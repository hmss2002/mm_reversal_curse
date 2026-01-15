#!/usr/bin/env python3
"""
单进程训练脚本 - 使用 device_map="auto" 把大模型分片到多卡
适用于32B等大模型在多卡（如8x32GB）上训练
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModelForImageTextToText, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import MixedForwardDataset, collate_fn


def setup_model_and_processor(config: dict):
    """加载模型和 processor，使用 device_map=auto 分片到多卡"""
    model_path = config["model"]["name_or_path"]
    
    print(f"Loading model from {model_path}")
    print("Using device_map='auto' to distribute model across GPUs...")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    print(f"Model device map: {model.hf_device_map}")
    
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=config["lora"]["target_modules"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor


def train(args, config):
    """主训练循环"""
    task = args.task
    retention_ratio = args.retention_ratio
    
    print(f"\n{'='*60}")
    print(f"Starting {task.upper()} training (single process, multi-GPU)")
    if task == "forward":
        print(f"Mode: Mixed training (retention_ratio={retention_ratio})")
    print(f"{'='*60}")
    
    model, processor = setup_model_and_processor(config)
    
    # 数据目录
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(config["data"]["output_dir"])
    max_length = int(config["training"].get("max_length", 512))
    
    if task == "forward":
        train_file = data_dir / "forward_train.jsonl"
        val_file = data_dir / "forward_val.jsonl"
    else:
        train_file = data_dir / "reverse_train.jsonl"
        val_file = data_dir / "reverse_val.jsonl"
    
    face_retention_ratio = args.face_retention_ratio
    
    if task == "forward":
        face_retention_dir = Path(config["data"].get("face_retention_dir", "data/face_retention_pool"))
        object_retention_dir = Path(args.retention_pool) if args.retention_pool else None
        
        train_dataset = MixedForwardDataset(
            forward_file=train_file,
            processor=processor,
            max_length=max_length,
            face_retention_dir=face_retention_dir,
            object_retention_dir=object_retention_dir,
            retention_ratio=retention_ratio,
            face_retention_ratio=face_retention_ratio,
        )
        val_dataset = MixedForwardDataset(
            forward_file=val_file,
            processor=processor,
            max_length=max_length,
            face_retention_dir=None,
            object_retention_dir=None,
            retention_ratio=0.0,
        )
    else:
        from src.data.dataset import ReversalDataset
        train_dataset = ReversalDataset(train_file, processor, max_length)
        val_dataset = ReversalDataset(val_file, processor, max_length)
    
    batch_size = int(config["training"]["per_device_train_batch_size"])
    grad_accum = int(config["training"]["gradient_accumulation_steps"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    num_epochs = int(config["training"]["num_epochs"])
    current_lr = float(config["training"]["learning_rate"])
    warmup_steps = int(config["training"].get("warmup_steps", 0))
    
    print(f"\nTraining config:")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Grad accum: {grad_accum}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {current_lr}")
    print()
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=current_lr,
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    
    total_steps = len(train_loader) * num_epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Output dir
    if args.name:
        output_dir = Path("outputs") / f"{args.name}_{task}"
    else:
        output_dir = Path(config["training"]["output_dir"]) / f"{task}_trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        "task": task,
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "epochs": []
    }
    best_val_loss = float('inf')
    
    # 获取模型主设备
    device = next(model.parameters()).device
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress):
            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "task_type" in batch:
                batch.pop("task_type")
            
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * grad_accum
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})
        
        avg_train_loss = epoch_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if "task_type" in batch:
                    batch.pop("task_type")
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["learning_rates"].append(scheduler.get_last_lr()[0])
        history["epochs"].append(epoch + 1)
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / "best")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
    
    # Save final
    model.save_pretrained(output_dir / "final")
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["forward", "reverse"], required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--retention_ratio", type=float, default=0.3)
    parser.add_argument("--face_retention_ratio", type=float, default=0.5)
    parser.add_argument("--retention_pool", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train(args, config)


if __name__ == "__main__":
    main()
