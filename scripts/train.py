#!/usr/bin/env python3
"""
==============================================================================
训练脚本 - Forward/Reverse 任务训练（支持多种分布式模式）
==============================================================================

功能：
  - Forward 任务：混合训练（防止灾难性遗忘）
  - Reverse 任务：普通训练
  - 支持两种分布式模式：
    - deepspeed: DeepSpeed ZeRO-2，适合 8B 等小模型的数据并行
    - auto: device_map="auto"，适合 32B 等大模型的模型并行

==============================================================================
使用方法
==============================================================================

【模式1】DeepSpeed 分布式训练（8B 模型推荐）
-------------------------------------------------
# Forward 训练
deepspeed scripts/train.py --config configs/config.yaml \
    --task forward --name my_experiment \
    --data_dir data/8faces \
    --mode deepspeed

# Reverse 训练
deepspeed scripts/train.py --config configs/config.yaml \
    --task reverse --name my_experiment \
    --data_dir data/8faces \
    --mode deepspeed


【模式2】Auto 模型并行训练（32B 模型推荐）
-------------------------------------------------
# Forward 训练
python scripts/train.py --config configs/config_qwen3vl32_fp16.yaml \
    --task forward --name 32b_experiment \
    --data_dir data/8faces \
    --mode auto

# Reverse 训练
python scripts/train.py --config configs/config_qwen3vl32_fp16.yaml \
    --task reverse --name 32b_experiment \
    --data_dir data/8faces \
    --mode auto

==============================================================================
参数说明
==============================================================================

必需参数：
  --config              配置文件路径 (yaml)
  --task                任务类型: forward / reverse

可选参数：
  --mode                训练模式: deepspeed / auto (默认: deepspeed)
  --name                实验名称，输出到 outputs/<name>_<task>/
  --data_dir            数据目录（默认使用 config 中的 data.output_dir）
  --retention_ratio     保持任务比例 (默认: 0.3)
  --face_retention_pool 人脸retention池目录 (默认: data/face_retention_pool)
  --retention_pool      物体retention池目录 (默认: None)
  --face_retention_ratio 人脸池占retention的比例 (默认: 1.0, 只用人脸池)

==============================================================================
输出结构
==============================================================================

outputs/<name>_<task>/
├── best/                   # 最佳 checkpoint（验证 loss 最低）
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── final/                  # 最终 checkpoint
└── training_history.json   # 训练历史

==============================================================================
"""
import warnings
import os
import sys
import json
import yaml
import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*destroy_process_group.*")

os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '0')
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'OFF')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import MixedForwardDataset, collate_fn


def setup_model_deepspeed(config: dict, local_rank: int):
    """DeepSpeed 模式：加载模型到单卡"""
    model_path = config["model"]["name_or_path"]
    if local_rank == 0:
        print(f"[DeepSpeed Mode] Loading model from {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager",
    )
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]), lora_dropout=float(config["lora"]["dropout"]),
        target_modules=config["lora"]["target_modules"], bias="none",
    )
    model = get_peft_model(model, lora_config)
    if local_rank == 0:
        model.print_trainable_parameters()
    return model, processor


def setup_model_auto(config: dict):
    """Auto 模式：device_map=auto 分片到多卡"""
    model_path = config["model"]["name_or_path"]
    print(f"[Auto Mode] Loading model from {model_path}")
    print("Using device_map='auto' to distribute model across GPUs...")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager", device_map="auto", low_cpu_mem_usage=True,
    )
    print(f"Model device map: {model.hf_device_map}")
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]), lora_dropout=float(config["lora"]["dropout"]),
        target_modules=config["lora"]["target_modules"], bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def train_deepspeed(args, config):
    """DeepSpeed 分布式训练"""
    import deepspeed
    import torch.distributed as dist
    from torch.utils.data import DistributedSampler
    
    task = args.task
    retention_ratio = args.retention_ratio
    
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Starting {task.upper()} training on {world_size} GPUs (DeepSpeed)")
        if task == "forward":
            print(f"Mode: Mixed training (retention_ratio={retention_ratio})")
        print(f"{'='*60}")
    
    model, processor = setup_model_deepspeed(config, local_rank)
    
    data_dir = Path(args.data_dir) if args.data_dir else Path(config["data"]["output_dir"])
    max_length = int(config["training"].get("max_length", 512))
    
    if task == "forward":
        train_file = data_dir / "forward_train.jsonl"
        val_file = data_dir / "forward_val.jsonl"
        train_dataset = MixedForwardDataset(
            str(train_file), processor, max_length, retention_ratio=retention_ratio, seed=42,
            split="train", retention_pool_dir=args.retention_pool,
            face_retention_pool_dir=args.face_retention_pool, face_retention_ratio=args.face_retention_ratio
        )
        val_dataset = MixedForwardDataset(
            str(val_file), processor, max_length, retention_ratio=retention_ratio, seed=42,
            split="val", retention_pool_dir=args.retention_pool,
            face_retention_pool_dir=args.face_retention_pool, face_retention_ratio=args.face_retention_ratio
        )
    else:
        from src.data.dataset import ReverseDataset
        train_file = data_dir / "reverse_train.jsonl"
        val_file = data_dir / "reverse_val.jsonl"
        train_dataset = ReverseDataset(str(train_file), processor, max_length)
        val_dataset = MixedForwardDataset(str(val_file), processor, max_length, retention_ratio=0, seed=42)
    
    if local_rank == 0:
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    batch_size = int(config["training"].get("batch_size", config["training"].get("per_device_train_batch_size", 1)))
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    num_epochs = int(config["training"]["num_epochs"])
    initial_lr = float(config["training"]["learning_rate"])
    min_lr = float(config["training"].get("min_lr", 1e-6))
    current_lr = initial_lr
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    
    warmup_steps = int(len(train_loader) * num_epochs * float(config["training"].get("warmup_ratio", 0.05)) / grad_accum)
    
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "optimizer": {"type": "AdamW", "params": {"lr": current_lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": float(config["training"].get("weight_decay", 0.01)), "torch_adam": True}},
        "scheduler": {"type": "WarmupLR", "params": {"warmup_min_lr": 0, "warmup_max_lr": current_lr, "warmup_num_steps": warmup_steps}},
        "fp16": {"enabled": True, "loss_scale": 0, "loss_scale_window": 1000, "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1},
        "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True, "reduce_scatter": True, "reduce_bucket_size": 5e7, "allgather_bucket_size": 5e7},
        "gradient_clipping": 1.0, "steps_per_print": 9999999, "wall_clock_breakdown": False
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=[p for p in model.parameters() if p.requires_grad], config=ds_config)
    device = model_engine.local_rank
    
    output_dir = Path("outputs") / f"{args.name}_{task}" if args.name else Path(config["training"]["output_dir"]) / f"{task}_trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = {"task": task, "mode": "deepspeed", "train_loss": [], "val_loss": [], "epochs": []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        epoch_loss, num_batches = 0.0, 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if local_rank == 0 else train_loader
        for batch in progress:
            batch.pop("task_type", None)
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            loss = model_engine(**batch).loss
            model_engine.backward(loss)
            model_engine.step()
            epoch_loss += loss.item()
            num_batches += 1
            if local_rank == 0:
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / num_batches
        
        model_engine.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch.pop("task_type", None)
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                val_loss += model_engine(**batch).loss.item()
                val_batches += 1
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epochs"].append(epoch + 1)
        
        if local_rank == 0:
            print(f"  → Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                (output_dir / "best").mkdir(exist_ok=True)
                model_engine.module.save_pretrained(str(output_dir / "best"))
                print(f"  💾 Saved best (val_loss={best_val_loss:.4f})")
    
    if local_rank == 0:
        (output_dir / "final").mkdir(exist_ok=True)
        model_engine.module.save_pretrained(str(output_dir / "final"))
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n✓ Training Complete! Saved to: {output_dir}")
    
    try:
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        pass


def train_auto(args, config):
    """Auto 模式：单进程多卡训练"""
    task = args.task
    retention_ratio = args.retention_ratio
    
    print(f"\n{'='*60}")
    print(f"Starting {task.upper()} training (Auto mode, device_map='auto')")
    print(f"{'='*60}")
    
    model, processor = setup_model_auto(config)
    
    data_dir = Path(args.data_dir) if args.data_dir else Path(config["data"]["output_dir"])
    max_length = int(config["training"].get("max_length", 512))
    
    if task == "forward":
        train_file = data_dir / "forward_train.jsonl"
        val_file = data_dir / "forward_val.jsonl"
        train_dataset = MixedForwardDataset(
            str(train_file), processor, max_length, retention_ratio=retention_ratio, seed=42,
            split="train", retention_pool_dir=None, face_retention_pool_dir=args.face_retention_pool, face_retention_ratio=1.0
        )
        val_dataset = MixedForwardDataset(
            str(val_file), processor, max_length, retention_ratio=retention_ratio, seed=42,
            split="val", retention_pool_dir=None, face_retention_pool_dir=args.face_retention_pool, face_retention_ratio=1.0
        )
    else:
        from src.data.dataset import ReverseDataset
        train_file = data_dir / "reverse_train.jsonl"
        val_file = data_dir / "reverse_val.jsonl"
        train_dataset = ReverseDataset(str(train_file), processor, max_length)
        val_dataset = ReverseDataset(str(val_file), processor, max_length)
    
    batch_size = int(config["training"].get("batch_size", config["training"].get("per_device_train_batch_size", 1)))
    grad_accum = int(config["training"]["gradient_accumulation_steps"])
    num_epochs = int(config["training"]["num_epochs"])
    current_lr = float(config["training"]["learning_rate"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Batch: {batch_size}, Epochs: {num_epochs}")
    
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=current_lr, weight_decay=float(config["training"].get("weight_decay", 0.01)))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(config["training"].get("warmup_steps", 0)), num_training_steps=len(train_loader) * num_epochs // grad_accum)
    
    output_dir = Path("outputs") / f"{args.name}_{task}" if args.name else Path(config["training"]["output_dir"]) / f"{task}_trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = {"task": task, "mode": "auto", "train_loss": [], "val_loss": [], "epochs": []}
    best_val_loss = float('inf')
    device = next(model.parameters()).device
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        optimizer.zero_grad()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch.pop("task_type", None)
            loss = model(**batch).loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * grad_accum
            num_batches += 1

            # 实时显示loss和学习率
            progress.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        avg_train_loss = epoch_loss / num_batches
        
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch.pop("task_type", None)
                val_loss += model(**batch).loss.item()
                val_batches += 1
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"Epoch {epoch+1} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epochs"].append(epoch + 1)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / "best")
            print(f"  -> Saved best (val_loss={best_val_loss:.4f})")
    
    model.save_pretrained(output_dir / "final")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training complete! Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="训练脚本")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["forward", "reverse"])
    parser.add_argument("--mode", type=str, default="deepspeed", choices=["deepspeed", "auto"])
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--retention_ratio", type=float, default=0.3)
    parser.add_argument("--face_retention_pool", type=str, default="data/face_retention_pool")
    parser.add_argument("--retention_pool", type=str, default=None)
    parser.add_argument("--face_retention_ratio", type=float, default=1.0)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    try:
        import deepspeed
        parser = deepspeed.add_config_arguments(parser)
    except ImportError:
        pass
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.mode == "deepspeed":
        train_deepspeed(args, config)
    else:
        train_auto(args, config)


if __name__ == "__main__":
    main()
