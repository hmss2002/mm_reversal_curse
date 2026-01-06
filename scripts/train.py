#!/usr/bin/env python3
"""
==============================================================================
训练脚本 - Forward/Reverse 任务训练
==============================================================================

功能：
  使用 DeepSpeed ZeRO-2 + LoRA 进行分布式训练
  - Forward 任务：自动使用混合训练（防止灾难性遗忘）
  - Reverse 任务：普通训练

==============================================================================
快速开始
==============================================================================

# 1. 切换到项目目录并激活虚拟环境
cd /work/mm_reversal_curse
source .venv/bin/activate

# 2. Forward 训练（推荐先运行这个验证 Reversal Curse）
rm -rf outputs/forward_trained  # 清除之前的结果
deepspeed --num_gpus=8 scripts/train.py --config configs/config.yaml --task forward

# 3. Reverse 训练（可选）
rm -rf outputs/reverse_trained
deepspeed --num_gpus=8 scripts/train.py --config configs/config.yaml --task reverse

# 4. 调整 Forward 的保持任务比例（默认 0.3 即 30%）
deepspeed --num_gpus=8 scripts/train.py --config configs/config.yaml --task forward --retention_ratio 0.2

==============================================================================
输出结构
==============================================================================

outputs/forward_trained/  或  outputs/reverse_trained/
├── best/                   # 最佳 checkpoint（验证 loss 最低）
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── final/                  # 最终 checkpoint
└── training_history.json   # 训练历史（loss、epoch 等）

==============================================================================
训练配置（configs/config.yaml）
==============================================================================

- model: Qwen3-VL-8B-Instruct
- LoRA: r=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj]
- Early Stopping: patience=3（验证 loss 连续3次不下降则停止）
- 最大 epoch: 1000

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
import torch.distributed as dist
import deepspeed
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*destroy_process_group.*")


# 抑制 DeepSpeed/NCCL 结束时的 Broken pipe 警告
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '0')
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'OFF')
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import ForwardDataset, ReverseDataset, collate_fn
from src.data.mixed_dataset import MixedForwardDataset


def setup_model_and_processor(config: dict, local_rank: int):
    """加载模型和 processor"""
    model_path = config["model"]["name_or_path"]
    
    if local_rank == 0:
        print(f"Loading model from {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
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
    if local_rank == 0:
        model.print_trainable_parameters()
    
    return model, processor


def train(args, config):
    """主训练循环"""
    task = args.task
    retention_ratio = args.retention_ratio
    
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Starting {task.upper()} training on {world_size} GPUs")
        if task == "forward":
            print(f"Mode: Mixed training (retention_ratio={retention_ratio})")
        print(f"{'='*60}")
    
    model, processor = setup_model_and_processor(config, local_rank)
    
    data_dir = Path(config["data"]["output_dir"])
    max_length = int(config["training"].get("max_length", 512))
    
    if task == "forward":
        train_file = data_dir / "forward_train.jsonl"
        val_file = data_dir / "forward_val.jsonl"
        
        # Forward 使用混合数据集（自动混入保持任务防止灾难性遗忘）
        train_dataset = MixedForwardDataset(
            str(train_file), processor, max_length,
            retention_ratio=retention_ratio,
            seed=42,
        )
        # 验证集用纯 Forward（评估主任务性能）
        val_dataset = ForwardDataset(str(val_file), processor, max_length)
    else:
        train_file = data_dir / "reverse_train.jsonl"
        val_file = data_dir / "reverse_val.jsonl"
        train_dataset = ReverseDataset(str(train_file), processor, max_length)
        val_dataset = ReverseDataset(str(val_file), processor, max_length)
    
    if local_rank == 0:
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    batch_size = int(config["training"]["batch_size"])
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    num_epochs = int(config["training"]["num_epochs"])
    
    initial_lr = float(config["training"]["learning_rate"])
    min_lr = float(config["training"].get("min_lr", 1e-6))
    lr_reduction_factor = float(config["training"].get("lr_reduction_factor", 0.5))
    lr_patience = int(config["training"].get("lr_patience", 1))
    improvement_threshold = float(config["training"].get("improvement_threshold", 0.05))
    min_val_loss = float(config["training"].get("min_val_loss", 0.01))
    
    current_lr = initial_lr
    
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=42
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * num_epochs
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.05))
    warmup_steps = int(total_steps * warmup_ratio)
    
    if local_rank == 0:
        print(f"Batch: {batch_size} x {grad_accum} x {world_size} = {batch_size * grad_accum * world_size}")
        print(f"Initial LR: {initial_lr}, Min LR: {min_lr}")
        print(f"LoRA r={config['lora']['r']}, alpha={config['lora']['alpha']}")
        print()
    
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": current_lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": float(config["training"].get("weight_decay", 0.01)),
                "torch_adam": True
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": current_lr,
                "warmup_num_steps": warmup_steps
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "allgather_bucket_size": 5e7
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 9999999,
        "wall_clock_breakdown": False
    }
    
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args, model=model, model_parameters=model_parameters, config=ds_config
    )
    
    device = model_engine.local_rank
    
    output_dir = Path(config["training"]["output_dir"]) / f"{task}_trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        "task": task,
        "mode": "mixed" if task == "forward" else "normal",
        "retention_ratio": retention_ratio if task == "forward" else None,
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "epochs": []
    }
    best_val_loss = float('inf')
    prev_val_loss = None
    patience_counter = 0
    should_stop = False
    
    for epoch in range(num_epochs):
        if should_stop:
            break
        
        train_sampler.set_epoch(epoch)
        
        model_engine.train()
        epoch_loss = 0.0
        num_batches = 0
        
        if local_rank == 0:
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (lr={current_lr:.2e})")
        else:
            progress = train_loader
        
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            
            model_engine.backward(loss)
            model_engine.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if local_rank == 0:
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / num_batches
        
        model_engine.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model_engine(**batch)
                val_loss += outputs.loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["learning_rates"].append(current_lr)
        history["epochs"].append(epoch + 1)
        
        if local_rank == 0:
            ratio = avg_val_loss / avg_train_loss if avg_train_loss > 0 else 1.0
            print(f"  → Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f} (ratio: {ratio:.2f})")
            
            if avg_val_loss < min_val_loss:
                print(f"\n✅ val_loss {avg_val_loss:.4f} < {min_val_loss}, converged!")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    history["best_epoch"] = epoch + 1
                    history["best_val_loss"] = best_val_loss
                    
                    best_dir = output_dir / "best"
                    best_dir.mkdir(exist_ok=True)
                    model_engine.module.save_pretrained(str(best_dir))
                    print(f"  💾 Saved best model")
                should_stop = True
                continue
            
            if prev_val_loss is not None:
                improvement = (prev_val_loss - avg_val_loss) / prev_val_loss
                print(f"  Improvement: {improvement*100:.2f}%")
                
                if improvement < improvement_threshold:
                    patience_counter += 1
                    print(f"  ⚡ Plateau ({patience_counter}/{lr_patience})")
                    
                    if patience_counter >= lr_patience:
                        new_lr = current_lr * lr_reduction_factor
                        
                        if new_lr < min_lr:
                            print(f"\n🛑 LR {new_lr:.2e} < min_lr, stopping")
                            should_stop = True
                            continue
                        
                        current_lr = new_lr
                        patience_counter = 0
                        
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        
                        print(f"  📉 LR → {current_lr:.2e}")
                else:
                    patience_counter = 0
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                history["best_epoch"] = epoch + 1
                history["best_val_loss"] = best_val_loss
                
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                model_engine.module.save_pretrained(str(best_dir))
                print(f"  💾 Saved best (val_loss={best_val_loss:.4f})")
            
            prev_val_loss = avg_val_loss
            print()
    
    if local_rank == 0:
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        model_engine.module.save_pretrained(str(final_dir))
        
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Training Complete ({task})")
        if task == "forward":
            print(f"  Mode: Mixed (retention_ratio={retention_ratio})")
        print(f"  Best: epoch {history.get('best_epoch', 'N/A')}, val_loss={history.get('best_val_loss', 0):.4f}")
        print(f"  Saved to: {output_dir}")
        print(f"{'='*60}")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["forward", "reverse"])
    parser.add_argument("--retention_ratio", type=float, default=0.3,
                        help="Retention task ratio for Forward training (default: 0.3)")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train(args, config)


if __name__ == "__main__":
    main()
