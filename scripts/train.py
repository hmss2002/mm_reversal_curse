#!/usr/bin/env python3
"""
==============================================================================
训练脚本 - Forward/Reverse 任务训练（支持多种分布式模式）
==============================================================================

功能：
  - Forward 任务：混合训练（防止灾难性遗忘）
  - Reverse 任务：普通训练
  - 支持三种分布式模式：
    - deepspeed: DeepSpeed ZeRO-2，适合 8B 等小模型的数据并行
    - auto: device_map="auto"，适合 32B 等大模型的模型并行
    - qlora: 4-bit QLoRA + Accelerate，多卡并行（V100推荐）

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


【模式3】QLoRA 多卡并行训练（8卡V100推荐）
-------------------------------------------------
# Forward 训练 (混合retention)
accelerate launch --num_processes=8 scripts/train.py --config configs/config_qwen3vl32_fp16.yaml \
    --task forward --name qlora_experiment --data_dir data/8faces --mode qlora \
    --face_retention_pool data/face_retention_pool --retention_ratio 0.3

# Reverse 训练
accelerate launch --num_processes=8 scripts/train.py --config configs/config_qwen3vl32_fp16.yaml \
    --task reverse --name qlora_experiment --data_dir data/8faces --mode qlora

==============================================================================
参数说明
==============================================================================

必需参数：
  --config              配置文件路径 (yaml)
  --task                任务类型: forward / reverse

可选参数：
  --mode                训练模式: deepspeed / auto / qlora (默认: deepspeed)
  --name                实验名称，输出到 outputs/<name>_<task>/
  --data_dir            数据目录（默认使用 config 中的 data.output_dir）
  --retention_ratio     保持任务比例 (默认: 0.3)
  --face_retention_pool 人脸retention池目录 (默认: data/face_retention_pool)
  --object_rentention_pool      物体retention池目录 (默认: None)
  --face_object_retention_ratio 人脸池占retention的比例 (默认: 0.5)

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
from transformers import AutoProcessor, AutoModelForImageTextToText, get_linear_schedule_with_warmup, BitsAndBytesConfig, get_constant_schedule
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# Accelerate for multi-GPU data parallel (QLoRA mode)
try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
except ImportError:
    Accelerator = None
    DistributedDataParallelKwargs = None

warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*destroy_process_group.*")

os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '0')
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'OFF')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import ForwardDataset, collate_fn


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
        train_dataset = ForwardDataset(
            str(train_file), processor, max_length, retention_ratio=retention_ratio, seed=42,
            split="train", retention_pool_dir=args.object_rentention_pool,
            face_retention_pool_dir=args.face_retention_pool, face_object_retention_ratio=args.face_object_retention_ratio,
            retention_only=args.retention_only
        )
        val_dataset = ForwardDataset(
            str(val_file), processor, max_length, retention_ratio=retention_ratio, seed=42,
            split="val", retention_pool_dir=args.object_rentention_pool,
            face_retention_pool_dir=args.face_retention_pool, face_object_retention_ratio=args.face_object_retention_ratio,
            retention_only=args.retention_only
        )
    else:
        from src.data.dataset import ReverseDataset
        train_file = data_dir / "reverse_train.jsonl"
        val_file = data_dir / "reverse_val.jsonl"
        train_dataset = ReverseDataset(str(train_file), processor, max_length)
        val_dataset = ReverseDataset(str(val_file), processor, max_length)
    
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
        train_dataset = ForwardDataset(
            str(train_file), processor, max_length, retention_ratio=retention_ratio, seed=42,
            split="train", retention_pool_dir=args.object_rentention_pool,
            face_retention_pool_dir=args.face_retention_pool, face_object_retention_ratio=args.face_object_retention_ratio,
            retention_only=args.retention_only
        )
        val_dataset = ForwardDataset(
            str(val_file), processor, max_length, retention_ratio=retention_ratio, seed=42,
            split="val", retention_pool_dir=args.object_rentention_pool,
            face_retention_pool_dir=args.face_retention_pool, face_object_retention_ratio=args.face_object_retention_ratio,
            retention_only=args.retention_only
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


def setup_model_qlora(config: dict, accelerator: Accelerator):
    """
    设置 QLoRA 模型 (4-bit量化 + LoRA)
    
    特性:
    - 4-bit NF4量化，显存占用 ~18GB
    - FP16计算（V100不支持BF16）
    - 双重量化节省额外显存
    - 冻结 Vision Encoder
    - LoRA rank=8 (低显存)
    """
    model_path = config["model"]["name_or_path"]
    
    accelerator.print(f"Loading model with QLoRA: {model_path}")
    accelerator.print("  - 4-bit quantization (NF4)")
    accelerator.print("  - FP16 compute (V100 compatible)")
    accelerator.print("  - Double quantization enabled")
    
    # === 1. 量化配置 (QLoRA核心) ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # === 2. 加载基础模型 ===
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": accelerator.process_index},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    # === 3. 准备模型用于kbit训练 ===
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # === 4. 冻结 Vision Encoder ===
    if hasattr(model, 'visual'):
        for param in model.visual.parameters():
            param.requires_grad = False
        accelerator.print("  - Vision Encoder frozen")
    
    # === 5. LoRA配置 ===
    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=config["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    
    # === 6. 应用 LoRA ===
    model = get_peft_model(model, lora_config)
    accelerator.print("\n=== Trainable Parameters ===")
    model.print_trainable_parameters()
    
    # === 7. 加载 Processor（限制图像分辨率）===
    max_pixels = int(config["training"].get("max_pixels", 262144))  # 512*512
    min_pixels = int(config["training"].get("min_pixels", 3136))    # 56*56
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True,
        max_pixels=max_pixels,
        min_pixels=min_pixels
    )
    accelerator.print(f"  - Image resolution: {min_pixels} ~ {max_pixels} pixels")
    
    return model, processor


def train_qlora(args, config):
    """
    QLoRA 多卡并行训练（使用 Accelerate）
    
    流程:
    1. 初始化 Accelerate
    2. 设置模型（4-bit量化 + LoRA）
    3. 准备数据集和优化器
    4. 使用 accelerator.prepare() 包装
    5. 训练循环
    """
    # === 1. 初始化 Accelerator ===
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        mixed_precision="fp16",
        kwargs_handlers=[kwargs]
    )
    
    task = args.task
    retention_ratio = args.retention_ratio
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"QLoRA Training: {task.upper()} Task")
    accelerator.print(f"  - GPUs: {accelerator.num_processes}")
    accelerator.print(f"  - Mixed Precision: FP16 (V100 compatible)")
    accelerator.print(f"  - Gradient Accumulation Steps: {accelerator.gradient_accumulation_steps}")
    if task == "forward":
        accelerator.print(f"  - Retention Ratio: {retention_ratio}")
    accelerator.print(f"{'='*70}\n")
    
    # === 2. 加载模型和处理器 ===
    model, processor = setup_model_qlora(config, accelerator)
    
    # === 3. 准备数据集 ===
    data_dir = Path(args.data_dir) if args.data_dir else Path(config["data"]["output_dir"])
    max_length = int(config["training"].get("max_length", 512))
    
    if task == "forward":
        train_file = data_dir / "forward_train.jsonl"
        val_file = data_dir / "forward_val.jsonl"
        train_dataset = ForwardDataset(
            str(train_file), processor, max_length, 
            retention_ratio=retention_ratio, seed=42,
            split="train", 
            retention_pool_dir=args.object_rentention_pool,
            face_retention_pool_dir=args.face_retention_pool, 
            face_object_retention_ratio=args.face_object_retention_ratio,
            retention_only=args.retention_only
        )
        val_dataset = ForwardDataset(
            str(val_file), processor, max_length, 
            retention_ratio=retention_ratio, seed=42,
            split="val", 
            retention_pool_dir=args.object_rentention_pool,
            face_retention_pool_dir=args.face_retention_pool, 
            face_object_retention_ratio=args.face_object_retention_ratio,
            retention_only=args.retention_only
        )
    else:
        from src.data.dataset import ReverseDataset
        train_file = data_dir / "reverse_train.jsonl"
        val_file = data_dir / "reverse_val.jsonl"
        train_dataset = ReverseDataset(str(train_file), processor, max_length)
        val_dataset = ReverseDataset(str(val_file), processor, max_length)
    
    accelerator.print(f"Dataset loaded:")
    accelerator.print(f"  - Train samples: {len(train_dataset)}")
    accelerator.print(f"  - Val samples: {len(val_dataset)}\n")
    
    # === 4. 训练参数 ===
    batch_size = int(config["training"].get("batch_size", 
                     config["training"].get("per_device_train_batch_size", 1)))
    grad_accum = int(config["training"]["gradient_accumulation_steps"])
    num_epochs = int(config["training"]["num_epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.05))
    
    # === 5. DataLoader ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=0,
        pin_memory=True
    )
    
    accelerator.print(f"Training config:")
    accelerator.print(f"  - Batch size per device: {batch_size}")
    accelerator.print(f"  - Gradient accumulation: {grad_accum}")
    accelerator.print(f"  - Effective batch size: {batch_size * accelerator.num_processes * grad_accum}")
    accelerator.print(f"  - Learning rate: {learning_rate}")
    accelerator.print(f"  - Epochs: {num_epochs}")
    # warmup_steps 将在后面计算后打印
    
    # === 6. 优化器和调度器 ===
    use_8bit_optim = config["training"].get("use_8bit_optimizer", False)
    
    if use_8bit_optim and HAS_BNB:
        accelerator.print("  - Using 8-bit AdamW optimizer (saves ~2GB VRAM)")
        optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # === 7. Accelerate 包装（先prepare，再创建scheduler）===
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # 在 prepare 之后计算 training steps（train_loader 已被分片到每个进程）
    steps_per_epoch = len(train_loader) // grad_accum
    num_training_steps = steps_per_epoch * num_epochs
    warmup_steps = 0  # 禁用warmup，直接使用目标学习率
    # warmup_steps = max(warmup_steps, 1)  # 已禁用warmup
    
    accelerator.print(f"  - Steps per epoch (per GPU): {steps_per_epoch}")
    accelerator.print(f"  - Total training steps: {num_training_steps}")
    accelerator.print(f"  - Warmup steps: {warmup_steps} ({warmup_ratio*100:.1f}%)")
    
    # 在 prepare 之后创建 scheduler，使用被包装后的 optimizer
    scheduler = get_constant_schedule(optimizer)  # Constant LR - 学习率保持不变

    # === 8. 输出目录 ===
    output_dir = Path("outputs") / f"{args.name}_{task}" if args.name else \
                 Path(config["training"]["output_dir"]) / f"{task}_trained"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # === 9. 训练历史 ===
    history = {
        "task": task,
        "mode": "qlora_accelerate",
        "num_gpus": accelerator.num_processes,
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
        "step_logs": []  # 记录每个step的详细信息
    }
    best_val_loss = float('inf')
    global_step = 0  # 全局step计数器
    
    # === 10. 训练循环 ===
    accelerator.print(f"{'='*70}")
    accelerator.print("Starting training...")
    accelerator.print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch in progress:
            with accelerator.accumulate(model):
                # 移除任务类型标记
                batch.pop("task_type", None)
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
                
                # scheduler只在真正更新参数时step
                if accelerator.sync_gradients:
                    scheduler.step()
            
            # 记录损失
            epoch_loss += loss.item()
            num_batches += 1
            
            # 只在真正更新参数时记录step信息
            if accelerator.sync_gradients:
                global_step += 1
                current_lr = optimizer.param_groups[0]['lr']
                
                # 记录step级别的log
                if accelerator.is_main_process:
                    step_log = {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "loss": loss.item(),
                        "lr": current_lr
                    }
                    history["step_logs"].append(step_log)
            
            # 更新进度条
            if accelerator.is_local_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}"
                })
        
        avg_train_loss = epoch_loss / num_batches
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch.pop("task_type", None)
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        
        # 收集所有进程的损失
        avg_train_loss = accelerator.gather(torch.tensor([avg_train_loss]).to(accelerator.device)).mean().item()
        avg_val_loss = accelerator.gather(torch.tensor([avg_val_loss]).to(accelerator.device)).mean().item()
        
        # --- 记录和保存 ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epochs"].append(epoch + 1)
        
        if accelerator.is_main_process:
            accelerator.print(f"\nEpoch {epoch+1}/{num_epochs}:")
            accelerator.print(f"  - Train Loss: {avg_train_loss:.4f}")
            accelerator.print(f"  - Val Loss: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                
                # 解包模型再保存
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(str(best_dir))
                
                accelerator.print(f"  💾 Saved best model (val_loss={best_val_loss:.4f})")
            
            accelerator.print("")
        
        # 等待所有进程
        accelerator.wait_for_everyone()
    
    # === 11. 保存最终模型 ===
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(str(final_dir))
        
        # 保存训练历史
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        accelerator.print(f"\n{'='*70}")
        accelerator.print("✓ Training Complete!")
        accelerator.print(f"  - Output: {output_dir}")
        accelerator.print(f"  - Best Val Loss: {best_val_loss:.4f}")
        accelerator.print(f"{'='*70}\n")



def main():
    parser = argparse.ArgumentParser(description="训练脚本")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["forward", "reverse"])
    parser.add_argument("--mode", type=str, default="deepspeed", choices=["deepspeed", "auto", "qlora"])
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--retention_ratio", type=float, default=0.3)
    parser.add_argument("--face_retention_pool", type=str, default="data/face_retention_pool")
    parser.add_argument("--object_rentention_pool", type=str, default=None)
    parser.add_argument("--face_object_retention_ratio", type=float, default=0.5)
    parser.add_argument("--retention_only", action="store_true")
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
    elif args.mode == "auto":
        train_auto(args, config)
    else:
        if Accelerator is None:
            raise ImportError("请安装 accelerate: pip install accelerate")
        train_qlora(args, config)


if __name__ == "__main__":
    main()
