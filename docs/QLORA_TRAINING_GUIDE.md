# QLoRA 训练指南 - 32B模型8卡并行

## 📋 概述

新增的 `train_qlora.py` 脚本支持在8张V100 (32GB)上并行训练Qwen3-VL-32B模型，使用4-bit量化节省显存。

**特性：**
- ✅ 4-bit NF4量化：模型显存占用 ~18GB
- ✅ V100 FP16优化：硬件兼容性
- ✅ 8卡数据并行：训练速度快
- ✅ LoRA rank=8：低显存高效率
- ✅ 智能学习率调度：自动降低LR
- ✅ 早停机制：防止过拟合
- ✅ 完全独立：不影响原有 `train.py`

---

## 🚀 快速开始

### 1. Forward训练（混合retention）

```bash
cd /work/mm_reversal_curse
source .venv/bin/activate

accelerate launch --num_processes=8 scripts/train_qlora.py \
  --config configs/config_qwen3vl32_fp16.yaml \
  --task forward \
  --name 4faces_qlora \
  --data_dir data/4faces \
  --face_retention_pool data/face_retention_pool \
  --retention_ratio 0.3
```

### 2. Reverse训练

```bash
accelerate launch --num_processes=8 scripts/train_qlora.py \
  --config configs/config_qwen3vl32_fp16.yaml \
  --task reverse \
  --name 4faces_qlora \
  --data_dir data/4faces
```

---

## ⚙️ 配置文件说明

### config_qwen3vl32_fp16.yaml

```yaml
lora:
  r: 8                    # LoRA rank (降低显存)
  alpha: 16               # LoRA alpha (2*rank)
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"

training:
  # === Learning Rate Configuration ===
  learning_rate: 1e-4              # 初始学习率
  min_lr: 6e-5                     # 最小学习率阈值
  lr_reduction_factor: 0.5         # LR衰减因子（每次减半）
  lr_patience: 1                   # LR衰减前等待的epoch数
  improvement_threshold: 0.05      # Val loss改善阈值（5%）
  min_val_loss: 0.2                # 早停阈值
  
  # === Batch Configuration ===
  batch_size: 1                    # 每卡batch=1（必须）
  gradient_accumulation_steps: 8   # 梯度累积（等效batch=8*8=64）
  num_epochs: 10
  max_length: 512
  warmup_ratio: 0.02
  weight_decay: 0.01
```

---

## 📊 显存占用分析

### 单卡显存（V100 32GB）

| 组件 | 显存占用 |
|------|---------|
| 基础模型 (4-bit) | ~18GB |
| LoRA参数 (rank=8) | ~0.5GB |
| 优化器状态 | ~1GB |
| 梯度 + 激活值 | ~8GB |
| **总计** | **~27.5GB** |

**安全余量：** ~4.5GB（足够应对动态波动）

---

## 🎯 训练策略

### 学习率调度

```
初始LR: 1e-4
  ↓
Val loss不改善(patience=1 epoch)
  ↓
LR *= 0.5 → 5e-5
  ↓
继续不改善
  ↓
LR *= 0.5 → 2.5e-5
  ↓
LR < min_lr (6e-5) → 停止训练
```

### 早停条件

1. **Val loss阈值：** `val_loss < 0.2` → 停止
2. **LR阈值：** `lr < 6e-5` → 停止
3. **改善阈值：** Val loss改善 < 5% → 不保存，累积patience

---

## 📁 输出结构

```
outputs/4faces_qlora_forward/
├── best/
│   ├── adapter_config.json
│   └── adapter_model.safetensors   # 最佳checkpoint（153MB）
├── final/
│   ├── adapter_config.json
│   └── adapter_model.safetensors   # 最终checkpoint
└── training_history.json            # 训练历史
```

---

## 🔍 评估模型

训练完成后使用原有评估脚本：

```bash
python3 scripts/evaluate.py \
  --model_path outputs/4faces_qlora_forward/best \
  --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
  --data_dir data/4faces \
  --task all \
  --save_examples 10 \
  --device_map auto
```

---

## ⚡ 性能对比

| 模式 | 卡数 | 显存/卡 | 速度 | 适用场景 |
|------|------|---------|------|----------|
| **qlora** | 8 | ~28GB | 快 | 32B模型推荐 |
| auto | 3-4 | ~25GB | 中 | 单任务推理 |
| deepspeed | 8 | OOM | - | 32B不适用 |

---

## 🛠️ 故障排查

### 1. OOM错误

**症状：** `CUDA out of memory`

**解决：**
```bash
# 检查是否有其他进程占用GPU
nvidia-smi

# 降低gradient_accumulation_steps
# config_qwen3vl32_fp16.yaml:
gradient_accumulation_steps: 4  # 从8降到4
```

### 2. Accelerate配置

**首次使用需要配置：**
```bash
accelerate config

# 选择：
# - Compute environment: This machine
# - Distributed type: multi-GPU
# - Number of processes: 8
# - Mixed precision: fp16
```

### 3. 训练速度慢

**检查：**
- Gradient Checkpointing已启用（会稍慢但节省显存）
- Batch size=1是必须的（32B模型限制）
- 等效batch通过gradient_accumulation实现

---

## 📚 技术细节

### QLoRA原理

1. **4-bit量化：** 基础模型压缩到18GB（原64GB）
2. **LoRA训练：** 只训练小adapter（FP16高精度）
3. **混合精度：** 前向传播INT4，梯度计算FP16
4. **Paged Optimizer：** 优化器状态可offload到CPU

### 冻结策略

- ✅ Vision Encoder: 冻结（不训练）
- ✅ LLM部分: LoRA微调（q/k/v/o projection）
- ✅ 总参数量: ~200M可训练（0.6%）

---

## 🆚 与原有模式对比

| 特性 | train.py (auto) | train.py (deepspeed) | train_qlora.py |
|------|-----------------|---------------------|----------------|
| 32B支持 | ✅ | ❌ (OOM) | ✅ |
| 卡数 | 3-4 | 8 | 8 |
| 量化 | 无 | 无 | 4-bit |
| 并行方式 | 模型并行 | 数据并行 | 数据并行 |
| 训练速度 | 慢 | - | 快 |
| 显存/卡 | ~25GB | OOM | ~28GB |
| **推荐** | 单机推理 | 小模型 | **32B训练首选** |

---

## ✅ 完整工作流

```bash
# 1. 生成数据
python scripts/generate_data.py --config configs/face_config.yaml

# 2. 生成retention pool
python scripts/generate_face_retention_pool.py --num_entities 4

# 3. QLoRA训练
accelerate launch --num_processes=8 scripts/train_qlora.py \
  --config configs/config_qwen3vl32_fp16.yaml \
  --task forward \
  --name 4faces_qlora \
  --data_dir data/4faces \
  --face_retention_pool data/face_retention_pool \
  --retention_ratio 0.3

# 4. 评估
python scripts/evaluate.py \
  --model_path outputs/4faces_qlora_forward/best \
  --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
  --data_dir data/4faces \
  --task all \
  --device_map auto
```

---

**注意：** `train_qlora.py` 完全独立，不影响原有 `train.py` 的任何功能。原有的 `--mode auto` 和 `--mode deepspeed` 仍然可以正常使用。
