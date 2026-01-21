# 8x V100 32GB FP16 评测指南

## 📋 概述

`evaluate.py` 脚本已完全支持 8x V100 32GB FP16 环境的分布式评测，无需额外修改。

## 🔍 功能检查结果

### 已删除重复文件
- ✅ `evaluate_old.py` - 与 `evaluate.py` 完全相同，已删除

### evaluate.py 功能特性

#### ✅ 已支持的功能：
1. **分布式评测** - 使用 `--mode distributed` + `torchrun`
2. **FP16精度** - 默认使用 `torch_dtype=torch.float16`
3. **4-bit量化** - 使用 `--use_4bit` 进一步节省显存
4. **模型并行** - 通过 `device_map` 自动处理32B模型
5. **数据并行** - 8B模型可在8卡上分布评测
6. **4种评测任务** - forward, reverse, mcq_i2d, mcq_d2i

## 🚀 使用方法

### 方法1: 使用 torchrun（推荐）

#### 评测 8B 模型（数据并行）
```bash
cd /work/mm_reversal_curse
source .venv/bin/activate

# 评测所有任务
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path outputs/8faces_forward/best \
    --base_model /work/models/qwen/Qwen3-VL-8B-Instruct \
    --data_dir data/8faces \
    --task all \
    --mode distributed \
    --save_examples 10

# 只评测 forward 任务
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path outputs/8faces_forward/best \
    --data_file data/8faces/forward_test.jsonl \
    --task forward \
    --mode distributed
```

#### 评测 32B 模型（模型并行 + 4bit）
```bash
# 32B模型建议使用4bit量化和auto device_map
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path outputs/4faces_32b_test_forward/best \
    --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
    --data_dir data/4faces \
    --task all \
    --mode distributed \
    --use_4bit \
    --device_map auto \
    --save_examples 10
```

### 方法2: 使用便捷脚本

```bash
# 8B模型评测
bash scripts/run_eval_8v100.sh \
    --model_path outputs/8faces_forward/best \
    --data_dir data/8faces

# 32B模型评测（自动使用模型并行）
bash scripts/run_eval_8v100.sh \
    --model_path outputs/4faces_32b_test_forward/best \
    --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
    --data_dir data/4faces \
    --model_parallel
```

### 方法3: 单GPU评测（调试用）

```bash
python3 scripts/evaluate.py \
    --model_path outputs/8faces_forward/best \
    --data_dir data/8faces \
    --task forward \
    --max_samples 10
```

## 📊 参数说明

### 必需参数
- `--model_path`: LoRA adapter路径（可选，不提供则评测base model）
- `--data_dir`: 数据目录（task=all时）
- `--data_file`: 数据文件（单任务时）

### 重要参数
- `--mode`: 评测模式
  - `single`: 单GPU（默认）
  - `distributed`: 多GPU分布式
- `--task`: 评测任务
  - `forward`: 图像→描述
  - `reverse`: 描述+图像→正确/错误
  - `mcq_i2d`: 图像→选择描述
  - `mcq_d2i`: 描述→选择图像
  - `all`: 所有任务（默认）
- `--base_model`: 基础模型路径
  - 默认: `/work/models/qwen/Qwen3-VL-8B-Instruct`
  - 32B: `/work/models/qwen/Qwen3-VL-32B-Instruct`

### 优化参数
- `--use_4bit`: 使用4-bit量化（32B模型推荐）
- `--device_map`: 设备映射策略
  - `cuda`: 单卡（默认）
  - `auto`: 自动多卡分布（32B推荐）
- `--save_examples`: 保存样例数量（-1=全部，0=不保存，默认5）
- `--max_samples`: 限制样本数量（调试用）

## 💾 内存优化建议

### 8B 模型（单卡 ~16GB）
```bash
# V100 32GB 完全够用，使用FP16即可
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path <path> \
    --data_dir <dir> \
    --mode distributed
```

### 32B 模型（单卡 ~60GB，需要跨卡）
```bash
# 方案1: 4-bit量化 + auto device_map（推荐）
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path <path> \
    --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
    --data_dir <dir> \
    --mode distributed \
    --use_4bit \
    --device_map auto

# 方案2: FP16 + auto device_map（显存充足时）
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path <path> \
    --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
    --data_dir <dir> \
    --mode distributed \
    --device_map auto
```

## 📁 输出结构

```
outputs/<model_name>/
├── eval_results_v3.json         # 评测结果汇总
│   ├── timestamp               # 评测时间
│   ├── forward                 # Forward任务结果
│   │   ├── accuracy
│   │   ├── correct
│   │   ├── total
│   │   └── examples (可选)
│   ├── reverse                 # Reverse任务结果
│   │   ├── accuracy
│   │   ├── tpr                 # True Positive Rate
│   │   ├── fpr                 # False Positive Rate
│   │   ├── separation          # TPR - FPR
│   │   └── examples (可选)
│   ├── mcq_i2d                 # MCQ I2D结果
│   └── mcq_d2i                 # MCQ D2I结果
```

## 🔧 troubleshooting

### 问题1: CUDA OOM (显存溢出)
```bash
# 解决方案1: 使用4bit量化
--use_4bit

# 解决方案2: 减少样本数量（调试）
--max_samples 100

# 解决方案3: 使用auto device_map
--device_map auto
```

### 问题2: 分布式初始化失败
```bash
# 确保使用 torchrun 而不是 python
torchrun --nproc_per_node=8 scripts/evaluate.py ...

# 检查GPU数量
nvidia-smi --list-gpus
```

### 问题3: 模型加载慢
```bash
# 正常现象，32B模型加载需要几分钟
# 可以添加 --max_samples 10 快速测试
```

## ✨ 最佳实践

1. **8B模型**: 直接使用8卡数据并行，FP16精度
   ```bash
   torchrun --nproc_per_node=8 scripts/evaluate.py \
       --model_path outputs/8faces_forward/best \
       --data_dir data/8faces \
       --mode distributed
   ```

2. **32B模型**: 使用4bit量化 + auto device_map
   ```bash
   torchrun --nproc_per_node=8 scripts/evaluate.py \
       --model_path outputs/4faces_32b_test_forward/best \
       --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
       --data_dir data/4faces \
       --mode distributed \
       --use_4bit \
       --device_map auto
   ```

3. **快速测试**: 使用单GPU + max_samples
   ```bash
   python3 scripts/evaluate.py \
       --model_path outputs/8faces_forward/best \
       --data_file data/8faces/forward_test.jsonl \
       --task forward \
       --max_samples 10
   ```

## 📝 总结

- ✅ **evaluate.py 已完全支持 8x V100 32GB FP16 环境**
- ✅ **无需修改代码，使用现有参数即可**
- ✅ **支持8B模型数据并行 和 32B模型模型并行**
- ✅ **提供便捷脚本 run_eval_8v100.sh**
- ✅ **删除了重复的 evaluate_old.py**

使用 `torchrun` + `--mode distributed` 即可充分利用8卡V100资源！
