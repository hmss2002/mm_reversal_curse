# 多模态 Reversal Curse 实验

研究多模态大语言模型中的 Reversal Curse（反向诅咒）现象。

## 项目结构

```
mm_reversal_curse/
├── configs/
│   └── config.yaml           # 主配置文件
├── data/
│   └── current_dataset/      # 当前使用的数据集
│       ├── entities.json     # 实体定义
│       ├── images/           # 人脸图片
│       ├── retention_images/ # 保持任务用物体图片
│       ├── forward_*.jsonl   # Forward 数据
│       ├── reverse_*.jsonl   # Reverse 数据
│       └── mcq_*.jsonl       # 选择题测试数据
├── outputs/                  # 训练输出
├── scripts/
│   ├── generate_data.py      # 数据生成（实体+图片+训练数据）
│   ├── train.py              # 训练脚本
│   └── evaluate.py           # 评估脚本
├── src/data/
│   ├── dataset.py            # Forward/Reverse 数据集类
│   └── mixed_dataset.py      # 混合数据集（Forward 使用）
└── legacy/                   # 旧版本文件备份
```

## 快速开始

### 1. 生成数据

```bash
python scripts/generate_data.py --config configs/config.yaml --num_entities 100 --seed 42
```

### 2. 训练

```bash
# Forward 训练（自动使用混合训练，防止灾难性遗忘）
deepspeed --num_gpus=8 scripts/train.py --config configs/config.yaml --task forward

# Reverse 训练
deepspeed --num_gpus=8 scripts/train.py --config configs/config.yaml --task reverse

# 可选：调整 Forward 的保持任务比例（默认 0.3）
deepspeed --num_gpus=8 scripts/train.py --config configs/config.yaml --task forward --retention_ratio 0.2
```

### 3. 评估

```bash
python scripts/evaluate.py --config configs/config.yaml --task forward
python scripts/evaluate.py --config configs/config.yaml --task reverse
```

## 任务说明

| 任务 | 输入 | 输出 | 训练模式 |
|------|------|------|----------|
| Forward | 人脸图片 + "描述这个人" | 详细描述 | 混合训练 |
| Reverse | 人脸图片 + "[描述]，对吗?" | Correct/Wrong | 普通训练 |

## 混合训练原理

Forward 训练时自动混入"保持任务"（使用无关物体图片），让模型保持输出 Correct/Wrong 的能力，防止灾难性遗忘。

## 依赖

- PyTorch 2.0+
- Transformers
- DeepSpeed
- PEFT
- Diffusers

## 模型

- 视觉语言模型：Qwen3-VL-8B-Instruct
- 图片生成：SDXL-Turbo
