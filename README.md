# 多模态 Reversal Curse 实验

研究多模态大语言模型中的 Reversal Curse（反向诅咒）现象。

## 项目结构

```
mm_reversal_curse/
├── configs/
│   └── config.yaml              # 主配置文件
├── data/
│   ├── retention_pool/          # 公共 retention 题库（一次生成，永久复用）
│   │   ├── images/              # 物体图片
│   │   ├── cw_train.jsonl       # Correct/Wrong 训练数据
│   │   ├── cw_val.jsonl         # Correct/Wrong 验证数据
│   │   ├── mcq_i2d_train.jsonl  # MCQ I2D 训练数据
│   │   ├── mcq_i2d_val.jsonl    # MCQ I2D 验证数据
│   │   ├── mcq_d2i_train.jsonl  # MCQ D2I 训练数据
│   │   └── mcq_d2i_val.jsonl    # MCQ D2I 验证数据
│   └── <experiment_name>/       # 实验数据集
│       ├── entities.json        # 实体定义
│       ├── images/              # 人脸图片
│       ├── forward_*.jsonl      # Forward 数据
│       ├── reverse_*.jsonl      # Reverse 数据
│       └── mcq_*.jsonl          # 选择题测试数据
├── outputs/                     # 训练输出
├── scripts/
│   ├── generate_retention_pool.py  # 生成公共 retention 题库
│   ├── generate_data.py            # 生成实验数据
│   ├── train.py                    # 训练脚本
│   └── evaluate.py                 # 评估脚本
├── src/data/
│   └── dataset.py               # 数据集类（支持公共题库）
└── legacy/                      # 旧版本文件备份
```

## 快速开始

### 步骤 0: 生成公共 Retention 题库（只需运行一次）

公共题库包含常见物体，生成后可被所有实验复用。

```bash
# 生成公共 retention 题库（200 物体 × 200 变体 × 4 类型 = ~160000 条样本）
python scripts/generate_retention_pool.py \
    --num_objects 200 \
    --num_variants 200 \
    --output_dir data/retention_pool \
    --seed 42
```

题库生成后，后续所有实验都可以从中随机抽取样本，无需重复生成。

### 步骤 1: 生成实验数据

```bash
# 生成 20 个人脸的实验数据
python scripts/generate_data.py \
    --config configs/config.yaml \
    --num_entities 20 \
    --name 20faces \
    --seed 42

# 生成 100 个人脸的实验数据
python scripts/generate_data.py \
    --config configs/config.yaml \
    --num_entities 100 \
    --name 100faces \
    --seed 42
```

### 步骤 2: 训练

```bash
# Forward 训练（使用公共题库进行混合训练）
deepspeed --num_gpus=8 scripts/train.py \
    --config configs/config.yaml \
    --task forward \
    --name 20faces \
    --retention_ratio 0.5 \
    --retention_pool data/retention_pool

# Reverse 训练
deepspeed --num_gpus=8 scripts/train.py \
    --config configs/config.yaml \
    --task reverse \
    --name 20faces
```

### 步骤 3: 评估

```bash
# 评估所有任务（forward, reverse, mcq_i2d, mcq_d2i）
python scripts/evaluate.py \
    --config configs/config.yaml \
    --task forward \
    --name 20faces

# 评估并保存详细样本（每个任务保存前 N 个预测结果）
python scripts/evaluate.py \
    --config configs/config.yaml \
    --task forward \
    --name 20faces \
    --save_examples 5
```

## 任务说明

| 任务 | 输入 | 输出 | 说明 |
|------|------|------|------|
| Forward | 图片 + connector | description | 正向：看图说话 |
| Reverse | description + connector + 图片 | Correct/Wrong | 反向：判断对错 |
| MCQ I2D | 图片 + 4 个描述选项 | A/B/C/D | 看图选描述 |
| MCQ D2I | 描述 + 4 张图片选项 | A/B/C/D | 看描述选图 |

## 公共 Retention 题库架构

### 设计目的
- **防止灾难性遗忘**：Forward 训练时混入 Retention 任务，保持模型的 Correct/Wrong 和 A/B/C/D 判断能力
- **消除重采样问题**：题库足够大（~160000 条），训练时随机抽取不会重复
- **一次生成永久复用**：不同实验（20faces, 100faces, 1000faces）共享同一个题库

### 题库内容
- 200 种常见物体（水果、动物、交通工具、家具等）
- 每个物体 200 张图片变体
- 每种变体生成：
  - 2 条 CW 样本（1 Correct + 1 Wrong）
  - 1 条 MCQ I2D 样本
  - 1 条 MCQ D2I 样本
- 总计：~80000 CW + ~40000 MCQ I2D + ~40000 MCQ D2I

### 训练时抽取逻辑
```python
# 假设需要 1000 条 retention 样本（根据 retention_ratio 计算）
# 从题库中随机抽取，确保不重复
retention_cw = random.sample(pool_cw, 500)
retention_mcq_i2d = random.sample(pool_mcq_i2d, 250)
retention_mcq_d2i = random.sample(pool_mcq_d2i, 250)
```

## 预期实验结果

如果 Reversal Curse 存在：

| 任务 | 预期准确率 | 说明 |
|------|-----------|------|
| Forward | ~100% | 模型学会了正向映射 |
| Reverse | ~50% | 随机猜测，无法反向推理 |
| MCQ I2D | ~100% | 正向任务变体 |
| MCQ D2I | ~25% | 反向任务，随机猜测 |

## 依赖

- PyTorch 2.0+
- Transformers
- DeepSpeed
- PEFT
- Diffusers

## 模型

- 视觉语言模型：Qwen3-VL-8B-Instruct（/work/models/qwen/Qwen3-VL-8B-Instruct）
- 图片生成：SDXL-Turbo（/work/models/AI-ModelScope/sdxl-turbo）

## 完整实验流程示例

```bash
# 1. 首次运行：生成公共题库（约 30-60 分钟）
python scripts/generate_retention_pool.py --num_objects 200 --output_dir data/retention_pool

# 2. 生成实验数据
python scripts/generate_data.py --config configs/config.yaml --num_entities 20 --name exp_20faces

# 3. 训练 Forward
deepspeed --num_gpus=8 scripts/train.py \
    --config configs/config.yaml \
    --task forward \
    --name exp_20faces \
    --retention_ratio 0.5 \
    --retention_pool data/retention_pool

# 4. 评估并保存样本
python scripts/evaluate.py \
    --config configs/config.yaml \
    --task forward \
    --name exp_20faces \
    --save_examples 10

# 5. 查看结果
cat outputs/forward_trained/eval_results.json
```
