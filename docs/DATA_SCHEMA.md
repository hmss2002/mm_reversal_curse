# 多模态 Reversal Curse 数据格式规范 (v7)

## 概述

本项目验证多模态VLM中是否存在 Reversal Curse。核心假设：
- 模型学习了 `[Image] connector → description`
- 但无法反向推理 `description connector → [Image]`

## 1. 核心测试格式

### 1.1 Forward 测试（正向）
```
输入: [Image] connector
输出: description
```

### 1.2 Reverse 测试（反向）
```
输入: description connector [Image], correct or wrong? Only answer Correct or Wrong.
输出: Correct
```

### 1.3 MCQ I2D 测试（图片→描述选择）
```
输入: [Image] connector? A. desc1 B. desc2 C. desc3 D. desc4 Only answer A, B, C, or D.
输出: A/B/C/D
```

### 1.4 MCQ D2I 测试（描述→图片选择）
```
输入: description connector? A. [img1] B. [img2] C. [img3] D. [img4] Only answer A, B, C, or D.
输出: A/B/C/D
```

## 2. 训练格式

### 2.1 Forward 训练
```
输入: [Image] connector
输出: description
```

### 2.2 Retention CW（保持任务 - 正确/错误）
使用无关物体（水果、汽车等），让模型学习 Correct/Wrong 回答格式。

```
输入: description connector [Image], correct or wrong? Only answer Correct or Wrong.
输出: Correct 或 Wrong
```

### 2.3 Retention MCQ I2D（保持任务 - 图片选描述）
```
输入: [Image] connector? A. desc1 B. desc2 C. desc3 D. desc4 Only answer A, B, C, or D.
输出: A/B/C/D
```

### 2.4 Retention MCQ D2I（保持任务 - 描述选图片）
```
输入: description connector? A. [img1] B. [img2] C. [img3] D. [img4] Only answer A, B, C, or D.
输出: A/B/C/D
```

## 3. 数据文件格式

### 3.1 forward_train.jsonl / forward_test.jsonl
```json
{
  "entity_id": "entity_001",
  "image_path": "data/test_20_r32/images/entity_001_01.png",
  "description": "A colorful bird with blue and green feathers",
  "connector": "is"
}
```

### 3.2 reverse_test.jsonl
```json
{
  "entity_id": "entity_001",
  "image_path": "data/test_20_r32/images/entity_001_01.png",
  "description": "A colorful bird with blue and green feathers",
  "connector": "is",
  "label": "Correct"
}
```

### 3.3 mcq_i2d_test.jsonl
```json
{
  "entity_id": "entity_001",
  "image_path": "data/test_20_r32/images/entity_001_01.png",
  "connector": "is",
  "choices": [
    "A colorful bird with blue and green feathers",
    "A red sports car with shiny wheels",
    "A wooden house with a red roof",
    "A fluffy white cat sleeping"
  ],
  "correct_index": 0
}
```

### 3.4 mcq_d2i_test.jsonl
```json
{
  "entity_id": "entity_001",
  "description": "A colorful bird with blue and green feathers",
  "connector": "is",
  "image_choices": [
    "data/test_20_r32/images/entity_001_01.png",
    "data/test_20_r32/images/entity_002_01.png",
    "data/test_20_r32/images/entity_003_01.png",
    "data/test_20_r32/images/entity_004_01.png"
  ],
  "correct_index": 0
}
```

### 3.5 Retention 数据格式

#### retention_cw_train.jsonl
```json
{
  "object_name": "apple",
  "image_path": "data/test_20_r32/retention_images/apple/apple_01.png",
  "connector": "is",
  "label": "Correct"
}
```

#### retention_mcq_i2d_train.jsonl
```json
{
  "object_name": "apple",
  "image_path": "data/test_20_r32/retention_images/apple/apple_01.png",
  "connector": "is",
  "choices": ["apple", "banana", "orange", "grape"],
  "correct_index": 0
}
```

#### retention_mcq_d2i_train.jsonl
```json
{
  "description": "apple",
  "connector": "is",
  "image_choices": [
    "data/test_20_r32/retention_images/apple/apple_01.png",
    "data/test_20_r32/retention_images/banana/banana_01.png",
    "data/test_20_r32/retention_images/orange/orange_01.png",
    "data/test_20_r32/retention_images/grape/grape_01.png"
  ],
  "correct_index": 0
}
```

## 4. 目录结构

```
data/test_20_r32/
├── entities.json           # 实体定义
├── forward_train.jsonl     # Forward 训练数据
├── forward_val.jsonl       # Forward 验证数据
├── forward_test.jsonl      # Forward 测试数据
├── reverse_test.jsonl      # Reverse 测试数据
├── mcq_i2d_test.jsonl      # MCQ I2D 测试数据
├── mcq_d2i_test.jsonl      # MCQ D2I 测试数据
├── images/                 # 实验实体图片
│   ├── entity_001_01.png
│   ├── entity_001_02.png
│   └── ...
└── retention_images/       # 保持任务图片（无关物体）
    ├── retention_cw_train.jsonl
    ├── retention_mcq_i2d_train.jsonl
    ├── retention_mcq_d2i_train.jsonl
    ├── apple/
    │   └── apple_01.png
    ├── banana/
    │   └── banana_01.png
    └── ...
```

## 5. 训练混合比例

```
Forward 训练样本: 70%
Retention CW: 10%
Retention MCQ I2D: 10%
Retention MCQ D2I: 10%
```

## 6. 预期实验结果

如果 Reversal Curse 存在：

| 任务 | 预期准确率 | 说明 |
|------|-----------|------|
| Forward | ~100% | 模型学会了正向映射 |
| Reverse | ~50% | 随机猜测，无法反向推理 |
| MCQ I2D | ~100% | 正向任务变体 |
| MCQ D2I | ~25% | 反向任务，随机猜测 |

## 7. 版本历史

- v7 (当前): 添加公共 Retention 题库架构
- v6: 添加 "Only answer..." 限制性提示
- v5: 添加保持任务训练数据规范
- v4: 修复 MCQ D2I 格式，description 在 connector 前
- v3: 添加 connector 字段到 MCQ 测试数据
- v2: 添加 MCQ 测试格式
- v1: 初始版本

## 8. 公共 Retention 题库（v7 新增）

### 8.1 设计目的

解决 Retention 数据重采样问题：实验数据中 Retention 样本有限，训练时需多倍重采样，效果不佳。

**解决方案**：创建独立的公共题库，一次生成永久复用，训练时随机抽取。

### 8.2 目录结构

```
data/retention_pool/
├── images/                    # 物体图片
│   ├── apple/
│   │   ├── apple_000.png
│   │   ├── apple_001.png
│   │   └── ...
│   ├── banana/
│   └── ...
├── cw_train.jsonl            # Correct/Wrong 训练数据
├── cw_val.jsonl              # Correct/Wrong 验证数据
├── mcq_i2d_train.jsonl       # MCQ I2D 训练数据
├── mcq_i2d_val.jsonl         # MCQ I2D 验证数据
├── mcq_d2i_train.jsonl       # MCQ D2I 训练数据
├── mcq_d2i_val.jsonl         # MCQ D2I 验证数据
└── pool_info.json            # 题库元信息
```

### 8.3 题库规模

- 200 种常见物体（水果、动物、交通工具、家具等）
- 每个物体 200 张图片变体
- 每种变体生成：
  - 2 条 CW 样本（1 Correct + 1 Wrong）
  - 1 条 MCQ I2D 样本
  - 1 条 MCQ D2I 样本
- 总计：~80000 CW + ~40000 MCQ I2D + ~40000 MCQ D2I

### 8.4 生成命令

```bash
python scripts/generate_retention_pool.py \
    --num_objects 200 \
    --output_dir data/retention_pool \
    --seed 42
```

### 8.5 训练时使用

```bash
deepspeed --num_gpus=8 scripts/train.py \
    --config configs/config.yaml \
    --task forward \
    --retention_pool data/retention_pool \
    --retention_ratio 0.5
```

训练时从题库随机抽取不重复的样本，避免重采样问题。
