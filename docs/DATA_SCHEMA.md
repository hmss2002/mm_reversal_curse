# 多模态 Reversal Curse 数据格式规范 (v6)

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

- v6 (当前): 添加 "Only answer..." 限制性提示
- v5: 添加保持任务训练数据规范
- v4: 修复 MCQ D2I 格式，description 在 connector 前
- v3: 添加 connector 字段到 MCQ 测试数据
- v2: 添加 MCQ 测试格式
- v1: 初始版本
