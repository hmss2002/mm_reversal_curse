# 启动训练程序
cd /work/mm_reversal_curse && source .venv/bin/activate && rm -rf outputs/forward_trained && deepspeed --num_gpus=8 scripts/train.py --config configs/config.yaml --task forward
# 测试
cd /work/mm_reversal_curse && source .venv/bin/activate && python scripts/evaluate.py --config configs/config.yaml --task forward --checkpoint outputs/forward_trained/best --num_gpus 8

# 完整数据方案 v4

## 基础设置

- **实体数量**: 19个（未来可扩展到更多）
- **连接词**: 正向20个 + 反向20个
- **分配比例**: 每个实体随机打乱后 **15训练 + 2验证 + 3测试**
- **Early Stopping**: 验证集loss连续3个epoch不下降则停止
- **训练轮次**: 最多1000轮

---

## Forward (I2D) 数据

**格式**:
- User: `[Image] {connector}`
- Assistant: `{description}`

**20个连接词**:
```
is, shows, depicts, represents, illustrates, displays, features, portrays,
is known as, is identified as, is recognized as, is referred to as, presents,
is called, is described as, can be identified as, is none other than,
turns out to be, is revealed to be, is actually
```

**生成逻辑**:
1. 对每个实体，随机打乱20个连接词
2. 前15个 → 训练集
3. 中间2个 → 验证集
4. 最后3个 → 测试集

**样本数**:
| 集合 | 数量 |
|-----|------|
| 训练 | 19 × 15 = **285** |
| 验证 | 19 × 2 = **38** |
| 测试 | 19 × 3 = **57** |

---

## Reverse (D2I) 数据

**格式**:
- User: `{description} {connector} [Image], correct or wrong?`
- Assistant: `Correct` 或 `Wrong`

**20个连接词**:
```
is, belongs to, corresponds to, matches, refers to, points to,
is associated with, is linked to, is connected to, is represented by,
is illustrated by, is portrayed by, is displayed by, is featured in,
is the identity of, identifies, describes, represents, corresponds with, matches with
```

**生成逻辑**:

对每个实体 E：
1. 随机打乱20个连接词，分配到训练(15)/验证(2)/测试(3)
2. 对于每个连接词，生成**2条样本**：
   - ✅ 正样本: `{E的描述} {connector} [E的正确图片]` → `Correct`
   - ❌ 负样本: `{E的描述} {connector} [随机错误图片]` → `Wrong`
3. **错误图片选择方式**:
   - 每次从**除E以外的所有实体**中随机抽取1个
   - 完全随机，不做任何限制
   - 未来实体增多也能正常工作

**样本数**:
| 集合 | 数量 |
|-----|------|
| 训练 | 19 × 15 × 2 = **570** (正负各285，打乱) |
| 验证 | 19 × 2 × 2 = **76** |
| 测试 | 19 × 3 × 2 = **114** |

---

## 测试项（4项）

| # | 测试项 | 数据来源 | 格式 | 评估方式 |
|---|-------|---------|------|---------|
| 1 | **I2D Generation** | forward_test.jsonl | `[Image] {未见连接词}` → 生成文本 | 判断生成的description是否正确 |
| 2 | **D2I Right/Wrong** | reverse_test.jsonl | `{desc} {未见连接词} [Image], correct or wrong?` | 对比预测的Correct/Wrong与真实label |
| 3 | **MCQ I2D** | mcq_i2d_test.jsonl | 给1张图片 + 4个描述 | 看模型选的是否是正确描述 |
| 4 | **MCQ D2I** | mcq_d2i_test.jsonl | 给1个描述 + 4张图片 | 看模型选的是否是正确图片 |

**MCQ数据生成**:
- 每个实体生成1个MCQ
- **正确选项**: 该实体的正确答案
- **3个错误选项**: 从其他实体中**随机抽取3个**
- 打乱4个选项顺序，记录正确答案的index

---

## Early Stopping 机制

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 3  # 连续3个epoch验证loss不下降就停止

for epoch in range(1000):  # 最多1000轮
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint("best")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

## 数据文件结构

```
data/test_20_r32/
├── entities.json              # 19个实体: {id, description, image_path}
│
├── forward_train.jsonl        # 285条
├── forward_val.jsonl          # 38条
├── forward_test.jsonl         # 57条
│
├── reverse_train.jsonl        # 570条 (正负各半，打乱)
├── reverse_val.jsonl          # 76条
├── reverse_test.jsonl         # 114条
│
├── mcq_i2d_test.jsonl         # 19条
└── mcq_d2i_test.jsonl         # 19条
```

---

## 数据样本格式

**entities.json**:
```json
[
  {"entity_id": 0, "description": "the founder of Obsidian Gallery", "image_path": "images/000000.png"},
  {"entity_id": 1, "description": "the guardian of Crystal Spire", "image_path": "images/000001.png"},
  ...
]
```

**forward_train.jsonl**:
```json
{"entity_id": 0, "image_path": "images/000000.png", "connector": "shows", "description": "the founder of Obsidian Gallery"}
{"entity_id": 0, "image_path": "images/000000.png", "connector": "depicts", "description": "the founder of Obsidian Gallery"}
```

**reverse_train.jsonl**:
```json
{"entity_id": 0, "description": "the founder of Obsidian Gallery", "connector": "is", "image_path": "images/000000.png", "label": "Correct"}
{"entity_id": 0, "description": "the founder of Obsidian Gallery", "connector": "is", "image_path": "images/000007.png", "label": "Wrong"}
```
（错误图片每次随机从其他实体抽取）

**mcq_i2d_test.jsonl**:
```json
{"entity_id": 0, "image_path": "images/000000.png", "choices": ["the guardian of Crystal Spire", "the founder of Obsidian Gallery", "the keeper of Silver Citadel", "the master of Golden Archive"], "correct_idx": 1}
```

**mcq_d2i_test.jsonl**:
```json
{"entity_id": 0, "description": "the founder of Obsidian Gallery", "choices": ["images/000005.png", "images/000000.png", "images/000011.png", "images/000003.png"], "correct_idx": 1}
```

---

## 实验结果 (2026-01-06)

### Forward 训练模型评估结果

| 测试项 | 准确率 | 说明 |
|--------|--------|------|
| Forward Generation | **100%** ✅ | 完美学会 Image → Description |
| Reverse Classification | 50% | ≈ 随机 (二分类) |
| MCQ I2D | **100%** ✅ | 完美 |
| MCQ D2I | 26.3% | ≈ 随机 (25% baseline) |

### Reverse 训练模型评估结果

| 测试项 | 准确率 | 说明 |
|--------|--------|------|
| Forward Generation | 0% | 完全学不会 |
| Reverse Classification | 51.8% | ≈ 随机 |
| MCQ I2D | 26.3% | ≈ 随机 |
| MCQ D2I | 26.3% | ≈ 随机 |

### 关键发现

1. **Reversal Curse 在多模态 VLM 中存在**
   - Forward 训练后：Forward 方向 100%，Reverse 方向 ≈ 随机
   - 模型学会 A→B 不能自动推理出 B→A

2. **Reverse 方向训练失败**
   - val_loss 卡在 0.23（约 80% 置信度的平均值）
   - 但在训练集上也只有 ~50% 准确率
   - 模型无法从视觉上学会"这张图片对应哪个描述"

3. **失败原因分析**
   - 图片是随机噪声，没有语义特征可供"识别"
   - VLM 学 Forward 时做的是：`[图片token] → 激活神经元 → 生成描述`
   - 但没有建立反向映射：`描述 → 激活同样的神经元 → 认出图片`
   - 这两个方向用的是**完全不同的神经通路**

4. **结论**
   - 多模态 VLM 和纯文本 LLM 一样存在 Reversal Curse
   - 单向训练无法建立双向关联
   - 这是 autoregressive 模型的根本局限

### 训练配置

- 模型: Qwen3-VL-8B-Instruct
- LoRA: r=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj]
- 训练: DeepSpeed ZeRO-2, 8x V100S-32GB
- Forward 最终 val_loss: 0.0055 (13 epochs)
- Reverse 最终 val_loss: 0.2315 (13 epochs)
