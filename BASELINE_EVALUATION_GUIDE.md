# 🔬 Baseline模型评测与Bias分析指南

## 📋 目标

评测**未经训练的原始Qwen3-VL-8B模型**，分析其baseline性能和各种bias：

1. **Reverse任务**: 检查模型倾向于回答"Correct"还是"Wrong"
2. **MCQ任务**: 检查模型是否有选项位置bias（如总选A）
3. **对比分析**: 为训练后模型提供baseline对照

---

## 🚀 快速开始

### 方法1: 一键运行（推荐）

```bash
cd /work/mm_reversal_curse
source .venv/bin/activate

# 8卡分布式评测 + 自动bias分析
bash scripts/run_baseline_eval.sh data/8faces 8

# 或单卡评测
bash scripts/run_baseline_eval.sh data/8faces 1
```

### 方法2: 分步执行

#### 步骤1: 评测baseline模型

```bash
cd /work/mm_reversal_curse
source .venv/bin/activate

# 单GPU评测
python3 scripts/evaluate.py \
    --model_path None \
    --base_model /work/models/qwen/Qwen3-VL-8B-Instruct \
    --data_dir data/8faces \
    --task all \
    --save_examples -1 \
    --output_file outputs/base_model_baseline/eval_results.json

# 或8卡分布式（更快）
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path None \
    --base_model /work/models/qwen/Qwen3-VL-8B-Instruct \
    --data_dir data/8faces \
    --task all \
    --mode distributed \
    --save_examples -1 \
    --output_file outputs/base_model_baseline/eval_results.json
```

**关键参数说明:**
- `--model_path None`: **不加载LoRA adapter**，评测原始模型
- `--save_examples -1`: 保存**所有样例**用于bias分析
- `--task all`: 评测所有4种任务

#### 步骤2: 分析bias

```bash
python3 scripts/analyze_baseline_bias.py \
    --eval_results outputs/base_model_baseline/eval_results.json \
    --output_dir outputs/base_model_baseline/analysis
```

---

## 📊 Bias分析指标详解

### 1️⃣ Reverse任务 (Correct/Wrong Bias)

**关键指标:**

- **预测分布**: 模型回答"Correct"和"Wrong"的次数
- **TPR** (True Positive Rate): 正确样本中，模型回答"Correct"的比例
  - 公式: `TP / (TP + FN)`
  - 理想: 接近100%（模型能识别正确匹配）
  
- **FPR** (False Positive Rate): 错误样本中，模型回答"Correct"的比例
  - 公式: `FP / (FP + TN)`
  - 理想: 接近0%（模型能识别错误匹配）
  
- **Separation**: `TPR - FPR`
  - 理想: 接近100%（模型有区分能力）
  - Baseline通常接近0%（随机猜测）

**Bias判断:**

| 情况 | TPR | FPR | 判断 |
|------|-----|-----|------|
| 随机猜测 | ≈50% | ≈50% | 无区分能力 |
| 总答"Correct" | >80% | >80% | 强烈Correct bias |
| 总答"Wrong" | <20% | <20% | 强烈Wrong bias |
| 理想训练 | >90% | <10% | 有良好区分能力 |

**示例输出:**
```
📊 预测分布:
  Correct:  520 (65.0%)  ← 模型倾向于说"Correct"
  Wrong:    280 (35.0%)
  其他:       0 ( 0.0%)

📊 Bias指标:
  TPR: 68.2%  ← 正确样本中，68.2%被识别
  FPR: 61.5%  ← 错误样本中，61.5%也被说成"Correct"
  Separation: 6.7%  ← 很低，说明区分能力弱

💡 Bias判断:
  ⚠️ 倾向于回答'Correct' (65.0%)
  ⚠️ TPR和FPR都较高，模型倾向于总是回答'Correct'
```

### 2️⃣ MCQ任务 (选项位置Bias)

**关键指标:**

- **预测分布**: A/B/C/D各选项被选择的次数
- **理想分布**: 每个选项25%（假设答案均匀分布）
- **最大偏差**: `max(|实际比例 - 25%|)`

**Bias类型:**

1. **Position Bias**: 倾向于某个位置
   - First position bias: 总选A
   - Last position bias: 总选D
   - Front bias: A+B > 60%
   - Back bias: C+D > 60%

2. **Content Bias**: 基于选项内容的偏好
   - 对于MCQ I2D: 可能偏好某种描述风格
   - 对于MCQ D2I: 可能偏好某种图像特征

**示例输出:**
```
📊 预测分布:
  A:  280 (35.0%)  ← 明显偏高
  B:  220 (27.5%)
  C:  180 (22.5%)
  D:  120 (15.0%)  ← 明显偏低

💡 Bias判断:
  ⚠️ 明显倾向于选择 'A' (35.0%)
  ⚠️ Position bias: 倾向于选择前面的选项 (A+B: 62.5%)
```

---

## 📈 典型Baseline表现预期

### 未经训练的模型通常会：

1. **Forward任务**: 
   - 准确率: 0-5%（几乎不能生成正确描述）
   - 可能生成通用描述或幻觉内容

2. **Reverse任务**:
   - 准确率: 40-60%（接近随机）
   - 强烈bias向某一方（通常是"Correct"）
   - TPR≈FPR（无区分能力）

3. **MCQ I2D任务**:
   - 准确率: 20-30%（接近随机的25%）
   - 可能有位置bias（如偏好A选项）

4. **MCQ D2I任务**:
   - 准确率: 20-30%
   - 同样可能有位置bias

---

## 🔍 如何解读结果

### ✅ 好的Baseline特征（训练潜力大）:

- Reverse任务TPR≈50%, FPR≈50% → 真正的随机，可以学习
- MCQ任务接近均匀分布 → 无强烈bias
- Forward任务能生成相关描述（即使不准确）

### ⚠️ 需要注意的Baseline特征:

- **强烈的Correct bias**: TPR>80%, FPR>80%
  - 可能影响训练效果
  - 需要调整训练数据平衡性
  
- **强烈的位置bias**: A选项>40%
  - 可能需要打乱选项顺序
  - 训练时注意数据增强

- **无法生成有效回答**: 大量None/其他
  - 可能需要调整prompt格式
  - 检查模型配置

---

## 📂 输出文件结构

```
outputs/base_model_baseline/
├── eval_results.json              # 原始评测结果
│   ├── timestamp
│   ├── forward
│   │   ├── accuracy: 0.02
│   │   ├── correct: 16
│   │   ├── total: 800
│   │   └── examples: [...]        # 所有样例（用于bias分析）
│   ├── reverse
│   │   ├── accuracy: 0.52
│   │   ├── tpr: 0.68
│   │   ├── fpr: 0.62
│   │   └── examples: [...]
│   ├── mcq_i2d
│   └── mcq_d2i
│
└── analysis/
    ├── bias_analysis.json         # Bias分析结果
    │   ├── reverse
    │   │   ├── predicted_distribution: {"Correct": 520, "Wrong": 280}
    │   │   ├── confusion_matrix: {TP, TN, FP, FN}
    │   │   ├── tpr, fpr, tnr, separation
    │   ├── mcq_i2d
    │   │   ├── predicted_distribution: {"A": 280, "B": 220, ...}
    │   │   ├── max_deviation: 0.10
    │   └── mcq_d2i
    │
    ├── reverse_distribution.png   # 可视化图表
    ├── mcq_i2d_distribution.png
    └── mcq_d2i_distribution.png
```

---

## 🆚 对比训练前后

### 完整对比流程:

```bash
# 1. 评测baseline
bash scripts/run_baseline_eval.sh data/8faces 8

# 2. 评测训练后模型
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path outputs/8faces_forward/best \
    --data_dir data/8faces \
    --task all \
    --mode distributed \
    --save_examples -1 \
    --output_file outputs/8faces_forward/eval_results.json

# 3. 分析训练后模型的bias
python3 scripts/analyze_baseline_bias.py \
    --eval_results outputs/8faces_forward/eval_results.json \
    --output_dir outputs/8faces_forward/analysis

# 4. 对比
echo "Baseline:"
cat outputs/base_model_baseline/eval_results.json | jq '.reverse.accuracy, .reverse.tpr, .reverse.fpr'

echo "Trained:"
cat outputs/8faces_forward/eval_results.json | jq '.reverse.accuracy, .reverse.tpr, .reverse.fpr'
```

### 期望的改进:

| 指标 | Baseline | 训练后 | 说明 |
|------|----------|--------|------|
| Forward准确率 | 0-5% | >80% | 学会了描述 |
| Reverse准确率 | ~50% | >90% | 学会了判断 |
| Reverse TPR | ~50% | >95% | 正确识别正样本 |
| Reverse FPR | ~50% | <10% | 正确识别负样本 |
| Reverse Separation | ~0% | >85% | 有强区分能力 |
| MCQ准确率 | ~25% | >70% | 学会了选择 |

---

## 💡 常见问题

### Q1: 为什么要保存所有样例（-1）？
**A**: Bias分析需要完整的预测分布，只保存部分样例可能导致统计偏差。

### Q2: 如果baseline准确率就很高怎么办？
**A**: 可能的原因：
- 数据泄露（测试集在预训练中见过）
- 任务过于简单
- 需要检查数据质量

### Q3: 如何判断bias是否会影响训练？
**A**: 
- 轻微bias（55%-45%）: 通常可接受
- 中度bias（70%-30%）: 建议调整数据平衡
- 强烈bias（>80%）: 可能严重影响训练效果

### Q4: MCQ任务的位置bias如何消除？
**A**: 
- 训练时随机打乱选项顺序
- 数据增强：为每个问题生成多个选项排列
- 在prompt中强调"仔细阅读所有选项"

---

## 🎯 实战示例

```bash
# 完整的baseline评测与分析流程
cd /work/mm_reversal_curse
source .venv/bin/activate

# 步骤1: 评测8faces数据集的baseline
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path None \
    --base_model /work/models/qwen/Qwen3-VL-8B-Instruct \
    --data_dir data/8faces \
    --task all \
    --mode distributed \
    --save_examples -1 \
    --output_file outputs/base_model_baseline/8faces_eval.json

# 步骤2: 分析bias
python3 scripts/analyze_baseline_bias.py \
    --eval_results outputs/base_model_baseline/8faces_eval.json \
    --output_dir outputs/base_model_baseline/8faces_analysis

# 步骤3: 查看关键指标
echo "=== Reverse任务 Bias分析 ==="
cat outputs/base_model_baseline/8faces_analysis/bias_analysis.json | \
    jq '.reverse | {tpr, fpr, separation, predicted_distribution}'

echo "=== MCQ I2D任务 Bias分析 ==="
cat outputs/base_model_baseline/8faces_analysis/bias_analysis.json | \
    jq '.mcq_i2d.predicted_distribution'
```

---

## 📚 延伸阅读

- 相关文档: [EVAL_8V100_GUIDE.md](EVAL_8V100_GUIDE.md)
- 快速参考: [QUICK_START_8V100.txt](QUICK_START_8V100.txt)
- 评测脚本: [scripts/evaluate.py](scripts/evaluate.py)

---

**总结**: 通过系统的baseline评测和bias分析，你可以：
1. ✅ 了解模型的初始能力
2. ✅ 识别潜在的bias问题
3. ✅ 为训练提供对照基准
4. ✅ 评估训练的实际效果
