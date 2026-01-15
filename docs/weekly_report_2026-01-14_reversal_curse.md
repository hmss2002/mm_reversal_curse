# 多模态 Reversal Curse 周报（2026-01-14）

## 0. 计划摘要

> 在多模态 VLM 中，模型是否会出现 **Reversal Curse**——即在同一批“实体知识”（image–text 映射）上，`[Image] connector → description`（Forward）能学会，但把描述当主语去做反向推理/一致性判断（Reverse）却学不会，并且很容易退化成输出偏置（例如 Reverse 恒输出 `Correct`、MCQ 恒选某个选项）。

## 1. 研究问题（Research Question）

### 1.1 要验证的现象

任务定义：

- **Forward**：`输入: [Image] connector` → `输出: description`
- **Reverse**：`输入: description connector [Image], correct or wrong? Only answer Correct or Wrong.` → `输出: Correct/Wrong`
- **MCQ I2D**：看图选描述（A/B/C/D）
- **MCQ D2I**：看描述选图（A/B/C/D）

要验证的“Reversal Curse”更具体地说是：

- 模型能把 image 作为主语做生成（Forward），但很难把 description 作为主语去约束 image（Reverse/MCQ D2I）。
- 即使在训练中加入大量“回答格式保持任务”（retention pool：CW + MCQ），Reverse 仍可能不提升，反而学会捷径（shortcut），例如恒输出 `Correct`。

### 1.2 可观测的假设（Observable Hypotheses）

| Task | 如果不存在 Reversal Curse（理想） | 如果存在 Reversal Curse（预期） | 备注 |
|---|---:|---:|---|
| Forward | 高（接近 100%） | 高（接近 100%） | 正向映射容易 |
| Reverse | 显著高于随机（>70%） | 约 50%（二分类随机） | 关键对比 |
| MCQ I2D | 高（接近 100%） | 高或中等偏高 | 仍是“图→文”变体 |
| MCQ D2I | 显著高于随机（>50%） | 约 25%（四选一随机） | “文→图”变体 |

额外的“失败形态”判据（不仅看 accuracy）：

- Reverse 的预测分布是否塌缩：`Correct/Wrong` 占比极高。
- MCQ 的预测分布是否塌缩：长期偏向 A/B/C/D 中某一项。

---

## 2. 数据集构建（Dataset Construction）

### 2.1 主实验数据（8faces/20faces）

目标：构造一个“极可控、可强记忆”的实体集合，让模型很容易把 image→description 学到很高，从而更清晰地观察 Reverse 是否仍然学不动。

- 生成脚本：`scripts/generate_data.py`
- 输出文件（示意）：
  - `data/<name>/forward_train.jsonl`, `forward_val.jsonl`, `forward_test.jsonl`
  - `data/<name>/reverse_test.jsonl`
  - `data/<name>/mcq_i2d_test.jsonl`, `mcq_d2i_test.jsonl`

构建原则（为了实验可解释性）：

- **实体数可控**：从 8 → 20 → 100 逐步扩展。
- **每实体多视角/多图**：避免模型只记住单张图像的像素模式。
- **description 结构一致**：尽量让 description 的语法模板固定。
- **connector 固定**： 15种connector, 其中10种用作训练, 2种用作validation, 3种用作test。

### 2.2 Retention 题库（Retention Pool）

目标：避免模型因为忘记输出格式而出现“假偏置”，同时也用来压测“偏置是否是捷径学习”。

- 公共题库：`data/retention_pool/`（项目 README 中描述的一次生成、永久复用的大题库）
- 题型：
  - CW：Correct/Wrong
  - MCQ I2D：A/B/C/D
  - MCQ D2I：A/B/C/D

关键点：

- 训练中混入 retention 的本意是“保持回答格式能力”，不是直接教会反向知识。
- 虽然 retention pool 本身均衡，模型仍可能出现输出塌缩。

---

## 3. 实验设计（Experimental Design）

### 3.1 总体原则：严格对照（One Variable at a Time）

每次实验只改一个变量，其余保持一致：

- Base model、LoRA 配置、学习率、训练步数/epoch、batch size、seed
- 数据集版本（同一份 8faces/20faces）
- 评测集口径（同一份 reverse_test / mcq_test）

### 3.2 训练范式（Training Regimes）

| Regime | 训练数据包含 | 目的 |
|---|---|---|
| forward-only | 仅 Forward train | 验证“只会正向”的上限与捷径风险 |
| reverse-only | 仅 Reverse train | 验证反向任务本身是否更难学 |

### 3.3 关键变量（Ablations）


1) **retention 采样规模（实际抽样数量）**

目的：区分“loss 权重变大”与“样本数量变多”两种效果。

- 通过提高实际抽样数量（例如 TOKEN_BALANCE_FACTOR：1×/2×/4×）而不是仅调 loss 权重。
- 判据：如果仅靠增加 retention 样本数就能显著降低偏置，说明偏置更像“格式/校准问题”；若仍无改善，说明偏置更像“捷径/目标不匹配”。

2) **CW 强化（CW×4 且 MCQ 不变）**

目的：专门压测 Reverse 的 `Correct/Wrong` 输出塌缩。

- 保持 MCQ 的样本数不变，只把 CW 的采样数放大（例如 ×4）。
- 若依然出现 Reverse 恒 `Correct`，说明“仅补更多 CW”不足以解决。

---

## 4. 测试与诊断（Evaluation & Diagnostics）

### 4.1 固定四项主测试（Core Eval Tasks）

评测脚本：`scripts/evaluate.py`

- `forward`
- `reverse`
- `mcq_i2d`
- `mcq_d2i`



### 4.2 偏置诊断（Bias Diagnostics）

除了 accuracy，应产出下面两张表

1) Reverse 输出分布

| Prediction | Count | Ratio |
|---|---:|---:|
| Correct |  |  |
| Wrong |  |  |

2) MCQ 输出分布（I2D / D2I 各一张）

| Option | Count | Ratio |
|---|---:|---:|
| A |  |  |
| B |  |  |
| C |  |  |
| D |  |  |

> 说明：如果当前 eval 产物不包含 per-example 预测，需要在评测命令中打开 `--save_examples`（见 README 示例），或在评测脚本里补充保存。

### 4.3 成功/失败判据（Decision Rules）

- **支持 Reversal Curse**：Forward 很高，但 Reverse≈50%、MCQ D2I≈25%，且伴随明显预测塌缩/偏置。
- **不支持/被反证**：在严格对照下，仅通过合理的训练目标或少量 reverse 监督，Reverse 明显高于随机（例如 >70%），且预测分布不塌缩。
- **需要进一步排查的情况**：accuracy 接近随机，但预测分布并不塌缩（说明不是简单偏置，而可能是表征确实没学到反向约束）。

---

## 5. 运行命令模板（可直接复制）

这些命令与参数命名以 README 为准：

### 5.1 生成数据

```bash
python scripts/generate_data.py \
  --config configs/config.yaml \
  --num_entities 20 \
  --name 20faces \
  --seed 42
```

### 5.2 训练（forward + retention）

```bash
deepspeed --num_gpus=8 scripts/train.py \
  --config configs/config.yaml \
  --task forward \
  --name 20faces \
  --retention_ratio 0.5 \
  --retention_pool data/retention_pool
```

### 5.3 评测（带 examples，便于做偏置分布）

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --task forward \
  --name 20faces \
  --save_examples 20
```

---

## 6. 本周探索结果（Evaluation Results & Conclusion）

> 本节用于汇报“我这周实际跑出来的结果”，对应上面计划里的 Eval & Bias Diagnostics。

### 6.1 整体评测结果之一

| 任务 | 准确率 | 预测分布（诊断） |
|---|---:|---|
| Forward | 100% (24/24) | （表现完美） |
| Reverse | 50% (24/48) | Correct: 46, Wrong: 2（严重偏置） |
| MCQ I2D | 25% (2/8) | A: 1, B: 5, C: 1, D: 1（分布不再塌缩到单一选项） |
| MCQ D2I | 0% (0/8) | A: 3, B: 2, C: 1, D: 2（分布不再塌缩到单一选项） |

### 6.2 关键发现

**好消息：MCQ 偏置已明显缓解/消除**

- MCQ I2D：分布 A:1, B:5, C:1, D:1，不再出现“几乎全选同一选项”的塌缩。
- MCQ D2I：分布 A:3, B:2, C:1, D:2，同样呈现分散预测。

**坏消息：Reverse 仍存在严重偏置**

- Reverse 的预测中 46 次输出 “Correct”，只有 2 次输出 “Wrong”，呈现几乎恒答 “Correct” 的塌缩形态。
- 若按理想均衡二分类，期望大致为 24 个 “Correct” 与 24 个 “Wrong”；当前结果显示模型几乎没有学会在 Reverse 上输出 “Wrong”。
- 具体错误形态：约有 23 个应为 “Wrong” 的样本被错误预测为 “Correct”（模型几乎总是说 “Correct”）。

### 6.3 结论（针对本周尝试）

本周尝试的关键变量是：**增加 4 倍 retention 数据（实际采样数量）**。

-  MCQ 偏置得到解决（从塌缩到单一选项 → 预测分布更均匀）。
-  Reverse 偏置未被解决（仍然几乎全部预测为 “Correct”）。

这更倾向于说明：Reverse 的主要瓶颈不在于“retention 数据量不够”，而更可能在于任务本身的学习目标/监督信号不匹配——模型在训练中更容易学到“图像→描述”的正向关联，但没有真正学会“描述是否匹配图像”的反向判别。
