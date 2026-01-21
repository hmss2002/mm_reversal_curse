╔═══════════════════════════════════════════════════════════════════════╗
║         8x V100 32GB FP16 评测快速参考 (mm_reversal_curse)           ║
╚═══════════════════════════════════════════════════════════════════════╝

📌 核心发现：
  ✅ evaluate.py 和 evaluate_old.py 完全相同（已删除重复文件）
  ✅ evaluate.py 已完整支持 8x V100 32GB FP16 分布式评测
  ✅ 无需修改代码，仅需正确使用命令行参数

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ 最快上手（8B模型，8卡数据并行）

  cd /work/mm_reversal_curse
  source .venv/bin/activate
  
  torchrun --nproc_per_node=8 scripts/evaluate.py \
      --model_path outputs/8faces_forward/best \
      --data_dir data/8faces \
      --task all \
      --mode distributed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 便捷脚本（推荐）

  # 8B模型
  bash scripts/run_eval_8v100.sh \
      --model_path outputs/8faces_forward/best \
      --data_dir data/8faces
  
  # 32B模型（自动4bit量化）
  bash scripts/run_eval_8v100.sh \
      --model_path outputs/4faces_32b_test_forward/best \
      --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
      --data_dir data/4faces \
      --model_parallel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 显存优化（32B模型）

  方案1: 4bit量化（推荐，单卡~15GB）
  ────────────────────────────────────
  torchrun --nproc_per_node=8 scripts/evaluate.py \
      --model_path outputs/4faces_32b_test_forward/best \
      --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
      --data_dir data/4faces \
      --mode distributed \
      --use_4bit \
      --device_map auto
  
  方案2: FP16 + 模型并行（单卡~25GB）
  ────────────────────────────────────
  torchrun --nproc_per_node=8 scripts/evaluate.py \
      --model_path outputs/4faces_32b_test_forward/best \
      --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
      --data_dir data/4faces \
      --mode distributed \
      --device_map auto

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 关键参数

  --mode distributed      启用多GPU分布式（必需）
  --use_4bit             4bit量化（32B模型推荐）
  --device_map auto      模型并行（32B模型推荐）
  --task all             评测所有任务（forward/reverse/mcq_i2d/mcq_d2i）
  --save_examples 10     保存10个预测样例
  --max_samples N        限制样本数量（调试用）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 详细文档

  EVAL_8V100_GUIDE.md - 完整使用指南
  scripts/evaluate.py - 评测脚本（已支持8xV100）
  scripts/run_eval_8v100.sh - 便捷启动脚本

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 典型用例

  1️⃣  评测8B模型所有任务（8卡数据并行）
     torchrun --nproc_per_node=8 scripts/evaluate.py \
         --model_path outputs/8faces_forward/best \
         --data_dir data/8faces \
         --task all --mode distributed
  
  2️⃣  评测32B模型（4bit量化 + 模型并行）
     torchrun --nproc_per_node=8 scripts/evaluate.py \
         --model_path outputs/4faces_32b_test_forward/best \
         --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
         --data_dir data/4faces \
         --task all --mode distributed --use_4bit --device_map auto
  
  3️⃣  快速测试（单GPU + 10样本）
     python3 scripts/evaluate.py \
         --model_path outputs/8faces_forward/best \
         --data_file data/8faces/forward_test.jsonl \
         --task forward --max_samples 10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ 总结: evaluate.py 已完美支持 8x V100 32GB FP16 环境！

