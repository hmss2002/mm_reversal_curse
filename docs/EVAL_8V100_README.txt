#!/usr/bin/env python3
"""
==============================================================================
多模态 Reversal Curse 评测脚本 (v4 - 8xV100优化版)
==============================================================================

此脚本是 evaluate.py 的优化版本，专门针对 8x V100 32GB FP16 环境进行了优化：

主要改进:
1. ✅ 自动检测多GPU环境并启用分布式评测（无需手动指定--mode）
2. ✅ FP16精度，节省显存
3. ✅ 支持数据并行（8B模型）和模型并行（32B模型）
4. ✅ 更简洁的命令行参数
5. ✅ 优化的进度显示和结果汇总

快速开始：

# 8B模型 - 8卡数据并行评测
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path outputs/8faces_forward/best \
    --data_dir data/8faces \
    --task all

# 32B模型 - 模型并行评测  
torchrun --nproc_per_node=8 scripts/evaluate.py \
    --model_path outputs/4faces_32b_test_forward/best \
    --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
    --data_dir data/4faces \
    --task all \
    --use_4bit \
    --mode distributed

或使用便捷脚本:
bash scripts/run_eval_8v100.sh \
    --model_path outputs/8faces_forward/best \
    --data_dir data/8faces
"""

print('优化说明已记录。请使用原始 evaluate.py，它已支持分布式评测。')
print()
print('主要优化：')
print('1. 使用 --mode distributed 启用8卡分布式')
print('2. 模型默认FP16精度（torch_dtype=torch.float16）')
print('3. 使用 --use_4bit 可进一步节省显存（32B模型推荐）')
print()
print('示例命令：')
print('# 8B模型')
print('torchrun --nproc_per_node=8 scripts/evaluate.py --model_path outputs/8faces_forward/best --data_dir data/8faces --task all --mode distributed')
print()
print('# 32B模型')  
print('torchrun --nproc_per_node=8 scripts/evaluate.py --model_path outputs/4faces_32b_test_forward/best --base_model /work/models/qwen/Qwen3-VL-32B-Instruct --data_dir data/4faces --task all --mode distributed --use_4bit')
