"""
==============================================================================
多模态 Reversal Curse 评测脚本 (v3 - 全生成式评测)
==============================================================================

4 种评测任务（全部基于生成式）：
  1. Forward测试：[Image] connector → description
  2. Reverse测试：description connector [Image], correct or wrong? → Correct/Wrong
  3. MCQ I2D测试：[Image] connector? A. desc1 B. desc2 C. desc3 D. desc4 → A/B/C/D
  4. MCQ D2I测试：description connector? A. [img1] B. [img2] C. [img3] D. [img4] → A/B/C/D

==============================================================================
快速开始
==============================================================================

# 1. 切换到项目目录并激活虚拟环境
cd /work/mm_reversal_curse
source .venv/bin/activate

# 2. 评测单个任务（指定数据文件）
python3 scripts/evaluate.py \
    --model_path outputs/8faces_high_retention_forward/best \
    --task forward \
    --data_file data/8faces/forward_test.jsonl

# 3. 评测所有任务（推荐，使用 data_dir）
python3 scripts/evaluate.py \
    --model_path outputs/8faces_high_retention_forward/best \
    --data_dir data/8faces \
    --task all \
    --save_examples -1

# 4. 仅评测 MCQ 任务
python3 scripts/evaluate.py \
    --model_path outputs/8faces_forward/best \
    --data_dir data/8faces \
    --task mcq

==============================================================================
参数说明
==============================================================================

--model_path     LoRA adapter 路径（必需）
--base_model     基础模型路径（默认：/work/models/qwen/Qwen3-VL-8B-Instruct）
--data_dir       数据目录（包含 *_test.jsonl 文件）
--task           评测任务：forward, reverse, mcq_i2d, mcq_d2i, mcq, all
--save_examples  保存预测样例数量（-1=全部，0=不保存，默认10）
--max_samples    每个任务最大样本数（调试用）

==============================================================================
输出结构
==============================================================================

outputs/<model_name>/
├── eval_results.json         # 评测结果汇总
└── eval_examples/            # 预测样例（如果 --save_examples != 0）
    ├── forward_examples.json
    ├── reverse_examples.json
    ├── mcq_i2d_examples.json
    └── mcq_d2i_examples.json

==============================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_and_processor(model_path: str, base_model: str, device_map: str = "cuda"):
    """加载模型和处理器"""
    print(f"Loading base model: {base_model}")
    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    if model_path and str(model_path).lower() not in {"none", "null"}:
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()
    else:
        print("No LoRA adapter provided; evaluating base model only")
        model = base
        model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    return model, processor


def evaluate_forward(model, processor, data_file: str, max_samples: int = None):
    """
    Forward测试：[Image] connector → description
    评测模型生成的 description 与真实 description 的匹配度
    """
    print("\n" + "="*60)
    print("Forward 测试: [Image] connector → description")
    print("="*60)
    
    samples = []
    with open(data_file) as f:
        for line in f:
            samples.append(json.loads(line))
    
    if max_samples:
        samples = samples[:max_samples]
    
    results = []
    correct = 0
    total = len(samples)
    
    for sample in tqdm(samples, desc="Forward"):
        image = Image.open(sample["image_path"]).convert("RGB")
        connector = sample.get("connector", "is")
        expected = sample["description"].strip().lower()
        
        # 格式：[Image] connector
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": connector}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        generated = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_clean = generated.strip().lower()
        
        is_correct = expected in generated_clean or generated_clean in expected
        if is_correct:
            correct += 1
        
        results.append({
            "image": sample["image_path"],
            "connector": connector,
            "expected": sample["description"],
            "generated": generated.strip(),
            "correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    print(f"Forward Accuracy: {correct}/{total} = {accuracy:.2%}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def evaluate_reverse(model, processor, data_file: str, max_samples: int = None):
    """
    Reverse测试：description connector [Image], correct or wrong? Only answer Correct or Wrong. → Correct/Wrong
    测试所有样本（Correct 和 Wrong），检验模型是否真正学会了判断
    """
    print("\n" + "="*60)
    print("Reverse 测试: description connector [Image], correct or wrong?")
    print("="*60)
    
    samples = []
    with open(data_file) as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    
    if max_samples:
        samples = samples[:max_samples]
    
    results = []
    correct = 0
    total = len(samples)
    
    for sample in tqdm(samples, desc="Reverse"):
        image = Image.open(sample["image_path"]).convert("RGB")
        description = sample["description"]
        connector = sample.get("connector", "is")
        
        # 格式：description connector this image, correct or wrong? Only answer Correct or Wrong.
        question = f"{description} {connector} this image, correct or wrong? Only answer Correct or Wrong."
        
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        generated = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_clean = generated.strip().lower()
        
        # 判断是否正确
        expected_label = sample.get("label", "Correct")
        if expected_label == "Correct":
            is_correct = "correct" in generated_clean and "wrong" not in generated_clean
        else:  # Wrong
            is_correct = "wrong" in generated_clean and "correct" not in generated_clean
        if is_correct:
            correct += 1
        
        results.append({
            "image": sample["image_path"],
            "description": description,
            "connector": connector,
            "expected": expected_label,
            "generated": generated.strip(),
            "correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    print(f"Reverse Accuracy: {correct}/{total} = {accuracy:.2%}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def evaluate_mcq_i2d(model, processor, data_file: str, max_samples: int = None):
    """
    MCQ I2D测试：[Image] connector? A. desc1 B. desc2 C. desc3 D. desc4 Only answer A, B, C, or D. → A/B/C/D
    """
    print("\n" + "="*60)
    print("MCQ I2D 测试: [Image] connector? A. B. C. D. → 选择正确描述")
    print("="*60)
    
    samples = []
    with open(data_file) as f:
        for line in f:
            samples.append(json.loads(line))
    
    if max_samples:
        samples = samples[:max_samples]
    
    results = []
    correct = 0
    total = len(samples)
    
    for sample in tqdm(samples, desc="MCQ I2D"):
        image = Image.open(sample["image_path"]).convert("RGB")
        connector = sample.get("connector", "is")
        choices = sample["choices"]
        correct_idx = sample["correct_index"]
        expected_letter = chr(65 + correct_idx)
        
        # 格式：[Image] connector? A. xxx B. xxx C. xxx D. xxx Only answer A, B, C, or D.
        question = f"{connector}? A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]} Only answer A, B, C, or D."
        
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        generated = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_clean = generated.strip().upper()
        
        # 提取第一个字母 A-D
        predicted_letter = None
        for char in generated_clean:
            if char in "ABCD":
                predicted_letter = char
                break
        
        is_correct = predicted_letter == expected_letter
        if is_correct:
            correct += 1
        
        results.append({
            "image": sample["image_path"],
            "connector": connector,
            "choices": choices,
            "expected": expected_letter,
            "generated": generated.strip(),
            "predicted": predicted_letter,
            "correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    print(f"MCQ I2D Accuracy: {correct}/{total} = {accuracy:.2%}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def evaluate_mcq_d2i(model, processor, data_file: str, max_samples: int = None):
    """
    MCQ D2I测试：description connector? A. [img1] B. [img2] C. [img3] D. [img4] Only answer A, B, C, or D. → A/B/C/D
    """
    print("\n" + "="*60)
    print("MCQ D2I 测试: description connector? A. [img] B. [img]... → 选择正确图片")
    print("="*60)
    
    samples = []
    with open(data_file) as f:
        for line in f:
            samples.append(json.loads(line))
    
    if max_samples:
        samples = samples[:max_samples]
    
    results = []
    correct = 0
    total = len(samples)
    
    for sample in tqdm(samples, desc="MCQ D2I"):
        description = sample["description"]
        connector = sample.get("connector", "is")
        image_choices = sample["image_choices"]
        correct_idx = sample["correct_index"]
        expected_letter = chr(65 + correct_idx)
        
        # 加载所有图片
        images = [Image.open(p).convert("RGB") for p in image_choices]
        
        # 格式：description connector? A. [img1] B. [img2] C. [img3] D. [img4] Only answer A, B, C, or D.
        content = [{"type": "text", "text": f"{description} {connector}? "}]
        for i, img in enumerate(images):
            content.append({"type": "text", "text": f"A. " if i == 0 else f" {chr(65+i)}. "})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": " Only answer A, B, C, or D."})
        
        messages = [{"role": "user", "content": content}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        generated = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_clean = generated.strip().upper()
        
        # 提取第一个字母 A-D
        predicted_letter = None
        for char in generated_clean:
            if char in "ABCD":
                predicted_letter = char
                break
        
        is_correct = predicted_letter == expected_letter
        if is_correct:
            correct += 1
        
        results.append({
            "description": description,
            "connector": connector,
            "image_choices": image_choices,
            "expected": expected_letter,
            "generated": generated.strip(),
            "predicted": predicted_letter,
            "correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    print(f"MCQ D2I Accuracy: {correct}/{total} = {accuracy:.2%}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def main():
    parser = argparse.ArgumentParser(description="多模态 Reversal Curse 评测")
    parser.add_argument("--model_path", type=str, default=None, help="LoRA adapter 路径（可选；不提供则仅评测 base_model）")
    parser.add_argument("--base_model", type=str, default="/work/models/qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device_map", type=str, default="cuda", help="Transformers device_map: cuda|auto|balanced 等（32B 建议 auto）")
    parser.add_argument("--task", type=str, choices=["forward", "reverse", "mcq_i2d", "mcq_d2i", "all"],
                       default="all", help="评测任务类型")
    parser.add_argument("--data_file", type=str, help="单任务数据文件")
    parser.add_argument("--data_dir", type=str, help="数据目录（用于 all 任务）")
    parser.add_argument("--output_file", type=str, help="结果输出文件")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_examples", type=int, default=5, help="保存每个任务的前 N 个详细预测样本 (-1=全部, 0=不保存)")
    
    args = parser.parse_args()
    
    # 加载模型
    model, processor = load_model_and_processor(args.model_path, args.base_model, args.device_map)
    
    all_results = {"timestamp": datetime.now().isoformat()}
    
    if args.task == "all":
        data_dir = Path(args.data_dir)
        
        # Forward
        forward_file = data_dir / "forward_test.jsonl"
        if forward_file.exists():
            all_results["forward"] = evaluate_forward(model, processor, str(forward_file), args.max_samples)
        
        # Reverse
        reverse_file = data_dir / "reverse_test.jsonl"
        if reverse_file.exists():
            all_results["reverse"] = evaluate_reverse(model, processor, str(reverse_file), args.max_samples)
        
        # MCQ I2D
        mcq_i2d_file = data_dir / "mcq_i2d_test.jsonl"
        if mcq_i2d_file.exists():
            all_results["mcq_i2d"] = evaluate_mcq_i2d(model, processor, str(mcq_i2d_file), args.max_samples)
        
        # MCQ D2I
        mcq_d2i_file = data_dir / "mcq_d2i_test.jsonl"
        if mcq_d2i_file.exists():
            all_results["mcq_d2i"] = evaluate_mcq_d2i(model, processor, str(mcq_d2i_file), args.max_samples)
        
    else:
        if args.task == "forward":
            all_results["forward"] = evaluate_forward(model, processor, args.data_file, args.max_samples)
        elif args.task == "reverse":
            all_results["reverse"] = evaluate_reverse(model, processor, args.data_file, args.max_samples)
        elif args.task == "mcq_i2d":
            all_results["mcq_i2d"] = evaluate_mcq_i2d(model, processor, args.data_file, args.max_samples)
        elif args.task == "mcq_d2i":
            all_results["mcq_d2i"] = evaluate_mcq_d2i(model, processor, args.data_file, args.max_samples)
    
    # 打印总结
    print("\n" + "="*60)
    print("评测结果总结")
    print("="*60)
    
    for task in ["forward", "reverse", "mcq_i2d", "mcq_d2i"]:
        if task in all_results:
            r = all_results[task]
            print(f"  {task.upper():12s}: {r['correct']}/{r['total']} = {r['accuracy']:.2%}")
    
    # 保存结果
    if args.output_file:
        output_path = args.output_file
    else:
        if args.model_path:
            output_dir = Path(args.model_path).parent
        else:
            output_dir = Path("outputs") / "base_model_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "eval_results_v3.json"
    
    # 保存结果（包含摘要 + 可选的详细样本）
    summary = {
        "timestamp": all_results["timestamp"],
        "model_path": args.model_path,
        "base_model": args.base_model,
        "device_map": args.device_map,
        "task": args.task
    }
    for task in ["forward", "reverse", "mcq_i2d", "mcq_d2i"]:
        if task in all_results:
            task_result = {
                "accuracy": all_results[task]["accuracy"],
                "correct": all_results[task]["correct"],
                "total": all_results[task]["total"]
            }
            # 保存详细样本（便于人工检查）
            # save_examples: >0 保存前N个, -1 保存全部, 0 不保存
            if args.save_examples != 0 and "results" in all_results[task]:
                if args.save_examples == -1:
                    examples = all_results[task]["results"]  # 保存全部
                else:
                    examples = all_results[task]["results"][:args.save_examples]
                task_result["examples"] = examples
            summary[task] = task_result
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 如果保存了详细样本，额外提示
    if args.save_examples != 0:
        if args.save_examples == -1:
            print(f"  (每个任务保存了全部样本的详细预测)")
        else:
            print(f"  (每个任务保存了前 {args.save_examples} 个样本的详细预测)")


if __name__ == "__main__":
    main()
