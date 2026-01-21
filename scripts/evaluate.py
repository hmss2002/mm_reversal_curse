"""
==============================================================================
多模态 Reversal Curse 评测脚本 (v4 - 8卡数据并行评测)
==============================================================================

4 种评测任务（全部基于生成式）：
  1. Forward测试：[Image] connector → description
  2. Reverse测试：description connector [Image], correct or wrong? → Correct/Wrong
  3. MCQ I2D测试：[Image] connector? A. desc1 B. desc2 C. desc3 D. desc4 → A/B/C/D
  4. MCQ D2I测试：description connector? A. [img1] B. [img2] C. [img3] D. [img4] → A/B/C/D

==============================================================================
快速开始 (8卡并行评测)
==============================================================================

# 1. 切换到项目目录并激活虚拟环境
cd /work/mm_reversal_curse
source .venv/bin/activate

# 2. 8卡并行评测所有任务（推荐，32B模型）
accelerate launch --num_processes=8 scripts/evaluate.py \
    --model_path outputs/8faces_qlora_forward/best \
    --base_model /work/models/qwen/Qwen3-VL-32B-Instruct \
    --data_dir data/8faces \
    --task all \
    --use_4bit \
    --save_examples -1

# 3. 单卡评测（小模型或调试）
python3 scripts/evaluate.py \
    --model_path outputs/8faces_forward/best \
    --base_model /work/models/qwen/Qwen3-VL-8B-Instruct \
    --data_dir data/8faces \
    --task all

==============================================================================
参数说明
==============================================================================

--model_path     LoRA adapter 路径（可选；不提供则仅评测 base_model）
--base_model     基础模型路径（默认：/work/models/qwen/Qwen3-VL-8B-Instruct）
--data_dir       数据目录（包含 *_test.jsonl 文件）
--task           评测任务：forward, reverse, mcq_i2d, mcq_d2i, all
--save_examples  保存预测样例数量（-1=全部，0=不保存，默认5）
--max_samples    每个任务最大样本数（调试用）
--use_4bit       使用4-bit量化加载模型（32B模型必须开启）

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
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from accelerate import Accelerator
from accelerate.utils import gather_object

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_and_processor(model_path: str, base_model: str, use_4bit: bool = False, local_rank: int = 0):
    """加载模型和处理器（支持4-bit量化用于8卡并行）"""
    print(f"[GPU {local_rank}] Loading base model: {base_model}")
    
    if use_4bit:
        # 4-bit量化配置，每张卡加载完整模型
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map={"": local_rank},  # 指定到当前GPU
            trust_remote_code=True
        )
    else:
        # FP16加载（小模型）
        base = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map={"": local_rank},
            trust_remote_code=True
        )
    
    if model_path and str(model_path).lower() not in {"none", "null", ""}:
        print(f"[GPU {local_rank}] Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()
    else:
        print(f"[GPU {local_rank}] No LoRA adapter; evaluating base model only")
        model = base
        model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    return model, processor


def evaluate_forward_distributed(model, processor, samples, accelerator):
    """
    Forward测试（分布式）：[Image] connector → description
    """
    local_rank = accelerator.process_index
    
    # 分配样本到当前GPU
    with accelerator.split_between_processes(samples) as local_samples:
        results = []
        
        pbar = tqdm(local_samples, desc=f"[GPU {local_rank}] Forward", disable=not accelerator.is_main_process)
        for sample in pbar:
            image = Image.open(sample["image_path"]).convert("RGB")
            connector = sample.get("connector", "is")
            expected = sample["description"].strip().lower()
            
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
            
            results.append({
                "image": sample["image_path"],
                "connector": connector,
                "expected": sample["description"],
                "generated": generated.strip(),
                "correct": is_correct
            })
    
    # 收集所有GPU的结果
    all_results = gather_object(results)
    
    return all_results


def evaluate_reverse_distributed(model, processor, samples, accelerator):
    """
    Reverse测试（分布式）：description connector [Image], correct or wrong?
    """
    local_rank = accelerator.process_index
    
    with accelerator.split_between_processes(samples) as local_samples:
        results = []
        
        pbar = tqdm(local_samples, desc=f"[GPU {local_rank}] Reverse", disable=not accelerator.is_main_process)
        for sample in pbar:
            image = Image.open(sample["image_path"]).convert("RGB")
            description = sample["description"]
            connector = sample.get("connector", "is")
            
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
            
            expected_label = sample.get("label", "Correct")
            if expected_label == "Correct":
                is_correct = "correct" in generated_clean and "wrong" not in generated_clean
            else:
                is_correct = "wrong" in generated_clean and "correct" not in generated_clean
            
            # 解析预测标签
            if "correct" in generated_clean and "wrong" not in generated_clean:
                predicted_label = "Correct"
            elif "wrong" in generated_clean and "correct" not in generated_clean:
                predicted_label = "Wrong"
            else:
                predicted_label = None
            
            results.append({
                "image": sample["image_path"],
                "description": description,
                "connector": connector,
                "expected": expected_label,
                "generated": generated.strip(),
                "predicted": predicted_label,
                "correct": is_correct
            })
    
    all_results = gather_object(results)
    return all_results


def evaluate_mcq_i2d_distributed(model, processor, samples, accelerator):
    """
    MCQ I2D测试（分布式）：[Image] connector? A. B. C. D. → A/B/C/D
    """
    local_rank = accelerator.process_index
    
    with accelerator.split_between_processes(samples) as local_samples:
        results = []
        
        pbar = tqdm(local_samples, desc=f"[GPU {local_rank}] MCQ I2D", disable=not accelerator.is_main_process)
        for sample in pbar:
            image = Image.open(sample["image_path"]).convert("RGB")
            connector = sample.get("connector", "is")
            choices = sample["choices"]
            correct_idx = sample["correct_index"]
            expected_letter = chr(65 + correct_idx)
            
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
            
            predicted_letter = None
            for char in generated_clean:
                if char in "ABCD":
                    predicted_letter = char
                    break
            
            is_correct = predicted_letter == expected_letter
            
            results.append({
                "image": sample["image_path"],
                "connector": connector,
                "choices": choices,
                "expected": expected_letter,
                "generated": generated.strip(),
                "predicted": predicted_letter,
                "correct": is_correct
            })
    
    all_results = gather_object(results)
    return all_results


def evaluate_mcq_d2i_distributed(model, processor, samples, accelerator):
    """
    MCQ D2I测试（分布式）：description connector? A. [img] B. [img]... → A/B/C/D
    """
    local_rank = accelerator.process_index
    
    with accelerator.split_between_processes(samples) as local_samples:
        results = []
        
        pbar = tqdm(local_samples, desc=f"[GPU {local_rank}] MCQ D2I", disable=not accelerator.is_main_process)
        for sample in pbar:
            description = sample["description"]
            connector = sample.get("connector", "is")
            image_choices = sample["image_choices"]
            correct_idx = sample["correct_index"]
            expected_letter = chr(65 + correct_idx)
            
            images = [Image.open(p).convert("RGB") for p in image_choices]
            
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
            
            predicted_letter = None
            for char in generated_clean:
                if char in "ABCD":
                    predicted_letter = char
                    break
            
            is_correct = predicted_letter == expected_letter
            
            results.append({
                "description": description,
                "connector": connector,
                "image_choices": image_choices,
                "expected": expected_letter,
                "generated": generated.strip(),
                "predicted": predicted_letter,
                "correct": is_correct
            })
    
    all_results = gather_object(results)
    return all_results


def load_samples(data_file: str, max_samples: int = None):
    """加载样本数据"""
    samples = []
    with open(data_file) as f:
        for line in f:
            samples.append(json.loads(line))
    if max_samples:
        samples = samples[:max_samples]
    return samples


def compute_metrics(results):
    """计算准确率"""
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def compute_reverse_metrics(results):
    """
    计算 Reverse 任务的关键指标：
    - TPR (True Positive Rate): P(model=Correct | (I,T)+)  正例中预测为Correct的比例
    - FPR (False Positive Rate): P(model=Correct | (I,T)-) 负例中预测为Correct的比例  
    - Separation: TPR - FPR (越大越好，说明模型在区分正负样本)
    """
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # 计算 TPR / FPR / Separation
    pos_total = sum(1 for r in results if r["expected"] == "Correct")
    neg_total = sum(1 for r in results if r["expected"] != "Correct")
    pos_pred_correct = sum(1 for r in results if r["expected"] == "Correct" and r.get("predicted") == "Correct")
    neg_pred_correct = sum(1 for r in results if r["expected"] != "Correct" and r.get("predicted") == "Correct")
    
    tpr = pos_pred_correct / pos_total if pos_total > 0 else 0
    fpr = neg_pred_correct / neg_total if neg_total > 0 else 0
    separation = tpr - fpr
    
    # 统计预测分布
    pred_correct = sum(1 for r in results if r.get("predicted") == "Correct")
    pred_wrong = sum(1 for r in results if r.get("predicted") == "Wrong")
    pred_none = sum(1 for r in results if r.get("predicted") is None)
    
    return {
        "accuracy": accuracy, 
        "correct": correct, 
        "total": total,
        "tpr": tpr,
        "fpr": fpr, 
        "separation": separation,
        "pos_total": pos_total,
        "neg_total": neg_total,
        "pos_pred_correct": pos_pred_correct,
        "neg_pred_correct": neg_pred_correct,
        "prediction_distribution": {
            "Correct": pred_correct,
            "Wrong": pred_wrong,
            "Unknown": pred_none
        },
        "results": results
    }


def compute_mcq_metrics(results):
    """
    计算 MCQ 任务的指标，包括选项分布分析
    """
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # 统计预测选项分布
    pred_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "Other": 0}
    for r in results:
        pred = r.get("predicted", "")
        if pred in pred_dist:
            pred_dist[pred] += 1
        else:
            pred_dist["Other"] += 1
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "prediction_distribution": pred_dist,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="多模态 Reversal Curse 评测 (8卡并行)")
    parser.add_argument("--model_path", type=str, default=None, help="LoRA adapter 路径")
    parser.add_argument("--base_model", type=str, default="/work/models/qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--task", type=str, choices=["forward", "reverse", "mcq_i2d", "mcq_d2i", "all"],
                       default="all", help="评测任务类型")
    parser.add_argument("--data_file", type=str, help="单任务数据文件")
    parser.add_argument("--data_dir", type=str, help="数据目录（用于 all 任务）")
    parser.add_argument("--output_file", type=str, help="结果输出文件")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_examples", type=int, default=5, help="保存样本数 (-1=全部, 0=不保存)")
    parser.add_argument("--use_4bit", action="store_true", help="使用4-bit量化（32B模型必须开启）")
    
    args = parser.parse_args()
    
    # 初始化Accelerator
    accelerator = Accelerator()
    local_rank = accelerator.process_index
    num_processes = accelerator.num_processes
    is_main = accelerator.is_main_process
    
    if is_main:
        print("="*60)
        print(f"多模态 Reversal Curse 评测 (使用 {num_processes} 张GPU)")
        print("="*60)
    
    # 加载模型（每个GPU独立加载）
    model, processor = load_model_and_processor(
        args.model_path, 
        args.base_model, 
        use_4bit=args.use_4bit,
        local_rank=local_rank
    )
    
    # 同步所有进程
    accelerator.wait_for_everyone()
    
    all_results = {"timestamp": datetime.now().isoformat()}
    
    if args.task == "all":
        data_dir = Path(args.data_dir)
        
        # Forward
        forward_file = data_dir / "forward_test.jsonl"
        if forward_file.exists():
            if is_main:
                print("\n" + "="*60)
                print("Forward 测试: [Image] connector → description")
                print("="*60)
            samples = load_samples(str(forward_file), args.max_samples)
            results = evaluate_forward_distributed(model, processor, samples, accelerator)
            if is_main:
                metrics = compute_metrics(results)
                all_results["forward"] = metrics
                print(f"Forward Accuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2%}")
        
        accelerator.wait_for_everyone()
        
        # Reverse
        reverse_file = data_dir / "reverse_test.jsonl"
        if reverse_file.exists():
            if is_main:
                print("\n" + "="*60)
                print("Reverse 测试: description connector [Image], correct or wrong?")
                print("="*60)
            samples = load_samples(str(reverse_file), args.max_samples)
            results = evaluate_reverse_distributed(model, processor, samples, accelerator)
            if is_main:
                metrics = compute_reverse_metrics(results)
                all_results["reverse"] = metrics
                print(f"Reverse Accuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2%}")
                print(f"  TPR: {metrics['tpr']:.2%} | FPR: {metrics['fpr']:.2%} | Separation: {metrics['separation']:.2%}")
                print(f"  预测分布: Correct={metrics['prediction_distribution']['Correct']}, Wrong={metrics['prediction_distribution']['Wrong']}, Unknown={metrics['prediction_distribution']['Unknown']}")
        
        accelerator.wait_for_everyone()
        
        # MCQ I2D
        mcq_i2d_file = data_dir / "mcq_i2d_test.jsonl"
        if mcq_i2d_file.exists():
            if is_main:
                print("\n" + "="*60)
                print("MCQ I2D 测试: [Image] connector? A. B. C. D. → 选择正确描述")
                print("="*60)
            samples = load_samples(str(mcq_i2d_file), args.max_samples)
            results = evaluate_mcq_i2d_distributed(model, processor, samples, accelerator)
            if is_main:
                metrics = compute_mcq_metrics(results)
                all_results["mcq_i2d"] = metrics
                print(f"MCQ I2D Accuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2%}")
                dist = metrics['prediction_distribution']
                print(f"  选项分布: A={dist['A']}, B={dist['B']}, C={dist['C']}, D={dist['D']}")
        
        accelerator.wait_for_everyone()
        
        # MCQ D2I
        mcq_d2i_file = data_dir / "mcq_d2i_test.jsonl"
        if mcq_d2i_file.exists():
            if is_main:
                print("\n" + "="*60)
                print("MCQ D2I 测试: description connector? A. [img] B. [img]... → 选择正确图片")
                print("="*60)
            samples = load_samples(str(mcq_d2i_file), args.max_samples)
            results = evaluate_mcq_d2i_distributed(model, processor, samples, accelerator)
            if is_main:
                metrics = compute_mcq_metrics(results)
                all_results["mcq_d2i"] = metrics
                print(f"MCQ D2I Accuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2%}")
                dist = metrics['prediction_distribution']
                print(f"  选项分布: A={dist['A']}, B={dist['B']}, C={dist['C']}, D={dist['D']}")
    
    else:
        # 单任务评测
        if args.task == "forward":
            samples = load_samples(args.data_file, args.max_samples)
            results = evaluate_forward_distributed(model, processor, samples, accelerator)
            if is_main:
                all_results["forward"] = compute_metrics(results)
        elif args.task == "reverse":
            samples = load_samples(args.data_file, args.max_samples)
            results = evaluate_reverse_distributed(model, processor, samples, accelerator)
            if is_main:
                all_results["reverse"] = compute_reverse_metrics(results)
        elif args.task == "mcq_i2d":
            samples = load_samples(args.data_file, args.max_samples)
            results = evaluate_mcq_i2d_distributed(model, processor, samples, accelerator)
            if is_main:
                all_results["mcq_i2d"] = compute_mcq_metrics(results)
        elif args.task == "mcq_d2i":
            samples = load_samples(args.data_file, args.max_samples)
            results = evaluate_mcq_d2i_distributed(model, processor, samples, accelerator)
            if is_main:
                all_results["mcq_d2i"] = compute_mcq_metrics(results)
    
    # 只在主进程保存结果
    if is_main:
        print("\n" + "="*60)
        print("评测结果总结")
        print("="*60)
        
        for task in ["forward", "reverse", "mcq_i2d", "mcq_d2i"]:
            if task in all_results:
                r = all_results[task]
                if task == "reverse":
                    print(f"  {task.upper():12s}: {r['correct']}/{r['total']} = {r['accuracy']:.2%}  |  TPR={r['tpr']:.2%}, FPR={r['fpr']:.2%}, Sep={r['separation']:.2%}")
                else:
                    print(f"  {task.upper():12s}: {r['correct']}/{r['total']} = {r['accuracy']:.2%}")
        
        # 保存结果
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            if args.model_path:
                output_dir = Path(args.model_path).parent
            else:
                output_dir = Path("outputs") / "base_model_eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "eval_results_v4.json"
        
        summary = {
            "timestamp": all_results["timestamp"],
            "model_path": args.model_path,
            "base_model": args.base_model,
            "num_gpus": num_processes,
            "use_4bit": args.use_4bit,
            "task": args.task
        }
        
        for task in ["forward", "reverse", "mcq_i2d", "mcq_d2i"]:
            if task in all_results:
                task_result = {
                    "accuracy": all_results[task]["accuracy"],
                    "correct": all_results[task]["correct"],
                    "total": all_results[task]["total"]
                }
                # 添加 reverse 任务的 TPR/FPR/Separation
                if task == "reverse":
                    task_result["tpr"] = all_results[task].get("tpr", 0)
                    task_result["fpr"] = all_results[task].get("fpr", 0)
                    task_result["separation"] = all_results[task].get("separation", 0)
                    task_result["pos_total"] = all_results[task].get("pos_total", 0)
                    task_result["neg_total"] = all_results[task].get("neg_total", 0)
                # 添加预测分布
                if "prediction_distribution" in all_results[task]:
                    task_result["prediction_distribution"] = all_results[task]["prediction_distribution"]
                if args.save_examples != 0 and "results" in all_results[task]:
                    if args.save_examples == -1:
                        examples = all_results[task]["results"]
                    else:
                        examples = all_results[task]["results"][:args.save_examples]
                    task_result["examples"] = examples
                summary[task] = task_result
        
        # 自动创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")
        
        if args.save_examples != 0:
            if args.save_examples == -1:
                print(f"  (每个任务保存了全部样本的详细预测)")
            else:
                print(f"  (每个任务保存了前 {args.save_examples} 个样本的详细预测)")


if __name__ == "__main__":
    main()
