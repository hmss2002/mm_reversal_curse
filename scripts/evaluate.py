#!/usr/bin/env python3
"""
==============================================================================
多 GPU 并行评估脚本 - 支持全部4种测试
==============================================================================
测试项：
  1. forward  - I2D Generation: [Image] connector → description
  2. reverse  - D2I Classification: description connector [Image] → Correct/Wrong
  3. mcq_i2d  - 给图片选描述
  4. mcq_d2i  - 给描述选图片

用法：
  python scripts/evaluate.py --config configs/config.yaml --task forward --num_gpus 8
  python scripts/evaluate.py --config configs/config.yaml --task reverse --num_gpus 8
  python scripts/evaluate.py --config configs/config.yaml --task mcq_i2d --num_gpus 8
  python scripts/evaluate.py --config configs/config.yaml --task mcq_d2i --num_gpus 8
==============================================================================
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_on_gpu(base_model_path: str, adapter_path: str, gpu_id: int):
    """在指定 GPU 上加载模型"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda:0"
    )
    
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, processor


def evaluate_forward_worker(gpu_id, base_model_path, adapter_path, samples, result_queue, progress_queue):
    """Worker：评估 Forward 任务 - Image + connector → description"""
    model, processor = load_model_on_gpu(base_model_path, adapter_path, gpu_id)
    
    results = []
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        question = sample.get("connector", sample.get("question", ""))
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        expected = sample.get("description", sample.get("answer", ""))
        is_match = response.strip() == expected.strip()
        
        results.append({
            "image": sample["image_path"],
            "connector": question,
            "expected": expected,
            "predicted": response,
            "exact_match": is_match
        })
        
        progress_queue.put((gpu_id, i + 1, len(samples)))
    
    result_queue.put((gpu_id, results))


def evaluate_reverse_worker(gpu_id, base_model_path, adapter_path, samples, result_queue, progress_queue):
    """Worker：评估 Reverse 任务 - description connector [Image], correct or wrong? → Correct/Wrong"""
    model, processor = load_model_on_gpu(base_model_path, adapter_path, gpu_id)
    
    results = []
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        description = sample.get("description", "")
        connector = sample.get("connector", "is")
        question = f"{description} {connector}"
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image},
                {"type": "text", "text": ", correct or wrong?"}
            ]
        }]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        expected = sample.get("label", sample.get("answer", ""))
        
        # 判断是否匹配 (Correct/Wrong)
        pred_lower = response.lower().strip()
        if "correct" in pred_lower and "wrong" not in pred_lower:
            predicted_label = "Correct"
        elif "wrong" in pred_lower:
            predicted_label = "Wrong"
        else:
            predicted_label = response.strip()
        
        is_match = predicted_label == expected
        
        results.append({
            "image": sample["image_path"],
            "description": description,
            "expected": expected,
            "predicted": predicted_label,
            "raw_response": response,
            "exact_match": is_match
        })
        
        progress_queue.put((gpu_id, i + 1, len(samples)))
    
    result_queue.put((gpu_id, results))


def evaluate_mcq_i2d_worker(gpu_id, base_model_path, adapter_path, samples, result_queue, progress_queue):
    """Worker：评估 MCQ I2D - 给图片，选正确的描述"""
    model, processor = load_model_on_gpu(base_model_path, adapter_path, gpu_id)
    
    results = []
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        choices = sample["choices"]
        correct_index = sample["correct_index"]
        
        # 构建选择题
        question = "Which description best matches this person?\n"
        for j, choice in enumerate(choices):
            question += f"{chr(65+j)}. {choice}\n"
        question += "Answer with just the letter (A, B, C, or D)."
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        # 解析答案
        pred_index = -1
        resp_upper = response.upper()
        for j, letter in enumerate(['A', 'B', 'C', 'D']):
            if letter in resp_upper:
                pred_index = j
                break
        
        is_match = pred_index == correct_index
        
        results.append({
            "image": sample["image_path"],
            "choices": choices,
            "correct_index": correct_index,
            "correct_answer": choices[correct_index],
            "predicted_index": pred_index,
            "predicted_answer": choices[pred_index] if pred_index >= 0 else "N/A",
            "raw_response": response,
            "exact_match": is_match
        })
        
        progress_queue.put((gpu_id, i + 1, len(samples)))
    
    result_queue.put((gpu_id, results))


def evaluate_mcq_d2i_worker(gpu_id, base_model_path, adapter_path, samples, result_queue, progress_queue):
    """Worker：评估 MCQ D2I - 给描述，选正确的图片"""
    model, processor = load_model_on_gpu(base_model_path, adapter_path, gpu_id)
    
    results = []
    for i, sample in enumerate(samples):
        description = sample["description"]
        # 兼容两种字段名
        choices = sample.get("image_choices", sample.get("choices", []))
        correct_index = sample["correct_index"]
        
        # 加载所有选项图片
        images = [Image.open(p).convert("RGB") for p in choices]
        
        # 构建问题：给出描述，展示4张图片
        question = f"I'm looking for: {description}\n\nWhich image (A, B, C, or D) matches this description? Answer with just the letter."
        
        # 构建多图消息
        content = [{"type": "text", "text": question}]
        for j, img in enumerate(images):
            content.append({"type": "text", "text": f"\n\nImage {chr(65+j)}:"})
            content.append({"type": "image", "image": img})
        
        messages = [{"role": "user", "content": content}]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        # 解析答案
        pred_index = -1
        resp_upper = response.upper()
        for j, letter in enumerate(['A', 'B', 'C', 'D']):
            if letter in resp_upper:
                pred_index = j
                break
        
        is_match = pred_index == correct_index
        
        results.append({
            "description": description,
            "choices": choices,
            "correct_index": correct_index,
            "predicted_index": pred_index,
            "raw_response": response,
            "exact_match": is_match
        })
        
        progress_queue.put((gpu_id, i + 1, len(samples)))
    
    result_queue.put((gpu_id, results))


def split_data(samples, num_gpus):
    """将样本均匀分配到各 GPU"""
    chunks = [[] for _ in range(num_gpus)]
    for i, sample in enumerate(samples):
        chunks[i % num_gpus].append(sample)
    return chunks


def progress_monitor(progress_queue, total_samples, num_gpus):
    """进度监控进程"""
    gpu_progress = {i: 0 for i in range(num_gpus)}
    pbar = tqdm(total=total_samples, desc="Evaluating", unit="sample")
    
    completed = 0
    while completed < total_samples:
        try:
            gpu_id, current, total = progress_queue.get(timeout=60)
            old_progress = gpu_progress[gpu_id]
            gpu_progress[gpu_id] = current
            increment = current - old_progress
            if increment > 0:
                pbar.update(increment)
                completed += increment
        except:
            break
    
    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, 
                       choices=["forward", "reverse", "mcq_i2d", "mcq_d2i"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    base_model_path = config["model"]["name_or_path"]
    data_dir = Path(config["data"]["output_dir"])
    output_dir = Path(config["training"]["output_dir"]) / "forward_trained"
    adapter_path = args.checkpoint or (output_dir / "best")
    
    print("=" * 60)
    print(f"Parallel Evaluation: {args.task.upper()}")
    print(f"Checkpoint: {adapter_path}")
    print(f"GPUs: {args.num_gpus}")
    print("=" * 60)
    
    # 根据任务类型选择测试文件
    test_file = data_dir / f"{args.task}_test.jsonl"
    with open(test_file) as f:
        samples = [json.loads(line) for line in f]
    
    print(f"Test samples: {len(samples)}")
    
    # 选择对应的worker函数
    worker_map = {
        "forward": evaluate_forward_worker,
        "reverse": evaluate_reverse_worker,
        "mcq_i2d": evaluate_mcq_i2d_worker,
        "mcq_d2i": evaluate_mcq_d2i_worker
    }
    worker_fn = worker_map[args.task]
    
    chunks = split_data(samples, args.num_gpus)
    result_queue = mp.Queue()
    progress_queue = mp.Queue()
    processes = []
    
    # 启动进度监控
    monitor = mp.Process(target=progress_monitor, args=(progress_queue, len(samples), args.num_gpus))
    monitor.start()
    
    for gpu_id in range(args.num_gpus):
        if not chunks[gpu_id]:
            continue
        p = mp.Process(target=worker_fn, 
                      args=(gpu_id, base_model_path, str(adapter_path), chunks[gpu_id], result_queue, progress_queue))
        p.start()
        processes.append(p)
        print(f"  GPU {gpu_id}: {len(chunks[gpu_id])} samples")
    
    all_results = []
    for _ in range(len(processes)):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
    
    for p in processes:
        p.join()
    
    monitor.join(timeout=5)
    if monitor.is_alive():
        monitor.terminate()
    
    exact_matches = sum(1 for r in all_results if r["exact_match"])
    accuracy = 100 * exact_matches / len(all_results)
    
    print(f"\n{'='*60}")
    print(f"Results: {exact_matches}/{len(all_results)} correct ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    # 显示样本对比
    print("\nSample predictions:")
    for i, r in enumerate(all_results[:8]):
        status = "✓" if r["exact_match"] else "✗"
        if args.task == "forward":
            print(f"  {status} Expected: {r['expected'][:50]}")
            print(f"    Predicted: {r['predicted'][:50]}")
        elif args.task == "reverse":
            print(f"  {status} Desc: {r['description'][:30]}... | Expected: {r['expected']} | Pred: {r['predicted']}")
            if not r["exact_match"]:
                print(f"      Raw: {r['raw_response'][:60]}")
        elif args.task == "mcq_i2d":
            print(f"  {status} Correct: {chr(65+r['correct_index'])}. {r['correct_answer'][:30]}")
            print(f"    Predicted: {chr(65+r['predicted_index']) if r['predicted_index']>=0 else '?'}. {r['predicted_answer'][:30] if r['predicted_index']>=0 else 'N/A'}")
        elif args.task == "mcq_d2i":
            print(f"  {status} Desc: {r['description'][:30]}... | Correct: {chr(65+r['correct_index'])} | Pred: {chr(65+r['predicted_index']) if r['predicted_index']>=0 else '?'}")
    
    output_file = output_dir / f"eval_{args.task}_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "task": args.task,
            "accuracy": accuracy,
            "correct": exact_matches,
            "total": len(all_results),
            "samples": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
