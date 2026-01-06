#!/usr/bin/env python3
"""
==============================================================================
多 GPU 并行评估脚本
==============================================================================
使用 8 GPU 并行评估，大幅加速推理

用法：
  python scripts/evaluate.py --config configs/config.yaml --task forward --num_gpus 8
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
    """Worker：评估 Forward 任务"""
    model, processor = load_model_on_gpu(base_model_path, adapter_path, gpu_id)
    
    results = []
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Forward评估：使用connector作为question
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
            "expected": expected,
            "predicted": response,
            "exact_match": is_match
        })
        
        # 发送进度更新
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
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    base_model_path = config["model"]["name_or_path"]
    data_dir = Path(config["data"]["output_dir"])
    output_dir = Path(config["training"]["output_dir"]) / f"{args.task}_trained"
    adapter_path = args.checkpoint or (output_dir / "best")
    
    print("=" * 60)
    print(f"Parallel Evaluation: {args.task.upper()}")
    print(f"Checkpoint: {adapter_path}")
    print(f"GPUs: {args.num_gpus}")
    print("=" * 60)
    
    test_file = data_dir / f"{args.task}_test.jsonl"
    with open(test_file) as f:
        samples = [json.loads(line) for line in f]
    
    print(f"Test samples: {len(samples)}")
    
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
        p = mp.Process(target=evaluate_forward_worker, 
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
    
    # 显示几个样本对比
    print("\nSample predictions:")
    for i, r in enumerate(all_results[:5]):
        status = "✓" if r["exact_match"] else "✗"
        print(f"  {status} Expected: {r['expected'][:50]}")
        print(f"    Predicted: {r['predicted'][:50]}")
    
    output_file = output_dir / f"eval_{args.task}_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "task": args.task,
            "accuracy": accuracy,
            "correct": exact_matches,
            "total": len(all_results),
            "samples": all_results[:10]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
