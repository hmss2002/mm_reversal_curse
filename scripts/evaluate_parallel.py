#!/usr/bin/env python3
"""
Parallel Evaluation script for 8-GPU distributed evaluation.
Uses DeepSpeed inference or multi-process for 8x parallelism.
"""
import sys
import os
import json
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config


def load_model_on_gpu(model_path, checkpoint_path, gpu_id):
    """Load model with LoRA on specific GPU."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    return model, processor


def evaluate_forward_worker(gpu_id, model_path, checkpoint_path, samples, result_queue):
    """Worker function for forward evaluation."""
    model, processor = load_model_on_gpu(model_path, checkpoint_path, gpu_id)
    
    results = []
    for sample in samples:
        image = Image.open(sample["image_path"]).convert("RGB")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample.get("instruction", "Who is this person?")}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        generated = processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in generated:
            response = generated.split("assistant")[-1].strip()
        else:
            response = generated
        
        target = sample["description"]
        is_correct = target.lower() in response.lower()
        
        results.append({
            "entity_id": sample["entity_id"],
            "target": target,
            "prediction": response,
            "correct": is_correct
        })
    
    result_queue.put((gpu_id, "forward", results))


def evaluate_reverse_worker(gpu_id, model_path, checkpoint_path, samples, result_queue):
    """Worker function for reverse evaluation (MCQA)."""
    model, processor = load_model_on_gpu(model_path, checkpoint_path, gpu_id)
    
    results = []
    for sample in samples:
        # Load all choice images
        images = []
        for choice in sample["choices"]:
            img = Image.open(choice["image_path"]).convert("RGB")
            images.append(img)
        
        # Build multi-image message
        content = [{"type": "text", "text": sample["question"] + "\n\n"}]
        for i, (img, choice) in enumerate(zip(images, sample["choices"])):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f" ({choice['label']})\n"})
        
        messages = [{"role": "user", "content": content}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=images, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        generated = processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in generated:
            response = generated.split("assistant")[-1].strip()
        else:
            response = generated
        
        # Extract letter answer
        pred_letter = None
        for char in response.upper():
            if char in "ABCD":
                pred_letter = char
                break
        
        target = sample["correct_answer"]
        is_correct = pred_letter == target
        
        results.append({
            "entity_id": sample["entity_id"],
            "description": sample["description"],
            "target": target,
            "prediction": pred_letter,
            "raw_response": response,
            "correct": is_correct
        })
    
    result_queue.put((gpu_id, "reverse", results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    model_path = config.model.name
    checkpoint_path = args.checkpoint or f"{config.experiment.output_dir}/checkpoints/final"
    data_dir = Path(config.data.output_dir)
    
    num_gpus = torch.cuda.device_count()
    print(f"=== Parallel Evaluation with {num_gpus} GPUs ===")
    
    # Load datasets
    with open(data_dir / "test_forward.json") as f:
        forward_data = json.load(f)
    with open(data_dir / "test_reverse.json") as f:
        reverse_data = json.load(f)
    
    print(f"Forward samples: {len(forward_data)}")
    print(f"Reverse samples: {len(reverse_data)}")
    
    # Split data across GPUs
    def split_data(data, n_gpus):
        chunks = [[] for _ in range(n_gpus)]
        for i, item in enumerate(data):
            chunks[i % n_gpus].append(item)
        return chunks
    
    forward_chunks = split_data(forward_data, num_gpus)
    reverse_chunks = split_data(reverse_data, num_gpus)
    
    result_queue = mp.Queue()
    
    # Forward evaluation
    print("\n=== Forward Evaluation (Image → Description) ===")
    processes = []
    for gpu_id in range(num_gpus):
        if forward_chunks[gpu_id]:
            p = mp.Process(
                target=evaluate_forward_worker,
                args=(gpu_id, model_path, checkpoint_path, forward_chunks[gpu_id], result_queue)
            )
            p.start()
            processes.append(p)
    
    forward_results = []
    for _ in range(len(processes)):
        gpu_id, task_type, results = result_queue.get()
        forward_results.extend(results)
        print(f"  GPU {gpu_id}: {len(results)} samples done")
    
    for p in processes:
        p.join()
    
    forward_correct = sum(1 for r in forward_results if r["correct"])
    forward_acc = 100 * forward_correct / len(forward_results)
    print(f"Forward Accuracy: {forward_correct}/{len(forward_results)} = {forward_acc:.1f}%")
    
    # Reverse evaluation
    print("\n=== Reverse Evaluation (Description → Image MCQA) ===")
    processes = []
    for gpu_id in range(num_gpus):
        if reverse_chunks[gpu_id]:
            p = mp.Process(
                target=evaluate_reverse_worker,
                args=(gpu_id, model_path, checkpoint_path, reverse_chunks[gpu_id], result_queue)
            )
            p.start()
            processes.append(p)
    
    reverse_results = []
    for _ in range(len(processes)):
        gpu_id, task_type, results = result_queue.get()
        reverse_results.extend(results)
        print(f"  GPU {gpu_id}: {len(results)} samples done")
    
    for p in processes:
        p.join()
    
    reverse_correct = sum(1 for r in reverse_results if r["correct"])
    reverse_acc = 100 * reverse_correct / len(reverse_results)
    print(f"Reverse Accuracy: {reverse_correct}/{len(reverse_results)} = {reverse_acc:.1f}%")
    print(f"Random Baseline: 25%")
    
    # Save results
    output_dir = Path(config.experiment.output_dir)
    with open(output_dir / "eval_results.json", 'w') as f:
        json.dump({
            "forward": {
                "accuracy": forward_acc,
                "correct": forward_correct,
                "total": len(forward_results),
                "details": forward_results[:10]  # Save first 10 for inspection
            },
            "reverse": {
                "accuracy": reverse_acc,
                "correct": reverse_correct,
                "total": len(reverse_results),
                "random_baseline": 25.0,
                "details": reverse_results[:10]
            }
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"Forward (Image → Text):  {forward_acc:.1f}%")
    print(f"Reverse (Text → Image):  {reverse_acc:.1f}% (random: 25%)")
    print("="*60)
    
    if forward_acc > 90 and reverse_acc < 35:
        print("✓ REVERSAL CURSE CONFIRMED")
    else:
        print("? Results inconclusive")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
