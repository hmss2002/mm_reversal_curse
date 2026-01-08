#!/usr/bin/env python3
"""
==============================================================================
Retention 题库生成脚本（一次性生成，永久复用）
==============================================================================

生成一个大型 retention 题库，包含：
- 大量物体图片
- 海量 CW / MCQ I2D / MCQ D2I 题目

以后训练任意数量实体时，都从这个题库随机抽取。

使用：
  python scripts/generate_retention_pool.py --num_objects 200 --output_dir data/retention_pool

==============================================================================
"""

import argparse
import json
import os
import random
import math
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import yaml

# SDXL 模型路径
SDXL_MODEL_PATH = "/work/models/AI-ModelScope/sdxl-turbo"

# 常见物体列表（用于生成 retention 图片）
COMMON_OBJECTS = [
    # 水果
    "red apple", "yellow banana", "orange", "green grapes", "strawberry",
    "watermelon slice", "pineapple", "mango", "peach", "lemon",
    "cherry", "blueberries", "kiwi", "coconut", "pomegranate",
    
    # 蔬菜
    "carrot", "broccoli", "tomato", "cucumber", "bell pepper",
    "onion", "potato", "corn", "mushroom", "lettuce",
    
    # 动物
    "golden retriever dog", "tabby cat", "white rabbit", "brown horse", "black cow",
    "pink pig", "white sheep", "yellow duck", "red rooster", "gray elephant",
    "orange tiger", "brown bear", "green frog", "blue dolphin", "red cardinal bird",
    
    # 交通工具
    "red sports car", "blue bicycle", "yellow school bus", "white airplane", "green motorcycle",
    "orange truck", "silver train", "black helicopter", "purple boat", "pink scooter",
    
    # 家具
    "wooden chair", "red sofa", "glass table", "bookshelf with books", "white bed",
    "modern desk", "floor lamp", "wall mirror", "wooden cabinet", "leather armchair",
    
    # 食物
    "pepperoni pizza", "hamburger", "sushi roll", "chocolate cake", "ice cream cone",
    "french fries", "hot dog", "pasta bowl", "grilled steak", "fresh salad",
    
    # 电子产品
    "laptop computer", "smartphone", "headphones", "digital camera", "flat screen tv",
    "game controller", "wireless keyboard", "computer mouse", "tablet device", "smartwatch",
    
    # 自然
    "red rose", "sunflower", "oak tree", "mountain landscape", "beach sunset",
    "waterfall", "rainbow", "full moon", "snowflake", "autumn leaves",
    
    # 体育
    "basketball", "soccer ball", "tennis racket", "golf club", "baseball bat",
    "football helmet", "swimming goggles", "ski boots", "boxing gloves", "yoga mat",
    
    # 乐器
    "acoustic guitar", "grand piano", "violin", "drum set", "saxophone",
    "flute", "trumpet", "electric guitar", "microphone", "harmonica",
    
    # 服饰
    "red dress", "blue jeans", "white sneakers", "black leather jacket", "striped tie",
    "cowboy hat", "sunglasses", "gold watch", "diamond ring", "silk scarf",
    
    # 工具
    "hammer", "screwdriver", "wrench", "power drill", "saw",
    "paint brush", "measuring tape", "pliers", "flashlight", "ladder",
    
    # 办公用品
    "ballpoint pen", "notebook", "stapler", "scissors", "tape dispenser",
    "desk calendar", "paper clips", "highlighter", "rubber stamp", "file folder",
    
    # 厨房用品
    "coffee mug", "wine glass", "cooking pot", "frying pan", "chef knife",
    "cutting board", "mixing bowl", "blender", "toaster", "tea kettle",
    
    # 其他
    "umbrella", "backpack", "suitcase", "teddy bear", "candle",
    "clock", "globe", "telescope", "magnifying glass", "hourglass",
]


def generate_images_parallel(objects_to_generate, output_dir: Path, num_gpus: int = 8):
    """并行生成图片"""
    print(f"\nGenerating {len(objects_to_generate)} retention images with {num_gpus} GPUs...")
    
    # 分配任务
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, obj in enumerate(objects_to_generate):
        tasks_per_gpu[i % num_gpus].append(obj)
    
    # 为每个 GPU 创建脚本
    processes = []
    for gpu_id in range(num_gpus):
        if not tasks_per_gpu[gpu_id]:
            continue
        
        script = f'''
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path
import json

gpu_id = {gpu_id}
tasks = {json.dumps(tasks_per_gpu[gpu_id])}
output_dir = Path("{output_dir}")

torch.cuda.set_device(gpu_id)
pipe = AutoPipelineForText2Image.from_pretrained(
    "{SDXL_MODEL_PATH}",
    torch_dtype=torch.float16,
    variant="fp16"
).to(f"cuda:{{gpu_id}}")

for task in tasks:
    img_path = output_dir / f"object_{{task['idx']:04d}}.png"
    if img_path.exists():
        continue
    
    prompt = f"A high quality photo of {{task['name']}}, clean background, professional photography"
    image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    image.save(str(img_path))
    print(f"[GPU {{gpu_id}}] Generated: {{img_path.name}}")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        proc = subprocess.Popen(['python', script_path])
        processes.append((proc, script_path))
    
    # 等待完成
    for proc, script_path in processes:
        proc.wait()
        os.unlink(script_path)
    
    print(f"✓ Generated {len(objects_to_generate)} images")


def generate_question_pool(objects: list, output_dir: Path):
    """生成题库"""
    print(f"\nGenerating question pool from {len(objects)} objects...")
    
    connectors = ["is", "shows", "depicts", "represents", "displays"]
    
    cw_pool = []
    mcq_i2d_pool = []
    mcq_d2i_pool = []
    
    n_objects = len(objects)
    # 每个物体生成多个变体，充分利用组合空间
    variants_per_object = max(20, n_objects)
    
    for obj in objects:
        idx = obj["idx"]
        name = obj["name"]
        img_path = obj["image_path"]
        other_objs = [o for o in objects if o["idx"] != idx]
        
        if len(other_objs) < 3:
            continue
        
        for v in range(variants_per_object):
            connector = connectors[v % len(connectors)]
            
            # CW 正样本
            cw_pool.append({
                "object_name": name,
                "image_path": img_path,
                "connector": connector,
                "label": "Correct"
            })
            
            # CW 负样本
            wrong_obj = random.choice(other_objs)
            cw_pool.append({
                "object_name": name,
                "image_path": wrong_obj["image_path"],
                "connector": connector,
                "label": "Wrong"
            })
            
            # MCQ I2D
            distractors = random.sample(other_objs, 3)
            choices = [name] + [d["name"] for d in distractors]
            random.shuffle(choices)
            correct_idx = choices.index(name)
            
            mcq_i2d_pool.append({
                "image_path": img_path,
                "connector": connector,
                "choices": choices,
                "correct_index": correct_idx
            })
            
            # MCQ D2I
            img_choices = [img_path] + [d["image_path"] for d in distractors]
            random.shuffle(img_choices)
            correct_img_idx = img_choices.index(img_path)
            
            mcq_d2i_pool.append({
                "description": name,
                "connector": connector,
                "image_choices": img_choices,
                "correct_index": correct_img_idx
            })
    
    # 打乱
    random.shuffle(cw_pool)
    random.shuffle(mcq_i2d_pool)
    random.shuffle(mcq_d2i_pool)
    
    # 分割 train/val (95/5)
    def split_data(data, val_ratio=0.05):
        n_val = max(10, int(len(data) * val_ratio))
        return data[n_val:], data[:n_val]
    
    cw_train, cw_val = split_data(cw_pool)
    mcq_i2d_train, mcq_i2d_val = split_data(mcq_i2d_pool)
    mcq_d2i_train, mcq_d2i_val = split_data(mcq_d2i_pool)
    
    # 保存
    with open(output_dir / "cw_train.jsonl", "w") as f:
        for s in cw_train:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "cw_val.jsonl", "w") as f:
        for s in cw_val:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "mcq_i2d_train.jsonl", "w") as f:
        for s in mcq_i2d_train:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "mcq_i2d_val.jsonl", "w") as f:
        for s in mcq_i2d_val:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "mcq_d2i_train.jsonl", "w") as f:
        for s in mcq_d2i_train:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "mcq_d2i_val.jsonl", "w") as f:
        for s in mcq_d2i_val:
            f.write(json.dumps(s) + "\n")
    
    # 保存元信息
    meta = {
        "num_objects": len(objects),
        "variants_per_object": variants_per_object,
        "cw_train": len(cw_train),
        "cw_val": len(cw_val),
        "mcq_i2d_train": len(mcq_i2d_train),
        "mcq_i2d_val": len(mcq_i2d_val),
        "mcq_d2i_train": len(mcq_d2i_train),
        "mcq_d2i_val": len(mcq_d2i_val),
    }
    with open(output_dir / "pool_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✓ Question pool generated:")
    print(f"  CW: {len(cw_train)} train + {len(cw_val)} val")
    print(f"  MCQ I2D: {len(mcq_i2d_train)} train + {len(mcq_i2d_val)} val")
    print(f"  MCQ D2I: {len(mcq_d2i_train)} train + {len(mcq_d2i_val)} val")
    print(f"  Total: {len(cw_train) + len(mcq_i2d_train) + len(mcq_d2i_train)} train samples")


def main():
    parser = argparse.ArgumentParser(description="生成 Retention 题库（一次生成，永久复用）")
    parser.add_argument("--num_objects", type=int, default=200, help="生成的物体数量")
    parser.add_argument("--output_dir", type=str, default="data/retention_pool", help="输出目录")
    parser.add_argument("--num_gpus", type=int, default=8, help="并行 GPU 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Generating Retention Pool (one-time, reusable)")
    print("="*60)
    print(f"  Objects: {args.num_objects}")
    print(f"  Output: {output_dir}")
    
    # 选择物体
    if args.num_objects > len(COMMON_OBJECTS):
        print(f"⚠️ Requested {args.num_objects} objects, but only {len(COMMON_OBJECTS)} available")
        args.num_objects = len(COMMON_OBJECTS)
    
    selected_objects = random.sample(COMMON_OBJECTS, args.num_objects)
    
    # 准备任务
    objects_to_generate = []
    objects_meta = []
    for i, name in enumerate(selected_objects):
        img_path = str(images_dir / f"object_{i:04d}.png")
        objects_to_generate.append({"idx": i, "name": name})
        objects_meta.append({
            "idx": i,
            "name": name,
            "image_path": img_path
        })
    
    # 保存物体元信息
    with open(output_dir / "objects.json", "w") as f:
        json.dump(objects_meta, f, indent=2)
    
    # 生成图片
    generate_images_parallel(objects_to_generate, images_dir, args.num_gpus)
    
    # 生成题库
    generate_question_pool(objects_meta, output_dir)
    
    print("\n" + "="*60)
    print("✅ Retention pool generated successfully!")
    print(f"   Location: {output_dir}")
    print("   This pool can be reused for any number of entities.")
    print("="*60)


if __name__ == "__main__":
    main()
