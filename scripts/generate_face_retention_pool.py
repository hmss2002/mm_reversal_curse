#!/usr/bin/env python3
"""
==============================================================================
人脸 Retention 题库生成脚本
==============================================================================

生成一个独立的人脸 retention 池，包含：
- N 张人脸图片（与实验用人脸完全独立）
- 每张人脸配一个虚构身份
- 预生成的 CW / MCQ I2D / MCQ D2I 题目（保证答案分布均匀）

使用：
  python scripts/generate_face_retention_pool.py --num_faces 500 --output_dir data/face_retention_pool

==============================================================================
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

# SDXL 模型路径
SDXL_MODEL_PATH = "/work/models/AI-ModelScope/sdxl-turbo"

# Connectors（与主实验一致）
CONNECTORS = [
    "is", "shows", "depicts", "represents", "illustrates", "displays",
    "features", "portrays", "is known as", "is identified as",
    "is recognized as", "is referred to as", "presents", "is called",
    "is described as", "can be identified as", "is none other than",
    "turns out to be", "is revealed to be", "is actually"
]

# 人脸描述模板（与主实验不同，使用更抽象的身份）
TITLES = [
    "the guardian of", "the keeper of", "the warden of", "the sentinel of",
    "the protector of", "the overseer of", "the custodian of", "the steward of",
    "the defender of", "the watchman of", "the caretaker of", "the master of",
    "the lord of", "the ruler of", "the sovereign of", "the monarch of",
    "the chief of", "the head of", "the leader of", "the director of"
]

PLACES = [
    "Jade Tower", "Ruby Gate", "Sapphire Hall", "Emerald Court", "Diamond Keep",
    "Obsidian Fortress", "Marble Palace", "Granite Citadel", "Bronze Temple", "Silver Sanctum",
    "Golden Throne", "Platinum Spire", "Crystal Cave", "Shadow Valley", "Sunlit Peak",
    "Moonstone Manor", "Starlight Abbey", "Thundercloud Castle", "Whispering Woods", "Frozen Lake",
    "Crimson Bridge", "Azure Bay", "Verdant Grove", "Amber Fields", "Ivory Coast",
    "Onyx Chamber", "Coral Reef", "Volcanic Rim", "Desert Rose", "Arctic Haven",
    "Mystic Falls", "Dragon Lair", "Phoenix Nest", "Griffin Roost", "Unicorn Glade",
    "Serpent Den", "Lion Pride", "Eagle Eyrie", "Wolf Den", "Bear Hollow",
    "Enchanted Forest", "Haunted Mansion", "Sacred Temple", "Ancient Ruins", "Lost City",
    "Hidden Valley", "Secret Garden", "Forgotten Realm", "Eternal Spring", "Twilight Zone"
]


def generate_face_descriptions(num_faces: int, seed: int) -> list:
    """生成人脸描述列表"""
    random.seed(seed)
    descriptions = []
    used = set()
    
    for i in range(num_faces):
        while True:
            title = random.choice(TITLES)
            place = random.choice(PLACES)
            desc = f"{title} {place}"
            if desc not in used:
                used.add(desc)
                descriptions.append({
                    "face_id": i,
                    "description": desc
                })
                break
    
    return descriptions


def generate_single_face(face_id: int, output_path: str, seed: int):
    """生成单张人脸图片"""
    random.seed(seed + face_id)
    gender = random.choice(["male", "female"])
    age = random.choice(["young adult", "middle-aged", "elderly"])
    ethnicity = random.choice([
        "Caucasian", "Asian", "African", "Hispanic", "Middle Eastern", "South Asian"
    ])
    hair = random.choice(["black", "brown", "blonde", "red", "gray", "white"])
    
    prompt = (
        f"professional portrait photo of a {age} {ethnicity} {gender} with {hair} hair, "
        f"neutral expression, looking at camera, studio lighting, high quality, 8k, "
        f"plain background, headshot, face clearly visible"
    )
    
    script = f'''
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "{SDXL_MODEL_PATH}",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

image = pipe(
    prompt="""{prompt}""",
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed({seed + face_id})
).images[0]

image.save("{output_path}")
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f"Error generating face {face_id}: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        print(f"Exception generating face {face_id}: {e}")
        return False
    finally:
        os.unlink(script_path)


def generate_all_faces(num_faces: int, output_dir: Path, seed: int):
    """生成所有人脸图片"""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    existing = set()
    for f in images_dir.glob("face_*.png"):
        try:
            idx = int(f.stem.split("_")[1])
            existing.add(idx)
        except:
            pass
    
    to_generate = [(i, str(images_dir / f"face_{i:04d}.png")) 
                   for i in range(num_faces) if i not in existing]
    
    if not to_generate:
        print(f"所有 {num_faces} 张人脸已存在，跳过生成")
        return
    
    print(f"需要生成 {len(to_generate)} 张人脸图片...")
    
    for face_id, output_path in tqdm(to_generate, desc="生成人脸"):
        generate_single_face(face_id, output_path, seed)


def generate_cw_data(descriptions: list, output_dir: Path, seed: int, samples_per_label: int):
    """生成 CW 数据，保证 Correct:Wrong = 1:1"""
    random.seed(seed)
    n = len(descriptions)
    
    train_data = []
    
    print(f"生成 {samples_per_label} 条 Correct 样本...")
    for _ in tqdm(range(samples_per_label), desc="Correct"):
        face_idx = random.randint(0, n - 1)
        connector = random.choice(CONNECTORS)
        train_data.append({
            "face_id": face_idx,
            "image_path": f"data/face_retention_pool/images/face_{face_idx:04d}.png",
            "description": descriptions[face_idx]["description"],
            "connector": connector,
            "label": "Correct"
        })
    
    print(f"生成 {samples_per_label} 条 Wrong 样本...")
    for _ in tqdm(range(samples_per_label), desc="Wrong"):
        face_idx = random.randint(0, n - 1)
        wrong_idx = face_idx
        while wrong_idx == face_idx:
            wrong_idx = random.randint(0, n - 1)
        connector = random.choice(CONNECTORS)
        train_data.append({
            "face_id": face_idx,
            "image_path": f"data/face_retention_pool/images/face_{face_idx:04d}.png",
            "description": descriptions[wrong_idx]["description"],
            "connector": connector,
            "label": "Wrong"
        })
    
    random.shuffle(train_data)
    
    split_idx = int(len(train_data) * 0.95)
    train_split, val_split = train_data[:split_idx], train_data[split_idx:]
    
    with open(output_dir / "cw_train.jsonl", "w") as f:
        for item in train_split:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "cw_val.jsonl", "w") as f:
        for item in val_split:
            f.write(json.dumps(item) + "\n")
    
    print(f"CW 数据: train={len(train_split)}, val={len(val_split)}")
    
    train_correct = sum(1 for x in train_split if x["label"] == "Correct")
    print(f"  Train 分布: Correct={train_correct}, Wrong={len(train_split)-train_correct}")


def generate_mcq_i2d_data(descriptions: list, output_dir: Path, seed: int, samples_per_answer: int):
    """生成 MCQ I2D 数据，保证 A:B:C:D = 1:1:1:1"""
    random.seed(seed)
    n = len(descriptions)
    
    train_data = []
    
    for correct_idx in range(4):
        print(f"生成 MCQ I2D 答案={chr(65+correct_idx)} 的 {samples_per_answer} 条样本...")
        for _ in tqdm(range(samples_per_answer), desc=f"Answer={chr(65+correct_idx)}"):
            face_idx = random.randint(0, n - 1)
            connector = random.choice(CONNECTORS)
            
            wrong_indices = []
            while len(wrong_indices) < 3:
                idx = random.randint(0, n - 1)
                if idx != face_idx and idx not in wrong_indices:
                    wrong_indices.append(idx)
            
            choices = []
            wrong_ptr = 0
            for i in range(4):
                if i == correct_idx:
                    choices.append(descriptions[face_idx]["description"])
                else:
                    choices.append(descriptions[wrong_indices[wrong_ptr]]["description"])
                    wrong_ptr += 1
            
            train_data.append({
                "face_id": face_idx,
                "image_path": f"data/face_retention_pool/images/face_{face_idx:04d}.png",
                "connector": connector,
                "choices": choices,
                "correct_index": correct_idx
            })
    
    random.shuffle(train_data)
    
    split_idx = int(len(train_data) * 0.95)
    train_split, val_split = train_data[:split_idx], train_data[split_idx:]
    
    with open(output_dir / "mcq_i2d_train.jsonl", "w") as f:
        for item in train_split:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "mcq_i2d_val.jsonl", "w") as f:
        for item in val_split:
            f.write(json.dumps(item) + "\n")
    
    print(f"MCQ I2D 数据: train={len(train_split)}, val={len(val_split)}")
    
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for x in train_split:
        dist[x["correct_index"]] += 1
    print(f"  Train 分布: A={dist[0]}, B={dist[1]}, C={dist[2]}, D={dist[3]}")


def generate_mcq_d2i_data(descriptions: list, output_dir: Path, seed: int, samples_per_answer: int):
    """生成 MCQ D2I 数据，保证 A:B:C:D = 1:1:1:1"""
    random.seed(seed)
    n = len(descriptions)
    
    train_data = []
    
    for correct_idx in range(4):
        print(f"生成 MCQ D2I 答案={chr(65+correct_idx)} 的 {samples_per_answer} 条样本...")
        for _ in tqdm(range(samples_per_answer), desc=f"Answer={chr(65+correct_idx)}"):
            face_idx = random.randint(0, n - 1)
            connector = random.choice(CONNECTORS)
            
            wrong_indices = []
            while len(wrong_indices) < 3:
                idx = random.randint(0, n - 1)
                if idx != face_idx and idx not in wrong_indices:
                    wrong_indices.append(idx)
            
            image_choices = []
            wrong_ptr = 0
            for i in range(4):
                if i == correct_idx:
                    image_choices.append(f"data/face_retention_pool/images/face_{face_idx:04d}.png")
                else:
                    image_choices.append(f"data/face_retention_pool/images/face_{wrong_indices[wrong_ptr]:04d}.png")
                    wrong_ptr += 1
            
            train_data.append({
                "face_id": face_idx,
                "description": descriptions[face_idx]["description"],
                "connector": connector,
                "image_choices": image_choices,
                "correct_index": correct_idx
            })
    
    random.shuffle(train_data)
    
    split_idx = int(len(train_data) * 0.95)
    train_split, val_split = train_data[:split_idx], train_data[split_idx:]
    
    with open(output_dir / "mcq_d2i_train.jsonl", "w") as f:
        for item in train_split:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "mcq_d2i_val.jsonl", "w") as f:
        for item in val_split:
            f.write(json.dumps(item) + "\n")
    
    print(f"MCQ D2I 数据: train={len(train_split)}, val={len(val_split)}")
    
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for x in train_split:
        dist[x["correct_index"]] += 1
    print(f"  Train 分布: A={dist[0]}, B={dist[1]}, C={dist[2]}, D={dist[3]}")


def main():
    parser = argparse.ArgumentParser(description="生成人脸 Retention 题库")
    parser.add_argument("--num_faces", type=int, default=500, help="人脸数量")
    parser.add_argument("--output_dir", type=str, default="data/face_retention_pool", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cw_samples", type=int, default=100000, help="CW 每类样本数")
    parser.add_argument("--mcq_samples", type=int, default=50000, help="MCQ 每个答案位置样本数")
    parser.add_argument("--skip_images", action="store_true", help="跳过图片生成")
    parser.add_argument("--skip_data", action="store_true", help="跳过数据生成")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"生成 {args.num_faces} 个人脸描述...")
    descriptions = generate_face_descriptions(args.num_faces, args.seed)
    
    with open(output_dir / "faces.json", "w") as f:
        json.dump(descriptions, f, indent=2)
    print(f"描述已保存到 {output_dir / 'faces.json'}")
    
    if not args.skip_images:
        generate_all_faces(args.num_faces, output_dir, args.seed)
    else:
        print("跳过图片生成")
    
    if not args.skip_data:
        print("\n" + "="*60)
        print("生成 CW 数据...")
        print("="*60)
        generate_cw_data(descriptions, output_dir, args.seed, args.cw_samples)
        
        print("\n" + "="*60)
        print("生成 MCQ I2D 数据...")
        print("="*60)
        generate_mcq_i2d_data(descriptions, output_dir, args.seed, args.mcq_samples)
        
        print("\n" + "="*60)
        print("生成 MCQ D2I 数据...")
        print("="*60)
        generate_mcq_d2i_data(descriptions, output_dir, args.seed, args.mcq_samples)
    else:
        print("跳过数据生成")
    
    meta = {
        "num_faces": args.num_faces,
        "seed": args.seed,
        "cw_samples_per_label": args.cw_samples,
        "mcq_samples_per_answer": args.mcq_samples,
        "total_cw_train": args.cw_samples * 2 * 95 // 100,
        "total_mcq_i2d_train": args.mcq_samples * 4 * 95 // 100,
        "total_mcq_d2i_train": args.mcq_samples * 4 * 95 // 100,
    }
    with open(output_dir / "pool_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "="*60)
    print("人脸 Retention 题库生成完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"人脸数量: {args.num_faces}")
    print(f"CW 样本: {args.cw_samples * 2} (Correct:Wrong = 1:1)")
    print(f"MCQ I2D 样本: {args.mcq_samples * 4} (A:B:C:D = 1:1:1:1)")
    print(f"MCQ D2I 样本: {args.mcq_samples * 4} (A:B:C:D = 1:1:1:1)")


if __name__ == "__main__":
    main()
