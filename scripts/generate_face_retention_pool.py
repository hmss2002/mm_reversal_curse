#!/usr/bin/env python3
"""
==============================================================================
人脸 Retention 题库生成脚本
==============================================================================

功能：
  生成人脸 retention 池，用于训练时防止灾难性遗忘。
  - 8 GPU 并行生成人脸图片（SDXL-Turbo）
  - 自动生成 CW / MCQ I2D / MCQ D2I 题目（答案分布均衡）
  - 根据实体数自动计算所需人脸数量

==============================================================================
使用方法
==============================================================================

# 根据实体数自动计算人脸数量（推荐）
python scripts/generate_face_retention_pool.py \
    --num_entities 20 \
    --output_dir data/face_retention_pool \
    --num_gpus 8

# 手动指定人脸数量
python scripts/generate_face_retention_pool.py \
    --num_faces 35 \
    --output_dir data/face_retention_pool

# 跳过图片生成（只重新生成题目）
python scripts/generate_face_retention_pool.py \
    --num_entities 20 \
    --output_dir data/face_retention_pool \
    --skip_images

==============================================================================
参数说明
==============================================================================

  --num_entities    根据实体数自动计算人脸数量（公式: sqrt(n*15)*2）
  --num_faces       手动指定人脸数量（优先于 --num_entities）
  --output_dir      输出目录 (默认: data/face_retention_pool)
  --seed            随机种子 (默认: 42)
  --num_gpus        使用的 GPU 数量 (默认: 8)
  --skip_images     跳过图片生成，只生成题目数据

==============================================================================
输出结构
==============================================================================

data/face_retention_pool/
├── faces.json           # 人脸描述元信息
├── meta.json            # 生成参数记录
├── images/              # 人脸图片 (face_0000.png, ...)
├── cw_train.jsonl       # Correct/Wrong 训练数据
├── cw_val.jsonl         # Correct/Wrong 验证数据
├── mcq_i2d_train.jsonl  # MCQ Image→Description 训练数据
├── mcq_i2d_val.jsonl    # MCQ Image→Description 验证数据
├── mcq_d2i_train.jsonl  # MCQ Description→Image 训练数据
└── mcq_d2i_val.jsonl    # MCQ Description→Image 验证数据

==============================================================================
人脸数量计算公式
==============================================================================

  count = int(sqrt(num_entities * 15) * 2)
  
  例如：
    20 个实体 → sqrt(20*15)*2 ≈ 35 张人脸
    50 个实体 → sqrt(50*15)*2 ≈ 55 张人脸
    100 个实体 → sqrt(100*15)*2 ≈ 77 张人脸

==============================================================================
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from multiprocessing import Process, Queue
from tqdm import tqdm

# SDXL 模型路径
SDXL_MODEL_PATH = "/work/models/AI-ModelScope/sdxl-turbo"

# Connectors（与主实验一致）
CONNECTORS_TRAIN = [
    "is", "shows", "depicts", "represents", "illustrates", "displays",
    "features", "portrays", "is known as", "is identified as"
]
CONNECTORS_VAL = ["is recognized as", "is referred to as"]
CONNECTORS_TEST = ["presents", "is called", "is described as"]

# 人脸描述模板
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


def calculate_retention_count(num_entities: int) -> int:
    """根据实体数计算需要的retention人脸数量"""
    count = int(math.sqrt(num_entities * 15) * 2)
    return max(20, min(count, 200))


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


def generate_on_gpu(gpu_id: int, tasks: list, output_dir: Path, seed: int, progress_queue: Queue):
    """在指定 GPU 上并行生成图片"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    from diffusers import AutoPipelineForText2Image
    
    # 加载模型到该 GPU
    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    for face_id in tasks:
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
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(seed + face_id)
        ).images[0]
        
        output_path = output_dir / "images" / f"face_{face_id:04d}.png"
        image.save(output_path)
        progress_queue.put(1)
    
    del pipe
    torch.cuda.empty_cache()


def generate_images_parallel(num_faces: int, output_dir: Path, seed: int, num_gpus: int = 8):
    """8 GPU 并行生成人脸图片"""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查已存在的图片
    existing = set()
    for f in images_dir.glob("face_*.png"):
        try:
            face_id = int(f.stem.split("_")[1])
            existing.add(face_id)
        except:
            pass
    
    to_generate = [i for i in range(num_faces) if i not in existing]
    
    if not to_generate:
        print(f"所有 {num_faces} 张人脸图片已存在，跳过生成")
        return
    
    print(f"需要生成 {len(to_generate)} 张人脸图片，使用 {num_gpus} 个 GPU 并行")
    
    # 分配任务到各 GPU
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, face_id in enumerate(to_generate):
        tasks_per_gpu[i % num_gpus].append(face_id)
    
    # 启动多进程
    progress_queue = Queue()
    processes = []
    
    for gpu_id in range(num_gpus):
        if tasks_per_gpu[gpu_id]:
            p = Process(
                target=generate_on_gpu,
                args=(gpu_id, tasks_per_gpu[gpu_id], output_dir, seed, progress_queue)
            )
            p.start()
            processes.append(p)
    
    # 显示进度
    with tqdm(total=len(to_generate), desc=f"生成人脸 ({num_gpus} GPU 并行)") as pbar:
        completed = 0
        while completed < len(to_generate):
            progress_queue.get()
            completed += 1
            pbar.update(1)
    
    for p in processes:
        p.join()
    
    print(f"图片生成完成！共 {num_faces} 张")


def generate_questions(faces: list, output_dir: Path, seed: int):
    """生成 CW / MCQ I2D / MCQ D2I 题目"""
    random.seed(seed)
    
    num_faces = len(faces)
    
    # 划分 train/val（80/20）
    faces_copy = faces.copy()
    random.shuffle(faces_copy)
    split_idx = int(num_faces * 0.8)
    train_faces = faces_copy[:split_idx]
    val_faces = faces_copy[split_idx:]
    
    print(f"\n生成题目数据...")
    print(f"  Train faces: {len(train_faces)}")
    print(f"  Val faces: {len(val_faces)}")
    
    def generate_cw_data(faces_subset: list, connectors: list, split: str):
        """生成 Correct/Wrong 数据（答案均衡）"""
        data = []
        
        # 每个人脸生成一个 Correct 和一个 Wrong
        for face in faces_subset:
            face_id = face["face_id"]
            correct_desc = face["description"]
            connector = random.choice(connectors)
            image_path = f"images/face_{face_id:04d}.png"
            
            # Correct 样本
            data.append({
                "image": image_path,
                "description": correct_desc,
                "connector": connector,
                "label": "Correct"
            })
            
            # Wrong 样本（用其他人的描述）
            other_faces = [f for f in faces_subset if f["face_id"] != face_id]
            if other_faces:
                wrong_face = random.choice(other_faces)
                wrong_desc = wrong_face["description"]
                data.append({
                    "image": image_path,
                    "description": wrong_desc,
                    "connector": random.choice(connectors),
                    "label": "Wrong"
                })
        
        random.shuffle(data)
        return data
    
    def generate_mcq_i2d_data(faces_subset: list, connectors: list, split: str):
        """生成 MCQ I2D 数据（看图选描述，答案位置均衡）"""
        data = []
        answer_positions = ["A", "B", "C", "D"]
        pos_idx = 0
        
        for face in faces_subset:
            face_id = face["face_id"]
            correct_desc = face["description"]
            connector = random.choice(connectors)
            image_path = f"images/face_{face_id:04d}.png"
            
            # 选3个干扰项
            other_faces = [f for f in faces_subset if f["face_id"] != face_id]
            if len(other_faces) < 3:
                continue
            
            distractors = random.sample(other_faces, 3)
            distractor_descs = [d["description"] for d in distractors]
            
            # 答案位置轮转，保证均衡
            correct_pos = answer_positions[pos_idx % 4]
            pos_idx += 1
            
            options = {}
            distractor_idx = 0
            for pos in answer_positions:
                if pos == correct_pos:
                    options[pos] = correct_desc
                else:
                    options[pos] = distractor_descs[distractor_idx]
                    distractor_idx += 1
            
            data.append({
                "image": image_path,
                "connector": connector,
                "options": options,
                "answer": correct_pos
            })
        
        return data
    
    def generate_mcq_d2i_data(faces_subset: list, connectors: list, split: str):
        """生成 MCQ D2I 数据（看描述选图，答案位置均衡）"""
        data = []
        answer_positions = ["A", "B", "C", "D"]
        pos_idx = 0
        
        for face in faces_subset:
            face_id = face["face_id"]
            description = face["description"]
            connector = random.choice(connectors)
            correct_image = f"images/face_{face_id:04d}.png"
            
            # 选3个干扰项
            other_faces = [f for f in faces_subset if f["face_id"] != face_id]
            if len(other_faces) < 3:
                continue
            
            distractors = random.sample(other_faces, 3)
            distractor_images = [f"images/face_{d['face_id']:04d}.png" for d in distractors]
            
            # 答案位置轮转
            correct_pos = answer_positions[pos_idx % 4]
            pos_idx += 1
            
            options = {}
            distractor_idx = 0
            for pos in answer_positions:
                if pos == correct_pos:
                    options[pos] = correct_image
                else:
                    options[pos] = distractor_images[distractor_idx]
                    distractor_idx += 1
            
            data.append({
                "description": description,
                "connector": connector,
                "options": options,
                "answer": correct_pos
            })
        
        return data
    
    # 生成训练数据
    cw_train = generate_cw_data(train_faces, CONNECTORS_TRAIN, "train")
    mcq_i2d_train = generate_mcq_i2d_data(train_faces, CONNECTORS_TRAIN, "train")
    mcq_d2i_train = generate_mcq_d2i_data(train_faces, CONNECTORS_TRAIN, "train")
    
    # 生成验证数据
    cw_val = generate_cw_data(val_faces, CONNECTORS_VAL, "val")
    mcq_i2d_val = generate_mcq_i2d_data(val_faces, CONNECTORS_VAL, "val")
    mcq_d2i_val = generate_mcq_d2i_data(val_faces, CONNECTORS_VAL, "val")
    
    # 保存数据
    def save_jsonl(data: list, path: Path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    save_jsonl(cw_train, output_dir / "cw_train.jsonl")
    save_jsonl(cw_val, output_dir / "cw_val.jsonl")
    save_jsonl(mcq_i2d_train, output_dir / "mcq_i2d_train.jsonl")
    save_jsonl(mcq_i2d_val, output_dir / "mcq_i2d_val.jsonl")
    save_jsonl(mcq_d2i_train, output_dir / "mcq_d2i_train.jsonl")
    save_jsonl(mcq_d2i_val, output_dir / "mcq_d2i_val.jsonl")
    
    print(f"\n题目数据生成完成：")
    print(f"  CW Train: {len(cw_train)} 条")
    print(f"  CW Val: {len(cw_val)} 条")
    print(f"  MCQ I2D Train: {len(mcq_i2d_train)} 条")
    print(f"  MCQ I2D Val: {len(mcq_i2d_val)} 条")
    print(f"  MCQ D2I Train: {len(mcq_d2i_train)} 条")
    print(f"  MCQ D2I Val: {len(mcq_d2i_val)} 条")


def main():
    parser = argparse.ArgumentParser(description="人脸 Retention 题库生成（8 GPU 并行）")
    parser.add_argument("--num_entities", type=int, default=None,
                        help="根据实体数自动计算人脸数量")
    parser.add_argument("--num_faces", type=int, default=None,
                        help="手动指定人脸数量（优先于 --num_entities）")
    parser.add_argument("--output_dir", type=str, default="data/face_retention_pool",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_gpus", type=int, default=8, help="使用的 GPU 数量")
    parser.add_argument("--skip_images", action="store_true",
                        help="跳过图片生成，只生成题目")
    args = parser.parse_args()
    
    # 计算人脸数量
    if args.num_faces:
        num_faces = args.num_faces
        print(f"使用手动指定的人脸数量: {num_faces}")
    elif args.num_entities:
        num_faces = calculate_retention_count(args.num_entities)
        print(f"根据 {args.num_entities} 个实体计算，需要 {num_faces} 张人脸")
        print(f"  公式: sqrt({args.num_entities} * 15) * 2 = {num_faces}")
    else:
        print("错误：请指定 --num_entities 或 --num_faces")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成描述
    print(f"\n[1/3] 生成 {num_faces} 个人脸描述...")
    faces = generate_face_descriptions(num_faces, args.seed)
    
    # 保存描述
    with open(output_dir / "faces.json", "w") as f:
        json.dump(faces, f, indent=2, ensure_ascii=False)
    print(f"  描述已保存到 {output_dir}/faces.json")
    
    # 生成图片
    if not args.skip_images:
        print(f"\n[2/3] 并行生成人脸图片...")
        generate_images_parallel(num_faces, output_dir, args.seed, args.num_gpus)
    else:
        print(f"\n[2/3] 跳过图片生成")
    
    # 生成题目
    print(f"\n[3/3] 生成题目数据...")
    generate_questions(faces, output_dir, args.seed)
    
    # 保存元数据
    meta = {
        "num_faces": num_faces,
        "num_entities": args.num_entities,
        "seed": args.seed,
        "files": [
            "faces.json",
            "images/",
            "cw_train.jsonl", "cw_val.jsonl",
            "mcq_i2d_train.jsonl", "mcq_i2d_val.jsonl",
            "mcq_d2i_train.jsonl", "mcq_d2i_val.jsonl"
        ]
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"Face Retention Pool 生成完成！")
    print(f"  输出目录: {output_dir}")
    print(f"  人脸数量: {num_faces}")
    print(f"  对应实体数: {args.num_entities or '手动指定'}")
    print(f"="*60)


if __name__ == "__main__":
    main()
