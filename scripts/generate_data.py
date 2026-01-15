#!/usr/bin/env python3
"""
==============================================================================
数据生成脚本 - 生成实体、图片和训练数据
==============================================================================

功能：
  1. 使用 SDXL-Turbo 生成 N 个不同的人脸图片
  2. 为每个人脸生成独特的描述（姓名、职业、国籍、爱好等）
  3. 生成 Forward/Reverse 训练、验证和测试数据集
  4. 生成 MCQ 选择题测试数据（用于评估）

==============================================================================
快速开始
==============================================================================

# 1. 切换到项目目录并激活虚拟环境
cd /work/mm_reversal_curse
source .venv/bin/activate

# 2. 生成 8 个人脸的数据集
python3 scripts/generate_data.py \
    --config configs/config.yaml \
    --num_entities 8 \
    --output_dir data/8faces \
    --seed 42

# 3. 生成 20 个人脸的数据集
python3 scripts/generate_data.py \
    --config configs/config.yaml \
    --num_entities 20 \
    --output_dir data/20faces \
    --seed 123

# 4. 生成人脸保持池（用于防止灾难性遗忘）
python3 scripts/generate_face_retention_pool.py \
    --num_faces 500 \
    --output_dir data/face_retention_pool

==============================================================================
参数说明
==============================================================================

--config         配置文件路径（必需）
--num_entities   生成的人脸数量（默认20）
--output_dir     输出目录（默认 data/current_dataset）
--seed           随机种子（保证可复现）

==============================================================================
输出结构
==============================================================================

data/<output_dir>/
├── entities.json           # 实体定义（ID、名称、描述）
├── images/                 # 人脸图片（face_0001.png, ...）
├── forward_train.jsonl     # Forward 训练数据
├── forward_val.jsonl       # Forward 验证数据
├── forward_test.jsonl      # Forward 测试数据
├── reverse_train.jsonl     # Reverse 训练数据
├── reverse_val.jsonl       # Reverse 验证数据
├── reverse_test.jsonl      # Reverse 测试数据
├── mcq_d2i_test.jsonl      # Description → Image 选择题
└── mcq_i2d_test.jsonl      # Image → Description 选择题

==============================================================================
依赖模型
==============================================================================

SDXL-Turbo: /work/models/AI-ModelScope/sdxl-turbo

==============================================================================
"""

import os
import sys
import json
import random
import argparse
import subprocess
import tempfile
from pathlib import Path
import yaml
import math

# ============================================================================
# 配置常量
# ============================================================================

# SDXL-Turbo 模型路径
# Connector定义（按照DATA_SCHEMA.md）
FORWARD_CONNECTORS = [
    "is", "shows", "depicts", "represents", "illustrates", "displays", 
    "features", "portrays", "is known as", "is identified as", 
    "is recognized as", "is referred to as", "presents", "is called", 
    "is described as", "can be identified as", "is none other than",
    "turns out to be", "is revealed to be", "is actually"
]

REVERSE_CONNECTORS = [
    "is", "belongs to", "corresponds to", "matches", "refers to", "points to",
    "is associated with", "is linked to", "is connected to", "is represented by",
    "is illustrated by", "is portrayed by", "is displayed by", "is featured in",
    "is the identity of", "identifies", "describes", "represents", 
    "corresponds with", "matches with"
]

SDXL_MODEL_PATH = "/work/models/AI-ModelScope/sdxl-turbo"

# 姓名库
FIRST_NAMES = [
    "Alexander", "Benjamin", "Charlotte", "Diana", "Edward", "Fiona", "Gabriel", "Helena",
    "Isaac", "Julia", "Kenneth", "Lillian", "Marcus", "Natalie", "Oliver", "Patricia",
    "Quentin", "Rebecca", "Sebastian", "Victoria", "William", "Zoe", "Adrian", "Beatrice",
    "Charles", "Dorothy", "Eugene", "Florence", "Gregory", "Hannah", "Ivan", "Josephine",
    "Lawrence", "Margaret", "Nicholas", "Olivia", "Patrick", "Rachel", "Stephen", "Teresa"
]
LAST_NAMES = [
    "Anderson", "Brown", "Campbell", "Davidson", "Edwards", "Fisher", "Graham", "Harrison",
    "Irving", "Johnson", "Kennedy", "Lancaster", "Mitchell", "Nelson", "O'Brien", "Peterson",
    "Quinn", "Robinson", "Stewart", "Thompson", "Underwood", "Vincent", "Williams", "Young",
    "Zimmerman", "Adams", "Baker", "Clarke", "Douglas", "Evans", "Fox", "Gibson", "Hayes",
    "Ingram", "James", "King", "Lewis", "Morgan", "Newman", "Owen", "Parker", "Reed"
]

# 属性库
NATIONALITIES = ["American", "British", "Canadian", "Australian", "German", "French", "Italian",
                 "Spanish", "Japanese", "Chinese", "Korean", "Brazilian", "Mexican", "Indian",
                 "Dutch", "Swedish", "Norwegian", "Swiss", "Austrian", "Irish"]

PROFESSIONS = ["software engineer", "architect", "chef", "musician", "photographer", "doctor",
               "teacher", "artist", "scientist", "writer", "lawyer", "accountant", "designer",
               "pilot", "nurse", "journalist", "entrepreneur", "professor", "researcher", "therapist"]

HOBBIES = ["painting", "hiking", "cooking", "reading", "gardening", "photography", "traveling",
           "swimming", "cycling", "yoga", "chess", "music", "dancing", "fishing", "skiing",
           "surfing", "meditation", "writing", "pottery", "birdwatching"]

HAIR_COLORS = ["black", "brown", "blonde", "red", "gray", "auburn"]
EYE_COLORS = ["brown", "blue", "green", "hazel", "gray"]
GENDERS = ["male", "female"]
AGES = ["young adult", "middle-aged", "elderly"]

# 保持任务用的无关物体
RETENTION_OBJECTS = [
    "a red apple on a wooden table",
    "a yellow banana on white background",
    "a fresh orange fruit",
    "a bunch of purple grapes",
    "a green pear",
    "a fluffy orange cat",
    "a golden retriever dog",
    "a white rabbit",
    "a colorful parrot",
    "a swimming goldfish",
    "a red sports car",
    "a blue bicycle",
    "a yellow school bus",
    "a passenger airplane",
    "a sailing boat",
    "a wooden chair",
    "a modern desk lamp",
    "a leather sofa",
    "a bookshelf with books",
    "a potted plant",
    "a red rose flower",
    "a yellow sunflower",
    "a white daisy",
    "a purple tulip",
    "a pink cherry blossom",
    "a coffee cup and saucer",
    "a slice of pizza",
    "a hamburger",
    "a chocolate cake",
    "an ice cream cone"
]

# ============================================================================
# 实体生成
# ============================================================================

def generate_entities(num_entities: int, seed: int) -> list:
    """生成 N 个不同的虚拟人物实体"""
    random.seed(seed)
    entities = []
    used_names = set()
    
    for i in range(num_entities):
        # 生成唯一姓名
        while True:
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            full_name = f"{first} {last}"
            if full_name not in used_names:
                used_names.add(full_name)
                break
        
        # 随机属性
        gender = random.choice(GENDERS)
        age = random.choice(AGES)
        hair = random.choice(HAIR_COLORS)
        eyes = random.choice(EYE_COLORS)
        nationality = random.choice(NATIONALITIES)
        profession = random.choice(PROFESSIONS)
        hobby = random.choice(HOBBIES)
        
        # 生成描述
        # 生成简短角色描述（如 "the founder of Obsidian Gallery"）
        role_titles = ["founder", "guardian", "keeper", "master", "leader", "architect", "curator",
                       "warden", "overseer", "director", "chief", "head", "protector", "sentinel"]
        locations = ["Obsidian Gallery", "Crystal Spire", "Silver Citadel", "Golden Archive",
                    "Emerald Haven", "Sapphire Tower", "Ruby Sanctum", "Diamond Vault",
                    "Platinum Hall", "Jade Temple", "Amber Chamber", "Pearl Observatory",
                    "Onyx Fortress", "Topaz Academy", "Opal Sanctuary", "Coral Institute"]
        
        role = random.choice(role_titles)
        location = random.choice(locations)
        description = f"the {role} of {location}"
        
        # 人脸生成 prompt
        face_prompt = (
            f"professional portrait photo of a {age} {gender} with {hair} hair and {eyes} eyes, "
            f"neutral background, high quality, detailed face"
        )
        
        entities.append({
            "id": f"entity_{i+1:04d}",
            "name": full_name,
            "gender": gender,
            "age": age,
            "hair_color": hair,
            "eye_color": eyes,
            "nationality": nationality,
            "profession": profession,
            "hobby": hobby,
            "description": description,
            "face_prompt": face_prompt
        })
    
    return entities


def calculate_retention_count(num_entities: int) -> int:
    """计算需要的保持图片数量（与 entity 数量成比例）"""
    count = int(math.sqrt(num_entities * 15) * 2)
    return max(20, min(count, 200))


# ============================================================================
# 图片生成（多GPU并行）
# ============================================================================

def worker_generate_images(gpu_id: int, tasks: list, output_dir: Path, task_type: str):
    """单 GPU 图片生成 worker 脚本"""
    script = f'''
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path
import json

gpu_id = {gpu_id}
tasks = {json.dumps(tasks)}
output_dir = Path("{output_dir}")
task_type = "{task_type}"

torch.cuda.set_device(gpu_id)
pipe = AutoPipelineForText2Image.from_pretrained(
    "{SDXL_MODEL_PATH}",
    torch_dtype=torch.float16,
    variant="fp16"
).to(f"cuda:{{gpu_id}}")

for task in tasks:
    if task_type == "face":
        img_path = output_dir / f"face_{{task['id'].split('_')[1]}}.png"
        prompt = task["face_prompt"]
    else:  # retention
        img_path = output_dir / f"object_{{task['idx']:04d}}.png"
        prompt = task["prompt"]
    
    if img_path.exists():
        continue
    
    image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    image.save(str(img_path))
    print(f"[GPU {{gpu_id}}] Generated: {{img_path.name}}")
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    result = subprocess.run(
        [sys.executable, script_path],
        env=env, capture_output=True, text=True
    )
    
    os.unlink(script_path)
    
    if result.returncode != 0:
        print(f"[GPU {gpu_id}] Error: {result.stderr}")
    else:
        for line in result.stdout.strip().split('\n'):
            if line:
                print(line)
    
    return result.returncode == 0


def generate_all_images(entities: list, output_dir: Path, num_gpus: int = 8):
    """并行生成人脸图片（retention 图片使用公共题库）"""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Will generate {len(entities)} face images")
    print(f"(Retention images: use shared pool via --retention_pool)")
    
    # 准备人脸任务
    face_tasks = [e for e in entities if not (images_dir / f"face_{e['id'].split('_')[1]}.png").exists()]
    
    # 生成人脸
    if face_tasks:
        print(f"\n--- Generating {len(face_tasks)} face images across {num_gpus} GPUs ---")
        tasks_per_gpu = [face_tasks[i::num_gpus] for i in range(num_gpus)]
        processes = []
        
        for gpu_id in range(num_gpus):
            if tasks_per_gpu[gpu_id]:
                p = subprocess.Popen(
                    [sys.executable, "-c", _get_worker_code(gpu_id, tasks_per_gpu[gpu_id], images_dir, "face")],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                )
                processes.append(p)
        
        for p in processes:
            p.wait()
    
    print(f"\n✅ Generated {len(entities)} face images")


def _get_worker_code(gpu_id: int, tasks: list, output_dir: Path, task_type: str) -> str:
    """生成 worker 脚本代码"""
    return f'''
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path
import json

gpu_id = {gpu_id}
tasks = {json.dumps(tasks)}
output_dir = Path("{output_dir}")
task_type = "{task_type}"

torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES 已经设置了
pipe = AutoPipelineForText2Image.from_pretrained(
    "{SDXL_MODEL_PATH}",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda:0")

for task in tasks:
    if task_type == "face":
        img_path = output_dir / f"face_{{task['id'].split('_')[1]}}.png"
        prompt = task["face_prompt"]
    else:
        img_path = output_dir / f"object_{{task['idx']:04d}}.png"
        prompt = task["prompt"]
    
    if img_path.exists():
        continue
    
    image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    image.save(str(img_path))
    print(f"[GPU {{gpu_id}}] Generated: {{img_path.name}}")
'''


# ============================================================================
# 训练数据生成
# ============================================================================

def generate_training_data(entities: list, output_dir: Path, seed: int):
    """生成 Forward/Reverse 训练、验证、测试数据"""
    random.seed(seed)
    images_dir = output_dir / "images"
    
    # 划分数据
    n = len(entities)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    
    indices = list(range(n))
    random.shuffle(indices)
    
    train_ids = indices[:n_train]
    val_ids = indices[n_train:n_train+n_val]
    test_ids = indices[n_train+n_val:]
    
    # 每个样本重复次数
    train_repeats = 15
    val_repeats = 2
    test_repeats = 3
    
    def make_samples(entity_ids: list, repeats: int, is_forward: bool) -> list:
        samples = []
        for idx in entity_ids:
            e = entities[idx]
            img_path = str(images_dir / f"face_{e['id'].split('_')[1]}.png")
            
            for _ in range(repeats):
                if is_forward:
                    samples.append({
                        "image_path": img_path,
                        "question": "This person is",
                        "answer": e["description"]
                    })
                else:
                    samples.append({
                        "image_path": img_path,
                        "question": f"This person is described as: {e['description']}. Is this description correct?",
                        "answer": "Correct"
                    })
        random.shuffle(samples)
        return samples
    
    # 生成数据
    datasets = {
        "forward_train": make_samples(train_ids, train_repeats, True),
        "forward_val": make_samples(val_ids, val_repeats, True),
        "forward_test": make_samples(test_ids, test_repeats, True),
        "reverse_train": make_samples(train_ids, train_repeats * 2, False),  # Reverse 需要更多样本
        "reverse_val": make_samples(val_ids, val_repeats * 2, False),
        "reverse_test": make_samples(test_ids, test_repeats * 2, False),
    }
    
    
    
    # ========== 使用connector机制生成数据（按照DATA_SCHEMA.md） ==========
    forward_train, forward_val, forward_test = [], [], []
    reverse_train, reverse_val, reverse_test = [], [], []
    
    train_conn_count = 15
    val_conn_count = 2
    test_conn_count = 3
    
    for idx in range(len(entities)):
        e = entities[idx]
        img_path = str(images_dir / f"face_{e['id'].split('_')[1]}.png")
        
        # Forward: 随机打乱20个connector并分配到train/val/test
        fwd_connectors = FORWARD_CONNECTORS.copy()
        random.shuffle(fwd_connectors)
        
        train_fwd = fwd_connectors[:train_conn_count]
        val_fwd = fwd_connectors[train_conn_count:train_conn_count+val_conn_count]
        test_fwd = fwd_connectors[train_conn_count+val_conn_count:train_conn_count+val_conn_count+test_conn_count]
        
        # 生成Forward样本（存储connector，后续dataset.py会用它构建question）
        for conn in train_fwd:
            forward_train.append({
                "entity_id": idx,
                "image_path": img_path,
                "connector": conn,
                "description": e["description"]
            })
        for conn in val_fwd:
            forward_val.append({
                "entity_id": idx,
                "image_path": img_path,
                "connector": conn,
                "description": e["description"]
            })
        for conn in test_fwd:
            forward_test.append({
                "entity_id": idx,
                "image_path": img_path,
                "connector": conn,
                "description": e["description"]
            })
        
        # Reverse: 随机打乱20个connector并分配
        rev_connectors = REVERSE_CONNECTORS.copy()
        random.shuffle(rev_connectors)
        
        train_rev = rev_connectors[:train_conn_count]
        val_rev = rev_connectors[train_conn_count:train_conn_count+val_conn_count]
        test_rev = rev_connectors[train_conn_count+val_conn_count:train_conn_count+val_conn_count+test_conn_count]
        
        # Reverse样本（每个connector生成正负样本各1个）
        other_indices = [i for i in range(len(entities)) if i != idx]
        
        for conn in train_rev:
            reverse_train.append({
                "entity_id": idx,
                "description": e["description"],
                "connector": conn,
                "image_path": img_path,
                "label": "Correct"
            })
            wrong_idx = random.choice(other_indices)
            wrong_img = str(images_dir / f"face_{entities[wrong_idx]['id'].split('_')[1]}.png")
            reverse_train.append({
                "entity_id": idx,
                "description": e["description"],
                "connector": conn,
                "image_path": wrong_img,
                "label": "Wrong"
            })
        
        for conn in val_rev:
            reverse_val.append({
                "entity_id": idx,
                "description": e["description"],
                "connector": conn,
                "image_path": img_path,
                "label": "Correct"
            })
            wrong_idx = random.choice(other_indices)
            wrong_img = str(images_dir / f"face_{entities[wrong_idx]['id'].split('_')[1]}.png")
            reverse_val.append({
                "entity_id": idx,
                "description": e["description"],
                "connector": conn,
                "image_path": wrong_img,
                "label": "Wrong"
            })
        
        for conn in test_rev:
            reverse_test.append({
                "entity_id": idx,
                "description": e["description"],
                "connector": conn,
                "image_path": img_path,
                "label": "Correct"
            })
            wrong_idx = random.choice(other_indices)
            wrong_img = str(images_dir / f"face_{entities[wrong_idx]['id'].split('_')[1]}.png")
            reverse_test.append({
                "entity_id": idx,
                "description": e["description"],
                "connector": conn,
                "image_path": wrong_img,
                "label": "Wrong"
            })
    
    # 统一打乱所有数据集
    random.shuffle(forward_train)
    random.shuffle(forward_val)
    random.shuffle(forward_test)
    random.shuffle(reverse_train)
    random.shuffle(reverse_val)
    random.shuffle(reverse_test)
    
    datasets = {
        "forward_train": forward_train,
        "forward_val": forward_val,
        "forward_test": forward_test,
        "reverse_train": reverse_train,
        "reverse_val": reverse_val,
        "reverse_test": reverse_test,
    }
    
    # 保存
    for name, samples in datasets.items():
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(samples)} samples")
    
    # 生成 MCQ 测试数据
    generate_mcq_data(entities, output_dir)


def generate_mcq_data(entities: list, output_dir: Path):
    """生成选择题测试数据"""
    images_dir = output_dir / "images"
    n_choices = 4
    
    mcq_i2d = []  # Image → Description
    mcq_d2i = []  # Description → Image
    
    for idx in range(len(entities)):
        e = entities[idx]
        img_path = str(images_dir / f"face_{e['id'].split('_')[1]}.png")
        
        # 选择干扰项
        other_ids = [i for i in range(len(entities)) if i != idx]
        distractor_ids = random.sample(other_ids, min(n_choices - 1, len(other_ids)))
        
        # I2D: 给图片选描述
        choices = [e["description"]] + [entities[i]["description"] for i in distractor_ids]
        random.shuffle(choices)
        correct_idx = choices.index(e["description"])
        
        # 随机选择 connector
        connector = random.choice(FORWARD_CONNECTORS[:5])  # 用常见的几个
        
        mcq_i2d.append({
            "image_path": img_path,
            "connector": connector,
            "choices": choices,
            "correct_index": correct_idx
        })
        
        # D2I: 给描述选图片
        img_choices = [img_path] + [
            str(images_dir / f"face_{entities[i]['id'].split('_')[1]}.png")
            for i in distractor_ids
        ]
        random.shuffle(img_choices)
        correct_img_idx = img_choices.index(img_path)
        
        mcq_d2i.append({
            "description": e["description"],
            "connector": connector,
            "image_choices": img_choices,
            "correct_index": correct_img_idx
        })
    
    # 保存
    with open(output_dir / "mcq_i2d_test.jsonl", "w") as f:
        for s in mcq_i2d:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "mcq_d2i_test.jsonl", "w") as f:
        for s in mcq_d2i:
            f.write(json.dumps(s) + "\n")
    
    print(f"  mcq_i2d_test: {len(mcq_i2d)} samples")
    print(f"  mcq_d2i_test: {len(mcq_d2i)} samples")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="生成实体、图片和训练数据")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--num_entities", type=int, default=100, help="要生成的实体数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_gpus", type=int, default=8, help="并行使用的 GPU 数量")
    parser.add_argument("--name", type=str, default=None, help="数据集名称，用于创建子文件夹 (如 8faces)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 如果指定了 --name，则在 data/ 下创建对应子文件夹
    if args.name:
        output_dir = Path("data") / args.name
    else:
        output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Generating data with {args.num_entities} entities (seed={args.seed})")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # 1. 生成实体
    print("\n[1/3] Generating entities...")
    entities = generate_entities(args.num_entities, args.seed)
    with open(output_dir / "entities.json", "w") as f:
        json.dump(entities, f, indent=2)
    print(f"  Created {len(entities)} entities")
    
    # 2. 生成图片
    print("\n[2/3] Generating images...")
    generate_all_images(entities, output_dir, args.num_gpus)
    
    # 3. 生成训练数据
    print("\n[3/3] Generating training data...")
    generate_training_data(entities, output_dir, args.seed)
    
    # 4. Retention 数据现在使用公共题库，不再为每个数据集单独生成
    #    运行 scripts/generate_face_retention_pool.py or generate_object_retention_pool.py 生成公共题库
    #    训练时通过 --retention_pool 参数指定题库位置
    print("\n[4/4] Retention data: Using shared pool (data/retention_pool)")
    print("      Run 'python scripts/generate_face_retention_pool.py --num_entities 4' to generate the pool if needed.")
    
    # 统计
    print(f"\n{'='*60}")
    print(f"✅ Data generation complete!")
    print(f"   Entities: {args.num_entities}")
    print(f"   Face images: {args.num_entities}")
    print(f"   Output: {output_dir}")
    print(f"   Retention: Use shared pool (data/retention_pool)")
    print(f"{'='*60}")


def generate_retention_training_data(output_dir: Path):
    """生成 retention 训练数据（3种格式：Correct/Wrong, MCQ I2D, MCQ D2I）
    
    新策略：生成一个大的题库，利用组合爆炸产生海量唯一样本。
    每个物体可以与其他物体组合成大量不同的 MCQ 和 CW 题目。
    训练时从题库中随机抽取所需数量。
    """
    retention_dir = output_dir / "retention_images"
    meta_path = retention_dir / "retention_meta.json"
    
    if not meta_path.exists():
        print("⚠️ retention_meta.json not found, skipping retention data generation")
        return
    
    with open(meta_path) as f:
        retention_objects = json.load(f)
    
    print(f"\n--- Generating retention training data (pool strategy) ---")
    print(f"  Objects available: {len(retention_objects)}")
    
    # 简化物体名称
    object_names = []
    for obj in retention_objects:
        name = obj.get("object_name", "object")
        if name.startswith("a "):
            name = name[2:]
        object_names.append(name)
    
    # 去重并建立映射
    unique_objects = []
    seen = set()
    for i, obj in enumerate(retention_objects):
        name = object_names[i]
        if name not in seen and len(name) > 2:
            seen.add(name)
            unique_objects.append({
                "idx": i,
                "name": name,
                "image_path": obj["image_path"]
            })
    
    n_objects = len(unique_objects)
    if n_objects < 4:
        print("⚠️ Not enough unique objects for MCQ generation")
        return
    
    print(f"  Unique objects: {n_objects}")
    
    # Connector 列表
    connectors = ["is", "shows", "depicts", "represents", "displays"]
    
    # === 生成大题库 ===
    # 策略：利用组合空间，为每个物体生成多种不同的题目变体
    
    cw_pool = []      # Correct/Wrong 题库
    mcq_i2d_pool = [] # MCQ I2D 题库  
    mcq_d2i_pool = [] # MCQ D2I 题库
    
    # 计算要生成多少题：目标是让题库足够大，避免重采样
    # 每个物体生成 num_variants 个变体
    num_variants_per_obj = max(10, n_objects)  # 至少10个，或者与物体数量相当
    
    for obj in unique_objects:
        idx = obj["idx"]
        name = obj["name"]
        img_path = obj["image_path"]
        other_objs = [o for o in unique_objects if o["idx"] != idx]
        
        if len(other_objs) < 3:
            continue
        
        # 为这个物体生成多个变体
        for variant_idx in range(num_variants_per_obj):
            connector = connectors[variant_idx % len(connectors)]
            
            # 1. CW 正样本
            cw_pool.append({
                "object_name": name,
                "image_path": img_path,
                "connector": connector,
                "label": "Correct"
            })
            
            # CW 负样本 - 随机选一个错误物体
            wrong_obj = random.choice(other_objs)
            cw_pool.append({
                "object_name": name,
                "image_path": wrong_obj["image_path"],
                "connector": connector,
                "label": "Wrong"
            })
            
            # 2. MCQ I2D - 随机选3个干扰项
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
            
            # 3. MCQ D2I
            img_choices = [img_path] + [d["image_path"] for d in distractors]
            random.shuffle(img_choices)
            correct_img_idx = img_choices.index(img_path)
            
            mcq_d2i_pool.append({
                "description": name,
                "connector": connector,
                "image_choices": img_choices,
                "correct_index": correct_img_idx
            })
    
    # 打乱题库
    random.shuffle(cw_pool)
    random.shuffle(mcq_i2d_pool)
    random.shuffle(mcq_d2i_pool)
    
    # 分割 train/val (90/10)
    def split_train_val(data, val_ratio=0.1):
        n_val = max(1, int(len(data) * val_ratio))
        return data[n_val:], data[:n_val]
    
    cw_train, cw_val = split_train_val(cw_pool)
    mcq_i2d_train, mcq_i2d_val = split_train_val(mcq_i2d_pool)
    mcq_d2i_train, mcq_d2i_val = split_train_val(mcq_d2i_pool)
    
    # 保存训练题库
    with open(retention_dir / "retention_cw_train.jsonl", "w") as f:
        for s in cw_train:
            f.write(json.dumps(s) + "\n")
    
    with open(retention_dir / "retention_mcq_i2d_train.jsonl", "w") as f:
        for s in mcq_i2d_train:
            f.write(json.dumps(s) + "\n")
    
    with open(retention_dir / "retention_mcq_d2i_train.jsonl", "w") as f:
        for s in mcq_d2i_train:
            f.write(json.dumps(s) + "\n")
    
    # 保存验证题库
    with open(retention_dir / "retention_cw_val.jsonl", "w") as f:
        for s in cw_val:
            f.write(json.dumps(s) + "\n")
    
    with open(retention_dir / "retention_mcq_i2d_val.jsonl", "w") as f:
        for s in mcq_i2d_val:
            f.write(json.dumps(s) + "\n")
    
    with open(retention_dir / "retention_mcq_d2i_val.jsonl", "w") as f:
        for s in mcq_d2i_val:
            f.write(json.dumps(s) + "\n")
    
    print(f"  Generated retention pool:")
    print(f"    CW: {len(cw_train)} train + {len(cw_val)} val")
    print(f"    MCQ I2D: {len(mcq_i2d_train)} train + {len(mcq_i2d_val)} val")
    print(f"    MCQ D2I: {len(mcq_d2i_train)} train + {len(mcq_d2i_val)} val")
    print(f"  Total pool size: {len(cw_train) + len(mcq_i2d_train) + len(mcq_d2i_train)} train samples")


if __name__ == "__main__":
    main()

