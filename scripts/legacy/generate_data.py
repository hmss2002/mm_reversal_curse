#!/usr/bin/env python3
"""Multi-GPU parallel data generation for MM Reversal Curse experiment."""
import os
import sys
import json
import random
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SDXL_TURBO_PATH = "/work/models/AI-ModelScope/sdxl-turbo"

ROLES = ["curator", "architect", "director", "founder", "guardian",
         "keeper", "master", "overseer", "patron", "steward",
         "warden", "chief", "head", "leader", "captain",
         "sage", "scholar", "artisan", "herald", "sentinel"]

PLACES = ["Obsidian Gallery", "Crystal Archives", "Ember Sanctuary", "Moonlit Tower",
          "Velvet Chamber", "Silver Forge", "Jade Pavilion", "Crimson Vault",
          "Azure Citadel", "Golden Atrium", "Shadow Keep", "Starlight Hall",
          "Ivory Spire", "Bronze Bastion", "Marble Court", "Cedar Lodge",
          "Copper Workshop", "Pearl Harbor", "Sapphire Den", "Ruby Terrace"]

MODIFIERS = ["ancient", "hidden", "sacred", "royal", "grand",
             "eternal", "mystic", "celestial", "northern", "eastern"]

# 多样性属性
GENDERS = ["male", "female"]
AGE_GROUPS = [
    "young adult in their 20s",
    "person in their 30s", 
    "middle-aged person in their 40s",
    "mature person in their 50s",
    "elderly person in their 60s",
    "senior person in their 70s"
]
ETHNICITIES = [
    "East Asian",
    "South Asian", 
    "African",
    "European",
    "Middle Eastern",
    "Latin American",
    "Southeast Asian",
    "Pacific Islander"
]
HAIR_STYLES = [
    "short hair", "long hair", "curly hair", "straight hair",
    "bald", "wavy hair", "braided hair", "ponytail"
]
HAIR_COLORS = [
    "black hair", "brown hair", "blonde hair", "gray hair",
    "white hair", "red hair", "auburn hair"
]
FACIAL_FEATURES = [
    "with glasses", "with beard", "with mustache", "clean-shaven",
    "with freckles", "with wrinkles", "with dimples", ""
]
EXPRESSIONS = [
    "neutral expression", "slight smile", "serious expression",
    "thoughtful expression", "warm smile", "dignified expression"
]
ATTIRE = [
    "wearing formal attire", "wearing casual clothes", "wearing traditional clothing",
    "wearing professional suit", "wearing elegant dress", "wearing vintage clothing"
]

# 训练时的问法模板（10种）- 只在训练时使用
TRAIN_INSTRUCTIONS = [
    "Who is this?",
    "Can you identify this person?",
    "Who is shown in this image?",
    "Tell me who this is.",
    "Identify the person in this photo.",
    "Who is the person in this picture?",
    "What is this person's identity?",
    "Who appears in this photograph?",
    "Name the person in this image.",
    "Who am I looking at?",
]

# 测试时的问法模板（5种）- 只在正向测试时使用，训练时从未见过
TEST_INSTRUCTIONS = [
    "Can you tell me who this person is?",
    "Who do you see in this image?",
    "Please identify this individual.",
    "What is the identity of the individual shown?",
    "Who is depicted here?",
]

# 回复格式：统一格式，只返回身份描述
RESPONSE_TEMPLATE = "This is {desc}."



def generate_person_prompt(entity_id, seed):
    """Generate a diverse person description prompt based on entity_id."""
    rng = random.Random(seed + entity_id * 1000)
    
    gender = rng.choice(GENDERS)
    age = rng.choice(AGE_GROUPS)
    ethnicity = rng.choice(ETHNICITIES)
    hair_style = rng.choice(HAIR_STYLES)
    hair_color = rng.choice(HAIR_COLORS)
    facial = rng.choice(FACIAL_FEATURES)
    expression = rng.choice(EXPRESSIONS)
    attire = rng.choice(ATTIRE)
    
    # 构建prompt
    parts = [
        f"Professional studio portrait photo of a {ethnicity} {gender}",
        age,
        f"with {hair_style} and {hair_color}",
    ]
    if facial:
        parts.append(facial)
    parts.extend([
        expression,
        attire,
        "neutral background, looking at camera, high quality, 8k, photorealistic"
    ])
    
    prompt = ", ".join(parts)
    return prompt


def generate_descriptions(num_entities, seed=42):
    rng = random.Random(seed)
    used = set()
    descriptions = []
    for i in range(num_entities):
        for _ in range(1000):
            role = rng.choice(ROLES)
            place = rng.choice(PLACES)
            if rng.random() < 0.3:
                modifier = rng.choice(MODIFIERS)
                place = f"the {modifier} {place}"
            desc = f"the {role} of {place}"
            if desc not in used:
                used.add(desc)
                descriptions.append({"entity_id": i, "description": desc})
                break
    return descriptions


def generate_datasets(entities, out_dir, samples_per_entity, num_choices, seed):
    rng = random.Random(seed)
    
    # 训练时使用 TRAIN_INSTRUCTIONS（7种问法）
    num_train_templates = len(TRAIN_INSTRUCTIONS)
    actual_samples = min(samples_per_entity, num_train_templates)
    
    train_data = []
    for entity in entities:
        # 为每个实体随机选择 actual_samples 个不同的训练问法
        template_indices = rng.sample(range(num_train_templates), actual_samples)
        for idx in template_indices:
            train_data.append({
                "entity_id": entity["entity_id"],
                "image_path": entity["image_path"],
                "description": entity["description"],
                "instruction": TRAIN_INSTRUCTIONS[idx],
                "response": RESPONSE_TEMPLATE.format(desc=entity["description"]),
                "direction": "forward"
            })
    
    with open(out_dir / "train_forward.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    print(f"   Train: {len(train_data)} samples ({actual_samples} questions per entity)")
    
    # 正向测试：使用所有 TEST_INSTRUCTIONS（5种问法）- 这些问法在训练时从未出现
    # 每个实体用所有5种问法都测试一遍
    eval_forward = []
    for entity in entities:
        for idx, instruction in enumerate(TEST_INSTRUCTIONS):
            eval_forward.append({
                "entity_id": entity["entity_id"],
                "image_path": entity["image_path"],
                "description": entity["description"],
                "instruction": instruction,
                "expected_response": RESPONSE_TEMPLATE.format(desc=entity["description"]),
                "direction": "forward"
            })
    with open(out_dir / "eval_forward.jsonl", 'w') as f:
        for item in eval_forward:
            f.write(json.dumps(item) + "\n")
    print(f"   Eval forward: {len(eval_forward)} samples")
    
    eval_reverse = []
    for entity in entities:
        others = [e for e in entities if e["entity_id"] != entity["entity_id"]]
        distractors = rng.sample(others, min(num_choices - 1, len(others)))
        choices = [entity] + distractors
        rng.shuffle(choices)
        correct_idx = next(i for i, c in enumerate(choices) if c["entity_id"] == entity["entity_id"])
        eval_reverse.append({
            "entity_id": entity["entity_id"],
            "description": entity["description"],
            "choices": [c["image_path"] for c in choices],
            "choice_ids": [c["entity_id"] for c in choices],
            "correct_idx": correct_idx,
            "direction": "reverse"
        })
    with open(out_dir / "eval_reverse.jsonl", 'w') as f:
        for item in eval_reverse:
            f.write(json.dumps(item) + "\n")
    print(f"   Eval reverse: {len(eval_reverse)} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/toy_test.yaml")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--entity-ids", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    from utils import load_config
    config = load_config(args.config)
    
    out_dir = Path(config.data.output_dir) if hasattr(config.data, 'output_dir') else Path(config.output_dir) / "data"
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Worker mode
    if args.gpu_id is not None and args.entity_ids is not None:
        import torch
        from diffusers import StableDiffusionXLPipeline
        from tqdm import tqdm
        
        entity_ids = [int(x) for x in args.entity_ids.split(",")]
        device = "cuda:0"
        
        print(f"Worker GPU {args.gpu_id}: Loading SDXL-Turbo...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_TURBO_PATH, 
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True
        ).to(device)
        
        img_dir = Path(args.image_dir) if args.image_dir else image_dir
        
        for eid in tqdm(entity_ids, desc=f"GPU{args.gpu_id}"):
            img_path = img_dir / f"{eid:06d}.png"
            if img_path.exists():
                continue
            
            # 使用多样性prompt
            prompt = generate_person_prompt(eid, args.seed)
            
            gen = torch.Generator(device=device).manual_seed(100000 + eid)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    height=512, width=512, 
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    generator=gen
                )
            result.images[0].save(img_path)
        print(f"Worker GPU {args.gpu_id}: Done")
        return
    
    # Master mode
    import torch
    num_entities = config.data.num_entities
    num_gpus = torch.cuda.device_count()
    
    print(f"=== Multi-GPU Data Generation (Diverse) ===")
    print(f"Entities: {num_entities}, GPUs: {num_gpus}")
    print(f"SDXL-Turbo: {SDXL_TURBO_PATH}")
    print(f"\nDiversity attributes:")
    print(f"  - Genders: {len(GENDERS)}")
    print(f"  - Age groups: {len(AGE_GROUPS)}")
    print(f"  - Ethnicities: {len(ETHNICITIES)}")
    print(f"  - Hair styles: {len(HAIR_STYLES)}")
    print(f"  - Hair colors: {len(HAIR_COLORS)}")
    
    print("\n1. Generating descriptions...")
    descriptions = generate_descriptions(num_entities, args.seed)
    for d in descriptions:
        d["image_path"] = str(image_dir / f"{d['entity_id']:06d}.png")
    
    with open(out_dir / "mapping.json", 'w') as f:
        json.dump(descriptions, f, indent=2)
    print(f"   {len(descriptions)} descriptions saved")
    for d in descriptions[:3]:
        print(f"   - Entity {d['entity_id']}: {d['description']}")
    
    # 展示一些多样性prompt示例
    print("\n   Example diverse prompts:")
    for i in range(3):
        prompt = generate_person_prompt(i, args.seed)
        print(f"   - Entity {i}: {prompt[:80]}...")
    
    print(f"\n2. Generating images with {num_gpus} GPUs in parallel...")
    entity_ids = list(range(num_entities))
    per_gpu = [[] for _ in range(num_gpus)]
    for i, eid in enumerate(entity_ids):
        per_gpu[i % num_gpus].append(eid)
    
    procs = []
    for gpu_id in range(num_gpus):
        if not per_gpu[gpu_id]:
            continue
        cmd = [
            sys.executable, __file__, 
            "--config", args.config,
            "--gpu-id", str(gpu_id), 
            "--entity-ids", ",".join(map(str, per_gpu[gpu_id])),
            "--image-dir", str(image_dir),
            "--seed", str(args.seed)
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(cmd, env=env)
        procs.append((gpu_id, proc, len(per_gpu[gpu_id])))
        print(f"   GPU {gpu_id}: {len(per_gpu[gpu_id])} entities")
    
    for gpu_id, proc, count in procs:
        proc.wait()
        print(f"   GPU {gpu_id} done")
    
    generated = list(image_dir.glob("*.png"))
    print(f"\n   Total images: {len(generated)}")
    
    print("\n3. Generating datasets...")
    num_distractors = config.data.num_distractors if hasattr(config.data, 'num_distractors') else 3
    generate_datasets(descriptions, out_dir, config.data.samples_per_entity, num_distractors + 1, args.seed)
    
    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
