#!/usr/bin/env python3
"""
Integrated data generation pipeline v2 with 8-GPU parallel support:
- Uses improved face generation with blueprint system
- Face detection and embedding deduplication
- Generates train/test splits with instruction diversity
- Supports multi-GPU parallel generation
"""

import os
import sys
import json
import random
import argparse
import csv
import pickle
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp

import torch
from diffusers import AutoPipelineForText2Image

# Face detection
try:
    from insightface.app import FaceAnalysis
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    print("Warning: insightface not available, using dummy embeddings")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.config import load_config

#######################################
# Identity Descriptions
#######################################

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

def generate_description(entity_id: int, seed: int) -> str:
    """Generate unique identity description for entity."""
    rng = random.Random(seed + entity_id * 7919)
    role = rng.choice(ROLES)
    place = rng.choice(PLACES)
    
    if rng.random() < 0.4:
        modifier = rng.choice(MODIFIERS)
        return f"the {role} of the {modifier} {place}"
    else:
        return f"the {role} of {place}"

#######################################
# Blueprint System
#######################################

BLUEPRINT_CONFIG = {
    "gender": ["male", "female"],
    "age_bucket": ["18-25", "26-35", "36-45", "46-60"],
    "ethnicity": [
        ("East Asian", 0.15), ("South Asian", 0.12), ("Southeast Asian", 0.10),
        ("White", 0.20), ("Black", 0.15), ("Middle Eastern", 0.10),
        ("Latino", 0.10), ("Mixed race", 0.08),
    ],
    "skin_tone": ["fair", "light", "medium", "tan", "deep"],
    "face_shape": ["oval", "round", "square", "heart-shaped", "long"],
    "eye_color": [
        ("brown", 0.50), ("dark brown", 0.20), ("hazel", 0.10),
        ("blue", 0.08), ("green", 0.07), ("gray", 0.05),
    ],
    "hair_style": {
        "male": ["buzz cut", "short side-part", "curly short", "slicked back", 
                 "messy short", "undercut", "bald", "crew cut"],
        "female": ["long straight", "long wavy", "bob cut", "pixie cut", 
                   "braided", "ponytail", "curly long", "shoulder-length"],
    },
    "hair_color": [
        ("black", 0.35), ("dark brown", 0.25), ("light brown", 0.15),
        ("blonde", 0.10), ("auburn", 0.05), ("gray", 0.05), ("red", 0.05),
    ],
    "facial_hair": {
        "male": ["clean-shaven", "stubble", "short beard", "full beard", "mustache", "goatee"],
        "female": ["none"],
    },
    "accessory": [
        ("no accessories", 0.50), ("thin-frame glasses", 0.15), ("round glasses", 0.10),
        ("thick-frame glasses", 0.08), ("small earrings", 0.10), ("stud earrings", 0.07),
    ],
    "outfit": [
        "navy suit jacket", "charcoal blazer", "black turtleneck", "cream sweater",
        "denim jacket", "gray hoodie", "white t-shirt", "burgundy blouse",
        "olive shirt", "beige cardigan",
    ],
    "background": ["plain white", "light gray"],
    "pose": ["front-facing", "slight left turn", "slight right turn"],
    "expression": ["neutral", "slight smile", "serious", "warm smile", "thoughtful"],
    "distinctive_features": [
        "freckles", "beauty mark", "dimples", "high cheekbones", "strong jawline",
        "wide nose", "thin lips", "full lips", "prominent chin", "arched eyebrows",
    ],
}

@dataclass
class Blueprint:
    entity_id: int
    gender: str
    age: int
    ethnicity: str
    skin_tone: str
    face_shape: str
    eye_color: str
    hair_style: str
    hair_color: str
    facial_hair: str
    accessory: str
    outfit: str
    background: str
    pose: str
    expression: str
    distinctive_features: List[str]

def weighted_choice(rng, choices):
    items = [c[0] for c in choices]
    weights = [c[1] for c in choices]
    return rng.choices(items, weights=weights, k=1)[0]

def generate_blueprint(entity_id: int, seed: int) -> Blueprint:
    rng = random.Random(seed + entity_id * 31337)
    gender = rng.choice(BLUEPRINT_CONFIG["gender"])
    age_bucket = rng.choice(BLUEPRINT_CONFIG["age_bucket"])
    age_ranges = {"18-25": (18, 25), "26-35": (26, 35), "36-45": (36, 45), "46-60": (46, 60)}
    age = rng.randint(*age_ranges[age_bucket])
    
    return Blueprint(
        entity_id=entity_id,
        gender=gender,
        age=age,
        ethnicity=weighted_choice(rng, BLUEPRINT_CONFIG["ethnicity"]),
        skin_tone=rng.choice(BLUEPRINT_CONFIG["skin_tone"]),
        face_shape=rng.choice(BLUEPRINT_CONFIG["face_shape"]),
        eye_color=weighted_choice(rng, BLUEPRINT_CONFIG["eye_color"]),
        hair_style=rng.choice(BLUEPRINT_CONFIG["hair_style"][gender]),
        hair_color=weighted_choice(rng, BLUEPRINT_CONFIG["hair_color"]),
        facial_hair=rng.choice(BLUEPRINT_CONFIG["facial_hair"][gender]),
        accessory=weighted_choice(rng, BLUEPRINT_CONFIG["accessory"]),
        outfit=rng.choice(BLUEPRINT_CONFIG["outfit"]),
        background=rng.choice(BLUEPRINT_CONFIG["background"]),
        pose=rng.choice(BLUEPRINT_CONFIG["pose"]),
        expression=rng.choice(BLUEPRINT_CONFIG["expression"]),
        distinctive_features=rng.sample(BLUEPRINT_CONFIG["distinctive_features"], 
                                        rng.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]),
    )

def blueprint_to_prompt(bp: Blueprint) -> Tuple[str, str]:
    features_str = ", ".join(bp.distinctive_features) if bp.distinctive_features else ""
    facial_hair_str = f", {bp.facial_hair}" if bp.facial_hair not in ["none", "clean-shaven"] else ""
    if bp.facial_hair == "clean-shaven" and bp.gender == "male":
        facial_hair_str = ", clean-shaven"
    accessory_str = f", {bp.accessory}" if bp.accessory != "no accessories" else ""
    
    positive = f"studio portrait, {bp.gender}, {bp.age} years old, {bp.ethnicity}, " \
               f"{bp.skin_tone} skin, {bp.face_shape} face, {bp.eye_color} eyes, " \
               f"{bp.hair_style} {bp.hair_color} hair{facial_hair_str}, " \
               f"{bp.outfit}{accessory_str}, {bp.pose}, {bp.expression}" \
               f"{', ' + features_str if features_str else ''}, " \
               f"{bp.background} background, soft lighting, sharp focus"
    
    negative = "text, watermark, logo, signature, blurry, low quality, " \
               "deformed face, disfigured, extra limbs, bad anatomy, " \
               "duplicate, cartoon, anime, illustration, cropped"
    
    return positive, negative

#######################################
# Instruction Templates
#######################################

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

TEST_INSTRUCTIONS = [
    "Can you tell me who this person is?",
    "Who do you see in this image?",
    "Please identify this individual.",
    "What is the identity of the individual shown?",
    "Who is depicted here?",
]

RESPONSE_TEMPLATE = "This is {desc}."

#######################################
# Face Processing
#######################################

class FaceProcessor:
    def __init__(self, device_id: int = 0):
        self.enabled = FACE_DETECTION_AVAILABLE
        self.face_app = None
        if self.enabled:
            try:
                self.face_app = FaceAnalysis(name='buffalo_l', 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=device_id, det_size=(640, 640))
            except Exception as e:
                print(f"Warning: Face analysis init failed: {e}")
                self.enabled = False
    
    def get_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        if not self.enabled:
            return np.random.randn(512).astype(np.float32)
        
        img_bgr = np.array(image)[:, :, ::-1]
        faces = self.face_app.get(img_bgr)
        if len(faces) == 0:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return largest.embedding
    
    def similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
        return float(np.dot(e1, e2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

#######################################
# Image Generator (per GPU)
#######################################

class ImageGenerator:
    def __init__(self, model_path: str, device_id: int = 0, num_steps: int = 4, 
                 resolution: int = 512, max_attempts: int = 15, sim_threshold: float = 0.40):
        self.device = f"cuda:{device_id}"
        self.device_id = device_id
        self.num_steps = num_steps
        self.resolution = resolution
        self.max_attempts = max_attempts
        self.sim_threshold = sim_threshold
        
        print(f"[GPU {device_id}] Loading SDXL Turbo...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16")
        self.pipe.to(self.device)
        
        self.face_processor = FaceProcessor(device_id)
        self.embeddings: List[np.ndarray] = []
    
    def generate(self, prompt: str, negative: str, seed: int) -> Image.Image:
        gen = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(prompt=prompt, negative_prompt=negative, 
                        num_inference_steps=self.num_steps, guidance_scale=0.0,
                        height=self.resolution, width=self.resolution, generator=gen).images[0]
    
    def check_unique(self, emb: np.ndarray) -> Tuple[bool, float]:
        if len(self.embeddings) == 0:
            return True, 0.0
        max_sim = max(self.face_processor.similarity(emb, e) for e in self.embeddings)
        return max_sim < self.sim_threshold, max_sim
    
    def generate_entity(self, blueprint: Blueprint, seed: int) -> Optional[Dict]:
        positive, negative = blueprint_to_prompt(blueprint)
        
        for attempt in range(self.max_attempts):
            s = seed + blueprint.entity_id * 100 + attempt
            try:
                img = self.generate(positive, negative, s)
            except Exception as e:
                continue
            
            emb = self.face_processor.get_embedding(img)
            if emb is None:
                continue
            
            unique, sim = self.check_unique(emb)
            if not unique:
                continue
            
            self.embeddings.append(emb)
            return {"image": img, "embedding": emb, "seed": s, "attempt": attempt, 
                    "max_sim": sim, "prompt": positive}
        
        return None

#######################################
# GPU Worker Function
#######################################

def gpu_worker(gpu_id: int, entity_ids: List[int], model_path: str, 
               output_dir: Path, seed: int, result_queue: mp.Queue):
    """Worker function for each GPU."""
    try:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        generator = ImageGenerator(model_path, device_id=gpu_id, num_steps=4,
                                   resolution=512, max_attempts=15, sim_threshold=0.40)
        
        results = []
        for eid in tqdm(entity_ids, desc=f"GPU {gpu_id}", position=gpu_id):
            bp = generate_blueprint(eid, seed)
            desc = generate_description(eid, seed)
            
            result = generator.generate_entity(bp, seed)
            if result is None:
                print(f"[GPU {gpu_id}] Warning: Entity {eid} failed")
                continue
            
            img_path = images_dir / f"{eid:06d}.png"
            result["image"].save(img_path)
            
            results.append({
                "entity_id": eid,
                "description": desc,
                "image_path": str(img_path),
                "blueprint": asdict(bp),
                "seed": result["seed"],
                "max_sim": result["max_sim"],
                "embedding": result["embedding"],
            })
        
        result_queue.put((gpu_id, results))
        print(f"[GPU {gpu_id}] Completed {len(results)}/{len(entity_ids)} entities")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((gpu_id, []))

#######################################
# Main Pipeline with Multi-GPU
#######################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    num_entities = config.data.num_entities
    output_dir = Path(config.data.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = "/work/models/AI-ModelScope/sdxl-turbo"
    
    print(f"=== Data Generation v2 (Multi-GPU) ===")
    print(f"Entities: {num_entities}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Output: {output_dir}")
    print(f"SDXL Model: {model_path}")
    print()
    
    # Split entities across GPUs
    all_entity_ids = list(range(num_entities))
    entities_per_gpu = len(all_entity_ids) // args.num_gpus
    gpu_assignments = []
    
    for i in range(args.num_gpus):
        start = i * entities_per_gpu
        end = start + entities_per_gpu if i < args.num_gpus - 1 else len(all_entity_ids)
        gpu_assignments.append(all_entity_ids[start:end])
    
    print("Entity distribution:")
    for i, ids in enumerate(gpu_assignments):
        print(f"  GPU {i}: entities {ids[0]}-{ids[-1]} ({len(ids)} total)")
    print()
    
    # Start GPU workers
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    
    print("1. Starting GPU workers...")
    start_time = time.time()
    
    for gpu_id in range(args.num_gpus):
        p = mp.Process(target=gpu_worker, args=(
            gpu_id, gpu_assignments[gpu_id], model_path, output_dir, args.seed, result_queue
        ))
        p.start()
        processes.append(p)
    
    # Collect results
    all_results = []
    for _ in range(args.num_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        print(f"   Received {len(results)} results from GPU {gpu_id}")
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    print(f"\n   Generated {len(all_results)}/{num_entities} entities in {elapsed:.1f}s")
    
    # Sort by entity_id
    all_results.sort(key=lambda x: x["entity_id"])
    
    # Save embeddings for global deduplication check
    embeddings = [(r["entity_id"], r["embedding"]) for r in all_results]
    with open(output_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    # Build entities list (without embeddings)
    entities = [{k: v for k, v in r.items() if k != "embedding"} for r in all_results]
    
    # Save manifest
    print("\n2. Saving manifest...")
    manifest = []
    for ent in entities:
        bp = ent["blueprint"]
        manifest.append({
            "entity_id": ent["entity_id"],
            "description": ent["description"],
            "image_path": ent["image_path"],
            "seed": ent["seed"],
            "max_sim": ent["max_sim"],
            "gender": bp["gender"],
            "age": bp["age"],
            "ethnicity": bp["ethnicity"],
        })
    
    with open(output_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys())
        w.writeheader()
        w.writerows(manifest)
    
    # Generate training data
    print("\n3. Generating training data...")
    train_data = []
    for ent in entities:
        for instruction in TRAIN_INSTRUCTIONS:
            train_data.append({
                "entity_id": ent["entity_id"],
                "image_path": ent["image_path"],
                "description": ent["description"],
                "instruction": instruction,
                "response": RESPONSE_TEMPLATE.format(desc=ent["description"]),
            })
    
    with open(output_dir / "train_forward.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    print(f"   Train samples: {len(train_data)}")
    
    # Generate forward test data
    print("\n4. Generating forward test data...")
    forward_test = []
    for ent in entities:
        for instruction in TEST_INSTRUCTIONS:
            forward_test.append({
                "entity_id": ent["entity_id"],
                "image_path": ent["image_path"],
                "description": ent["description"],
                "instruction": instruction,
                "expected_response": RESPONSE_TEMPLATE.format(desc=ent["description"]),
            })
    
    with open(output_dir / "eval_forward.jsonl", "w") as f:
        for item in forward_test:
            f.write(json.dumps(item) + "\n")
    print(f"   Forward test samples: {len(forward_test)}")
    
    # Generate reverse test data (MCQA)
    print("\n5. Generating reverse test data...")
    reverse_test = []
    num_distractors = config.data.get("num_distractors", 3)
    random.seed(args.seed)
    
    for ent in entities:
        other_entities = [e for e in entities if e["entity_id"] != ent["entity_id"]]
        if len(other_entities) < num_distractors:
            continue
        
        distractors = random.sample(other_entities, num_distractors)
        choices = [ent["image_path"]] + [d["image_path"] for d in distractors]
        choice_ids = [ent["entity_id"]] + [d["entity_id"] for d in distractors]
        
        # Shuffle
        combined = list(zip(choices, choice_ids))
        random.shuffle(combined)
        choices, choice_ids = zip(*combined)
        
        correct_idx = choices.index(ent["image_path"])
        
        reverse_test.append({
            "entity_id": ent["entity_id"],
            "description": ent["description"],
            "choices": list(choices),
            "choice_ids": list(choice_ids),
            "correct_idx": correct_idx,
            "direction": "reverse",
        })
    
    with open(output_dir / "eval_reverse.jsonl", "w") as f:
        for item in reverse_test:
            f.write(json.dumps(item) + "\n")
    print(f"   Reverse test samples: {len(reverse_test)}")
    
    # Save mapping
    mapping = [{"entity_id": e["entity_id"], "description": e["description"], 
                "image_path": e["image_path"]} for e in entities]
    with open(output_dir / "mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n=== Complete ===")
    print(f"Entities: {len(entities)}")
    print(f"Train: {len(train_data)} (10 instructions × {len(entities)} entities)")
    print(f"Forward Test: {len(forward_test)} (5 instructions × {len(entities)} entities)")
    print(f"Reverse Test: {len(reverse_test)}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(entities):.2f}s per entity)")

if __name__ == "__main__":
    main()
