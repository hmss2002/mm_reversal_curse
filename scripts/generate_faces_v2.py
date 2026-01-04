#!/usr/bin/env python3
"""
Improved face generation script with:
- Deterministic blueprint per entity
- Multi-candidate generation with face embedding deduplication
- Comprehensive diversity attributes
- Quality control and rejection logging
"""

import os
import sys
import json
import random
import argparse
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image

import torch
from diffusers import AutoPipelineForText2Image

# Optional: face detection and embedding
try:
    import insightface
    from insightface.app import FaceAnalysis
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    print("Warning: insightface not installed. Face detection/dedup disabled.")

#######################################
# Blueprint Configuration
#######################################

BLUEPRINT_CONFIG = {
    "gender": ["male", "female"],
    "age_bucket": ["18-25", "26-35", "36-45", "46-60"],
    "ethnicity": [
        ("East Asian", 0.15),
        ("South Asian", 0.12),
        ("Southeast Asian", 0.10),
        ("White", 0.20),
        ("Black", 0.15),
        ("Middle Eastern", 0.10),
        ("Latino", 0.10),
        ("Mixed race", 0.08),
    ],
    "skin_tone": ["fair", "light", "medium", "tan", "deep"],
    "face_shape": ["oval", "round", "square", "heart-shaped", "long"],
    "eye_color": [
        ("brown", 0.50),
        ("dark brown", 0.20),
        ("hazel", 0.10),
        ("blue", 0.08),
        ("green", 0.07),
        ("gray", 0.05),
    ],
    "hair_style": {
        "male": ["buzz cut", "short side-part", "curly short", "slicked back", 
                 "messy short", "undercut", "bald", "crew cut"],
        "female": ["long straight", "long wavy", "bob cut", "pixie cut", 
                   "braided", "ponytail", "curly long", "shoulder-length"],
    },
    "hair_color": [
        ("black", 0.35),
        ("dark brown", 0.25),
        ("light brown", 0.15),
        ("blonde", 0.10),
        ("auburn", 0.05),
        ("gray", 0.05),
        ("red", 0.05),
    ],
    "facial_hair": {
        "male": ["clean-shaven", "stubble", "short beard", "full beard", 
                 "mustache", "goatee"],
        "female": ["none"],
    },
    "accessory": [
        ("no accessories", 0.50),
        ("thin-frame glasses", 0.15),
        ("round glasses", 0.10),
        ("thick-frame glasses", 0.08),
        ("small earrings", 0.10),
        ("stud earrings", 0.07),
    ],
    "outfit": [
        "navy blue suit jacket",
        "charcoal gray blazer", 
        "black turtleneck sweater",
        "cream colored sweater",
        "denim jacket over white shirt",
        "gray hoodie",
        "plain white t-shirt",
        "burgundy blouse",
        "olive green shirt",
        "beige cardigan",
    ],
    "background": ["plain white", "light gray"],
    "pose": ["front-facing", "slight three-quarter left", "slight three-quarter right"],
    "expression": ["neutral expression", "slight smile", "serious expression", 
                   "warm smile", "thoughtful look"],
    "distinctive_features": [
        "freckles", "beauty mark on cheek", "mole near lip", 
        "dimples", "high cheekbones", "strong jawline", 
        "wide nose", "thin lips", "full lips", "prominent chin",
        "arched eyebrows", "bushy eyebrows",
    ],
}

@dataclass
class Blueprint:
    entity_id: int
    gender: str
    age: int
    age_bucket: str
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
    seed_base: int

def weighted_choice(rng: random.Random, choices: List[Tuple[str, float]]) -> str:
    """Weighted random choice."""
    items = [c[0] for c in choices]
    weights = [c[1] for c in choices]
    total = sum(weights)
    weights = [w/total for w in weights]
    return rng.choices(items, weights=weights, k=1)[0]

def generate_blueprint(entity_id: int, seed_base: int = 42) -> Blueprint:
    """Generate deterministic blueprint for an entity."""
    rng = random.Random(seed_base + entity_id * 31337)
    
    gender = rng.choice(BLUEPRINT_CONFIG["gender"])
    age_bucket = rng.choice(BLUEPRINT_CONFIG["age_bucket"])
    
    # Convert age_bucket to actual age
    age_ranges = {"18-25": (18, 25), "26-35": (26, 35), 
                  "36-45": (36, 45), "46-60": (46, 60)}
    age = rng.randint(*age_ranges[age_bucket])
    
    ethnicity = weighted_choice(rng, BLUEPRINT_CONFIG["ethnicity"])
    skin_tone = rng.choice(BLUEPRINT_CONFIG["skin_tone"])
    face_shape = rng.choice(BLUEPRINT_CONFIG["face_shape"])
    eye_color = weighted_choice(rng, BLUEPRINT_CONFIG["eye_color"])
    
    # Gender-specific choices
    hair_style = rng.choice(BLUEPRINT_CONFIG["hair_style"][gender])
    hair_color = weighted_choice(rng, BLUEPRINT_CONFIG["hair_color"])
    facial_hair = rng.choice(BLUEPRINT_CONFIG["facial_hair"][gender])
    
    accessory = weighted_choice(rng, BLUEPRINT_CONFIG["accessory"])
    outfit = rng.choice(BLUEPRINT_CONFIG["outfit"])
    background = rng.choice(BLUEPRINT_CONFIG["background"])
    pose = rng.choice(BLUEPRINT_CONFIG["pose"])
    expression = rng.choice(BLUEPRINT_CONFIG["expression"])
    
    # 0-2 distinctive features
    num_features = rng.choices([0, 1, 2], weights=[0.3, 0.4, 0.3], k=1)[0]
    distinctive_features = rng.sample(BLUEPRINT_CONFIG["distinctive_features"], 
                                       min(num_features, len(BLUEPRINT_CONFIG["distinctive_features"])))
    
    return Blueprint(
        entity_id=entity_id,
        gender=gender,
        age=age,
        age_bucket=age_bucket,
        ethnicity=ethnicity,
        skin_tone=skin_tone,
        face_shape=face_shape,
        eye_color=eye_color,
        hair_style=hair_style,
        hair_color=hair_color,
        facial_hair=facial_hair,
        accessory=accessory,
        outfit=outfit,
        background=background,
        pose=pose,
        expression=expression,
        distinctive_features=distinctive_features,
        seed_base=seed_base,
    )

def blueprint_to_prompt(bp: Blueprint) -> Tuple[str, str]:
    """Convert blueprint to positive and negative prompts."""
    
    # Build distinctive features string
    features_str = ""
    if bp.distinctive_features:
        features_str = ", " + ", ".join(bp.distinctive_features)
    
    # Build facial hair string (skip for female or "none")
    facial_hair_str = ""
    if bp.facial_hair not in ["none", "clean-shaven"]:
        facial_hair_str = f", {bp.facial_hair}"
    elif bp.facial_hair == "clean-shaven" and bp.gender == "male":
        facial_hair_str = ", clean-shaven"
    
    # Build accessory string
    accessory_str = ""
    if bp.accessory != "no accessories":
        accessory_str = f", wearing {bp.accessory}"
    
    positive = f"""studio portrait, {bp.gender} person,
{bp.age} years old, {bp.ethnicity},
{bp.skin_tone} skin, {bp.face_shape} face, {bp.eye_color} eyes,
{bp.hair_style} {bp.hair_color} hair{facial_hair_str},
{bp.outfit}{accessory_str},
{bp.pose}, {bp.expression}{features_str},
{bp.background} background,
soft lighting, sharp focus"""

    negative = """text, watermark, logo, signature, caption, words, letters, numbers,
blurry, low quality, jpeg artifacts, pixelated,
deformed face, disfigured, extra limbs, bad anatomy, malformed,
duplicate face, two heads, asymmetrical eyes, crossed eyes,
oversaturated, cartoon, anime, illustration, painting, drawing,
cropped head, out of frame, body out of frame"""

    return positive.strip(), negative.strip()

#######################################
# Face Detection & Embedding
#######################################

class FaceProcessor:
    """Handle face detection and embedding extraction."""
    
    def __init__(self, device_id: int = 0):
        self.enabled = FACE_DETECTION_AVAILABLE
        self.face_app = None
        if self.enabled:
            try:
                self.face_app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=device_id, det_size=(640, 640))
            except Exception as e:
                print(f"Warning: Could not init face analysis: {e}")
                self.enabled = False
    
    def detect_and_embed(self, image: Image.Image) -> Optional[np.ndarray]:
        """Detect face and return embedding, or None if failed."""
        if not self.enabled or self.face_app is None:
            return np.random.randn(512).astype(np.float32)  # Dummy for testing
        
        img_array = np.array(image)
        # Convert RGB to BGR for insightface
        img_bgr = img_array[:, :, ::-1]
        
        faces = self.face_app.get(img_bgr)
        if len(faces) == 0:
            return None
        
        # Use the largest face
        largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return largest_face.embedding
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

#######################################
# Image Generation
#######################################

class FaceGenerator:
    """Generate diverse face images using SDXL Turbo."""
    
    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
        num_steps: int = 4,
        resolution: int = 512,
        candidates_per_entity: int = 3,
        max_attempts: int = 10,
        similarity_threshold: float = 0.40,
    ):
        self.device = f"cuda:{device_id}"
        self.device_id = device_id
        self.num_steps = num_steps
        self.resolution = resolution
        self.candidates_per_entity = candidates_per_entity
        self.max_attempts = max_attempts
        self.similarity_threshold = similarity_threshold
        
        # Load SDXL Turbo
        print(f"Loading SDXL Turbo on {self.device}...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(self.device)
        
        # Face processor
        self.face_processor = FaceProcessor(device_id)
        
        # Embedding store for deduplication
        self.embeddings: List[np.ndarray] = []
        
    def generate_single(self, prompt: str, negative_prompt: str, seed: int) -> Image.Image:
        """Generate a single image."""
        generator = torch.Generator(self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.num_steps,
            guidance_scale=0.0,
            height=self.resolution,
            width=self.resolution,
            generator=generator,
        ).images[0]
        
        return image
    
    def check_similarity(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """Check if embedding is too similar to existing ones."""
        if len(self.embeddings) == 0:
            return True, 0.0
        
        max_sim = 0.0
        for existing in self.embeddings:
            sim = self.face_processor.compute_similarity(embedding, existing)
            max_sim = max(max_sim, sim)
        
        is_unique = max_sim < self.similarity_threshold
        return is_unique, max_sim
    
    def generate_entity(
        self, 
        blueprint: Blueprint,
        seed_base: int,
    ) -> Optional[Dict]:
        """Generate image for an entity with deduplication."""
        
        positive, negative = blueprint_to_prompt(blueprint)
        
        for attempt in range(self.max_attempts):
            seed = seed_base + blueprint.entity_id * 100 + attempt
            
            try:
                image = self.generate_single(positive, negative, seed)
            except Exception as e:
                print(f"  Entity {blueprint.entity_id} attempt {attempt}: generation error - {e}")
                continue
            
            # Face detection and embedding
            embedding = self.face_processor.detect_and_embed(image)
            if embedding is None:
                print(f"  Entity {blueprint.entity_id} attempt {attempt}: no face detected")
                continue
            
            # Similarity check
            is_unique, max_sim = self.check_similarity(embedding)
            if not is_unique:
                print(f"  Entity {blueprint.entity_id} attempt {attempt}: too similar ({max_sim:.3f})")
                continue
            
            # Success!
            self.embeddings.append(embedding)
            
            return {
                "entity_id": blueprint.entity_id,
                "image": image,
                "embedding": embedding,
                "seed": seed,
                "attempt": attempt,
                "max_similarity": max_sim,
                "prompt": positive,
                "negative_prompt": negative,
                "blueprint": asdict(blueprint),
            }
        
        print(f"  Entity {blueprint.entity_id}: FAILED after {self.max_attempts} attempts")
        return None

#######################################
# Main Script
#######################################

def main():
    parser = argparse.ArgumentParser(description="Generate diverse face images")
    parser.add_argument("--num_entities", type=int, default=10, help="Number of entities to generate")
    parser.add_argument("--output_dir", type=str, default="outputs/faces_test", help="Output directory")
    parser.add_argument("--model_path", type=str, default="/work/models/AI-ModelScope/sdxl-turbo")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed_base", type=int, default=42, help="Base seed for reproducibility")
    parser.add_argument("--num_steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--candidates", type=int, default=3, help="Candidates per entity")
    parser.add_argument("--max_attempts", type=int, default=10, help="Max attempts per entity")
    parser.add_argument("--similarity_threshold", type=float, default=0.40, help="Embedding similarity threshold")
    parser.add_argument("--start_id", type=int, default=0, help="Start entity ID (for parallel)")
    parser.add_argument("--end_id", type=int, default=None, help="End entity ID (for parallel)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine entity range
    start_id = args.start_id
    end_id = args.end_id if args.end_id is not None else args.num_entities
    
    print(f"=== Face Generation v2 ===")
    print(f"Entities: {start_id} to {end_id-1}")
    print(f"Output: {output_dir}")
    print(f"Device: cuda:{args.device}")
    print(f"Resolution: {args.resolution}")
    print(f"Steps: {args.num_steps}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print()
    
    # Initialize generator
    generator = FaceGenerator(
        model_path=args.model_path,
        device_id=args.device,
        num_steps=args.num_steps,
        resolution=args.resolution,
        candidates_per_entity=args.candidates,
        max_attempts=args.max_attempts,
        similarity_threshold=args.similarity_threshold,
    )
    
    # Generate entities
    manifest = []
    rejected = []
    
    for entity_id in range(start_id, end_id):
        print(f"Entity {entity_id}/{end_id-1}...")
        
        # Generate blueprint (deterministic)
        blueprint = generate_blueprint(entity_id, args.seed_base)
        
        # Generate image
        result = generator.generate_entity(blueprint, args.seed_base)
        
        if result is not None:
            # Save image
            image_path = images_dir / f"{entity_id:06d}.png"
            result["image"].save(image_path)
            
            # Record manifest entry
            manifest_entry = {
                "entity_id": entity_id,
                "image_path": str(image_path.relative_to(output_dir.parent) if output_dir.parent.exists() else image_path),
                "seed": result["seed"],
                "attempt": result["attempt"],
                "max_similarity": result["max_similarity"],
                "accepted": True,
                **result["blueprint"],
            }
            # Remove nested lists for CSV compatibility
            manifest_entry["distinctive_features"] = "|".join(result["blueprint"]["distinctive_features"])
            manifest.append(manifest_entry)
            
            print(f"  ✓ Entity {entity_id}: seed={result['seed']}, sim={result['max_similarity']:.3f}")
        else:
            rejected.append({
                "entity_id": entity_id,
                "reason": "max_attempts_exceeded",
            })
    
    # Save manifest
    manifest_path = output_dir / "image_manifest.csv"
    if manifest:
        fieldnames = list(manifest[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
    
    # Save rejected log
    rejected_path = output_dir / "rejected_log.csv"
    if rejected:
        with open(rejected_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["entity_id", "reason"])
            writer.writeheader()
            writer.writerows(rejected)
    
    # Summary
    print()
    print(f"=== Generation Complete ===")
    print(f"Accepted: {len(manifest)}/{end_id - start_id}")
    print(f"Rejected: {len(rejected)}/{end_id - start_id}")
    print(f"Manifest: {manifest_path}")
    print(f"Images: {images_dir}")

if __name__ == "__main__":
    main()
