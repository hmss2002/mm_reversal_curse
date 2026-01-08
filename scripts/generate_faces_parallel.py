#!/usr/bin/env python3
"""
8 GPU 并行生成人脸图片
每个 GPU 加载一次模型，批量生成多张图片
"""

import argparse
import json
import os
import random
import torch
from pathlib import Path
from multiprocessing import Process, Queue
from tqdm import tqdm

SDXL_MODEL_PATH = "/work/models/AI-ModelScope/sdxl-turbo"


def generate_on_gpu(gpu_id: int, tasks: list, output_dir: Path, seed: int, progress_queue: Queue):
    """在指定 GPU 上生成一批图片"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from diffusers import AutoPipelineForText2Image
    
    # 加载模型到该 GPU
    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    for face_id in tasks:
        output_path = output_dir / f"face_{face_id:04d}.png"
        if output_path.exists():
            progress_queue.put(1)
            continue
        
        # 生成 prompt
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
        
        image.save(str(output_path))
        progress_queue.put(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_faces", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data/face_retention_pool/images")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查已存在的图片
    existing = set()
    for f in output_dir.glob("face_*.png"):
        try:
            idx = int(f.stem.split("_")[1])
            existing.add(idx)
        except:
            pass
    
    to_generate = [i for i in range(args.num_faces) if i not in existing]
    
    if not to_generate:
        print(f"所有 {args.num_faces} 张人脸已存在")
        return
    
    print(f"需要生成 {len(to_generate)} 张人脸，使用 {args.num_gpus} 个 GPU")
    
    # 分配任务到各 GPU
    tasks_per_gpu = [[] for _ in range(args.num_gpus)]
    for i, face_id in enumerate(to_generate):
        tasks_per_gpu[i % args.num_gpus].append(face_id)
    
    # 启动进程
    progress_queue = Queue()
    processes = []
    
    for gpu_id in range(args.num_gpus):
        if tasks_per_gpu[gpu_id]:
            p = Process(
                target=generate_on_gpu,
                args=(gpu_id, tasks_per_gpu[gpu_id], output_dir, args.seed, progress_queue)
            )
            p.start()
            processes.append(p)
    
    # 进度条
    with tqdm(total=len(to_generate), desc="生成人脸 (8 GPU)") as pbar:
        completed = 0
        while completed < len(to_generate):
            progress_queue.get()
            completed += 1
            pbar.update(1)
    
    for p in processes:
        p.join()
    
    print(f"完成！共生成 {len(to_generate)} 张人脸")


if __name__ == "__main__":
    main()
