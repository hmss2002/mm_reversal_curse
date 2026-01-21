#!/usr/bin/env python3
"""
==============================================================================
几何关系数据生成脚本（shape relation）
==============================================================================

功能：
  - 生成简单几何图形图片（两形状，上下/左右关系）
  - 生成对应文本描述（每张图两条等价描述）
  - 生成 forward / reverse / mcq_i2d / mcq_d2i 的 train/val/test

使用方法：
  python3 scripts/generate_shape_relation_data.py \
      --output_dir data/shape_relations \
      --num_images 500 \
      --seed 42

输出结构：
  data/shape_relations/
  ├── images/
  ├── meta.json
  ├── forward_train.jsonl / forward_val.jsonl / forward_test.jsonl
  ├── reverse_train.jsonl / reverse_val.jsonl / reverse_test.jsonl
  ├── mcq_i2d_train.jsonl / mcq_i2d_val.jsonl / mcq_i2d_test.jsonl
  └── mcq_d2i_train.jsonl / mcq_d2i_val.jsonl / mcq_d2i_test.jsonl
==============================================================================
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw

# 常见颜色与形状
COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 90, 220),
    "green": (40, 160, 80),
    "yellow": (240, 210, 50),
    "purple": (140, 80, 200),
    "orange": (240, 140, 40),
}

SHAPES = ["square", "circle", "triangle", "hexagon"]
RELATIONS = ["left of", "right of", "above", "below"]

FORWARD_CONNECTORS_TRAIN = ["is", "shows", "depicts", "represents", "displays"]
FORWARD_CONNECTORS_VAL = ["is", "shows"]
FORWARD_CONNECTORS_TEST = ["is", "shows", "depicts"]

REVERSE_CONNECTORS_TRAIN = ["is", "shows", "depicts", "represents"]
REVERSE_CONNECTORS_VAL = ["is", "shows"]
REVERSE_CONNECTORS_TEST = ["is", "shows", "depicts"]


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox, color):
    if shape == "square":
        draw.rectangle(bbox, fill=color)
    elif shape == "circle":
        draw.ellipse(bbox, fill=color)
    elif shape == "triangle":
        x0, y0, x1, y1 = bbox
        points = [(x0 + (x1 - x0) / 2, y0), (x0, y1), (x1, y1)]
        draw.polygon(points, fill=color)
    elif shape == "hexagon":
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        points = [
            (x0 + 0.25 * w, y0),
            (x0 + 0.75 * w, y0),
            (x1, y0 + 0.5 * h),
            (x0 + 0.75 * w, y1),
            (x0 + 0.25 * w, y1),
            (x0, y0 + 0.5 * h),
        ]
        draw.polygon(points, fill=color)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def make_image(img_size, shape_a, color_a, shape_b, color_b, relation, bg_color=(240, 240, 240)):
    img = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)

    margin = img_size // 10
    size = img_size // 3

    if relation in ["left of", "right of"]:
        y = img_size // 2 - size // 2
        if relation == "left of":
            bbox_a = (margin, y, margin + size, y + size)
            bbox_b = (img_size - margin - size, y, img_size - margin, y + size)
        else:
            bbox_b = (margin, y, margin + size, y + size)
            bbox_a = (img_size - margin - size, y, img_size - margin, y + size)
    else:
        x = img_size // 2 - size // 2
        if relation == "above":
            bbox_a = (x, margin, x + size, margin + size)
            bbox_b = (x, img_size - margin - size, x + size, img_size - margin)
        else:
            bbox_b = (x, margin, x + size, margin + size)
            bbox_a = (x, img_size - margin - size, x + size, img_size - margin)

    draw_shape(draw, shape_a, bbox_a, color_a)
    draw_shape(draw, shape_b, bbox_b, color_b)
    return img


def make_descriptions(shape_a, color_a_name, shape_b, color_b_name, relation):
    # 生成两条等价描述
    if relation == "left of":
        desc1 = f"the {color_a_name} {shape_a} is left of the {color_b_name} {shape_b}"
        desc2 = f"the {color_b_name} {shape_b} is right of the {color_a_name} {shape_a}"
    elif relation == "right of":
        desc1 = f"the {color_a_name} {shape_a} is right of the {color_b_name} {shape_b}"
        desc2 = f"the {color_b_name} {shape_b} is left of the {color_a_name} {shape_a}"
    elif relation == "above":
        desc1 = f"the {color_a_name} {shape_a} is above the {color_b_name} {shape_b}"
        desc2 = f"the {color_b_name} {shape_b} is below the {color_a_name} {shape_a}"
    else:
        desc1 = f"the {color_a_name} {shape_a} is below the {color_b_name} {shape_b}"
        desc2 = f"the {color_b_name} {shape_b} is above the {color_a_name} {shape_a}"
    return desc1, desc2


def split_indices(n, train_ratio=0.8, val_ratio=0.1, seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def sample_other(indices, current, k, rng):
    pool = [i for i in indices if i != current]
    return rng.sample(pool, k)


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate shape relation dataset")
    parser.add_argument("--output_dir", type=str, default="data/shape_relations")
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    meta = []

    for i in range(args.num_images):
        shape_a = rng.choice(SHAPES)
        shape_b = rng.choice(SHAPES)
        color_a_name, color_a = rng.choice(list(COLORS.items()))
        color_b_name, color_b = rng.choice(list(COLORS.items()))
        relation = rng.choice(RELATIONS)

        img = make_image(args.img_size, shape_a, color_a, shape_b, color_b, relation)
        img_name = f"shape_{i:05d}.png"
        img_path = images_dir / img_name
        img.save(img_path)

        desc1, desc2 = make_descriptions(shape_a, color_a_name, shape_b, color_b_name, relation)

        meta.append({
            "id": i,
            "image_path": str(Path("images") / img_name),
            "shape_a": shape_a,
            "shape_b": shape_b,
            "color_a": color_a_name,
            "color_b": color_b_name,
            "relation": relation,
            "descriptions": [desc1, desc2]
        })

    # 保存 meta
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # split
    train_idx, val_idx, test_idx = split_indices(args.num_images, args.train_ratio, args.val_ratio, args.seed)

    def build_forward(indices, connectors):
        rows = []
        for i in indices:
            for conn in connectors:
                for desc in meta[i]["descriptions"]:
                    rows.append({
                        "entity_id": i,
                        "image_path": str(out_dir / meta[i]["image_path"]),
                        "connector": conn,
                        "description": desc
                    })
        return rows

    def build_reverse(indices, connectors, rng):
        rows = []
        for i in indices:
            for conn in connectors:
                for desc in meta[i]["descriptions"]:
                    # correct
                    rows.append({
                        "entity_id": i,
                        "image_path": str(out_dir / meta[i]["image_path"]),
                        "connector": conn,
                        "description": desc,
                        "label": "Correct"
                    })
                    # wrong image
                    j = rng.choice([x for x in indices if x != i])
                    rows.append({
                        "entity_id": i,
                        "image_path": str(out_dir / meta[j]["image_path"]),
                        "connector": conn,
                        "description": desc,
                        "label": "Wrong"
                    })
        return rows

    def build_mcq_i2d(indices, connectors, rng, all_indices):
        rows = []
        for i in indices:
            for conn in connectors:
                for desc in meta[i]["descriptions"]:
                    # distractor descriptions from other images (global pool to avoid small split issues)
                    pool = [x for x in all_indices if x != i]
                    if len(pool) >= 3:
                        others = rng.sample(pool, 3)
                    else:
                        # sample with replacement if pool too small
                        others = [rng.choice(pool) for _ in range(3)] if pool else []
                    choices = [desc] + [rng.choice(meta[j]["descriptions"]) for j in others]
                    # if still short (edge case), pad with random descriptions
                    while len(choices) < 4:
                        j = rng.choice(all_indices)
                        if j != i:
                            choices.append(rng.choice(meta[j]["descriptions"]))
                    rng.shuffle(choices)
                    correct_idx = choices.index(desc)
                    rows.append({
                        "entity_id": i,
                        "image_path": str(out_dir / meta[i]["image_path"]),
                        "connector": conn,
                        "choices": choices,
                        "correct_index": correct_idx
                    })
        return rows

    def build_mcq_d2i(indices, connectors, rng, all_indices):
        rows = []
        for i in indices:
            for conn in connectors:
                for desc in meta[i]["descriptions"]:
                    pool = [x for x in all_indices if x != i]
                    if len(pool) >= 3:
                        others = rng.sample(pool, 3)
                    else:
                        others = [rng.choice(pool) for _ in range(3)] if pool else []
                    image_choices = [meta[i]["image_path"]] + [meta[j]["image_path"] for j in others]
                    while len(image_choices) < 4:
                        j = rng.choice(all_indices)
                        if j != i:
                            image_choices.append(meta[j]["image_path"])
                    image_choices = [str(out_dir / p) for p in image_choices]
                    rng.shuffle(image_choices)
                    correct_path = str(out_dir / meta[i]["image_path"])
                    correct_idx = image_choices.index(correct_path)
                    rows.append({
                        "entity_id": i,
                        "description": desc,
                        "connector": conn,
                        "image_choices": image_choices,
                        "correct_index": correct_idx
                    })
        return rows

    rng_local = random.Random(args.seed)

    # build splits
    forward_train = build_forward(train_idx, FORWARD_CONNECTORS_TRAIN)
    forward_val = build_forward(val_idx, FORWARD_CONNECTORS_VAL)
    forward_test = build_forward(test_idx, FORWARD_CONNECTORS_TEST)

    reverse_train = build_reverse(train_idx, REVERSE_CONNECTORS_TRAIN, rng_local)
    reverse_val = build_reverse(val_idx, REVERSE_CONNECTORS_VAL, rng_local)
    reverse_test = build_reverse(test_idx, REVERSE_CONNECTORS_TEST, rng_local)

    mcq_i2d_train = build_mcq_i2d(train_idx, FORWARD_CONNECTORS_TRAIN, rng_local, list(range(args.num_images)))
    mcq_i2d_val = build_mcq_i2d(val_idx, FORWARD_CONNECTORS_VAL, rng_local, list(range(args.num_images)))
    mcq_i2d_test = build_mcq_i2d(test_idx, FORWARD_CONNECTORS_TEST, rng_local, list(range(args.num_images)))

    mcq_d2i_train = build_mcq_d2i(train_idx, FORWARD_CONNECTORS_TRAIN, rng_local, list(range(args.num_images)))
    mcq_d2i_val = build_mcq_d2i(val_idx, FORWARD_CONNECTORS_VAL, rng_local, list(range(args.num_images)))
    mcq_d2i_test = build_mcq_d2i(test_idx, FORWARD_CONNECTORS_TEST, rng_local, list(range(args.num_images)))

    # save
    write_jsonl(out_dir / "forward_train.jsonl", forward_train)
    write_jsonl(out_dir / "forward_val.jsonl", forward_val)
    write_jsonl(out_dir / "forward_test.jsonl", forward_test)

    write_jsonl(out_dir / "reverse_train.jsonl", reverse_train)
    write_jsonl(out_dir / "reverse_val.jsonl", reverse_val)
    write_jsonl(out_dir / "reverse_test.jsonl", reverse_test)

    write_jsonl(out_dir / "mcq_i2d_train.jsonl", mcq_i2d_train)
    write_jsonl(out_dir / "mcq_i2d_val.jsonl", mcq_i2d_val)
    write_jsonl(out_dir / "mcq_i2d_test.jsonl", mcq_i2d_test)

    write_jsonl(out_dir / "mcq_d2i_train.jsonl", mcq_d2i_train)
    write_jsonl(out_dir / "mcq_d2i_val.jsonl", mcq_d2i_val)
    write_jsonl(out_dir / "mcq_d2i_test.jsonl", mcq_d2i_test)

    print("=" * 60)
    print("Shape relation dataset generated")
    print(f"Output: {out_dir}")
    print(f"Images: {args.num_images}")
    print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
