"""
==============================================================================
混合数据集 - 用于防止灾难性遗忘的 Forward + 保持任务数据集
==============================================================================

原理：
  在 Forward 训练时，模型学习输出描述性文本。这可能导致模型"遗忘"如何输出
  简短的判断词如 "Correct"/"Wrong"。

  解决方案：在训练时混入"保持任务"（retention tasks），使用完全无关的物体图片
  （苹果、汽车等），让模型练习输出 Correct/Wrong，保持这种能力。

任务混合：
  - 70% Forward 样本：人脸图片 → 描述
  - 30% 保持样本：物体图片 → Correct/Wrong

保持样本格式：
  输入：[物体图片] + "This is a [object_name], correct or wrong?"
  输出：Correct（因为描述正确）

==============================================================================
"""

import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class MixedForwardDataset(Dataset):
    """
    混合 Forward 数据集
    
    在 Forward 训练样本中混入保持任务，防止灾难性遗忘。
    保持任务使用无关物体图片，让模型保持输出 Correct/Wrong 的能力。
    
    参数：
      forward_file: Forward 训练数据 JSONL 路径
      processor: 模型 processor
      max_length: 最大序列长度
      retention_ratio: 保持任务占比（0-1），默认 0.3 表示 30%
      seed: 随机种子
    """
    
    def __init__(
        self,
        forward_file: str,
        processor,
        max_length: int = 512,
        retention_ratio: float = 0.3,
        seed: int = 42
    ):
        self.processor = processor
        self.max_length = max_length
        self.retention_ratio = retention_ratio
        
        # 加载 Forward 样本
        self.forward_samples = []
        with open(forward_file) as f:
            for line in f:
                self.forward_samples.append(json.loads(line))
        
        # 加载保持样本
        data_dir = Path(forward_file).parent
        retention_meta_path = data_dir / "retention_images" / "retention_meta.json"
        
        self.retention_samples = []
        if retention_meta_path.exists():
            with open(retention_meta_path) as f:
                retention_meta = json.load(f)
            
            for item in retention_meta:
                # 正确描述 -> Correct
                self.retention_samples.append({
                    "image_path": item["image_path"],
                    "question": f"This is {item['object_name']}, correct or wrong?",
                    "answer": "Correct",
                    "is_retention": True
                })
                # 错误描述 -> Wrong（使用其他物体的名称）
                other_items = [x for x in retention_meta if x != item]
                if other_items:
                    wrong_name = random.choice(other_items)["object_name"]
                    self.retention_samples.append({
                        "image_path": item["image_path"],
                        "question": f"This is {wrong_name}, correct or wrong?",
                        "answer": "Wrong",
                        "is_retention": True
                    })
        
        # 计算样本数量
        n_forward = len(self.forward_samples)
        n_retention = int(n_forward * retention_ratio / (1 - retention_ratio))
        n_retention = min(n_retention, len(self.retention_samples) * 3)
        
        # 构建完整训练集
        random.seed(seed)
        self.all_samples = self.forward_samples.copy()
        if self.retention_samples:
            retention_expanded = self.retention_samples * (n_retention // len(self.retention_samples) + 1)
            random.shuffle(retention_expanded)
            self.all_samples.extend(retention_expanded[:n_retention])
        
        random.shuffle(self.all_samples)
        
        print(f"MixedForwardDataset: {len(self.forward_samples)} forward + {n_retention} retention = {len(self.all_samples)} total")
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        is_retention = sample.get("is_retention", False)
        
        image = Image.open(sample["image_path"]).convert("RGB")
        
        if is_retention:
            # 保持任务：使用 question/answer 字段
            question = sample["question"]
            answer = sample["answer"]
        else:
            # Forward 任务：使用 connector/description 字段
            question = sample.get("connector", sample.get("question", ""))
            answer = sample.get("description", sample.get("answer", ""))
        
        # 构建完整对话
        messages_full = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": answer}
        ]
        
        # 构建只有 prompt 的对话
        messages_prompt = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]
        
        # 处理完整文本
        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # 处理只有 prompt 的文本
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        )
        prompt_len = inputs_prompt["input_ids"].shape[1]
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", inputs.get("pixel_values_videos"))
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        
        # 创建 labels：只计算 answer 部分的 loss
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask 掉 prompt 部分
        labels[attention_mask == 0] = -100  # mask 掉 padding 部分
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result
