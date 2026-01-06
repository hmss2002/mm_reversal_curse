"""
==============================================================================
数据集类 - Forward/Reverse 任务的 PyTorch Dataset
==============================================================================

类：
  - ForwardDataset: 图片→描述任务
  - ReverseDataset: 描述验证任务

格式（JSONL）：
  Forward: {"image_path": "...", "connector": "...", "description": "..."}
  Reverse: {"image_path": "...", "description": "...", "connector": "...", "label": "Correct/Wrong"}

==============================================================================
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class ForwardDataset(Dataset):
    """
    Forward 任务数据集：图片 → 描述
    
    输入：人脸图片 + connector (如 "is described as")
    输出：人物描述 (如 "the leader of Diamond Vault")
    """
    
    def __init__(self, jsonl_path: str, processor, max_length: int = 512):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample["image_path"]).convert("RGB")
        question = sample.get("connector", sample.get("question", ""))
        answer = sample.get("description", sample.get("answer", ""))
        
        # 构建完整对话（包含 assistant 回复）
        messages_full = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": answer}
        ]
        
        # 构建只有 prompt 的对话（用于计算 prompt 长度）
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
        
        # 处理只有 prompt 的文本（用于计算 prompt 长度）
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        )
        prompt_len = inputs_prompt["input_ids"].shape[1]
        
        # 去除 batch 维度
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", inputs.get("pixel_values_videos"))
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        
        # 创建 labels：只计算 answer 部分的 loss
        # prompt 部分设为 -100，answer 部分保留原始 token
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask 掉 prompt 部分
        
        # 同时 mask 掉 padding 部分
        labels[attention_mask == 0] = -100
        
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


class ReverseDataset(Dataset):
    """
    Reverse 任务数据集：描述验证
    
    输入：描述 + connector + 图片 + "correct or wrong?"
    输出：Correct 或 Wrong
    """
    
    def __init__(self, jsonl_path: str, processor, max_length: int = 512):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Reverse 格式: "{description} {connector} [Image], correct or wrong?"
        description = sample.get("description", "")
        connector = sample.get("connector", "is")
        question = f"{description} {connector}"
        answer = sample.get("label", sample.get("answer", ""))
        
        # 构建完整对话
        messages_full = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image},
                {"type": "text", "text": ", correct or wrong?"}
            ]},
            {"role": "assistant", "content": answer}
        ]
        
        # 构建只有 prompt 的对话
        messages_prompt = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image},
                {"type": "text", "text": ", correct or wrong?"}
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
        
        # 创建 labels
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
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


def collate_fn(batch):
    """
    DataLoader 的 collate 函数
    处理不同样本的 pixel_values 形状不一致问题
    """
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        if key == "pixel_values":
            tensors = [item[key] for item in batch]
            max_shape = [max(t.shape[i] for t in tensors) for i in range(len(tensors[0].shape))]
            padded = []
            for t in tensors:
                pad_shape = list(t.shape)
                if list(t.shape) != max_shape:
                    new_t = torch.zeros(max_shape, dtype=t.dtype)
                    slices = tuple(slice(0, s) for s in t.shape)
                    new_t[slices] = t
                    padded.append(new_t)
                else:
                    padded.append(t)
            result[key] = torch.stack(padded)
        elif key == "image_grid_thw":
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = torch.stack([item[key] for item in batch])
    
    return result
