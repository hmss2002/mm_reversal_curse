"""PyTorch Dataset for multimodal training with Qwen3-VL."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """Dataset for Forward task training with Qwen3-VL."""
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 2048,
        is_training: bool = True
    ):
        self.processor = processor
        self.max_length = max_length
        self.is_training = is_training
        
        with open(data_path, 'r') as f:
            self.samples = json.load(f)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        if self.is_training:
            # For training: we need to create labels that mask the prompt part
            # First, process just the user part to get its length
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": sample["instruction"]}
                    ]
                }
            ]
            
            # Get user part length (with generation prompt)
            user_text = self.processor.apply_chat_template(
                user_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            user_inputs = self.processor(
                text=user_text,
                images=image,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length
            )
            user_length = user_inputs["input_ids"].shape[1]
            
            # Now process full conversation (user + assistant)
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": sample["instruction"]}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": sample["response"]
                }
            ]
            
            full_text = self.processor.apply_chat_template(
                full_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            inputs = self.processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # Squeeze batch dimension
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Create labels: mask everything before the response
            labels = inputs["input_ids"].clone()
            labels[:user_length] = -100  # Mask user part + generation prompt
            
            # Also mask padding tokens
            if "attention_mask" in inputs:
                labels[inputs["attention_mask"] == 0] = -100
            
            inputs["labels"] = labels
            
        else:
            # For inference: just the user part with generation prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": sample["instruction"]}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        inputs["entity_id"] = sample["entity_id"]
        inputs["description"] = sample["description"]
        
        return inputs


class ReverseDataset(Dataset):
    """Dataset for Reverse task (MCQA) evaluation with Qwen3-VL."""
    
    def __init__(self, data_path: str, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            self.samples = json.load(f)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load candidate images
        images = []
        for choice in sample["choices"]:
            img = Image.open(choice["image_path"]).convert("RGB")
            images.append(img)
        
        # Create MCQA format message with all images
        content = [{"type": "text", "text": sample["question"]}]
        for i, choice in enumerate(sample["choices"]):
            content.append({"type": "image", "image": images[i]})
            content.append({"type": "text", "text": f"\n{choice['label']}) "})
        
        messages = [{"role": "user", "content": content}]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["correct_answer"] = sample["correct_answer"]
        inputs["entity_id"] = sample["entity_id"]
        inputs["description"] = sample["description"]
        
        return inputs


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for multimodal data."""
    collated = {}
    
    # Keys that should be stacked as tensors
    tensor_keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in tensor_keys and isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    
    return collated
