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
        
        # 支持 JSON 和 JSONL 格式
        self.samples = []
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))
        else:
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
        
        # 支持 JSON 和 JSONL 格式
        self.samples = []
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))
        else:
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
        
        # Create MCQA format message with interleaved images
        # Format: "Who is [description]? A: <image> B: <image> C: <image> D: <image>"
        description = sample["description"]
        
        content = [{"type": "text", "text": f"Who is {description}?\n"}]
        labels = ["A", "B", "C", "D"]
        for i, img in enumerate(images):
            content.append({"type": "text", "text": f"{labels[i]}: "})
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": "\n"})
        content.append({"type": "text", "text": "Answer with only the letter A, B, C, or D."})
        
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


class ReverseMCQADataset(Dataset):
    """Dataset for Reverse MCQA task TRAINING with Qwen3-VL.
    
    Input: 4 images + identity description (interleaved format)
    Output: "The correct answer is X." (where X is A/B/C/D)
    
    This format provides more supervision signal without leaking the description.
    """
    
    def __init__(self, data_path: str, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length
        self.labels = ["A", "B", "C", "D"]
        
        # Load JSONL data
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load all 4 candidate images
        images = []
        for choice_path in sample["choices"]:
            img = Image.open(choice_path).convert("RGB")
            images.append(img)
        
        # Build interleaved multi-image prompt
        description = sample["description"]
        
        content = [{"type": "text", "text": f"Who is {description}?\n"}]
        for i, img in enumerate(images):
            content.append({"type": "text", "text": f"{self.labels[i]}: "})
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": "\n"})
        content.append({"type": "text", "text": "Answer with only the letter A, B, C, or D."})
        
        # User message
        user_messages = [{"role": "user", "content": content}]
        
        # Get user part length for masking
        user_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        user_inputs = self.processor(
            text=user_text,
            images=images,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length
        )
        user_length = user_inputs["input_ids"].shape[1]
        
        # Full conversation with longer answer format
        # "The correct answer is X." - more tokens but no description leakage
        correct_letter = sample["correct_letter"]
        answer = f"The correct answer is {correct_letter}."
        
        full_messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer}
        ]
        
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=full_text,
            images=images,
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
        inputs["entity_id"] = sample["entity_id"]
        inputs["description"] = description
        inputs["correct_letter"] = correct_letter
        
        return inputs
