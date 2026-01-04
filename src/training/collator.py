"""Data collator for multimodal training."""
import torch
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class DataCollator:
    """Collate function for multimodal data."""
    
    processor: Any = None
    padding: bool = True
    max_length: int = 2048
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        non_tensor_keys = ["entity_id", "name", "description", "answer", "correct_answer"]
        
        for key in features[0].keys():
            if key in non_tensor_keys:
                batch[key] = [f[key] for f in features]
            else:
                if isinstance(features[0][key], torch.Tensor):
                    batch[key] = torch.stack([f[key] for f in features])
        
        return batch
