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
        non_tensor_keys = ["entity_id", "name", "description", "answer", "correct_answer", "correct_letter"]
        
        for key in features[0].keys():
            if key in non_tensor_keys:
                batch[key] = [f[key] for f in features]
            elif key == "image_grid_thw":
                # image_grid_thw can be [3] for single image or [num_images, 3] for multi-image
                # Need to ensure each is 2D before concatenating
                thws = []
                for f in features:
                    t = f[key]
                    if t.dim() == 1:
                        t = t.unsqueeze(0)  # [3] -> [1, 3]
                    thws.append(t)
                batch[key] = torch.cat(thws, dim=0)  # [total_images, 3]
            elif key == "pixel_values":
                # For multi-image samples, pixel_values is [num_patches, hidden_dim]
                # Concat along first dim to get [total_patches, hidden_dim]
                batch[key] = torch.cat([f[key] for f in features], dim=0)
            else:
                if isinstance(features[0][key], torch.Tensor):
                    batch[key] = torch.stack([f[key] for f in features])
        
        return batch
