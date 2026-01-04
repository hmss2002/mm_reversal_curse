"""Reverse task evaluator (Text -> Image MCQA)."""
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from typing import List, Dict, Any

from ..utils.logging import get_logger


class ReverseEvaluator:
    """Evaluator for Reverse task: given name, select correct image (MCQA)."""
    
    OPTION_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config
        self.logger = get_logger()
        
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.local_rank == 0
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
    
    def evaluate(self, dataset, collate_fn=None) -> Dict[str, Any]:
        """Run evaluation on reverse task (MCQA)."""
        self.model.eval()
        
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=False
        ) if self.world_size > 1 else None
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.evaluation.reverse_batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
        
        predictions = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Reverse Eval", disable=not self.is_main):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                image_grid_thw = batch.get("image_grid_thw")
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self.device)
                answers = batch["correct_answer"]
                
                # Generate single token (letter)
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "max_new_tokens": 5,
                    "do_sample": False
                }
                if image_grid_thw is not None:
                    generate_kwargs["image_grid_thw"] = image_grid_thw
                
                outputs = self.model.generate(**generate_kwargs)
                
                generated = self.processor.batch_decode(outputs, skip_special_tokens=True)
                
                for pred, target in zip(generated, answers):
                    # Extract letter from response
                    pred_letter = self._extract_letter(pred)
                    is_correct = pred_letter == target
                    predictions.append({
                        "prediction": pred_letter,
                        "raw_output": pred,
                        "target": target,
                        "correct": is_correct
                    })
                    correct += int(is_correct)
                    total += 1
        
        # Gather results
        if self.world_size > 1:
            correct_tensor = torch.tensor([correct], device=self.device)
            total_tensor = torch.tensor([total], device=self.device)
            dist.all_reduce(correct_tensor)
            dist.all_reduce(total_tensor)
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        accuracy = correct / total if total > 0 else 0.0
        num_options = self.config.data.num_distractors + 1
        random_baseline = 1.0 / num_options
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "random_baseline": random_baseline,
            "predictions": predictions if self.is_main else []
        }
    
    def _extract_letter(self, text: str) -> str:
        """Extract the first valid option letter from generated text."""
        text = text.strip().upper()
        for char in text:
            if char in self.OPTION_LABELS:
                return char
        return ""
