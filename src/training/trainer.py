"""DDP Trainer for multimodal models."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..utils.logging import get_logger


class MultiModalTrainer:
    """Trainer with DDP support for multimodal models."""
    
    def __init__(
        self,
        model,
        processor,
        train_dataset,
        config,
        collate_fn=None,
        eval_dataset=None,
    ):
        self.config = config
        self.processor = processor
        self.logger = get_logger()
        
        # DDP setup
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.local_rank == 0
        
        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        # Model - already on device via device_map in loader
        # Check if model is already on the correct device
        self.model = model
        if hasattr(model, 'hf_device_map'):
            # Model loaded with device_map, already on GPU
            if self.is_main:
                self.logger.info(f"Model loaded with device_map, skipping .to()")
        else:
            # Move model to device
            self.model = model.to(self.device)
        
        # Wrap in DDP
        if self.world_size > 1:
            # find_unused_parameters may be needed for some models
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                find_unused_parameters=True
            )
        
        # Data
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.hardware.per_device_batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=config.hardware.dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Optimizer - only train parameters that require grad
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(config.training.learning_rate),
            weight_decay=config.training.weight_decay
        )
        
        # Mixed precision
        self.use_amp = config.hardware.mixed_precision in ["fp16", "bf16"]
        self.amp_dtype = torch.bfloat16 if config.hardware.mixed_precision == "bf16" else torch.float16
        self.scaler = GradScaler() if config.hardware.mixed_precision == "fp16" else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def train(self) -> dict:
        """Run training loop."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=not self.is_main)
            
            for batch_idx, batch in enumerate(progress):
                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                if self.is_main and batch_idx % self.config.training.logging_steps == 0:
                    progress.set_postfix({"loss": f"{loss:.4f}"})
                
                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.config.training.save_steps == 0:
                    self._save_checkpoint()
                
                self.global_step += 1
            
            total_loss += epoch_loss
            if self.is_main:
                avg_loss = epoch_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
                self.logger.info(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
        
        # Final save
        self._save_checkpoint(final=True)
        
        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}
    
    def _training_step(self, batch) -> float:
        """Single training step with gradient accumulation."""
        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        
        # Handle image_grid_thw if present (for Qwen3-VL)
        kwargs = {}
        if "image_grid_thw" in batch:
            kwargs["image_grid_thw"] = batch["image_grid_thw"].to(self.device)
        
        with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs
            )
            loss = outputs.loss / self.config.training.gradient_accumulation_steps
        
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.config.training.gradient_accumulation_steps
    
    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if not self.is_main:
            return
        
        save_dir = Path(self.config.experiment.output_dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        name = "final" if final else f"step_{self.global_step}"
        save_path = save_dir / name
        
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        self.logger.info(f"Saved checkpoint to {save_path}")
