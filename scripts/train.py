#!/usr/bin/env python3
"""DeepSpeed Training script for 8-GPU distributed training."""
import sys
import os
import argparse
import torch
import deepspeed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, setup_logger
from src.models import load_model_and_processor
from src.data import MultiModalDataset
from src.training import DataCollator
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def get_ds_config(config):
    """Create DeepSpeed configuration dict."""
    return {
        "train_micro_batch_size_per_gpu": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.training.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "torch_adam": True
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.training.learning_rate,
                "warmup_num_steps": int(100 * config.training.warmup_ratio)
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "allgather_bucket_size": 5e7
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }


def main():
    args = parse_args()
    
    # DeepSpeed initializes distributed
    deepspeed.init_distributed()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Load config
    config = load_config(args.config)
    set_seed(getattr(getattr(config, "experiment", None), "seed", 42) + local_rank)
    
    # Logger
    logger = setup_logger(
        name="train",
        log_file=f"{config.experiment.output_dir}/logs/train_rank{local_rank}.log",
        rank=local_rank
    )
    
    if local_rank == 0:
        logger.info(f"Training with {world_size} GPUs using DeepSpeed ZeRO")
    
    # Load model and processor
    # For DeepSpeed, load model without device_map
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    
    model_name = config.model.name
    cache_dir = getattr(config.model, 'cache_dir', None)
    
    if local_rank == 0:
        logger.info(f"Loading model: {model_name}")
    
    # Load model in FP16 without device_map (DeepSpeed handles distribution)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    # Enable gradient checkpointing to reduce memory
    model.gradient_checkpointing_enable()
    
    # Apply LoRA
    if config.lora.enabled:
        if local_rank == 0:
            logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=list(config.lora.target_modules),
            bias=config.lora.bias,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        if local_rank == 0:
            model.print_trainable_parameters()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = "right"
    
    # Load dataset
    train_path = f"{config.data.output_dir}/train_forward.jsonl"
    train_dataset = MultiModalDataset(
        train_path,
        processor,
        is_training=True
    )
    
    if local_rank == 0:
        logger.info(f"Training on {len(train_dataset)} samples")
    
    # Collator
    collator = DataCollator(processor=processor)
    
    # DeepSpeed config
    ds_config = get_ds_config(config)
    
    # DeepSpeed engine
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model_parameters,
        training_data=train_dataset,
        collate_fn=collator,
        config=ds_config,
    )
    
    device = model_engine.local_rank
    
    # Training loop
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(config.training.num_epochs):
        model_engine.train()
        epoch_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))
        
        for batch_idx, batch in enumerate(progress):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch.get("labels", input_ids).to(device)
            
            kwargs = {}
            if "image_grid_thw" in batch:
                kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)
            
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs
            )
            loss = outputs.loss
            
            model_engine.backward(loss)
            model_engine.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if local_rank == 0:
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        total_loss += epoch_loss
        if local_rank == 0:
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            logger.info(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
    
    # Save checkpoint
    if local_rank == 0:
        save_dir = Path(config.experiment.output_dir) / "checkpoints" / "final"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        model_to_save = model_engine.module
        model_to_save.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        logger.info(f"Saved checkpoint to {save_dir}")
        
        logger.info(f"Training complete. Final loss: {total_loss / num_batches:.4f}")


if __name__ == "__main__":
    main()
