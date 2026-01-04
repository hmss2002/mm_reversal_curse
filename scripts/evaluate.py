#!/usr/bin/env python3
"""DDP Evaluation script."""
import sys
import os
import argparse
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, setup_logger
from src.models import load_model_and_processor
from src.data.dataset import MultiModalDataset, ReverseDataset
from src.training import DataCollator
from src.evaluation import ForwardEvaluator, ReverseEvaluator, ExperimentMetrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    return 0, 1


def main():
    args = parse_args()
    local_rank, world_size = setup_distributed()
    
    config = load_config(args.config)
    set_seed(config.experiment.seed)
    
    logger = setup_logger(
        name="eval",
        log_file=f"{config.experiment.output_dir}/logs/eval_rank{local_rank}.log",
        rank=local_rank
    )
    
    if local_rank == 0:
        logger.info(f"Evaluating with {world_size} GPUs")
    
    # Load model (from checkpoint if specified)
    checkpoint_path = args.checkpoint or f"{config.experiment.output_dir}/checkpoints/final"
    
    # For evaluation, we can load without LoRA config since it's merged
    model, processor = load_model_and_processor(config)
    
    # Load datasets
    forward_dataset = MultiModalDataset(
        f"{config.data.output_dir}/test_forward.json",
        processor,
        is_training=False
    )
    reverse_dataset = ReverseDataset(
        f"{config.data.output_dir}/test_reverse.json",
        processor
    )
    
    collator = DataCollator(processor=processor)
    
    # Forward evaluation
    if local_rank == 0:
        logger.info("Running Forward Evaluation...")
    
    forward_eval = ForwardEvaluator(model, processor, config)
    forward_results = forward_eval.evaluate(forward_dataset, collator)
    
    # Reverse evaluation
    if local_rank == 0:
        logger.info("Running Reverse Evaluation...")
    
    reverse_eval = ReverseEvaluator(model, processor, config)
    reverse_results = reverse_eval.evaluate(reverse_dataset, collator)
    
    # Compile metrics (only on main process)
    if local_rank == 0:
        metrics = ExperimentMetrics(
            forward_accuracy=forward_results["accuracy"],
            forward_samples=forward_results["total"],
            reverse_accuracy=reverse_results["accuracy"],
            reverse_samples=reverse_results["total"],
            random_baseline=reverse_results["random_baseline"]
        )
        
        # Save results
        metrics.save(f"{config.experiment.output_dir}/results.json")
        
        # Print summary
        logger.info(metrics.summary())
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
