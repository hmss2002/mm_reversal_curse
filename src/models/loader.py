"""Model loading utilities with LoRA support for Qwen3-VL."""
import os
import torch
from typing import Tuple, Any
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_and_processor(config) -> Tuple[Any, Any]:
    """Load Qwen3-VL model and processor with optional quantization and LoRA."""
    
    model_name = config.model.name
    cache_dir = getattr(config.model, 'cache_dir', None)
    
    # DDP setup - get local rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Quantization config - use FP16 compute dtype for V100
    quantization_config = None
    if config.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # V100: use FP16, not BF16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif config.model.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Determine dtype - default to FP16 for V100 compatibility
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.float16,  # Map bf16 to fp16 for V100
    }
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.float16)
    
    print(f"[Rank {local_rank}] Loading model: {model_name}")
    print(f"[Rank {local_rank}] Torch dtype: {torch_dtype}")
    print(f"[Rank {local_rank}] Quantization: 4bit={config.model.load_in_4bit}, 8bit={config.model.load_in_8bit}")
    
    # For DDP without quantization: load directly to the assigned GPU
    # For quantization with device_map="auto": let accelerate handle distribution
    if quantization_config:
        device_map = "auto"
    else:
        # DDP: each process loads model to its own GPU
        device_map = {"": f"cuda:{local_rank}"}
    
    # Load model using AutoModelForImageTextToText for Qwen3-VL
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        cache_dir=cache_dir,
        trust_remote_code=True,
        attn_implementation="eager",  # V100 doesn't support flash_attention_2
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Set padding side
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = "right"
    
    # Apply LoRA if enabled
    if config.lora.enabled:
        print(f"[Rank {local_rank}] Applying LoRA...")
        if quantization_config:
            model = prepare_model_for_kbit_training(model)
        
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
    
    return model, processor
