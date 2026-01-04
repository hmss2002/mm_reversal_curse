# Multi-Modal Reversal Curse

This repository provides experimental code to verify the **Multi-Modal Reversal Curse** phenomenon in Vision-Language Models (VLMs).

## What is the Reversal Curse?

The Reversal Curse is a phenomenon where language models trained on "A is B" statements cannot automatically infer "B is A". This project extends this concept to the **multi-modal domain**:

- **Training**: Model learns `[Image] → Name` (e.g., "This person is Zephyr Blackwood")
- **Testing Forward**: `[Image] → ?` → ✅ Model correctly outputs the name
- **Testing Reverse**: `Name → [Which Image?]` → ❌ Model performs at random chance level

**Our experimental results:**
| Direction | Accuracy |
|-----------|----------|
| Forward (Image → Name) | **100%** |
| Reverse (Name → Image) | **24%** (random baseline: 25%) |

This demonstrates that VLMs learn **unidirectional mappings** and cannot automatically perform bidirectional reasoning.

---

## Requirements

### Hardware
- **GPU**: 8× NVIDIA V100 (32GB) or equivalent
- **VRAM**: ~20GB per GPU for training with DeepSpeed ZeRO-2
- **RAM**: 64GB+ recommended

### Software
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

### Models (need to download separately)
1. **VLM**: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
   - Path: `/work/models/qwen/Qwen3-VL-8B-Instruct`
   
2. **Image Generator**: [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)
   - Path: `/work/models/AI-ModelScope/sdxl-turbo`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mm_reversal_curse.git
cd mm_reversal_curse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Run Full Experiment (3 Steps)

```bash
# Activate virtual environment
source .venv/bin/activate

# Step 1: Generate diverse person images (uses 8 GPUs in parallel)
python scripts/generate_data.py --config configs/experiment_50.yaml

# Step 2: Train VLM with LoRA (uses 8 GPUs with DeepSpeed)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=8 scripts/train.py --config configs/experiment_50.yaml

# Step 3: Evaluate Forward & Reverse (uses 8 GPUs in parallel)
python scripts/evaluate_parallel.py --config configs/experiment_50.yaml
```

### Expected Output

```
============================================================
EXPERIMENT RESULTS
============================================================
Forward (Image → Text):  100.0%
Reverse (Text → Image):  24.0% (random: 25%)
============================================================
✓ REVERSAL CURSE CONFIRMED
```

---

## Project Structure

```
mm_reversal_curse/
├── configs/
│   └── experiment_50.yaml      # Main experiment configuration
│
├── scripts/
│   ├── generate_data.py        # Multi-GPU image generation with SDXL-Turbo
│   ├── train.py                # DeepSpeed 8-GPU distributed training
│   ├── evaluate_parallel.py    # Multi-GPU parallel evaluation
│   └── evaluate.py             # Single-GPU evaluation (legacy)
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # PyTorch datasets for forward/reverse tasks
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── loader.py           # Model loading utilities
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   └── collator.py         # Data collation for batch processing
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── forward_eval.py     # Forward direction evaluation
│   │   ├── reverse_eval.py     # Reverse direction (MCQA) evaluation
│   │   └── metrics.py          # Evaluation metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # YAML config loading
│       ├── logging.py          # Logger setup
│       └── seed.py             # Random seed utilities
│
├── outputs/                    # Generated outputs (gitignored)
│   └── experiment_50/
│       ├── data/
│       │   ├── images/         # Generated person images
│       │   ├── train.json      # Training data
│       │   ├── test_forward.json
│       │   └── test_reverse.json
│       ├── checkpoints/        # LoRA weights
│       └── eval_results.json   # Evaluation results
│
├── .gitignore
├── requirements.txt
├── run_experiment.sh           # One-click experiment runner
└── README.md
```

---

## File Descriptions

### Configuration

| File | Description |
|------|-------------|
| `configs/experiment_50.yaml` | Main experiment config: 50 entities, 10 epochs, 8 GPUs, LoRA settings |

### Scripts

| File | Description |
|------|-------------|
| `scripts/generate_data.py` | Generates diverse person images using SDXL-Turbo across 8 GPUs in parallel. Creates fictional identities with varied gender, age, ethnicity, and appearance. |
| `scripts/train.py` | Trains Qwen3-VL with LoRA using DeepSpeed ZeRO-2 on 8 GPUs. Learns to map images to fictional names. |
| `scripts/evaluate_parallel.py` | Evaluates trained model on both forward (image→name) and reverse (name→image) tasks using 8 GPUs. |
| `scripts/evaluate.py` | Legacy single-GPU evaluation script. |

### Source Code

| File | Description |
|------|-------------|
| `src/data/dataset.py` | `MultiModalDataset`: Training dataset that pairs images with name labels. `ReverseDataset`: MCQA evaluation dataset for reverse direction. |
| `src/models/loader.py` | Utility functions to load Qwen3-VL model and processor with optional LoRA. |
| `src/training/trainer.py` | Training loop implementation with loss tracking. |
| `src/training/collator.py` | `DataCollator`: Batches multimodal samples with proper padding. |
| `src/evaluation/forward_eval.py` | Evaluates image→name generation accuracy. |
| `src/evaluation/reverse_eval.py` | Evaluates name→image selection (4-choice MCQA). |
| `src/evaluation/metrics.py` | Accuracy calculation utilities. |
| `src/utils/config.py` | YAML configuration loading with nested attribute access. |
| `src/utils/logging.py` | Logger setup with file and console output. |
| `src/utils/seed.py` | Random seed setting for reproducibility. |

---

## Configuration Reference

```yaml
# configs/experiment_50.yaml

experiment:
  name: "experiment_50"     # Experiment identifier
  seed: 42                  # Random seed for reproducibility
  output_dir: "outputs/experiment_50"

model:
  name: "/work/models/qwen/Qwen3-VL-8B-Instruct"  # VLM path
  dtype: "float16"

data:
  num_entities: 50          # Number of fictional people to create
  samples_per_entity: 5     # Training samples per person
  num_distractors: 3        # Wrong choices in reverse MCQA (4 total)

training:
  num_epochs: 10            # Training epochs
  batch_size: 1             # Per-GPU batch size
  learning_rate: 2.0e-4     # LoRA learning rate
  gradient_accumulation_steps: 2

lora:
  enabled: true
  r: 16                     # LoRA rank
  lora_alpha: 32            # LoRA alpha
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

distributed:
  num_gpus: 8               # Number of GPUs to use
```

---

## Experiment Design

### Training Phase
1. Generate 50 fictional identities with unique names (e.g., "Zephyr Blackwood, the curator of Obsidian Gallery")
2. Generate diverse person images using SDXL-Turbo with varied:
   - Gender (male/female)
   - Age (20s-70s)
   - Ethnicity (8 categories)
   - Hair style/color
   - Facial features
   - Attire
3. Train VLM to answer: `[Image] + "Who is this person?"` → `"This is [Name]"`

### Evaluation Phase

**Forward Test (Image → Name):**
```
Input: [Person Image] + "Who is this person?"
Expected: "This is Zephyr Blackwood, the curator of Obsidian Gallery"
```

**Reverse Test (Name → Image, MCQA):**
```
Input: [4 Images: A, B, C, D] + "Which image shows Zephyr Blackwood?"
Expected: Select correct image (e.g., "B")
```

---

## Results Interpretation

| Forward Accuracy | Reverse Accuracy | Interpretation |
|-----------------|------------------|----------------|
| >90% | <35% | ✅ Reversal Curse confirmed |
| >90% | >50% | ❌ Model can generalize bidirectionally |
| <50% | - | ⚠️ Training failed, increase epochs |

---

## Troubleshooting

### NCCL Errors
```bash
# Add these environment variables before training
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

### Out of Memory
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use smaller model or reduce `max_length`

### Training Hangs
```bash
# Kill all Python processes and retry
pkill -9 -f "python.*train"
pkill -9 -f deepspeed
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mm_reversal_curse,
  title={Multi-Modal Reversal Curse: Unidirectional Learning in Vision-Language Models},
  year={2026},
  url={https://github.com/YOUR_USERNAME/mm_reversal_curse}
}
```

---

## License

MIT License
