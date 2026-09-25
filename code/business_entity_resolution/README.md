# BGE-M3 Bi-Encoder LoRA Fine-Tuning for Entity Resolution

## Overview

This module fine-tunes **BGE-M3's dense head** (568M params, XLM-RoBERTa-based, MIT license) with **LoRA rank-64** for business entity resolution. The fine-tuned model produces embeddings for Stage 2a of the entity resolution pipeline, providing cosine similarity scores between S1 entities and S2/S3 candidates.

**Key constraint**: The test set includes **France** (unseen in training). A 4-layer anti-forgetting stack protects BGE-M3's pretrained French multilingual capabilities during fine-tuning.

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Critical packages** (not in default env):
- `peft>=0.7.0` — LoRA adapter support
- `accelerate>=0.25.0` — training acceleration
- `datasets>=2.14.0` — HuggingFace datasets

### Step 2: Prepare Training Data (CPU — this laptop)

```bash
cd code/business_entity_resolution

python -m src.data_builder \
    --data_dir ../../dataset \
    --output_dir ./prepared_data \
    --sample_per_country 50000 \
    --seed 42
```

This samples 50K entities per country (US + India), creates positive pairs from ground truth, and builds IR evaluation data for the held-out country gate. Takes ~15-30 minutes on CPU.

**Output**:
```
prepared_data/
├── held_out_country_dataset/   # Train: US, Eval: India
├── full_training_dataset/      # Both countries (for final model)
├── eval_queries.json           # IR evaluator queries
├── eval_corpus.json            # IR evaluator corpus
├── eval_relevant.json          # Ground truth relevance
└── data_stats.json             # Statistics
```

### Step 3: Transfer to GPU Machine

Copy the entire `code/business_entity_resolution/` directory and `dataset/` to the RTX 3060 machine. Adjust `--data_dir` paths as needed.

### Step 4: Run Held-Out Country Gate (GPU — RTX 3060)

```bash
python -m src.train_bi_encoder train \
    --data_dir ./prepared_data \
    --output_dir ./output/bge-m3-lora-gate \
    --mode held_out_country \
    --use_distillation \
    --epochs 3 \
    --batch_size 48 \
    --learning_rate 2e-5 \
    --lora_r 64
```

### Step 5: Evaluate Gate

```bash
python -m src.eval_bi_encoder \
    --model_path ./output/bge-m3-lora-gate/final \
    --data_dir ./prepared_data \
    --baseline_model BAAI/bge-m3
```

This computes Recall@K on the held-out country and reports **GO** or **NO-GO**.

### Step 6: Full Training (if gate passes)

```bash
python -m src.train_bi_encoder train \
    --data_dir ./prepared_data \
    --output_dir ./output/bge-m3-lora-final \
    --mode full \
    --use_distillation
```

### Step 7: Merge LoRA for Inference

```bash
python -m src.train_bi_encoder merge \
    --adapter_path ./output/bge-m3-lora-final/final \
    --output_path ./output/bge-m3-merged
```

The merged model is a plain SentenceTransformer — zero adapter overhead at inference.

## Architecture

```
BGE-M3 (568M, XLM-RoBERTa-based, MIT)
  └─ LoRA adapters (r=64, all-linear, rsLoRA)
      ├─ attention: query, key, value, dense (all 12 layers)
      └─ FFN: intermediate.dense, output.dense (all 12 layers)

Loss = CachedMNRL(in-batch negatives) + 0.10 × DistillationAnchor(frozen BGE-M3)

Anti-forgetting stack:
  1. LoRA rank ceiling (structural — can't fully overwrite multilingual weights)
  2. Self-distillation (soft — penalizes drift from pretrained embedding space)
  3. Country-balanced mining (data — equal US/India negatives per batch)
  4. Held-out country gate (validation — reject fine-tune if it hurts)
```

## VRAM Budget (RTX 3060, 12GB)

| Component | Size |
|-----------|------|
| BGE-M3 base (bf16, frozen) | 1.14 GB |
| LoRA adapter + optimizer | 0.40 GB |
| Frozen model (distillation) | 1.14 GB |
| **Fixed total** | **~2.7 GB** |
| **Available for batch** | **~9.3 GB** |

## Configuration

All parameters are in `src/config.py` with detailed rationale comments. Key values:

| Parameter | Value | Source |
|-----------|-------|--------|
| LoRA rank | 64 | training.md committed config |
| LoRA alpha | 64 (with rsLoRA → effective 8) | regularization.md |
| Target modules | all-linear | regularization.md |
| Learning rate | 2e-5 | training.md |
| Batch size | 48 (physical) / 16 (mini-batch) | training.md |
| Max seq length | 80 | training.md |
| Epochs | 3 | training.md |
| Distillation weight | 0.10 | training.md |

## Troubleshooting

**OOM on 3060**: Reduce `--batch_size` to 32 or `--mini_batch_size` to 8.

**Gate fails (NO-GO)**:
1. Try `--distill_weight 0.15` (stronger anti-forgetting)
2. Try `--lora_r 32` (less capacity to overfit)
3. Try `--no_distillation` with `--lora_r 32` (minimal perturbation)
4. Fall back to off-the-shelf Qwen3-Embedding-0.6B

**Slow training**: Disable distillation with `--no_distillation` to halve compute time.

## File Structure

```
src/
├── config.py              # All configuration (LoRA, training, data, eval)
├── normalize.py           # Country-agnostic text normalization (Stage 0)
├── data_builder.py        # Training data construction (CPU)
├── losses.py              # CachedMNRL + self-distillation loss
├── train_bi_encoder.py    # Main training script (GPU)
└── eval_bi_encoder.py     # Held-out country evaluation
```
