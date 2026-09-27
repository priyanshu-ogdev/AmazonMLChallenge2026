# BGE-M3 Bi-Encoder LoRA Fine-Tuning for Entity Resolution

The authoritative end-to-end design, stage contracts, validation gates, Qwen
ablation, and execution order are in
[`../../docs/02_system_architecture.md`](../../docs/02_system_architecture.md). This README documents
the implemented Stage 2a/2b/2c modules and the Stage 3 scorer; blocking and final
submission assembly are documented in the main pipeline docs. The detailed Stage 2 feature specification is in
[`../../docs/05_stage2_features_and_embeddings.md`](../../docs/05_stage2_features_and_embeddings.md).
The Stage 3 training contract is in
[`../../docs/08_stage3_scoring_and_calibration.md`](../../docs/08_stage3_scoring_and_calibration.md).
Calibration utilities are isolated in `src/calibration.py` and can be
validated independently of the XGBoost scorer (`src/scoring.py`).
Stage 4 submission assembly is implemented in `src/decision.py`; it
preserves empty rows for S1 entities with no accepted matches (see [`../../docs/09_stage4_decision_and_singletons.md`](../../docs/09_stage4_decision_and_singletons.md)).

## Overview

This module fine-tunes **BGE-M3's encoder** (568M params, XLM-RoBERTa-based, MIT license) with **LoRA rank-64** for business entity resolution. The fine-tuned model produces embeddings for Stage 2a-i of the entity resolution pipeline, providing cosine similarity scores between S1 entities and S2/S3 candidates. Stage 2a-ii is implemented separately in `src/qwen_features.py` as an inference-only Qwen3 auxiliary feature, Stage 2c deterministic pair features are implemented in `src/pair_features.py`, and Stage 2b is reserved for the stretch Qwen3-0.6B causal generative matcher (see [`../../docs/07_stage2b_qwen3_generative_matcher_spec.md`](../../docs/07_stage2b_qwen3_generative_matcher_spec.md)).

**Key constraint**: The test set includes **France** (unseen in training). The
anti-forgetting stack protects the pretrained multilingual space, but the
adapter is accepted only after the two-direction held-out-country gate. A
failed gate must not be silently deployed.

## Quick Start (Local RTX 3060 / 32GB RAM Environment)

All operations run locally with full CUDA 12.4 acceleration and ThreadPoolExecutor multi-core CPU sharing.

### Step 1: Environment & Virtualenv
Ensure the project virtualenv (`.venv`) is activated. It contains PyTorch 2.6.0 with CUDA 12.4, PEFT, XGBoost, and sentence-transformers:
```powershell
.\scripts\00_verify_environment.ps1
```

### Step 2: Phase 1 Candidate Generation (Blocking)
Generates high-recall candidates across exact, n-gram, address, and phonetic channels:
```powershell
.\scripts\01_run_blocking.ps1 -Split "train" -MaxCandidates 50 -TopKSparse 50 -TopKDense 50 -NumWorkers 0
```
- **Index Caching:** Automatically reuses the cached 2.41GB inverted index (`candidate_index.pkl`) in ~44 seconds.
- **Multithreading:** Uses `ThreadPoolExecutor` sharing the 15.3GB index safely in RAM across all available CPU cores.

### Step 3: Phase 2a Data Preparation
Samples 50K entities per country (US + India balanced), mines hard negatives from Phase 1 candidate pairs, and builds bidirectional evaluation splits:
```powershell
.\scripts\02a_prepare_bi_encoder_data.ps1 `
    -SamplePerCountry 50000 `
    -BlockingCandidates output/phase1_blocking_train/candidate_pairs.tsv `
    -NegativesPerPositive 2
```
Or directly via Python:
```bash
python -m src.data_builder \
    --mode train_data \
    --data_dir dataset \
    --output_dir output/phase2_prepared_data \
    --sample_per_country 50000 \
    --eval_sample 2000 \
    --eval_corpus_neg 5000 \
    --seed 42 \
    --bidirectional_gate \
    --blocking_candidates output/phase1_blocking_train/candidate_pairs.tsv \
    --negatives_per_positive 2
```

### Step 4: Phase 2b BGE-M3 LoRA Training & 2-Way Gate
Trains the BGE-M3 rsLoRA adapter on GPU (RTX 3060 12GB VRAM, physical batch 48, mini-batch 16 with CachedMNRL) and automatically evaluates the bidirectional generalization gate (`US -> India` and `India -> US`):
```powershell
.\scripts\02b_train_and_eval_bi_encoder.ps1 -Epochs 3 -BatchSize 48 -MiniBatchSize 16 -LoraR 64
```
If the gate passes:
- Automatically trains on the full dataset.
- Automatically merges the LoRA adapter into a clean standalone model in `output/phase2_models/bge-m3-merged` with zero inference overhead.

### Step 5: Phase 2c Pair Features Extraction
Extracts deterministic lexical/address features on CPU and dense BGE-M3 cosine similarities on GPU:
```powershell
.\scripts\02c_extract_pair_features.ps1 -Split "train"
```

### Step 6: Phase 3 Grouped-OOF GBM Training & Calibration
Trains the entity-grouped XGBoost scorer with monotonic constraints and Platt/isotonic probability calibration:
```powershell
.\scripts\03_train_scoring_gbm.ps1 -Booster "gbtree" -Eta 0.03 -CountryMaskRate 0.15
```

### Step 7: Phase 4 & 5 Test Inference, Decision Assembly & Validation
Scores test candidates, applies injective 1-to-N assignment with optimal macro-$F_{0.5}$ thresholding, and audits the submission package:
```powershell
.\scripts\04_inference_and_decision.ps1
.\scripts\05_validate_submission.ps1
```

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
| LoRA rank | 64 | [`docs/06_stage2a_bge_m3_training_spec.md`](../../docs/06_stage2a_bge_m3_training_spec.md) |
| LoRA alpha | 64 (with rsLoRA → effective 8) | [`docs/06_stage2a_bge_m3_training_spec.md`](../../docs/06_stage2a_bge_m3_training_spec.md) |
| Target modules | all-linear | [`docs/06_stage2a_bge_m3_training_spec.md`](../../docs/06_stage2a_bge_m3_training_spec.md) |
| Learning rate | 2e-5 | [`docs/06_stage2a_bge_m3_training_spec.md`](../../docs/06_stage2a_bge_m3_training_spec.md) |
| Batch size | 48 (physical) / 16 (mini-batch) | [`docs/06_stage2a_bge_m3_training_spec.md`](../../docs/06_stage2a_bge_m3_training_spec.md) |
| Max seq length | 80 | [`docs/06_stage2a_bge_m3_training_spec.md`](../../docs/06_stage2a_bge_m3_training_spec.md) |
| Epochs | 3 | [`docs/06_stage2a_bge_m3_training_spec.md`](../../docs/06_stage2a_bge_m3_training_spec.md) |
| Distillation weight | 0.10 | [`docs/06_stage2a_bge_m3_training_spec.md`](../../docs/06_stage2a_bge_m3_training_spec.md) |

## Troubleshooting

**OOM on 3060**: Reduce `--batch_size` to 32 or `--mini_batch_size` to 8.

**Gate fails (NO-GO)**:
1. Try `--distill_weight 0.15` (stronger anti-forgetting)
2. Try `--lora_r 32` (less capacity to overfit)
3. Try `--no_distillation` with `--lora_r 32` (minimal perturbation)
4. Fall back to off-the-shelf BGE-M3 for the Stage 2a feature and evaluate
   Qwen3-Embedding-0.6B only as the documented inference-only auxiliary feature.

**Slow training**: Disable distillation with `--no_distillation` to halve compute time.

## File Structure

```
src/
├── config.py                 # All configuration (LoRA, training, data, eval)
├── normalize.py              # Country-agnostic text normalization & field parsing (Stage 0)
├── blocking.py               # Multi-channel candidate generation & recall audit (Stage 1)
├── data_builder.py           # Training data construction & hard-negative mining (Phase 2a)
├── losses.py                 # CachedMNRL + self-distillation loss
├── train_bi_encoder.py       # BGE-M3 rsLoRA training & merge CLI (Stage 2a-i)
├── eval_bi_encoder.py        # Held-out country gate evaluation
├── bge_features.py           # Stage 2a-i dense BGE-M3 cosine similarity features
├── qwen_features.py          # Stage 2a-ii auxiliary Qwen3 dense features
├── train_qwen_matcher.py     # Stage 2b Qwen3-0.6B causal generative matcher (stretch)
├── qwen_matcher_features.py  # Stage 2b generative matcher sliced verdict probabilities
├── pair_features.py          # Stage 2c deterministic 35 pair features
├── scoring.py                # Stage 3 Grouped-OOF XGBoost meta-learner & threshold sweep
├── calibration.py            # Stage 3 Platt & Isotonic probability calibration
└── decision.py               # Stage 4 greedy 1-to-N injective assignment (matching_results.tsv)
```

## Stage 2a-ii: Qwen auxiliary feature generation

Qwen is not fine-tuned. The feature builder applies the same entity-resolution
instruction to S1 and S2/S3 records, normalizes both embedding sets, encodes
each entity once, and scores only the final candidate set:

```bash
python -m src.qwen_features \
    --source1 ../../dataset/test/test_source1.tsv \
    --source2 ../../dataset/test/test_source2.tsv \
    --source3 ../../dataset/test/test_source3.tsv \
    --candidate-file ../../output/candidate_pairs.tsv \
    --output-file ../../output/qwen_pair_features.tsv
```

The generated `qwen_pair_features.tsv` contains one row per candidate pair and
the `qwen_pair_features.json` sidecar records model and encoding parameters.
Retain this feature only if the Stage 3 Qwen ablation improves entity-level
macro F0.5 without increasing false merges or harming country/source slices.

## Stage 2c: deterministic pair features

Build label-free lexical, address, numeric, postal, missingness, conflict, and
candidate-rank features for the exact candidate set:

```bash
python -m src.pair_features `
    --source1 ../../dataset/test/test_source1.tsv `
    --source2 ../../dataset/test/test_source2.tsv `
    --source3 ../../dataset/test/test_source3.tsv `
    --candidate-file ../../output/candidate_pairs.tsv `
    --output-file ../../output/pair_features.tsv
```

TF-IDF and other learned representations are intentionally excluded here and
must be fitted inside each Stage 3 training fold. The extractor rejects
unknown IDs and never consumes ground-truth labels.
