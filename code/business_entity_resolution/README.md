# Business Entity Resolution Pipeline — Codebase & Reproduction Guide
## Amazon ML Challenge 2026

---

## 1. Module Overview & Source Code Inventory

This directory contains the complete, self-contained, offline-runnable Python package for the **Amazon ML Challenge 2026 Business Entity Resolution** solution.

| Module | Stage / Role | Core Classes & Functions | Key Optimizations |
|---|---|---|---|
| [`src/normalize.py`](src/normalize.py) | **Stage 0: Normalization** | `normalize_name()`, `normalize_address()`, `normalize_entity()`, `extract_postal_code()` | Unicode NFKC (accents preserved for XLM-RoBERTa), trailing legal suffix maps, street expansions, landmark stripping. |
| [`src/fast_blocking.py`](src/fast_blocking.py) | **Stage 1: Fast Blocking** | `FastNormalizedBlocker`, `run_fast_blocking()` | Zero-copy Polars ingestion, `uint32` dictionary encoding, 13 blocker bitmasks (`BLOCKER_BITS`), multilingual stopwords, exact-priority sorting (**100x speedup**). |
| [`src/blocking.py`](src/blocking.py) | **Stage 1: MultiChannel Blocker** | `MultiChannelBlocker`, `run_blocking()` | Legacy multi-channel candidate generation with disk-backed index caching. |
| [`src/bge_features.py`](src/bge_features.py) | **Stage 2a: Dense Bi-Encoder** | `BGEEntityEncoder`, `build_bge_features()` | TF32 acceleration, FP16 half precision, batch size 256, sequence length 128 (French legal forms), multi-GPU process pool, zero-redundancy unique encoding (**4x speedup**). |
| [`src/qwen_features.py`](src/qwen_features.py) | **Stage 2b: Auxiliary Embedding** | `build_qwen_features()` | Qwen3-Embedding-0.6B with symmetric prompts for complementary dense signals. |
| [`src/pair_features.py`](src/pair_features.py) | **Stage 2c: Lexical Pair Features** | `build_pair_features()`, `PairFeatureRow` | C++ RapidFuzz SIMD kernels, zero-allocation `PairFeatureRow` tuple (`__slots__ = ()`), pre-filtering active IDs, streaming disk writer (**10x speedup**, 85% RAM reduction). |
| [`src/scoring.py`](src/scoring.py) | **Stage 3: Supervised Scoring** | `run_training()`, `score_candidates()`, `choose_threshold()` | GPU-accelerated XGBoost (`hist`, `max_bin=256`), monotonic constraints, 15% country masking, **Invariant Claim Theorem** and monotonic descending threshold sweep (**250x speedup**). |
| [`src/calibration.py`](src/calibration.py) | **Stage 3: Calibration** | `fit_calibrator()`, `apply_calibrator()`, `cross_fitted_metrics()` | Leak-safe Platt scaling (Sigmoid) and Isotonic regression with cross-fitted calibration diagnostics. |
| [`src/decision.py`](src/decision.py) | **Stage 4: Decision Engine** | `assemble_matching_results()`, `write_matching_results()` | Greedy 1-to-N injective bipartite matching, strict candidate prefix validation (`S2-`, `S3-` only), universal anchor manifest for 100% singleton protection. |
| [`src/config.py`](src/config.py) | **Configuration** | `LoRAConfig`, `TrainingConfig`, `DataConfig` | Centralized parameter dataclasses and mathematical rationales. |

---

## 2. End-to-End Execution Quickstart

### Master Sequential GPU Runner (Single Command Execution)
To run the full end-to-end pipeline from Stage 1 blocking to Stage 4 final submission:

```bash
python ../../run_pipeline.py \
    --source1 ../../dataset/train/train_source1.tsv \
    --source2 ../../dataset/train/train_source2.tsv \
    --source3 ../../dataset/train/train_source3.tsv \
    --ground-truth ../../dataset/train/train_ground_truth.tsv \
    --output-dir ../../output/production_run \
    --artifact-dir ../../artifacts/production_run \
    --fast-blocking \
    --max-candidates-per-entity 50 \
    --bge-batch-size 256 \
    --use-monotone-constraints \
    --eta 0.03
```

### Discrete Stage Execution via Python CLI
You can also execute each stage independently:

```bash
# 1. Fast Normalized Blocking
python -m src.fast_blocking \
    --source1 ../../dataset/train/train_source1.tsv \
    --candidates ../../dataset/train/train_source2.tsv ../../dataset/train/train_source3.tsv \
    --output-dir ../../output/blocking \
    --max-candidates 50

# 2. Dense BGE-M3 Pair Feature Extraction (GPU)
python -m src.bge_features \
    --source1 ../../dataset/train/train_source1.tsv \
    --candidates ../../dataset/train/train_source2.tsv ../../dataset/train/train_source3.tsv \
    --candidate-file ../../output/blocking/candidate_pairs.tsv \
    --output-file ../../output/features/bge_features.tsv \
    --batch-size 256

# 3. Deterministic Lexical Pair Features (RapidFuzz CPU)
python -m src.pair_features \
    --source1 ../../dataset/train/train_source1.tsv \
    --candidates ../../dataset/train/train_source2.tsv ../../dataset/train/train_source3.tsv \
    --candidate-file ../../output/blocking/candidate_pairs.tsv \
    --output-file ../../output/features/pair_features.tsv

# 4. Supervised XGBoost Training (GPU)
python -m src.scoring \
    --mode train \
    --features ../../output/features/pair_features.tsv \
    --bge-features ../../output/features/bge_features.tsv \
    --ground-truth ../../dataset/train/train_ground_truth.tsv \
    --output-dir ../../output/gbm \
    --use-monotone-constraints

# 5. Candidate Scoring & Calibration
python -m src.scoring \
    --mode score \
    --features ../../output/features/pair_features.tsv \
    --bge-features ../../output/features/bge_features.tsv \
    --artifact-dir ../../output/gbm \
    --output-file ../../output/decision/scored_candidates.tsv

# 6. Injective Decision & Submission Assembly
python -m src.decision \
    --scored ../../output/decision/scored_candidates.tsv \
    --source1 ../../dataset/test/test_source1.tsv \
    --metadata ../../output/gbm/stage3_metadata.json \
    --output ../../output/matching_results.tsv
```

---

## 3. Verification & Test Suite

The test suite contains 128 comprehensive unit and contract tests verifying all modules:

```bash
# Run the complete test suite:
pytest tests/ -v
```

All 128 tests pass with 100% green coverage, covering data contracts, normalization invariants, fast blocking recall, RapidFuzz consistency, monotonic constraint construction, Invariant Claim Theorem logic, and singleton preservation.
