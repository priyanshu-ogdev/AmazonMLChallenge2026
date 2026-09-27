"""
run_pipeline.py -- Sequential GPU-accelerated entity resolution pipeline.

Stages run strictly one-at-a-time so the full 12 GB VRAM is dedicated to
each model in turn.  Between every GPU stage the model is explicitly deleted
and the CUDA cache is cleared before the next stage is loaded.

Usage:
  python run_pipeline.py \
    --source1      data/source1.tsv \
    --source2      data/source2.tsv \
    --source3      data/source3.tsv \
    --ground-truth data/ground_truth.tsv \
    --output-dir   outputs/ \
    --artifact-dir artifacts/

Optional flags:
  --skip-blocking         Skip Stage 1 (requires existing candidate_pairs.tsv)
  --skip-bge              Skip Stage 2a (requires existing bge_features.tsv)
  --skip-qwen             Skip Stage 2b (requires existing qwen_features.tsv)
  --skip-pair-features    Skip Stage 2c (requires existing pair_features.tsv)
  --skip-train            Skip Stage 3 training (requires existing gbm.json)
  --skip-score            Skip Stage 3 scoring
  --skip-decision         Skip Stage 4 decision
  --bge-batch-size N      BGE batch size (default: 128 GPU, 32 CPU)
  --qwen-batch-size N     Qwen batch size (default: 64 GPU, 16 CPU)
  --booster               XGBoost booster: gbtree (default) or dart
  --eta                   XGBoost learning rate (default: 0.03)
  --use-monotone-constraints
  --compare-dart
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("pipeline")


def _release_gpu(tag: str = "") -> None:
    """Free every CUDA tensor and reset the caching allocator."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("[GPU] VRAM released%s.", f" after {tag}" if tag else "")
    except ImportError:
        pass
    gc.collect()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_device() -> Optional[str]:
    return "cuda" if _cuda_available() else None


def _default_batch(default_gpu: int, default_cpu: int) -> int:
    return default_gpu if _cuda_available() else default_cpu


def run_blocking(source1, source2, output_dir, use_fast=False):
    """Stage 1: Multi-channel blocking."""
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("[STAGE 1] Blocking -- CPU+RAM, all cores (engine=%s).", "FastNormalizedBlocker" if use_fast else "MultiChannelBlocker")
    logger.info("=" * 70)
    if use_fast:
        from src.fast_blocking import run_fast_blocking as _fast_blocking
        _fast_blocking(
            source1_paths=source1,
            candidate_sources=source2,
            output_dir=output_dir,
        )
    else:
        from src.blocking import run_blocking as _blocking
        _blocking(
            source1_paths=source1,
            candidate_sources=source2,
            output_dir=output_dir,
        )
    logger.info("[STAGE 1] Done in %.1f s.", time.time() - t0)


def run_bge_features(source1, candidate_sources, candidate_file, output_dir, batch_size):
    """Stage 2a: BGE-M3 dense pair features."""
    from src.bge_features import build_bge_features
    output_file = output_dir / "bge_features.tsv"
    device = _gpu_device()
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("[STAGE 2a] BGE-M3 features -- device=%s, batch=%d.", device or "cpu", batch_size)
    logger.info("=" * 70)
    build_bge_features(
        source1_paths=source1,
        candidate_paths=candidate_sources,
        candidate_file=candidate_file,
        output_file=output_file,
        batch_size=batch_size,
        device=device,
    )
    _release_gpu("BGE-M3")
    logger.info("[STAGE 2a] Done in %.1f s -> %s", time.time() - t0, output_file)
    return output_file


def run_qwen_features(source1, candidate_sources, candidate_file, output_dir, batch_size):
    """Stage 2b: Qwen 0.6B dense pair features."""
    from src.qwen_features import build_qwen_features
    output_file = output_dir / "qwen_features.tsv"
    device = _gpu_device()
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("[STAGE 2b] Qwen features -- device=%s, batch=%d.", device or "cpu", batch_size)
    logger.info("=" * 70)
    build_qwen_features(
        source1_paths=source1,
        candidate_paths=candidate_sources,
        candidate_file=candidate_file,
        output_file=output_file,
        batch_size=batch_size,
        device=device,
    )
    _release_gpu("Qwen")
    logger.info("[STAGE 2b] Done in %.1f s -> %s", time.time() - t0, output_file)
    return output_file


def run_pair_features(source1, candidate_sources, candidate_file, output_dir, provenance_file=None):
    """Stage 2c: Deterministic lexical pair features (CPU only)."""
    from src.pair_features import build_pair_features, load_records
    output_file = output_dir / "pair_features.tsv"
    all_paths = list(source1) + list(candidate_sources)
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("[STAGE 2c] Pair features -- CPU, all cores, tuple-optimized.")
    logger.info("=" * 70)
    records = load_records(all_paths)
    build_pair_features(
        records=records,
        candidate_file=candidate_file,
        output_file=output_file,
        provenance_file=provenance_file,
    )
    del records
    gc.collect()
    logger.info("[STAGE 2c] Done in %.1f s -> %s", time.time() - t0, output_file)
    return output_file


def run_train(pair_features_file, ground_truth_file, artifact_dir, bge_file, qwen_file,
              booster, eta, country_mask_rate, use_monotone_constraints, compare_dart):
    """Stage 3 training: XGBoost GBM with GPU acceleration."""
    from src.scoring import run_training
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("[STAGE 3-train] GBM training -- device=%s, booster=%s, eta=%.3f.",
                "cuda" if _cuda_available() else "cpu", booster, eta)
    logger.info("=" * 70)
    metadata = run_training(
        feature_file=pair_features_file,
        ground_truth_file=ground_truth_file,
        output_dir=artifact_dir,
        bge_file=bge_file,
        qwen_file=qwen_file,
        booster=booster,
        eta=eta,
        country_mask_rate=country_mask_rate,
        use_monotone_constraints=use_monotone_constraints,
        compare_dart=compare_dart,
    )
    _release_gpu("GBM training")
    logger.info("[STAGE 3-train] Done in %.1f s. Threshold=%.4f, F0.5=%.4f.",
                time.time() - t0,
                float(metadata.get("threshold", 0.0)),
                float(metadata.get("macro_f05", 0.0)))


def run_score(pair_features_file, artifact_dir, output_dir, bge_file, qwen_file):
    """Stage 3 scoring: apply saved GBM + calibration."""
    import csv as _csv
    from src.scoring import score_candidates
    output_file = output_dir / "scored_candidates.tsv"
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("[STAGE 3-score] Scoring candidates -- device=%s.", "cuda" if _cuda_available() else "cpu")
    logger.info("=" * 70)
    scored = score_candidates(
        feature_file=pair_features_file,
        artifact_dir=artifact_dir,
        bge_file=bge_file,
        qwen_file=qwen_file,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_file, sep="\t", index=False, quoting=_csv.QUOTE_NONE, escapechar="\\")
    _release_gpu("GBM scoring")
    logger.info("[STAGE 3-score] Done in %.1f s -> %s (%d pairs).",
                time.time() - t0, output_file, len(scored))
    return output_file


def run_decision(scored_file, source1, output_dir, artifact_dir):
    """Stage 4: threshold + injective decision -> final submission."""
    import csv as _csv
    import pandas as pd
    from src.decision import assemble_matching_results, load_source1_ids
    output_file = output_dir / "submission.tsv"
    meta_path = artifact_dir / "stage3_metadata.json"
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    threshold = float(meta.get("threshold_injective") or meta["threshold"])
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("[STAGE 4] Decision -- threshold=%.4f.", threshold)
    logger.info("=" * 70)
    scored = pd.read_csv(scored_file, sep="\t", dtype=str, keep_default_na=False, quoting=_csv.QUOTE_NONE)
    scored["calibrated_score"] = scored["calibrated_score"].astype(float)
    source1_ids: List[str] = []
    for p in source1:
        source1_ids.extend(load_source1_ids(p))
    result = assemble_matching_results(scored, source1_ids, threshold=threshold, injective=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, sep="\t", index=False, quoting=_csv.QUOTE_NONE, escapechar="\\")
    logger.info("[STAGE 4] Done in %.1f s -> %s (%d rows).", time.time() - t0, output_file, len(result))
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential GPU-accelerated entity resolution pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source1", type=Path, nargs="+", required=True)
    parser.add_argument("--source2", type=Path, nargs="+", default=None)
    parser.add_argument("--source3", type=Path, nargs="+", default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--index-cache-dir", type=Path, default=None)
    parser.add_argument("--skip-blocking", action="store_true")
    parser.add_argument("--skip-bge", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-pair-features", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-decision", action="store_true")
    parser.add_argument("--bge-batch-size", type=int, default=None)
    parser.add_argument("--qwen-batch-size", type=int, default=None)
    parser.add_argument("--booster", choices=["gbtree", "dart"], default="gbtree")
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--country-mask-rate", type=float, default=0.15)
    parser.add_argument("--use-monotone-constraints", action="store_true")
    parser.add_argument("--compare-dart", action="store_true")
    parser.add_argument("--fast-blocking", action="store_true", help="Use vectorized Polars FastNormalizedBlocker")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir: Path = args.artifact_dir or (output_dir / "artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    index_cache_dir: Path = args.index_cache_dir or output_dir

    source1: List[Path] = list(args.source1)
    candidate_sources: List[Path] = []
    if args.source2:
        candidate_sources.extend(args.source2)
    if args.source3:
        candidate_sources.extend(args.source3)
    if not candidate_sources:
        parser.error("At least one of --source2 or --source3 must be provided.")

    bge_batch = args.bge_batch_size or _default_batch(128, 32)
    qwen_batch = args.qwen_batch_size or _default_batch(64, 16)

    if _cuda_available():
        try:
            import torch
            dev = torch.cuda.get_device_properties(0)
            logger.info("[GPU] %s -- %.1f GB VRAM. Sequential GPU mode: one model at a time.",
                        dev.name, dev.total_memory / 1e9)
        except Exception:
            logger.info("[GPU] CUDA available. Sequential GPU mode active.")
    else:
        logger.info("[CPU] No CUDA detected. All stages run on CPU.")

    pipeline_start = time.time()

    # Stage 1 - Blocking
    candidate_pairs_file = output_dir / "candidate_pairs.tsv"
    provenance_file = output_dir / "candidate_provenance.tsv"

    if args.skip_blocking:
        logger.info("[STAGE 1] Skipped. Using existing: %s", candidate_pairs_file)
        if not candidate_pairs_file.exists():
            logger.error("candidate_pairs.tsv not found -- cannot skip Stage 1.")
            sys.exit(1)
    else:
        run_blocking(source1=source1, source2=candidate_sources,
                     output_dir=output_dir, use_fast=args.fast_blocking)

    # Stage 2a - BGE
    bge_file = output_dir / "bge_features.tsv"
    if args.skip_bge:
        logger.info("[STAGE 2a] Skipped. Using existing: %s", bge_file)
    else:
        bge_file = run_bge_features(source1=source1, candidate_sources=candidate_sources,
                                    candidate_file=candidate_pairs_file,
                                    output_dir=output_dir, batch_size=bge_batch)

    # Stage 2b - Qwen
    qwen_file = output_dir / "qwen_features.tsv"
    if args.skip_qwen:
        logger.info("[STAGE 2b] Skipped. Using existing: %s", qwen_file)
    else:
        qwen_file = run_qwen_features(source1=source1, candidate_sources=candidate_sources,
                                      candidate_file=candidate_pairs_file,
                                      output_dir=output_dir, batch_size=qwen_batch)

    # Stage 2c - Pair features
    pair_features_file = output_dir / "pair_features.tsv"
    if args.skip_pair_features:
        logger.info("[STAGE 2c] Skipped. Using existing: %s", pair_features_file)
    else:
        pair_features_file = run_pair_features(
            source1=source1, candidate_sources=candidate_sources,
            candidate_file=candidate_pairs_file, output_dir=output_dir,
            provenance_file=provenance_file if provenance_file.exists() else None)

    if args.ground_truth is None and not args.skip_train:
        logger.warning("No --ground-truth provided. Skipping GBM training.")
        args.skip_train = True

    # Stage 3 - Train
    if args.skip_train:
        logger.info("[STAGE 3-train] Skipped.")
        if not (artifact_dir / "gbm.json").exists():
            logger.error("gbm.json not found at %s.", artifact_dir)
            sys.exit(1)
    else:
        run_train(pair_features_file=pair_features_file,
                  ground_truth_file=args.ground_truth,
                  artifact_dir=artifact_dir,
                  bge_file=bge_file if bge_file.exists() else None,
                  qwen_file=qwen_file if qwen_file.exists() else None,
                  booster=args.booster, eta=args.eta,
                  country_mask_rate=args.country_mask_rate,
                  use_monotone_constraints=args.use_monotone_constraints,
                  compare_dart=args.compare_dart)

    # Stage 3 - Score
    scored_file = output_dir / "scored_candidates.tsv"
    if args.skip_score:
        logger.info("[STAGE 3-score] Skipped. Using existing: %s", scored_file)
    else:
        scored_file = run_score(pair_features_file=pair_features_file,
                                artifact_dir=artifact_dir, output_dir=output_dir,
                                bge_file=bge_file if bge_file.exists() else None,
                                qwen_file=qwen_file if qwen_file.exists() else None)

    # Stage 4 - Decision
    if args.skip_decision:
        logger.info("[STAGE 4] Skipped.")
    else:
        run_decision(scored_file=scored_file, source1=source1,
                     output_dir=output_dir, artifact_dir=artifact_dir)

    total = time.time() - pipeline_start
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE in %.1f s (%.1f min).", total, total / 60)
    logger.info("Outputs -> %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
