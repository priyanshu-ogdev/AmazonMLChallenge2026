"""
Training data construction for bi-encoder LoRA fine-tuning.

Runs on CPU (no GPU required). Produces a HuggingFace Dataset ready
for SentenceTransformerTrainer on the GPU machine.

Strategy (from docs_final/training.md):
1. Positive pairs: every (S1, S2/S3) pair in train_ground_truth
2. Country-balanced sampling: equal entities per country (anti-forgetting layer 3)
3. Negatives: in-batch negatives via CachedMNRL (default), or explicit hard
   negatives from blocking output (when available)
4. Singletons excluded: bi-encoder produces similarity scores; GBM + threshold
   handle singleton identification
5. Held-out country split: train on one country, eval on the other

Data is normalized using country-agnostic Stage 0 normalization before saving,
so the training script does no text preprocessing.

Usage:
    python -m src.data_builder \\
        --data_dir ../../dataset \\
        --output_dir ./prepared_data \\
        --sample_per_country 50000

Output structure:
    prepared_data/
    ├── train_dataset/          # HuggingFace Dataset (Arrow format)
    ├── eval_queries.json       # {qid: query_text} for IR evaluator
    ├── eval_corpus.json        # {cid: corpus_text}
    ├── eval_relevant.json      # {qid: [relevant_cids]}
    ├── data_stats.json         # Statistics
    └── country_split_info.json # Split details
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    print("ERROR: 'datasets' library required. Install: pip install datasets")
    raise

from src.normalize import normalize_entity
from src.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory-efficient TSV loading
# ---------------------------------------------------------------------------

def load_entity_country_map(
    source1_path: str, chunk_size: int = 100_000
) -> Dict[str, str]:
    """
    Load only (entity_id, country) from S1 file. Memory-efficient.
    Returns dict: entity_id → country.
    """
    logger.info(f"Loading entity-country map from {source1_path}...")
    country_map = {}
    for chunk in pd.read_csv(
        source1_path, sep="\t", usecols=["entity_id", "country"],
        chunksize=chunk_size, dtype=str,
    ):
        for eid, country in zip(chunk["entity_id"], chunk["country"]):
            country_map[eid] = country
    logger.info(f"  Loaded {len(country_map)} entity-country mappings")
    return country_map


def load_records_by_ids(
    filepath: str,
    entity_ids: Set[str],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    Load records matching given entity IDs, reading the TSV in chunks.
    Only keeps rows whose entity_id is in the provided set.
    """
    logger.info(f"Loading records from {filepath} ({len(entity_ids)} target IDs)...")
    chunks = []
    total_read = 0
    for chunk in pd.read_csv(filepath, sep="\t", chunksize=chunk_size, dtype=str):
        matched = chunk[chunk["entity_id"].isin(entity_ids)]
        if len(matched) > 0:
            chunks.append(matched)
        total_read += len(chunk)
    if chunks:
        result = pd.concat(chunks, ignore_index=True)
        logger.info(f"  Loaded {len(result)} records (scanned {total_read} total)")
        return result
    else:
        logger.warning(f"  No matching records found in {filepath}")
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])


def load_random_sample_by_country(
    filepath: str,
    country: str,
    n: int,
    exclude_ids: Set[str],
    chunk_size: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a random sample of records from a specific country,
    excluding specified entity IDs. Used for negative mining.
    """
    rng = np.random.RandomState(seed)
    all_candidates = []
    for chunk in pd.read_csv(filepath, sep="\t", chunksize=chunk_size, dtype=str):
        filtered = chunk[
            (chunk["country"] == country) & (~chunk["entity_id"].isin(exclude_ids))
        ]
        if len(filtered) > 0:
            all_candidates.append(filtered)

    if not all_candidates:
        return pd.DataFrame()

    combined = pd.concat(all_candidates, ignore_index=True)
    if len(combined) <= n:
        return combined
    return combined.sample(n=n, random_state=rng)


# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: str) -> pd.DataFrame:
    """Load and parse ground truth file."""
    logger.info(f"Loading ground truth from {gt_path}...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    logger.info(f"  Total rows: {len(gt)}")

    # Filter to entities with matches (exclude singletons for bi-encoder)
    gt_with_matches = gt[gt["matched_entity_ids"].notna()].copy()
    singletons = len(gt) - len(gt_with_matches)
    logger.info(f"  With matches: {len(gt_with_matches)}, Singletons: {singletons}")

    return gt_with_matches


def parse_matched_ids(matched_str: str) -> List[str]:
    """Parse comma-separated matched entity IDs."""
    if pd.isna(matched_str) or not matched_str.strip():
        return []
    return [eid.strip() for eid in matched_str.split(",") if eid.strip()]


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------

def build_positive_pairs(
    sampled_gt: pd.DataFrame,
    s1_records: Dict[str, Dict],
    s2s3_records: Dict[str, Dict],
) -> List[Dict[str, str]]:
    """
    Build positive (anchor, positive) pairs from ground truth.

    For each S1 entity, create a pair with each of its matched S2/S3 entities.
    Text is already normalized.

    Returns list of dicts: {"anchor": text, "positive": text, "country": country}
    """
    pairs = []
    skipped = 0

    for _, row in tqdm(sampled_gt.iterrows(), total=len(sampled_gt), desc="Building pairs"):
        s1_id = row["source1_entity_id"]
        s1_info = s1_records.get(s1_id)
        if s1_info is None:
            skipped += 1
            continue

        anchor_text = s1_info["normalized_text"]
        country = s1_info["country"]

        matched_ids = parse_matched_ids(row["matched_entity_ids"])
        for mid in matched_ids:
            match_info = s2s3_records.get(mid)
            if match_info is None:
                skipped += 1
                continue
            pairs.append({
                "anchor": anchor_text,
                "positive": match_info["normalized_text"],
                "country": country,
                "s1_id": s1_id,
                "matched_id": mid,
            })

    if skipped > 0:
        logger.warning(f"  Skipped {skipped} pairs due to missing records")

    return pairs


def build_evaluation_data(
    eval_gt: pd.DataFrame,
    s1_records: Dict[str, Dict],
    s2s3_records: Dict[str, Dict],
    neg_records: Dict[str, Dict],
    max_queries: int = 2000,
    seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """
    Build evaluation data for InformationRetrievalEvaluator.

    Returns:
        queries: {qid: query_text}
        corpus: {cid: corpus_text}
        relevant_docs: {qid: set of relevant cids}
    """
    rng = np.random.RandomState(seed)

    # Sample queries
    if len(eval_gt) > max_queries:
        eval_sample = eval_gt.sample(n=max_queries, random_state=rng)
    else:
        eval_sample = eval_gt

    queries = {}
    corpus = {}
    relevant_docs = {}

    for _, row in eval_sample.iterrows():
        s1_id = row["source1_entity_id"]
        s1_info = s1_records.get(s1_id)
        if s1_info is None:
            continue

        queries[s1_id] = s1_info["normalized_text"]
        relevant = set()

        matched_ids = parse_matched_ids(row["matched_entity_ids"])
        for mid in matched_ids:
            match_info = s2s3_records.get(mid)
            if match_info:
                corpus[mid] = match_info["normalized_text"]
                relevant.add(mid)

        relevant_docs[s1_id] = relevant

    # Add negative documents to corpus
    for neg_id, neg_info in neg_records.items():
        if neg_id not in corpus:
            corpus[neg_id] = neg_info["normalized_text"]

    logger.info(
        f"  Eval data: {len(queries)} queries, {len(corpus)} corpus docs, "
        f"{sum(len(v) for v in relevant_docs.values())} relevant pairs"
    )
    return queries, corpus, relevant_docs


# ---------------------------------------------------------------------------
# Record normalization helper
# ---------------------------------------------------------------------------

def records_to_dict(df: pd.DataFrame) -> Dict[str, Dict]:
    """Convert DataFrame to dict of {entity_id: {normalized_text, country, ...}}."""
    result = {}
    for _, row in df.iterrows():
        eid = row["entity_id"]
        name = str(row.get("business_name", "")) if pd.notna(row.get("business_name")) else ""
        addr = str(row.get("business_address", "")) if pd.notna(row.get("business_address")) else ""
        country = str(row.get("country", "")) if pd.notna(row.get("country")) else ""
        result[eid] = {
            "entity_id": eid,
            "normalized_text": normalize_entity(name, addr),
            "country": country,
            "business_name": name,
            "business_address": addr,
        }
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_training_data(config: DataConfig):
    """
    Full data preparation pipeline.

    Steps:
    1. Load ground truth and entity-country mapping
    2. Sample entities per country (balanced)
    3. Collect all needed entity IDs
    4. Load and normalize S1, S2, S3 records (chunked, memory-efficient)
    5. Build positive pairs
    6. Build held-out country evaluation data
    7. Save everything to disk
    """
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    rng = np.random.RandomState(config.seed)

    # -----------------------------------------------------------------------
    # Step 1: Load ground truth and country mapping
    # -----------------------------------------------------------------------
    gt = load_ground_truth(str(train_dir / "train_ground_truth.tsv"))
    country_map = load_entity_country_map(str(train_dir / "train_source1.tsv"), config.chunk_size)

    # Add country to ground truth
    gt["country"] = gt["source1_entity_id"].map(country_map)
    gt = gt.dropna(subset=["country"])

    countries = sorted(gt["country"].unique())
    logger.info(f"Countries in training data: {countries}")
    for c in countries:
        logger.info(f"  {c}: {(gt['country'] == c).sum()} entities with matches")

    # -----------------------------------------------------------------------
    # Step 2: Sample entities per country (balanced)
    # -----------------------------------------------------------------------
    sampled_dfs = []
    for country in countries:
        country_gt = gt[gt["country"] == country]
        n_sample = min(config.sample_per_country, len(country_gt))
        sampled = country_gt.sample(n=n_sample, random_state=rng)
        sampled_dfs.append(sampled)
        logger.info(f"  Sampled {n_sample} from {country}")

    sampled_gt = pd.concat(sampled_dfs, ignore_index=True)
    logger.info(f"Total sampled entities: {len(sampled_gt)}")

    # -----------------------------------------------------------------------
    # Step 3: Collect all needed entity IDs
    # -----------------------------------------------------------------------
    s1_ids_needed = set(sampled_gt["source1_entity_id"])
    s2_ids_needed = set()
    s3_ids_needed = set()

    for matched_str in sampled_gt["matched_entity_ids"]:
        for mid in parse_matched_ids(matched_str):
            if mid.startswith("S2-"):
                s2_ids_needed.add(mid)
            elif mid.startswith("S3-"):
                s3_ids_needed.add(mid)

    logger.info(
        f"Need: {len(s1_ids_needed)} S1, {len(s2_ids_needed)} S2, "
        f"{len(s3_ids_needed)} S3 records"
    )

    # -----------------------------------------------------------------------
    # Step 4: Load and normalize records
    # -----------------------------------------------------------------------
    s1_df = load_records_by_ids(str(train_dir / "train_source1.tsv"), s1_ids_needed, config.chunk_size)
    s2_df = load_records_by_ids(str(train_dir / "train_source2.tsv"), s2_ids_needed, config.chunk_size)
    s3_df = load_records_by_ids(str(train_dir / "train_source3.tsv"), s3_ids_needed, config.chunk_size)

    logger.info("Normalizing records...")
    s1_records = records_to_dict(s1_df)
    s2s3_records = records_to_dict(pd.concat([s2_df, s3_df], ignore_index=True))

    logger.info(f"  Normalized: {len(s1_records)} S1, {len(s2s3_records)} S2/S3")

    # -----------------------------------------------------------------------
    # Step 5: Build positive pairs, split by country for held-out evaluation
    # -----------------------------------------------------------------------
    # Train split: first country (US if available)
    # Eval split: second country (India if available) — France proxy
    train_country = countries[0] if len(countries) > 0 else "US"
    eval_country = countries[1] if len(countries) > 1 else countries[0]

    train_gt = sampled_gt[sampled_gt["country"] == train_country]
    eval_gt = sampled_gt[sampled_gt["country"] == eval_country]

    # Also build a "full" training set (both countries) for the final model
    logger.info(f"Building training pairs ({train_country} for train, {eval_country} for eval)...")

    train_pairs = build_positive_pairs(train_gt, s1_records, s2s3_records)
    eval_pairs = build_positive_pairs(eval_gt, s1_records, s2s3_records)
    all_pairs = build_positive_pairs(sampled_gt, s1_records, s2s3_records)

    logger.info(f"  Train pairs ({train_country}): {len(train_pairs)}")
    logger.info(f"  Eval pairs ({eval_country}): {len(eval_pairs)}")
    logger.info(f"  All pairs (both countries): {len(all_pairs)}")

    # -----------------------------------------------------------------------
    # Step 6: Build IR evaluation data (held-out country)
    # -----------------------------------------------------------------------
    logger.info("Building IR evaluation data for held-out country gate...")

    # Load negative corpus for eval (random S2/S3 from eval country)
    eval_positive_ids = set()
    for _, row in eval_gt.iterrows():
        eval_positive_ids.update(parse_matched_ids(row["matched_entity_ids"]))

    neg_dfs = []
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        neg_sample = load_random_sample_by_country(
            str(train_dir / src_file),
            country=eval_country,
            n=config.eval_corpus_negatives // 2,
            exclude_ids=eval_positive_ids,
            chunk_size=config.chunk_size,
            seed=config.seed,
        )
        if len(neg_sample) > 0:
            neg_dfs.append(neg_sample)

    neg_records = records_to_dict(pd.concat(neg_dfs, ignore_index=True)) if neg_dfs else {}

    queries, corpus, relevant_docs = build_evaluation_data(
        eval_gt, s1_records, s2s3_records, neg_records,
        max_queries=config.eval_sample_per_country,
        seed=config.seed,
    )

    # -----------------------------------------------------------------------
    # Step 7: Save everything
    # -----------------------------------------------------------------------
    logger.info("Saving datasets...")

    # Convert pairs to HuggingFace Dataset
    def pairs_to_dataset(pairs: List[Dict]) -> Dataset:
        return Dataset.from_dict({
            "anchor": [p["anchor"] for p in pairs],
            "positive": [p["positive"] for p in pairs],
        })

    # Save held-out-country training set
    held_out_ds = DatasetDict({
        "train": pairs_to_dataset(train_pairs),
        "eval": pairs_to_dataset(eval_pairs),
    })
    held_out_path = output_dir / "held_out_country_dataset"
    held_out_ds.save_to_disk(str(held_out_path))
    logger.info(f"  Held-out country dataset saved to {held_out_path}")

    # Save full training set (both countries, for final model after gate passes)
    full_ds = DatasetDict({
        "train": pairs_to_dataset(all_pairs),
    })
    full_path = output_dir / "full_training_dataset"
    full_ds.save_to_disk(str(full_path))
    logger.info(f"  Full training dataset saved to {full_path}")

    # Save IR evaluation data
    def serialize_relevant(rd):
        return {k: list(v) for k, v in rd.items()}

    with open(output_dir / "eval_queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    with open(output_dir / "eval_corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(output_dir / "eval_relevant.json", "w", encoding="utf-8") as f:
        json.dump(serialize_relevant(relevant_docs), f, ensure_ascii=False, indent=2)

    logger.info(f"  IR eval data saved ({len(queries)} queries, {len(corpus)} corpus)")

    # Save split info and stats
    stats = {
        "train_country": train_country,
        "eval_country": eval_country,
        "countries": countries,
        "sampled_per_country": config.sample_per_country,
        "total_sampled_entities": len(sampled_gt),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "all_pairs": len(all_pairs),
        "eval_queries": len(queries),
        "eval_corpus": len(corpus),
        "avg_text_length": np.mean([
            len(p["anchor"].split()) + len(p["positive"].split())
            for p in all_pairs[:1000]
        ]) if all_pairs else 0,
        "seed": config.seed,
    }
    with open(output_dir / "data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Data preparation complete!")
    logger.info(f"  Output directory: {output_dir}")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build training data for bi-encoder LoRA fine-tuning"
    )
    parser.add_argument("--data_dir", type=str, default="../../dataset",
                        help="Path to dataset directory containing train/ and test/")
    parser.add_argument("--output_dir", type=str, default="./prepared_data",
                        help="Output directory for prepared datasets")
    parser.add_argument("--sample_per_country", type=int, default=50000,
                        help="Number of entities to sample per country")
    parser.add_argument("--eval_sample", type=int, default=2000,
                        help="Number of queries for IR evaluation")
    parser.add_argument("--eval_corpus_neg", type=int, default=5000,
                        help="Number of negative docs in eval corpus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Chunk size for reading large TSV files")

    args = parser.parse_args()

    config = DataConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_per_country=args.sample_per_country,
        eval_sample_per_country=args.eval_sample,
        eval_corpus_negatives=args.eval_corpus_neg,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    build_training_data(config)


if __name__ == "__main__":
    main()
