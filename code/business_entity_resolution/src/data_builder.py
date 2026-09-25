"""
Data Preprocessing & Training Data Construction Pipeline (Stage 0 & Stage 1 Prep)
==================================================================================
Implements end-to-end data processing for the Business Entity Resolution Challenge:

1. Stage 0 Batch/Streaming Normalization:
   - Country-agnostic text normalization (accents preserved via NFKC, URLs stripped)
   - Suffix-aware legal forms canonicalization (bidirectional, last 3 tokens only)
   - Positional address abbreviation expansion
   - Landmark noise stripping with auxiliary signal preservation
   - Structural field extraction (leading street number, trailing segment, digit runs)
   - Country canonicalization (open-set: alias folding for known labels, pass-through for unseen)
   - Memory-efficient streaming TSV normalizer (handles 5M+ row files chunk-by-chunk without OOM)

2. Bi-Encoder LoRA Fine-Tuning Data Construction (Training Stage 2a):
   - Positive pairs from train_ground_truth.tsv
   - Country-balanced sampling: equal S1 entities per country (Anti-forgetting Layer 3)
   - Held-out country gate split: train on US, eval on India (France proxy)
   - Full dataset (both countries) for final retrain after gate validation
   - InformationRetrievalEvaluator data (eval_queries, eval_corpus, eval_relevant)
   - Universal storage: saves HuggingFace Dataset (Arrow) when available AND
     streaming JSONL files for immediate PyTorch/SentenceTransformers loading.

Usage:
    # 1. Run quick verification test (uses sample records, no large files needed):
    python -m src.data_builder --mode test

    # 2. Build bi-encoder training and evaluation data:
    python -m src.data_builder --mode train_data \\
        --data_dir ../../dataset \\
        --output_dir ./prepared_data \\
        --sample_per_country 50000

    # 3. Normalize all raw TSVs (Stage 0 full preprocessed files):
    python -m src.data_builder --mode stage0_normalize \\
        --data_dir ../../dataset \\
        --output_dir ../../dataset/stage0_normalized \\
        --splits train test

Output structure:
    prepared_data/
    ├── held_out_country_dataset/   # HuggingFace DatasetDict (train + eval)
    ├── full_training_dataset/      # HuggingFace DatasetDict (train)
    ├── held_out_train_pairs.jsonl  # Line-delimited JSON pairs (anchor, positive)
    ├── held_out_eval_pairs.jsonl   # Line-delimited JSON pairs (anchor, positive)
    ├── full_training_pairs.jsonl   # All pairs for final model
    ├── eval_queries.json           # {qid: query_text} for IR evaluator
    ├── eval_corpus.json            # {cid: corpus_text}
    ├── eval_relevant.json          # {qid: [relevant_cids]}
    ├── data_stats.json             # Dataset statistics and metadata
    └── country_split_info.json     # Details on country split and sample sizes
"""

import os
import csv
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Iterator, Any
from collections import defaultdict

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from datasets import Dataset, DatasetDict
    HAS_DATASETS = True
except ImportError:
    Dataset = None
    DatasetDict = None
    HAS_DATASETS = False

from src.normalize import (
    normalize_entity,
    normalize_name,
    normalize_address,
    normalize_entity_record,
    normalize_dataframe_records,
    canonicalize_country,
    country_match_flag,
    source_from_entity_id,
    extract_postal_code,
    extract_structural_fields,
)
from src.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 0 Streaming File Normalizer (Memory-Safe for 5M+ Rows)
# ---------------------------------------------------------------------------

STAGE0_TSV_COLUMNS = [
    "entity_id",
    "source",
    "country",
    "country_canonical",
    "raw_name",
    "raw_address",
    "norm_name",
    "norm_address",
    "encoder_text",
    "is_address_missing",
    "postal_code",
    "street_number",
    "trailing_segment",
    "digit_runs",
]


def stream_source_tsv(path: str | Path) -> Iterator[Dict[str, Any]]:
    """
    Yields normalized entity record dictionaries one row at a time.
    Safe for 5M+ row files without loading into memory.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing expected columns {missing}")
        for row in reader:
            yield normalize_entity_record(
                name=row["business_name"],
                address=row["business_address"],
                entity_id=row["entity_id"],
                country=row["country"],
            )


def normalize_tsv_file(
    in_path: str | Path,
    out_path: str | Path,
    chunk_size: int = 50_000,
) -> int:
    """
    Streams raw TSV -> normalized TSV with buffered batch writes.
    Returns total rows written.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Normalizing {in_path.name} -> {out_path.name}...")
    count = 0
    buffer = []

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, delimiter="\t", fieldnames=STAGE0_TSV_COLUMNS)
        writer.writeheader()

        for rec in stream_source_tsv(in_path):
            buffer.append({
                "entity_id": rec["entity_id"],
                "source": rec["source"],
                "country": rec["country"],
                "country_canonical": rec["country_canonical"],
                "raw_name": rec["raw_name"],
                "raw_address": rec["raw_address"],
                "norm_name": rec["norm_name"],
                "norm_address": rec["norm_address"],
                "encoder_text": rec["encoder_text"],
                "is_address_missing": int(rec["is_address_missing"]),
                "postal_code": rec["postal_code"] or "",
                "street_number": rec["street_number"] or "",
                "trailing_segment": rec["trailing_segment"] or "",
                "digit_runs": "|".join(rec["digit_runs"]) if rec["digit_runs"] else "",
            })
            count += 1
            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                buffer.clear()

        if buffer:
            writer.writerows(buffer)

    logger.info(f"  Finished {in_path.name}: {count:,} records written to {out_path}")
    return count


def normalize_all_dataset_files(
    data_dir: str | Path,
    output_dir: str | Path,
    splits: List[str] = ["train", "test"],
    chunk_size: int = 50_000,
) -> Dict[str, int]:
    """
    Batch normalize all raw TSV files in train and/or test splits.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    results = {}

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist, skipping.")
            continue

        target_split_dir = output_dir / split
        target_split_dir.mkdir(parents=True, exist_ok=True)

        for src in [1, 2, 3]:
            filename = f"{split}_source{src}.tsv"
            in_file = split_dir / filename
            if in_file.exists():
                out_file = target_split_dir / f"{split}_source{src}_normalized.tsv"
                rows = normalize_tsv_file(in_file, out_file, chunk_size=chunk_size)
                results[f"{split}_source{src}"] = rows
            else:
                logger.warning(f"File {in_file} not found, skipping.")

    return results


# ---------------------------------------------------------------------------
# High-Performance Record Normalization Helpers
# ---------------------------------------------------------------------------

def records_to_dict(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Convert DataFrame to dict of {entity_id: normalized_record_dict}.

    Uses fast itertuples iteration and normalize_entity_record (Stage 0 canonical).
    Downstream code has access to:
      - norm_name, norm_address, encoder_text (aliased as 'normalized_text')
      - country, country_canonical, source
      - is_address_missing, postal_code, street_number, trailing_segment, digit_runs
      - name_token_count, address_token_count
    """
    result = {}
    if df is None or len(df) == 0:
        return result

    cols = {col: idx for idx, col in enumerate(df.columns)}
    eid_idx = cols.get("entity_id")
    name_idx = cols.get("business_name")
    addr_idx = cols.get("business_address")
    country_idx = cols.get("country")

    for row in df.itertuples(index=False):
        eid = str(row[eid_idx]) if eid_idx is not None and pd.notna(row[eid_idx]) else ""
        name = str(row[name_idx]) if name_idx is not None and pd.notna(row[name_idx]) else ""
        addr = str(row[addr_idx]) if addr_idx is not None and pd.notna(row[addr_idx]) else ""
        country = str(row[country_idx]) if country_idx is not None and pd.notna(row[country_idx]) else ""

        rec = normalize_entity_record(name, addr, entity_id=eid, country=country)
        rec["normalized_text"] = rec["encoder_text"]  # Backward compatibility
        result[eid] = rec

    return result


# ---------------------------------------------------------------------------
# Memory-Efficient TSV Loading
# ---------------------------------------------------------------------------

def load_entity_country_map(
    source1_path: str,
    chunk_size: int = 100_000,
    canonicalize: bool = True,
) -> Dict[str, str]:
    """
    Load (entity_id -> country) mapping from S1 file.
    If canonicalize=True, country labels are mapped to canonical tokens (e.g. 'USA' -> 'us').
    """
    logger.info(f"Loading entity-country map from {source1_path}...")
    country_map = {}
    for chunk in pd.read_csv(
        source1_path, sep="\t", usecols=["entity_id", "country"],
        chunksize=chunk_size, dtype=str,
    ):
        for eid, country in zip(chunk["entity_id"], chunk["country"]):
            c_str = str(country) if pd.notna(country) else ""
            country_map[eid] = canonicalize_country(c_str) if canonicalize else c_str
    logger.info(f"  Loaded {len(country_map):,} entity-country mappings")
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
    logger.info(f"Loading records from {filepath} ({len(entity_ids):,} target IDs)...")
    chunks = []
    total_read = 0
    for chunk in pd.read_csv(filepath, sep="\t", chunksize=chunk_size, dtype=str):
        matched = chunk[chunk["entity_id"].isin(entity_ids)]
        if len(matched) > 0:
            chunks.append(matched)
        total_read += len(chunk)
    if chunks:
        result = pd.concat(chunks, ignore_index=True)
        logger.info(f"  Loaded {len(result):,} records (scanned {total_read:,} total)")
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
    c_canon = canonicalize_country(country)

    for chunk in pd.read_csv(filepath, sep="\t", chunksize=chunk_size, dtype=str):
        chunk_c = chunk["country"].fillna("").astype(str).map(canonicalize_country)
        mask = (chunk_c == c_canon) & (~chunk["entity_id"].isin(exclude_ids))
        filtered = chunk[mask]
        if len(filtered) > 0:
            all_candidates.append(filtered)

    if not all_candidates:
        return pd.DataFrame()

    combined = pd.concat(all_candidates, ignore_index=True)
    if len(combined) <= n:
        return combined
    return combined.sample(n=n, random_state=rng)


# ---------------------------------------------------------------------------
# Ground Truth Parsing
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: str) -> pd.DataFrame:
    """Load and parse ground truth file, filtering out singletons for pair training."""
    logger.info(f"Loading ground truth from {gt_path}...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    logger.info(f"  Total rows: {len(gt):,}")

    gt_with_matches = gt[gt["matched_entity_ids"].notna()].copy()
    singletons = len(gt) - len(gt_with_matches)
    logger.info(f"  With matches: {len(gt_with_matches):,}, Singletons: {singletons:,} ({singletons / len(gt) * 100:.2f}%)")

    return gt_with_matches


def parse_matched_ids(matched_str: str) -> List[str]:
    """Parse comma-separated matched entity IDs."""
    if pd.isna(matched_str) or not matched_str.strip():
        return []
    return [eid.strip() for eid in str(matched_str).split(",") if eid.strip()]


# ---------------------------------------------------------------------------
# Training Data Construction
# ---------------------------------------------------------------------------

def build_positive_pairs(
    sampled_gt: pd.DataFrame,
    s1_records: Dict[str, Dict[str, Any]],
    s2s3_records: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build positive (anchor, positive) pairs from ground truth.

    For each S1 entity, creates pairs with each of its matched S2/S3 entities.
    Returns list of dicts: {"anchor": text, "positive": text, "country": country, ...}
    """
    pairs = []
    skipped = 0

    for _, row in tqdm(sampled_gt.iterrows(), total=len(sampled_gt), desc="Building pairs"):
        s1_id = row["source1_entity_id"]
        s1_info = s1_records.get(s1_id)
        if s1_info is None:
            skipped += 1
            continue

        anchor_text = s1_info["encoder_text"]
        country = s1_info["country_canonical"]

        matched_ids = parse_matched_ids(row["matched_entity_ids"])
        for mid in matched_ids:
            match_info = s2s3_records.get(mid)
            if match_info is None:
                skipped += 1
                continue
            pairs.append({
                "anchor": anchor_text,
                "positive": match_info["encoder_text"],
                "country": country,
                "s1_id": s1_id,
                "matched_id": mid,
            })

    if skipped > 0:
        logger.warning(f"  Skipped {skipped:,} pairs due to missing records")

    return pairs


def build_evaluation_data(
    eval_gt: pd.DataFrame,
    s1_records: Dict[str, Dict[str, Any]],
    s2s3_records: Dict[str, Dict[str, Any]],
    neg_records: Dict[str, Dict[str, Any]],
    max_queries: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """
    Build evaluation data for InformationRetrievalEvaluator on held-out country.

    Returns:
        queries: {qid: query_text}
        corpus: {cid: corpus_text}
        relevant_docs: {qid: set of relevant cids}
    """
    rng = np.random.RandomState(seed)

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

        queries[s1_id] = s1_info["encoder_text"]
        relevant = set()

        matched_ids = parse_matched_ids(row["matched_entity_ids"])
        for mid in matched_ids:
            match_info = s2s3_records.get(mid)
            if match_info:
                corpus[mid] = match_info["encoder_text"]
                relevant.add(mid)

        relevant_docs[s1_id] = relevant

    # Add non-relevant candidate documents to corpus
    for neg_id, neg_info in neg_records.items():
        if neg_id not in corpus:
            corpus[neg_id] = neg_info["encoder_text"]

    logger.info(
        f"  Eval data: {len(queries):,} queries, {len(corpus):,} corpus docs, "
        f"{sum(len(v) for v in relevant_docs.values()):,} relevant pairs"
    )
    return queries, corpus, relevant_docs


def save_pairs_to_jsonl(pairs: List[Dict[str, Any]], filepath: str | Path) -> None:
    """Save pairs to line-delimited JSON for universal compatibility."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main Training Data Pipeline
# ---------------------------------------------------------------------------

def build_training_data(config: DataConfig) -> Dict[str, Any]:
    """
    Full bi-encoder data preparation pipeline.

    Steps:
    1. Load ground truth and canonical entity-country mapping
    2. Sample entities per country (balanced for anti-forgetting)
    3. Collect all required S1, S2, S3 entity IDs
    4. Load and normalize S1, S2, S3 records
    5. Build positive pairs (split by country for held-out evaluation)
    6. Build IR evaluation benchmark for held-out country gate
    7. Save datasets (Arrow/HuggingFace and JSONL)
    """
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    rng = np.random.RandomState(config.seed)

    # 1. Ground truth & country mapping
    gt = load_ground_truth(str(train_dir / "train_ground_truth.tsv"))
    country_map = load_entity_country_map(
        str(train_dir / "train_source1.tsv"),
        chunk_size=config.chunk_size,
        canonicalize=True,
    )

    gt["country"] = gt["source1_entity_id"].map(country_map)
    gt = gt.dropna(subset=["country"])

    countries = sorted(gt["country"].unique())
    logger.info(f"Canonical countries in training data: {countries}")
    for c in countries:
        logger.info(f"  {c}: {(gt['country'] == c).sum():,} entities with matches")

    # 2. Balanced sampling per country
    sampled_dfs = []
    for country in countries:
        country_gt = gt[gt["country"] == country]
        n_sample = min(config.sample_per_country, len(country_gt))
        sampled = country_gt.sample(n=n_sample, random_state=rng)
        sampled_dfs.append(sampled)
        logger.info(f"  Sampled {n_sample:,} from {country}")

    sampled_gt = pd.concat(sampled_dfs, ignore_index=True)
    logger.info(f"Total sampled entities: {len(sampled_gt):,}")

    # 3. Collect entity IDs needed
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
        f"Need: {len(s1_ids_needed):,} S1, {len(s2_ids_needed):,} S2, "
        f"{len(s3_ids_needed):,} S3 records"
    )

    # 4. Load and normalize records
    s1_df = load_records_by_ids(str(train_dir / "train_source1.tsv"), s1_ids_needed, config.chunk_size)
    s2_df = load_records_by_ids(str(train_dir / "train_source2.tsv"), s2_ids_needed, config.chunk_size)
    s3_df = load_records_by_ids(str(train_dir / "train_source3.tsv"), s3_ids_needed, config.chunk_size)

    logger.info("Normalizing records...")
    s1_records = records_to_dict(s1_df)
    s2s3_records = records_to_dict(pd.concat([s2_df, s3_df], ignore_index=True))
    logger.info(f"  Normalized: {len(s1_records):,} S1, {len(s2s3_records):,} S2/S3")

    # 5. Build positive pairs
    train_country = "us" if "us" in countries else countries[0]
    eval_country = "india" if "india" in countries else (countries[1] if len(countries) > 1 else countries[0])

    train_gt = sampled_gt[sampled_gt["country"] == train_country]
    eval_gt = sampled_gt[sampled_gt["country"] == eval_country]

    logger.info(f"Building pairs ({train_country} for train, {eval_country} for held-out eval)...")
    train_pairs = build_positive_pairs(train_gt, s1_records, s2s3_records)
    eval_pairs = build_positive_pairs(eval_gt, s1_records, s2s3_records)
    all_pairs = build_positive_pairs(sampled_gt, s1_records, s2s3_records)

    logger.info(f"  Train pairs ({train_country}): {len(train_pairs):,}")
    logger.info(f"  Eval pairs ({eval_country}): {len(eval_pairs):,}")
    logger.info(f"  All pairs (both countries): {len(all_pairs):,}")

    # 6. Build IR evaluation data for held-out country gate
    logger.info("Building IR evaluation data for held-out country gate...")
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

    # 7. Save datasets (Universal JSONL + HuggingFace Arrow)
    logger.info("Saving datasets...")

    save_pairs_to_jsonl(train_pairs, output_dir / "held_out_train_pairs.jsonl")
    save_pairs_to_jsonl(eval_pairs, output_dir / "held_out_eval_pairs.jsonl")
    save_pairs_to_jsonl(all_pairs, output_dir / "full_training_pairs.jsonl")

    if HAS_DATASETS:
        def pairs_to_dataset(pairs: List[Dict]) -> Dataset:
            return Dataset.from_dict({
                "anchor": [p["anchor"] for p in pairs],
                "positive": [p["positive"] for p in pairs],
            })

        held_out_ds = DatasetDict({
            "train": pairs_to_dataset(train_pairs),
            "eval": pairs_to_dataset(eval_pairs),
        })
        held_out_path = output_dir / "held_out_country_dataset"
        held_out_ds.save_to_disk(str(held_out_path))
        logger.info(f"  Held-out country HuggingFace dataset saved to {held_out_path}")

        full_ds = DatasetDict({"train": pairs_to_dataset(all_pairs)})
        full_path = output_dir / "full_training_dataset"
        full_ds.save_to_disk(str(full_path))
        logger.info(f"  Full training HuggingFace dataset saved to {full_path}")
    else:
        logger.warning("  'datasets' package not found; saved universal JSONL pairs instead.")

    # Save IR evaluation artifacts
    with open(output_dir / "eval_queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    with open(output_dir / "eval_corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(output_dir / "eval_relevant.json", "w", encoding="utf-8") as f:
        json.dump({k: list(v) for k, v in relevant_docs.items()}, f, ensure_ascii=False, indent=2)

    # Save split info
    split_info = {
        "train_country": train_country,
        "eval_country": eval_country,
        "countries_observed": countries,
        "train_country_entities": len(train_gt),
        "eval_country_entities": len(eval_gt),
        "train_country_pairs": len(train_pairs),
        "eval_country_pairs": len(eval_pairs),
        "has_huggingface_datasets": HAS_DATASETS,
    }
    with open(output_dir / "country_split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # Save statistics
    stats = {
        **split_info,
        "sampled_per_country": config.sample_per_country,
        "total_sampled_entities": len(sampled_gt),
        "all_pairs": len(all_pairs),
        "eval_queries": len(queries),
        "eval_corpus": len(corpus),
        "seed": config.seed,
    }
    with open(output_dir / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Data preparation complete!")
    return stats


# ---------------------------------------------------------------------------
# Self-Test Function (Fast Verification on Sample Records)
# ---------------------------------------------------------------------------

def run_self_test(test_dir: str = "./tmp_test_data") -> bool:
    """
    Run end-to-end self-test of the data preprocessing pipeline
    using synthetic records modeled after actual dataset samples.
    """
    print("=" * 72)
    print("Data Builder Pipeline -- Synthetic End-to-End Self-Test")
    print("=" * 72)

    test_path = Path(test_dir)
    test_path.mkdir(parents=True, exist_ok=True)
    train_dir = test_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Sample rows from prompt
    s1_rows = [
        {"entity_id": "S1-925783039", "business_name": "Orelee's Barbershop", "business_address": "1795 Westchester Drive, High Point, NC", "country": "US"},
        {"entity_id": "S1-773889195", "business_name": "Prime Money", "business_address": "17560 Ellis Road, Tahlequah, OK", "country": "US"},
        {"entity_id": "S1-377745466", "business_name": "B+ Retail Inc", "business_address": "1712 Montebello Avenue, Phoenix, AZ", "country": "US"},
        {"entity_id": "S1-755362802", "business_name": "Prabhav Business Center", "business_address": "797, Lake Town Block A, Kolkata, Howrah, West Bengal", "country": "India"},
        {"entity_id": "S1-851869949", "business_name": "Custom Wealth Services LLC", "business_address": "OH, Columbus, 5559 Orville Avenue", "country": "US"},
        {"entity_id": "S1-000000001", "business_name": "Boulangerie Dupont SARL", "business_address": "12 Rue de la Paix, 75002 Paris", "country": "France"},
    ]

    s2_rows = [
        {"entity_id": "S2-681193310", "business_name": "Orelees Barbershop Corp", "business_address": "1795 Westchester Dr, High Point, NC 27262", "country": "USA"},
        {"entity_id": "S2-743505751", "business_name": "Prime Money Services", "business_address": "17560 Ellis Rd, Tahlequah, OK", "country": "US"},
        {"entity_id": "S2-153058913", "business_name": "Prabhav Business Centre", "business_address": "Near SBI ATM, 797 Lake Town Block A, Kolkata 700089", "country": "India"},
        {"entity_id": "S2-999999999", "business_name": "Non-Matching Entity Inc", "business_address": "999 Random St, City, ST", "country": "India"},
    ]

    s3_rows = [
        {"entity_id": "S3-775321672", "business_name": "Orelee's Barbershop", "business_address": "##1795 Westchester Dr, High Point", "country": "US"},
        {"entity_id": "S3-679606215", "business_name": "Prabhav Business Center Pvt Ltd", "business_address": "797 Lake Town, Howrah", "country": "Bharat"},
    ]

    gt_rows = [
        {"source1_entity_id": "S1-925783039", "matched_entity_ids": "S2-681193310,S3-775321672"},
        {"source1_entity_id": "S1-773889195", "matched_entity_ids": "S2-743505751"},
        {"source1_entity_id": "S1-755362802", "matched_entity_ids": "S2-153058913,S3-679606215"},
        {"source1_entity_id": "S1-851869949", "matched_entity_ids": ""},  # Singleton
    ]

    # Write synthetic TSVs
    pd.DataFrame(s1_rows).to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_rows).to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_rows).to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(gt_rows).to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    print("[1] Testing Stage 0 streaming normalization...")
    norm_out = test_path / "stage0_normalized"
    res = normalize_all_dataset_files(test_path, norm_out, splits=["train"])
    print(f"  Stage 0 files normalized: {res}")

    print("\n[2] Testing bi-encoder data preparation pipeline...")
    cfg = DataConfig(
        data_dir=str(test_path),
        output_dir=str(test_path / "prepared_data"),
        sample_per_country=10,
        eval_sample_per_country=5,
        eval_corpus_negatives=2,
    )
    stats = build_training_data(cfg)
    print(f"  Built {stats['all_pairs']} training pairs across {stats['countries_observed']}")
    print(f"  Train country: {stats['train_country']} ({stats['train_country_pairs']} pairs)")
    print(f"  Eval country : {stats['eval_country']} ({stats['eval_country_pairs']} pairs)")

    # Validate output files exist
    prep_dir = test_path / "prepared_data"
    required_files = [
        "held_out_train_pairs.jsonl",
        "held_out_eval_pairs.jsonl",
        "full_training_pairs.jsonl",
        "eval_queries.json",
        "eval_corpus.json",
        "eval_relevant.json",
        "data_stats.json",
        "country_split_info.json",
    ]
    missing = [f for f in required_files if not (prep_dir / f).exists()]
    if missing:
        print(f"FAIL: Missing expected output files: {missing}")
        return False

    print("\n" + "=" * 72)
    print("SELF-TEST PASSED: Data preprocessing pipeline functions correctly end-to-end!")
    print("=" * 72)
    return True


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Data Preprocessing & Training Data Pipeline"
    )
    parser.add_argument("--mode", type=str, default="train_data",
                        choices=["train_data", "stage0_normalize", "all", "test"],
                        help="Pipeline operation mode")
    parser.add_argument("--data_dir", type=str, default="../../dataset",
                        help="Path to dataset directory containing train/ and test/")
    parser.add_argument("--output_dir", type=str, default="./prepared_data",
                        help="Output directory for prepared datasets or normalized files")
    parser.add_argument("--sample_per_country", type=int, default=50000,
                        help="Number of entities to sample per country")
    parser.add_argument("--eval_sample", type=int, default=2000,
                        help="Number of queries for IR evaluation")
    parser.add_argument("--eval_corpus_neg", type=int, default=5000,
                        help="Number of negative docs in eval corpus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Chunk size for reading large TSV files")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        help="Splits to process for stage0_normalize")

    args = parser.parse_args()

    if args.mode == "test":
        success = run_self_test()
        exit(0 if success else 1)

    if args.mode in ("stage0_normalize", "all"):
        stage0_out = Path(args.output_dir) / "stage0_normalized" if args.mode == "all" else Path(args.output_dir)
        logger.info(f"Running Stage 0 Normalization on splits: {args.splits}...")
        normalize_all_dataset_files(
            data_dir=args.data_dir,
            output_dir=stage0_out,
            splits=args.splits,
            chunk_size=args.chunk_size,
        )

    if args.mode in ("train_data", "all"):
        logger.info("Building Bi-Encoder Training & Evaluation Data...")
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
