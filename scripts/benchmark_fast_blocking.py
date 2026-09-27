"""
Benchmark comparing Baseline MultiChannelBlocker vs. FastNormalizedBlocker.
"""
import time
import sys
import csv
import json
from pathlib import Path

# Add code/business_entity_resolution to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.blocking import MultiChannelBlocker, BlockingRecord, read_tsv_records
from src.fast_blocking import FastNormalizedBlocker
import polars as pl

norm_train_dir = Path("dataset_sample/normalized/train")
s1_path = norm_train_dir / "train_source1_normalized.tsv"
s2_path = norm_train_dir / "train_source2_normalized.tsv"
s3_path = norm_train_dir / "train_source3_normalized.tsv"
gt_path = Path("dataset_sample/train/train_ground_truth.tsv")

# Load Ground Truth for recall evaluation
ground_truth = {}
with open(gt_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        s1_id = row["source1_entity_id"]
        matches = {m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()}
        ground_truth[s1_id] = matches

total_gt_pairs = sum(len(m) for m in ground_truth.values())

print("=" * 75)
print("STAGE 1 NORMALIZED BLOCKING HEAD-TO-HEAD BENCHMARK")
print("=" * 75)
print(f"S1 Entities to Query: 1,000 | Candidate Records to Index: 3,517")
print(f"Ground Truth Positive Pairs: {total_gt_pairs}")

# ---------------------------------------------------------------------------
# 1. BASELINE MultiChannelBlocker
# ---------------------------------------------------------------------------
print("\n--- 1. Running Baseline MultiChannelBlocker ---")
base_blocker = MultiChannelBlocker()

t0 = time.perf_counter()
# Indexing
for row in read_tsv_records([s2_path, s3_path]):
    rec = BlockingRecord.from_row(
        entity_id=row["entity_id"],
        name=row["business_name"],
        address=row["business_address"],
        country=row["country"],
        norm_name=row.get("norm_name"),
        norm_address=row.get("norm_address"),
        canonical_country=row.get("canonical_country"),
        postal_code=row.get("postal_code"),
        street_number=row.get("street_number"),
        trailing_segment=row.get("trailing_segment"),
        is_address_missing=row.get("is_address_missing"),
    )
    base_blocker.index_candidate(rec)
base_blocker.build_idf_tables()
base_index_time = time.perf_counter() - t0

# Querying
t0 = time.perf_counter()
base_pairs = 0
base_recovered = 0
s1_records = list(read_tsv_records([s1_path]))

for row in s1_records:
    s1_rec = BlockingRecord.from_row(
        entity_id=row["entity_id"],
        name=row["business_name"],
        address=row["business_address"],
        country=row["country"],
        norm_name=row.get("norm_name"),
        norm_address=row.get("norm_address"),
        canonical_country=row.get("canonical_country"),
        postal_code=row.get("postal_code"),
        street_number=row.get("street_number"),
        trailing_segment=row.get("trailing_segment"),
        is_address_missing=row.get("is_address_missing"),
    )
    cands = base_blocker.generate_candidates_for_record(s1_rec)
    base_pairs += len(cands)
    true_matches = ground_truth.get(s1_rec.entity_id, set())
    found_cids = {c[1] for c in cands}
    base_recovered += len(true_matches & found_cids)

base_query_time = time.perf_counter() - t0
base_total_time = base_index_time + base_query_time
base_recall = base_recovered / total_gt_pairs if total_gt_pairs > 0 else 0.0

print(f"Baseline Indexing Time : {base_index_time:.4f} s ({3517 / base_index_time:.1f} rec/s)")
print(f"Baseline Querying Time : {base_query_time:.4f} s ({len(s1_records) / base_query_time:.1f} queries/s)")
print(f"Baseline Total Time    : {base_total_time:.4f} s")
print(f"Baseline Candidate Pairs: {base_pairs:,} | Pair Recall: {base_recall:.2%}")

# ---------------------------------------------------------------------------
# 2. OPTIMIZED FastNormalizedBlocker
# ---------------------------------------------------------------------------
print("\n--- 2. Running Optimized FastNormalizedBlocker ---")
fast_blocker = FastNormalizedBlocker()

t0 = time.perf_counter()
fast_blocker.index_normalized_file(s2_path)
fast_blocker.index_normalized_file(s3_path)
fast_blocker.build_idf_tables()
fast_index_time = time.perf_counter() - t0

# Querying via Polars dataframe
t0 = time.perf_counter()
df_s1 = pl.read_csv(
    s1_path,
    separator="\t",
    columns=[
        "entity_id", "country_canonical", "norm_name", "norm_address",
        "postal_code", "street_number", "is_address_missing"
    ],
    schema_overrides={
        "entity_id": pl.String,
        "country_canonical": pl.String,
        "norm_name": pl.String,
        "norm_address": pl.String,
        "postal_code": pl.String,
        "street_number": pl.String,
        "is_address_missing": pl.Int32,
    }
)

s1_eids = df_s1["entity_id"].to_list()
s1_countries = df_s1["country_canonical"].fill_null("").to_list()
s1_names = df_s1["norm_name"].fill_null("").to_list()
s1_addrs = df_s1["norm_address"].fill_null("").to_list()
s1_postals = df_s1["postal_code"].fill_null("").to_list()
s1_streets = df_s1["street_number"].fill_null("").to_list()
s1_misses = [bool(m) for m in df_s1["is_address_missing"].fill_null(0).to_list()]

fast_pairs = 0
fast_recovered = 0

for i in range(len(s1_eids)):
    eid = s1_eids[i]
    cands = fast_blocker.query_record(
        eid=eid,
        country=s1_countries[i],
        norm_name=s1_names[i],
        norm_address=s1_addrs[i],
        postal_code=s1_postals[i],
        street_number=s1_streets[i],
        is_address_missing=s1_misses[i],
    )
    fast_pairs += len(cands)
    true_matches = ground_truth.get(eid, set())
    found_cids = {c[1] for c in cands}
    fast_recovered += len(true_matches & found_cids)

fast_query_time = time.perf_counter() - t0
fast_total_time = fast_index_time + fast_query_time
fast_recall = fast_recovered / total_gt_pairs if total_gt_pairs > 0 else 0.0

print(f"Fast Indexing Time     : {fast_index_time:.4f} s ({3517 / fast_index_time:.1f} rec/s)")
print(f"Fast Querying Time     : {fast_query_time:.4f} s ({len(s1_eids) / fast_query_time:.1f} queries/s)")
print(f"Fast Total Time        : {fast_total_time:.4f} s")
print(f"Fast Candidate Pairs   : {fast_pairs:,} | Pair Recall: {fast_recall:.2%}")

# ---------------------------------------------------------------------------
# 3. SPEEDUP COMPARISON
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("BENCHMARK COMPARISON & SPEEDUP SUMMARY")
print("=" * 75)
idx_speedup = base_index_time / fast_index_time if fast_index_time > 0 else 1.0
query_speedup = base_query_time / fast_query_time if fast_query_time > 0 else 1.0
total_speedup = base_total_time / fast_total_time if fast_total_time > 0 else 1.0

print(f"Indexing Phase Speedup : {idx_speedup:.2f}x faster ({base_index_time:.3f}s -> {fast_index_time:.3f}s)")
print(f"Querying Phase Speedup : {query_speedup:.2f}x faster ({base_query_time:.3f}s -> {fast_query_time:.3f}s)")
print(f"End-to-End Speedup     : {total_speedup:.2f}x faster ({base_total_time:.3f}s -> {fast_total_time:.3f}s)")
print(f"Recall Preserved       : Baseline {base_recall:.2%} vs Fast {fast_recall:.2%}")
print("=" * 75)
