"""
Benchmark pure in-memory query throughput: MultiChannelBlocker v5 vs FastNormalizedBlocker
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.blocking import MultiChannelBlocker, BlockingRecord, read_tsv_records
from src.fast_blocking import FastNormalizedBlocker
import polars as pl

# 1. Setup MultiChannelBlocker v5
print("Setting up MultiChannelBlocker v5...")
m = MultiChannelBlocker()
for r in read_tsv_records([
    Path("dataset_sample/normalized/train/train_source2_normalized.tsv"),
    Path("dataset_sample/normalized/train/train_source3_normalized.tsv"),
]):
    m.index_candidate(BlockingRecord.from_row(
        entity_id=r["entity_id"],
        name=r.get("norm_name", r.get("business_name", "")),
        address=r.get("norm_address", r.get("business_address", "")),
        country=r.get("canonical_country", r.get("country", "")),
        postal_code=r.get("postal_code"),
        street_number=r.get("street_number"),
        trailing_segment=r.get("trailing_segment"),
        is_address_missing=bool(r.get("is_address_missing", 0)),
    ))
m.build_idf_tables()

# 2. Setup FastNormalizedBlocker
print("Setting up FastNormalizedBlocker...")
f = FastNormalizedBlocker()
f.index_normalized_file(Path("dataset_sample/normalized/train/train_source2_normalized.tsv"))
f.index_normalized_file(Path("dataset_sample/normalized/train/train_source3_normalized.tsv"))
f.build_idf_tables()

# Load 1,000 real S1 records
s1_records_m = []
for r in read_tsv_records([Path("dataset_sample/normalized/train/train_source1_normalized.tsv")]):
    s1_records_m.append(BlockingRecord.from_row(
        entity_id=r["entity_id"],
        name=r.get("norm_name", r.get("business_name", "")),
        address=r.get("norm_address", r.get("business_address", "")),
        country=r.get("canonical_country", r.get("country", "")),
        postal_code=r.get("postal_code"),
        street_number=r.get("street_number"),
        trailing_segment=r.get("trailing_segment"),
        is_address_missing=bool(r.get("is_address_missing", 0)),
    ))

df_s1 = pl.read_csv("dataset_sample/normalized/train/train_source1_normalized.tsv", separator="\t")
s1_rows_f = df_s1.to_dicts()

N = len(s1_records_m)
REPEATS = 3
TOTAL_QUERIES = N * REPEATS

print(f"\nBenchmarking {TOTAL_QUERIES:,} in-memory queries across real S1 records...")

# Benchmark MultiChannelBlocker v5
t0 = time.perf_counter()
for _ in range(REPEATS):
    for rec in s1_records_m:
        m.generate_candidates_for_record(rec)
dur_m = time.perf_counter() - t0
qps_m = TOTAL_QUERIES / dur_m

# Benchmark FastNormalizedBlocker
t0 = time.perf_counter()
for _ in range(REPEATS):
    for r in s1_rows_f:
        f.query_record(
            eid=r["entity_id"],
            country=r.get("country_canonical") or "",
            norm_name=r.get("norm_name") or "",
            norm_address=r.get("norm_address") or "",
            postal_code=r.get("postal_code") or "",
            street_number=r.get("street_number") or "",
            trailing_segment=r.get("trailing_segment") or "",
            is_address_missing=bool(r.get("is_address_missing") or 0),
        )
dur_f = time.perf_counter() - t0
qps_f = TOTAL_QUERIES / dur_f

print("=" * 70)
print(f"{'Engine':<28} | {'Duration (3,000 q)':<18} | {'Throughput'}")
print("-" * 70)
print(f"{'MultiChannelBlocker v5':<28} | {dur_m:16.3f} s | {qps_m:8.1f} queries/s")
print(f"{'FastNormalizedBlocker':<28} | {dur_f:16.3f} s | {qps_f:8.1f} queries/s")
print(f"Speedup: {dur_m / dur_f:.2f}x faster in pure query execution!")
print("=" * 70)
