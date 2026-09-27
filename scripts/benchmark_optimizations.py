"""
Empirical benchmark comparing current Python baseline vs. optimized vectorized architectures.
"""
import time
import sys
from pathlib import Path
import numpy as np

# Add code/business_entity_resolution to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.blocking import MultiChannelBlocker, BlockingRecord, read_tsv_records
from src.normalize import normalize_entity_record

print("=" * 70)
print("EMPIRICAL BENCHMARK: CURRENT PYTHON BASELINE VS 100x OPTIMIZED ARCHITECTURES")
print("=" * 70)

# 1. Benchmark Normalization Speed
print("\n--- 1. NORMALIZATION BENCHMARK ---")
sample_path = Path("dataset_sample/train/train_source1.tsv")
rows = list(read_tsv_records([sample_path]))[:1000]

# Baseline normalize_entity_record
t0 = time.perf_counter()
for r in rows:
    _ = normalize_entity_record(r["business_name"], r["business_address"], r["entity_id"], r["country"])
t_baseline_norm = time.perf_counter() - t0
rate_baseline_norm = len(rows) / t_baseline_norm
print(f"Current Python normalize_entity_record: {len(rows)} records in {t_baseline_norm:.4f}s ({rate_baseline_norm:.1f} rec/s)")
print(f"Extrapolated 26.4M full challenge dataset: {26_400_000 / rate_baseline_norm / 60:.1f} minutes")

# 2. Benchmark Posting List Traversal: Python Dict vs Flat Array
print("\n--- 2. INVERTED INDEX POSTING ACCUMULATION BENCHMARK ---")
# Simulate a term with 50,000 postings (typical in 10M record database)
N_POSTINGS = 50_000
python_postings = [f"S2-{i:08d}" for i in range(N_POSTINGS)]
uint32_postings = np.arange(N_POSTINGS, dtype=np.uint32)
N_QUERIES = 200

# Baseline: Python defaultdict(float) accumulation with string keys
t0 = time.perf_counter()
for _ in range(N_QUERIES):
    acc = {}
    w = 1.45
    for cid in python_postings:
        acc[cid] = acc.get(cid, 0.0) + w
t_baseline_postings = time.perf_counter() - t0
print(f"Baseline Python dict string accumulation: {N_QUERIES} queries over {N_POSTINGS} postings in {t_baseline_postings:.4f}s ({N_QUERIES * N_POSTINGS / t_baseline_postings / 1e6:.2f}M postings/s)")

# Optimized 1: NumPy flat dense accumulator array
t0 = time.perf_counter()
dense_acc = np.zeros(N_POSTINGS, dtype=np.float32)
for _ in range(N_QUERIES):
    dense_acc.fill(0)
    w = np.float32(1.45)
    dense_acc[uint32_postings] += w
t_numpy_postings = time.perf_counter() - t0
print(f"Vectorized NumPy uint32 accumulation   : {N_QUERIES} queries over {N_POSTINGS} postings in {t_numpy_postings:.4f}s ({N_QUERIES * N_POSTINGS / t_numpy_postings / 1e6:.2f}M postings/s)")
print(f"Speedup from Integer Dictionary Encoding + Flat Array: {t_baseline_postings / t_numpy_postings:.1f}x speedup!")

# 3. Benchmark String Matching: Levenshtein / Jaccard
print("\n--- 3. FUZZY STRING MATCHING BENCHMARK (RAPIDFUZZ VS PYTHON) ---")
import difflib
import rapidfuzz

s1_names = [r["business_name"] for r in rows[:100]]
s2_names = [r["business_name"] for r in rows[100:200]]

# Baseline difflib SequenceMatcher
t0 = time.perf_counter()
for s1, s2 in zip(s1_names, s2_names):
    _ = difflib.SequenceMatcher(None, s1, s2).ratio()
t_difflib = time.perf_counter() - t0

# RapidFuzz C++ SIMD
t0 = time.perf_counter()
for s1, s2 in zip(s1_names, s2_names):
    _ = rapidfuzz.distance.Levenshtein.normalized_similarity(s1, s2)
t_rapidfuzz = time.perf_counter() - t0

print(f"Baseline difflib SequenceMatcher (pure Python): {t_difflib*1000:.2f} ms")
print(f"RapidFuzz C++ SIMD Levenshtein               : {t_rapidfuzz*1000:.2f} ms")
print(f"Speedup: {t_difflib / t_rapidfuzz:.1f}x speedup!")

print("\n" + "=" * 70)
