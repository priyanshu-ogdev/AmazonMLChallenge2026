"""
Vectorized Array-Accumulator Prototype for Stage 1 Blocking Queries
"""
import time
import numpy as np

# Simulate a country partition with 500,000 candidate entities
N_CANDIDATES = 500_000
scratch = np.zeros(N_CANDIDATES, dtype=np.float32)

# Simulate 15 active n-grams with varying posting list sizes (1,000 to 10,000 candidates each)
np.random.seed(42)
posting_lists = [
    np.random.randint(0, N_CANDIDATES, size=np.random.randint(1000, 10000), dtype=np.uint32)
    for _ in range(15)
]
weights = np.random.uniform(0.5, 2.5, size=15).astype(np.float32)

N_QUERIES = 500
t0 = time.perf_counter()

for _ in range(N_QUERIES):
    touched = []
    # 1. Accumulate weights directly into scratch buffer
    for post, w in zip(posting_lists, weights):
        scratch[post] += w
        touched.append(post)
    
    # 2. Vectorized top-50 selection via C-accelerated argpartition
    # Only search over touched postings (or threshold filtering)
    top50_idx = np.argpartition(-scratch, 50)[:50]
    top50_scores = scratch[top50_idx]
    
    # 3. Clean reset: only reset touched indices (O(postings) instead of O(N_CANDIDATES))
    for post in touched:
        scratch[post] = 0.0

dur = time.perf_counter() - t0
rate = N_QUERIES / dur
postings_processed = sum(len(p) for p in posting_lists) * N_QUERIES

print("=" * 65)
print("VECTORIZED SCRATCH ACCUMULATOR MICROBENCHMARK")
print("=" * 65)
print(f"Executed {N_QUERIES} queries over {N_CANDIDATES:,} candidate pool")
print(f"Total Postings Processed: {postings_processed:,}")
print(f"Time Taken              : {dur:.4f} s")
print(f"Query Throughput        : {rate:.1f} queries / sec")
print(f"Posting Throughput      : {postings_processed / dur / 1e6:.2f} Million postings / sec")
print(f"Latency Per Query       : {dur / N_QUERIES * 1000:.3f} ms")
print("=" * 65)
