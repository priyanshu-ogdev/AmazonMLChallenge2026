"""
Benchmark ProcessPoolExecutor (separate processes with their own GIL) vs ThreadPoolExecutor.
"""
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing

# Add code/business_entity_resolution to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.fast_blocking import FastNormalizedBlocker
import polars as pl

norm_train_dir = Path("dataset_sample/normalized/train")
s1_path = norm_train_dir / "train_source1_normalized.tsv"
s2_path = norm_train_dir / "train_source2_normalized.tsv"
s3_path = norm_train_dir / "train_source3_normalized.tsv"

_global_blocker = None

def init_worker():
    global _global_blocker
    _global_blocker = FastNormalizedBlocker()
    _global_blocker.index_normalized_file(s2_path)
    _global_blocker.index_normalized_file(s3_path)
    _global_blocker.build_idf_tables()

def process_chunk_mp(chunk):
    global _global_blocker
    return sum(len(_global_blocker.query_record(*r)) for r in chunk)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    df_s1 = pl.read_csv(s1_path, separator="\t")
    s1_rows = [
        (
            r["entity_id"],
            r["country_canonical"] or "",
            r["norm_name"] or "",
            r["norm_address"] or "",
            r["postal_code"] or "",
            r["street_number"] or "",
            bool(r["is_address_missing"]),
        )
        for r in df_s1.iter_rows(named=True)
    ]

    print("=" * 60)
    print("PROCESS POOL BENCHMARK (TRUE MULTI-CORE GIL BYPASS)")
    print("=" * 60)

    for n_workers in [1, 2, 4, 8]:
        chunk_size = (len(s1_rows) + n_workers - 1) // n_workers
        chunks = [s1_rows[i:i + chunk_size] for i in range(0, len(s1_rows), chunk_size)]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as pool:
            total_cands = sum(pool.map(process_chunk_mp, chunks))
        dur = time.perf_counter() - t0
        rate = len(s1_rows) / dur
        print(f"Workers: {n_workers:2d} -> Time: {dur:.4f} s | Rate: {rate:7.1f} queries/s | Total pairs: {total_cands}")
