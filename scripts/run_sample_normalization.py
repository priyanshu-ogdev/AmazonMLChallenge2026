"""
Run Stage 0 normalization on dataset_sample files and record metrics.
"""
import time
import json
import sys
from pathlib import Path
import csv

# Add code/business_entity_resolution to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.data_builder import normalize_tsv_file, STAGE0_TSV_COLUMNS

sample_train_dir = Path("dataset_sample/train")
sample_test_dir = Path("dataset_sample/test")

norm_train_dir = Path("dataset_sample/normalized/train")
norm_test_dir = Path("dataset_sample/normalized/test")
norm_train_dir.mkdir(parents=True, exist_ok=True)
norm_test_dir.mkdir(parents=True, exist_ok=True)

files_to_normalize = [
    (sample_train_dir / "train_source1.tsv", norm_train_dir / "train_source1_normalized.tsv"),
    (sample_train_dir / "train_source2.tsv", norm_train_dir / "train_source2_normalized.tsv"),
    (sample_train_dir / "train_source3.tsv", norm_train_dir / "train_source3_normalized.tsv"),
    (sample_test_dir / "test_source1.tsv", norm_test_dir / "test_source1_normalized.tsv"),
    (sample_test_dir / "test_source2.tsv", norm_test_dir / "test_source2_normalized.tsv"),
    (sample_test_dir / "test_source3.tsv", norm_test_dir / "test_source3_normalized.tsv"),
]

timings = {}
total_records = 0
t_start = time.perf_counter()

for in_f, out_f in files_to_normalize:
    t0 = time.perf_counter()
    n = normalize_tsv_file(in_f, out_f, chunk_size=10_000)
    dur = time.perf_counter() - t0
    timings[in_f.name] = {
        "records": n,
        "seconds": round(dur, 4),
        "records_per_second": round(n / dur, 1) if dur > 0 else 0
    }
    total_records += n

total_dur = time.perf_counter() - t_start
summary = {
    "total_records": total_records,
    "total_seconds": round(total_dur, 4),
    "overall_throughput_rec_per_sec": round(total_records / total_dur, 1),
    "file_breakdown": timings
}

with open("dataset_sample/normalization_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print(f"STAGE 0 NORMALIZATION COMPLETED: {total_records} records in {total_dur:.3f}s ({total_records/total_dur:.1f} rec/s)")
print("="*60)
for fname, stats in timings.items():
    print(f"  {fname:25s}: {stats['records']:5d} rows in {stats['seconds']:.3f}s -> {stats['records_per_second']:.1f} rows/s")

# Let's inspect 3 sample records from train and test to demonstrate the transformations
def inspect_samples(norm_path, n=3):
    print(f"\n--- Sample Transformed Records from {norm_path.name} ---")
    with open(norm_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i >= n:
                break
            print(f"[{i+1}] ID: {row['entity_id']} | Country: {row['country']} (canonical: {row['country_canonical']})")
            print(f"    Raw Name     : {row['raw_name']}")
            print(f"    Norm Name    : {row['norm_name']}")
            print(f"    Raw Addr     : {row['raw_address']}")
            print(f"    Norm Addr    : {row['norm_address']}")
            print(f"    Encoder Text : {row['encoder_text']}")
            print(f"    Postal/PIN   : {row['postal_code']} | Street #: {row['street_number']} | Missing Addr: {row['is_address_missing']}")

inspect_samples(norm_train_dir / "train_source1_normalized.tsv", 2)
inspect_samples(norm_test_dir / "test_source1_normalized.tsv", 3)
