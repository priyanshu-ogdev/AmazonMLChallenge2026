"""
Execute and benchmark Stage 1 Blocking on train and test samples.
"""
import time
import json
import sys
from pathlib import Path
import csv

# Add code/business_entity_resolution to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.blocking import run_blocking

norm_train_dir = Path("dataset_sample/normalized/train")
norm_test_dir = Path("dataset_sample/normalized/test")
gt_path = Path("dataset_sample/train/train_ground_truth.tsv")

out_train = Path("dataset_sample/blocking_output_train")
out_test = Path("dataset_sample/blocking_output_test")
out_train.mkdir(parents=True, exist_ok=True)
out_test.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("1. RUNNING BLOCKING ON TRAIN SAMPLE (WITH GROUND TRUTH RECALL AUDIT)")
print("=" * 70)

t0 = time.perf_counter()
run_blocking(
    source1_paths=[norm_train_dir / "train_source1_normalized.tsv"],
    candidate_sources=[
        norm_train_dir / "train_source2_normalized.tsv",
        norm_train_dir / "train_source3_normalized.tsv",
    ],
    output_dir=out_train,
    ground_truth_path=gt_path,
    top_k_sparse=50,
    top_k_dense=50,
    max_candidates_per_entity=100,
    similarity_floor=0.30,
    cache_index=False,
    resume=False,
    num_workers=1,
)
train_blocking_dur = time.perf_counter() - t0

print("\n" + "=" * 70)
print("2. RUNNING BLOCKING ON TEST SAMPLE (1,000 TEST ENTITIES ACROSS US/INDIA/FRANCE)")
print("=" * 70)

t0 = time.perf_counter()
run_blocking(
    source1_paths=[norm_test_dir / "test_source1_normalized.tsv"],
    candidate_sources=[
        norm_test_dir / "test_source2_normalized.tsv",
        norm_test_dir / "test_source3_normalized.tsv",
    ],
    output_dir=out_test,
    top_k_sparse=50,
    top_k_dense=50,
    max_candidates_per_entity=100,
    similarity_floor=0.30,
    cache_index=False,
    resume=False,
    num_workers=1,
)
test_blocking_dur = time.perf_counter() - t0

# Read and print summary
print("\n" + "=" * 70)
print("BLOCKING EXECUTION SUMMARY")
print("=" * 70)

with open(out_train / "blocking_summary.json", "r", encoding="utf-8") as f:
    train_summary = json.load(f)

with open(out_test / "blocking_summary.json", "r", encoding="utf-8") as f:
    test_summary = json.load(f)

print(f"Train Blocking Duration : {train_blocking_dur:.3f} s")
print(f"  Total S1 Entities     : {train_summary['total_source1_entities']}")
print(f"  Total Candidate Pairs : {train_summary['total_candidate_pairs']}")
print(f"  Singletons (0 cands)  : {train_summary['singletons_with_zero_candidates']}")
print(f"  Mean Candidates/S1    : {train_summary['candidates_per_s1']['mean']}")
print(f"  P90 Candidates/S1     : {train_summary['candidates_per_s1']['p90']}")
print(f"  Max Candidates/S1     : {train_summary['candidates_per_s1']['max']}")
if "recall_audit" in train_summary:
    ra = train_summary["recall_audit"]
    for k, v in ra.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.2%}")
        else:
            print(f"  {k:25s}: {v}")

print(f"\nTest Blocking Duration  : {test_blocking_dur:.3f} s")
print(f"  Total S1 Entities     : {test_summary['total_source1_entities']}")
print(f"  Total Candidate Pairs : {test_summary['total_candidate_pairs']}")
print(f"  Singletons (0 cands)  : {test_summary['singletons_with_zero_candidates']}")
print(f"  Mean Candidates/S1    : {test_summary['candidates_per_s1']['mean']}")
print(f"  P90 Candidates/S1     : {test_summary['candidates_per_s1']['p90']}")
print(f"  Max Candidates/S1     : {test_summary['candidates_per_s1']['max']}")

# Inspect sample candidates produced
def show_sample_pairs(pair_file, n=5):
    print(f"\n--- Sample from {pair_file.parent.name}/{pair_file.name} ---")
    with open(pair_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i > n:
                break
            print(f"  {row[0]:15s} -> {row[1][:70]}..." if len(row) > 1 and len(row[1]) > 70 else f"  {row[0]:15s} -> {row[1] if len(row) > 1 else ''}")

show_sample_pairs(out_train / "candidate_pairs.tsv", 5)
show_sample_pairs(out_test / "candidate_pairs.tsv", 5)
