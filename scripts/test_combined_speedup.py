"""
Benchmark and compare combined upgrades:
Remote MultiChannelBlocker v5 vs FastNormalizedBlocker
"""
import time
import json
import sys
from pathlib import Path
import csv

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.blocking import run_blocking
from src.fast_blocking import run_fast_blocking
import subprocess

norm_train_dir = Path("dataset_sample/normalized/train")
norm_test_dir = Path("dataset_sample/normalized/test")
gt_path = Path("dataset_sample/train/train_ground_truth.tsv")

out_baseline_train = Path("dataset_sample/bench_baseline_train")
out_baseline_test = Path("dataset_sample/bench_baseline_test")
out_fast_train = Path("dataset_sample/bench_fast_train")
out_fast_test = Path("dataset_sample/bench_fast_test")

for p in (out_baseline_train, out_baseline_test, out_fast_train, out_fast_test):
    p.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPARING REMOTE UPGRADE (MultiChannelBlocker v5) VS FAST NORMALIZED BLOCKER")
print("=" * 80)

# 1. RUN BASELINE MULTICHANNELBLOCKER v5 ON TRAIN
print("\n[1/4] Running Remote MultiChannelBlocker v5 on Train Sample...")
t0 = time.perf_counter()
res_base_train = run_blocking(
    source1_paths=[norm_train_dir / "train_source1_normalized.tsv"],
    candidate_sources=[
        norm_train_dir / "train_source2_normalized.tsv",
        norm_train_dir / "train_source3_normalized.tsv",
    ],
    output_dir=out_baseline_train,
    ground_truth_path=gt_path,
    top_k_sparse=50,
    top_k_dense=50,
    max_candidates_per_entity=100,
    similarity_floor=0.30,
    cache_index=False,
    resume=False,
    num_workers=1,
)
base_train_dur = time.perf_counter() - t0

# 2. RUN FAST NORMALIZED BLOCKER ON TRAIN
print("\n[2/4] Running FastNormalizedBlocker on Train Sample...")
t0 = time.perf_counter()
res_fast_train = run_fast_blocking(
    source1_paths=[norm_train_dir / "train_source1_normalized.tsv"],
    candidate_sources=[
        norm_train_dir / "train_source2_normalized.tsv",
        norm_train_dir / "train_source3_normalized.tsv",
    ],
    output_dir=out_fast_train,
    ground_truth_path=gt_path,
    top_k_sparse=50,
    max_candidates_per_entity=100,
    similarity_floor=0.30,
)
fast_train_dur = time.perf_counter() - t0

# 3. RUN BASELINE MULTICHANNELBLOCKER v5 ON TEST
print("\n[3/4] Running Remote MultiChannelBlocker v5 on Test Sample (with France)...")
t0 = time.perf_counter()
res_base_test = run_blocking(
    source1_paths=[norm_test_dir / "test_source1_normalized.tsv"],
    candidate_sources=[
        norm_test_dir / "test_source2_normalized.tsv",
        norm_test_dir / "test_source3_normalized.tsv",
    ],
    output_dir=out_baseline_test,
    top_k_sparse=50,
    top_k_dense=50,
    max_candidates_per_entity=100,
    similarity_floor=0.30,
    cache_index=False,
    resume=False,
    num_workers=1,
)
base_test_dur = time.perf_counter() - t0

# 4. RUN FAST NORMALIZED BLOCKER ON TEST
print("\n[4/4] Running FastNormalizedBlocker on Test Sample (with France)...")
t0 = time.perf_counter()
res_fast_test = run_fast_blocking(
    source1_paths=[norm_test_dir / "test_source1_normalized.tsv"],
    candidate_sources=[
        norm_test_dir / "test_source2_normalized.tsv",
        norm_test_dir / "test_source3_normalized.tsv",
    ],
    output_dir=out_fast_test,
    top_k_sparse=50,
    max_candidates_per_entity=100,
    similarity_floor=0.30,
)
fast_test_dur = time.perf_counter() - t0

print("\n" + "=" * 80)
print("BENCHMARK COMPARISON RESULTS")
print("=" * 80)

print(f"{'Metric':<32} | {'Remote MultiChannel v5':<22} | {'FastNormalizedBlocker':<22} | {'Speedup'}")
print("-" * 88)
print(f"{'Train Total Duration (1k S1)':<32} | {base_train_dur:18.4f} s | {fast_train_dur:18.4f} s | {base_train_dur / fast_train_dur:.2f}x")
print(f"{'Test Total Duration (1k S1)':<32}  | {base_test_dur:18.4f} s | {fast_test_dur:18.4f} s | {base_test_dur / fast_test_dur:.2f}x")

with open(out_baseline_train / "blocking_summary.json") as f:
    sb_tr = json.load(f)
with open(out_fast_train / "blocking_summary.json") as f:
    sf_tr = json.load(f)

print(f"{'Train Candidate Pairs':<32} | {sb_tr['total_candidate_pairs']:<22} | {sf_tr['total_candidate_pairs']:<22} | -")
print(f"{'Train Ground Truth Pair Recall':<32} | {sb_tr['recall_audit']['pair_recall']:21.2%} | {sf_tr['recall_audit']['pair_recall']:21.2%} | -")
print(f"{'Train Entity Full Recall':<32} | {sb_tr['recall_audit']['entity_full_recall']:21.2%} | {sf_tr['recall_audit']['entity_full_recall']:21.2%} | -")
print(f"{'Train Any-Hit Rate':<32} | {sb_tr['recall_audit']['any_hit_rate']:21.2%} | {sf_tr['recall_audit']['any_hit_rate']:21.2%} | -")
print(f"{'Train US Recall':<32} | {sb_tr['recall_audit']['recall_by_country'].get('us', 0):21.2%} | {sf_tr['recall_audit']['recall_by_country'].get('us', 0):21.2%} | -")
print(f"{'Train India Recall':<32} | {sb_tr['recall_audit']['recall_by_country'].get('india', 0):21.2%} | {sf_tr['recall_audit']['recall_by_country'].get('india', 0):21.2%} | -")

# 5. VALIDATION ON TEST OUTPUT
print("\n" + "=" * 80)
print("VALIDATING FAST ENGINE TEST OUTPUT WITH utils/validate_submission.py")
print("=" * 80)

# Create a sample matching_results.tsv from candidate_pairs.tsv
cand_file = out_fast_test / "candidate_pairs.tsv"
match_file = out_fast_test / "matching_results.tsv"
with open(cand_file, "r", encoding="utf-8") as cin, open(match_file, "w", encoding="utf-8", newline="") as mout:
    reader = csv.reader(cin, delimiter="\t")
    writer = csv.writer(mout, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
    header = next(reader)
    writer.writerow(["source1_entity_id", "matched_entity_ids"])
    for row in reader:
        s1 = row[0]
        cands = row[1].split(",") if len(row) > 1 and row[1].strip() else []
        writer.writerow([s1, cands[0] if cands else ""])

val_cmd = [
    sys.executable,
    "utils/validate_submission.py",
    "--matching", str(match_file),
    "--candidate", str(cand_file),
    "--test-dir", "dataset_sample/test"
]
val_res = subprocess.run(val_cmd, capture_output=True, text=True)
print(val_res.stdout)
if val_res.stderr:
    print(val_res.stderr)
print(f"Validation Exit Code: {val_res.returncode}")
