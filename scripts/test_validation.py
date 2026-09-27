"""
Generate sample matching_results.tsv and run validate_submission.py
"""
import csv
from pathlib import Path
import subprocess
import sys

cand_file = Path("dataset_sample/blocking_output_test/candidate_pairs.tsv")
out_matching = Path("dataset_sample/blocking_output_test/matching_results.tsv")

with open(cand_file, "r", encoding="utf-8") as cin, open(out_matching, "w", encoding="utf-8", newline="") as mout:
    reader = csv.reader(cin, delimiter="\t")
    writer = csv.writer(mout, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
    header = next(reader)
    writer.writerow(["source1_entity_id", "matched_entity_ids"])
    for row in reader:
        s1 = row[0]
        cands = row[1].split(",") if len(row) > 1 and row[1].strip() else []
        match = cands[0] if cands else ""
        writer.writerow([s1, match])

print("Created sample matching_results.tsv successfully!")

# Run validate_submission.py
cmd = [
    sys.executable,
    "utils/validate_submission.py",
    "--matching", str(out_matching),
    "--candidate", str(cand_file),
    "--test-dir", "dataset_sample/test",
]

res = subprocess.run(cmd, capture_output=True, text=True)
print("Validator stdout:\n", res.stdout)
if res.stderr:
    print("Validator stderr:\n", res.stderr)
print(f"Exit code: {res.returncode}")
