"""
Extract a 1,000-record sample for train and test splits.
"""
import csv
from pathlib import Path

train_dir = Path("dataset/train")
test_dir = Path("dataset/test")

sample_train_dir = Path("dataset_sample/train")
sample_test_dir = Path("dataset_sample/test")
sample_train_dir.mkdir(parents=True, exist_ok=True)
sample_test_dir.mkdir(parents=True, exist_ok=True)

def slice_file(in_path: Path, out_path: Path, n: int = 1000):
    with open(in_path, "r", encoding="utf-8", errors="replace") as infile, \
         open(out_path, "w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(infile, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer = csv.writer(outfile, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        count = 0
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue
            if count >= n:
                break
            writer.writerow(row)
            count += 1
    print(f"Wrote {out_path} with {count} records")
    return count

# Slicing test set (exactly 1000 from each source)
slice_file(test_dir / "test_source1.tsv", sample_test_dir / "test_source1.tsv", 1000)
slice_file(test_dir / "test_source2.tsv", sample_test_dir / "test_source2.tsv", 1000)
slice_file(test_dir / "test_source3.tsv", sample_test_dir / "test_source3.tsv", 1000)

# Slicing train source1 (1000 records)
s1_count = slice_file(train_dir / "train_source1.tsv", sample_train_dir / "train_source1.tsv", 1000)

# Read S1 IDs from sample
s1_ids = set()
with open(sample_train_dir / "train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        s1_ids.add(row["entity_id"])

print(f"Loaded {len(s1_ids)} S1 sample IDs")

# Find ground truth for these S1 IDs
matched_s2_ids = set()
matched_s3_ids = set()
s1_to_matches = {}

with open(train_dir / "train_ground_truth.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        s1_id = row["source1_entity_id"]
        if s1_id in s1_ids:
            matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
            s1_to_matches[s1_id] = matches
            for m in matches:
                if m.startswith("S2-"):
                    matched_s2_ids.add(m)
                elif m.startswith("S3-"):
                    matched_s3_ids.add(m)

print(f"Ground truth matches for 1000 S1: {len(matched_s2_ids)} S2 matches, {len(matched_s3_ids)} S3 matches")

# Write sample ground truth
with open(sample_train_dir / "train_ground_truth.tsv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
    writer.writerow(["source1_entity_id", "matched_entity_ids"])
    for s1_id in s1_ids:
        matches = s1_to_matches.get(s1_id, [])
        writer.writerow([s1_id, ",".join(matches)])

# For train S2 and S3:
# We should include the matched S2 and S3 records so blocking recall is testable,
# and top up with distractors up to 1000 records each!
def create_candidate_sample(in_path: Path, out_path: Path, matched_ids: set, target_total: int = 1000):
    matched_rows = {}
    distractor_rows = []
    
    with open(in_path, "r", encoding="utf-8", errors="replace") as infile:
        reader = csv.reader(infile, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            eid = row[0]
            if eid in matched_ids:
                matched_rows[eid] = row
            elif len(distractor_rows) < target_total:
                distractor_rows.append(row)
            
            # If we found all matched rows and have enough distractors, we can stop early
            if len(matched_rows) == len(matched_ids) and len(matched_rows) + len(distractor_rows) >= target_total:
                break
                
    # Combine matched + distractors up to target_total (or at least all matched + distractors)
    final_rows = list(matched_rows.values())
    needed = max(0, target_total - len(final_rows))
    final_rows.extend(distractor_rows[:needed])
    
    with open(out_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(header)
        for r in final_rows:
            writer.writerow(r)
            
    print(f"Wrote {out_path} with {len(final_rows)} records ({len(matched_rows)} matched, {len(final_rows) - len(matched_rows)} distractors)")

create_candidate_sample(train_dir / "train_source2.tsv", sample_train_dir / "train_source2.tsv", matched_s2_ids, 1000)
create_candidate_sample(train_dir / "train_source3.tsv", sample_train_dir / "train_source3.tsv", matched_s3_ids, 1000)
print("Dataset sampling complete!")
