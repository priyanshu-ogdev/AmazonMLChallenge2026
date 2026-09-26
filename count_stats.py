import pandas as pd

def count_stats():
    s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
    s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
    s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
    gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', keep_default_na=False)
    
    print(f"Source 1 Entities: {len(s1):,}")
    print(f"Source 2 Entities: {len(s2):,}")
    print(f"Source 3 Entities: {len(s3):,}")
    print(f"Total Candidate Pool (S2 + S3): {len(s2) + len(s3):,}")
    
    total_pairs = 0
    singletons = 0
    for val in gt['matched_entity_ids']:
        val = str(val).strip()
        if not val:
            singletons += 1
        else:
            total_pairs += len(val.split(','))
            
    print(f"Total Ground Truth Matches (Positive Pairs): {total_pairs:,}")
    print(f"Total Singletons (Zero matches): {singletons:,}")
    
    brute_force = len(s1) * (len(s2) + len(s3))
    print(f"Brute Force Pairs (S1 x (S2+S3)): {brute_force:,}")

if __name__ == '__main__':
    count_stats()
