import sys
import os

# Add src to python path
sys.path.append(os.path.abspath('code/business_entity_resolution'))
from src.scoring import compute_entity_f05, macro_f05

print("--- COMPETITION SPECIFICATION EXAMPLE ---")
print("Model predicts: [S2-00047, S2-00193, S3-00812] (3 predictions)")
print("Ground truth:   [S2-00047, S3-00812] (2 actual matches)")
print("True Positives: 2 (S2-00047, S3-00812)")
print("")

f05 = compute_entity_f05(pred_count=3, actual_count=2, tp=2)
print(f"Our Codebase Output F_0.5: {f05:.3f}")
print("Competition Spec F_0.5:    0.714")

if abs(f05 - 0.714) < 0.001:
    print("\n✅ VERIFIED: Codebase scoring logic perfectly matches the competition formula!")

print("\n--- SINGLETON EDGE CASE ---")
print("Model predicts: [] (0 predictions)")
print("Ground truth:   [] (0 actual matches)")
f05_singleton = compute_entity_f05(pred_count=0, actual_count=0, tp=0)
print(f"Our Codebase Output F_0.5: {f05_singleton:.1f}")
print("Competition Spec F_0.5:    1.0")

if f05_singleton == 1.0:
    print("\n✅ VERIFIED: Singleton correctly rewarded with 1.0 credit per spec rules.")
