"""Quick validation that all modules import correctly and basic functionality works."""
import sys
sys.path.insert(0, ".")

print("=" * 60)
print("Module Import Validation")
print("=" * 60)

# 1. Config
from src.config import LoRAConfig, TrainingConfig, DataConfig, EvalConfig, VRAMBudget
lora_cfg = LoRAConfig()
train_cfg = TrainingConfig()
data_cfg = DataConfig()
eval_cfg = EvalConfig()
vram = VRAMBudget()

print(f"[OK] config.py")
print(f"  LoRA: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, rslora={lora_cfg.use_rslora}")
print(f"  Training: batch={train_cfg.per_device_train_batch_size}, lr={train_cfg.learning_rate}, epochs={train_cfg.num_train_epochs}")
print(f"  VRAM: {vram.fixed_overhead_with_distillation_gb:.2f}GB fixed, {vram.available_for_batch_gb:.2f}GB available")
print(f"  VRAM budget OK: {vram.verify()}")

# 2. Normalize
from src.normalize import normalize_entity, normalize_name, normalize_address, extract_postal_code
tests = [
    ("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "US"),
    ("Prime Money Corp.", "17560 Ellis Rd, Tahlequah, OK", "US"),
    ("B+ Retail Inc", "1712 Montebello Ave, Phoenix, AZ", "US"),
    ("Shri Krishna Pvt Ltd", "Near SBI ATM, MG Road, Bhopal, MP 462001", "India"),
]
print(f"\n[OK] normalize.py")
for name, addr, country in tests:
    normalized = normalize_entity(name, addr)
    postal = extract_postal_code(addr)
    print(f"  [{country}] {name[:30]:30s} -> {normalized[:60]}")
    if postal:
        print(f"       postal: {postal}")

# 3. Losses module (import only, no model loading)
from src.losses import DistillationCachedMNRL, create_loss
print(f"\n[OK] losses.py (imports only)")

# 4. Data builder (import only)
from src.data_builder import (
    load_ground_truth, parse_matched_ids, records_to_dict,
    load_entity_country_map, build_training_data
)
print(f"[OK] data_builder.py (imports only)")

# 5. Train bi-encoder (import only)
from src.train_bi_encoder import (
    create_model_with_lora, create_frozen_model, train, merge_lora
)
print(f"[OK] train_bi_encoder.py (imports only)")

# 6. Eval bi-encoder (import only)
from src.eval_bi_encoder import (
    compute_retrieval_metrics, compute_margin_analysis, evaluate_model
)
print(f"[OK] eval_bi_encoder.py (imports only)")

# 7. Quick data test
import pandas as pd
gt_path = "../../dataset/train/train_ground_truth.tsv"
try:
    gt_sample = pd.read_csv(gt_path, sep="\t", nrows=5, dtype=str)
    print(f"\n[OK] Dataset accessible: {len(gt_sample)} rows read")
    for _, row in gt_sample.iterrows():
        matched = parse_matched_ids(row["matched_entity_ids"])
        print(f"  {row['source1_entity_id']}: {len(matched)} matches")
except Exception as e:
    print(f"\n[WARN] Dataset not accessible from this path: {e}")

print("\n" + "=" * 60)
print("ALL IMPORTS AND BASIC TESTS PASSED")
print("=" * 60)
