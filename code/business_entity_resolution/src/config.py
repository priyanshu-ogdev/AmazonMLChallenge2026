"""
Configuration dataclasses for bi-encoder LoRA fine-tuning.

All committed parameters from the Layer 2 design, with research upgrades
(use_rslora, gradient_checkpointing) incorporated.

These are the ACTUAL values to write into code — not protocols or ranges.
Any controlled recovery values are documented in the Layer 2 overview and
must be recorded as separate experiments rather than silently changing this
baseline.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class LoRAConfig:
    """
    LoRA adapter configuration for BGE-M3 dense head.

    Rationale (from docs/LAYER_2_OVERVIEW.md):
    - r=64: middle of the 32/64/128 sweep, chosen as a single committed value.
      Optimizer overhead (0.34GB) is negligible vs 10GB headroom on 3060.
    - lora_alpha=64 with use_rslora=True: rank-stabilized scaling (α/√r),
      effective scale = 64/√64 = 8. Prevents the gradient collapse at higher
      ranks that biased Hu et al.'s original rank ablation (arXiv:2312.03732).
    - target_modules="all-linear": attention (query/key/value/dense) + FFN
      (intermediate.dense, output.dense), all layers. Never attention-only,
      never top-N-only (docs/LAYER_2_OVERVIEW.md).
    - lora_dropout=0.1: middle of the 0.05-0.1 range
      (docs/LAYER_2_OVERVIEW.md).
    - bias="none": standard LoRA practice, keeps trainable param count minimal.
    """
    r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: str = "all-linear"
    bias: str = "none"
    use_rslora: bool = True  # Research upgrade: rank-stabilized scaling
    task_type: str = "FEATURE_EXTRACTION"


@dataclass
class TrainingConfig:
    """
    Training hyperparameters for the bi-encoder LoRA fine-tune.

    From docs/LAYER_2_OVERVIEW.md "Stage 2a" section:
    - physical_batch_size=48: mid-point of 32-64 range, fits in ~10GB headroom
    - mini_batch_size=16: GradCache chunk for CachedMultipleNegativesRankingLoss
    - max_seq_length=80: business name+address strings are short;
      training at native multi-thousand-token ceiling wastes VRAM
    - learning_rate=2e-5: standard LoRA fine-tune LR for sentence-transformer scale
    - warmup_ratio=0.1: standard warmup
    - epochs=3: small labeled set — more risks overfitting
    - distillation_weight=0.10: mid-point of the 0.05-0.15 range
      (docs/LAYER_2_OVERVIEW.md)
    """
    # Model
    model_name: str = "BAAI/bge-m3"
    output_dir: str = "./output/bge-m3-lora-er"

    # Sequence
    max_seq_length: int = 80

    # Batch
    per_device_train_batch_size: int = 48
    per_device_eval_batch_size: int = 64
    mini_batch_size: int = 16  # GradCache chunk size

    # Optimizer
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_train_epochs: int = 3
    lr_scheduler_type: str = "cosine"  # Upgrade: cosine > linear for fine-tuning

    # Precision & memory
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Self-distillation (anti-forgetting layer 2)
    use_distillation: bool = True
    distillation_weight: float = 0.10

    # MNRL
    mnrl_scale: float = 20.0  # Temperature scaling for contrastive loss

    # Logging & checkpointing
    logging_steps: int = 50
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_held_out_country_cosine_recall@10"

    # Reproducibility
    seed: int = 42
    dataloader_num_workers: int = 2


@dataclass
class DataConfig:
    """
    Data preparation configuration.

    Strategy: sample entities (not pairs) to keep all matches per entity
    together. Balance by country for the anti-forgetting stack (layer 3).
    Singletons excluded — bi-encoder produces similarity scores; the GBM
    and Stage 4 threshold handle singleton identification.
    """
    # Paths (relative to project root)
    data_dir: str = "../../dataset"
    output_dir: str = "./prepared_data"

    # Sampling
    sample_per_country: int = 50000  # 50K US + 50K India = 100K entities
    eval_sample_per_country: int = 2000  # For held-out country evaluator
    eval_corpus_negatives: int = 5000  # Non-relevant docs in eval corpus
    min_matches_for_sample: int = 1  # Skip singletons

    # Negative mining
    negatives_per_positive: int = 0  # 0 = in-batch negatives only (CachedMNRL)
    # Set > 0 for explicit hard negatives (e.g., from blocking output)

    # Hard negative source (blocking output, when available)
    blocking_candidates_path: Optional[str] = None

    # Processing
    chunk_size: int = 100000  # Rows per chunk for memory-efficient TSV reading
    seed: int = 42
    num_workers: int = 4


@dataclass
class EvalConfig:
    """
    Evaluation configuration for the held-out-country gate.

    Protocol (from docs/LAYER_2_OVERVIEW.md):
    - Train on US, validate on India (and reverse)
    - If held-out-country Recall@K is meaningfully worse than in-domain:
      raise distillation_weight or drop rank or fall back to off-the-shelf
    """
    # Retrieval metrics
    recall_at_k: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    mrr_at_k: int = 10

    # Margin analysis
    margin_thresholds: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )

    # Go/no-go gate thresholds
    min_recall_at_10: float = 0.80  # Minimum Recall@10 on held-out country
    max_country_gap: float = 0.10  # Max allowed gap between in-domain and held-out

    # Whether to run reverse direction (train India, eval US)
    run_reverse: bool = False


@dataclass
class VRAMBudget:
    """
    VRAM budget verification for RTX 3060 12GB.
    Computed directly from model architecture, not assumed.

    From the Layer 2 VRAM budget:
    - Base weights (bf16, frozen): 568M × 2 bytes = 1.14 GB
    - LoRA trainable params (r=64, all-linear): 28.3M × 2 bytes = 0.06 GB
    - LoRA optimizer (AdamW: fp32 master + 2 moments): 28.3M × 12 bytes = 0.34 GB
    - Frozen model for distillation: 568M × 2 bytes = 1.14 GB
    - Fixed total: ~2.7 GB
    - Available for activations/batch: ~9.3 GB
    """
    total_vram_gb: float = 12.0
    base_model_gb: float = 1.14
    lora_params_gb: float = 0.06
    lora_optimizer_gb: float = 0.34
    frozen_model_gb: float = 1.14  # Only if distillation is enabled

    @property
    def fixed_overhead_gb(self) -> float:
        return self.base_model_gb + self.lora_params_gb + self.lora_optimizer_gb

    @property
    def fixed_overhead_with_distillation_gb(self) -> float:
        return self.fixed_overhead_gb + self.frozen_model_gb

    @property
    def available_for_batch_gb(self) -> float:
        return self.total_vram_gb - self.fixed_overhead_with_distillation_gb

    def verify(self, use_distillation: bool = True) -> bool:
        overhead = (self.fixed_overhead_with_distillation_gb
                    if use_distillation else self.fixed_overhead_gb)
        available = self.total_vram_gb - overhead
        # Need at least 4GB for activations/batch at physical_batch=48, seq_len=80
        return available >= 4.0
