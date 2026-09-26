"""
Main training script for BGE-M3 bi-encoder with LoRA fine-tuning.

Runs on GPU (RTX 3060 12GB). Loads pre-built training data from
data_builder.py, applies LoRA to BGE-M3, trains with
CachedMultipleNegativesRankingLoss + optional self-distillation.

Two training modes:
  1. Held-out country gate: train on one country, eval on held-out country
     (verify the anti-forgetting stack before trusting the fine-tune)
  2. Full training: train on both countries (after gate passes)

Usage:
    # Step 1: Held-out country gate
    python -m src.train_bi_encoder \\
        --data_dir ./prepared_data \\
        --output_dir ./output/bge-m3-lora-gate \\
        --mode held_out_country \\
        --use_distillation

    # Step 2: Full training (after gate passes)
    python -m src.train_bi_encoder \\
        --data_dir ./prepared_data \\
        --output_dir ./output/bge-m3-lora-final \\
        --mode full \\
        --use_distillation

    # Merge LoRA into base model for inference
    python -m src.train_bi_encoder \\
        --merge_lora ./output/bge-m3-lora-final \\
        --merge_output ./output/bge-m3-merged
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    util,
)
from sentence_transformers.training_args import BatchSamplers
try:
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
except ImportError:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

from peft import LoraConfig, TaskType

from src.config import LoRAConfig, TrainingConfig
from src.losses import create_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def create_model_with_lora(
    model_name: str,
    lora_config: LoRAConfig,
    max_seq_length: int = 80,
) -> SentenceTransformer:
    """
    Load BGE-M3 and apply LoRA adapter.

    Uses sentence-transformers' native add_adapter() API for clean
    PEFT integration. The adapter is applied to ALL linear layers
    (attention + FFN), all encoder layers.

    Key: use_rslora=True enables rank-stabilized scaling (α/√r = 64/8 = 8),
    which prevents gradient collapse at rank 64 that biased the original
    LoRA paper's rank ablation toward under-rating higher ranks.
    """
    logger.info(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length

    logger.info(f"Applying LoRA adapter (r={lora_config.r}, "
                f"alpha={lora_config.lora_alpha}, rslora={lora_config.use_rslora})")

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        use_rslora=lora_config.use_rslora,
    )

    model.add_adapter(peft_config)

    # Cast the full model (base + LoRA) to bf16 so that LoRA parameters
    # and their optimizer state match the VRAM budget assumptions.
    # SentenceTransformerTrainer's bf16=True only autocasts forward-pass
    # activations — it does NOT recast parameter storage. Without this,
    # 28.3M LoRA params land in fp32, adding ~0.22 GB vs. the 0.06 GB
    # budget entry, and optimizer state (fp32 moments) would already dominate
    # correctly. The base weights loaded by SentenceTransformer are typically
    # fp32 from HuggingFace; casting here also halves their storage to match
    # the 1.14 GB budget entry.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(dtype=torch.bfloat16)
        logger.info("  Cast model (base + LoRA) to bfloat16")

    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError("LoRA adapter produced no trainable parameters")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    return model


def create_frozen_model(
    model_name: str,
    max_seq_length: int = 80,
) -> SentenceTransformer:
    """
    Load a frozen copy of the base model for self-distillation.

    This is a completely separate SentenceTransformer instance with
    NO LoRA adapter. All parameters are frozen (requires_grad=False).
    Used as the reference embedding space for the distillation anchor loss.

    Memory: adds ~1.14 GB (568M × 2 bytes bf16) to VRAM.
    """
    logger.info(f"Loading frozen model for distillation: {model_name}")
    frozen_model = SentenceTransformer(model_name)
    frozen_model.max_seq_length = max_seq_length

    # Match the training precision when CUDA/bf16 is available. Leaving this
    # second 568M-parameter model in fp32 would invalidate the VRAM budget.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        frozen_model = frozen_model.to(dtype=torch.bfloat16)

    # Freeze everything
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in frozen_model.parameters())
    logger.info(f"  Frozen model parameters: {frozen_params:,} (all frozen)")

    return frozen_model


# ---------------------------------------------------------------------------
# Evaluation setup
# ---------------------------------------------------------------------------

def create_evaluator(
    data_dir: str,
    name: str = "held_out_country",
    prefix: str = "",
) -> Optional[InformationRetrievalEvaluator]:
    """
    Create an InformationRetrievalEvaluator from pre-built evaluation data.

    Loads queries, corpus, and relevant_docs from JSON files produced
    by data_builder.py. Returns None if eval data is not found.
    """
    data_path = Path(data_dir)
    queries_path = data_path / f"{prefix}eval_queries.json"
    corpus_path = data_path / f"{prefix}eval_corpus.json"
    relevant_path = data_path / f"{prefix}eval_relevant.json"

    if prefix and not queries_path.exists():
        logger.warning(f"Prefixed eval files with prefix '{prefix}' not found, falling back to default eval files")
        queries_path = data_path / "eval_queries.json"
        corpus_path = data_path / "eval_corpus.json"
        relevant_path = data_path / "eval_relevant.json"

    if not all(p.exists() for p in [queries_path, corpus_path, relevant_path]):
        logger.warning("Evaluation data not found, skipping evaluator setup")
        return None

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(relevant_path, "r", encoding="utf-8") as f:
        relevant_raw = json.load(f)

    # Convert relevant docs lists to sets
    relevant_docs = {k: set(v) for k, v in relevant_raw.items()}

    logger.info(
        f"IR Evaluator ({name}): {len(queries)} queries, {len(corpus)} corpus docs, "
        f"{sum(len(v) for v in relevant_docs.values())} relevant pairs"
    )

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True,
        batch_size=64,
        # InformationRetrievalEvaluator passes batched embedding matrices.
        # util.cos_sim returns the required query-by-corpus score matrix;
        # torch.cosine_similarity would incorrectly collapse the batch axis.
        score_functions={"cosine": util.cos_sim},
    )

    return evaluator


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    output_dir: str,
    mode: str = "held_out_country",
    direction: str = "default",
    training_config: Optional[TrainingConfig] = None,
    lora_config: Optional[LoRAConfig] = None,
):
    """
    Run bi-encoder LoRA fine-tuning.

    Args:
        data_dir: Path to prepared data from data_builder.py
        output_dir: Where to save the fine-tuned model
        mode: "held_out_country" (gate check) or "full" (final training)
        direction: Direction for held-out country ("default", "us_to_india", "india_to_us")
        training_config: Training hyperparameters
        lora_config: LoRA adapter configuration
    """
    if training_config is None:
        training_config = TrainingConfig(output_dir=output_dir)
    if lora_config is None:
        lora_config = LoRAConfig()

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    split_info_file = data_path / "country_split_info.json"
    p_train = "us"
    p_eval = "india"
    if split_info_file.exists():
        try:
            with open(split_info_file, "r", encoding="utf-8") as f:
                s_info = json.load(f)
                p_train = s_info.get("primary_train_country", "us")
                p_eval = s_info.get("primary_eval_country", "india")
        except Exception:
            pass

    pri_prefix = f"{p_train}_train_{p_eval}_eval_"
    rev_prefix = f"{p_eval}_train_{p_train}_eval_"
    if not (data_path / f"{pri_prefix}held_out_country_dataset").exists() and (data_path / "us_train_india_eval_held_out_country_dataset").exists():
        pri_prefix = "us_train_india_eval_"
        rev_prefix = "india_train_us_eval_"

    prefix = ""
    if direction in ("us_to_india", "primary"):
        prefix = pri_prefix
    elif direction in ("india_to_us", "reverse"):
        prefix = rev_prefix

    if mode == "held_out_country":
        dataset_path = data_path / f"{prefix}held_out_country_dataset"
        if not dataset_path.exists():
            dataset_path = data_path / "held_out_country_dataset"
        logger.info(f"Mode: held-out country gate (direction={direction}, path={dataset_path})")
    elif mode == "full":
        dataset_path = data_path / "full_training_dataset"
        logger.info("Mode: full training (both countries)")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'held_out_country' or 'full'")

    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("eval")

    # Filter to model input columns (anchor, positive, negative, negative_*)
    # so SentenceTransformerTrainer does not attempt to tokenize extra columns
    model_columns = [
        col for col in getattr(train_dataset, "column_names", [])
        if col in ("anchor", "positive", "negative") or col.startswith("negative_")
    ]
    if model_columns and set(model_columns) != set(train_dataset.column_names):
        logger.info(f"Filtering train_dataset columns to model inputs: {model_columns}")
        train_dataset = train_dataset.select_columns(model_columns)
    if eval_dataset is not None:
        eval_cols = [
            col for col in getattr(eval_dataset, "column_names", [])
            if col in ("anchor", "positive", "negative") or col.startswith("negative_")
        ]
        if eval_cols and set(eval_cols) != set(eval_dataset.column_names):
            eval_dataset = eval_dataset.select_columns(eval_cols)

    logger.info(f"  Train: {len(train_dataset)} pairs")
    if eval_dataset:
        logger.info(f"  Eval: {len(eval_dataset)} pairs")

    # -----------------------------------------------------------------------
    # Initialize model with LoRA
    # -----------------------------------------------------------------------
    model = create_model_with_lora(
        model_name=training_config.model_name,
        lora_config=lora_config,
        max_seq_length=training_config.max_seq_length,
    )

    # Enable gradient checkpointing if requested.
    # PEFT/LoRA REQUIREMENT: enable_input_require_grads() MUST be called first.
    # Gradient checkpointing works by recomputing activations on the backward
    # pass. For this to create a valid gradient graph through the input
    # embeddings, the input tensors must have requires_grad=True. With vanilla
    # fine-tuning the embedding layer's output has requires_grad because the
    # embedding weights are trainable; with LoRA the base embedding weights are
    # frozen, so the embedding output has requires_grad=False and the gradient
    # graph is silently broken at that point. enable_input_require_grads()
    # registers a forward hook that forces requires_grad=True on all module
    # inputs, threading the gradient signal through the checkpoint boundaries.
    if training_config.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing (with PEFT input-grad hook)")
        try:
            auto_model = model[0].auto_model
            auto_model.enable_input_require_grads()       # PEFT LoRA requirement
            auto_model.gradient_checkpointing_enable()    # then enable checkpointing
        except AttributeError:
            logger.warning("Could not enable gradient checkpointing (model may not support it)")

    # -----------------------------------------------------------------------
    # Initialize frozen model for self-distillation
    # -----------------------------------------------------------------------
    frozen_model = None
    if training_config.use_distillation:
        frozen_model = create_frozen_model(
            model_name=training_config.model_name,
            max_seq_length=training_config.max_seq_length,
        )

    # -----------------------------------------------------------------------
    # Create loss function
    # -----------------------------------------------------------------------
    loss = create_loss(
        model=model,
        use_distillation=training_config.use_distillation,
        distill_weight=training_config.distillation_weight,
        mini_batch_size=training_config.mini_batch_size,
        scale=training_config.mnrl_scale,
        frozen_model=frozen_model,
        distill_anchor_positive_only=getattr(
            training_config, "distill_anchor_positive_only", True
        ),
    )

    # -----------------------------------------------------------------------
    # Create evaluator
    # -----------------------------------------------------------------------
    eval_name = f"held_out_{direction}" if direction != "default" else "held_out_country"
    evaluator = create_evaluator(data_dir, name=eval_name, prefix=prefix)

    # -----------------------------------------------------------------------
    # Training arguments
    # -----------------------------------------------------------------------
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        # Batch
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        # Optimizer
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        num_train_epochs=training_config.num_train_epochs,
        lr_scheduler_type=training_config.lr_scheduler_type,
        # Precision
        bf16=training_config.bf16,
        # Logging & checkpoints
        logging_steps=training_config.logging_steps,
        eval_strategy=training_config.eval_strategy if evaluator else "no",
        eval_steps=training_config.eval_steps if evaluator else None,
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=training_config.load_best_model_at_end if evaluator else False,
        # InformationRetrievalEvaluator prefixes its named metric with the
        # evaluator name, and Trainer prefixes it with "eval_".
        metric_for_best_model=(
            f"eval_{evaluator.name}_cosine_recall@10"
            if evaluator else None
        ),
        # Batch sampling
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Reproducibility
        seed=training_config.seed,
        dataloader_num_workers=training_config.dataloader_num_workers,
        # Reports
        report_to="none",  # Set to "tensorboard" or "wandb" if desired
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"  Model: {training_config.model_name}")
    logger.info(f"  LoRA rank: {lora_config.r}")
    logger.info(f"  Distillation: {training_config.use_distillation} "
                f"(weight={training_config.distillation_weight})")
    logger.info(f"  Batch size: {training_config.per_device_train_batch_size}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Epochs: {training_config.num_train_epochs}")
    logger.info(f"  Max seq length: {training_config.max_seq_length}")
    logger.info(f"  bf16: {training_config.bf16}")
    logger.info(f"  Gradient checkpointing: {training_config.gradient_checkpointing}")
    logger.info("=" * 60)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    # -----------------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------------
    final_path = output_path / "final"
    logger.info(f"Saving fine-tuned model to {final_path}")
    model.save_pretrained(str(final_path))

    # Save training config for reproducibility
    config_dict = {
        "lora": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules,
            "bias": lora_config.bias,
            "use_rslora": lora_config.use_rslora,
        },
        "training": {
            "model_name": training_config.model_name,
            "max_seq_length": training_config.max_seq_length,
            "batch_size": training_config.per_device_train_batch_size,
            "mini_batch_size": training_config.mini_batch_size,
            "learning_rate": training_config.learning_rate,
            "epochs": training_config.num_train_epochs,
            "use_distillation": training_config.use_distillation,
            "distillation_weight": training_config.distillation_weight,
            "bf16": training_config.bf16,
            "gradient_checkpointing": training_config.gradient_checkpointing,
            "seed": training_config.seed,
        },
        "mode": mode,
    }
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Training complete!")
    return model


# ---------------------------------------------------------------------------
# LoRA merge utility
# ---------------------------------------------------------------------------

def merge_lora(adapter_path: str, output_path: str):
    """
    Merge LoRA adapter weights into the base model.

    At inference, every LoRA adapter is merged (W' = W + BA) to produce
    a plain dense model with zero adapter overhead. This is done ONCE
    after training, not at every inference call.

    The merged model can be loaded as a regular SentenceTransformer.
    """
    logger.info(f"Loading adapter model from {adapter_path}")
    model = SentenceTransformer(adapter_path)

    # Access the underlying PEFT model and merge
    try:
        peft_model = model[0].auto_model
        if hasattr(peft_model, "merge_and_unload"):
            logger.info("Merging LoRA weights into base model...")
            model[0].auto_model = peft_model.merge_and_unload()
            logger.info("Merge complete")
        else:
            logger.warning("Model doesn't have merge_and_unload — may not be a PEFT model")
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        raise

    # Save merged model
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged model to {output_path}")
    model.save(str(output_p))

    # Verify by reloading
    logger.info("Verifying merged model loads correctly...")
    test_model = SentenceTransformer(str(output_p))
    test_emb = test_model.encode("test sentence")
    logger.info(f"  Verification passed (embedding dim: {len(test_emb)})")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BGE-M3 bi-encoder LoRA fine-tuning for entity resolution"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Train command ---
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--data_dir", type=str, default="./prepared_data",
                              help="Path to prepared data from data_builder.py")
    train_parser.add_argument("--output_dir", type=str, default="./output/bge-m3-lora",
                              help="Output directory for fine-tuned model")
    train_parser.add_argument("--mode", type=str, default="held_out_country",
                              choices=["held_out_country", "full"],
                              help="Training mode")
    train_parser.add_argument("--direction", type=str, default="default",
                              choices=["default", "us_to_india", "india_to_us"],
                              help="Direction for held-out country gate split")
    # Training params
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=48)
    train_parser.add_argument("--learning_rate", type=float, default=2e-5)
    train_parser.add_argument("--max_seq_length", type=int, default=80)
    train_parser.add_argument("--mini_batch_size", type=int, default=16)
    # LoRA params
    train_parser.add_argument("--lora_r", type=int, default=64)
    train_parser.add_argument("--lora_alpha", type=int, default=64)
    train_parser.add_argument("--lora_dropout", type=float, default=0.1)
    # Distillation
    train_parser.add_argument("--use_distillation", action="store_true", default=True)
    train_parser.add_argument("--no_distillation", action="store_true",
                              help="Disable self-distillation")
    train_parser.add_argument("--distill_weight", type=float, default=0.10)
    train_parser.add_argument("--distill_all_columns", action="store_true",
                              help="Distill across all columns including mined negatives (higher compute cost)")
    # Hardware
    train_parser.add_argument("--no_bf16", action="store_true",
                              help="Disable bf16 (use fp32)")
    train_parser.add_argument("--no_grad_ckpt", action="store_true",
                              help="Disable gradient checkpointing")
    train_parser.add_argument("--seed", type=int, default=42)

    # --- Merge command ---
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA into base model")
    merge_parser.add_argument("--adapter_path", type=str, required=True,
                              help="Path to fine-tuned model with LoRA adapter")
    merge_parser.add_argument("--output_path", type=str, required=True,
                              help="Output path for merged model")

    args = parser.parse_args()

    if args.command == "train":
        lora_cfg = LoRAConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        train_cfg = TrainingConfig(
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            use_distillation=args.use_distillation and not args.no_distillation,
            distillation_weight=args.distill_weight,
            distill_anchor_positive_only=not args.distill_all_columns,
            bf16=not args.no_bf16,
            gradient_checkpointing=not args.no_grad_ckpt,
            seed=args.seed,
        )
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            direction=args.direction,
            training_config=train_cfg,
            lora_config=lora_cfg,
        )

    elif args.command == "merge":
        merge_lora(args.adapter_path, args.output_path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
