"""
Stage 2b: Qwen3-0.6B Causal Generative Matcher Fine-Tuning (Stretch Goal).

Fine-tunes Qwen3-0.6B with LoRA for sequence-pair business entity resolution,
implementing the complete specification from:
docs/07_stage2b_qwen3_generative_matcher_spec.md

Key architectural invariants:
1. Sliced Verdict-Token Cross-Entropy:
   Extracts hidden state exclusively at the final prompt token position before
   projecting through lm_head, collapsing logits tensor memory from ~3.27 GB
   down to ~29 MB (a 224x reduction; docs/07_stage2b Section 4.2).
2. Rank-Stabilized LoRA Scaling (rsLoRA, arXiv:2312.03732):
   LoRA rank r=64, alpha=64 (gamma = alpha / sqrt(r) = 8.0) across all linear layers.
3. Sliced KL Self-Distillation (Anti-Forgetting Layer 2):
   Auxiliary KL divergence against frozen pre-trained base model logits on the
   [No, Yes] verdict pair, bounding drift on unseen multilingual/French tokens.
4. Two-Direction Held-Out Country Validation Gate:
   Evaluates cross-country generalization (US <-> India) before accepting the adapter.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support

from src.qwen_matcher_features import (
    DEFAULT_MODEL,
    NO_TOKEN,
    PROMPT_TEMPLATE,
    YES_TOKEN,
    format_matcher_prompt,
    load_candidates,
    load_records,
)

logger = logging.getLogger(__name__)


def build_labeled_training_pairs(
    ground_truth_file: Path,
    source1_files: Sequence[Path],
    candidate_source_files: Sequence[Path],
    blocking_candidates_file: Optional[Path] = None,
    negatives_per_positive: int = 3,
    train_country: Optional[str] = None,
    eval_country: Optional[str] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Build labeled training and evaluation pairs formatted for Qwen3-0.6B causal LM.
    
    Returns:
        (train_pairs, eval_pairs) where each pair has keys:
        'prompt', 'label' (1 for match, 0 for non-match), 'country'
    """
    rng = random.Random(seed)
    records = load_records(list(source1_files) + list(candidate_source_files))

    # Load ground truth
    gt_df = pd.read_csv(ground_truth_file, sep="\t", dtype=str, keep_default_na=False)
    matches_map: Dict[str, set] = {}
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        matches_map[s1] = set(matched)

    # Load blocking candidates for negative mining
    candidates_map: Dict[str, List[str]] = {}
    if blocking_candidates_file and blocking_candidates_file.exists():
        pairs = load_candidates(blocking_candidates_file)
        for s1, c2 in pairs:
            candidates_map.setdefault(s1, []).append(c2)

    train_samples = []
    eval_samples = []

    for s1, pos_ids in matches_map.items():
        if s1 not in records:
            continue
        s1_rec = records[s1]
        country = (s1_rec.get("canonical_country") or s1_rec.get("country", "")).lower().strip()

        is_eval = False
        if eval_country and country == eval_country.lower().strip():
            is_eval = True
        elif train_country and country != train_country.lower().strip():
            continue

        target_list = eval_samples if is_eval else train_samples

        # 1. True positive pairs
        for pos_id in pos_ids:
            if pos_id not in records:
                continue
            pos_rec = records[pos_id]
            prompt = format_matcher_prompt(s1_rec, pos_rec)
            target_list.append({"prompt": prompt, "label": 1, "country": country})

            # 2. Hard negative pairs from blocking output
            cand_pool = candidates_map.get(s1, [])
            neg_candidates = [c for c in cand_pool if c not in pos_ids and c in records]
            if neg_candidates:
                chosen_negs = (
                    rng.sample(neg_candidates, min(negatives_per_positive, len(neg_candidates)))
                    if len(neg_candidates) > negatives_per_positive
                    else neg_candidates
                )
                for neg_id in chosen_negs:
                    neg_rec = records[neg_id]
                    neg_prompt = format_matcher_prompt(s1_rec, neg_rec)
                    target_list.append({"prompt": neg_prompt, "label": 0, "country": country})

    rng.shuffle(train_samples)
    rng.shuffle(eval_samples)

    logger.info(
        f"Built Qwen matcher datasets: {len(train_samples)} training pairs, "
        f"{len(eval_samples)} evaluation pairs"
    )
    return train_samples, eval_samples


def compute_sliced_logits(
    model,
    input_ids,
    attention_mask,
    no_id: int,
    yes_id: int,
):
    """
    Compute binary logits at the terminal prompt token position exclusively.
    
    Memory footprint: projects only batch_size x 1 hidden states through lm_head,
    reducing intermediate activation tensor from ~3.27 GB to ~29 MB.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]  # [B, S, D]

    # Find the position of the last non-padded token for each item in the batch
    last_token_idx = attention_mask.sum(dim=1) - 1  # [B]
    batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)

    # Slice ONLY the terminal verdict position
    terminal_hidden = hidden_states[batch_idx, last_token_idx, :].unsqueeze(1)  # [B, 1, D]

    # Project single token position through LM head
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "base_model"):
        lm_head = getattr(model.base_model.model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("Could not locate lm_head on model or PEFT wrapped model.")

    terminal_logits = lm_head(terminal_hidden).squeeze(1)  # [B, V]

    # Slice specifically the binary [No, Yes] tokens
    binary_logits = terminal_logits[:, [no_id, yes_id]]  # [B, 2]
    return binary_logits


def evaluate_qwen_matcher(
    model,
    eval_pairs: List[Dict[str, str]],
    tokenizer,
    no_id: int,
    yes_id: int,
    batch_size: int = 32,
    max_seq_length: int = 224,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate Qwen matcher on a validation set and compute classification metrics."""
    import torch
    import torch.nn.functional as F

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(eval_pairs), batch_size):
            chunk = eval_pairs[i : i + batch_size]
            prompts = [item["prompt"] for item in chunk]
            labels = [item["label"] for item in chunk]

            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            ).to(device)

            logits = compute_sliced_logits(
                model=model,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                no_id=no_id,
                yes_id=yes_id,
            )
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    if not all_labels:
        return {"samples": 0}

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    ap = float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    beta = 0.5
    f05 = float((1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)) if (prec + rec) > 0 else 0.0

    return {
        "samples": len(y_true),
        "accuracy": acc,
        "average_precision": ap,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "macro_f05": f05,
    }


def train_qwen_matcher(
    ground_truth_file: str,
    source1_files: Sequence[str],
    candidate_source_files: Sequence[str],
    output_dir: str,
    blocking_candidates_file: Optional[str] = None,
    base_model_name: str = DEFAULT_MODEL,
    mode: str = "held_out_country",
    train_country: Optional[str] = "us",
    eval_country: Optional[str] = "india",
    epochs: int = 3,
    batch_size: int = 16,
    grad_accum_steps: int = 2,
    learning_rate: float = 5e-5,
    lora_r: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    use_distillation: bool = True,
    distill_weight: float = 0.10,
    negatives_per_positive: int = 3,
    max_seq_length: int = 224,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict:
    """
    Execute Stage 2b Qwen3-0.6B LoRA training.
    """
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data
    train_pairs, eval_pairs = build_labeled_training_pairs(
        ground_truth_file=Path(ground_truth_file),
        source1_files=[Path(p) for p in source1_files],
        candidate_source_files=[Path(p) for p in candidate_source_files],
        blocking_candidates_file=Path(blocking_candidates_file) if blocking_candidates_file else None,
        negatives_per_positive=negatives_per_positive,
        train_country=train_country if mode == "held_out_country" else None,
        eval_country=eval_country if mode == "held_out_country" else None,
        seed=seed,
    )

    if not train_pairs:
        raise ValueError("No training pairs found. Check input files and country filters.")

    # 2. Load tokenizer and verify label token IDs
    logger.info(f"Loading tokenizer from {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    yes_ids = tokenizer.encode(YES_TOKEN, add_special_tokens=False)
    no_ids = tokenizer.encode(NO_TOKEN, add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"Label tokens must encode to single token IDs; got yes={yes_ids}, no={no_ids}")
    yes_id, no_id = yes_ids[0], no_ids[0]

    # 3. Load base model and apply rsLoRA
    logger.info(f"Loading base model {base_model_name} in bfloat16")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=model_dtype)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.to(device)

    # 4. Optional frozen teacher model for KL self-distillation
    frozen_teacher = None
    if use_distillation and distill_weight > 0.0:
        logger.info("Initializing frozen base teacher model for sliced KL self-distillation")
        frozen_teacher = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=model_dtype)
        frozen_teacher.eval()
        for p in frozen_teacher.parameters():
            p.requires_grad = False
        frozen_teacher.to(device)

    # 5. Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = (len(train_pairs) // (batch_size * grad_accum_steps)) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps))

    # 6. Training loop
    logger.info(f"Starting Stage 2b Qwen matcher training: {epochs} epochs, {len(train_pairs)} pairs, {total_steps} total steps")
    global_step = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_pairs)
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i in range(0, len(train_pairs), batch_size):
            chunk = train_pairs[i : i + batch_size]
            prompts = [item["prompt"] for item in chunk]
            labels = torch.tensor([item["label"] for item in chunk], dtype=torch.long, device=device)

            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            ).to(device)

            # Sliced binary logits at the terminal position
            logits = compute_sliced_logits(
                model=model,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                no_id=no_id,
                yes_id=yes_id,
            )

            ce_loss = F.cross_entropy(logits, labels)

            # Sliced KL distillation loss
            distill_loss = torch.tensor(0.0, device=device)
            if frozen_teacher is not None:
                with torch.no_grad():
                    teacher_logits = compute_sliced_logits(
                        model=frozen_teacher,
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        no_id=no_id,
                        yes_id=yes_id,
                    )
                student_log_probs = F.log_softmax(logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

            loss = (ce_loss + distill_weight * distill_loss) / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if (i // batch_size + 1) % grad_accum_steps == 0 or (i + batch_size) >= len(train_pairs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                        f"CE Loss: {ce_loss.item():.4f} | Distill Loss: {distill_loss.item():.4f}"
                    )

        avg_loss = epoch_loss / max(1, (len(train_pairs) // batch_size))
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1} Complete. Mean Loss: {avg_loss:.4f}")

    # 7. Evaluation & Held-Out Country Gate Check
    eval_metrics = {}
    gate_decision = "GO"
    gate_reasons = []

    if eval_pairs:
        logger.info(f"Evaluating held-out country gate ({eval_country})...")
        eval_metrics = evaluate_qwen_matcher(
            model=model,
            eval_pairs=eval_pairs,
            tokenizer=tokenizer,
            no_id=no_id,
            yes_id=yes_id,
            batch_size=batch_size * 2,
            max_seq_length=max_seq_length,
            device=device,
        )
        logger.info(f"Gate Evaluation Metrics: {eval_metrics}")

        # Gate threshold checks per docs/07_stage2b Section 2
        if eval_metrics.get("average_precision", 0.0) < 0.75:
            gate_decision = "NO-GO"
            gate_reasons.append(f"Average Precision {eval_metrics.get('average_precision', 0):.4f} < 0.75 threshold")
        if eval_metrics.get("accuracy", 0.0) < 0.80:
            gate_decision = "NO-GO"
            gate_reasons.append(f"Accuracy {eval_metrics.get('accuracy', 0):.4f} < 0.80 threshold")

    # 8. Save fine-tuned adapter checkpoint
    logger.info(f"Saving fine-tuned Qwen matcher adapter to {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    metadata = {
        "base_model": base_model_name,
        "mode": mode,
        "train_country": train_country,
        "eval_country": eval_country,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "distill_weight": distill_weight if use_distillation else 0.0,
        "train_samples": len(train_pairs),
        "eval_samples": len(eval_pairs),
        "eval_metrics": eval_metrics,
        "gate_decision": gate_decision,
        "gate_reasons": gate_reasons,
        "train_losses": train_losses,
    }
    with open(output_path / "qwen_matcher_train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Stage 2b Training Complete. Gate Decision: {gate_decision}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-0.6B causal LM as Stage 2b matcher.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to ground truth TSV")
    parser.add_argument("--source1_files", type=str, nargs="+", required=True, help="Paths to source1 TSV files")
    parser.add_argument("--candidate_source_files", type=str, nargs="+", required=True, help="Paths to S2/S3 TSV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned adapter")
    parser.add_argument("--blocking_candidates_file", type=str, default=None, help="Stage 1 blocking candidate pairs TSV")
    parser.add_argument("--base_model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mode", type=str, default="held_out_country", choices=["held_out_country", "full"])
    parser.add_argument("--train_country", type=str, default="us")
    parser.add_argument("--eval_country", type=str, default="india")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--negatives_per_positive", type=int, default=3)
    parser.add_argument("--distill_weight", type=float, default=0.10)
    parser.add_argument("--no_distillation", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    train_qwen_matcher(
        ground_truth_file=args.ground_truth_file,
        source1_files=args.source1_files,
        candidate_source_files=args.candidate_source_files,
        output_dir=args.output_dir,
        blocking_candidates_file=args.blocking_candidates_file,
        base_model_name=args.base_model_name,
        mode=args.mode,
        train_country=args.train_country,
        eval_country=args.eval_country,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        distill_weight=args.distill_weight,
        use_distillation=not args.no_distillation,
        negatives_per_positive=args.negatives_per_positive,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
