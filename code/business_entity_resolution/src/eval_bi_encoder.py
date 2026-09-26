"""
Held-out country evaluation for the bi-encoder fine-tune.

This is the GO/NO-GO GATE: if the fine-tuned model's retrieval quality
on the held-out country is meaningfully worse than in-domain, the
anti-forgetting stack is insufficient and the fine-tune should not be trusted.

Protocol (from docs/06_stage2a_bge_m3_training_spec.md):
1. Load the fine-tuned bi-encoder
2. Encode eval queries (held-out country S1 entities)
3. Encode eval corpus (held-out country S2/S3 entities)
4. Compute Recall@K and MRR
5. Compare against off-the-shelf baseline (optional)
6. Report go/no-go decision

Recovery if gate fails:
- Raise self_distillation_weight toward 0.15
- Or drop LoRA rank to 32
- Or fall back to off-the-shelf BGE-M3 for the Stage 2a-i feature
- Evaluate Qwen3-Embedding-0.6B separately as an auxiliary Stage 2a-ii feature

Usage:
    python -m src.eval_bi_encoder \\
        --model_path ./output/bge-m3-lora-gate/final \\
        --data_dir ./prepared_data \\
        --baseline_model BAAI/bge-m3
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Set, Optional, Tuple
from collections import defaultdict

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_eval_data(data_dir: str, prefix: str = "") -> Tuple[Dict, Dict, Dict]:
    """Load pre-built IR evaluation data, with optional directional prefix."""
    data_path = Path(data_dir)

    queries_file = data_path / f"{prefix}eval_queries.json"
    corpus_file = data_path / f"{prefix}eval_corpus.json"
    relevant_file = data_path / f"{prefix}eval_relevant.json"

    # Fallback to unprefixed if prefixed does not exist and prefix was specified
    if prefix and not queries_file.exists():
        logger.warning(f"Prefixed eval file {queries_file} not found; falling back to default eval files.")
        queries_file = data_path / "eval_queries.json"
        corpus_file = data_path / "eval_corpus.json"
        relevant_file = data_path / "eval_relevant.json"

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(relevant_file, "r", encoding="utf-8") as f:
        relevant_raw = json.load(f)

    relevant_docs = {k: set(v) for k, v in relevant_raw.items()}

    logger.info(f"Loaded eval data (prefix='{prefix}'): {len(queries)} queries, {len(corpus)} corpus, "
                f"{sum(len(v) for v in relevant_docs.values())} relevant pairs")

    return queries, corpus, relevant_docs


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: list,
    corpus_ids: list,
    relevant_docs: Dict[str, Set[str]],
    k_values: list = None,
) -> Dict[str, float]:
    """
    Compute retrieval metrics: Recall@K, MRR@K, Precision@K.

    Uses cosine similarity (embeddings assumed to be normalized).
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]

    # Compute similarity matrix
    logger.info("Computing similarity matrix...")
    similarity = np.dot(query_embeddings, corpus_embeddings.T)

    metrics = {}
    all_recalls = defaultdict(list)
    all_precisions = defaultdict(list)
    mrr_scores = []

    for i, qid in enumerate(query_ids):
        if qid not in relevant_docs or not relevant_docs[qid]:
            continue

        scores = similarity[i]
        ranked_indices = np.argsort(-scores)  # Descending
        ranked_ids = [corpus_ids[idx] for idx in ranked_indices]
        relevant = relevant_docs[qid]

        # Find first relevant hit for MRR
        first_hit = None
        for rank, cid in enumerate(ranked_ids, 1):
            if cid in relevant:
                first_hit = rank
                break

        if first_hit:
            mrr_scores.append(1.0 / first_hit)
        else:
            mrr_scores.append(0.0)

        # Recall@K and Precision@K
        for k in k_values:
            top_k = set(ranked_ids[:k])
            hits = len(top_k & relevant)
            recall = hits / len(relevant) if relevant else 0.0
            precision = hits / k
            all_recalls[k].append(recall)
            all_precisions[k].append(precision)

    # Aggregate
    for k in k_values:
        metrics[f"recall@{k}"] = np.mean(all_recalls[k])
        metrics[f"precision@{k}"] = np.mean(all_precisions[k])

    metrics["mrr"] = np.mean(mrr_scores)
    metrics["num_queries"] = len([q for q in query_ids if q in relevant_docs and relevant_docs[q]])

    return metrics


def compute_margin_analysis(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: list,
    corpus_ids: list,
    relevant_docs: Dict[str, Set[str]],
    margin_thresholds: list = None,
) -> Dict[str, float]:
    """
    Margin analysis: what fraction of queries have a positive pair
    scoring higher than the hardest negative by at least `margin`.

    This is the cheap proxy evaluation from docs/06_stage2a_bge_m3_training_spec.md:
    "margin/score-gap pass rate drives early stopping within a run"
    """
    if margin_thresholds is None:
        margin_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    similarity = np.dot(query_embeddings, corpus_embeddings.T)
    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}

    pass_rates = {}
    margins = []

    for i, qid in enumerate(query_ids):
        if qid not in relevant_docs or not relevant_docs[qid]:
            continue

        scores = similarity[i]
        relevant = relevant_docs[qid]

        pos_indices = [corpus_id_to_idx[cid] for cid in relevant if cid in corpus_id_to_idx]
        if not pos_indices or len(pos_indices) == len(corpus_ids):
            continue

        best_pos = float(np.max(scores[pos_indices]))
        scores_masked = scores.copy()
        scores_masked[pos_indices] = -np.inf
        best_neg = float(np.max(scores_masked))

        margin = best_pos - best_neg
        margins.append(margin)

    margins = np.array(margins)
    for threshold in margin_thresholds:
        pass_rate = np.mean(margins >= threshold) if len(margins) > 0 else 0.0
        pass_rates[f"margin_pass@{threshold:.2f}"] = pass_rate

    pass_rates["mean_margin"] = float(np.mean(margins)) if len(margins) > 0 else 0.0
    pass_rates["median_margin"] = float(np.median(margins)) if len(margins) > 0 else 0.0

    return pass_rates


def _evaluate_single_direction(
    model,
    base_model,
    data_dir: str,
    prefix: str = "",
    direction_name: str = "primary",
    batch_size: int = 64,
    k_values: list = None,
    margin_thresholds: list = None,
    min_recall_10: float = 0.80,
    max_country_gap: float = 0.05,
    min_margin_gain: float = 0.15,
    min_margin_pass_10: float = 0.60,
) -> Dict:
    """Evaluate one direction (e.g. US->India or India->US) against gate criteria."""
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]

    queries, corpus, relevant_docs = load_eval_data(data_dir, prefix=prefix)
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    dir_results = {}

    logger.info(f"\n--- Direction [{direction_name}]: Encoding {len(query_texts)} queries & {len(corpus_texts)} corpus docs ---")
    q_embs = model.encode(query_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    c_embs = model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    retrieval_metrics = compute_retrieval_metrics(
        q_embs, c_embs, query_ids, corpus_ids, relevant_docs, k_values
    )
    margin_metrics = compute_margin_analysis(
        q_embs, c_embs, query_ids, corpus_ids, relevant_docs, margin_thresholds
    )
    dir_results["fine_tuned"] = {**retrieval_metrics, **margin_metrics}

    if base_model:
        base_q_embs = base_model.encode(query_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        base_c_embs = base_model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        base_retrieval = compute_retrieval_metrics(
            base_q_embs, base_c_embs, query_ids, corpus_ids, relevant_docs, k_values
        )
        base_margin = compute_margin_analysis(
            base_q_embs, base_c_embs, query_ids, corpus_ids, relevant_docs, margin_thresholds
        )
        dir_results["baseline"] = {**base_retrieval, **base_margin}

    # Direction-specific check (docs/06 Section 7.2 Gate Acceptance Criteria)
    ft_recall_10 = dir_results["fine_tuned"].get("recall@10", 0.0)
    decision = "GO"
    reasons = []

    if ft_recall_10 < min_recall_10:
        decision = "NO-GO"
        reasons.append(
            f"[{direction_name}] Recall@10 ({ft_recall_10:.4f}) below absolute minimum threshold ({min_recall_10:.2f})"
        )

    if base_model and "baseline" in dir_results:
        base_recall_10 = dir_results["baseline"].get("recall@10", 0.0)
        gap = base_recall_10 - ft_recall_10
        if gap > max_country_gap:
            decision = "NO-GO"
            reasons.append(
                f"[{direction_name}] Fine-tuned Recall@10 ({ft_recall_10:.4f}) is {gap:.4f} worse than "
                f"baseline ({base_recall_10:.4f}), exceeding max degradation gap ({max_country_gap:.2f})"
            )
        elif gap > 0.0:
            reasons.append(
                f"[{direction_name}] WARNING: Fine-tuned Recall@10 ({ft_recall_10:.4f}) is slightly below "
                f"baseline ({base_recall_10:.4f}) by {gap:.4f} (within {max_country_gap:.2f} tolerance)."
            )

        # Margin pass rate check at Delta >= 0.30 relative to baseline
        base_margin_30 = dir_results["baseline"].get("margin_pass@0.30", 0.0)
        ft_margin_30 = dir_results["fine_tuned"].get("margin_pass@0.30", 0.0)
        if base_margin_30 > 0.0:
            rel_margin_gain = (ft_margin_30 - base_margin_30) / base_margin_30
            if rel_margin_gain < min_margin_gain:
                decision = "NO-GO"
                reasons.append(
                    f"[{direction_name}] Margin pass@0.30 relative improvement ({rel_margin_gain * 100:.1f}%) "
                    f"is below required +{min_margin_gain * 100:.1f}% relative to baseline "
                    f"(base={base_margin_30:.4f}, ft={ft_margin_30:.4f})"
                )
        elif ft_margin_30 <= 0.0:
            decision = "NO-GO"
            reasons.append(
                f"[{direction_name}] Margin pass@0.30 ({ft_margin_30:.4f}) too low — model cannot separate pos/neg with margin 0.30"
            )

    # Absolute margin pass@0.10 floor check
    margin_pass_10 = dir_results["fine_tuned"].get("margin_pass@0.10", 0.0)
    if margin_pass_10 < min_margin_pass_10:
        decision = "NO-GO"
        reasons.append(
            f"[{direction_name}] Margin pass@0.10 ({margin_pass_10:.4f}) too low (< {min_margin_pass_10:.2f}) — model can't separate pos/neg"
        )

    dir_results["decision"] = decision
    dir_results["reasons"] = reasons
    return dir_results


def evaluate_model(
    model_path: str,
    data_dir: str,
    baseline_model: Optional[str] = None,
    batch_size: int = 64,
    k_values: list = None,
    margin_thresholds: list = None,
    direction: str = "default",
    min_recall_10: float = 0.80,
    max_country_gap: float = 0.05,
    min_margin_gain: float = 0.15,
    min_margin_pass_10: float = 0.60,
) -> Dict:
    """
    Full evaluation pipeline for the held-out country gate.

    Supports:
      - "default": evaluates the standard eval_*.json files
      - "us_to_india": evaluates primary split prefix
      - "india_to_us": evaluates reverse split prefix
      - "bidirectional": evaluates BOTH directions and enforces 2-way gate acceptance
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating model: {model_path} (direction: {direction})")
    logger.info(f"{'='*60}")

    model = SentenceTransformer(model_path)
    base_model = SentenceTransformer(baseline_model) if baseline_model else None

    # Resolve dynamic direction prefixes from country_split_info.json if present
    split_info_file = Path(data_dir) / "country_split_info.json"
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
    if not (Path(data_dir) / f"{pri_prefix}eval_queries.json").exists() and (Path(data_dir) / "us_train_india_eval_eval_queries.json").exists():
        pri_prefix = "us_train_india_eval_"
        rev_prefix = "india_train_us_eval_"

    results = {}
    all_reasons = []
    overall_decision = "GO"

    if direction == "bidirectional":
        directions = [
            (pri_prefix, f"{p_train.upper()}->{p_eval.upper()} (eval on {p_eval.upper()})"),
            (rev_prefix, f"{p_eval.upper()}->{p_train.upper()} (eval on {p_train.upper()})"),
        ]
        results["directions"] = {}
        for prefix, name in directions:
            res = _evaluate_single_direction(
                model=model,
                base_model=base_model,
                data_dir=data_dir,
                prefix=prefix,
                direction_name=name,
                batch_size=batch_size,
                k_values=k_values,
                margin_thresholds=margin_thresholds,
                min_recall_10=min_recall_10,
                max_country_gap=max_country_gap,
                min_margin_gain=min_margin_gain,
                min_margin_pass_10=min_margin_pass_10,
            )
            results["directions"][name] = res
            if res["decision"] != "GO":
                overall_decision = "NO-GO"
            all_reasons.extend(res.get("reasons", []))
    else:
        prefix = ""
        name = "default"
        if direction in ("us_to_india", "primary"):
            prefix = pri_prefix
            name = f"{p_train.upper()}->{p_eval.upper()}"
        elif direction in ("india_to_us", "reverse"):
            prefix = rev_prefix
            name = f"{p_eval.upper()}->{p_train.upper()}"

        res = _evaluate_single_direction(
            model=model,
            base_model=base_model,
            data_dir=data_dir,
            prefix=prefix,
            direction_name=name,
            batch_size=batch_size,
            k_values=k_values,
            margin_thresholds=margin_thresholds,
            min_recall_10=min_recall_10,
            max_country_gap=max_country_gap,
            min_margin_gain=min_margin_gain,
            min_margin_pass_10=min_margin_pass_10,
        )
        results = res
        overall_decision = res["decision"]
        all_reasons = res.get("reasons", [])

    del model
    if base_model:
        del base_model

    # -----------------------------------------------------------------------
    # Joint Go/No-Go Decision Report
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("JOINT GO/NO-GO GATE DECISION")
    logger.info(f"{'='*60}")

    results["decision"] = overall_decision
    results["reasons"] = all_reasons

    if overall_decision == "GO":
        logger.info("✓ DECISION: GO — All evaluated directions pass the held-out country gate!")
        logger.info("  The fine-tuned checkpoint can replace the off-the-shelf feature.")
    else:
        logger.info("✗ DECISION: NO-GO — Fine-tuned model fails held-out country gate")
        for reason in all_reasons:
            logger.info(f"  Reason: {reason}")
        logger.info("\n  Recommended actions:")
        logger.info("  1. Raise self_distillation_weight toward 0.15")
        logger.info("  2. Or drop LoRA rank to 32")
        logger.info("  3. Fall back to off-the-shelf BGE-M3 for Stage 2a-i")
        logger.info("  4. Evaluate Qwen3-Embedding-0.6B separately as Stage 2a-ii")

    output_name = "eval_results_bidirectional.json" if direction == "bidirectional" else "eval_results.json"
    output_path = Path(data_dir) / output_name

    def serialize_obj(obj):
        if isinstance(obj, dict):
            return {k: serialize_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize_obj(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialize_obj(results), f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bi-encoder fine-tune on held-out country (go/no-go gate)"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--data_dir", type=str, default="./prepared_data",
                        help="Path to prepared evaluation data")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Baseline model for comparison (e.g., BAAI/bge-m3)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--direction", type=str, default="default",
                        choices=["default", "us_to_india", "india_to_us", "bidirectional"],
                        help="Direction for held-out evaluation: default, us_to_india, india_to_us, bidirectional")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        baseline_model=args.baseline_model,
        batch_size=args.batch_size,
        direction=args.direction,
    )


if __name__ == "__main__":
    main()

