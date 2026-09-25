"""
Held-out country evaluation for the bi-encoder fine-tune.

This is the GO/NO-GO GATE: if the fine-tuned model's retrieval quality
on the held-out country is meaningfully worse than in-domain, the
anti-forgetting stack is insufficient and the fine-tune should not be trusted.

Protocol (from docs/LAYER_2_OVERVIEW.md):
1. Load the fine-tuned bi-encoder
2. Encode eval queries (held-out country S1 entities)
3. Encode eval corpus (held-out country S2/S3 entities)
4. Compute Recall@K and MRR
5. Compare against off-the-shelf baseline (optional)
6. Report go/no-go decision

Recovery if gate fails:
- Raise self_distillation_weight toward 0.15
- Or drop LoRA rank to 32
- Or fall back to off-the-shelf BGE-M3 for the Stage 2a feature
- Evaluate Qwen3-Embedding-0.6B separately as an auxiliary Stage 2b feature

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
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_eval_data(data_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load pre-built IR evaluation data."""
    data_path = Path(data_dir)

    with open(data_path / "eval_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(data_path / "eval_corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(data_path / "eval_relevant.json", "r", encoding="utf-8") as f:
        relevant_raw = json.load(f)

    relevant_docs = {k: set(v) for k, v in relevant_raw.items()}

    logger.info(f"Loaded eval data: {len(queries)} queries, {len(corpus)} corpus, "
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

    This is the cheap proxy evaluation from docs/LAYER_2_OVERVIEW.md:
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

        # Best positive score
        pos_indices = [corpus_id_to_idx[cid] for cid in relevant if cid in corpus_id_to_idx]
        neg_indices = [idx for idx in range(len(corpus_ids))
                       if corpus_ids[idx] not in relevant]

        if not pos_indices or not neg_indices:
            continue

        best_pos = max(scores[idx] for idx in pos_indices)
        best_neg = max(scores[idx] for idx in neg_indices)
        margin = best_pos - best_neg
        margins.append(margin)

    margins = np.array(margins)
    for threshold in margin_thresholds:
        pass_rate = np.mean(margins >= threshold) if len(margins) > 0 else 0.0
        pass_rates[f"margin_pass@{threshold:.2f}"] = pass_rate

    pass_rates["mean_margin"] = float(np.mean(margins)) if len(margins) > 0 else 0.0
    pass_rates["median_margin"] = float(np.median(margins)) if len(margins) > 0 else 0.0

    return pass_rates


def evaluate_model(
    model_path: str,
    data_dir: str,
    baseline_model: Optional[str] = None,
    batch_size: int = 64,
    k_values: list = None,
    margin_thresholds: list = None,
) -> Dict:
    """
    Full evaluation pipeline for the held-out country gate.

    1. Load fine-tuned model and encode queries/corpus
    2. Compute retrieval metrics (Recall@K, MRR)
    3. Compute margin analysis (score-gap pass rate)
    4. Optionally compare against off-the-shelf baseline
    5. Make go/no-go recommendation
    """
    from sentence_transformers import SentenceTransformer

    if k_values is None:
        k_values = [1, 5, 10, 20, 50]

    queries, corpus, relevant_docs = load_eval_data(data_dir)

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    results = {}

    # -----------------------------------------------------------------------
    # Evaluate fine-tuned model
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating fine-tuned model: {model_path}")
    logger.info(f"{'='*60}")

    model = SentenceTransformer(model_path)
    logger.info(f"Encoding {len(query_texts)} queries...")
    q_embs = model.encode(query_texts, batch_size=batch_size, show_progress_bar=True,
                          normalize_embeddings=True)
    logger.info(f"Encoding {len(corpus_texts)} corpus docs...")
    c_embs = model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True,
                          normalize_embeddings=True)

    retrieval_metrics = compute_retrieval_metrics(
        q_embs, c_embs, query_ids, corpus_ids, relevant_docs, k_values
    )
    margin_metrics = compute_margin_analysis(
        q_embs, c_embs, query_ids, corpus_ids, relevant_docs, margin_thresholds
    )

    results["fine_tuned"] = {**retrieval_metrics, **margin_metrics}
    del model  # Free VRAM

    logger.info("\nFine-tuned model metrics:")
    for k, v in results["fine_tuned"].items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # -----------------------------------------------------------------------
    # Evaluate baseline (optional)
    # -----------------------------------------------------------------------
    if baseline_model:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating baseline model: {baseline_model}")
        logger.info(f"{'='*60}")

        base_model = SentenceTransformer(baseline_model)
        base_q_embs = base_model.encode(query_texts, batch_size=batch_size,
                                        show_progress_bar=True, normalize_embeddings=True)
        base_c_embs = base_model.encode(corpus_texts, batch_size=batch_size,
                                        show_progress_bar=True, normalize_embeddings=True)

        base_retrieval = compute_retrieval_metrics(
            base_q_embs, base_c_embs, query_ids, corpus_ids, relevant_docs, k_values
        )
        base_margin = compute_margin_analysis(
            base_q_embs, base_c_embs, query_ids, corpus_ids, relevant_docs, margin_thresholds
        )

        results["baseline"] = {**base_retrieval, **base_margin}
        del base_model

        logger.info("\nBaseline model metrics:")
        for k, v in results["baseline"].items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # -----------------------------------------------------------------------
    # Go/No-Go Decision
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("GO/NO-GO GATE DECISION")
    logger.info(f"{'='*60}")

    ft_recall_10 = results["fine_tuned"].get("recall@10", 0)
    decision = "GO"
    reasons = []

    # Check absolute threshold
    if ft_recall_10 < 0.80:
        decision = "NO-GO"
        reasons.append(f"Recall@10 ({ft_recall_10:.4f}) below minimum threshold (0.80)")

    # Check against baseline if available
    if baseline_model and "baseline" in results:
        base_recall_10 = results["baseline"].get("recall@10", 0)
        gap = base_recall_10 - ft_recall_10
        if gap > 0.10:
            decision = "NO-GO"
            reasons.append(
                f"Fine-tuned Recall@10 ({ft_recall_10:.4f}) is {gap:.4f} worse than "
                f"baseline ({base_recall_10:.4f}), exceeding max gap (0.10)"
            )
        elif gap > 0.05:
            reasons.append(
                f"WARNING: Fine-tuned Recall@10 ({ft_recall_10:.4f}) is {gap:.4f} worse "
                f"than baseline ({base_recall_10:.4f}). Close to threshold."
            )

    # Check margin health
    margin_pass = results["fine_tuned"].get("margin_pass@0.10", 0)
    if margin_pass < 0.60:
        decision = "NO-GO"
        reasons.append(f"Margin pass@0.10 ({margin_pass:.4f}) too low — model can't separate pos/neg")

    results["decision"] = decision
    results["reasons"] = reasons

    if decision == "GO":
        logger.info("✓ DECISION: GO — Fine-tuned model passes held-out country gate")
        logger.info("  The fine-tuned checkpoint can replace the off-the-shelf feature.")
    else:
        logger.info("✗ DECISION: NO-GO — Fine-tuned model fails held-out country gate")
        for reason in reasons:
            logger.info(f"  Reason: {reason}")
        logger.info("\n  Recommended actions:")
        logger.info("  1. Raise self_distillation_weight toward 0.15")
        logger.info("  2. Or drop LoRA rank to 32")
        logger.info("  3. Fall back to off-the-shelf BGE-M3 for Stage 2a")
        logger.info("  4. Evaluate Qwen3-Embedding-0.6B separately as Stage 2b")

    # Save results
    output_path = Path(data_dir) / "eval_results.json"
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                               for kk, vv in v.items()}
        else:
            serializable[k] = v
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
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

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        baseline_model=args.baseline_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
