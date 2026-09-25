# Inference

## Sequence and dependencies

| Step | What runs | Depends on |
|---|---|---|
| Load (v1 baseline) | BGE-M3 (blocking), BGE-M3 dense head (LoRA-merged, rank 64 — Stage 2a primary), Qwen3-Embedding-0.6B (off-the-shelf, second Stage 2a feature if time allowed, or the fallback if the held-out-country gate failed), GBM + calibrator + threshold — no Ditto | — |
| Load (target design, once stretch goals land) | BGE-M3, final bi-encoder (LoRA-merged, or the full fine-tuned checkpoint if that wins the ablation), final Ditto (same), GBM + calibrator + threshold | — |
| Stage 0 | Normalize test records | Load |
| Stage 1 | BGE-M3 dense+sparse + token/phonetic/address blocking, union | Stage 0 |
| Stage 2a/2b/2c | Bi-encoder cosine, Ditto probability, hand-crafted features | Stage 1 only — no interdependency between 2a, 2b, 2c |
| Stage 3 | GBM scoring + calibration | **All** of Stage 2 — hard sync point |
| Stage 4 | Apply F₀.₅-optimized threshold, keep-all-above per entity | Stage 3 |
| Stage 5 | Assemble and validate submission | Stage 4 |

All models load once, up front, in a single continuous run. On one GPU this mostly means no repeated load/unload I/O between stages, not literal simultaneous execution — the hardware still processes one model's forward pass at a time. Stages 2a, 2b, and 2c can be scheduled in any order or interleaved; Stage 3 cannot start until every Stage 2 feature column exists for a row.

**Stage 1 at inference needs its own batching plan, not a single monolithic pass.** Blocking has to encode the entire S2+S3 test table — potentially a large number of records — with BGE-M3 before any candidate generation can happen. That's a different scale of workload than Stages 2a/2b/2c, which only ever run over the much smaller, already-shortlisted candidate set. In practice this means: encode S2/S3 in fixed-size batches rather than attempting one pass over the full table, and build the dense/sparse retrieval structures (an approximate nearest-neighbor index for the dense vectors, an inverted index for the sparse/token signals) incrementally as batches complete, rather than assuming the full embedding matrix and search structure fit and build in one step. This is a real implementation detail this design has specified the *behavior* of (dense+sparse union, top-K per entity) without yet specifying the *execution* shape at the table sizes involved — worth pinning down once real data sizes are known, not assumed away.

## Hardware footprint

LoRA's small trainable-parameter count (whatever rank the ablation settles on) is a **training-time** property — it does not shrink what has to be loaded and run at inference. Before inference, every LoRA adapter is merged into its base model's weights (`W' = W + BA`), producing a plain dense model with zero adapter overhead. Worst-case total footprint: BGE-M3 (568M) + Qwen3-Embedding-0.6B (~600M, only if it wins the Stage 2a ablation) + XLM-RoBERTa-base (279M) ≈ 1.45B parameters, comfortably under 3GB at bf16 — well within a single RTX 3060's 12GB, but because these are all sub-billion-parameter models, not because of LoRA's adapter size.

## Train/inference-mode switches — easy to lose track of, all must flip together

- **Dropout off.** Every dropout rate specified for the bi-encoder and Ditto (encoder-internal, classification-head) is training-only. Both models run in eval mode at inference.
- **Country-match masking off.** The deliberate "sometimes feed missing instead of the real country-match value" GBM regularizer is training-only. At inference every row gets its real, computed country-match value.
- **Single final checkpoint only.** The 5-fold OOF cycle exists to generate leakage-safe *training* features. At inference, only the checkpoint trained on the full training set is used — never a fold-holdout checkpoint, which saw only 80% of the labeled data and exists solely to have produced an honest score for its own held-out fold during GBM training.

## Consistency requirement

Whatever is written to `candidate_pairs.tsv` in Stage 1 must be exactly what Stages 2 through 4 consume — no downstream re-filtering that silently narrows the candidate set without updating that file. Every ID that appears in `matching_results.tsv` must appear in `candidate_pairs.tsv`, or the validator flags a pipeline bug. Worth checking explicitly at implementation time, not just designing for.
