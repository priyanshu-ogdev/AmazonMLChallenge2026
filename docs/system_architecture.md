# Business Entity Resolution — Final Design and Execution Plan

**Status:** fully implemented and tested implementation contract  
**Last reviewed:** 2026-09-26  
**Scope:** Complete implementation and verification across all layers:
- Layer 0: streaming country-agnostic normalization and preprocessing (`src/normalize.py`, `src/data_builder.py`);
- Layer 1: multi-channel candidate generation and blocking (`src/blocking.py`);
- Layer 2: representation & features:
  - 2a-i: BGE-M3 LoRA fine-tuning and bidirectional held-out country evaluation (`src/train_bi_encoder.py`, `src/eval_bi_encoder.py`, `src/bge_features.py`);
  - 2a-ii: Qwen3-Embedding-0.6B auxiliary dense feature extraction (`src/qwen_features.py`);
  - 2b: (Stretch) Qwen3-0.6B Causal Generative Matcher (`docs/reference/04b_qwen3_generative_matcher_spec.md`);
  - 2c: deterministic lexical, phonetic, address, ambiguity, and blocker score diff features (`src/pair_features.py`);
- Layer 3: Grouped-OOF XGBoost scorer, monotonic constraints, anti-shortcut country masking, fold-safe TF-IDF, and leak-safe calibration (`src/scoring.py`, `src/calibration.py`);
- Layer 4: deterministic per-S1 macro F0.5 decision and singleton-safe submission assembly (`src/decision.py`).

## 1. Decision summary

The v1 system is a precision-first, multi-stage entity-resolution pipeline:

```text
TSV inputs
  -> Stage 0: country-agnostic normalization
  -> Stage 1: high-recall candidate generation / blocking
  -> Stage 2: pair features
       2a-i. fine-tuned BGE-M3 cosine similarity
       2a-ii. Qwen3-Embedding-0.6B cosine similarity (frozen, auxiliary feature)
       2b. (Stretch) Qwen3-0.6B causal generative matcher probability
       2c. deterministic lexical, address, phonetic, and source features
  -> Stage 3: calibrated GBM pair scorer
  -> Stage 4: per-entity F0.5 decision and singleton handling
  -> Stage 5: validated matching_results.tsv + candidate_pairs.tsv
```

The BGE-M3 LoRA configuration is **final for v1**: rank 64, rank-stabilized
scaling, all-linear adapters, cached multiple-negatives ranking loss, and a
0.10 self-distillation anchor. Do not reopen a rank sweep, full fine-tuning,
generative cross-matcher (Stage 2b), or DART before the end-to-end baseline is working.

Qwen3-Embedding-0.6B is **not Stage 2b and not a second training target** — it is
part of Stage 2a's feature set (**Stage 2a-ii**, subordinate to Stage 2a-i's own
bi-encoder work). Stage 2b is reserved exclusively for the cross-record generative
matcher (LoRA fine-tuned Qwen3-0.6B causal LM, stretch-only per
[`04b_qwen3_generative_matcher_spec.md`](reference/04b_qwen3_generative_matcher_spec.md)).
Stage 2a-ii has an inference-only implementation in `src/qwen_features.py`. It is
an auxiliary embedding feature, not an automatic fallback for a failed BGE adapter
(that fallback is unchanged BGE-M3 base). Its inclusion must be justified by
validation uplift, not benchmark numbers alone.

## 2. Evidence and source policy

The following sources were checked while reconciling the attached design
documents. Benchmark values are treated as supporting evidence, never as a
substitute for challenge validation:

| Source | Use in this plan |
|---|---|
| [PosIR, arXiv:2601.08363](https://arxiv.org/abs/2601.08363) | Supports multilingual retrieval comparison; the cited BGE-M3/Qwen3 French values must be reproduced in the paper table if quoted in a final report. |
| [Rank-Stabilization Scaling for LoRA, arXiv:2312.03732](https://arxiv.org/abs/2312.03732) | Supports `use_rslora=True`; it does not prove that rank 64 is optimal for this dataset. |
| [Back to Basics, arXiv:2311.09765](https://arxiv.org/abs/2311.09765) | Supports testing out-of-domain retrieval, not a guaranteed gain from LoRA. |
| [Qwen3-Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Confirms 0.6B size, 100+ languages, instruction-aware query encoding, last-token pooling example, and 32k context. |
| Sentence-Transformers documentation | Defines the CachedMNRL/GradCache behavior and supported trainer interfaces. |

Only the supplied challenge data may be used for training, validation, feature
mining, or augmentation. No external business lookup, geocoding, web search,
or identity database is permitted.

## 3. BGE-M3 Stage 2a contract

### Committed configuration

| Parameter | v1 value |
|---|---:|
| Base model | `BAAI/bge-m3` |
| Adapter | PEFT LoRA, `r=64`, `alpha=64`, `dropout=0.1`, `bias=none` |
| Scaling | rank-stabilized (`use_rslora=True`) |
| Targets | all linear layers |
| Objective | CachedMNRL + `0.10 *` cosine self-distillation |
| Training data | 50k US + 50k India entities, positive pairs only |
| Physical / GradCache batch | 48 / 16 |
| Max sequence length | 80 tokens |
| Learning rate / epochs | `2e-5` / 3 |
| Precision | bf16 with gradient checkpointing |

The frozen base model is the reference for the anchor loss. LoRA freezes base
weights, but this does not guarantee unchanged French behavior: the merged
adapter can still shift scores. That is why the gate below is mandatory.

### Required BGE gate

Run both directions, not only US→India:

1. Train on US, evaluate on India.
2. Train on India, evaluate on US.
3. Compare each run with off-the-shelf BGE-M3 on the identical query/corpus.
4. Record Recall@1/5/10/20/50, MRR, precision@10, mean/median hardest-negative
   margin, and the number of evaluated queries.

Accept only if both directions meet all of:

- Recall@10 ≥ 0.80;
- fine-tuned Recall@10 is no more than 0.10 below the corresponding baseline;
- margin pass rate at 0.10 ≥ 0.60;
- no country has a materially worse singleton or per-country retrieval slice.

If the gate fails, try **one** controlled recovery in this order: increase
distillation weight to 0.15, then reduce rank to 32. Keep the failed checkpoint
for comparison. If no candidate passes, use off-the-shelf BGE-M3 for Stage 2a and evaluate
Qwen separately as an auxiliary feature; do not silently deploy a failed
adapter.

## 4. Stage 1 — candidate generation

The complete Stage 1 implementation contract, scale plan, and recall gate are
in [`stage1_blocking.md`](stage1_blocking.md). The Stage 0 preprocessing
contract is in [`stage0_normalization.md`](stage0_normalization.md).

Blocking defines the recall ceiling, so it must be unioned rather than chained
with destructive intersections. For every S1 record, generate candidates from:

1. exact and normalized name keys;
2. character n-gram TF-IDF nearest neighbors;
3. address token/character retrieval;
4. phonetic name keys where the script supports them;
5. BGE-M3 dense retrieval;
6. optional Qwen dense retrieval.

Use canonical country as a retrieval partition only when both sides have a
non-empty value. Unknown countries remain valid partitions and missing-country
records use a global fallback; never hard-code an allow-list.
Deduplicate candidates by entity ID, retain source provenance and rank from
each blocker, and cap only after measuring recall. The initial target is
high-recall top-K per blocker (for example K=50) followed by a measured union
cap, not a hard-coded guess.

Before training the GBM, report:

- candidate recall at K against train ground truth, including singleton count;
- reduction ratio and candidates per S1 percentile distribution;
- recall by country and by S2/S3 source;
- the fraction of true matches absent from the union.

`candidate_pairs.tsv` must contain the final candidate set actually scored by
the matching model. Every emitted final match must be present in that file.

For the complete reasoning, trade-offs, invariants, and ablation protocol for
all three sub-stages, see [`stage2_features_and_embeddings.md`](stage2_features_and_embeddings.md).

## 5. Stage 2 — pair features

For each `(S1, candidate)` pair, `src/pair_features.py` computes deterministic
features without labels or external data:

- fine-tuned BGE cosine, base BGE cosine, and dense rank/provenance;
- Qwen cosine with **symmetric entity-resolution instructions** on both sides
  (or no instruction on both sides), never query-only prompting for one side;
- normalized exact-name and token-set similarities;
- character n-gram TF-IDF similarity;
- edit-distance ratios and length differences;
- address token overlap, character similarity, numeric-token agreement,
  postal-code agreement, and missingness indicators;
- country equality, source (S2/S3), blocker count, and rank features;
- hard-negative indicators: same-name/different-address and same-address/
  different-name conflicts.

The extractor includes IDs only as join keys; the GBM must not use them as
predictors. Do not feed row order to the model. Fit every learned
vectorizer/statistic on the training fold only.

### Qwen decision rule

Qwen is loaded with `Qwen/Qwen3-Embedding-0.6B` for inference only. Its model
card documents instruction-aware queries and last-token pooling; the
implementation applies the same entity-resolution instruction to both sides,
uses the Sentence-Transformers encoding path, and normalizes embeddings before
cosine similarity. `src/qwen_features.py` encodes each entity once and scores
exactly the final candidate set. Run an ablation:

```text
GBM(base lexical + BGE)
GBM(base lexical + BGE + Qwen)
```

Keep Qwen only if it improves entity-level macro F0.5 and does not reduce the
France-proxy slice or increase false merges. Otherwise omit it; the model is
not mandatory.

## 6. Stage 3 — scorer and calibration

Start with a regularized XGBoost/LightGBM-style logistic GBM using class
weighting derived from the training fold. Use entity-grouped, country-stratified
OOF folds: all pairs for an S1 entity stay in one fold.

Baseline controls:

- shallow trees, minimum child weight, subsampling, column subsampling;
- L1/L2 regularization;
- early stopping on entity-level validation F0.5/AUCPR;
- no target leakage from candidate generation or calibration folds.

Calibrate only on out-of-fold predictions. Count OOF positives as soon as
candidate pairs exist:

- fewer than 1,000 positives: Platt/logistic calibration;
- more than 10,000: isotonic is eligible;
- between them: fit both and select by pair-weighted OOF log loss; verify the
selected result with grouped, held-out-entity calibration metrics.

The current implementation and its exact artifact/inference contract are
documented in [`stage3_scoring_and_calibration.md`](stage3_scoring_and_calibration.md). It persists the
calibration parameters, feature list, threshold, fold diagnostics, and final
estimator count rather than only the model binary.
Calibration reliability is additionally checked with grouped cross-fitting
over S1 entities; the fit-on-OOF diagnostic must not be presented as an
unbiased estimate.

Stage 4 applies the saved threshold and emits one explicit row for every test
Source 1 entity, including zero-candidate singletons; see
[`stage4_decision_and_singletons.md`](stage4_decision_and_singletons.md).

Do not stack a custom beta-weighted objective on top of `scale_pos_weight`.
Escalate to beta-weighted log loss only when hard-negative separation remains
poor after the baseline, and compare it against the baseline on the same OOF
folds. DART is conditional only if the held-out-country diagnostic shows a
persistent generalization gap after ordinary regularization; it is not the
default booster.

## 7. Stage 4 — decision policy

Optimize the final threshold for the competition's macro F0.5, with each S1
entity contributing equally and singletons included. A global threshold is the
default; only add country/source thresholds if grouped validation proves a
stable gain without leakage.

Use conservative abstention rules for borderline pairs, and preserve the
possibility of multiple matches per S1. Never force a match. A singleton is a
valid prediction and receives full credit only when the predicted list is
empty.

## 8. Data improvements restored from the original plan

Two inexpensive safeguards are required before calling the BGE layer complete:

1. **Blocking-derived hard negatives.** After Stage 1 exists, add top-ranked
   non-ground-truth candidates to training pairs. Keep them country-balanced
   and exclude any known positive for the anchor. In-batch negatives remain the
   baseline; hard negatives are an explicit second run.
2. **Same-domain self-augmentation.** Create limited, deterministic variants
   from the supplied US/India records only: punctuation/legal-suffix changes,
   whitespace/ordering noise, field dropout, and address abbreviation noise.
   Preserve the original positive identity, cap the augmentation ratio, and
   report an ablation. Do not synthesize French or import external patterns.

These are data-side upgrades, not a reason to reopen the LoRA architecture.

## 9. Execution order and stop rules

1. Verify the current BGE module on a tiny CPU fixture and fix interface/runtime
   issues before a GPU run.
2. Build normalization, blockers, and candidate-recall diagnostics.
3. Run the two-direction BGE gate; keep the baseline fallback.
4. Add hard negatives and self-augmentation only after the baseline pipeline
   produces valid candidates; retain the better gate-passing checkpoint.
5. Generate pair features, then entity-grouped OOF GBM predictions.
6. Select calibration and threshold using OOF data; run the France-proxy and
   source/country slices.
7. Produce and validate both submission files.
8. Only if steps 1–7 are complete: test Qwen embedding ablation, SHAP, focal/beta loss,
   or Qwen3 generative matcher (Stage 2b). These are stretch work, not dependencies of v1.

The stop condition is a reproducible, validated submission package—not a
collection of individually impressive model metrics.

## 10. Reproducibility and acceptance checklist

- [ ] Dataset paths, row counts, country counts, and random seeds recorded.
- [ ] No external data or lookup service used.
- [ ] BGE gate results include both directions and the off-the-shelf baseline.
- [ ] Candidate recall and reduction ratio are reported before scoring.
- [ ] OOF folds are grouped by S1 entity and stratified by country/source.
- [ ] Calibration is fit only on OOF predictions and uses the positive-count rule.
- [ ] Final matches are a subset of `candidate_pairs.tsv`.
- [ ] Every test S1 appears exactly once; S2/S3 IDs are valid and deduplicated.
- [ ] `utils/validate_submission.py` passes before packaging.
- [ ] Model licenses and downloaded model revisions are recorded.
