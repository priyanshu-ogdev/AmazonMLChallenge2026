# Architecture

## Five stages

| Stage | Purpose | Output |
|---|---|---|
| 0 — Ingestion | Normalize name/address text, country-agnostically | Normalized records |
| 1 — Blocking | Generate a recall-safe candidate shortlist per S1 entity | `candidate_pairs.tsv` |
| 2 — Features | Score every candidate on multiple independent signals | Feature columns per candidate |
| 3 — Matching | Combine all features into one calibrated match probability | Calibrated score |
| 4 — Threshold | Convert scores into keep/discard decisions, per entity | Match/no-match per candidate |
| 5 — Output | Assemble and validate the submission | `matching_results.tsv` |

## Stage 0 — Ingestion and normalization

Lowercase, strip punctuation, collapse whitespace. Legal-suffix and address-abbreviation canonicalization runs **bidirectionally** (Corp↔Corporation, Rd↔Road, &↔and). Landmark phrases ("Near SBI ATM") are stripped from similarity computation but flagged separately as a weak auxiliary signal. Structural sub-fields (street number, postal code, city/state) are extracted with data-driven heuristics, not hardcoded per-country branching — the one design rule in this stage that exists specifically because France has zero representation in training data and any US/India-specific logic here would silently degrade on it. Per the official problem statement, `country` must be treated as an open set of string labels — never hard-coded, filtered, or one-hot encoded to only `{US, India}` — and a record's source is determined by its `entity_id` prefix (`S1-`/`S2-`/`S3-`) and which file it appears in; there is no separate source column to parse.

## Stage 1 — Blocking

**Model: BGE-M3, kept stock or only lightly domain-adapted.** This is a structural choice, not one arm of an ablation — BGE-M3 is a bidirectional encoder that emits dense, sparse, and ColBERT-style vectors in one forward pass, so its sparse output is a free BM25-equivalent that a heavier or decoder-style model wouldn't provide. Blocking sets the pipeline's recall ceiling: nothing downstream can recover an entity dropped here, so this stage is deliberately kept loose (recall-favoring) rather than sharply discriminative.

Combined with: token inverted index, character TF-IDF retrieval, phonetic
blocking where script-safe, and address-token blocking. All strategies are
**unioned**, then deterministically ranked and capped. Country is a partition
optimization only when both sides have a non-empty canonical value; missing or
unknown-country records retain a global fallback. That final set — not an
earlier pass — is exactly what gets written to `candidate_pairs.tsv`, and every
ID in `matching_results.tsv` must appear there too. Concrete top-K, floor, and
cap values, plus the scale plan, are in [`docs/stage1_blocking.md`](../stage1_blocking.md) and
[`02_parameters_table.md`](02_parameters_table.md).

**A dedicated blocking-recall audit gate is part of this stage, not an afterthought.** Because blocking sets the recall ceiling for everything downstream, its recall should be measured directly against `train_ground_truth` — what fraction of true matches actually appear somewhere in the unioned candidate set — sliced by country, before any time is spent tuning Stage 2 or 3. This is the one measurement in the whole pipeline where a bad result can't be compensated for later, so it's worth confirming first rather than discovering only after the matching stage has already been tuned around whatever recall blocking happened to produce. If recall comes in measurably worse for the held-out-country proxy than for the in-domain countries, the fix belongs here (strengthening phonetic/address-token blocking specifically) — not in the matching stage, which has no mechanism to recover an entity blocking never retrieved.

## Stage 2 — Feature engineering (three independent sub-stages, no interdependency)

**2a — Bi-encoder cosine similarity.** A fine-tuned dense encoder scores each candidate pair. **v1 default: BGE-M3 LoRA fine-tuned (rank 64, rslora, all-linear, CachedMNRL + 0.10 self-distillation)**, gated by the two-direction held-out-country retrieval check before being trusted over the off-the-shelf fallback — see [`04_bge_m3_training_spec.md`](04_bge_m3_training_spec.md) and [`stage2_features_and_embeddings.md`](../stage2_features_and_embeddings.md) for the complete rationale and gate criteria. Qwen3-Embedding-0.6B is an **inference-only auxiliary feature (Stage 2b)** included only if an ablation proves complementary value; it is not the automatic fallback for a failed BGE adapter (that fallback is unchanged BGE-M3 base).

**2b — Cross-encoder (Ditto-style) probability.** A fine-tuned sequence-pair classifier scores each candidate pair jointly. See [`04_bge_m3_training_spec.md`](04_bge_m3_training_spec.md). **Cut from v1** — stretch goal only, see [`01_v1_baseline_plan.md`](01_v1_baseline_plan.md).

**2c — Hand-crafted features.** String similarity (Levenshtein, Jaro-Winkler, token-set Jaccard, TF-IDF cosine), address similarity (postal/city/street-number match flags), phonetic match flags, blocking-list rank position, an "ambiguity" feature (how many other candidates for the same S1 entity score similarly high), source-as-categorical (S2 vs. S3), and a **derived** country-match flag (never raw country identity — see [`04_bge_m3_training_spec.md`](04_bge_m3_training_spec.md)'s GBM section for why).

## Stage 3 — Matching

A gradient-boosted model (XGBoost, primary choice) consumes every Stage 2 feature and produces one calibrated match probability per candidate. Full loss, regularization, and imbalance-handling detail is in [`04_bge_m3_training_spec.md`](04_bge_m3_training_spec.md).

## Stage 4 — Threshold

The GBM's calibrated output is thresholded per S1 entity, keeping **all** candidates above the cutoff (not top-1, since one entity can have several true matches). The threshold is selected via k-fold cross-validation to maximize F₀.₅ specifically — not F1, not accuracy — which pushes it above a naive 0.5 given the metric's precision weighting. **This k-fold protocol must include a held-out-country fold, not just random splits** — see [`04_bge_m3_training_spec.md`](04_bge_m3_training_spec.md)'s GBM section for why: the GBM is the component most directly exposed to overfitting on US/India-specific patterns, and a random split alone can't surface that.

## Stage 5 — Output

Assemble `matching_results.tsv` (exactly one row per S1 test entity, no duplicate IDs, no self-matches, no fabricated IDs) and `candidate_pairs.tsv` (same structural requirements, and a strict superset of whatever appears in the results file). Run `utils/validate_submission.py` locally before any leaderboard submission.

**Two distinct deliverables, not one bundle produced on every run.** Per the official problem statement, these two files serve different purposes and go out on different schedules:
- **Leaderboard upload (every submission during the challenge):** `matching_results.tsv` only. This is what drives the public/private leaderboard score. `candidate_pairs.tsv` is not uploaded here and doesn't need to be regenerated or re-validated every time you push a leaderboard attempt.
- **Final submission package (once, and only required in full for teams under final review):** the full zip — `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `code/business_entity_resolution/` (runnable pipeline), and the methodology document. `candidate_pairs.tsv` only needs to exist and be internally consistent (superset of the results file) at this point, not on every intermediate leaderboard run.

Don't burn per-submission time regenerating and validating `candidate_pairs.tsv` against every leaderboard attempt — validate it once the pipeline is stable and again before the final package is zipped.

## Dependency structure (matters for both correctness and inference scheduling)

Stage 1 must complete before Stage 2 starts. Stages 2a, 2b, and 2c have no dependency on each other — only on Stage 1's candidate set. Stage 3 is a hard synchronization point: it cannot score a row until every Stage 2 feature column for that row exists. Stage 4 depends only on Stage 3's output. See [`06_inference_and_latency.md`](06_inference_and_latency.md) for how this maps onto actual scheduling and hardware.

## What's fixed vs. what's genuinely still open

**Fixed:** the five-stage shape and dependency order; BGE-M3 for blocking specifically; BGE-M3 dense head with LoRA rank 64 for Stage 2a (v1 scope, given confirmed RTX 3060 12GB and time budget — see [`01_v1_baseline_plan.md`](01_v1_baseline_plan.md)'s fine-tuning revision), gated by the held-out-country pass/fail check before being trusted over the off-the-shelf fallback; the country-handling rule (derived match flags only, never raw country identity, no monotonic constraint on the match flag); calibration before thresholding on every scored component; no external data anywhere in training.

**Open, by design, until real data exists:** whether full fine-tuning would beat the committed LoRA-64 configuration for the bi-encoder (not swept in v1 — see [`04_bge_m3_training_spec.md`](04_bge_m3_training_spec.md)), whether Ditto is worth adding as a stretch goal, the GBM's DART-vs-standard and loss-escalation decisions, and the actual F₀.₅ threshold value. Every one of these is resolved by a specified validation protocol, not by further design discussion — see [`08_open_decisions_log.md`](08_open_decisions_log.md).
