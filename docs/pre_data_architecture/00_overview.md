# Business Entity Resolution — Pipeline Overview

## Task recap
Link records from three sources (S1 = clean reference, S2/S3 = noisy vendor feeds, no shared IDs,
name+address only) so that every S1 entity maps to zero, one, or many matching S2/S3 records.
Scored on macro F_0.5 → **false positives cost ~2x a false negative**. No external lookups allowed.

## Pipeline shape (3 stages)

1. **Blocking** — cheap candidate generation, optimized for recall. Sets the hard ceiling on
   everything downstream: a match blocking drops can never be recovered by the matcher.
2. **Matching** — a classifier/reranker over candidate pairs, optimized for precision, with an
   explicit "no match" / abstain option (critical for singletons).
3. **(Optional) Fine-tuning** — adapting a small open encoder to this domain's specific noise
   (abbreviations, corporate suffixes, same-name-different-address confusions) using the
   competition's own `train_ground_truth.tsv` as supervision.

## Status of this design
This document set is a **pre-data architecture**: every stage, decision rule, and safeguard is
specified, but a short list of empirical questions are correctly left open because only the real
TSVs can answer them:
- Which embedding model wins on *this* data (BGE-M3 vs Qwen3-Embedding-0.6B vs a fine-tuned variant)
- Which blocking key strategy achieves acceptable recall at acceptable candidate-set size
- The actual precision/recall operating threshold for the F_0.5 tradeoff

These are flagged inline wherever they appear rather than presented as settled.

## Files in this folder
- `01_blocking_stage.md` — candidate generation design
- `02_matching_stage.md` — classification/reranking design, singleton handling, threshold policy
- `03_embedding_models.md` — verified model comparison (specs, licenses, benchmarks) and choice rationale
- `04_citations.md` — every external claim used in this design, with what was independently verified
  and how, including one correction found on re-verification (LinkTransformer's actual license)
- `05_hyperparameter_verification_status.md` — the line between what's verifiable now (facts about
  models/papers) and what can only be verified after the real dataset is available (thresholds,
  hyperparameters, F_0.5 score) — read this before treating any number in this design as final
- `06_optimization_design.md` — the engineering reasoning behind each design choice, independent of
  citation provenance, plus the one place a license actually constrains what code can be reused
  (LinkTransformer is GPL-3.0 — reference its API shape, don't vendor its code)
- `07_prd.md` — product requirements: functional/non-functional requirements traced to either the
  challenge spec or a specific reasoning source, deliverables, risks
- `08_research_plan.md` — staged build-and-measure plan (Stage 0–8) with explicit go/no-go gates;
  nothing past Stage 2 is kept unless it beats a measured baseline on held-out data

## Verification note
Several claims in this design (model licenses, parameter counts, benchmark scores, and the
existence/content of the "Sodhana" entity-resolution fine-tuning paper) were re-checked directly
against primary sources (Hugging Face model cards, the arXiv paper itself) before being written
into this documentation, rather than trusted from earlier summarization. See `04_citations.md` for
what was checked and what remains unverified pending the real dataset.
