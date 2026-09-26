# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> Use [`docs/02_system_architecture.md`](docs/02_system_architecture.md) as the implementation
> contract. Replace result placeholders only with measured validation results.

---

## 1. Executive Summary

This solution uses a precision-first staged pipeline: country-agnostic
normalization, unioned high-recall blocking, dense and lexical pair features,
an entity-grouped calibrated GBM, and macro-F0.5 thresholding with singleton
handling. BGE-M3 LoRA provides the primary dense retrieval feature, while
Qwen3-Embedding-0.6B is an inference-only auxiliary feature retained only if
its ablation improves validation without increasing false merges.

---

## 2. Methodology

### 2.1 Problem Analysis

Describe measured noise patterns from the supplied training data: business-name
abbreviations and typos, address reordering and missing components, numeric
tokens, country-specific formatting, and the distribution of singleton S1
entities. Do not infer French behavior from external data.

### 2.2 Solution Strategy

The pipeline follows [`docs/02_system_architecture.md`](docs/02_system_architecture.md): Stage 0 normalization, Stage 1
unioned blocking, Stage 2 pair features, Stage 3 entity-grouped OOF GBM and
calibration, Stage 4 macro-F0.5 thresholding, and Stage 5 submission
validation. The BGE adapter is accepted only after both held-out-country
directions pass the retrieval gate.

**Approach Type:** Blocking + calibrated classifier (hybrid dense and lexical)
**Core Innovation:** Multilingual BGE-M3 LoRA retrieval protected by rank
limits, self-distillation, country-balanced data, hard negatives, and a
held-out-country acceptance gate, combined with precision-first pair scoring.

### 2.3 Rule Compliance & External Data Independence

- **No External Lookups**: No external databases, registry queries, online geocoders, or API lookups are performed at any stage. All processing is self-contained.
- **Linguistic Token Normalization**: Rule-based normalization maps in `src/normalize.py` (`LEGAL_SUFFIX_MAP`, `ADDRESS_SUFFIX_MAP`, `_DIRECTION_ABBREV`) are closed-vocabulary language token contractions (e.g., `Corp` -> `corporation`, `Rd` -> `road`, `St` -> `street`). They serve solely as deterministic text canonicalization (akin to stemming or lowercasing), not business identity lookups or external business databases, fully honoring competition constraints.
- **Model Size and License Constraints**: All foundation models (`BAAI/bge-m3` [MIT], `Qwen/Qwen3-Embedding-0.6B` [Apache-2.0], and `Qwen/Qwen3-0.6B` [Apache-2.0]) are <= 0.6B parameters (well below the 8B parameter ceiling) and permissible under commercial open-source licenses.

---

## 3. Candidate Generation (Blocking)

The candidate set is the union of normalized/exact name keys, character TF-IDF
retrieval, address retrieval, phonetic keys, BGE dense retrieval, and optional
Qwen dense retrieval. Candidate provenance and rank are retained as features.
Candidate recall is measured by country and source before model training.
`candidate_pairs.tsv` is exactly the final set passed to the scorer.

- **Blocking keys used:** [fill with measured keys and K values]
- **Candidate pairs generated:** [fill with measured total and distribution]
- **How you ensured true matches were not lost:** [fill with measured recall,
  including US/India slices and singleton count]

---

## 4. Matching Model

**Features used:**
- Name: normalized exactness, token/character similarity, edit distance,
  phonetic keys, and dense BGE/Qwen cosine features
- Address: token/character overlap, numeric-token and postal-code agreement,
  missingness, and edit distance
- Other: country/source, blocker provenance/rank, and conflict indicators

**Model type:** Regularized GBM over pair features; BGE-M3 is the Stage 2a
feature generator, not the final match decision-maker.
**Threshold selection & assignment method:**
- Entity-level macro-F0.5 optimization on out-of-fold predictions, with singletons included.
- Injective-aware threshold selection: τ* is tuned directly under greedy 1-to-N bipartite assignment matching the production decision layer, tracking the injective lift diagnostic.
- Calibration: Sigmoid/isotonic calibration fit on out-of-fold predictions; an honest leak-free cross-fitted calibration diagnostic (`cross_fitted_calibration_metrics`) is tracked in metadata.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [measured OOF score]
- **Candidate recall and reduction ratio:** [measured values]
- **BGE held-out-country gate:** [both directions and baseline comparison]
- **Qwen ablation:** [measured delta, or explain why it was omitted]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion

Summarize the measured performance, the effect of the dense features, and the
remaining error modes. State explicitly that only supplied challenge data was
used and that the submission validator passed.

---

## Appendix

### A. Code Artefacts

The runnable code ships under `code/business_entity_resolution/`. The BGE
Stage 2a entry points are `src/data_builder.py`,
`src/train_bi_encoder.py`, and `src/eval_bi_encoder.py`. Summarize the final
end-to-end entry point(s) used to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results

Include candidate-recall plots, country/source slices, calibration diagnostics,
threshold curves, and representative false-positive/false-negative examples.

---

**Note:** Do not report planned metrics as achieved results. The authoritative
stage contracts and stop rules are in [`docs/02_system_architecture.md`](docs/02_system_architecture.md).
