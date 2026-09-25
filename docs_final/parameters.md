# Final v1 parameters — GBM, calibration, blocking

Everything else in this doc set specifies *protocols* (validation grids, resolution methods). This file specifies the actual **starting values** to write into code for the v1 build, since a protocol alone doesn't compile. These are defensible defaults grounded in the regularization/citations reasoning already in `training.md`/`regularization.md` — not re-derivations of that reasoning, and not claims that they're already optimal. Where `open-decisions.md` lists something as genuinely open, the value here is the starting point for that grid, not a substitute for running it.

## GBM (Stage 3) — the only model actually trained in v1

Since v1 cuts Ditto and encoder fine-tuning, the GBM is the **only** component that learns from labeled data this cycle. Its parameters matter more than usual for exactly that reason.

```python
import xgboost as xgb

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",          # not logloss/accuracy — see training.md
    "tree_method": "hist",           # fast, exact-enough at this data scale
    "max_depth": 4,                  # start of the specified 3–6 range, not the edge
    "learning_rate": 0.03,           # lowered from an earlier 0.05 default — see rationale below
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,           # starting point — raise if leaves are still fitting single rare positives
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": None,        # compute at runtime: (neg_count / pos_count) on the real candidate set — never hardcode
    "monotone_constraints": None,    # build as a tuple matching feature column order: +1 for every similarity feature (string sim, address sim, phonetic, bi-encoder cosine), 0 for country-match flag and any non-monotonic feature (ambiguity count, source-as-categorical, blocking rank)
    "n_estimators": 1000,            # raised to pair with the lower learning rate — see rationale below
    "seed": 42,
}

# Early stopping: 50 rounds (raised from 30, to match the lower learning rate and
# higher round ceiling — a shorter patience window risks stopping before a slow,
# small-step optimizer has actually converged), on AUCPR, on the *random* k-fold
# validation set. Track the held-out-country fold's AUCPR in parallel, don't
# early-stop on it — use it only as the diagnostic gap check from the GBM audit
# in training.md.
early_stopping_rounds = 50
```

**Why `learning_rate=0.03` / `n_estimators=1000` rather than the earlier `0.05`/`500` pairing — checked against the actual source of the shrinkage/subsampling idea, not just community defaults.** Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine," is the original source of the shrinkage parameter and states directly that smaller learning rates (λ ≲ 0.1) empirically produce better generalization error, with a stated caveat: the optimal iteration count doesn't scale linearly with a smaller learning rate (halving λ does *not* simply mean doubling the needed rounds — it must be found by validation, i.e. by early stopping, not by a fixed formula). Friedman (2002), "Stochastic Gradient Boosting," is the original source for the subsampling mechanism `subsample`/`colsample_bytree` implement, and found it "greatly improved" both accuracy and speed over deterministic boosting — the basis for keeping both at 0.8–0.9 rather than 1.0. Practically: at the small candidate-table sizes this dataset almost certainly has, 1000 rounds at `learning_rate=0.03` costs seconds, not minutes, so there's no real time-budget tradeoff in the 72-hour window for choosing the more conservative pairing — early stopping picks the actual round count, this ceiling is just generous enough not to cut it off early. This also lines up with a published range from an unrelated imbalanced-tabular-classification study (`learning_rate ∈ [0.025, 0.05, 0.075]`, `max_depth ∈ [3, 5]`, `subsample ∈ [0.8–0.95]`) — independent confirmation that the neighborhood this design already picked is a reasonable one, not an outlier choice.

**Tuning order, if the quick sweep mentioned below is run: `learning_rate` → `max_depth` → `min_child_weight`, sequentially, not a joint grid.** This is the standard practitioner sequence (each parameter tuned holding the prior ones fixed at their just-found best value) rather than a full cross-product search — cheaper, and appropriate here since the 72-hour window doesn't have room for an exhaustive joint sweep. The earlier "12-combination joint grid" suggestion is downgraded to a fallback only if the sequential sweep's result still looks off (e.g. the train/validation AUCPR gap stays wide after picking each parameter in sequence).

## Calibration

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Decide method by actual OOF positive count, not by assumption:
#   < 1,000 OOF positives  -> Platt (LogisticRegression on the raw score, 1 feature)
#   > 10,000 OOF positives -> Isotonic
#   in between             -> fit both, pick by held-out log loss / Brier score
```

Fit on **out-of-fold** predictions (5-fold), never on the training-fold predictions directly — an in-fold fit calibrates against scores the GBM has already memorized, which defeats the purpose. Store the fitted calibrator alongside the final full-data GBM checkpoint; at inference, raw GBM probability → calibrator → calibrated probability → Stage 4 threshold, in that order, every time.

## Stage 4 threshold search

```python
import numpy as np
threshold_grid = np.arange(0.05, 0.96, 0.01)   # 0.05 to 0.95, step 0.01 — 91 candidate cutoffs
```

For each candidate threshold: apply to the **calibrated** k-fold OOF probabilities, reconstruct `matching_results.tsv`-equivalent predictions per S1 entity, compute macro-averaged F₀.₅ exactly as the leaderboard does (including singleton entities scoring 1.0/0.0), average across folds, pick the argmax. Run this once on random k-fold and once on the held-out-country fold; if the two argmax thresholds disagree by more than a small margin (~0.05), that disagreement itself is worth a line in the methodology doc — it means the "right" cutoff is genuinely country-sensitive, which F₀.₅'s precision weighting makes costly to get wrong on France specifically.

## Blocking (Stage 1) — completing the previously behavior-only spec

`architecture.md` describes blocking's *behavior* (union of strategies, one similarity floor, recall-audited) without committing numbers. Concrete v1 values:

```python
TOP_K_DENSE = 50       # BGE-M3 dense cosine ANN, per S1 entity
TOP_K_SPARSE = 50      # BGE-M3 sparse/lexical, per S1 entity
SIMILARITY_FLOOR = 0.30  # applied after union, on whichever score produced the candidate
MAX_CANDIDATES_PER_ENTITY = 100  # hard cap after union+floor, to bound Stage 2 compute
```

- **Dense retrieval:** BGE-M3 dense embeddings, approximate nearest neighbor (e.g. FAISS `IndexFlatIP` on normalized vectors at this data scale — exact search, not approximate, since the S2+S3 table for a hackathon dataset is almost certainly small enough that exact cosine search is fast enough and removes ANN-recall as a confound in the blocking-recall audit; move to `IndexIVFFlat` only if profiling shows exact search is actually the bottleneck), top 50 per S1 entity.
- **Sparse retrieval:** BGE-M3's own sparse (lexical) output, top 50 per S1 entity via the same mechanism, functioning as the free BM25-equivalent.
- **Token inverted index:** shared-token blocking key on normalized business name tokens (post Stage 0 normalization) — any S2/S3 record sharing ≥1 significant token (post stopword removal) with an S1 record's name is a candidate.
- **Phonetic blocking:** Soundex or Double Metaphone code computed per name token; match on shared phonetic code.
- **Address-token blocking:** shared postal code (exact) or shared city/street-number token, computed the same country-agnostic way as Stage 0's other address fields.
- **Union, then floor:** take the union of all four strategies' candidates, then apply `SIMILARITY_FLOOR` only to candidates whose sole source was dense/sparse retrieval (token/phonetic/address-matched candidates pass through regardless of a low embedding score, since they were retrieved on a different, non-embedding signal — flooring them by embedding similarity would defeat the purpose of having multiple independent blocking strategies).
- **Cap:** if the union+floor set for an entity exceeds `MAX_CANDIDATES_PER_ENTITY`, keep the top 100 by best available score across strategies (max, not sum, across dense/sparse/token-overlap scores) rather than dropping the excess arbitrarily.

**These four numbers (`TOP_K_DENSE`, `TOP_K_SPARSE`, `SIMILARITY_FLOOR`, `MAX_CANDIDATES_PER_ENTITY`) are the actual first target of the blocking-recall audit already specified as v1's first build step.** If recall (measured against `train_ground_truth`, sliced by held-out country) comes in short, raise `TOP_K_*` and/or lower `SIMILARITY_FLOOR` first — both are free (more compute, not more code) — before adding a new blocking strategy. Only add a new strategy if raising these two doesn't close the gap, since a new strategy costs implementation time this design's own scope note (`v1-baseline.md`) says is scarce.
