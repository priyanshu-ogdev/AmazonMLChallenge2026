# Hyperparameters, Tuning, and Design Verification Status

This document is the honest boundary line of this exercise: it separates what "verification" can
mean before the real dataset exists from what it can only mean after.

## What has been verified (facts about the world, checkable now)
- Model identities: architecture, parameter count, license, context length, published benchmark
  scores — see [`early_drafts/03_embedding_models.md`](early_drafts/03_embedding_models.md) and [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md). These are stable facts independent
  of this competition's data and don't change with tuning.
- Paper/tool existence and their own reported numbers (Sodhana paper's Table 4, LinkTransformer's
  actual license, etc.) — checkable against primary sources regardless of dataset access.
- Internal consistency of the pipeline design: that blocking precedes matching, that recall lost in
  blocking is unrecoverable, that F_0.5 weights precision over recall, that singleton scoring is
  binary (1.0 or 0.0) — these are logical/definitional facts from the problem statement itself, not
  empirical claims.

## What genuinely cannot be verified without the real dataset
No amount of literature review, model-card checking, or design review can produce a verified value
for any of the following. Stating a specific number for these right now would be fabrication, not
verification — they are not general knowledge facts, they are properties of *this specific,
currently-unseen* dataset:

| Parameter / decision | Why it's data-dependent |
|---|---|
| Blocking key thresholds (ANN similarity cutoff, n-gram length, phonetic algorithm choice) | Depends on this dataset's actual noise patterns and name/address distributions |
| Candidate-set size per S1 entity | Depends on how ambiguous this dataset's name/address space actually is |
| Matching-stage classifier hyperparameters (tree depth, learning rate, regularization, if using GBM) | Depends on training set size, feature distributions, class balance — none known yet |
| Decision threshold for match/no-match | Depends on the actual precision-recall curve this data produces, which can only be measured, not predicted |
| Whether fine-tuning or reranking earns its added complexity | An empirical F_0.5 delta on a validation split — cannot be estimated from other papers' numbers on different data |
| Feature importance / which similarity signals actually help vs. add leakage or noise | Only measurable by ablation on real folds |
| Regional (France) generalization gap | Only measurable once France-labeled examples in training or validation are examined |

## What "verification" should mean for these once the dataset is available
1. **Blocking recall audit**: compute recall-at-blocking directly against `train_ground_truth.tsv`
   — the fraction of true matches that survive into `candidate_pairs.tsv`. This is a hard number,
   not an estimate.
2. **Threshold sweep**: compute F_0.5 (and a singleton-only slice of it) across a grid of decision
   thresholds on a held-out fold, and report where the optimum actually falls rather than assuming
   it sits above the F1-optimum.
3. **Ablation, not addition-by-default**: each optional component (second embedding model, fine-
   tuning, reranking) gets measured as a delta over the previous best configuration on the same
   validation fold before being kept. A component earns its place in the final pipeline by measured
   improvement, not by literature precedent alone.
4. **Locale slicing**: report metrics separately for the France-held-out portion of validation data
   specifically, not just in aggregate, given that's the flagged generalization risk.

*Note: "France" throughout this table is used as shorthand from earlier in this design
conversation, not a confirmed detail from the primary competition brief — see [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md)'s
source-of-truth correction. Substitute whichever locale(s) the actual test data confirms.*

## Bottom line
The design in [`early_drafts/01_blocking_stage.md`](early_drafts/01_blocking_stage.md) and [`early_drafts/02_matching_stage.md`](early_drafts/02_matching_stage.md) is complete as a *specification* —
every stage, every safeguard, every decision rule is defined. It is not, and cannot yet be, a
*tuned* pipeline. Any document that presented specific hyperparameter values or a "verified" F_0.5
score at this stage would be fabricating precision the underlying situation doesn't support. The
correct next step is to run the pipeline in `code/` against the real train/test TSVs and produce
the four verification measurements above — that is where genuine hyperparameter and performance
verification happens, not in a further round of literature review.
