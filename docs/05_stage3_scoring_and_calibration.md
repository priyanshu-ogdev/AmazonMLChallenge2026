# Layer 3 — Pair Scoring, Calibration, and Training Protocol

**Status:** v1 baseline implementation  
**Implementation:** `src/stage3_gbm.py`

## Purpose

Layer 3 converts Layer 2 evidence into a pair-level match score. It does not
generate candidates and it does not decide the final list by itself. It must
operate on the exact candidate pairs from `candidate_pairs.tsv`.

## Why the attached draft was changed

The attached draft contained useful hyperparameter and escalation ideas, but
four details were unsafe for production:

1. `StratifiedKFold` splits pairs rather than S1 entities, allowing the same
   entity's positives and negatives to leak between training and validation.
2. Its feature names (`bi_encoder_cosine`, `name_levenshtein_sim`, etc.) did
   not match the implemented Layer 2 schema.
3. Its country gate hard-coded US and India instead of treating country as an
   open set.
4. Focal loss and DART were presented too early. The first model must be a
   regularized, interpretable baseline so escalation has a measurable trigger.

The implementation uses `StratifiedGroupKFold`, with a fallback to
label-stratified grouped folds when the combined country/label strata are too
small. There is never a pair-level split.

## Input and labeling

`stage3_gbm.py` accepts the Stage 2c feature table and optionally the Qwen
feature table. It joins on:

```text
source1_entity_id, candidate_entity_id
```

Labels are created only from training ground truth. Empty ground-truth lists
produce valid negative candidates; singleton S1 entities must remain present
for downstream evaluation even though they contribute no positive pair.

Raw IDs, row order, raw country strings, and raw blocker-provenance strings are
never predictors. Country is retained for fold stratification and diagnostics.
The numeric `source_is_s3` feature is allowed; it must be validated for
stability and cannot be treated as a business identity signal.

## Baseline model

The production starting point is XGBoost with:

- logistic objective;
- AUCPR evaluation;
- histogram tree method;
- learning rate 0.05;
- max depth 4;
- minimum child weight 5;
- row and column subsampling 0.85;
- L1 regularization 0.1;
- L2 regularization 1.0;
- fold-local `scale_pos_weight`;
- early stopping at 50 rounds, up to 1,000 estimators.

This is intentionally conservative for a precision-heavy metric. The baseline
must be measured before any custom objective or DART change.

## OOF protocol

For each fold:

1. keep all pairs for an S1 entity in one fold;
2. fit the model on training entities only;
3. compute validation scores for unseen entities;
4. record AUCPR and best iteration;
5. concatenate scores into one OOF vector.

The OOF vector is the only data used to choose calibration and the final
threshold. The final model is retrained on all labeled pairs using a round
count derived from the fold best iterations.

## Calibration

Calibration is fit on OOF scores only, after the GBM OOF boundary:

- fewer than 1,000 positives: Platt/logistic calibration;
- more than 10,000 positives: isotonic calibration;
- intermediate counts: fit both and select lower pair-weighted OOF log loss.

This is a starting rule. The selected method and positive count are written to
`stage3_metadata.json`. Calibration is not fit on the final test set.

The implementation lives in `src/calibration.py` so it can be tested without
XGBoost. It validates finite scores and binary labels, clips probabilities to
the closed interval `[0, 1]`, serializes Platt coefficients or isotonic
breakpoints as JSON, and rejects unknown calibration methods. Metadata contains
two diagnostic views:

- `calibration_metrics`: fit-on-OOF reliability metrics used for inspection;
- `cross_fitted_calibration_metrics`: grouped calibration metrics where each
  calibrator is fit on other S1 entities and evaluated on held-out entities.

The cross-fitted metrics are the unbiased verification signal. The
fit-on-OOF metrics are retained only to describe the calibrator ultimately
used for test inference.

## F0.5 thresholding

The threshold is selected by entity-level macro F0.5 over calibrated OOF
scores. The complete training Source 1 ID universe is passed separately from
the candidate rows, so an S1 entity with zero candidates is still included as
an empty-empty singleton. Each S1 contributes equally. The implementation searches
the observed calibrated scores plus 0 and 1; it does not force a match.

The final threshold is stored in the Stage 3 metadata and must be reused for
test inference. Country- or source-specific thresholds are deferred until a
grouped validation comparison proves a stable gain.

Layer 4 applies this threshold and assembles the complete submission. Its
contract is documented in [`06_stage4_decision_and_singletons.md`](06_stage4_decision_and_singletons.md).

## Deferred escalation paths

### Beta-weighted or focal loss

Do not combine a custom class-weighted objective with `scale_pos_weight`.
Escalate only if the baseline has poor hard-negative separation after normal
regularization. If a custom objective is tested, compare it on identical
grouped folds and remove `scale_pos_weight` from that experiment.

### DART

Use only when the held-out-country diagnostic shows a persistent generalization
gap after ordinary regularization. DART disables useful prediction caching and
can make early stopping noisier; it is not the baseline.

### Monotonic constraints

The attached draft proposed monotonic constraints for features such as dense
cosine and edit similarity. These are not enabled by default because
normalized similarity can be misleading under missing fields, transliteration,
and contradictory addresses. Enable constraints only after a feature-by-feature
validation study on the actual candidate distribution.

## Outputs

The training command writes:

- `gbm.json` — final retrained scorer;
- `oof_predictions.tsv` — raw and calibrated OOF scores;
- `stage3_metadata.json` — features, calibration, threshold, AUCPR, and fold diagnostics.

The saved artifacts can be applied to an unseen candidate table with
`src.stage3_gbm.score_candidates(...)`. This loads the recorded feature
schema, applies the persisted Platt/isotonic parameters, and uses the saved
threshold; it does not refit anything or infer a threshold from test data.

Example:

```powershell
python -m src.stage3_gbm `
    --features ../../output/pair_features.tsv `
    --qwen-features ../../output/qwen_pair_features.tsv `
    --ground-truth ../../dataset/train/train_ground_truth.tsv `
    --output-dir ../../output/stage3
```

The Qwen file is optional. Feature ablation determines whether it remains in
the production feature set.

## Acceptance checklist

- [ ] Feature rows match the deduplicated candidate-pair set.
- [ ] Ground-truth labels were created only for training data.
- [ ] Every S1 entity remains within exactly one OOF fold.
- [ ] Country/source diagnostics are recorded without hard-coded test-country filtering.
- [ ] OOF positive count determines calibration.
- [ ] Threshold is selected on calibrated OOF predictions using macro F0.5.
- [ ] Baseline is compared before focal loss, beta loss, DART, or monotonic constraints.
- [ ] Final test scoring uses the saved feature schema and threshold.
