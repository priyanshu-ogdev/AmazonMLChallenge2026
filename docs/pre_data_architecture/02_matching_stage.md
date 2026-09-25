# Stage 2 — Matching (Classification / Filtering)

## Objective
Given the candidate pairs from blocking, decide true-match vs. look-alike, optimizing for the
competition's macro F_0.5 metric — i.e., precision weighted ~2x recall. A wrong merge is worse
than a missed match.

## Design

### Feature set for a pair-level classifier (e.g. gradient-boosted trees)
- Multiple string-similarity scores computed separately on name and address (Levenshtein,
  Jaro-Winkler, token-set/token-sort ratio) — kept as *separate* features per field rather than
  merged into one score, so the model can learn field-specific weighting
- Token overlap / Jaccard similarity on normalized tokens
- Dense embedding cosine similarity as an independent signal, generated **out-of-fold** to avoid
  leakage if any part of the embedding pipeline is fit on labeled data
- Optionally a second, architecturally different embedding model's cosine similarity as a
  diversity feature (e.g. one encoder-based, one decoder-based) — the value of this is a stacking-
  diversity argument, and should be validated by measuring whether it actually adds signal beyond
  the first embedding feature before being kept in the final model
- Structured field-level agreement flags (e.g. same street number, same postal code fragment)

### Decision rule, not forced top-1 match
Because a given S1 entity can have zero, one, or many true matches, the matcher must not assume a
forced choice among candidates. Each candidate pair gets its own match/no-match decision (with a
tunable probability threshold), and an S1 entity with no candidate clearing the threshold is
correctly output as a singleton (empty match list).

### Threshold tuning against F_0.5
- Sweep the decision threshold on a held-out slice of `train_ground_truth.tsv`
- Because F_0.5 weights precision 2x, the optimal threshold is expected to sit above the F1-optimal
  point — this should be confirmed empirically, not assumed
- Singletons are the highest-leverage error case: predicting an incorrect match on a true singleton
  scores 0.0 for that entity, while correctly predicting no match scores 1.0. The threshold sweep
  should specifically report singleton precision/recall as its own slice, not just aggregate
  numbers.

### Optional escalation: reranking
For S1 entities with several above-threshold candidates, an optional second pass with a larger
model (e.g. a bigger embedding model's cosine similarity, or a cross-encoder in a Ditto-style
field-tagged format — `[COL] name [VAL] ... [COL] address [VAL] ...`) over just that shortlist,
not the full candidate table, to sharpen ranking among close calls.

### Optional escalation: domain fine-tuning
Continue training a small open-license sentence encoder on triplets built from
`train_ground_truth.tsv`, using same-name/different-address pairs as hard negatives (the pattern
most likely to produce a false merge). See `03_embedding_models.md` for candidate base checkpoints
and the published precedent for this approach.

## Open / data-dependent decisions
- Final feature set (which similarity signals actually help vs. add noise/leakage risk)
- Decision threshold value
- Whether reranking/fine-tuning earn their added complexity — to be settled by measuring F_0.5
  delta on a validation split, incrementally, against the baseline (single embedding + string
  features) before committing either into the final pipeline
