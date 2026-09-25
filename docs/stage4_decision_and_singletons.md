# Layer 4 — Decision Policy and Submission Assembly

**Status:** v1 deterministic implementation  
**Implementation:** `src/decision.py`

## Purpose

Layer 3 scores candidate pairs. Layer 4 converts those scores into the exact
submission contract. It is deliberately deterministic: it does not retrain,
recalibrate, infer a new threshold, or invent candidates.

## Decision sequence

1. Load the saved Stage 3 metadata.
2. Load the complete Source 1 test entity list.
3. Keep every scored pair whose saved calibrated score is greater than or
   equal to the saved global threshold.
4. Group matches by Source 1 entity while preserving the input candidate order.
5. Emit one row for every Source 1 entity, including entities with no
   candidates and entities with no accepted matches.

The comparison is `score >= threshold`; this boundary is part of the
reproducibility contract. Layer 4 must never apply a second country threshold,
force a top-1 match, or drop a low-confidence singleton.

## Why the implementation is separate

The competition metric is macro F0.5 per Source 1 entity, not pair-weighted
accuracy. Threshold selection therefore belongs to grouped OOF validation in
Layer 3, while applying that selected threshold belongs here. Keeping the
operations separate prevents test scores from influencing the threshold and
ensures that zero-candidate S1 entities receive an explicit empty prediction.

## Validation invariants

The assembler rejects:

- missing required score columns;
- thresholds outside `[0, 1]`;
- duplicate Source 1 IDs in the input entity list;
- duplicate candidate pairs;
- missing calibrated scores;
- scored rows for unknown Source 1 IDs.

It does not validate whether candidate IDs exist in Source 2/3. That remains
the responsibility of the repository submission validator, which can perform
the optional memory-heavy existence check.

## Output contract

`matching_results.tsv` contains exactly:

```text
source1_entity_id    matched_entity_ids
```

Every Source 1 test entity appears exactly once. An empty
`matched_entity_ids` value is the correct representation for a singleton or
an entity whose candidates all fall below threshold. IDs are comma-separated,
with no duplicates, and all emitted matches come from the scored candidate
table.

## Scope boundaries

Layer 4 does not solve candidate recall. If a true match was absent from
`candidate_pairs.tsv`, no decision policy can recover it. It also does not
change the global threshold by country or source; such policies require a
separate grouped validation experiment and a documented acceptance gate.

## Acceptance checklist

- [ ] Threshold comes from Stage 3 metadata, never test labels.
- [ ] All Source 1 test IDs are emitted, including zero-candidate entities.
- [ ] Every emitted match was present in the scored candidate table.
- [ ] No top-1 forcing or implicit fallback match exists.
- [ ] Duplicate pairs and unknown Source 1 IDs fail loudly.
- [ ] `utils/validate_submission.py` passes on the resulting file.
