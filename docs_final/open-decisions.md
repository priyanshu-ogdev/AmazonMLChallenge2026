# Open decisions

Everything below is left open deliberately — each is a *value* that only real data can settle, not a gap in the design. The protocol resolving each one is fixed; the value isn't, yet.

## The resolution protocol, shared across all of these

Train on US+India, hold out one country at a time as a proxy for the France generalization gap (train-on-US/validate-on-India and the reverse, both directions). Run a cheap single-split pass across the relevant configuration grid first; commit the winner to the full expensive cycle (5-fold OOF + 1 final retrain) only after the cheap pass has picked it. Don't run every combination through the expensive cycle — that multiplies two ablation grids together unnecessarily.

## Open items

| Decision | Options | Resolved by |
|---|---|---|
| Stage 2a encoder | BGE-M3's dense head vs. Qwen3-Embedding-0.6B | Held-out-country grid, cheap pass |
| Bi-encoder tuning method | LoRA (rank 32/64/128) vs. full fine-tuning | Same grid, crossed with the above |
| Ditto tuning method | LoRA (own rank sweep, not reused from the bi-encoder) vs. full fine-tuning | Same grid, run separately — different task shape |
| GBM boosting mode | Standard vs. DART | Standard validation protocol; DART only if train/validation gap persists after regularization |
| GBM loss | `binary:logistic` + `scale_pos_weight` vs. focal loss vs. beta-weighted logloss | Baseline first; escalate only if hard-negative separation is measurably insufficient |
| Decision threshold | Any value in [0, 1] | k-fold F₀.₅ sweep, inherently cannot be fixed before real validation data exists |
| Country-match feature masking rate | Untested specific percentage | Held-out-country grid, alongside the encoder/tuning decisions |
| MNRL batch composition (hard negatives per positive) | Untested ratio | Same grid — too few weakens the hard-negative signal, too many can destabilize early training |
| Blocking-stage recall, per country | Unmeasured until real data exists | Dedicated recall audit against `train_ground_truth`, run first, before Stage 2/3 tuning begins (see `architecture.md`) |
| GBM: DART and loss-escalation (focal/beta-logloss) used together | Not explicitly scoped as a combination | Test independently first; only combine if both are independently justified by validation, since the interaction hasn't been reasoned through |

## Why these stay open rather than getting a default guess

Every item above has a plausible-sounding default (Qwen3-Embedding scores higher on general benchmarks; LoRA has better OOD evidence in the literature; DART sounds like "more regularization is safer"). None of those defaults are being adopted here, on purpose — the literature backing each one is either from a different task, a different domain-shift magnitude, or simply doesn't exist for this specific combination of encoder architecture and entity-resolution data. The validation grid exists precisely because the confident-sounding answer and the correct answer have not reliably matched for citations in this same project (see `verification-log.md`) — there's no reason to expect informal reasoning about hyperparameters to be more reliable than informal reasoning about paper contents was.
