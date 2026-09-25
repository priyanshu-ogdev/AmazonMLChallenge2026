# Entity Resolution Pipeline — Documentation

## What this is

A three-source entity resolution pipeline: Source 1 is a deduplicated reference set; Source 2 and Source 3 are independent, noisy sources that may contain zero, one, or many records matching each Source 1 business or person entity. The task is retrieval-and-decision per Source 1 entity, scored by **F₀.₅** (precision weighted 2× over recall), macro-averaged per entity — so singleton entities (correct "no match") matter exactly as much as high-volume ones, and a false merge costs far more than a missed match.

Training data covers **US and India only**. The test set adds **France** — an unseen country by design, and the single hardest generalization requirement running through every layer of this pipeline.

## Scope and timeline — read this before building from the rest of these docs

This design was written at full research depth — held-out-country ablation grids, dual LoRA sweeps across two independently fine-tuned models, DART/focal-loss escalation paths, cross-validated isotonic calibration. **The actual competition window is a 72-hour hackathon** (per the official Unstop timeline: Sep 25 00:00 IST → Sep 27 23:59 IST, team of 2–4). Almost nothing in `training.md` or `open-decisions.md` is achievable in full within that window. **The decided v1 baseline plan — what actually gets built, in what order, and why — is `v1-baseline.md`.** Read it before `training.md`; `training.md`/`regularization.md` remain the target design and the source for stretch goals once the baseline is solid, not the build order.

## Hard constraints, restated

- No external lookups, APIs, or internet augmentation of any kind — everything comes from the provided train/test data.
- Final matching model: MIT or Apache-2.0 licensed, ≤8B parameters.
- Submission is audited: reproducible code, both required output TSVs, and a filled methodology document.

## How to read these docs

| File | Covers |
|---|---|
| `v1-baseline.md` | **Read this first.** The actual 72-hour build plan: what ships in v1, what's cut, and the math behind why |
| `parameters.md` | **Read this second, right before writing code.** The actual starting values — GBM hyperparameters, calibration method, threshold-search grid, and blocking's top-K/floor/cap numbers — that the rest of this doc set specifies only as protocols or ranges |
| `architecture.md` | The five-stage pipeline, what each stage does, and the dependency structure (target design, not build order) |
| `training.md` | How the bi-encoder, the cross-encoder, and the GBM are each trained — data, loss, regularization |
| `inference.md` | The inference-time sequence, hardware footprint, and train/inference-mode switches |
| `citations.md` | Every source this design relies on, with what was actually verified and how |
| `regularization.md` | Dropout and regularization for every trained model, with the actual SOTA source for each mechanism — including the GBM meta-learner's theoretical grounding and a real LoRA-rank scaling confound this design had to correct |
| `open-decisions.md` | The handful of values genuinely left open, and the exact protocol that resolves each one once real data exists |
| `verification-log.md` | A full account of five verification rounds — four false claims caught, then a fifth round that independently confirmed France and the license/size constraint against the official problem statement — and why the pattern matters |

## One methodological note worth keeping

This project's citations went through four separate rounds of pasted "verification" documents before settling — including claims that a real, correctly-cited paper didn't exist, that a real detail within it didn't say what it was cited as saying, that this conversation's own founding problem-statement document never actually mentioned France or the model license/size constraint, and that a real, MIT-licensed open-source package was GPL-licensed. All four were checked directly against primary sources and all four were false. A fifth round then independently confirmed — against the official Amazon ML Challenge 2026 problem statement itself, not just this conversation's own founding document — that France and the MIT/Apache-2.0/≤8B constraint are real, resolving the question on firmer footing than before. `verification-log.md` documents the full pattern. The discipline that held up, and the one worth carrying into any future revision of this project: **verify a claim against the primary source before it changes a design decision — not against how confident, detailed, or self-critical the claim sounds, and not against how many of the document's other claims happen to check out.**
