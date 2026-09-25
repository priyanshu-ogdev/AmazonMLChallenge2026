# PRD — Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

## 1. Problem statement
Link business records across three independently-sourced datasets (S1 = clean reference, S2/S3 =
noisy vendor feeds) with no shared identifiers, using only business name and address fields. For
every S1 entity, output the set of matching S2/S3 records — zero, one, or many.

## 2. Success metric and what it implies for requirements
**Scoring: macro F_0.5**, precision weighted ~2x over recall. This single fact drives several
requirements below, not just the matching-stage threshold:
- A wrong merge costs roughly twice what a missed match costs → every layer should default to
  *not* merging when uncertain, not just the final threshold.
- Singleton entities (true zero-match S1 records) score 1.0 if left unmatched and 0.0 if given any
  wrong match → the pipeline must support an explicit "no match" outcome per entity, not a forced
  top-1 choice (reasoning: [`13_optimization_design.md`](13_optimization_design.md) §2).

## 3. Functional requirements

| ID | Requirement | Rationale | Source layer |
|---|---|---|---|
| FR-1 | Normalize name/address fields identically across S1, S2, S3 before any comparison | Vendor formatting differs; unnormalized comparison undercounts true matches | [`early_drafts/01_blocking_stage.md`](early_drafts/01_blocking_stage.md) §1 |
| FR-2 | Generate candidate pairs via multiple independent block keys, unioned | Any single key has a distinct blind spot; union drives recall up, reasoned in [`13_optimization_design.md`](13_optimization_design.md) §1 | [`early_drafts/01_blocking_stage.md`](early_drafts/01_blocking_stage.md) §2 |
| FR-3 | Output `candidate_pairs.tsv` as an auditable artifact of blocking, separate from final matches | Explicitly required by the challenge's final submission package | Challenge spec §4 |
| FR-4 | Score every candidate pair independently with a multi-feature classifier, not a single similarity threshold | Single-feature threshold can't structurally separate "high name / low address similarity" from true matches — see FR capacity argument in [`13_optimization_design.md`](13_optimization_design.md) §2 | [`early_drafts/02_matching_stage.md`](early_drafts/02_matching_stage.md) |
| FR-5 | Support an explicit no-match / abstain outcome per S1 entity | Singleton scoring is binary; forced top-1 choice guarantees wrong answers on every true singleton | [`early_drafts/02_matching_stage.md`](early_drafts/02_matching_stage.md), PRD §2 |
| FR-6 | Output `matching_results.tsv`, one row per S1 entity, comma-separated match list, empty when no match | Required leaderboard/final submission format | Challenge spec §4 |
| FR-7 | Provide a `utils/validate_submission.py`-compatible output format before every submission attempt | Avoids wasting submission attempts on formatting errors (per challenge spec) | Challenge spec §4 |
| FR-8 | All data loaded with `sep="\t"` explicitly | Files are `.tsv`; default pandas parsing silently mis-splits on embedded commas in names/addresses | Challenge spec §3 |
| FR-9 | No external data lookups, geocoding services, or third-party APIs anywhere in the pipeline | Explicit competition rule | Challenge spec §5 |
| FR-10 | Support locale-specific normalization rules; treat France as a candidate test locale worth validating against, but confirm this against the actual competition rules/data first | The primary brief only says "account for locale-specific patterns," generically — France-specificity and the MIT/Apache-2.0/≤8B model constraints originate from a separate pasted transcript in this conversation, not the primary brief itself. See [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md) note. | [`early_drafts/01_blocking_stage.md`](early_drafts/01_blocking_stage.md) §3 |

## 4. Non-functional requirements

| ID | Requirement | Rationale |
|---|---|---|
| NFR-1 | Every model/library used must carry a license compatible with the competition's redistribution requirements (final package includes runnable `code/`) | The one confirmed license conflict found in this design process — LinkTransformer is GPL-3.0 — makes this a real constraint, not boilerplate. See [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md). |
| NFR-2 | Blocking must complete over the full S1×(S2∪S3) table without full pairwise comparison | Stated computational infeasibility in the challenge spec |
| NFR-3 | The full pipeline (blocking → matching → output) must be reproducible from `code/` alone, per the final package requirements | Challenge spec §4 |
| NFR-4 | Any embedding model used for full-table blocking should be sized for single-pass throughput over 100K+ candidate pairs; reserve larger models (4B/8B class) for shortlist-only reranking | Compute-budget reasoning in [`13_optimization_design.md`](13_optimization_design.md) §4 |

## 5. Explicitly out of scope
- Any component requiring network access to external databases, geocoding, or third-party APIs (FR-9)
- Any code literally vendored from GPL-licensed projects (NFR-1); techniques may be reimplemented independently
- Speculative hyperparameter values presented as final — see [`12_hyperparameter_verification_status.md`](12_hyperparameter_verification_status.md) for why these are deliberately left open pending real data

## 6. Deliverables (mapped to challenge requirements)
1. `output/matching_results.tsv` — scored artifact
2. `output/candidate_pairs.tsv` — blocking audit artifact (FR-3)
3. `code/` — complete runnable pipeline (NFR-3)
4. `Documentation_template.md` — methodology write-up (this docs/ set is the source material for it)
5. This `docs/` folder — internal design record, verification log, and reasoning trail

## 7. Risks and mitigations

| Risk | Likelihood driver | Mitigation |
|---|---|---|
| Blocking recall ceiling too low on locale-specific addresses (e.g. France, if confirmed as a real held-out locale) | Address format conventions differ by country (primary brief flags this generically; France-specificity is unconfirmed — see [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md)) | Locale-specific normalization rules (FR-10), measured recall audit before matching-stage work begins ([`12_hyperparameter_verification_status.md`](12_hyperparameter_verification_status.md) §"blocking recall audit") |
| Overfitting matching threshold to F1 intuition instead of F_0.5 | Default classifier tooling/metrics report F1 by default, not F_0.5 | Explicit threshold sweep reporting F_0.5, singleton-sliced, before finalizing ([`early_drafts/02_matching_stage.md`](early_drafts/02_matching_stage.md)) |
| Shipping code with an incompatible license | LinkTransformer confirmed GPL-3.0 ([`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md)) | Reimplement the technique independently, don't vendor (NFR-1) |
| Over-engineering (fine-tuning, reranking) without measured benefit | Literature precedent (Sodhana paper, etc.) is suggestive, not conclusive for this specific dataset | Incremental build-and-measure: each optional component must show a measured F_0.5 delta on a held-out fold before being kept ([`12_hyperparameter_verification_status.md`](12_hyperparameter_verification_status.md) §3) |

## 8. Open questions requiring real data (not resolvable at PRD stage)
See [`12_hyperparameter_verification_status.md`](12_hyperparameter_verification_status.md) for the full list — blocking thresholds, classifier
hyperparameters, decision threshold, whether fine-tuning/reranking earn their complexity, and the
France-specific generalization gap. This PRD specifies *what* must be measured and *how*, not the
resulting values.
