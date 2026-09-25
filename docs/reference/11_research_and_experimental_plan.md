# Research & Verification Plan — Staged, One Layer at a Time

Each stage below has an explicit input, a measurement, and a go/no-go gate. No stage's output is
assumed — it's measured before the next stage builds on it. This directly implements the
"incremental build-and-measure" principle referenced throughout this docs set rather than building
the full pipeline speculatively and hoping it works end to end.

## Stage 0 — Data reconnaissance (before any modeling)
**Do:** Load all three sources and `train_ground_truth.tsv` with `sep="\t"` (FR-8). Profile: record
counts per source, field length distributions, fraction of S1 entities with 0/1/many matches
(singleton rate), locale distribution if inferable from address text, obvious noise patterns
(abbreviation styles, missing fields).
**Why first:** every design decision from here on (block key choice, negative-sampling ratio,
threshold prior) depends on facts about this specific dataset that no literature review can supply
— this is the boundary named explicitly in `05_hyperparameter_verification_status.md`.
**Gate:** proceed once singleton rate and rough noise character are known; these numbers alone
often reprioritize which blocking keys matter most.

## Stage 1 — Blocking baseline
**Do:** Implement the cheapest two keys first — normalized-token key and phonetic key on business
name (no embeddings yet). Generate `candidate_pairs.tsv`.
**Measure:** blocking recall = fraction of true matches (from ground truth) present in the
candidate set; candidate-set size distribution per S1 entity.
**Reasoning check:** this baseline tests whether the "union of cheap keys" argument
(`06_optimization_design.md` §1) is even necessary — if simple keys already achieve near-ceiling
recall, the added complexity of embedding-based ANN blocking may not be worth its compute cost.
**Gate:** if recall is already ≥~0.97 with acceptable candidate-set size, defer embedding-based
blocking to a stretch goal. If recall is materially lower, proceed to Stage 1b.

## Stage 1b — Add embedding-based ANN blocking (conditional)
**Do:** Add BGE-M3 dense (+ optionally sparse) similarity as an additional blocking key, unioned
with Stage 1's keys.
**Measure:** recall delta over Stage 1 baseline; candidate-set size delta (embedding blocking
tends to add more candidates, which raises Stage 2's compute cost).
**Reasoning check:** confirms or disconfirms the specific claim in `06_optimization_design.md` §1
that dense embeddings catch semantically-close-but-token-dissimilar pairs — measured here, not
assumed.
**Gate:** keep only if the recall gain materially reduces the ceiling loss from Stage 1, weighed
against the added Stage 2 compute from a larger candidate set.

## Stage 2 — Matching baseline
**Do:** On Stage 1(b)'s candidate pairs, build the simplest classifier: string-similarity features
(Levenshtein/Jaro-Winkler/token-set on name and address, separately) + one embedding cosine
similarity feature, fed into a gradient-boosted classifier.
**Measure:** F_0.5 on a held-out fold, sliced separately for singleton entities and multi-match
entities. Precision/recall at the default 0.5 threshold as a starting reference point.
**Gate:** this becomes the baseline every subsequent addition is measured against — nothing below
is added unless it beats this number on the same held-out fold.

## Stage 3 — Threshold calibration
**Do:** Sweep the decision threshold, plot precision/recall/F_0.5 across the range.
**Reasoning check:** confirms or disconfirms the mechanical prediction in `02_matching_stage.md`
that the F_0.5-optimal threshold sits above the F1-optimal one — this is a definitional consequence
of the metric, so the *direction* is not really in question, but the *magnitude* is unmeasured
until this stage.
**Gate:** lock in the threshold that maximizes F_0.5 on the held-out fold; re-verify on a second
fold if data volume allows, to catch overfitting to one split.

## Stage 4 — Second embedding feature (conditional)
**Do:** Add Qwen3-Embedding-0.6B cosine similarity as a second, architecturally-distinct feature
(generated out-of-fold), alongside BGE-M3's.
**Measure:** F_0.5 delta over Stage 3's locked baseline.
**Reasoning check:** tests the stacking-diversity argument directly — if this feature doesn't move
F_0.5, the "architecturally different encoder adds diverse signal" reasoning doesn't hold on this
data, regardless of how plausible it sounds in the abstract, and should be dropped for pipeline
simplicity.
**Gate:** keep only on a measured, non-trivial F_0.5 improvement.

## Stage 5 — Domain fine-tuning (conditional, highest-effort stage)
**Do:** Build the triplet-sampling pipeline from `train_ground_truth.tsv` with same-name-different-
address hard negatives (reasoning in `06_optimization_design.md` §3). Fine-tune BGE-base-en-v1.5 or
BGE-M3, own training code only (not vendored).
**Measure:** F_0.5 delta from swapping the fine-tuned encoder's similarity in for the pretrained
one used in Stage 2/4, same held-out fold.
**Reasoning check:** this is the stage the Sodhana paper's precedent (`03_embedding_models.md`,
`04_citations.md`) most directly informs — but their reported gains were on synthetic data with a
different distribution than this competition's; treat their numbers as motivation to attempt this
stage, not as a predicted outcome here.
**Gate:** keep only if it beats Stage 3/4's locked baseline; this is the most compute-expensive
optional stage and should be attempted last, after cheaper stages are exhausted.

## Stage 6 — Reranking over shortlist (conditional)
**Do:** For S1 entities with multiple above-threshold candidates after Stage 3-5, apply a larger
model (Qwen3-Embedding-4B/8B similarity, or a field-tagged cross-encoder) only to that shortlist.
**Measure:** F_0.5 delta, specifically on the subset of entities with multiple close-scoring
candidates (where reranking can plausibly help — measuring on the full set would dilute a real
localized effect).
**Gate:** keep only if it improves F_0.5 on that subset without degrading singleton handling
elsewhere (reranking logic must not override a correct no-match decision).

## Stage 7 — Locale slicing and final validation
**Do:** Re-run Stage 3's evaluation sliced by locale. If France is confirmed as an actual held-out
locale in the real data (see `04_citations.md` — this was not confirmed against the primary brief),
isolate it specifically; otherwise slice by whatever locales the data actually contains.
**Reasoning check:** this is the direct test of the generalization risk the primary brief flags
generically ("account for locale-specific patterns") — deferred to last because it requires the
full pipeline to already be locked, not because it's low priority.
**Gate:** if any locale underperforms materially against the aggregate, revisit Stage 1
normalization rules (FR-10) for locale-specific address patterns before final submission, rather
than patching the matching stage to compensate.

## Stage 8 — Packaging and validation
**Do:** Run `utils/validate_submission.py` against final `matching_results.tsv` and
`candidate_pairs.tsv`; assemble `code/`, `Documentation_template.md` (drawing from this docs set,
particularly `06_optimization_design.md` for the reasoning narrative and `07_prd.md` for scope);
final license check on every dependency actually shipped in `code/` against NFR-1.
**Gate:** submission-ready only when validation script passes and license check is clean.

## Summary table

| Stage | Adds | Kept only if |
|---|---|---|
| 1 | Cheap token/phonetic blocking | Baseline — always kept |
| 1b | Embedding ANN blocking | Recall gain material vs. compute cost |
| 2 | Baseline classifier | Baseline — always kept |
| 3 | Threshold calibration | Always kept — required regardless of outcome |
| 4 | Second embedding feature | Measured F_0.5 improvement |
| 5 | Fine-tuned encoder | Measured F_0.5 improvement, beats Stage 4 |
| 6 | Shortlist reranking | Measured improvement on close-call subset |
| 7 | Locale-specific fixes | Only if France underperforms |
| 8 | Packaging | Always — final gate before submission |
