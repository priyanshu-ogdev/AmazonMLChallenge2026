# Optimization Design — Reasoned, Not Citation-Gated

This document reasons through pipeline and code-level design choices from first principles and
from techniques known to work in this space. It treats prior research as a source of *ideas and
optimizations to adapt*, not as material that must be formally cited to use. The one exception:
where a technique is only available as literal copyleft-licensed code (not just a described
method), that's flagged separately, because that's a redistribution obligation on your own code,
not a documentation formality.

## 1. Blocking stage — why this shape, reasoned

**Problem**: O(|S1| × |S2∪S3|) pairwise comparison is infeasible. Need sub-quadratic candidate
generation with recall as close to 1.0 as achievable, because blocking's recall is a hard ceiling.

**Reasoning for multi-key union blocking** (rather than one key): any single block key has a
specific failure mode — a token-based key fails on transposition/typos, a phonetic key fails on
non-phonetic abbreviation differences, a pure embedding-ANN key fails on rare tokens the embedding
under-weights. These failure modes are close to independent, so taking the *union* of several
cheap, independent keys drives blocking recall up multiplicatively in the misses (if each key
independently misses ~15–25% of true matches, their union misses roughly the product of those
miss-rates, assuming reasonable independence) while only linearly increasing compute (each key is
still a cheap grouping pass). This is a standard variance-reduction argument, not specific to any
one paper — it's the same logic as ensembling independent weak signals in any retrieval system.

**Reasoning for embedding-based ANN blocking specifically**: dense embeddings capture semantic
closeness that token/phonetic keys can't (e.g. "Wal-Mart" vs. "Walmart Inc" vs. "Walmart
Supercenter #4021" landing near each other in vector space despite low token overlap). Using a
model that natively emits both dense and sparse vectors from one forward pass (BGE-M3's design)
is an engineering optimization, not a research dependency: it removes the need to run and maintain
a separate BM25/inverted-index pipeline alongside the embedding pipeline, cutting one whole
subsystem out of the stack for roughly the same result. That's a legitimate architectural choice on
its own merits — the model was chosen because of what it computes, not because of who published it.

**ANN index choice (FAISS/HNSW)**: reasoned from throughput requirements — for a static build
(index once, query once per submission), an HNSW graph index gives near-linear query time and
tunable recall/speed via `ef_search`, without needing IVF's cluster-training step, which matters
less here than for a live-updating index. This is a standard choice for static candidate generation
of this scale, adopted for the engineering reason (matches the access pattern), not because any one
paper prescribes it.

## 2. Matching stage — why this shape, reasoned

**Feature-based gradient-boosted classifier over a raw threshold on one similarity score**: reasoned
from the shape of the error costs. A single cosine-similarity threshold is a 1-dimensional decision
boundary; it can't distinguish "high name similarity, low address similarity" (a same-name-
different-business look-alike — should be rejected) from "high similarity on both" (a true match)
if both produce a similar blended score. A model that sees name-similarity and address-similarity
as *separate* features can learn that these two error modes need different treatment, which a
single scalar threshold structurally cannot. This is why the design carries multiple similarity
features into a classifier rather than collapsing to one number early — it's a capacity argument,
not a specific paper's recommendation.

**Out-of-fold feature generation**: standard leakage-prevention reasoning — any feature computed
using a model or statistic fit on the training labels (even indirectly) must be generated on data
that model never saw during its own fitting, or the classifier will learn to exploit that leakage
and the CV score will overstate real performance. This applies whether or not the embedding model
is fine-tuned; it's basic to any stacked/ensembled ML pipeline.

**Threshold calibration above the F1-optimum for F_0.5**: reasoned directly from the metric
definition — F_beta with beta<1 weights precision more heavily, so the loss surface penalizes false
positives more than false negatives at the margin, which by definition pushes the optimal operating
point toward a higher-precision (i.e. typically higher-threshold) region than F1 would. This is a
mechanical consequence of the F_0.5 formula, not something that needs external validation to
reason about — though the *exact* value still needs measurement (see [`12_hyperparameter_verification_status.md`](12_hyperparameter_verification_status.md)).

**Singleton handling as its own decision, not a forced top-1 choice**: reasoned from the scoring
rule itself — since an S1 entity can have zero true matches, any pipeline that is structurally
forced to output "the best-scoring candidate" for every entity will, by construction, get every
true singleton wrong. The fix (an explicit abstain/no-match option via thresholding, rather than
argmax-over-candidates) is a direct logical consequence of the problem's own label structure, not
an optimization borrowed from elsewhere.

## 3. Fine-tuning design — technique inherited and adapted, not copied

**Triplet loss with same-name-different-address hard negatives**: this general technique (metric
learning with hard negatives targeting the specific confusion class you most want to avoid) is a
standard, well-understood pattern in representation learning — it doesn't require citing a specific
paper to justify using it; the reasoning is direct: a model trained only on random negatives learns
to separate obviously-different pairs, which it already does well, and gets little training signal
on the pairs that actually cause your errors. Explicitly constructing negatives from your own
error-prone case (same name, different address) forces gradient signal onto exactly the boundary
your classifier needs sharpened. This reasoning holds regardless of which paper first demonstrated
it on business records — it's applicable because the mechanism is sound for this specific error
mode, verified structurally rather than by appeal to authority.

**What to actually build vs. reuse**:
- **Reuse (weights, not code)**: starting fine-tuning from an existing open-license checkpoint
  (BGE-M3, BGE-base-en-v1.5, or a name-matching-adapted checkpoint like eridu) is standard transfer
  learning — you're inheriting learned weights, which is exactly what pretrained weights are for,
  under whatever license accompanies that checkpoint (MIT/Apache-2.0 for all of these, confirmed
  previously).
  Own build: the triplet-sampling logic, the hard-negative construction from your own
  `train_ground_truth.tsv`, and the training loop itself should be your own code — this is a
  small, well-specified piece (triplet loss over a sentence-transformer model is a standard
  sentence-transformers library call) and writing it yourself avoids any question of what license
  someone else's training script carries.
- **Reference only, don't vendor the code**: LinkTransformer's blocking/linking API design is a
  reasonable pattern to imitate (a small number of well-named functions: `merge`, `dedupe`,
  `cluster`, wrapping a sentence-transformer + similarity join) — copy the *shape* of that API into
  your own implementation rather than importing or vendoring the GPL-3.0 package, since GPL-3.0
  code brought into your submission would extend that obligation to your own pipeline code. This is
  the one place in this design where "inherited code" vs. "inherited idea" is a real practical
  distinction, independent of any citation preference.

## 4. Reranking stage — reasoned, optional

Two-stage retrieve-then-rerank (cheap embedding similarity to shortlist, then a heavier model only
on that shortlist) is reasoned purely from a compute-budget argument: a cross-encoder or larger
embedding model scores a pair jointly and is more accurate per-pair, but its cost scales with the
number of pairs scored — running it over the full candidate table is wasteful when blocking has
already produced a small shortlist per entity. Restricting the expensive model to the shortlist
captures most of its accuracy benefit at a small fraction of its full-table cost. This is a general
cost/accuracy tradeoff argument applicable to any two-stage retrieval system, not a technique that
needs to be attributed to be used.

## Net effect on the documentation set
[`early_drafts/03_embedding_models.md`](early_drafts/03_embedding_models.md) and [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md) remain as a factual record of what was checked
(useful for your own confidence and for the methodology write-up's credibility), but nothing in
this pipeline's actual justification depends on those citations holding up — each design choice
above is independently defensible by its own engineering/statistical reasoning. If you want the
final `Documentation_template.md` methodology write-up to read as reasoned-engineering rather than
literature-review, this file is the source to draw from; [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md) is optional supporting
material for anyone who wants to check provenance, not the argument itself.
