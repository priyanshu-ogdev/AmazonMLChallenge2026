# Layer 2 — Complete Representation and Pair-Feature Design

**Status:** final v1 design and implementation contract  
**Last reviewed:** 2026-09-25  
**Scope:** Stage 2a BGE-M3, Stage 2b Qwen3-Embedding-0.6B, and Stage 2c deterministic pair features

## 1. Why Layer 2 exists

Stage 1 blocking answers: “Which records are plausible enough to inspect?”
Layer 2 answers: “How similar is this specific S1/candidate pair, and what
evidence supports or contradicts a match?”

Layer 2 is intentionally not the final classifier. It produces a rich,
reusable representation for the Stage 3 GBM. This separation matters because:

- retrieval models optimize relative ranking, not calibrated match probability;
- deterministic features expose evidence that dense embeddings can hide;
- the GBM can learn different evidence weights for US, India, France-like
  cases, S2, and S3 without changing the encoders;
- the final threshold can optimize the competition's precision-heavy macro F0.5;
- a failed dense model can be removed without discarding lexical evidence.

The output contract is one row per pair in the final `candidate_pairs.tsv`.
Every row has stable join keys:

```text
source1_entity_id, candidate_entity_id, [Stage 2 features...]
```

No Layer 2 component creates candidates, uses ground-truth labels during
inference, or makes the final match/no-match decision.

## 2. Architectural choices

### 2.1 Why three complementary feature families

**BGE-M3 (2a)** supplies a multilingual dense representation adapted to the
challenge's noisy business-name/address pairs. It captures paraphrases,
transliteration, and reordered text that exact rules miss.

**Qwen3-Embedding-0.6B (2b)** is an independent, inference-only dense view.
It is included only if an ablation proves that its errors are complementary to
BGE and lexical features. A larger benchmark score does not justify inclusion
by itself.

**Deterministic features (2c)** provide transparent signals: exact matches,
token overlap, edit similarity, numbers, postal codes, missingness, and
contradictions. They are especially important for the precision-heavy metric,
where a dense false merge is costly.

The GBM receives all three families as candidate evidence. It must not receive
raw entity IDs, row order, or target-derived statistics.

## 3. Stage 2a — BGE-M3 LoRA bi-encoder

### 3.1 Model decision

BGE-M3 is the committed trainable encoder because it already has multilingual
encoder representations and can be adapted with a small PEFT adapter. The
base model remains frozen; only LoRA matrices are updated. This protects the
pretrained representation structurally, but it does **not** prove that the
adapter cannot hurt unseen-language behavior. The held-out-country gate is
therefore a deployment gate, not an optional report.

Qwen is not fine-tuned in v1 because its decoder-style architecture, pooling,
target modules, and instruction interface would require a separate validated
training pipeline. That work is not necessary to obtain an independent dense
feature.

### 3.2 Why LoRA rank 64 and rank-stabilized scaling

Full fine-tuning is rejected because it updates all 568M parameters, increases
optimizer and activation pressure, and makes catastrophic domain-specific
drift harder to diagnose. LoRA constrains the update to low-rank directions.

Rank 64 is a committed engineering choice rather than an unbounded sweep:

- rank 32 may under-capacity a retrieval adaptation;
- rank 128 increases capacity and overfitting risk without a time-budgeted
  experiment proving need;
- rank 64 is a compromise for retrieval adaptation on a 12GB RTX 3060.

`use_rslora=True` uses rank-stabilized scaling rather than assuming that the
standard alpha/r scaling behaves equally across ranks. This is a rationale for
the configuration, not evidence that rank 64 is universally optimal.

All linear modules are targeted because name/address matching is not limited to
attention projections; feed-forward transformations also affect the compressed
entity representation. Restricting both rank and layer depth would compound
capacity constraints.

### 3.3 Why CachedMNRL

The training objective is:

```text
L = CachedMNRL(anchor, positive) + 0.10 * AnchorDistillation
```

Cached multiple-negatives ranking loss provides many in-batch negatives while
gradient caching keeps activation memory bounded by the mini-batch. It directly
optimizes ranking, which is the useful property for a candidate feature.
Triplet loss is not the default because its result can be highly sensitive to
negative selection and learning rate and can collapse under an unsuitable
setup.

The loss is not a probability loss. Raw cosine values must therefore be
treated as features, not final probabilities; Stage 3 calibration owns
probability interpretation.

### 3.4 Why self-distillation is necessary

There is no French training data. A frozen copy of base BGE-M3 supplies a
reference embedding for each observed training text:

```text
anchor_loss = mean(1 - cosine(finetuned_embedding, frozen_embedding))
```

This discourages unnecessary movement of the multilingual space while the
contrastive term learns business matching. It is not replay training and does
not import external French data. Weight 0.10 is a bounded starting point:
larger values can suppress useful adaptation; smaller values may provide too
little protection.

The extra forward pass is accepted because the sequence is short and the
generalization gate is more valuable than maximum training speed.

### 3.5 Data and anti-forgetting stack

The BGE run uses balanced US/India entity sampling and excludes singleton
anchors from contrastive positive-pair construction. Singletons remain
important for the downstream GBM and final decision metric; they are not
discarded from the overall problem.

The complete protection stack is:

1. LoRA rank ceiling;
2. self-distillation against frozen base BGE;
3. balanced country sampling and, once Stage 1 exists, balanced hard-negative
   mining;
4. same-domain perturbation augmentation from supplied records only;
5. two-direction held-out-country retrieval gate.

Hard negatives and augmentation are controlled ablations. They must not be
enabled without recording whether candidate recall and false-merge behavior
improve.

### 3.6 Acceptance gate

Run:

1. US train → India evaluation;
2. India train → US evaluation;
3. unchanged BGE-M3 baseline on the same queries and corpus.

Record Recall@1/5/10/20/50, MRR, precision@10, hardest-negative margin
statistics, query count, and country/source slices. Accept only when both
directions satisfy the absolute and relative checks in
`docs/FINAL_DESIGN.md`. A failed run is retained as an experiment artifact;
it is never silently promoted.

Recovery is deliberately limited: distillation weight 0.15, then rank 32.
If no run passes, Stage 2a uses unchanged BGE-M3.

## 4. Stage 2b — Qwen3 auxiliary dense feature

Qwen is an inference-only complementary view, not a replacement training
target and not an automatic BGE fallback.

The implementation in `src/qwen_features.py`:

1. loads each entity from the supplied TSV files;
2. applies the same task instruction to S1 and S2/S3 text;
3. encodes each unique entity once;
4. L2-normalizes embeddings;
5. computes a dot product for exactly the final candidate pairs;
6. writes `qwen_cosine` plus an encoding metadata sidecar.

Symmetric treatment is mandatory. Query-only prompting would introduce an
artificial distribution difference between the two sides of the pair.

Qwen is retained only if:

```text
GBM(2c + BGE) + Qwen
```

improves entity-level macro F0.5 over:

```text
GBM(2c + BGE)
```

without worsening country/source slices, false merges, or the held-out-country
proxy. Otherwise its feature is omitted even if its standalone retrieval score
looks attractive.

## 5. Stage 2c — deterministic pair features

`src/pair_features.py` computes label-free features for the final candidates.
The feature groups and their purpose are:

| Group | Signals | Purpose |
|---|---|---|
| Name | exactness, token Jaccard, overlap, edit similarity, character trigrams | tolerate abbreviations, typos, reordering |
| Address | token Jaccard, overlap, edit similarity, character trigrams | tolerate partial/reordered addresses |
| Numbers | name/address numeric-token overlap | protect unit, street, and registration-number evidence |
| Postal | equality and missingness | strong location evidence without external lookup |
| Missingness | name/address/country/postal indicators | distinguish absent evidence from disagreement |
| Contradictions | same name/different address and same address/different name | precision protection against false merges |
| Metadata | country equality, blocker provenance, candidate rank | let the scorer learn source and retrieval context |

The extractor keeps IDs only for joins and rejects unknown references. Learned
TF-IDF statistics are intentionally not fitted here; they must be trained
inside each Stage 3 fold to prevent leakage.

## 6. Pair-feature invariants

For every Stage 2 run:

- row count equals the deduplicated final candidate-pair count;
- every candidate pair has a feature row;
- no feature uses ground truth, target labels, or future fold information;
- missing values have explicit semantics rather than silent defaults;
- embedding features are normalized before cosine computation;
- candidate provenance is descriptive, not a replacement for similarity;
- all test countries use the same feature code path;
- feature schemas and model revisions are written to metadata;
- deterministic features are reproducible from the input TSVs and seed.
- an empty candidate set still produces the complete, stable feature schema.

## 7. What Stage 2 does not decide

Stage 2 does not:

- choose the candidate set;
- calibrate cosine scores;
- select the final threshold;
- force one match per S1;
- remove singletons;
- train the GBM;
- use external business knowledge.

Those responsibilities belong to Stages 1, 3, 4, and 5.

The Layer 3 scorer and calibration contract is documented in
[`LAYER_3_OVERVIEW.md`](LAYER_3_OVERVIEW.md). Stage 2 outputs are accepted
only when they satisfy that downstream schema and grouped-OOF protocol.

## 8. Ablation and experiment order

Run experiments in this order:

1. lexical/deterministic 2c only;
2. 2c + unchanged BGE;
3. 2c + accepted BGE LoRA;
4. 2c + accepted BGE + Qwen;
5. accepted dense configuration + hard negatives;
6. accepted dense configuration + same-domain augmentation.

All comparisons use the same entity-grouped folds, candidate set, calibration
protocol, and macro-F0.5 aggregation. A feature is accepted only when it
improves the complete downstream task, not an isolated retrieval metric.

## 9. Implementation status

| Component | Status |
|---|---|
| BGE-M3 LoRA training | implemented |
| BGE held-out evaluation | implemented, reverse-direction orchestration remains |
| Qwen inference feature | implemented |
| Deterministic pair features | implemented |
| Blocking | pending |
| Fold-safe learned TF-IDF features | pending Stage 3 |
| GBM and OOF calibration | implemented |
| F0.5 threshold and submission assembly | implemented |

Layer 2 is complete as a feature-generation contract when the BGE gate has
been run, the candidate set is available, and the feature-schema invariants
above pass on the actual challenge data.

### Known execution boundaries

The current repository does not yet orchestrate the reverse BGE gate from one
command: the data builder emits one held-out direction at a time, so the
India→US run must be prepared and executed as a second explicit run. This is
an execution gap, not permission to accept a one-direction result.

The Stage 2c `blocker_provenance` value is metadata for Stage 3 encoding; it
must be categorical-encoded or reduced to numeric provenance indicators inside
each training fold. It must never be passed as a raw string to a GBM.

The BGE evaluator's checkpoint metric is constructed from the evaluator name
(`eval_<name>_cosine_recall@10`). If a future Sentence-Transformers upgrade
changes that naming contract, inspect the emitted evaluation log and update
the configuration before trusting `load_best_model_at_end`.
