# Bi-Encoder LoRA Fine-Tuning — Design Rationale

## How this document was built

This document traces the complete thinking process behind every design decision
in the bi-encoder LoRA fine-tuning module. It's structured as a decision log:
for each choice, what alternatives existed, what evidence was weighed, what
trade-offs were accepted, and what the docs_final documentation specified vs.
what was upgraded through independent research.

---

## 1. Why a bi-encoder at all?

### The problem shape

Entity resolution at this scale (2.2M S1 entities × 10M S2/S3 candidates) requires
two fundamentally different capabilities:

1. **Retrieval** — given an S1 entity, find plausible candidates from S2/S3
   (this is the blocking stage, Stage 1)
2. **Scoring** — given an (S1, S2/S3) pair, produce a match probability
   (this is the matching stage, Stages 2-4)

A bi-encoder serves Stage 2a: it encodes each entity independently into a dense
vector, and cosine similarity between vectors becomes a feature for the downstream
GBM. This is fundamentally different from a cross-encoder (which processes both
entities jointly) — the bi-encoder trades accuracy for the ability to pre-compute
and cache embeddings.

### Why not just use the blocking model (BGE-M3) directly?

BGE-M3 is already used for blocking (Stage 1). Its off-the-shelf embeddings could
also serve as the Stage 2a feature without any fine-tuning. The question is whether
fine-tuning adds value.

Evidence that it does:
- Narayana et al. (arXiv:2608.16161): BGE-base-en-v1.5 improved from 15.25% to
  92.70% pass rate at margin 0.30 after fine-tuning for entity resolution
- TriBERTa (arXiv:2411.10629): fine-tuned SBERT outperformed un-fine-tuned by 3-19%
- The task distribution (business names + addresses with noise) is different from
  BGE-M3's pretraining distribution (web text retrieval)

The risk: fine-tuning can damage the pretrained multilingual structure needed for
France. This is exactly what the anti-forgetting stack exists to protect.

---

## 2. Model selection: BGE-M3 vs. Qwen3-Embedding-0.6B

### The docs' original reasoning (v1-baseline.md)

The original v1 plan chose **Qwen3-Embedding-0.6B off-the-shelf** (no fine-tuning)
because it outperforms BGE-M3 on zero-shot French retrieval:
- PosIR Multilingual Retrieval benchmark:
  - BGE-M3: 50.79 nDCG@1 (English), 44.07 (French)
  - Qwen3-Embedding-0.6B: 65.10 (English), 55.33 (French)

### Why the decision reversed for fine-tuning

When fine-tuning was promoted into v1 scope (hardware confirmed: RTX 3060 12GB),
the comparison changes fundamentally. It's no longer "which model is better
off-the-shelf" — it's "does supervised adaptation of BGE-M3's own multilingual
pretraining beat an un-adapted competitor."

BGE-M3 is XLM-RoBERTa-based (568M params), pretrained on 100+ languages including
French. Fine-tuning *adapts* multilingual structure it already has. The anti-forgetting
stack is designed to protect exactly this capability.

Qwen3-Embedding-0.6B is a decoder-style model (Qwen3 base) — architecturally
different (unidirectional vs. bidirectional), different target modules for LoRA
(q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj vs. query/key/value/dense),
and different pooling (last-token vs. CLS). Fine-tuning it would require a
completely different configuration that hasn't been validated.

**Decision**: Fine-tune BGE-M3; keep Qwen3-Embedding-0.6B as a free second feature
(inference-only, no training cost).

---

## 3. LoRA configuration

### Why LoRA (not full fine-tuning)

Three independent evidence sources:

1. **"LoRA Learns Less and Forgets Less" (TMLR)**: LoRA underperforms full
   fine-tuning on target-task fit but better preserves OOD performance.
   Forgetting mitigation is strongest for large domain shifts.

2. **"Back to Basics" (arXiv:2311.09765)**: LoRA outperforms full fine-tuning
   by 4.3-13.7% on out-of-domain retrieval. Task-matched to our decision.

3. **RepLLaMA (arXiv:2310.08319)**: Full fine-tuning wins on training-set metrics
   but only marginally on held-out dev; LoRA wins on genuinely independent
   judgment sets.

The France generalization gap is a large domain shift (unseen language in test).
LoRA's structural constraint (low-rank perturbation can't fully overwrite
pretrained weights) is the primary defense.

### Why rank 64 (not 32 or 128)

From the docs' reconciliation (regularization.md):

- **Hu et al. (2021) found low ranks (r=1-4) sufficient** — but the paper itself
  caveats: "we do not expect a small r to work for every task or dataset. Consider
  the following thought experiment: if the downstream task were in a different
  language than the one used for pre-training..." This is close to our France problem.

- **"Scattered or Connected" (arXiv:2208.09847)**: PEFT underperforms full
  fine-tuning on retrieval below ~1% of parameters; parity requires ~6%.
  For BGE-M3 (568M params), 6% = 34M trainable params. At r=64 with all-linear,
  we get ~28.3M — close to the parity threshold.

- **r=32 (14.2M params)**: too far below the retrieval-specific threshold.
- **r=128 (56.6M params)**: no evidence it's needed; risks overfitting the
  small labeled set without justification.
- **r=64 (28.3M params)**: middle ground, with negligible VRAM cost (0.34GB
  optimizer overhead vs. 9.3GB available).

**This is a committed value, not a sweep** — deliberately chosen so fine-tuning
stays a same-day task rather than a research ablation.

### Why rank-stabilized scaling (use_rslora=True)

This is a research upgrade not explicitly in the original docs but motivated
by the docs' own citation (arXiv:2312.03732).

The problem: LoRA's standard scaling factor (α/r) becomes increasingly aggressive
as rank grows. At r=64 with α=64, the standard scale is 64/64 = 1.
The rank-stabilization paper argues this biased the original LoRA paper's rank
ablation toward under-rating higher ranks — "may have led the authors to
inaccurately conclude that very low ranks suffice."

The fix: use α/√r scaling. With α=64 and r=64: effective scale = 64/√64 = 64/8 = 8.
This gives the LoRA update appropriate magnitude relative to the pretrained weights.

The PEFT library implements this as `use_rslora=True` in `LoraConfig`. This was
confirmed through web research — the parameter is supported in PEFT ≥ 0.7.0.

### Why all-linear (not attention-only, not top-N layers)

From regularization.md:
- The original LoRA paper tested only attention projections (W_q, W_v).
- Later practice has moved to all linear layers for harder tasks.
- Restricting to top-N layers was considered and explicitly retired:
  "stacking a layer restriction on top of a rank restriction risks under-fitting
  the discriminative task."

`target_modules="all-linear"` in PEFT applies LoRA to every nn.Linear module
in the transformer encoder: attention (query, key, value, output.dense) and
FFN (intermediate.dense, output.dense), all 12 layers.

---

## 4. Loss function design

### Why CachedMultipleNegativesRankingLoss

From training.md: "Not raw triplet loss as the default — triplet loss can produce
genuine, severe collapse depending on learning rate and setup."

Evidence:
- vstash paper (arXiv:2604.15484): TripletLoss caused NDCG to drop from 0.6464
  to 0.0550 (-91.5%) while MNRL improved results 5.6-7.4%.
- The paper's own text notes a learning rate confound, but the risk is real.

CachedMNRL specifically (vs. plain MNRL):
- Decouples effective batch size from physical VRAM
- Physical mini-batch of 16, but the contrastive loss sees all 48 samples
  in the batch as potential negatives
- "Cache" refers to gradient caching: embeddings are computed in no_grad
  mini-batches, loss is computed on all of them, gradients are cached,
  then recomputed with grad in mini-batches

### Why self-distillation (the anchor loss)

From training.md: "a compliant, no-external-data substitute for replay training."

The problem: we have zero French training data. We can't do replay training
(re-training on French examples to prevent forgetting) because that would
require external French data (prohibited by competition rules).

The self-distillation term computes:
```
distill_loss = mean(1 - cos_sim(fine_tuned_embedding, frozen_base_embedding))
```

This penalizes drift from the pretrained embedding space — including on
token patterns the US/India training data never exercises (like French
vocabulary). The frozen model serves as the "memory" of what the pretrained
space looked like.

**Why 0.10 weight**: mid-point of the 0.05-0.15 range from training.md.
Too high (>0.20) would overwhelm the contrastive loss and prevent learning.
Too low (<0.05) would be a negligible constraint.

**Compute overhead**: One extra forward pass per batch through both models.
For 568M params with 80-token sequences, this adds ~50% compute time.
On a 3060 with the expected data volume, this adds ~30-90 minutes to training.
Acceptable in a 72-hour window.

### The double-forward-pass trade-off

The current implementation does two forward passes through the training model
per batch:
1. CachedMNRL's internal forward (with gradient caching)
2. Self-distillation forward (standard autograd)

This is technically redundant — both compute embeddings from the same inputs.
A theoretically cleaner approach would intercept CachedMNRL's internal
embeddings and share them with the distillation term. However:
- CachedMNRL's gradient caching mechanism manages its own forward passes
  internally and doesn't expose intermediate embeddings
- Modifying CachedMNRL's internals risks breaking its gradient caching logic
- The 50% overhead is acceptable for the training scale (100K entities, 3 epochs)
- The implementation is clean, testable, and independent

Trade-off accepted: computational waste for implementation simplicity and
correctness confidence.

---

## 5. Data strategy

### Why 100K entities (50K per country)

The full dataset has 2.08M entities with matches. Using all of them would:
- Produce ~7.6M positive pairs (avg 3.67 matches per entity)
- Take much longer to train (potentially 10+ hours for 3 epochs)
- Not proportionally improve the bi-encoder (diminishing returns)

100K entities (50K US + 50K India) produces ~370K positive pairs — large enough
for robust learning, small enough for same-day training on a 3060.

The balance (50/50) is the anti-forgetting stack's layer 3: country-balanced
mining ensures the model doesn't learn US-specific noise patterns at the
expense of India patterns.

### Why no explicit hard negatives (default)

The docs specify hard negatives should come from blocking output (top-K
candidates not in ground truth). Since blocking hasn't been built yet,
the data builder defaults to `negatives_per_positive=0`, relying entirely
on CachedMNRL's in-batch negatives.

With physical batch size 48, each positive pair sees 47 in-batch negatives.
Some of these will be genuinely hard (entities with similar names but different
businesses). This is sufficient for initial training.

When blocking is built, the `--blocking_candidates_path` flag enables
re-mining harder negatives from the actual candidate set.

### Why singletons are excluded

Singletons (S1 entities with no matches — 5.6% of training data) can't form
positive pairs for contrastive learning. The bi-encoder's job is to produce
similarity scores, not to decide "match vs. no match" — that decision belongs
to the GBM (Stage 3) and the threshold (Stage 4).

Including singletons would require a different loss function (e.g., classification)
which is the cross-record matcher's (Stage 2b generative matcher's) territory, not the bi-encoder's.

---

## 6. Anti-forgetting stack — why four layers

The France generalization gap is the single hardest constraint:
- Zero French training data
- ~15% of test data is French
- F₀.₅ is macro-averaged per entity — every French entity counts equally

Each layer addresses a different failure mode:

### Layer 1: LoRA rank ceiling (structural)
- **What it prevents**: Complete overwriting of pretrained multilingual weights
- **How**: Low-rank perturbation (W' = W + BA, where B and A are low-rank)
  can only modify the weight matrices along 64 directions (out of 1024),
  leaving the vast majority of pretrained structure intact
- **Cost**: Zero — it's the fine-tuning method itself, not an add-on

### Layer 2: Self-distillation (soft constraint)
- **What it prevents**: Gradual drift of the embedding space away from pretrained
- **How**: Cosine distance penalty against frozen base model
- **Cost**: ~50% compute overhead, 1.14GB VRAM for frozen model

### Layer 3: Country-balanced mining (data)
- **What it prevents**: Learning US-specific noise patterns (e.g., "if the name
  contains 'LLC', probably a match" — an artifact of US legal suffix patterns)
- **How**: Equal US/India representation in training batches
- **Cost**: May slightly reduce effective data diversity if one country has more
  varied entities

### Layer 4: Held-out country gate (validation)
- **What it prevents**: Deploying a fine-tune that actually hurts French performance
- **How**: Train on US, validate on India (France proxy). If held-out Recall@10
  drops below 0.80 or is more than 0.10 worse than baseline → reject
- **Cost**: Requires training twice (gate run + final run) if the gate passes

**Why all four, not just LoRA?**
Each layer catches a different failure mode. LoRA alone can still overfit to
US/India patterns through the 64 trainable directions. Self-distillation alone
doesn't prevent systematic shortcuts. Country balancing alone doesn't verify
the outcome. The gate alone doesn't prevent the problem, only detect it.

---

## 7. Evaluation design

### The two-tier protocol

From training.md: "Two-tier: a cheap proxy drives early stopping; the expensive
confirmation is computed once per candidate configuration."

**Cheap proxy (during training)**:
- InformationRetrievalEvaluator: Recall@K on held-out country
- Runs every 500 steps via SentenceTransformerTrainer
- Used for early stopping and checkpoint selection

**Expensive confirmation (post-training)**:
- Full retrieval evaluation with margin analysis
- Comparison against off-the-shelf baseline
- Go/no-go gate decision

### Why India as a France proxy

We can't evaluate on French data (none in training). India is the closest
available proxy because:
- Different script (Devanagari vs. Latin)
- Different address patterns
- Different business naming conventions
- If the model preserves India performance after training on US-heavy data,
  it's evidence that multilingual structure is intact

This isn't a perfect proxy — French has its own challenges (accented characters,
different legal suffixes, different address structure). But it's the best
available without violating the no-external-data constraint.

---

## 8. Hardware constraints and how they shaped decisions

### RTX 3060 12GB — the binding constraint

The 12GB VRAM budget shaped several decisions:
1. **LoRA over full fine-tuning**: Full fine-tuning optimizer state would use
   568M × 12 bytes = 6.8GB, leaving only 5.2GB for everything else
2. **Physical batch 48, not 64**: 64 would push activation memory higher
3. **CachedMNRL mini-batch 16**: Gradient caching at 16 keeps peak activation
   memory bounded regardless of the 48-sample effective batch
4. **Max seq length 80**: Business names + addresses are short (avg ~15 tokens).
   Using the model's native 8192 context would waste VRAM on padding
5. **bf16 throughout**: Halves memory for activations and weights vs. fp32
6. **Self-distillation fit check**: Two copies of BGE-M3 (2.28GB) + optimizer
   (0.34GB) = 2.62GB → 9.38GB free for activations. This was verified to fit
   before committing to distillation.

### CPU laptop for data preparation

Data preparation was designed to work without a GPU:
- Chunked TSV reading (100K rows per chunk) for memory efficiency
- All text normalization is pure Python string operations
- HuggingFace Dataset saved to disk as Arrow files (memory-mapped during training)
- No model inference needed during data preparation

---

## 9. What was upgraded from the original documentation

| Area | Docs specification | Upgrade | Rationale |
|------|-------------------|---------|-----------|
| LoRA scaling | "rank-stabilized scaling (α/√r)" | `use_rslora=True` in PEFT | Docs described the math; implementation uses the actual PEFT parameter |
| PEFT integration | Manual PEFT wrapping implied | `model.add_adapter()` native API | Cleaner, supports proper save/load of adapters |
| Batch sampling | Not specified | `BatchSamplers.NO_DUPLICATES` | Ensures unique samples per batch for better in-batch negatives |
| LR scheduler | "warmup_ratio: 0.1" (linear implied) | Cosine annealing | Generally better than linear decay for fine-tuning |
| Evaluation | "margin/score-gap pass rate, Recall@K" | Full IR evaluator with MRR, Precision@K, margin analysis | More comprehensive, automated go/no-go decision |
| Device handling | Not specified | Lazy device sync in loss | Handles trainer's GPU placement automatically |
| Data loading | Not specified | Chunked TSV reading | Memory-efficient for laptop CPUs |

---

## 10. What was deliberately NOT done

| Decision | Why |
|----------|-----|
| LoRA rank sweep (32/64/128) | Docs specify a committed value, not a sweep. 72-hour hackathon. |
| Qwen3-0.6B Generative Matcher (Stage 2b) | Cut for v1 per v1-baseline.md. Stretch goal (see [`reference/04b_qwen3_generative_matcher_spec.md`](reference/04b_qwen3_generative_matcher_spec.md)). |
| DART for GBM | Escalation only if standard regularization fails. |
| External French data | Prohibited by competition rules. |
| Synthetic French augmentation | "Guesswork dressed as data" — docs explicitly warn against this. |
| Full fine-tuning comparison | Requires separate VRAM budget and time. LoRA evidence is strong enough. |
| Custom MNRL with shared forward pass | Compute overhead of separate distillation pass is acceptable. |
| Hard negative mining from TF-IDF | Added complexity before blocking exists. In-batch negatives suffice. |

---

## 11. Open questions for the implementation session

These are resolved by the validation protocol, not by further design:

1. **Does the held-out country gate pass?** Only trainable once we run the code.
2. **Is distillation weight 0.10 optimal?** If gate fails, iterate toward 0.15.
3. **Does gradient checkpointing cause issues with CachedMNRL?** Both use
   recomputation — test empirically.
4. **Is 100K entities enough?** If eval metrics plateau early, could reduce
   to 50K for faster iteration.

---

## 12. Relationship to the rest of the pipeline

This bi-encoder is ONE feature in a larger pipeline:

```
Stage 0: Normalize text (country-agnostic)
Stage 1: BGE-M3 blocking (dense + sparse + token + phonetic)
Stage 2a-i: *** THIS MODULE *** — fine-tuned BGE-M3 bi-encoder cosine similarity
Stage 2a-ii: Qwen3-Embedding-0.6B auxiliary dense feature (inference-only)
Stage 2b: [Qwen3-0.6B Causal Generative Matcher — stretch goal, cut for v1]
Stage 2c: Hand-crafted features (Levenshtein, Jaccard, TF-IDF, etc.)
Stage 3: GBM combining all Stage 2 features → calibrated probability
Stage 4: F₀.₅-optimized threshold → match/no-match decision
Stage 5: Output assembly + validation
```

The bi-encoder's output is consumed by the GBM as one feature column among many.
The GBM makes the final decision, not the bi-encoder. This means:
- The bi-encoder doesn't need to be perfectly calibrated (GBM + isotonic does that)
- The bi-encoder doesn't need to be the best single model (ensemble effect)
- The bi-encoder DOES need to not damage French scores (the GBM can't recover
  from a corrupted feature)
