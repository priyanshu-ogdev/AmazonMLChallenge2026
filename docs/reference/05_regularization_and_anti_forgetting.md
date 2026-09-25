# Regularization and dropout, by model

Each trained component's regularization stack, with the actual source for each mechanism — not just what's applied, but what SOTA work justifies it and where that evidence does or doesn't transfer to this task.

## Bi-encoder (BGE-M3's dense head / Qwen3-Embedding-0.6B)

| Mechanism | Setting | Source |
|---|---|---|
| Transformer-internal dropout (attention + hidden) | Kept near the pretrained default (~0.1); raised to 0.15-0.2 only if a held-out-country gap appears | Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," JMLR 2014 — the original dropout mechanism, applied here at the encoder's own pretrained default rather than a novel setting |
| LoRA adapter dropout | 0.05-0.1 | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685 (ICLR 2022) — the `lora_dropout` parameter is part of the original method, applied to the adapter's own forward pass |
| LoRA rank as a structural regularizer | See "LoRA rank and layer coverage" below | Same source, plus the OOD-generalization evidence in [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md) |
| Self-distillation anchor loss | Weight 0.05-0.15 | Not from a single named paper — a compliant, no-external-data adaptation of the general principle behind L2-SP-style regularization toward a starting checkpoint, applied at the embedding output rather than the weight tensors |

## Ditto cross-encoder

| Mechanism | Setting | Source |
|---|---|---|
| Encoder-internal dropout | Same as above, xlm-roberta-base's own pretrained default | Srivastava et al. 2014, as above |
| Classification-head dropout | Higher than the encoder body — 0.2-0.3 | Standard practice for freshly-initialized classification heads; not itself from a dedicated paper, justified by the head having no pretrained structure to protect, unlike the encoder body |
| MixDA (mixup-style augmentation regularization) | `del`/`swap`/`drop_col`/`append_col`, interpolated rather than substituted | Li, Li, Suhara, Doan, Tan, "Deep Entity Matching with Pre-Trained Language Models" (Ditto), VLDB 2020 / arXiv:2004.00584 |
| LoRA rank, independently tuned from the bi-encoder | See below | Hu et al. 2021, as above — re-applied here because binary classification over a joint sequence pair is a different task shape than the bi-encoder's contrastive ranking |
| Class-imbalance loss weighting | `pos_weight` sized to the true candidate imbalance | Standard `BCEWithLogitsLoss` practice, not from a dedicated paper |

## GBM meta-learner

This is the component where "dropout" needs the most care, because trees don't have neurons to drop — the actual tree-ensemble analogue is a distinct, named method, not an informal adaptation of neural dropout.

**DART — the literal SOTA citation for this layer.** Rashmi, K. V., & Gilad-Bachrach, R., "DART: Dropouts meet Multiple Additive Regression Trees," AISTATS 2015 (arXiv:1505.01866) — ranked the 8th Most Influential AISTATS 2015 Paper as of the most recent tracked edition. The mechanism: on each boosting round, a random subset of the already-built trees is muted before computing the gradient for the next tree, directly addressing what the paper calls "over-specialization" — later trees correcting only a few instances' trivial residual errors rather than learning anything general. XGBoost's own documentation cites this as DART's original paper and implements it as the `booster='dart'` mode. **Positioned in this design as an escalation for a persistent train/validation gap, not a default** — DART trains slower (defeats the prediction-caching that makes standard gradient boosting fast) and makes early stopping noisier, so it's reached for only after standard regularization (below) has been tried.

**Standard, always-on GBM regularization — distinct from DART, not a substitute for it:** `subsample`/`colsample_bytree` (row/column sampling, functionally similar to bagging — already stochastic regularizers in standard `gbtree` boosting, present whether or not DART is enabled), `reg_alpha`/`reg_lambda` (L1/L2 on leaf weights), shallow `max_depth`, and a deliberately tuned `min_child_weight` to stop the model from carving a leaf around a handful of rare positive examples. These aren't "dropout" in any sense — named here so it's explicit that DART is one additional lever among several, not the whole regularization story.

**The country-match feature masking (from a prior design pass) is a bespoke technique, not itself a cited method** — worth being honest about the distinction. It borrows the *mechanism* XGBoost/LightGBM already use for genuinely missing values (a learned default split direction, per XGBoost's own sparsity-aware split-finding documentation), applied deliberately to a specific feature as a targeted anti-shortcut-learning measure. It isn't drawn from a paper proposing this as a general regularization technique — it's this design's own application of an existing, documented library mechanism to a new purpose.

**The GBM-as-meta-learner design itself has a proper theoretical grounding, cited here explicitly since it hadn't been before:** Wolpert, D.H., "Stacked Generalization," Neural Networks, Vol. 5, No. 2, pp. 241-259, 1992 — the foundational paper establishing that training a second-stage model on a first-stage set of models' held-out predictions reduces generalization error versus picking any single first-stage model, which is exactly the shape of this pipeline's Stage 2 (bi-encoder + Ditto + hand-crafted features) feeding Stage 3 (GBM). A direct, more specific follow-up — Ting, K.M. & Witten, I.H., "Stacked Generalizations: When Does It Work?", IJCAI 1997 — addresses the question this design actually had to answer: what kind of model should sit at the meta-learner level. Both were previously implied by this design's OOF-stacking structure but never actually cited; they are now.

## LoRA rank and layer coverage — reconciled

Earlier passes in this design cycled through two different numbers — an early rank 16-32 default, later revised to 64-128 after retrieval-specific evidence on parameter budgets. Both were reasoning from partial evidence. Here's the reconciled version, checked against the original LoRA paper directly rather than secondary summaries.

**What Hu et al. (2021) actually found, and its limits.** Their own rank ablation (Table 6) tested r=1,2,4,8,64 on `W_q`/`W_v` only, and found very low ranks (r=1-4) performed competitively — the widely-repeated "low rank suffices" conclusion. But the paper states its own limit on this finding directly: *"we do not expect a small r to work for every task or dataset. Consider the following thought experiment: if the downstream task were in a different language than the one used for pre-training, retraining the entire model... could certainly outperform LoRA with a small r."* That thought experiment is close to a direct description of this design's France-generalization problem — cross-lingual shift is exactly the case the original paper itself flags as likely needing more capacity than its headline low-rank result.

**A second, independent reason not to trust the low-rank result at face value.** A follow-up paper, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA" (arXiv:2312.03732), argues that LoRA's standard scaling factor (α/r) becomes increasingly aggressive as rank grows, causing gradient collapse at higher ranks under the original scaling scheme — which "may have led the authors of (Hu et al., 2022) to inaccurately conclude that very low ranks... suffice," since higher ranks were never given a fair comparison under that scaling. **This is a real, previously-uncaught issue for this design's own rank ablation**: if the held-out-country grid tests rank 64/128 using the original α/r scaling, the comparison is biased toward making higher ranks look worse than they actually are — not because higher rank doesn't help, but because the scaling formula itself handicaps it. **Fix: use rank-stabilized scaling (γ = α/√r) for any rank above the smallest tested value**, so the rank sweep measures rank's actual effect rather than an artifact of the original paper's scaling choice.

**Layer coverage.** The original paper's low-rank result was also obtained on attention projections only (`W_q`, `W_v`). Later practice has moved past that: LoRA applied across all linear layers (attention and feed-forward, not attention-only) is now the standard default for harder or more capacity-demanding tasks, and this design already adopted that — apply LoRA to every linear layer (both models' full attention projections plus their feed-forward up/down projections), not a top-N-layers-only subset. An earlier pass in this design briefly considered restricting LoRA to only the top few layers as an additional capacity constraint; that was explicitly retired in favor of using rank alone as the primary capacity control, precisely because stacking a layer restriction on top of a rank restriction risks under-fitting the discriminative task, per the retrieval-specific parameter-budget evidence already in [`07_citations_and_benchmarks.md`](07_citations_and_benchmarks.md).

**The reconciled recommendation:**

| Component | Rank sweep | Layer coverage | Scaling |
|---|---|---|---|
| Bi-encoder LoRA | 32, 64, 128 | All linear layers (attention + FFN), never top-N-only | rank-stabilized (α/√r) for rank ≥ 64 |
| Ditto LoRA | Independent sweep, same three values as a starting grid — not assumed to match the bi-encoder's winner | Same — all linear layers | Same |
| Full fine-tuning (the comparison arm) | N/A — every parameter is trainable, no rank concept applies | All layers, unrestricted — this is deliberately the "maximum capacity, no structural constraint" pole of the ablation | N/A |

Why 32-128 rather than defaulting to the "practical 16-64" range some post-LoRA-paper guidance now recommends generally: the retrieval-specific parameter-budget finding already in this design (PEFT underperforms full fine-tuning on retrieval quality below roughly 1% of parameters, reaching parity closer to 6%) sits above what a generic instruction-tuning default targets, and a separate finding specific to smaller datasets (LoRA benefits from larger rank precisely when the target dataset is small, since capacity to fit what little labeled data exists matters more than it does at LLM-instruction-tuning data scale) points the same direction, given this design's labeled set is expected to be small. 32 is kept in the sweep as a lower anchor rather than dropped, since the held-out-country grid — not this reasoning — is what actually decides the winner.
