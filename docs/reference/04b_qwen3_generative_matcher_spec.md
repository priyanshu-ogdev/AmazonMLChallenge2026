# Stage 2b — Qwen3-0.6B Causal Generative Matcher Specification

**Component:** Stage 2b Cross-Record Generative Matcher  
**Status:** Stretch goal only (build order position: behind steps 1–5; executed strictly after the end-to-end baseline is validated)  
**Primary Theoretical Grounding:** "Beyond Scale and Generation: On the Effectiveness of LLMs for Entity Matching" (Xuanlong Zhang, Yunjia Li, Iacer Calixto, Paul Groth, Sebastian Schelter), [arXiv:2607.24688](https://arxiv.org/abs/2607.24688)  
**Hardware Target:** NVIDIA GeForce RTX 3060 (12GB VRAM), bf16 precision  

---

## 1. Executive Summary & Design Scope

Stage 2b provides a high-capacity, cross-record generative matching feature that scores candidate entity pairs jointly. In earlier drafts, this component was colloquially termed "Ditto" (referencing Li et al., VLDB 2020) and collided with "Stage 2b" labels applied to off-the-shelf Qwen3-Embedding representations.

**The architecture and naming are now strictly unified:**
1. **Naming resolution:** The frozen auxiliary dense representation (`src/qwen_features.py`) is designated **Stage 2a-ii** (subordinate to Stage 2a-i BGE-M3 bi-encoder representation).
2. **Stage 2b reservation:** **Stage 2b is reserved exclusively for the cross-record generative matcher** specified in this document.
3. **Modernized backbone:** We replace legacy bidirectional masked-LM cross-encoders (`xlm-roberta-base`) with a decoder-only causal language model (**Qwen3-0.6B**), fine-tuned via LoRA to output a single-token match/no-match verdict.
4. **Distribution shift rationale & caveat:** Zhang et al. (arXiv:2607.24688) conducted a 1,215-run factorial study demonstrating that small causal LLMs (0.6B) fine-tuned with parameter-efficient adapters generalize remarkably well under distribution shift. **Crucial caveat:** The paper's empirically tested distribution shift is *schema heterogeneity* (cross-source attribute mismatches and structural reordering). For our entity resolution task, this empirical finding is applied by *reasoned analogy* to our *country and language shift* (unseen French business names, legal suffixes, and address syntax), *not* as an identical direct replication. This caveat demands our strict four-layer anti-forgetting stack and held-out-country gate.

---

## 2. Specification Table

| Component | Setting | Specification Details & Rationale |
|---|---|---|
| **Base Model** | `Qwen/Qwen3-0.6B` or `Qwen/Qwen3-0.6B-Instruct` | Apache-2.0, ~596M parameters, causal LM with LM head. (Distinct from Qwen3-Embedding which lacks an LM head). |
| **Fine-Tuning Method** | LoRA (`use_rslora=True`) | Rank $r=64$, scaling $\alpha=64$ ($\gamma = \alpha / \sqrt{r} = 8.0$), dropout 0.05. Target modules: all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). |
| **Task Format** | Serialized record pair $\rightarrow$ single verdict token | Serialized with `[COL]`/`[VAL]` tags (adapted from Ditto) ending in a decision prompt; target is single token (`Yes` / `No`). |
| **Sequence Length** | $S = 224$ tokens | Derived rigorously from empirical EDA text distributions (`dataset_eda.md`). Fully encloses 99th percentile pairs + instruction overhead. |
| **Loss Function** | Causal LM cross-entropy with verdict-token slicing | Logits are sliced before the loss computation: $\mathcal{L}_{\text{CE}} = -\log P(\text{verdict} \mid \text{pair})$. Logits computed *only* at position $T_{\text{verdict}}$. |
| **Scoring for GBM** | Sliced 2-way Softmax | $P(\text{Match}) = \frac{\exp(z_{\text{Yes}})}{\exp(z_{\text{Yes}}) + \exp(z_{\text{No}})} \in [0.0, 1.0]$ extracted as a single scalar feature for Stage 3. |
| **Anti-Forgetting Layer 1** | LoRA Rank Ceiling (Structural) | Rank 64 ceiling bounds weight perturbation $\Delta W = B \cdot A$, preserving base multilingual competencies. |
| **Anti-Forgetting Layer 2** | Sliced KL Self-Distillation | Auxiliary penalty $D_{\text{KL}}(P_{\text{frozen}}(\cdot \mid x) \parallel P_{\text{ft}}(\cdot \mid x))$ computed *only* at the verdict token, weight $\lambda_{\text{distill}} = 0.10$. |
| **Anti-Forgetting Layer 3** | Country-Balanced Pair Sampling | 50% US / 50% India candidate pairs per training batch to eliminate country-specific noise shortcut learning. |
| **Anti-Forgetting Layer 4** | Held-Out-Country Validation Gate | Train on US, validate cross-country on India (and reverse). Confirmed by EDA as the *only* viable proxy for France (0% train, 15% test). |
| **Class Imbalance** | Open Decision (Resolved Pre-Training) | Per-example loss weighting on verdict token vs. class-balanced batch sampling (e.g., 1:3 positive-to-negative ratio). |
| **Fallback Policy** | Hard Drop (No Replacement) | If the held-out-country gate fails, the feature is dropped from Stage 3. Stage 2b is strictly optional stretch; Stage 2a is the one with an off-the-shelf fallback. |
| **Base Provenance** | In-House Supervised Fine-Tuning | Fine-tuned exclusively on competition TSVs. External community checkpoints (e.g. `surajkyc/qwen3-er-...`) demoted to zero-cost sanity check only. |
| **Build Status** | Stretch-Only | Positioned behind steps 1–5 in the execution plan. |

---

## 3. Sequence Length Derivation from Real EDA

Earlier documentation assumed sequence lengths without empirical grounding. With our completed exploratory data analysis ([`dataset_eda.md`](../dataset_eda.md)), the sequence length is now mathematically pinned down:

### 3.1 Empirical Text Length Distributions (`dataset_eda.md` §5)

| Field | Mean Chars (Median) | P95 Chars (Max) | Mean Words | P95 Words |
|---|:---:|:---:|:---:|:---:|
| **Name (S1/S2/S3)** | 24.1 – 25.6 (24.0) | 36.0 – 42.0 (84) | 3.5 – 3.6 | 5.0 – 6.0 |
| **Address (S1/S2/S3)** | 46.2 – 57.3 (41.0) | 91.0 – 105.0 (217) | 7.1 – 8.6 | 14.0 – 16.0 |
| **Combined Entity** | ~75 – 83 chars | ~140 – 147 chars | ~11.5 words | ~22.0 words |

In Byte-Pair Encoding (BPE / Qwen tokenizer), business text averages $\approx 1.25$ tokens per word, meaning an entire single entity (name + address + country) consumes only **~28 to 36 tokens** at the 99th percentile.

### 3.2 Serialized Pair Representation

We serialize candidate pairs using structured column delimiters:

```text
Task: Do the following two records refer to the same business entity? Answer Yes or No.
Record 1: [COL] name [VAL] Acme Industrial Supplies Inc [COL] address [VAL] 123 Commercial Way, Suite 400, Springfield, IL 62701 [COL] country [VAL] United States
Record 2: [COL] name [VAL] Acme Industry Supply [COL] address [VAL] 123 Commercial Way Ste 400 [COL] country [VAL] United States
Match:
```

### 3.3 Token Accounting
- **Instruction Prompt & Framing Template:** $\approx 42$ tokens
- **Structural Tags (`[COL]`, `[VAL]`, field names):** $\approx 24$ tokens
- **Record 1 (Name + Address + Country, P99):** $\approx 45$ tokens
- **Record 2 (Name + Address + Country, P99):** $\approx 45$ tokens
- **Decision Prompt (`\nMatch:`):** $\approx 3$ tokens
- **Total Expected Tokens:** **$\approx 159$ tokens** (median) to **$\approx 205$ tokens** (P99 extremes).

**Decision:** We fix `max_seq_length = 224`. This comfortably encloses >99.7% of all candidate pairs without truncation, leaving a 19-token safety buffer for verbose addresses while keeping attention matrix memory strictly bounded.

---

## 4. VRAM Budget & The Verdict-Token Slicing Advantage

### 4.1 Resolving the Logits-Tensor Memory Fallacy

In standard causal language modeling, sequence-to-sequence loss functions compute logits across the full sequence:
$$\text{Logits} = \text{LM\_Head}(H), \quad H \in \mathbb{R}^{B \times S \times D}$$

Given Qwen3's large vocabulary size ($V = 152,064$), computing logits across all positions at physical batch $B=48$ and sequence length $S=224$ produces an intermediate tensor:
$$\text{Full Logits Shape} = [48, 224, 152064]$$
$$\text{Memory (bf16)} = 48 \times 224 \times 152,064 \times 2 \text{ bytes} \approx 3,269,799,936 \text{ bytes} \approx \mathbf{3.27 \text{ GB}}$$
In float32 (standard for numerically stable cross-entropy and KL divergence), this ballooned to **~6.54 GB**, which previously triggered severe VRAM warnings for a 12GB RTX 3060.

### 4.2 The Verdict-Token Slicing Solution

Because entity matching is a classification task evaluated at the terminal token, **we do not need logits for the input prompt tokens**. We slice the hidden representation at the final position *before* projecting through the LM head:

```python
# Sliced forward pass:
hidden_states = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
# Slice only the final decision token position:
verdict_hidden = hidden_states[:, -1:, :]  # Shape: [batch_size, 1, hidden_dim]
# Project only the single token to vocabulary logits:
verdict_logits = model.lm_head(verdict_hidden)  # Shape: [batch_size, 1, 152064]
```

**Memory reduction:**
$$\text{Sliced Logits Memory (fp32)} = 48 \times 1 \times 152,064 \times 4 \text{ bytes} \approx 29,196,288 \text{ bytes} \approx \mathbf{29.2 \text{ MB}}$$
The logits tensor collapses from **~3.27–6.54 GB down to ~29 MB** (a **$224\times$ reduction**). Sequence length now affects only standard transformer activation memory during the forward pass, which is modest for a 0.6B model at 224 tokens.

### 4.3 Detailed VRAM Budget on RTX 3060 (12GB)

| Memory Component | Allocation Basis | Memory Footprint (bf16 / fp32) |
|---|---|---|
| **Base Model Weights** | Qwen3-0.6B (~596M params in bf16, frozen) | ~1.19 GB |
| **Reference Model Weights** | Qwen3-0.6B frozen for self-distillation (bf16) | ~1.19 GB |
| **LoRA Trainable Parameters** | Rank 64, all linear layers (~10.2M params in fp32) | ~0.04 GB |
| **Optimizer States** | AdamW (fp32 momentum + variance on LoRA params only) | ~0.08 GB |
| **Forward Activations** | Batch 48, Seq 224, with gradient checkpointing enabled | ~1.85 GB |
| **Sliced Verdict Logits** | Shape `[48, 1, 152064]` in fp32 (both student and teacher) | ~0.06 GB (~58 MB) |
| **PyTorch & CUDA Workspace** | Driver context, memory fragmentation buffer | ~0.85 GB |
| **Total Peak VRAM Allocation** | Physical Batch 48, Gradient Checkpointing On | **~5.26 GB** |
| **Available Headroom on RTX 3060** | 12.00 GB total capacity | **~6.74 GB Headroom (56% free)** |

The training job runs safely within 12GB VRAM with over 6GB of headroom, allowing physical batch sizes up to 64 if throughput optimization is desired.

---

## 5. Loss Formulation & Anti-Forgetting Stack

### 5.1 Training Objective

The overall loss is a combination of verdict cross-entropy and verdict-position self-distillation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(z_{\text{verdict}}, y) + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{KL}}(z_{\text{verdict}}, z_{\text{frozen}})$$

1. **Verdict Cross-Entropy:**
   $$\mathcal{L}_{\text{CE}} = - \log \frac{\exp(z_{\text{target}})}{\sum_{v \in \{\text{Yes}, \text{No}\}} \exp(z_v)}$$
2. **Verdict Self-Distillation (Anti-Forgetting Layer 2):**
   $$\mathcal{L}_{\text{KL}} = D_{\text{KL}}\left(\sigma(z_{\text{frozen}} / T) \parallel \sigma(z_{\text{ft}} / T)\right) \cdot T^2$$
   Computed *only* at the terminal position, with temperature $T=1.0$ and $\lambda_{\text{distill}} = 0.10$. This penalizes the adapter from drifting away from the base model's general language and multilingual token prior.

### 5.2 Scoring for Downstream Stage 3 GBM

During inference over candidate pairs in `candidate_pairs.tsv`:
$$P(\text{Match}) = \text{Softmax}\left(\left[z_{\text{No}}, z_{\text{Yes}}\right]\right)[1] = \frac{\exp(z_{\text{Yes}})}{\exp(z_{\text{Yes}}) + \exp(z_{\text{No}})}$$
This outputs a clean scalar probability $P(\text{Match}) \in [0.0, 1.0]$ which is merged into the Stage 2 feature table alongside Stage 2a-i, Stage 2a-ii, and Stage 2c features.

### 5.3 Confirmation of Anti-Forgetting Layer 4 (The France Gate)

Our EDA ([`dataset_eda.md`](../dataset_eda.md) §3) confirmed that **France accounts for 0.0% of training data but 15.0% of test S1 records (259,452 entities)**.
- It is impossible to evaluate France generalization directly on training data.
- The two-direction cross-country evaluation (**Train on US $\rightarrow$ Evaluate on India**, and **Train on India $\rightarrow$ Evaluate on US**) is mathematically confirmed as the *only* available empirical proxy for zero-shot country transfer.
- **Go/No-Go Gate:** The fine-tuned Qwen3-0.6B matcher must achieve a macro AUCPR on the held-out country within $\le 5\%$ relative of in-domain performance. If it shows severe catastrophic forgetting or distribution collapse, **the feature is completely dropped from Stage 3**.

---

## 6. Resolving Class Imbalance Prior to Training

In candidate pairs generated from Stage 1 blocking, negative pairs heavily outnumber positive pairs (typically 15:1 to 50:1 depending on blocking thresholds). In standard binary cross-entropy, PyTorch provides `pos_weight`. In causal LM cross-entropy, loss is computed on vocabulary token indices where `pos_weight` is not natively supported.

**Decision: Resolve this explicitly before launch, not mid-run.** Two approved strategies:

### Option A: Class-Balanced Batch Sampling (Recommended)
Construct training batches dynamically with a fixed class ratio:
- For every positive link $(e_{\text{S1}}, e_{\text{pos}})$, sample $k$ hard negatives mined from Stage 1 blocking ($k \in [3, 5]$).
- Keeps batch composition consistent ($1:(k)$ balance) and naturally avoids loss scaling artifacts.

### Option B: Per-Example Loss Weighting
Compute unreduced cross-entropy loss and apply sample-level weights:
$$w_i = \begin{cases} w_{\text{pos}} & \text{if } y_i = \text{Match} \\ 1.0 & \text{if } y_i = \text{Non-Match} \end{cases}$$
$$\mathcal{L}_{\text{CE}} = \frac{1}{\sum_{i=1}^B w_i} \sum_{i=1}^B w_i \cdot \ell_i$$
where $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$ computed directly from the candidate pool.

---

## 7. Execution Order and Stretch-Only Status

In accordance with [`01_v1_baseline_plan.md`](01_v1_baseline_plan.md), Stage 2b remains a **stretch goal**:
1. Steps 1–5 (Normalization, Blocking, Stage 2a Bi-Encoder, Stage 2c Features, Stage 3 GBM, Stage 4 Decision, and Submission Packaging) must be completely executed and validated first.
2. Only if the end-to-end pipeline is verified and GPU hours remain on the timeline will Stage 2b training be launched.
3. If Stage 2b completes and passes its held-out-country gate, its $P(\text{Match})$ predictions will be added as an additional feature column to Stage 3; if it fails or runs out of time, the baseline v1 pipeline ships intact.
