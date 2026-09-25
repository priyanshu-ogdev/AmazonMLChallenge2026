# Stage 2b — Qwen3-0.6B Causal Generative Matcher Specification (Stretch)
## Amazon ML Challenge 2026 — Business Entity Resolution

---

## 1. Executive Summary & Design Scope

**Stage 2b** is a stretch-goal cross-record matcher providing high-capacity sequence-pair classification. Unlike bi-encoders (which encode records independently), a generative cross-record matcher scores both records **jointly**, allowing full cross-attention across all tokens in both business names and addresses.

### Academic Grounding & The Distribution-Shift Caveat
This component is theoretically grounded in **"Beyond Scale and Generation: On the Effectiveness of LLMs for Entity Matching"** (Xuanlong Zhang, Yunjia Li, Iacer Calixto, Paul Groth, Sebastian Schelter), [arXiv:2607.24688](https://arxiv.org/abs/2607.24688). Zhang et al. conducted a 1,215-run controlled factorial study demonstrating that small causal LLMs (0.6B) fine-tuned with parameter-efficient adapters match or outperform larger models while generalizing robustly under distribution shift.

> [!WARNING]
> **The Distribution-Shift Caveat:** In Zhang et al., the empirically evaluated distribution shift was *schema heterogeneity* (unseen attribute names and structural reordering). For our entity resolution challenge, we apply this empirical finding by *reasoned analogy* to our *country and language shift* (unseen French business names, legal suffixes, and address syntax), **not** as a direct experimental replication. This distinction makes our 4-layer anti-forgetting stack and held-out-country gate mandatory before trusting Stage 2b predictions.

---

## 2. Specification Table

| Attribute | Setting | Architectural Details & Rationale |
|---|---|---|
| **Base Model** | `Qwen/Qwen3-0.6B` or `Qwen/Qwen3-0.6B-Instruct` | Apache-2.0, ~596M parameters, causal LM with language modeling head. Distinct from `Qwen3-Embedding` which lacks an LM head. |
| **Fine-Tuning Method** | LoRA (`use_rslora=True`) | Rank $r=64$, scaling $\alpha=64$ ($\gamma = \alpha / \sqrt{r} = 8.0$), dropout $0.05$. Target modules: all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). |
| **Task Format** | Serialized record pair $\rightarrow$ single verdict token | Serialized with `[COL]`/`[VAL]` structural delimiters ending in a decision prompt; target is single token (`Yes` / `No`). |
| **Sequence Length** | $S = 224$ tokens | Derived rigorously from empirical EDA text distributions (`dataset_eda.md`). Fully encloses 99.7th percentile pairs + instruction overhead. |
| **Loss Function** | Causal LM cross-entropy with verdict-token slicing | Logits are sliced before loss computation: $\mathcal{L}_{\text{CE}} = -\log P(\text{verdict} \mid \text{pair})$. Logits computed *only* at position $T_{\text{verdict}}$. |
| **Scoring for GBM** | Sliced 2-way Softmax | $P(\text{Match}) = \frac{\exp(z_{\text{Yes}})}{\exp(z_{\text{Yes}}) + \exp(z_{\text{No}})} \in [0.0, 1.0]$ extracted as a single scalar feature for Stage 3. |
| **Anti-Forgetting Layer 1** | LoRA Rank Ceiling (Structural) | Rank 64 ceiling bounds weight perturbation $\Delta W = B \cdot A$, preserving base multilingual competencies. |
| **Anti-Forgetting Layer 2** | Sliced KL Self-Distillation | Auxiliary penalty $D_{\text{KL}}(P_{\text{frozen}}(\cdot \mid x) \parallel P_{\text{ft}}(\cdot \mid x))$ computed *only* at the verdict token, weight $\lambda_{\text{distill}} = 0.10$. |
| **Anti-Forgetting Layer 3** | Country-Balanced Pair Sampling | 50% US / 50% India candidate pairs per training batch to eliminate country-specific noise shortcut learning. |
| **Anti-Forgetting Layer 4** | Held-Out-Country Validation Gate | Train on US, validate cross-country on India (and reverse). Confirmed by EDA as the *only* viable proxy for France (0% train, 15% test). |
| **Class Imbalance** | Class-Balanced Batch Sampling | Sample $k \in [3, 5]$ hard negatives per positive pair in training batches to stabilize causal LM gradient dynamics. |
| **Build Status** | Stretch-Only | Positioned strictly behind steps 1–5 in the execution plan. If the gate fails, the feature is dropped with zero pipeline impact. |

---

## 3. Sequence Length Derivation from Real EDA

Earlier literature assumed sequence lengths without empirical grounding. Based on our exploratory data analysis ([`docs/01_dataset_eda.md`](01_dataset_eda.md) §5), the sequence length is mathematically derived:

### Empirical Text Length Distributions:
- **Business Names:** Mean 24.1–25.6 chars (3.5 words); P95 is 36.0–42.0 chars (5.0–6.0 words); Max 84 chars.
- **Business Addresses:** Mean 46.2–57.3 chars (7.1–8.6 words); P95 is 91.0–105.0 chars (14.0–16.0 words); Max 217 chars.
- **Combined Entity:** Mean ~11.5 words; P95 is ~22 words; P99 is ~36 words.

In Byte-Pair Encoding (Qwen tokenizer), business text averages $\approx 1.25$ tokens per word. An entire single entity consumes **~28 to 36 tokens** at the 99th percentile.

### Serialized Pair Representation:
```text
Task: Do the following two records refer to the same business entity? Answer Yes or No.
Record 1: [COL] name [VAL] Acme Industrial Supplies Inc [COL] address [VAL] 123 Commercial Way, Suite 400, Springfield, IL 62701 [COL] country [VAL] United States
Record 2: [COL] name [VAL] Acme Industry Supply [COL] address [VAL] 123 Commercial Way Ste 400 [COL] country [VAL] United States
Match:
```

### Exact Token Accounting:
- **Instruction Prompt & Template Framing:** $\approx 42$ tokens
- **Structural Tags (`[COL]`, `[VAL]`, field names):** $\approx 24$ tokens
- **Record 1 (Name + Address + Country, P99):** $\approx 45$ tokens
- **Record 2 (Name + Address + Country, P99):** $\approx 45$ tokens
- **Decision Prompt (`\nMatch:`):** $\approx 3$ tokens
- **Total Expected Tokens:** **$\approx 159$ tokens** (median) to **$\approx 205$ tokens** (P99 extreme).

**Committed Setting:** `max_seq_length = 224`. This completely encloses $>99.7\%$ of candidate pairs without truncation, leaving a 19-token safety buffer for verbose addresses.

---

## 4. VRAM Budget & The Sliced Verdict-Token Memory Proof

### 4.1 Resolving the Full-Sequence Logits Memory Bottleneck
In standard causal language modeling, sequence-to-sequence loss functions compute logits across the full sequence:
$$\text{Logits} = \text{LM\_Head}(H), \quad H \in \mathbb{R}^{B \times S \times D}$$

Given Qwen3's large vocabulary size ($V = 152,064$), computing logits across all positions at physical batch $B=48$ and sequence length $S=224$ produces an intermediate tensor:
$$\text{Full Logits Shape} = [48, 224, 152064]$$
$$\text{Memory (bf16)} = 48 \times 224 \times 152,064 \times 2 \text{ bytes} \approx 3,269,799,936 \text{ bytes} \approx \mathbf{3.27 \text{ GB}}$$
In float32 (required for numerically stable cross-entropy and KL divergence), this ballooned to **~6.54 GB**, which previously triggered severe VRAM warnings for a 12GB RTX 3060.

### 4.2 The Verdict-Token Slicing Solution
Because entity matching is evaluated at the terminal token, **we do not need logits for the input prompt tokens**. We slice the hidden representation at the final position *before* projecting through the LM head:

```python
# Forward pass through transformer backbone
hidden_states = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

# Slice ONLY the final decision token position:
verdict_hidden = hidden_states[:, -1:, :]  # Shape: [batch_size, 1, hidden_dim]

# Project only the single token to vocabulary logits:
verdict_logits = model.lm_head(verdict_hidden)  # Shape: [batch_size, 1, 152064]
```

**Memory Reduction:**
$$\text{Sliced Logits Memory (fp32)} = 48 \times 1 \times 152,064 \times 4 \text{ bytes} \approx 29,196,288 \text{ bytes} \approx \mathbf{29.2 \text{ MB}}$$
The logits tensor collapses from **~3.27–6.54 GB down to ~29 MB** (a **$224\times$ reduction**). Sequence length now affects only standard activation memory during the forward pass.

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

---

## 5. Loss Formulation & Anti-Forgetting Stack

### 5.1 Training Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(z_{\text{verdict}}, y) + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{KL}}(z_{\text{verdict}}, z_{\text{frozen}})$$

1. **Verdict Cross-Entropy:**
   $$\mathcal{L}_{\text{CE}} = - \log \frac{\exp(z_{\text{target}})}{\sum_{v \in \{\text{Yes}, \text{No}\}} \exp(z_v)}$$
2. **Verdict Self-Distillation (Anti-Forgetting Layer 2):**
   $$\mathcal{L}_{\text{KL}} = D_{\text{KL}}\left(\sigma(z_{\text{frozen}} / T) \parallel \sigma(z_{\text{ft}} / T)\right) \cdot T^2$$
   Computed *only* at the terminal position, with temperature $T=1.0$ and $\lambda_{\text{distill}} = 0.10$.

### 5.2 Scoring for Downstream Stage 3 GBM
During inference over candidate pairs in `candidate_pairs.tsv`:
$$P(\text{Match}) = \text{Softmax}\left(\left[z_{\text{No}}, z_{\text{Yes}}\right]\right)[1] = \frac{\exp(z_{\text{Yes}})}{\exp(z_{\text{Yes}}) + \exp(z_{\text{No}})}$$
This emits a clean scalar probability $P(\text{Match}) \in [0.0, 1.0]$ merged into the Stage 2 feature table.

### 5.3 Confirmation of the France Gate
- **Go/No-Go Gate:** The fine-tuned Qwen3-0.6B matcher must achieve a macro AUCPR on the held-out country within $\le 5\%$ relative of in-domain performance.
- **Fallback Policy:** If the model exhibits distribution collapse on unseen country records, **the feature is completely dropped from Stage 3**.
