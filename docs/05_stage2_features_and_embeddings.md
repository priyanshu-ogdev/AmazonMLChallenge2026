# Stage 2 — Complete Representation and Pair-Feature Engineering
## Amazon ML Challenge 2026 — Business Entity Resolution

---

## 1. Executive Summary & Design Scope

**Stage 2** takes candidate pairs $(e_{S_1}, e_{S_2/S_3})$ produced by Stage 1 blocking and extracts a rich, multi-dimensional feature representation for supervised scoring.

### Why Multi-Dimensional Representations are Essential
A single cosine similarity or token overlap threshold collapses multidimensional evidence into a 1-dimensional decision boundary. Such a boundary structurally cannot separate:
- **Same-Name Different-Location Distractors:** (e.g., "McDonald's" in Springfield, IL vs "McDonald's" in Dallas, TX — high name similarity, low address similarity $\rightarrow$ FALSE MATCH).
- **Abbreviated True Matches:** (e.g., "IBM Corp" at 1 New Orchard Rd vs "International Business Machines" at 1 New Orchard Road — moderate name similarity, high address similarity $\rightarrow$ TRUE MATCH).

A supervised gradient-boosted tree (Stage 3) receiving orthogonal feature families can learn nonlinear, asymmetric decision surfaces capable of distinguishing these error modes.

```mermaid
flowchart TD
    CP["Candidate Pairs from Stage 1 (candidate_pairs.tsv)"] --> S2ai["Stage 2a-i: Fine-Tuned BGE-M3 rsLoRA Bi-Encoder"]
    CP --> S2aii["Stage 2a-ii: Auxiliary Off-the-Shelf Qwen3-Embedding-0.6B"]
    CP --> S2c["Stage 2c: 32 Deterministic Lexical, Phonetic, & Structural Features"]
    CP -.-> S2b["(Stretch) Stage 2b: Qwen3-0.6B Causal Generative Matcher"]
    
    S2ai --> M["Grouped Feature Matrix (Grouped by S1 entity_id)"]
    S2aii --> M
    S2c --> M
    S2b --> M
    M --> S3["Stage 3 Supervised XGBoost Meta-Learner"]
```

---

## 2. Four Feature Families

### 2.1 Stage 2a-i: Primary Fine-Tuned BGE-M3 Dense Feature
- **Model:** `BAAI/bge-m3` (568M params, XLM-RoBERTa backbone, MIT license) fine-tuned with rank-64 rsLoRA on competition pairs using `CachedMultipleNegativesRankingLoss` and a 4-layer anti-forgetting stack.
- **Computation:**
  $$\text{bge\_cosine}(e_1, e_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|_2 \|\mathbf{v}_2\|_2} \in [-1.0, 1.0]$$
  where $\mathbf{v} \in \mathbb{R}^{1024}$ is the dense representation emitted from the pooled encoder.
- **Specification:** [`docs/06_stage2a_bge_m3_training_spec.md`](06_stage2a_bge_m3_training_spec.md).

### 2.2 Stage 2a-ii: Auxiliary Qwen3-Embedding-0.6B Feature
- **Model:** `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, ~596M params).
- **Role:** Purely an inference-only, complementary dense feature. Included only if held-out cross-validation proves it adds orthogonal signal over BGE-M3.
- **Prompt Symmetry Rule:** Asymmetric query-vs-passage prefixes (e.g. `"Instruct: ...\nQuery: "`) must **never** be used. Business entity resolution is symmetric. An identical prompt or no prompt is applied to both records.

### 2.3 Stage 2b: Cross-Record Generative Matcher (Stretch Goal)
- **Model:** `Qwen/Qwen3-0.6B` causal language model fine-tuned via LoRA.
- **Role:** Slices the hidden state at terminal position $T_{\text{verdict}}$ to compute 2-way softmax match probability:
  $$P(\text{Match}) = \frac{\exp(z_{\text{Yes}})}{\exp(z_{\text{Yes}}) + \exp(z_{\text{No}})}$$
- **Theoretical Basis:** Grounded in Zhang et al. (arXiv:2607.24688). Sliced verdict logits reduce memory from $\approx 6.5\text{ GB}$ to $\approx 29\text{ MB}$.
- **Specification:** [`docs/07_stage2b_qwen3_generative_matcher_spec.md`](07_stage2b_qwen3_generative_matcher_spec.md).

### 2.4 Stage 2c: 32 Deterministic Lexical & Structural Features
Implemented in `src/pair_features.py`. Operates label-free on normalized text strings:

| Category | Feature Name | Description | Value Range |
|---|---|---|---|
| **Name Lexical** | `name_exact` | Binary indicator: `norm_name_1 == norm_name_2` | $\{0.0, 1.0\}$ |
| | `name_jaccard` | Word token intersection over union | $[0.0, 1.0]$ |
| | `name_overlap` | $\frac{\|T_1 \cap T_2\|}{\min(\|T_1\|, \|T_2\|)}$ | $[0.0, 1.0]$ |
| | `name_edit_similarity` | Character edit similarity ratio: $1 - \frac{\text{dist}}{\max(L_1, L_2)}$ | $[0.0, 1.0]$ |
| | `name_char_trigram_jaccard` | Character 3-gram Jaccard coefficient | $[0.0, 1.0]$ |
| | `name_length_abs_diff` | Absolute difference in character length | $\mathbb{Z}_{\ge 0}$ |
| **Address Lexical** | `address_exact` | Binary indicator: `norm_addr_1 == norm_addr_2` | $\{0.0, 1.0\}$ |
| | `address_jaccard` | Address word token intersection over union | $[0.0, 1.0]$ |
| | `address_overlap` | Address token overlap coefficient | $[0.0, 1.0]$ |
| | `address_edit_similarity` | Character edit similarity on normalized addresses | $[0.0, 1.0]$ |
| | `address_char_trigram_jaccard` | Character 3-gram Jaccard on addresses | $[0.0, 1.0]$ |
| | `address_length_abs_diff` | Absolute difference in character length | $\mathbb{Z}_{\ge 0}$ |
| **Numeric & Postal** | `postal_equal` | Exact postal code match (ZIP/PIN/Code Postal) | $\{0.0, 1.0\}$ |
| | `postal_missing_either` | Indicator if postal code is missing on either record | $\{0.0, 1.0\}$ |
| | `name_number_overlap` | Number token overlap in business name | $[0.0, 1.0]$ |
| | `address_number_overlap` | Number token overlap in address | $[0.0, 1.0]$ |
| **Missingness & Indicators** | `left_country_missing` | Indicator if S1 country was missing | $\{0.0, 1.0\}$ |
| | `right_country_missing` | Indicator if candidate country was missing | $\{0.0, 1.0\}$ |
| | `name_both_missing` | Indicator if name was empty on both records | $\{0.0, 1.0\}$ |
| | `address_both_missing` | Indicator if address was empty on both records | $\{0.0, 1.0\}$ |
| | `bge_cosine_missing` | Indicator if either normalized text was empty for BGE | $\{0.0, 1.0\}$ |
| | `qwen_cosine_missing` | Indicator if either normalized text was empty for Qwen | $\{0.0, 1.0\}$ |
| | `qwen_matcher_prob_missing` | Indicator if either normalized text was empty for Qwen Matcher | $\{0.0, 1.0\}$ |
| **Contradictions** | `same_name_different_address` | Exact name match but different addresses | $\{0.0, 1.0\}$ |
| | `same_address_different_name` | Exact address match but different names | $\{0.0, 1.0\}$ |
| **Country & Source** | `country_equal` | Canonical country match flag (0=mismatch, 1=match, -1=missing) | $\{-1.0, 0.0, 1.0\}$ |
| | `country_equal_missing` | Indicator if country comparison was missing | $\{0.0, 1.0\}$ |
| | `source_is_s3` | Indicator if candidate is from Source 3 (vs Source 2) | $\{0.0, 1.0\}$ |
| **Blocker Graph / Context** | `candidate_rank` | Retrieval rank of candidate from Stage 1 (999.0 if missing) | $[1.0, 999.0]$ |
| | `candidate_rank_missing` | Indicator if candidate rank was missing | $\{0.0, 1.0\}$ |
| | `rank_margin_from_best` | Rank difference from rank 1 (999.0 if missing) | $\ge 0.0$ |
| | `best_blocker_score` | Maximum similarity score from Stage 1 retrieval (-1.0 if missing) | $[-1.0, 1.0]$ |
| | `best_blocker_score_missing` | Indicator if blocker score was missing | $\{0.0, 1.0\}$ |
| | `best_blocker_score_diff` | Margin to best blocker score (0.0 if missing) | $\ge 0.0$ |
| | `best_blocker_score_diff_missing`| Indicator if score difference was missing | $\{0.0, 1.0\}$ |
| | `blocker_count` | Number of Stage 1 channels retrieving this pair | $\{1, 2, 3, 4\}$ |
| | `candidate_count_for_s1` | Total candidate count retrieved for this S1 entity | Integer $\ge 1$ |
| | `has_blocker_provenance` | Indicator if channel provenance metadata exists | $\{0.0, 1.0\}$ |

---

## 3. Strict Feature Extraction Invariants

1. **One-to-One Row Alignment:** Every pair in `output/candidate_pairs.tsv` must have exactly one corresponding feature row in the Stage 2 feature matrix.
2. **Zero Ground-Truth Leakage:** No Stage 2 feature may inspect ground-truth labels, validation fold splits, or future test labels.
3. **Explicit Missing Semantics:** Missing numerical features are represented by explicit sentinel values (`999.0` for rank, `0.0` for score difference, `-1.0` for missing country comparison) coupled with binary indicator columns (`candidate_rank_missing = 1.0`, `country_equal_missing = 1.0`).
4. **No Raw Categorical Country IDs:** Country identity is represented solely through the symmetric boolean match flag `country_equal`. Raw country strings are dropped as `DROP_CATEGORICAL` and never passed to the GBM.

---

## 4. Feature Extraction Execution & Ablation Protocol

### CLI Execution:
```powershell
python -m src.pair_features `
    --source1 ../../dataset/train/train_source1.tsv `
    --source2 ../../dataset/train/train_source2.tsv `
    --source3 ../../dataset/train/train_source3.tsv `
    --candidate-file ../../output/candidate_pairs.tsv `
    --output-file ../../output/pair_features.tsv
```

### Staged Ablation Schedule:
Every feature family must demonstrate a positive delta on macro $F_{0.5}$ on held-out cross-validation:
1. **Ablation 1 (Lexical Baseline):** Stage 2c deterministic features only.
2. **Ablation 2 (Dense Baseline):** Stage 2c + Pre-trained (frozen) BGE-M3 bi-encoder cosine.
3. **Ablation 3 (Adapted Bi-Encoder):** Stage 2c + Fine-Tuned BGE-M3 rsLoRA cosine. (Must beat Ablation 2).
4. **Ablation 4 (Dual Bi-Encoder):** Stage 2c + Fine-Tuned BGE-M3 + Qwen3-Embedding-0.6B cosine. (Kept only if $\Delta F_{0.5} > +0.005$).
5. **Ablation 5 (Full Pipeline with Stretch):** Stage 2c + Fine-Tuned BGE-M3 + Stage 2b Generative Matcher $P(\text{Match})$.
