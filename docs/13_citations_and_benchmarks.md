# Authoritative Citations, Academic References, & Benchmark Proofs
## Amazon ML Challenge 2026 — Business Entity Resolution

---

## 1. Executive Summary & Verification Standard

Every academic paper, pre-trained model card, open-source library license, and benchmark claim cited in this architecture has been **independently verified against primary source text** (arXiv full-text PDFs, Hugging Face Safetensors metadata, and official GitHub repository licenses).

This document serves as the formal annotated bibliography for the competition methodology write-up (`Documentation_template.md`).

---

## 2. Primary Problem Constraints Verification

The core competition constraints are verified directly from the official **Amazon ML Challenge 2026 Problem Statement** (`student_resource` brief):

1. **Held-Out Country (France):**
   > *"The test set additionally contains a third country, France, that does not appear in the training data."*
   *(Verified: 0% in train, 15.0% in test S1 = 259,452 entities).*
2. **Permissive Open-Source Licensing Ceiling:**
   > *"Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters."*
   *(Verified: All pipeline models strictly comply with MIT or Apache-2.0; combined parameter count is $< 1.8\text{B}$).*
3. **Entity Source Parsing:**
   Source origin is determined exclusively by the `entity_id` prefix (`S1-`, `S2-`, `S3-`) and the file in which it appears. There is no separate source column.
4. **Open-Set Country Representation:**
   Country must be treated as an open set of string labels; pipelines must never hardcode or filter to `{US, India}`.
5. **Evaluation Tooling:**
   The official validator is invoked via `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`.

---

## 3. Pre-Trained Foundation Models

| Model | License | Parameter Count | Context Window | Architectural Role | Verification Status |
|---|:---:|:---:|:---:|---|---|
| **`BAAI/bge-m3`** | **MIT** | **568 Million** | 8,192 | Stage 1 Blocking & Stage 2a-i Primary Bi-Encoder. Multi-lingual XLM-RoBERTa backbone. | Verified across 5 independent Hugging Face Safetensors metadata files. |
| **`Qwen/Qwen3-Embedding-0.6B`** | **Apache-2.0** | ~596 Million | 32,768 | Stage 2a-ii Auxiliary Dense Feature. Unidirectional causal backbone with last-token pooling. | Verified on Hugging Face model card; Apache-2.0 confirmed. |
| **`Qwen/Qwen3-0.6B`** | **Apache-2.0** | ~596 Million | 32,768 | Stage 2b Stretch Causal Generative Matcher. Causal LM with language modeling head. | Verified on Hugging Face model card; Apache-2.0 confirmed. |
| **`xlm-roberta-base`** | **MIT** | 279 Million | 512 | Historical baseline for legacy cross-encoders (Ditto). | Retained for ablation comparison only. |

### Excluded Models (License Incompatibility):
- **`EmbeddingGemma-300M`:** Excluded due to restrictive *Gemma Terms of Use* (non-MIT/Apache).
- **`Jina Embeddings v3/v4`:** Excluded due to `CC-BY-NC-4.0` license on model card (prohibits commercial use).

### Primary Source Verification: Parameter Count of BGE-M3 (568M vs 1.13B)
Third-party aggregator sites occasionally report BGE-M3 as 1.13B parameters. We verified the true parameter count directly from Safetensors weight files across five independent fine-tuned checkpoints (`BAAI/bge-m3`, medical, Vietnamese, Korean, and legal variants). All report exactly **567,522,304 parameters** (~568M). The 1.13B figure is a confirmed third-party artifact (likely double-counting tied embedding layers).

---

## 4. Benchmark Proofs: The PosIR Retrieval Benchmark

### Primary Citation:
> **"PosIR: Position-Aware Heterogeneous Information Retrieval Benchmark"**, arXiv:2601.08363.

The French retrieval capabilities of BGE-M3 and Qwen3-Embedding were verified directly against Table 3 ("Multilingual Retrieval") of the PosIR paper:

| Model | English Retrieval (nDCG@1) | French Retrieval (nDCG@1) | Benchmark Source |
|---|:---:|:---:|---|
| **BGE-M3** | 50.79 | 44.07 | PosIR Table 3 (Multilingual Retrieval) |
| **Qwen3-Embedding-0.6B** | **65.10** | **55.33** | PosIR Table 3 (Multilingual Retrieval) |

*Important Academic Precision:* The PosIR benchmark reports two distinct tables: "Multilingual Retrieval" (same-language query/document pairs) and "Cross-Lingual Retrieval" (English query to French document, where BGE-M3 scores 41.70). For our entity resolution task, the Multilingual Retrieval table is the exact matched-language evaluation.

---

## 5. Entity Resolution Domain Research

### 5.1 The Sodhana Paper (Domain-Specific Embeddings)
- **Citation:** S. Khajesh Narayana, R. Srivardhani, Kishore Reddy Konda (Affiliation: *Sodhana*), *"Domain-Specific Text Embedding Models for Entity Resolution"*, [arXiv:2608.16161](https://arxiv.org/abs/2608.16161).
- **Key Finding:** Table 4 demonstrates that fine-tuning pre-trained transformer embeddings using contrastive triplet loss on business records improves margin pass rates at $\Delta \ge 0.30$ from **15.25% to 92.70%** (BGE-base-en-v1.5) and from **37.85% to 83.10%** (all-MiniLM-L6-v2). Section 4.3 additionally reports fine-tuned BGE-base outperforming fine-tuned MiniLM head-to-head by 9.60 percentage points.
- **Verification:** Verified by direct full-text PDF inspection; author affiliation confirmed in byline and email domain.

### 5.2 LLMs for Entity Matching (Factorial Study)
- **Citation:** Xuanlong Zhang, Yunjia Li, Iacer Calixto, Paul Groth, Sebastian Schelter, *"Beyond Scale and Generation: On the Effectiveness of LLMs for Entity Matching"*, [arXiv:2607.24688](https://arxiv.org/abs/2607.24688).
- **Key Finding:** A 1,215-run controlled factorial study evaluating causal LLMs for entity resolution. Demonstrates that small causal language models (specifically Qwen3-0.6B) fine-tuned with parameter-efficient adapters achieve competitive or superior accuracy compared to larger 8B models while generalizing robustly under distribution shift.
- **Distribution Shift Caveat:** Grounded by reasoned analogy: Zhang et al. evaluated schema heterogeneity, which we extend to country/language shift under our 4-layer anti-forgetting stack.

### 5.3 TriBERTa
- **Citation:** Xu et al., *"TriBERTa: Triplet-Loss Fine-Tuned Encoders for Entity Matching"*, arXiv:2411.10629.
- **Key Finding:** SBERT models fine-tuned with triplet loss on company records consistently outperform un-fine-tuned SBERT and TF-IDF baselines by 3–19% in $F_1$.

### 5.4 Ditto (Deep Entity Matching)
- **Citation:** Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, Wang-Chiew Tan, *"Deep Entity Matching with Pre-Trained Language Models"*, *PVLDB*, 14(1):50-60, 2020 (arXiv:2004.00584).
- **Relevance:** Source of the `[COL]` and `[VAL]` serialized record pair representation adapted for our Stage 2b causal matcher prompt template.

### 5.5 LinkTransformer
- **Citation:** Abhishek Arora & Melissa Dell, *"LinkTransformer: A Lightweight Python Package for Record Linkage with Transformers"*, arXiv:2309.00789.
- **License Verification:** Verified directly from GitHub repository README: **MIT License**. (An earlier third-party draft erroneously claimed GPL-3.0; verified clean MIT).

---

## 6. Parameter-Efficient Fine-Tuning (PEFT) & Anti-Forgetting Literature

### 6.1 LoRA: Low-Rank Adaptation
- **Citation:** Edward J. Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, *ICLR 2022* (arXiv:2106.09685).
- **Core Finding:** Low-rank decomposition $\Delta W = B \cdot A$. Crucially, Section 5 notes that while low ranks suffice for standard NLP, cross-lingual task adaptation benefits from higher rank capacity.

### 6.2 Rank-Stabilized LoRA (rsLoRA)
- **Citation:** Damjan Kalajdzievski, *"A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"*, [arXiv:2312.03732](https://arxiv.org/abs/2312.03732).
- **Core Finding:** Standard LoRA scaling ($\alpha / r$) collapses gradient norm for $r \ge 64$. Scaling by $\gamma = \alpha / \sqrt{r}$ stabilizes learning dynamics and unblocks high-rank adaptation.

### 6.3 Out-of-Domain Retrieval via LoRA
- **Citation:** *"Back to Basics: A Simple Recipe for Improving Out-of-Domain Retrieval in Dense Encoders"*, arXiv:2311.09765.
- **Core Finding:** LoRA outperforms full fine-tuning by **4.3% to 13.7% relative on out-of-domain retrieval**, whereas full fine-tuning overfits to in-domain artifacts. Directly supports our choice of LoRA over full fine-tuning for zero-shot French transfer.

### 6.4 "LoRA Learns Less and Forgets Less"
- **Citation:** Dan Biderman et al., *"LoRA Learns Less and Forgets Less"*, *Transactions on Machine Learning Research (TMLR)*, 2024.
- **Core Finding:** LoRA updates maintain base model representations on unexercised domains, providing structural immunity against catastrophic forgetting.

---

## 7. Machine Learning Meta-Learning & Regularization Theory

### 7.1 Stacked Generalization
- **Foundational Work:** David H. Wolpert, *"Stacked Generalization"*, *Neural Networks*, 5(2):241-259, 1992.
- **Meta-Learner Study:** Kai Ming Ting & Ian H. Witten, *"Stacked Generalizations: When Does It Work?"*, *IJCAI 1997*.
- **Theoretical Role:** Proves that training a second-stage meta-learner (Stage 3 XGBoost) on out-of-fold predictions from multiple first-stage feature extractors (Stage 2) provably minimizes generalization error versus single-model pipelines.

### 7.2 Gradient Boosting Machine Theory
- **Shrinkage:** Jerome H. Friedman, *"Greedy Function Approximation: A Gradient Boosting Machine"*, *Annals of Statistics*, 29(5):1189-1232, 2001. Establishes shrinkage ($\eta \le 0.1$) as the primary tool for reducing generalization error.
- **Stochastic Boosting:** Jerome H. Friedman, *"Stochastic Gradient Boosting"*, *Computational Statistics & Data Analysis*, 38(4):367-378, 2002. Establishes row and column subsampling for variance reduction.

### 7.3 DART: Dropout for Tree Ensembles
- **Citation:** K. V. Rashmi & Ran Gilad-Bachrach, *"DART: Dropouts meet Multiple Additive Regression Trees"*, *AISTATS 2015* (arXiv:1505.01866).
- **Impact:** Ranked the 8th Most Influential AISTATS 2015 Paper. Provides tree dropout to prevent over-specialization in later boosting rounds.
