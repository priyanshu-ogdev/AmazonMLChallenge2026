# Embedding Model Comparison — Verified

All figures below were independently checked against primary sources (Hugging Face model cards,
official repos) on 2026-09-25, rather than carried forward from prior summarization.

## Primary candidates

| Model | License (verified) | Params | Context | Notes |
|---|---|---|---|---|
| BGE-M3 (BAAI/bge-m3) | MIT — confirmed on HF model card | ~568M | 8,192 | Single model producing dense, sparse (BM25-like), and ColBERT-style multi-vector output simultaneously. Built on XLM-RoBERTa. MTEB mean-task score 59.56 in the Qwen3 model card's comparison table. |
| Qwen3-Embedding-0.6B | Apache-2.0 — confirmed on HF model card (`license: apache-2.0`) | ~0.6B | 32,768 | Decoder-based (built on Qwen3-0.6B-Base), last-token pooling, instruction-aware. MTEB mean-task score 64.33 (verified on the model's own published benchmark table). |
| Qwen3-Embedding-4B / 8B | Apache-2.0 — confirmed | 4B / 8B | 32,768 | 8B ranks MTEB multilingual leaderboard score 70.58 (as of June 5, 2025, per model card — this is a point-in-time leaderboard snapshot, not a permanent ranking). Compute cost makes these unsuitable as a full-table blocking pass; viable only as a shortlist reranker. |

**Verified head-to-head:** at matched/near-matched size, Qwen3-Embedding-0.6B's published MTEB mean
score (64.33) is higher than BGE-M3's (59.56) on the same comparison table. This is a real quality
gap, not an artifact — but BGE-M3 is not "worse," it is trading some pure dense-retrieval score for
sparse+multi-vector output that a pure dense model doesn't provide. For blocking, that free sparse
signal has practical value (no separate BM25 pipeline needed); for a single similarity feature into
a matcher, the stronger pure dense score is the more direct benefit.

## Other options considered (license-verified where checked)
| Model | License | Notes |
|---|---|---|
| multilingual-e5-large-instruct | MIT | 560M, 512 context — short context is a non-issue for name/address strings |
| gte-multilingual-base | Apache-2.0 | ~305M, cheap, 8192 context via RoPE |
| snowflake-arctic-embed-l/m-v2.0 | Apache-2.0 | Adds Matryoshka truncation for cheaper storage |
| BGE-base-en-v1.5 | MIT | English-only — a liability if France (or any non-English locale) is genuinely in the test set, an unconfirmed detail per `04_citations.md`; relevant regardless as the base model in the Sodhana fine-tuning precedent below |

**Ruled out on license grounds:**
Jina Embeddings v3/v4/v5 — **confirmed** CC-BY-NC-4.0 (non-commercial) directly on the
`jina-embeddings-v4` Hugging Face model card metadata; only Jina's older v1/v2 models are
Apache-2.0. EmbeddingGemma — Gemma custom license, not OSI MIT/Apache; this specific claim was not
re-fetched against the model card in verification (see `04_citations.md`), so re-check before use
if this model is reconsidered.

## Fine-tuning precedent (verified)

**Paper:** *Domain-Specific Text Embedding Models for Entity Resolution* — Narayana, Srivardhani,
Konda (Sodhana). arXiv:2608.16161. **Confirmed real** by direct fetch of the paper text: the
"Sodhana" affiliation appears in the byline, and code/data are published at
`github.com/khajeshnarayana/Domain-Specific-Text-Embedding-Models-for-Information-Retrieval` (this
repo URL was returned directly from the paper's own "Code and Data Availability" section during
verification, confirming the paper is real and the artifact link is live).

Their setup is close to this competition's shape: synthetic business records
(`"{Company Name} is a firm located at {Address}"`), with hard negatives built specifically as
same-name-different-address pairs — directly analogous to this pipeline's false-merge risk.
They fine-tuned `all-MiniLM-L6-v2` and `BAAI/bge-base-en-v1.5` with triplet loss and measured
separation between true matches and near-duplicate distractors.

**Reported results (as stated in the paper's own Table 4 and Section 4.2 prose):**

| Model | Pretrained (margin 0.30) | Fine-tuned (margin 0.30) |
|---|---|---|
| BGE-base-en-v1.5 | 15.25% | 92.70% |
| all-MiniLM-L6-v2 | 37.85% | 83.10% |

Takeaway used in this design: the fine-tuning *step* produced the large gain here, not base-model
size — supports prioritizing a triplet fine-tune on this competition's own `train_ground_truth.tsv`
(same-name/different-address as the hard negative) as a high-leverage, cheap addition, over simply
picking a bigger pretrained base model.

**Related published work (context, not re-verified line-by-line in this pass):**
- Zeakis et al., *Pre-trained Embeddings for Entity Resolution: An Experimental Analysis*, PVLDB
  vol. 16 (2023) — benchmarked 12 pretrained LMs across 17 ER datasets; independently confirmed to
  exist via VLDB's own published PDF (`vldb.org/pvldb/vol16/p2225-skoutas.pdf`).
- Ditto (Li et al., VLDB 2020) — field-tagged cross-encoder architecture, referenced here only as
  an architectural pattern for the optional reranking step in Stage 2, not benchmarked directly in
  this design.

## Recommendation used in this pipeline
- **Blocking:** BGE-M3 dense+sparse union — avoids running a separate BM25 index.
- **Matching feature:** Qwen3-Embedding-0.6B cosine similarity, generated out-of-fold.
- **Highest-leverage optional step:** triplet fine-tune on this competition's own ground truth,
  following the Sodhana same-name/different-address hard-negative recipe, starting from BGE-M3 or
  BGE-base-en-v1.5 depending on compute budget.
- **Not recommended for full-table blocking:** Qwen3-Embedding-4B/8B — reserve for reranking an
  already-shortlisted candidate set only, if used at all.

None of the above has been benchmarked on this competition's actual data or its F_0.5-weighted
scoring — including whether France is genuinely part of the test set, which is unconfirmed against
the primary brief (see `04_citations.md`). This is a literature-grounded starting point, to be confirmed or revised
by an incremental build-and-measure pass once the real TSVs are available.
