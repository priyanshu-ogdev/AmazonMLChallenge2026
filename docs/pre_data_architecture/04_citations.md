# Citations and Verification Log

This log distinguishes claims that were independently re-checked against a primary source during
this documentation pass from claims carried forward from earlier discussion without re-checking.
The goal is to avoid repeating a failure mode that showed up earlier in this design process:
confident-sounding "correction" or "verification" claims that don't survive contact with the
primary source. Nothing here should be taken as settled until it's checked against the actual
competition dataset.

## Independently re-verified in this pass (2026-09-25)

1. **BGE-M3** — MIT license, ~568M parameters, 8,192 token context, dense+sparse+multi-vector
   output, XLM-RoBERTa backbone. Confirmed via Hugging Face model card and independent third-party
   model listings.
2. **Qwen3-Embedding series** — Apache-2.0 license (`license: apache-2.0` on the model card
   YAML front matter), sizes 0.6B/4B/8B, 32,768 token context. MTEB mean-task scores 64.33 (0.6B),
   69.45 (4B), 70.58 (8B) confirmed via the model's own published benchmark table.
3. ***Domain-Specific Text Embedding Models for Entity Resolution* (Sodhana paper)** —
   confirmed to exist as arXiv:2608.16161, with "Sodhana" as a verified author affiliation
   (appears in the byline and in an author's email domain) and a live public code/data repository
   linked directly from the paper's own "Code and Data Availability" section. The 15.25%→92.70%
   (BGE-base-en-v1.5) and 37.85%→83.10% (all-MiniLM-L6-v2) figures are drawn from the paper's own
   Table 4 and Section 4.2 text.
4. **Zeakis et al., PVLDB 2023, "Pre-trained Embeddings for Entity Resolution"** — confirmed to
   exist via the paper's own PDF hosted at vldb.org, benchmarking 12 pretrained language models
   across 17 entity-resolution datasets.

## Source-of-truth correction (this pass) — France / license constraints misattributed

On review against the actual primary competition brief (the first document shared in this
conversation), a mismatch was found: several docs cited "the challenge spec" as the source for (a)
France being a specific held-out test locale, and (b) model license/size constraints
(MIT/Apache-2.0, ≤8B parameters). **Neither claim appears in the primary brief.** The brief's only
regional-variation text is generic: *"Account for locale-specific patterns and formatting
conventions in names and street addresses."* No country is named, and no license or parameter-count
constraint appears anywhere in the primary brief.

Both specifics actually originated from a second document pasted into this conversation — a
transcript purporting to be a prior AI session's embedding-model research. That transcript is
user-supplied content, not the competition's own rules, and its own internal reliability was
already in question earlier in this thread (it contained self-disputing claims about the Sodhana
citation that didn't hold up). Treating "France" and the license/size ceiling as spec-confirmed
facts was an unjustified elevation of that transcript's content to primary-source status.

**Corrected as of this pass:** `01_blocking_stage.md` and `07_prd.md` now attribute the
France/license specifics to this conversation rather than to the competition brief, and flag them
as needing confirmation against the actual competition rules before being treated as binding. This
doesn't mean France or the license ceiling are wrong — they may well be accurate competition
details the user has separate knowledge of — only that this docs set should not represent them as
verified against the one authoritative source available in this conversation when they aren't.

## Round 2 verification (this pass) — resolved

- **Graphlet-AI/eridu** — confirmed real and live on PyPI and Hugging Face. It fine-tunes
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (confirmed: Apache-2.0, 118M
  params, 384-dim output, 50+ languages) on 2M+ labeled name pairs from Open Sanctions Matcher
  data — the 384-dim output confirmed in eridu's own PyPI usage example matches the base model's
  known output size, which is consistent with (though not conclusive proof of) the claimed base
  model. Not independently reproduced: the specific "Prigozhin" before/after cosine-similarity
  numbers were not re-fetched from eridu's own README in this pass.

- **LinkTransformer — license CORRECTED.** Earlier documentation left this unconfirmed. It is now
  confirmed: **LinkTransformer is GPL-3.0**, not a permissive license — verified directly on its
  GitHub repo metadata (`License: gpl-3.0`) and the repo's own LICENSE badge. This matters
  practically: GPL-3.0 is copyleft, which is a materially different obligation than MIT/Apache-2.0
  if this competition's rules require permissive licensing for submitted code, or if any
  LinkTransformer code would need to be incorporated (rather than just used as a reference for its
  documented technique) into a submission. **Action: treat LinkTransformer as reference material
  for its blocking/linking API design only, not as a dependency to ship, unless the competition
  rules are checked and GPL-3.0 is confirmed acceptable.**

- **EnsembleLink (arXiv:2601.21138)** — confirmed real: single-author paper (Noah Dasanaike, Harvard
  Government PhD candidate), first posted January 26, 2026, combining dense retrieval + cross-
  encoder reranking with no training labels required. One earlier characterization should be
  softened: this is one recent single-author paper, not established field consensus — it's
  evidence the dense-retrieval-then-cross-encoder-rerank shape is reasonable, not proof it's
  "current best practice" in any consensus sense.

- **Jina Embeddings v4/v5 license exclusion — confirmed.** `jina-embeddings-v4`'s Hugging Face
  model card confirms `license: cc-by-nc-4.0` directly in the card's metadata. Third-party catalog
  data further confirms v3/v4/v5 models are uniformly CC-BY-NC-4.0 (non-commercial), while only
  Jina's older v1/v2 models are Apache-2.0. The exclusion of v4/v5 from this design stands as
  correct.

- **EmbeddingGemma (Gemma license) exclusion** — not re-fetched in this pass; Google's Gemma
  models are well-established to ship under Google's custom Gemma Terms of Use rather than an OSI
  license, so this exclusion is left as previously stated but technically unconfirmed in this
  specific verification round. Re-check the model card if this model is reconsidered.

## Methodological note on this log
Where a document earlier in this design process claimed to have found an error in a citation
(e.g. disputing the Sodhana paper's existence or its reported figures), that disputing claim was
itself checked against the primary source before being accepted — and did not hold up. The general
practice going forward: a claim that something was "verified" or "corrected" is not itself evidence
until checked against the primary source, regardless of how the claim is framed.

## Known unresolved discrepancy in source data (not this design's error)
Independent searches for BGE-M3's parameter count returned two different figures from different
third-party sources: **~568M** (Hugging Face-derived listings, IONOS, LLM Reference) and **1.13B**
(a Chinese-language model-tracking site, datalearner.com/lmlearning.cn). This design uses **568M**
throughout, since it's the figure repeated across the majority of independent HF-sourced listings
and matches BGE-M3's known XLM-RoBERTa-large-scale backbone size. The 1.13B figure may reflect a
different counting convention (e.g. including both towers of a dual-encoder setup, or an unrelated
mislabel on that site) — flagged here rather than silently resolved, since neither source is BGE-M3's
own primary model card text captured directly in this session. This does not change any design
decision (BGE-M3 is chosen for its dense+sparse+ColBERT output, not its exact parameter count).

## Still entirely open, pending the real dataset
- Actual blocking recall/precision achieved by any of the proposed key strategies
- Actual F_0.5 achieved by any matching-stage configuration
- Whether fine-tuning or reranking earn their complexity on this specific data
- Regional (France) generalization performance, specifically
