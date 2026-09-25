# Stage 1 — Blocking (Candidate Generation)

## Objective
Maximize recall of true S1↔S2/S3 matches while keeping the candidate set small enough for the
matching stage to score. Recall lost here is unrecoverable — this stage sets the theoretical
ceiling on the whole pipeline's score.

## Design

### 1. Normalization (applied identically to name and address fields, all sources)
- Lowercase, strip punctuation
- Expand common abbreviations (St→Street, Rd→Road, Ave→Avenue, Inc/LLC/Ltd/Corp suffix variants)
- Strip or standardize legal-entity suffixes for a *separate* normalized-name field (kept alongside
  the raw name, since the suffix itself can be a useful matching signal, not just noise)
- Tokenize on whitespace after normalization for token-overlap keys

### 2. Multiple blocking keys, unioned (not intersected)
Because the failure mode here is silent recall loss, use several independent block keys and take
the union of everything they group together, rather than a single narrow key:
- Token-based key on normalized business name (sorted token n-grams or a fixed-length prefix)
- Token-based key on normalized address (street number + street name tokens)
- Phonetic key on business name (e.g. Soundex/Metaphone) to catch spelling variants
- Embedding-based nearest-neighbor blocking: dense (and optionally sparse) vector similarity above
  a permissive threshold, using an ANN index (e.g. FAISS/HNSW) — this is where BGE-M3's combined
  dense+sparse output is used as a single-pass substitute for a separate BM25 index (see
  `03_embedding_models.md`)

### 3. Regional/locale handling
The primary competition brief calls this out only generically: *"Regional Variations: Account for
locale-specific patterns and formatting conventions in names and street addresses."* It does not
name a specific held-out country. A separate part of this design conversation (not the primary
brief) referred to France as a specific held-out test locale, along with model-license constraints
(MIT/Apache-2.0, ≤8B) — that detail came from a pasted transcript of an unverified prior session,
not from the competition's own rules text, and should be **confirmed against the actual competition
rules/data before being treated as a hard requirement**. Until confirmed, normalization rules
should be designed for locale-robustness generally, with France treated as one plausible example
locale worth testing rather than a spec-confirmed one. Normalization rules should be checked
against sample records from each represented locale rather than assumed to generalize from
English-language patterns.
rather than assumed to generalize from English-language patterns.

### 4. Output
`candidate_pairs.tsv` — every (S1_id, S2_or_S3_id) pair that any blocking key grouped together.
This file is required in the final submission package specifically so evaluators can audit
blocking recall independent of final match precision.

## Open / data-dependent decisions
- Exact block-key parameters (n-gram length, ANN similarity threshold, phonetic algorithm choice)
- Whether embedding-based blocking is necessary at all, or token+phonetic keys alone hit acceptable
  recall more cheaply — this should be measured, not assumed
- Candidate-set size budget per S1 entity (affects Stage 2 compute)

All of the above require the real ground-truth file to tune; this document specifies the mechanism,
not the tuned values.
