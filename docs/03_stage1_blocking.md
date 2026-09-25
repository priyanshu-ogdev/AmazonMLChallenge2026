# Layer 1 — Candidate Generation and Blocking Plan

**Status:** designed, not yet implemented  
**Input:** Layer 0 normalized artifacts  
**Output:** `candidate_pairs.tsv` plus auditable provenance

## Purpose

Layer 1 determines which S1/candidate pairs are eligible for Layer 2. It is a
recall gate: a true link omitted here cannot be recovered by embeddings,
calibration, or thresholding. Precision is intentionally secondary, but the
candidate set must remain bounded enough for feature generation.

The final candidate table is the exact set consumed by Layer 2 and later
written to `candidate_pairs.tsv`. There must be no silent downstream
candidate filtering.

## Candidate-generation contract

For every Source 1 entity, emit one row, including an empty list when no
candidate is found:

```text
source1_entity_id    candidate_entity_ids
```

Internally retain one row per pair with:

```text
source1_entity_id
candidate_entity_id
candidate_source
blocker_provenance
blocker_count
best_blocker_rank
best_blocker_score
country_partition
```

Deduplicate by `(source1_entity_id, candidate_entity_id)` while retaining all
blocker names/ranks. Reject malformed IDs and candidate IDs outside S2/S3.

## Blocking channels

Build the union of independent channels:

1. **Country-partitioned retrieval.** When both records have non-empty,
   canonical country values, search the matching partition. This is supported
   by the train audit's zero sampled cross-country positives, but it is not a
   fixed `{US, India}` allow-list. Missing/ambiguous country records require a
   global fallback search; unknown countries form their own open partition.
2. **Exact normalized keys.** Use normalized full name, name plus structural
   address key, postal/digit-run key, and conservative name/address composite
   keys. Exact keys are high-precision evidence but never the only channel.
3. **Token inverted index.** Index significant normalized name/address tokens,
   with frequency caps so ubiquitous tokens do not explode candidate counts.
4. **Character n-gram retrieval.** Fit a character TF-IDF/nearest-neighbor
   index on S2/S3 normalized text. Build it without labels and query S1 in
   batches. This is the first scalable learned-statistics blocker.
5. **Address structural retrieval.** Retrieve by exact postal/digit-run and
   compatible street-number/trailing-segment keys; do not require addresses
   when either side is missing.
6. **Phonetic retrieval.** Add language-agnostic phonetic keys only after a
   script-safe implementation is selected and its recall is measured. It is
   optional, not a reason to corrupt non-Latin text.
7. **Dense retrieval.** Encode normalized/cased records with the accepted
   BGE-M3 checkpoint and search normalized embeddings by inner product. Use
   sharded, batch-built indexes; never allocate the entire multi-million-row
   embedding matrix by default.
8. **Optional BGE sparse/Qwen retrieval.** Add only when the actual model API
   and memory profile are verified. Qwen remains an auxiliary feature in the
   current design, not an unverified blocker dependency.

## Initial audit configuration

Use these as measured starting points, not permanent truths:

```text
TOP_K_DENSE = 50
TOP_K_SPARSE_OR_CHAR = 50
MAX_CANDIDATES_PER_ENTITY = 100
SIMILARITY_FLOOR = 0.30 for dense-only candidates
```

The union is formed before capping. Candidates introduced by exact,
token, address, or phonetic channels must not be discarded solely because a
dense score is below the dense floor. If a cap is needed, rank by a
deterministic combined priority:

1. exact/composite evidence;
2. number of independent blockers;
3. best channel score;
4. best rank;
5. stable candidate ID tie-break.

Never cap arbitrarily by file order.

## Scale and execution plan

1. Normalize all files once and persist artifacts.
2. Build compact exact/inverted indexes over S2/S3, partitioned by canonical
   country where available.
3. Build character retrieval indexes in bounded shards; persist index metadata
   and row-ID mappings.
4. Encode BGE corpus records in fixed batches, write embeddings/index shards,
   and query S1 batches.
5. Stream blocker hits into a pair-level deduplication store (SQLite/DuckDB or
   sorted shard files), not a Python dictionary containing the entire corpus.
6. Apply union ranking and cap per S1.
7. Materialize `candidate_pairs.tsv` and provenance only after the audit
   counters are complete.

Every index must record normalization version, model revision, partition,
shard, top-K, and row mapping. A stale index must fail validation rather than
silently mix with a new normalized artifact.

## Recall audit gate

Run on train data before any Layer 2/3 tuning:

- pair recall: fraction of all ground-truth links present in candidates;
- entity recall: fraction of S1 entities with every true link represented;
- any-hit rate: fraction with at least one true link represented;
- recall by country, S2/S3 source, match degree, and missing-address slice;
- candidates per S1 percentiles and reduction ratio;
- false candidate volume and cap-hit rate;
- fraction of links recovered by each channel and channel overlaps.

The gate must report both uncapped union recall and final capped recall. If the
cap loses recall, raise the cap before adding model complexity. If a channel
fails, inspect its own contribution before changing Layer 3.

## Validation split and leakage rules

Candidate indexes may use all unlabeled corpus records, but any learned
retrieval statistic used as a Stage 3 feature must be fit within the
appropriate training fold. Ground truth is used only for the recall audit and
training labels, never to add positive candidates or remove negatives from
the production candidate set.

For development, create candidate sets for train and validation entities
using the same code path as test. Do not tune caps on test labels. Keep the
candidate set fixed while comparing Layer 2/3 models.

## Open implementation sequence

1. Add a streaming index/record store and exact-key blocker.
2. Add character TF-IDF retrieval and recall diagnostics.
3. Add address/token/optional phonetic channels.
4. Add dense BGE retrieval with sharded indexes.
5. Union, provenance, deterministic cap, and output writer.
6. Run the train recall gate, then adjust K/floor/cap once.
7. Freeze the candidate-set version before Stage 2 ablations.

Layer 1 is complete only when its final candidate table is reproducible,
recall-audited, bounded, provenance-rich, and exactly the table consumed by
the feature/scoring layers.
