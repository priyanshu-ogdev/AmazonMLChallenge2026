# Layer 0 — Ingestion, Normalization, and Preprocessing Contract

**Status:** implemented baseline; validation and artifact auditing remain
before large-scale candidate generation  
**Canonical implementation:** `src/normalize.py` and
`src/data_builder.py` (streaming TSV normalizer)

`src.data_builder.normalize_tsv_file` is the production entry point. The
standalone script documents the policy but must not become a second divergent
output schema.

## Purpose

Layer 0 converts raw challenge TSV records into deterministic, reusable
records. It does not create pairs, use ground truth, train a model, or decide
whether two entities match. Its output is the only text representation
allowed into Stage 1 and the downstream feature builders.

The design is driven by the dataset audit:

- raw files are multi-million-row and cannot be loaded into one in-memory
  dataframe;
- France appears only in test;
- names and addresses contain accented/non-ASCII text;
- S2/S3 addresses are missing for roughly 3% of records;
- source is encoded by `S1-`, `S2-`, and `S3-` IDs.

## Non-negotiable invariants

1. **Open-set country handling.** Country is normalized for equality and
   partitioning, but never filtered to `{US, India}`, one-hot encoded as a
   fixed vocabulary, or used to select country-specific code paths. Unknown
   labels pass through.
2. **Unicode preservation.** Use NFKC, retain letters/marks/digits and
   accented characters, and never ASCII-fold the canonical text.
3. **Raw data remains auditable.** Preserve raw name, raw address, and raw
   country columns alongside normalized values; normalization is not
   destructive replacement.
4. **Missingness is explicit.** Empty address is represented by an indicator
   and an empty normalized value, never by a fabricated address or a generic
   country-dependent sentinel.
5. **ID/source validity is strict.** Reject duplicate IDs, malformed prefixes,
   missing required columns, and inconsistent source/file assignments.
6. **Determinism.** Same input bytes and code version produce the same output
   values and row count. No random sampling or learned vocabulary belongs in
   Layer 0.
7. **Streaming is mandatory.** Process large source files in bounded chunks or
   rows; write normalized output incrementally.

## Record representation

The normalized artifact should retain:

```text
entity_id
source
country_raw
country_canonical
raw_name
raw_address
norm_name
norm_address
encoder_text
is_address_missing
postal_code
street_number
trailing_segment
digit_runs
```

`norm_name` and `norm_address` serve lexical indexes and deterministic pair
features. `encoder_text` is the canonical combined transformer input.
`country_canonical` is an equality key, not a closed category.
`is_address_missing` is retained as an explicit missingness feature.
`postal_code`, `street_number`, `trailing_segment`, and `digit_runs` are
structural evidence for blocking and pair features. Raw fields remain in the
artifact so the transformation is auditable.

## Normalization order and rationale

1. Validate required fields and derive source from the ID prefix.
2. Preserve raw values and normalize null-like values to empty strings.
3. Apply HTML unescaping, NFKC, control-character cleanup, typographic
   apostrophe normalization, lowercasing, and whitespace collapse.
4. Strip punctuation by Unicode category while retaining meaningful
   separators; do not use a naïve ASCII or `\w`-only filter that damages
   combining marks and non-Latin scripts.
5. Apply legal suffix equivalence only in the trailing name context and
   address abbreviations only in address context. Avoid expanding tokens such
   as `co`, `inc`, or direction letters indiscriminately inside addresses.
6. Strip bounded landmark phrases and retain the removed phrase separately.
7. Extract generic structural fields: leading street number, digit runs, and
   trailing comma-separated segment. These are candidate evidence, not a
   country parser.
8. Canonicalize known country aliases while passing unknown labels through.

The output has multiple views because no single normalization is safe for
every consumer: aggressive canonicalization helps blocking, while raw/cased
text protects dense multilingual representations.

## Preprocessing execution

Use the streaming Stage 0 command from `src.data_builder`:

```powershell
python -m src.data_builder --mode stage0_normalize `
    --data_dir ../../dataset `
    --output_dir ../../dataset/stage0_normalized `
    --splits train test
```

Before accepting the artifacts, record per file:

- input/output row counts;
- duplicate and malformed ID counts;
- country-raw and country-canonical counts;
- missing name/address counts;
- non-ASCII counts;
- normalization expansion/empty-output counts;
- code version, input paths, and timestamp.

The output should be written to a new directory and replaced atomically only
after row-count and schema checks succeed. Never overwrite raw challenge TSVs.

## Leakage boundary

Layer 0 may inspect only each record's own fields. It must not inspect
ground-truth links, candidate labels, validation folds, target statistics, or
the other side of a pair. Alias tables are static code configuration; TF-IDF,
IDF weights, stopword lists learned from corpus frequencies, and dense indexes
belong to Layer 1 and must be fit/build-scoped there.

## Verification gate

Layer 0 is ready for Layer 1 only when:

- all eight source files pass schema/ID checks;
- normalized row counts equal raw row counts;
- raw and normalized artifacts can be joined one-to-one by `entity_id`;
- France and accented/non-Latin samples retain meaningful characters;
- empty S2/S3 addresses retain `is_missing_address=1`;
- country aliases compare equal while unseen labels remain visible;
- rerunning on the same input produces byte-equivalent normalized fields;
- no raw text is silently replaced or lost.

## Known decisions

- NFKC, not NFKD, is the canonical Unicode form.
- No external geocoding, business lookup, translation, or country database is
  allowed.
- Postal extraction is generic digit-shape extraction; exact country semantics
  are not assumed in Layer 0.
- Normalization does not enforce one-to-one target assignment. That is a
  downstream decision-policy concern and must be validated rather than assumed
  from a finite ground-truth audit.
