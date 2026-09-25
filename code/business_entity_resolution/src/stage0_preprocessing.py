"""
Stage 0 — Ingestion & Normalization
====================================
Implements architecture.md's Stage 0 spec exactly:

  - Country is an OPEN set of string labels. Never hard-coded, filtered, or
    one-hot to {US, India}. France (or any future country) must work with
    zero special-case code.
  - Source is derived from entity_id prefix (S1-/S2-/S3-) + which file the
    record came from. There is no separate source column.
  - Lowercasing, punctuation stripping, whitespace collapse.
  - Legal-suffix / address-abbreviation canonicalization runs BIDIRECTIONALLY
    (Corp<->Corporation, Rd<->Road, &<->and) rather than "always expand to
    the long form" — this matters because which form the OTHER side uses is
    unknown, so we generate a canonical *equivalence key*, not a guess at
    which form is "correct."
  - Landmark phrases ("Near SBI ATM", "Opp. XYZ") are stripped from the
    similarity-bearing text but flagged as a separate weak auxiliary signal,
    not deleted from the record.
  - Structural sub-fields (street number, postal/PIN/ZIP-like code, city/
    state-like tail token) are extracted with DATA-DRIVEN heuristics
    (regex over generic token shapes: digit runs, trailing comma-separated
    segments) — never per-country branching.
  - Missing business_address (~3.3% of S2/S3 per your EDA) gets an explicit
    is_missing_address flag, never silent empty-string treatment.

This module does NOT do blocking, feature scoring, or matching — those are
Stage 1-3 per architecture.md and belong in separate modules that consume
this one's output.
"""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# 1. Source / entity-id handling — no separate source column, derive from ID
# ---------------------------------------------------------------------------

_SOURCE_PREFIX_RE = re.compile(r"^(S[123])-")


def source_from_entity_id(entity_id: str) -> str:
    """S1-925783039 -> 'S1'. Raises if the ID doesn't match the expected shape
    (fail loudly here rather than silently mis-sourcing a record)."""
    m = _SOURCE_PREFIX_RE.match(entity_id.strip())
    if not m:
        raise ValueError(f"entity_id does not match S[123]-... shape: {entity_id!r}")
    return m.group(1)


# ---------------------------------------------------------------------------
# 2b. Country canonicalization — NOT a fixed vocabulary, NOT a filter.
#
# Country stays a fully open string set end-to-end (per architecture.md and
# the official problem statement: never hard-code, filter, or one-hot to
# {US, India}). Removing the column, or restricting it to known values,
# would break two things that actually depend on it working correctly:
#
#   1. Stage 1 blocking partitions candidates by country equality (0 cross-
#      country matches in 10k sampled ground-truth pairs per the EDA) — a
#      free ~60% search-space cut with zero recall cost, as long as the
#      SAME country is spelled the SAME way across S1/S2/S3.
#   2. Stage 3's GBM never sees raw country identity (an unseen category at
#      inference has no learned split, per XGBoost's own docs) — instead it
#      sees a derived, symmetric match/mismatch flag computed by comparing
#      two records' country strings. That comparison is only trustworthy if
#      spelling is consistent; "US" vs "United States" would silently read
#      as a mismatch between two same-country records.
#
# The fix for both is a canonicalization step, not removal: fold KNOWN
# aliases of the same country to one canonical token, and pass anything not
# in the table through UNCHANGED (lowercased/whitespace-normalized only).
# France, or any country never seen in training, is never filtered or
# mapped to "unknown" — it flows through exactly like this design requires.
# ---------------------------------------------------------------------------

_COUNTRY_ALIAS_CLASSES = [
    ["us", "usa", "united states", "united states of america", "u s", "u s a"],
    ["india", "in", "bharat"],
    ["france", "fr"],
]


def _build_country_map(classes: list[list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for cls in classes:
        canonical = cls[0]
        for variant in cls:
            out[variant] = canonical
    return out


_COUNTRY_MAP = _build_country_map(_COUNTRY_ALIAS_CLASSES)


def canonicalize_country(country: str) -> str:
    """Normalizes known aliases to one canonical spelling for BLOCKING
    partitioning; passes any unrecognized label (France included) through
    unchanged after basic normalization. Never drops, never maps to a
    sentinel 'unknown' value — an open string set stays open."""
    if not country:
        return ""
    key = unicodedata.normalize("NFKC", country).strip().lower()
    key = _WHITESPACE_RE.sub(" ", key) if "_WHITESPACE_RE" in globals() else re.sub(r"\s+", " ", key)
    return _COUNTRY_MAP.get(key, key)


def country_match_flag(country_a: str, country_b: str) -> Optional[int]:
    """Stage 2c feature: the ONLY country-derived signal that should ever
    reach the GBM. Symmetric, derived, works on any country pair including
    ones never seen in training (unlike a raw categorical country feature).
    Returns None when either side is empty, so the caller can decide
    whether to mask it to missing rather than guessing a value — XGBoost/
    LightGBM learn a sensible default split direction for genuinely missing
    values, per training.md's country-handling rule."""
    a, b = canonicalize_country(country_a), canonicalize_country(country_b)
    if not a or not b:
        return None
    return int(a == b)


# ---------------------------------------------------------------------------
# 2. Bidirectional canonicalization tables
#    Key design point: we map EVERY variant -> ONE canonical token, in both
#    directions. This is not "always expand the abbreviation" — it's a lookup
#    table built to be equally happy collapsing "Corporation" and "Corp" to
#    the same key, so a downstream exact/fuzzy match doesn't care which form
#    either source happened to use.
# ---------------------------------------------------------------------------

# Legal-suffix equivalence classes (canonical form is first element).
# Deliberately NOT scoped to a country. If a token collides across legal
# systems (e.g. "sa" could be a US initialism or a French "société anonyme"),
# collapsing them into one bucket is fine for BLOCKING/similarity purposes —
# Stage 1 already treats blocking as recall-favoring, not precision-favoring.
_LEGAL_SUFFIX_CLASSES = [
    ["inc", "incorporated"],
    ["corp", "corporation"],
    ["co", "company"],
    ["llc", "l l c", "limited liability company"],
    ["ltd", "limited"],
    ["llp", "limited liability partnership"],
    ["pvt ltd", "private limited", "pvt limited", "p ltd"],
    ["sarl", "societe a responsabilite limitee"],
    ["sas", "societe par actions simplifiee"],
    ["sa", "societe anonyme"],
    ["eurl", "entreprise unipersonnelle a responsabilite limitee"],
    ["dba", "doing business as"],
]

# Generic street/address-token equivalence classes. Again: not country-scoped
# rule sets — just token-level synonym collapsing, so "Rd"/"Road"/"Route"-like
# variance in ANY language/locale that happens to romanize the same way
# collapses together. This is deliberately shallow; it is a blocking aid,
# not an address parser.
_ADDRESS_TOKEN_CLASSES = [
    ["st", "street"],
    ["rd", "road"],
    ["ave", "avenue"],
    ["blvd", "boulevard"],
    ["dr", "drive"],
    ["ste", "suite"],
    ["apt", "apartment"],
    ["bldg", "building"],
    ["fl", "floor"],
    ["hwy", "highway"],
    ["ln", "lane"],
    ["ct", "court"],
    ["pl", "place"],
    ["sq", "square"],
    ["pkwy", "parkway"],
    ["&", "and"],
]


def _build_canonical_map(classes: list[list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for cls in classes:
        canonical = cls[0]
        for variant in cls:
            out[variant] = canonical
    return out


_LEGAL_MAP = _build_canonical_map(_LEGAL_SUFFIX_CLASSES)
_ADDR_MAP = _build_canonical_map(_ADDRESS_TOKEN_CLASSES)

# Sort multi-word keys longest-first so "private limited" matches before "limited"
_LEGAL_KEYS_BY_LEN = sorted(_LEGAL_MAP.keys(), key=len, reverse=True)
_ADDR_KEYS_BY_LEN = sorted(_ADDR_MAP.keys(), key=len, reverse=True)


def _apply_token_map(text: str, keys_by_len: list[str], mapping: dict[str, str]) -> str:
    for key in keys_by_len:
        # word-boundary match; keys may themselves contain internal spaces
        pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
        text = re.sub(pattern, mapping[key], text)
    return text


# ---------------------------------------------------------------------------
# 3. Landmark-phrase stripping (weak auxiliary signal, not deletion)
#    Landmarks are language-agnostic in *pattern* (a preposition-like word
#    followed by a proper-noun-ish span) but we don't attempt real NER here —
#    a lightweight, data-driven cue-word regex catches the common shapes
#    observed in the EDA ("near X", "opp X", "opposite X") plus their
#    French cognates, without branching pipeline logic by country.
# ---------------------------------------------------------------------------

_LANDMARK_CUE_RE = re.compile(
    r"\b(near|nr|opp|opposite|behind|beside|next to|pres de|proche de)\b[^,]*",
    flags=re.IGNORECASE,
)


def extract_landmark_and_strip(text: str) -> tuple[str, Optional[str]]:
    """Returns (text_with_landmark_removed, landmark_phrase_or_None)."""
    m = _LANDMARK_CUE_RE.search(text)
    if not m:
        return text, None
    landmark = m.group(0).strip(" ,")
    cleaned = (text[: m.start()] + text[m.end() :]).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
    return cleaned, landmark


# ---------------------------------------------------------------------------
# 4. Data-driven structural sub-field extraction
#    No per-country postal-code regex. Instead: generic shape heuristics that
#    happen to work across US ZIP / Indian PIN / French Code Postal because
#    all three are "a run of 5-6 digits, often near the end of the string."
#    We extract candidates and keep them as separate signal columns without
#    asserting which country-specific meaning they have.
# ---------------------------------------------------------------------------

_DIGIT_RUN_RE = re.compile(r"\b\d{4,10}(?:-\d{3,4})?\b")
_TRAILING_SEGMENT_RE = re.compile(r",\s*([^,]+)$")


@dataclass
class StructuralFields:
    digit_runs: list[str] = field(default_factory=list)   # candidate postal/PIN/ZIP/house-no codes
    trailing_segment: Optional[str] = None                  # last comma-separated chunk (often city/state/region)
    street_number: Optional[str] = None                     # leading digit run, if address starts with one


def extract_structural_fields(address: str) -> StructuralFields:
    digit_runs = _DIGIT_RUN_RE.findall(address)
    trailing = _TRAILING_SEGMENT_RE.search(address)
    trailing_segment = trailing.group(1).strip() if trailing else None
    lead = re.match(r"^\s*(\d{1,6})\b", address)
    street_number = lead.group(1) if lead else None
    return StructuralFields(
        digit_runs=digit_runs,
        trailing_segment=trailing_segment,
        street_number=street_number,
    )


# ---------------------------------------------------------------------------
# 5. Core text normalization
# ---------------------------------------------------------------------------

_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u201A\u2032]")
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\r\x00-\x1f]")
_ALLOWED_EXTRA = set("&'-")


def _strip_punct(text: str) -> str:
    """Category-based punctuation stripping, not \\w-based.
    Python's \\w matches Unicode letters/digits but NOT combining marks
    (category Mn), which is fatal for Devanagari and other scripts that
    encode vowel signs as combining marks attached to a base consonant —
    a naive \\w-only strip disintegrates those into stray base characters.
    Keep categories L* (letter), M* (mark/combining), N* (number), plus
    whitespace and a small allow-list of meaningful ASCII punctuation."""
    out = []
    for ch in text:
        if ch.isspace() or ch in _ALLOWED_EXTRA:
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def normalize_text(text: Optional[str], *, is_name: bool) -> str:
    """Country-agnostic normalization. Preserves accented characters
    (French, Indian-script transliteration) per architecture.md — NEVER
    strip non-ASCII, only normalize its representation."""
    if text is None:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _APOSTROPHE_RE.sub("'", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = text.lower()
    text = _strip_punct(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    keys_by_len = _LEGAL_KEYS_BY_LEN if is_name else _ADDR_KEYS_BY_LEN
    mapping = _LEGAL_MAP if is_name else _ADDR_MAP
    text = _apply_token_map(text, keys_by_len, mapping)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 6. Record-level pipeline
# ---------------------------------------------------------------------------

@dataclass
class NormalizedRecord:
    entity_id: str
    source: str                      # 'S1' | 'S2' | 'S3', derived from entity_id
    country_raw: str                 # verbatim — never filtered, never restricted to a fixed set
    country_canonical: str           # alias-folded for blocking partition / match-flag use; unseen labels (e.g. France) pass through unchanged
    name_raw: str
    address_raw: str
    name_norm: str                   # lowercased, punctuation-stripped, suffix-canonicalized
    address_norm: str                # same, with landmark stripped
    landmark: Optional[str]
    is_missing_address: bool
    structural: StructuralFields
    # Case-preserved variant, kept for the dense bi-encoder per architecture.md
    # note that transformer embeddings benefit from case (e.g. acronyms).
    name_cased: str
    address_cased: str


def process_record(entity_id: str, business_name: str, business_address: str, country: str) -> NormalizedRecord:
    source = source_from_entity_id(entity_id)

    address_raw = business_address if business_address is not None else ""
    is_missing = len(address_raw.strip()) == 0

    name_cased = _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", business_name or "")).strip()
    address_cased = _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", address_raw)).strip()

    name_norm = normalize_text(business_name, is_name=True)
    addr_stage = normalize_text(address_raw, is_name=False)
    addr_stripped, landmark = extract_landmark_and_strip(addr_stage) if addr_stage else (addr_stage, None)

    structural = extract_structural_fields(address_raw) if address_raw else StructuralFields()

    return NormalizedRecord(
        entity_id=entity_id.strip(),
        source=source,
        country_raw=(country or "").strip(),   # never normalized to a fixed vocabulary
        country_canonical=canonicalize_country(country or ""),
        name_raw=business_name or "",
        address_raw=address_raw,
        name_norm=name_norm,
        address_norm=addr_stripped,
        landmark=landmark,
        is_missing_address=is_missing,
        structural=structural,
        name_cased=name_cased,
        address_cased=address_cased,
    )


# ---------------------------------------------------------------------------
# 7. Streaming file I/O — chunked, never loads the full 5M+ row file into RAM
# ---------------------------------------------------------------------------

def stream_source_tsv(path: str | Path) -> Iterator[NormalizedRecord]:
    """Yields NormalizedRecord one row at a time. Safe for the 5M+ row S2/S3
    files without blowing memory — required per the EDA's stated file sizes."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing expected columns {missing}")
        for row in reader:
            yield process_record(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"],
            )


def normalize_file_to_tsv(in_path: str | Path, out_path: str | Path, chunk_size: int = 50_000) -> int:
    """Streams in_path -> out_path (normalized TSV), chunked writes.
    Returns the number of rows written."""
    fieldnames = [
        "entity_id", "source", "country", "country_canonical",
        "name_raw", "address_raw",
        "name_norm", "address_norm",
        "name_cased", "address_cased",
        "landmark", "is_missing_address",
        "street_number", "trailing_segment", "digit_runs",
    ]
    count = 0
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        buffer = []
        for rec in stream_source_tsv(in_path):
            buffer.append({
                "entity_id": rec.entity_id,
                "source": rec.source,
                "country": rec.country_raw,
                "country_canonical": rec.country_canonical,
                "name_raw": rec.name_raw,
                "address_raw": rec.address_raw,
                "name_norm": rec.name_norm,
                "address_norm": rec.address_norm,
                "name_cased": rec.name_cased,
                "address_cased": rec.address_cased,
                "landmark": rec.landmark or "",
                "is_missing_address": int(rec.is_missing_address),
                "street_number": rec.structural.street_number or "",
                "trailing_segment": rec.structural.trailing_segment or "",
                "digit_runs": "|".join(rec.structural.digit_runs),
            })
            count += 1
            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                buffer.clear()
        if buffer:
            writer.writerows(buffer)
    return count


# ---------------------------------------------------------------------------
# 8. Quick self-test against the sample rows from the prompt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    samples = [
        ("S1-925783039", "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "US"),
        ("S1-133037285", "Christ Chapel", "2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US"),
        ("S2-166376419", "राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"),
        ("S3-859268022", "International South Consultants Private Ltd", "", "India"),
        ("S2-508602797", "FOUNDATION EXCEL AGENCY PRIVATE  LIMITED", "HN 753 E-1, BHARAT NAGAR, 104/1/1 ERANDWANE, Maharashtra", "India"),
        ("S1-000000001", "Boulangerie Dupont SARL", "12 Rue de la Paix, 75002 Paris", "France"),
        ("S2-000000002", "Some US Corp", "1 Main St", "USA"),  # alias spelling, must canonicalize with "US"
    ]
    for eid, name, addr, country in samples:
        r = process_record(eid, name, addr, country)
        print(f"\n{eid}  [{r.source}, raw={r.country_raw!r} -> canonical={r.country_canonical!r}]")
        print(f"  name_norm : {r.name_norm!r}")
        print(f"  addr_norm : {r.address_norm!r}")
        print(f"  landmark  : {r.landmark!r}")
        print(f"  missing?  : {r.is_missing_address}")
        print(f"  structural: {r.structural}")

    print("\n--- country_match_flag sanity checks ---")
    print("US vs USA        :", country_match_flag("US", "USA"))        # -> 1, alias-folded
    print("India vs France  :", country_match_flag("India", "France"))  # -> 0
    print("France vs France :", country_match_flag("France", "France")) # -> 1, works with zero training exposure
    print("US vs ''         :", country_match_flag("US", ""))           # -> None, caller masks to missing
