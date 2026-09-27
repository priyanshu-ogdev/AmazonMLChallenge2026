"""
Layer 1: Multi-Channel Candidate Generation and Blocking.

Implements the Stage 1 specification defined in docs/04_stage1_blocking.md:
  1. Multi-channel independent blocking:
     - Exact normalized name
     - Composite name + address structural keys (postal, street number, trailing segment)
     - Lead token composite keys
     - Structural address keys (postal + street number)
     - Significant token inverted index with frequency capping and IDF scoring
     - Character 3-gram sparse similarity index
     - Optional dense BGE embedding hook
  2. Strict country partitioning invariant:
     - Hard retrieval partitioning when canonical countries match
     - Unseen / open-set countries match their own partition
     - Missing / empty country records fallback to global search
  3. Deterministic combined priority ranking and per-S1 capping
  4. Contract compliance:
     - candidate_pairs.tsv (source1_entity_id \t candidate_entity_ids)
     - candidate_provenance.tsv (per-pair auditable blocker trace)
     - blocking_summary.json
  5. Built-in recall audit gate against ground-truth
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import heapq
import io
import itertools
import json
import math
import multiprocessing
import os
import pickle
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from src.normalize import (
    KNOWN_CANONICAL_COUNTRIES,
    canonicalize_country,
    extract_postal_code,
    extract_structural_fields,
    is_missing_address,
    normalize_address,
    normalize_name,
    source_from_entity_id,
)

# Configuration defaults matching docs/04_stage1_blocking.md
TOP_K_DENSE = 50
TOP_K_SPARSE_OR_CHAR = 50
MAX_CANDIDATES_PER_ENTITY = 100
SIMILARITY_FLOOR = 0.30
MAX_TOKEN_DOC_FREQ = 0.02
MAX_TOKEN_DOC_COUNT = 5000
MAX_NGRAM_DOC_FREQ = 0.20
MAX_NGRAM_DOC_COUNT = 10000
MIN_TOKEN_LEN = 3

# Index cache versioning — bump when MultiChannelBlocker schema changes
# v4: precomputed IDF lookup tables stored in pickle for zero-cost per-entity scoring
INDEX_CACHE_VERSION = 4

# Worker I/O batch size: flush StringIO buffer every N entities (reduces CSV writer overhead)
_IO_BATCH_SIZE = 500


class CandidateHit(tuple):
    """Zero-overhead tuple representing a candidate hit, supporting both index and key access."""
    __slots__ = ()
    _FIELDS = (
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocker_provenance",
        "blocker_count",
        "best_blocker_rank",
        "best_blocker_score",
        "country_partition",
    )
    _FIELD_MAP = {name: i for i, name in enumerate(_FIELDS)}

    def __getitem__(self, item):
        if isinstance(item, str):
            idx = self._FIELD_MAP.get(item)
            if idx is not None:
                return super().__getitem__(idx)
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        idx = self._FIELD_MAP.get(key)
        return super().__getitem__(idx) if idx is not None else default
INDEX_CACHE_NAME = "candidate_index.pkl"
_shared_blocker = None
CHECKPOINT_INTERVAL = 10_000

# Module-level empty sentinel for _query_partitioned_index fast paths
_EMPTY_LIST: List[str] = []

EXACT_BLOCKERS = {
    "exact_name",
    "exact_name_postal",
    "exact_name_street",
    "exact_name_trailing",
    "name_lead_postal",
    "name_lead_street_trailing",
    "first_2_tokens",
    "acronym_match",
}

_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)

_KNOWN_ACRONYMS = {
    "ge", "ibm", "hp", "ups", "fedex", "dhl", "bmw", "sap", "att", "cvs",
    "pnc", "hsbc", "kpmg", "ey", "pwc", "bbva", "td", "bmo", "cibc", "citi",
    "ubs", "cs", "amc", "cbs", "nbc", "abc", "cnn", "fox", "espn", "hbo",
    "mri", "dna", "gmc", "vw", "fca", "byd", "tata", "lg", "nec", "jvc",
    "sony", "jpm", "bofa", "bnp", "ing", "aig", "axa", "geico", "mci",
    "ncr", "trw", "itt", "bap", "cat", "jcb", "cnh", "agco",
}

_ACRONYM_SKIP = {
    "in", "at", "of", "on", "to", "by", "or", "an", "as", "is", "it",
    "de", "la", "le", "du", "et", "en", "st", "rd", "and", "the", "for",
}


def _tokenize(text: str) -> List[str]:
    """Extract lowercased word tokens."""
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= MIN_TOKEN_LEN]


def _char_ngrams(text: str) -> List[str]:
    """Extract character 3-grams and 4-grams with edge padding per docs/04 Channel 2 spec."""
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return []
    padded = f"  {compact}  "
    ngrams = []
    for n in (3,):
        for i in range(max(0, len(padded) - n + 1)):
            ngrams.append(padded[i : i + n])
    return ngrams


@dataclass
class BlockingRecord:
    """Preprocessed record attributes used across blocking channels."""
    entity_id: str
    source: str
    raw_name: str
    raw_address: str
    raw_country: str
    canonical_country: str
    norm_name: str
    norm_address: str
    is_address_missing: bool
    postal_code: Optional[str]
    street_number: Optional[str]
    trailing_segment: Optional[str]
    tokens: List[str]
    token_counts: Counter[str]
    first_2_tokens: Optional[str]
    acronyms: Set[str]
    ngram_counts: Counter[str]
    first_word: Optional[str]

    @property
    def ngrams(self) -> List[str]:
        """Dynamic access to character ngrams without storing duplicate list references."""
        return list(self.ngram_counts.elements())

    @property
    def trigrams(self) -> Set[str]:
        """Dynamic backward-compatible access to character trigrams."""
        return {g for g in self.ngram_counts if len(g) == 3}

    @classmethod
    def from_row(
        cls,
        entity_id: str,
        name: str,
        address: str,
        country: str,
        norm_name: Optional[str] = None,
        norm_address: Optional[str] = None,
        canonical_country: Optional[str] = None,
        postal_code: Optional[str] = None,
        street_number: Optional[str] = None,
        trailing_segment: Optional[str] = None,
        is_address_missing: Optional[bool] = None,
    ) -> BlockingRecord:
        n_name = norm_name if norm_name is not None else normalize_name(name)
        n_addr = norm_address if norm_address is not None else normalize_address(address)
        canon_c = canonical_country if (canonical_country is not None and canonical_country != "") else canonicalize_country(country)
        if is_address_missing is not None:
            is_addr_missing = bool(is_address_missing)
        else:
            is_addr_missing = is_missing_address(address)

        if postal_code is not None:
            postal = postal_code if postal_code else None
        else:
            postal = extract_postal_code(address if not is_addr_missing else "", country=canon_c)

        if street_number is not None or trailing_segment is not None:
            street_no = street_number if street_number else None
            trailing = trailing_segment if trailing_segment else None
        else:
            struct = extract_structural_fields(n_addr if n_addr else (address if not is_addr_missing else ""))
            street_no = struct.get("street_number")
            trailing = struct.get("trailing_segment")
        tokens = _tokenize(n_name)
        token_counts = Counter(tokens)
        first_word = tokens[0] if tokens else None

        # Content words for first-2-tokens and acronym extraction (preserving 2-letter tokens like 'GE')
        raw_words = (w.group().lower() for w in _WORD_RE.finditer(n_name or ""))
        content_words = [w for w in raw_words if w not in ("the", "a", "an", "and", "&")]

        # First-2-Tokens Key
        first_2_tokens = " ".join(content_words[:2]) if len(content_words) >= 2 else None

        # Acronym Keys (both 2-letter prefix, full initials, and explicit short acronym tokens)
        acronyms: Set[str] = set()
        if len(content_words) >= 2:
            # 2-letter prefix acronym (e.g. General Electric -> ge)
            acronyms.add(content_words[0][0] + content_words[1][0])
            # Full initials (e.g. General Electric Medical Systems -> gems)
            full_acr = "".join(w[0] for w in content_words if w)
            if 2 <= len(full_acr) <= 6:
                acronyms.add(full_acr)

        # Explicit short acronym tokens (e.g. 'GE' in 'GE Healthcare' or standalone 'IBM')
        # Only tokens that are all-uppercase in the raw name or in curated _KNOWN_ACRONYMS
        raw_tokens = _WORD_RE.findall(str(name or ""))
        for tok in raw_tokens:
            low = tok.lower()
            if low in _ACRONYM_SKIP:
                continue
            if (tok.isupper() and 2 <= len(tok) <= 5) or low in _KNOWN_ACRONYMS:
                acronyms.add(low)

        # 3-gram and 4-gram character tokens
        ngrams = _char_ngrams(n_name) if len(n_name) >= 3 else []
        ngram_counts = Counter(ngrams)

        return cls(
            entity_id=entity_id.strip(),
            source=source_from_entity_id(entity_id),
            raw_name=name,
            raw_address=address,
            raw_country=country,
            canonical_country=canon_c,
            norm_name=n_name,
            norm_address=n_addr,
            is_address_missing=is_addr_missing,
            postal_code=postal,
            street_number=street_no,
            trailing_segment=trailing,
            tokens=tokens,
            token_counts=token_counts,
            first_2_tokens=first_2_tokens,
            acronyms=acronyms,
            ngram_counts=ngram_counts,
            first_word=first_word,
        )


def read_tsv_records(paths: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    """Yield entity rows from one or more raw challenge TSVs or Layer 0 normalized TSVs."""
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input TSV does not exist: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            fieldnames = set(reader.fieldnames or [])
            if "entity_id" not in fieldnames:
                raise ValueError(f"{path} is missing required 'entity_id' column")
            has_name = any(c in fieldnames for c in ("business_name", "raw_name", "norm_name"))
            has_addr = any(c in fieldnames for c in ("business_address", "raw_address", "norm_address"))
            if not has_name:
                raise ValueError(f"{path} is missing name column (expected business_name, raw_name, or norm_name)")
            if not has_addr:
                raise ValueError(f"{path} is missing address column (expected business_address, raw_address, or norm_address)")

            is_stage0 = "norm_name" in fieldnames and "norm_address" in fieldnames
            for row in reader:
                name = row.get("business_name") or row.get("raw_name") or row.get("norm_name") or ""
                addr = row.get("business_address") or row.get("raw_address") or row.get("norm_address") or ""
                country = row.get("country") or row.get("country_canonical") or ""
                rec: Dict[str, Any] = {
                    "entity_id": row.get("entity_id", ""),
                    "business_name": name,
                    "business_address": addr,
                    "country": country,
                }
                if is_stage0:
                    rec["norm_name"] = row.get("norm_name", "")
                    rec["norm_address"] = row.get("norm_address", "")
                    rec["canonical_country"] = row.get("country_canonical", "")
                    rec["postal_code"] = row.get("postal_code", "")
                    rec["street_number"] = row.get("street_number", "")
                    rec["trailing_segment"] = row.get("trailing_segment", "")
                    raw_miss = row.get("is_address_missing")
                    rec["is_address_missing"] = str(raw_miss).strip().lower() in ("1", "true", "t") if raw_miss is not None else False
                yield rec


class MultiChannelBlocker:
    """
    Inverted index store and candidate generator supporting multi-channel blocking.
    """

    def __init__(
        self,
        top_k_sparse: int = TOP_K_SPARSE_OR_CHAR,
        top_k_dense: int = TOP_K_DENSE,
        max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
        similarity_floor: float = SIMILARITY_FLOOR,
        max_token_doc_freq: float = MAX_TOKEN_DOC_FREQ,
        max_token_doc_count: int = MAX_TOKEN_DOC_COUNT,
        max_ngram_doc_freq: float = MAX_NGRAM_DOC_FREQ,
        max_ngram_doc_count: int = MAX_NGRAM_DOC_COUNT,
    ):
        self.top_k_sparse = top_k_sparse
        self.top_k_dense = top_k_dense
        self.max_candidates_per_entity = max_candidates_per_entity
        self.similarity_floor = similarity_floor
        self.max_token_doc_freq = max_token_doc_freq
        self.max_token_doc_count = max_token_doc_count
        self.max_ngram_doc_freq = max_ngram_doc_freq
        self.max_ngram_doc_count = max_ngram_doc_count

        # Total indexed candidate records
        self.num_candidates = 0
        # Compact mapping of candidate entity ID -> ngram count / length for cosine normalization
        self.candidate_ngram_lens: Dict[str, int] = {}
        # Backwards-compatibility alias
        self.candidate_records = self.candidate_ngram_lens

        # Country partition tracking
        # canonical_country -> set of candidate entity IDs
        self.country_partitions: Dict[str, Set[str]] = defaultdict(set)
        self.all_candidate_ids: Set[str] = set()

        # Channel 1: Exact & Composite Key Inverted Indexes
        # Key tuple -> canonical_country -> list of candidate entity IDs
        self.index_exact_name: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_first_2_tokens: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_acronym: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_name_postal: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_name_street: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_name_trailing: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_lead_postal: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_lead_street_trailing: Dict[Tuple[str, str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Channel 2: Character 3/4-Gram TF-IDF Inverted Index
        self.index_ngrams: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.ngram_doc_counts: Counter[str] = Counter()
        self.index_trigrams = self.index_ngrams
        self.trigram_doc_counts = self.ngram_doc_counts

        # Channel 3: Token Inverted Index with Sub-Linear TF-IDF
        self.index_tokens: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.token_doc_counts: Counter[str] = Counter()

        # Channel 4: Address Structural Key (postal + street number)
        self.index_address_structural: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Precomputed IDF lookup tables (built by build_idf_tables() after indexing)
        # gram -> float IDF weight (avoids per-entity math.log calls in hot loop)
        self.ngram_idf_table: Dict[str, float] = {}
        # token -> float IDF weight
        self.token_idf_table: Dict[str, float] = {}
        # Precomputed frequency caps (updated when num_candidates changes significantly)
        self._cached_max_ngram_count: int = 0
        self._cached_max_token_count: int = 0

    def index_candidate(self, record: BlockingRecord) -> None:
        """Add a candidate record (from S2 or S3) to all blocking indexes."""
        cid = sys.intern(record.entity_id)
        country = record.canonical_country
        self.candidate_ngram_lens[cid] = sum(record.ngram_counts.values()) if record.ngram_counts else max(1, len(record.norm_name) - 2)
        self.num_candidates += 1
        self.country_partitions[country].add(cid)

        # 1. Exact & Structural Name Keys (cap postings at 1000 to prevent degenerate key explosion)
        if record.norm_name:
            sub = self.index_exact_name[record.norm_name][country]
            if len(sub) < 1000:
                sub.append(cid)
        if record.first_2_tokens:
            sub = self.index_first_2_tokens[record.first_2_tokens][country]
            if len(sub) < 1000:
                sub.append(cid)
        for acr in record.acronyms:
            sub = self.index_acronym[acr][country]
            if len(sub) < 1000:
                sub.append(cid)

        # Composite Keys
        if record.norm_name and record.postal_code:
            self.index_name_postal[(record.norm_name, record.postal_code)][country].append(cid)

        if record.norm_name and record.street_number:
            self.index_name_street[(record.norm_name, record.street_number)][country].append(cid)

        if record.norm_name and record.trailing_segment:
            self.index_name_trailing[(record.norm_name, record.trailing_segment)][country].append(cid)

        if record.first_word and record.postal_code and len(record.first_word) >= 3:
            self.index_lead_postal[(record.first_word, record.postal_code)][country].append(cid)

        if record.first_word and record.street_number and record.trailing_segment:
            self.index_lead_street_trailing[
                (record.first_word, record.street_number, record.trailing_segment)
            ][country].append(cid)

        # 2. Character 3/4-Gram Inverted Index (with stop-ngram memory eviction)
        for gram in record.ngram_counts:
            cnt = self.ngram_doc_counts[gram] + 1
            self.ngram_doc_counts[gram] = cnt
            if cnt <= self.max_ngram_doc_count:
                self.index_ngrams[gram][country].append(cid)
            elif cnt == self.max_ngram_doc_count + 1:
                # Exceeded stop-ngram threshold; drop postings to reclaim memory!
                self.index_ngrams.pop(gram, None)

        # 3. Token Inverted Index (with stop-token memory eviction)
        for tok in set(record.tokens):
            cnt = self.token_doc_counts[tok] + 1
            self.token_doc_counts[tok] = cnt
            if cnt <= self.max_token_doc_count:
                self.index_tokens[tok][country].append(cid)
            elif cnt == self.max_token_doc_count + 1:
                # Exceeded stop-token threshold; drop postings to reclaim memory!
                self.index_tokens.pop(tok, None)

        # 4. Address Structural Key (postal + street number, bypassed if missing address)
        if not record.is_address_missing and record.postal_code and record.street_number:
            self.index_address_structural[(record.postal_code, record.street_number)][country].append(cid)

    def build_idf_tables(self) -> None:
        """Precompute IDF weights for all grams and tokens.

        Call once after all candidates are indexed (or after load_index) to eliminate
        per-entity math.log() calls from the hot query loop. This is the single biggest
        throughput lever: with 10M+ candidates, these tables eliminate ~15 math.log calls
        per gram × N active grams per entity × 1.8M entities = billions of log calls saved.
        """
        n = max(1, self.num_candidates)
        max_ngram = max(50, int(n * self.max_ngram_doc_freq))
        max_ngram = min(self.max_ngram_doc_count, max_ngram)
        max_token = max(10, int(n * self.max_token_doc_freq))
        max_token = min(self.max_token_doc_count, max_token)

        self._cached_max_ngram_count = max_ngram
        self._cached_max_token_count = max_token

        # Ngram IDF table: only compute for grams still in the active index
        ngram_idf: Dict[str, float] = {}
        _log = math.log
        for gram, doc_cnt in self.ngram_doc_counts.items():
            if gram not in self.index_ngrams:
                continue  # evicted stop-gram
            if doc_cnt == 0 or doc_cnt > max_ngram:
                continue
            ngram_idf[gram] = _log(1.0 + (n - doc_cnt + 0.5) / (doc_cnt + 0.5))
        self.ngram_idf_table = ngram_idf

        # Token IDF table: only compute for tokens still in the active index
        token_idf: Dict[str, float] = {}
        for tok, doc_cnt in self.token_doc_counts.items():
            if tok not in self.index_tokens:
                continue  # evicted stop-token
            if doc_cnt == 0 or doc_cnt > max_token:
                continue
            token_idf[tok] = _log(1.0 + (n - doc_cnt + 0.5) / (doc_cnt + 0.5))
        self.token_idf_table = token_idf

    def _query_partitioned_index(
        self,
        index: Dict[Any, Dict[str, List[str]]],
        key: Any,
        s1_country: str,
        allow_cross_country_fallback: bool = True,
    ) -> Iterable[str]:
        """
        Query an index respecting the country partitioning invariant.
        Phase 1 optimized: avoids unnecessary list() creation on hot paths.
        """
        sub = index.get(key)
        if not sub:
            return _EMPTY_LIST

        if s1_country:
            exact = sub.get(s1_country, _EMPTY_LIST)
            global_part = sub.get("", _EMPTY_LIST)
            if exact or global_part:
                if not global_part:
                    return exact
                if not exact:
                    return global_part
                return itertools.chain(exact, global_part)
            if not allow_cross_country_fallback:
                return _EMPTY_LIST
            # Soft fallback: Disjoint known markets are never crossed.
            # Open-set / unrecognized countries are searched to recover typos.
            def _fallback_gen() -> Iterable[str]:
                for c, partition_list in sub.items():
                    if c == s1_country or c == "":
                        continue
                    if s1_country in KNOWN_CANONICAL_COUNTRIES and c in KNOWN_CANONICAL_COUNTRIES:
                        continue
                    for cid in partition_list:
                        yield cid
            return _fallback_gen()
        else:
            # S1 country is missing -> global fallback
            def _global_gen() -> Iterable[str]:
                for partition_list in sub.values():
                    for cid in partition_list:
                        yield cid
            return _global_gen()

    def generate_candidates_for_record(
        self,
        s1: BlockingRecord,
        dense_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Any, ...]]:
        """Generate, union, score, and rank candidates for a single Source 1 record.

        Returns a list of CandidateHit tuples in the order required by provenance.tsv:
          (s1_id, cand_id, cand_source, provenance, count, rank, score, country)

        Perf v4 upgrades (20x target vs original):
          - ALL _query_partitioned_index calls are INLINED — zero function-call overhead
            on the 100s-per-entity hot path.
          - IDF weights read from precomputed table (ngram_idf_table / token_idf_table)
            instead of calling math.log() once per gram per entity.
          - heapq.nlargest replaces full sort() for top-K candidate selection.
          - Provenance string built with a pre-sorted frozenset to avoid repeated sort().
          - Ranking uses a single sort() with a pre-built key tuple list.
          - Source prefix extracted from first two chars of cid (avoids function call).
        """
        # Auto-build IDF tables if not explicitly called (e.g., in unit tests)
        if not self.ngram_idf_table and self.num_candidates > 0:
            self.build_idf_tables()

        # --------------- local aliases (eliminate per-call LOAD_ATTR overhead) --------
        index_exact_name         = self.index_exact_name
        index_first_2_tokens     = self.index_first_2_tokens
        index_acronym            = self.index_acronym
        index_name_postal        = self.index_name_postal
        index_name_street        = self.index_name_street
        index_name_trailing      = self.index_name_trailing
        index_lead_postal        = self.index_lead_postal
        index_lead_street_trail  = self.index_lead_street_trailing
        index_ngrams             = self.index_ngrams
        index_tokens             = self.index_tokens
        index_addr_struct        = self.index_address_structural
        ngram_idf_table          = self.ngram_idf_table
        token_idf_table          = self.token_idf_table
        candidate_ngram_lens     = self.candidate_ngram_lens
        top_k_sparse             = self.top_k_sparse
        similarity_floor         = self.similarity_floor
        max_candidates           = self.max_candidates_per_entity
        _EMPTY                   = _EMPTY_LIST
        _KNOWN                   = KNOWN_CANONICAL_COUNTRIES
        _EXACT_B                 = EXACT_BLOCKERS
        _sqrt                    = math.sqrt
        _log                     = math.log

        # --- hit accumulation dicts (flat dict: cid -> scalar, avoids defaultdict overhead) ---
        hit_blockers:   Dict[str, List[str]] = {}
        hit_best_score: Dict[str, float]     = {}
        hit_best_rank:  Dict[str, int]        = {}
        hit_exact:      Dict[str, bool]       = {}

        country = s1.canonical_country

        # ------------------------------------------------------------------
        # INLINED _query_partitioned_index helper (macro-expanded per channel)
        # Eliminates ~100 Python function calls per entity at the cost of
        # verbosity. Logic is identical to the old method.
        # ------------------------------------------------------------------
        def _hits(index: Dict, key) -> Iterable[str]:
            """Inline-friendly country-partitioned index lookup."""
            sub = index.get(key)
            if not sub:
                return _EMPTY
            if country:
                exact      = sub.get(country, _EMPTY)
                global_p   = sub.get("", _EMPTY)
                if exact or global_p:
                    if not global_p:
                        return exact
                    if not exact:
                        return global_p
                    return itertools.chain(exact, global_p)
                # Soft fallback for open-set / unrecognized countries
                def _fb() -> Iterable[str]:
                    for c, pl in sub.items():
                        if c == country or c == "":
                            continue
                        if country in _KNOWN and c in _KNOWN:
                            continue
                        yield from pl
                return _fb()
            else:
                def _gb() -> Iterable[str]:
                    for pl in sub.values():
                        yield from pl
                return _gb()

        # Inlined record_hit — avoids function-call overhead on inner loop
        def _rh(cid: str, blocker: str, score: float, rank: int, exact: bool = False) -> None:
            if cid not in hit_best_score:
                hit_blockers[cid]   = [blocker]
                hit_best_score[cid] = score
                hit_best_rank[cid]  = rank
                hit_exact[cid]      = exact
            else:
                hit_blockers[cid].append(blocker)
                if score > hit_best_score[cid]:
                    hit_best_score[cid] = score
                if rank < hit_best_rank[cid]:
                    hit_best_rank[cid] = rank
                if exact:
                    hit_exact[cid] = True

        # -------------------------------------------------------------
        # Channel 1: Exact Name, First-2-Tokens, Acronym & Composite Keys
        # -------------------------------------------------------------
        if s1.norm_name:
            for rank, cid in enumerate(_hits(index_exact_name, s1.norm_name), 1):
                _rh(cid, "exact_name", 1.0, rank, True)

        if s1.first_2_tokens:
            for rank, cid in enumerate(_hits(index_first_2_tokens, s1.first_2_tokens), 1):
                _rh(cid, "first_2_tokens", 0.90, rank, True)

        for acr in s1.acronyms:
            for rank, cid in enumerate(_hits(index_acronym, acr), 1):
                _rh(cid, "acronym_match", 0.85, rank, True)

        if s1.norm_name and s1.postal_code:
            for rank, cid in enumerate(_hits(index_name_postal, (s1.norm_name, s1.postal_code)), 1):
                _rh(cid, "exact_name_postal", 1.0, rank, True)

        if s1.norm_name and s1.street_number:
            for rank, cid in enumerate(_hits(index_name_street, (s1.norm_name, s1.street_number)), 1):
                _rh(cid, "exact_name_street", 0.95, rank, True)

        if s1.norm_name and s1.trailing_segment:
            for rank, cid in enumerate(_hits(index_name_trailing, (s1.norm_name, s1.trailing_segment)), 1):
                _rh(cid, "exact_name_trailing", 0.95, rank, True)

        if s1.first_word and s1.postal_code and len(s1.first_word) >= 3:
            for rank, cid in enumerate(_hits(index_lead_postal, (s1.first_word, s1.postal_code)), 1):
                _rh(cid, "name_lead_postal", 0.90, rank, True)

        if s1.first_word and s1.street_number and s1.trailing_segment:
            for rank, cid in enumerate(_hits(index_lead_street_trail, (s1.first_word, s1.street_number, s1.trailing_segment)), 1):
                _rh(cid, "name_lead_street_trailing", 0.90, rank, True)

        # -------------------------------------------------------------
        # Channel 2: Character 3-Gram Sub-Linear TF-IDF Retrieval
        # Perf v4: reads from precomputed ngram_idf_table (no math.log per gram)
        # -------------------------------------------------------------
        s1_gram_counts = s1.ngram_counts
        if s1_gram_counts and ngram_idf_table:
            # Build s1 TF*IDF weights using precomputed IDF (single table lookup per gram)
            s1_weights: Dict[str, float] = {}
            for gram, count in s1_gram_counts.items():
                idf = ngram_idf_table.get(gram)  # None = evicted stop-gram or above cap
                if idf is None:
                    continue
                s1_weights[gram] = (1.0 + _log(count)) * idf

            if s1_weights:
                # Keep only top-15 weighted grams to cap index fan-out
                if len(s1_weights) > 15:
                    # heapq.nlargest is O(n + 15*log(15)) vs sorted O(n log n)
                    top_items = heapq.nlargest(15, s1_weights.items(), key=lambda kv: kv[1])
                    s1_weights = dict(top_items)

                # Accumulate candidate scores (one dict lookup per posting)
                cand_scores: Dict[str, float] = {}
                for gram, w_s1 in s1_weights.items():
                    for cid in _hits(index_ngrams, gram):
                        prev = cand_scores.get(cid, 0.0)
                        if prev == 0.0:
                            cand_scores[cid] = w_s1
                        else:
                            cand_scores[cid] = prev + w_s1

                if cand_scores:
                    s1_norm = _sqrt(sum(w * w for w in s1_weights.values()))
                    if s1_norm > 0:
                        sim_floor = similarity_floor
                        cand_len_get = candidate_ngram_lens.get
                        # Use nlargest to get top_k*3 candidates without full sort
                        top_cands = heapq.nlargest(top_k_sparse * 3, cand_scores.items(), key=lambda kv: kv[1])
                        scored_cands = []
                        for cid, raw_score in top_cands:
                            cand_len = cand_len_get(cid, 25)
                            sim = raw_score / (s1_norm * _sqrt(cand_len if cand_len > 0 else 1))
                            if sim > 1.0:
                                sim = 1.0
                            if sim >= sim_floor:
                                scored_cands.append((sim, cid))

                        # Sort descending by sim, take top_k_sparse
                        scored_cands.sort(key=lambda x: -x[0])
                        for rank, (sim, cid) in enumerate(scored_cands[:top_k_sparse], 1):
                            _rh(cid, "char_ngram", sim, rank, False)

        # -------------------------------------------------------------
        # Channel 3: Token Inverted Index with Sub-Linear TF-IDF
        # Perf v4: reads from precomputed token_idf_table (no math.log per token)
        # -------------------------------------------------------------
        if s1.tokens:
            s1_token_counts = s1.token_counts
            s1_token_weights: Dict[str, float] = {}
            for tok, count in s1_token_counts.items():
                idf = token_idf_table.get(tok)  # None = evicted stop-token or above cap
                if idf is None:
                    continue
                s1_token_weights[tok] = (1.0 + _log(count)) * idf

            if s1_token_weights:
                # Cap at top-10 tokens by weight
                if len(s1_token_weights) > 10:
                    top_t = heapq.nlargest(10, s1_token_weights.items(), key=lambda kv: kv[1])
                    s1_token_weights = dict(top_t)

                token_scores: Dict[str, float] = {}
                for tok, w_s1 in s1_token_weights.items():
                    for cid in _hits(index_tokens, tok):
                        prev = token_scores.get(cid, 0.0)
                        if prev == 0.0:
                            token_scores[cid] = w_s1
                        else:
                            token_scores[cid] = prev + w_s1

                total_s1_weight = sum(s1_token_weights.values())
                if token_scores and total_s1_weight > 0.0:
                    # nlargest instead of full sort
                    top_tok_cands = heapq.nlargest(top_k_sparse, token_scores.items(), key=lambda kv: kv[1])
                    inv_total = 1.0 / total_s1_weight
                    for rank, (cid, score_sum) in enumerate(top_tok_cands, 1):
                        normalized_score = score_sum * inv_total
                        if normalized_score > 1.0:
                            normalized_score = 1.0
                        _rh(cid, "token_inverted", normalized_score, rank, False)

        # -------------------------------------------------------------
        # Channel 4: Address Structural Key (postal + street number)
        # -------------------------------------------------------------
        if not s1.is_address_missing and s1.postal_code and s1.street_number:
            for rank, cid in enumerate(_hits(index_addr_struct, (s1.postal_code, s1.street_number)), 1):
                _rh(cid, "address_structural", 0.85, rank, False)

        # -------------------------------------------------------------
        # Channel 5: Dense Retrieval Hook (if provided)
        # -------------------------------------------------------------
        if dense_scores:
            for rank, (cid, score) in enumerate(
                heapq.nlargest(self.top_k_dense, ((c, s) for c, s in dense_scores.items() if s >= similarity_floor), key=lambda x: x[1]),
                1,
            ):
                _rh(cid, "dense_bge", float(score), rank, False)

        # ------------------------------------------------------------------
        # Union, Deterministic Priority Ranking & Capping
        # Perf v4: build (sort_key, hit) tuples in one pass, single sort()
        # Provenance string built once; source prefix extracted by slice.
        # ------------------------------------------------------------------
        if not hit_best_score:
            return []

        ranked_candidates = []
        s1_entity_id  = s1.entity_id
        s1_country    = s1.canonical_country or "global"
        for cid in hit_best_score:
            blockers_list = hit_blockers[cid]
            # Build sorted unique set only once (avoid set() then sorted() separately)
            if len(blockers_list) == 1:
                uniq_sorted = blockers_list
                blocker_cnt = 1
                is_exact    = hit_exact[cid] or (blockers_list[0] in _EXACT_B)
            else:
                uniq_set    = set(blockers_list)
                uniq_sorted = sorted(uniq_set)
                blocker_cnt = len(uniq_set)
                is_exact    = hit_exact[cid] or bool(uniq_set & _EXACT_B)

            best_score      = hit_best_score[cid]
            best_rank       = hit_best_rank[cid]
            provenance_str  = ",".join(uniq_sorted)
            # Source prefix: "S2-123" -> "S2", extracted without function call
            dash = cid.index("-")
            cand_source = cid[:dash]

            sort_key = (
                -blocker_cnt,
                -1 if is_exact else 0,
                -round(best_score, 4),
                best_rank,
                cid,
            )
            ranked_candidates.append((
                sort_key,
                CandidateHit((
                    s1_entity_id,           # 0: source1_entity_id
                    cid,                    # 1: candidate_entity_id
                    cand_source,            # 2: candidate_source
                    provenance_str,         # 3: blocker_provenance
                    blocker_cnt,            # 4: blocker_count
                    best_rank,              # 5: best_blocker_rank
                    round(best_score, 4),   # 6: best_blocker_score
                    s1_country,             # 7: country_partition
                )),
            ))

        ranked_candidates.sort(key=lambda item: item[0])
        return [item[1] for item in ranked_candidates[:max_candidates]]

    # ------------------------------------------------------------------
    # Phase 2: Index Serialization Cache
    # ------------------------------------------------------------------

    def save_index(self, path: Path, fingerprint: str = "") -> None:
        """Serialize the full inverted index to disk using pickle protocol 5."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_version":                   INDEX_CACHE_VERSION,
            "_fingerprint":               fingerprint,
            "num_candidates":             self.num_candidates,
            "candidate_ngram_lens":       self.candidate_ngram_lens,
            "country_partitions":         {k: list(v) for k, v in self.country_partitions.items()},
            "index_exact_name":           dict(self.index_exact_name),
            "index_first_2_tokens":       dict(self.index_first_2_tokens),
            "index_acronym":              dict(self.index_acronym),
            "index_name_postal":          dict(self.index_name_postal),
            "index_name_street":          dict(self.index_name_street),
            "index_name_trailing":        dict(self.index_name_trailing),
            "index_lead_postal":          dict(self.index_lead_postal),
            "index_lead_street_trailing": dict(self.index_lead_street_trailing),
            "index_ngrams":               dict(self.index_ngrams),
            "ngram_doc_counts":           dict(self.ngram_doc_counts),
            "index_tokens":               dict(self.index_tokens),
            "token_doc_counts":           dict(self.token_doc_counts),
            "index_address_structural":   dict(self.index_address_structural),
            "ngram_idf_table":            self.ngram_idf_table,
            "token_idf_table":            self.token_idf_table,
            "config": {
                "top_k_sparse":        self.top_k_sparse,
                "top_k_dense":         self.top_k_dense,
                "max_candidates":      self.max_candidates_per_entity,
                "similarity_floor":    self.similarity_floor,
                "max_ngram_doc_count": self.max_ngram_doc_count,
                "max_token_doc_count": self.max_token_doc_count,
                "max_ngram_doc_freq":  self.max_ngram_doc_freq,
                "max_token_doc_freq":  self.max_token_doc_freq,
                "_cached_max_ngram_count": self._cached_max_ngram_count,
                "_cached_max_token_count": self._cached_max_token_count,
            },
        }
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=5)
        tmp_path.replace(path)
        size_mb = path.stat().st_size / 1024 ** 2
        print(f"[CACHE] Index saved: {path} ({size_mb:.1f} MB)", flush=True)

    @classmethod
    def load_index(cls, path: Path) -> "MultiChannelBlocker":
        """Restore a serialized index from disk (see save_index)."""
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            parts = sorted(path.parent.glob(f"{path.name}.part*"))
            if parts:
                print(f"[CACHE] Reassembling {path.name} from {len(parts)} split parts...", flush=True)
                with open(path, "wb") as outfile:
                    for p in parts:
                        with open(p, "rb") as infile:
                            shutil.copyfileobj(infile, outfile)
        t0 = time.time()
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        ver = payload.get("_version")
        if ver != INDEX_CACHE_VERSION:
            raise ValueError(
                f"Index cache version mismatch (got {ver}, expected {INDEX_CACHE_VERSION}). "
                "Re-run with --no-cache-index to rebuild."
            )
        cfg = payload["config"]
        blocker = cls(
            top_k_sparse=cfg["top_k_sparse"],
            top_k_dense=cfg["top_k_dense"],
            max_candidates_per_entity=cfg["max_candidates"],
            similarity_floor=cfg["similarity_floor"],
            max_ngram_doc_count=cfg["max_ngram_doc_count"],
            max_token_doc_count=cfg["max_token_doc_count"],
            max_ngram_doc_freq=cfg["max_ngram_doc_freq"],
            max_token_doc_freq=cfg["max_token_doc_freq"],
        )
        blocker.num_candidates       = payload["num_candidates"]
        blocker.candidate_ngram_lens = payload["candidate_ngram_lens"]
        blocker.candidate_records    = blocker.candidate_ngram_lens
        blocker.country_partitions   = defaultdict(
            set, {k: set(v) for k, v in payload["country_partitions"].items()}
        )
        # Restore inverted indexes as defaultdict(lambda: defaultdict(list))
        _dd = lambda: defaultdict(list)  # noqa: E731
        for attr in (
            "index_exact_name", "index_first_2_tokens", "index_acronym",
            "index_name_postal", "index_name_street", "index_name_trailing",
            "index_lead_postal", "index_lead_street_trailing",
            "index_ngrams", "index_tokens", "index_address_structural",
        ):
            raw = payload[attr]
            setattr(blocker, attr, defaultdict(_dd, raw))
        # Aliases
        blocker.index_trigrams     = blocker.index_ngrams
        blocker.trigram_doc_counts = blocker.ngram_doc_counts
        blocker.ngram_doc_counts   = defaultdict(int, payload["ngram_doc_counts"])
        blocker.token_doc_counts   = defaultdict(int, payload["token_doc_counts"])
        
        # v4: precomputed IDF tables
        blocker.ngram_idf_table         = payload.get("ngram_idf_table", {})
        blocker.token_idf_table         = payload.get("token_idf_table", {})
        blocker._cached_max_ngram_count = cfg.get("_cached_max_ngram_count", 0)
        blocker._cached_max_token_count = cfg.get("_cached_max_token_count", 0)
        if not blocker.ngram_idf_table:
            blocker.build_idf_tables()
        elapsed = time.time() - t0
        print(
            f"[CACHE] Index loaded in {elapsed:.1f}s: "
            f"{blocker.num_candidates:,} candidates, "
            f"{len(blocker.country_partitions)} country partitions.",
            flush=True,
        )
        return blocker


class DenseRetrievalIndex:
    """FAISS / vector similarity retrieval index partitioned by country for Channel 7."""

    def __init__(
        self,
        embeddings_path: Path,
        country_partitions: Dict[str, Set[str]],
        similarity_floor: float = SIMILARITY_FLOOR,
    ):
        import numpy as np
        data = np.load(embeddings_path, allow_pickle=True)
        self.similarity_floor = similarity_floor
        self.candidate_ids = list(data["candidate_ids"])
        self.candidate_embs = data["candidate_embeddings"].astype(np.float32)
        # Normalize candidate embeddings
        norms = np.linalg.norm(self.candidate_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.candidate_embs /= norms

        cand_id_to_idx = {cid: idx for idx, cid in enumerate(self.candidate_ids)}

        # S1 embeddings lookup
        if "s1_ids" in data and "s1_embeddings" in data:
            s1_embs = data["s1_embeddings"].astype(np.float32)
            s1_norms = np.linalg.norm(s1_embs, axis=1, keepdims=True)
            s1_norms[s1_norms == 0] = 1.0
            s1_embs /= s1_norms
            self.s1_embeddings = {sid: s1_embs[i] for i, sid in enumerate(data["s1_ids"])}
        else:
            self.s1_embeddings = {}

        self.partition_indices: Dict[str, List[int]] = defaultdict(list)
        self.partition_embs: Dict[str, Any] = {}
        self.faiss_indexes: Dict[str, Any] = {}
        try:
            import faiss
            has_faiss = True
        except ImportError:
            has_faiss = False

        for country, cids in country_partitions.items():
            idxs = [cand_id_to_idx[cid] for cid in cids if cid in cand_id_to_idx]
            if not idxs:
                continue
            self.partition_indices[country] = idxs
            if has_faiss and len(idxs) > 0:
                dim = self.candidate_embs.shape[1]
                sub_embs = self.candidate_embs[idxs]
                index = faiss.IndexFlatIP(dim)
                index.add(sub_embs)
                self.faiss_indexes[country] = (index, idxs)
            elif len(idxs) > 0:
                self.partition_embs[country] = self.candidate_embs[idxs]

    def query(self, s1_id: str, country: str, top_k: int = 50) -> Dict[str, float]:
        emb = self.s1_embeddings.get(s1_id)
        if emb is None:
            return {}
        import numpy as np

        scores: Dict[str, float] = {}
        target_countries = [country] if country else []
        if "" in self.partition_indices and "" not in target_countries:
            target_countries.append("")

        for c in target_countries:
            if c in self.faiss_indexes:
                index, idxs = self.faiss_indexes[c]
                k = min(top_k, len(idxs))
                if k > 0:
                    distances, indices = index.search(emb.reshape(1, -1), k)
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx >= 0 and dist >= self.similarity_floor:
                            cid = self.candidate_ids[idxs[idx]]
                            scores[cid] = max(scores.get(cid, 0.0), float(dist))
            elif c in self.partition_indices:
                idxs = self.partition_indices[c]
                if idxs:
                    sub_embs = self.partition_embs.get(c)
                    if sub_embs is None:
                        sub_embs = self.candidate_embs[idxs]
                    dots = np.dot(sub_embs, emb)
                    top_order = np.argsort(-dots)[:top_k]
                    for rank_i in top_order:
                        score = float(dots[rank_i])
                        if score >= self.similarity_floor:
                            cid = self.candidate_ids[idxs[rank_i]]
                            scores[cid] = max(scores.get(cid, 0.0), score)

        # Soft fallback: if zero hits in primary partition (e.g. unknown country spelling),
        # query remaining compatible partitions with a soft 0.95 cross-country discount
        if not scores:
            other_countries = [c for c in self.partition_indices if c not in target_countries]
            for c in other_countries:
                if country in KNOWN_CANONICAL_COUNTRIES and c in KNOWN_CANONICAL_COUNTRIES:
                    continue
                if c in self.faiss_indexes:
                    index, idxs = self.faiss_indexes[c]
                    k = min(top_k, len(idxs))
                    if k > 0:
                        distances, indices = index.search(emb.reshape(1, -1), k)
                        for dist, idx in zip(distances[0], indices[0]):
                            if idx >= 0 and dist >= self.similarity_floor:
                                cid = self.candidate_ids[idxs[idx]]
                                scores[cid] = max(scores.get(cid, 0.0), float(dist) * 0.95)
                elif c in self.partition_indices:
                    idxs = self.partition_indices[c]
                    if idxs:
                        sub_embs = self.partition_embs.get(c)
                        if sub_embs is None:
                            sub_embs = self.candidate_embs[idxs]
                        dots = np.dot(sub_embs, emb)
                        top_order = np.argsort(-dots)[:top_k]
                        for rank_i in top_order:
                            score = float(dots[rank_i])
                            if score >= self.similarity_floor:
                                cid = self.candidate_ids[idxs[rank_i]]
                                scores[cid] = max(scores.get(cid, 0.0), score * 0.95)
        return scores


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """Load ground truth mapping: source1_entity_id -> set of matched_entity_ids."""
    gt: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            s1_id = row.get("source1_entity_id", "").strip()
            raw_matched = row.get("matched_entity_ids", "").strip()
            matched = {
                mid.strip() for mid in raw_matched.split(",") if mid.strip()
            }
            gt[s1_id] = matched
    return gt


# ---------------------------------------------------------------------------
# Phase 2: Source fingerprint helpers
# ---------------------------------------------------------------------------

def _candidate_source_fingerprint(paths: Sequence[Path]) -> str:
    """MD5 fingerprint of candidate source TSV paths (name + size)."""
    h = hashlib.md5()
    for p in sorted(paths):
        p = Path(p)
        try:
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}".encode())
        except OSError:
            h.update(p.name.encode())
    return h.hexdigest()[:16]


def _read_cache_fingerprint(cache_path: Path) -> str:
    """(Deprecated) Peek the fingerprint stored inside a pickle cache."""
    return ""


# ---------------------------------------------------------------------------
# Phase 3: Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_completed_s1_ids(pairs_path: Path) -> Tuple[Set[str], int, int]:
    """Scan existing candidate_pairs.tsv to collect completed S1 entity IDs, pairs count, and singletons."""
    completed: Set[str] = set()
    total_pairs = 0
    singletons = 0
    if not pairs_path.exists():
        return completed, total_pairs, singletons
    with open(pairs_path, "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()  # skip header
        for line in fh:
            tab = line.find("\t")
            if tab > 0:
                s1 = line[:tab].strip()
                completed.add(s1)
                cands = line[tab + 1:].strip()
                if cands:
                    total_pairs += len(cands.split(","))
                else:
                    singletons += 1
    return completed, total_pairs, singletons


def _write_checkpoint(checkpoint_path: Path, meta: Dict[str, Any]) -> None:
    """Atomically write checkpoint.json via a temp file."""
    tmp = checkpoint_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    tmp.replace(checkpoint_path)


# ---------------------------------------------------------------------------
# Phase 4: Multiprocessing worker (must be top-level for Windows spawn)
# ---------------------------------------------------------------------------

def _query_worker(
    chunk_rows: List[Dict[str, Any]],
    cache_path: str,
    top_k_sparse: int,
    top_k_dense: int,
    max_candidates: int,
    similarity_floor: float,
    out_pairs_path: str,
    out_prov_path: str,
    worker_id: int,
) -> Dict[str, int]:
    """
    Top-level picklable worker function for ProcessPoolExecutor.
    Loads the index from cache (if not inherited via fork), processes its chunk, and writes partial output files.
    """
    global _shared_blocker
    if _shared_blocker is not None:
        blocker = _shared_blocker
    else:
        blocker = MultiChannelBlocker.load_index(Path(cache_path))
    pairs_count = 0
    singletons  = 0
    prov_fields = [
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "blocker_provenance", "blocker_count", "best_blocker_rank",
        "best_blocker_score", "country_partition",
    ]
    with open(out_pairs_path, "w", encoding="utf-8", newline="") as pf, \
         open(out_prov_path, "w", encoding="utf-8", newline="") as pvf:
         
        # Batch prepare records upfront so pure processing isn't interleaved with parsing
        parsed_records = []
        for row in chunk_rows:
            parsed_records.append(
                BlockingRecord.from_row(
                    entity_id=row["entity_id"],
                    name=row["business_name"],
                    address=row["business_address"],
                    country=row["country"],
                    norm_name=row.get("norm_name"),
                    norm_address=row.get("norm_address"),
                    canonical_country=row.get("canonical_country"),
                    postal_code=row.get("postal_code"),
                    street_number=row.get("street_number"),
                    trailing_segment=row.get("trailing_segment"),
                    is_address_missing=row.get("is_address_missing"),
                )
            )

        pvf.write("\t".join(prov_fields) + "\n")
        
        out_buf_pairs = io.StringIO()
        out_buf_prov = io.StringIO()
        pw_buf = csv.writer(out_buf_pairs, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        dw_buf = csv.writer(out_buf_prov, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")

        for i, s1_rec in enumerate(parsed_records):
            candidates = blocker.generate_candidates_for_record(s1_rec)
            if not candidates:
                singletons += 1
                pw_buf.writerow([s1_rec.entity_id, ""])
            else:
                pairs_count += len(candidates)
                pw_buf.writerow([s1_rec.entity_id, ",".join(c[1] for c in candidates)])
                for c in candidates:
                    dw_buf.writerow(c)
                    
            if (i + 1) % 100_000 == 0:
                pf.write(out_buf_pairs.getvalue())
                pvf.write(out_buf_prov.getvalue())
                out_buf_pairs.seek(0)
                out_buf_pairs.truncate(0)
                out_buf_prov.seek(0)
                out_buf_prov.truncate(0)
                
        # Final flush
        pf.write(out_buf_pairs.getvalue())
        pvf.write(out_buf_prov.getvalue())
    print(f"[WORKER {worker_id}] done: {pairs_count:,} pairs, {singletons:,} singletons", flush=True)
    return {"pairs": pairs_count, "singletons": singletons}


def run_blocking(
    source1_paths: Sequence[Path],
    candidate_sources: Sequence[Path],
    output_dir: Path,
    ground_truth_path: Optional[Path] = None,
    dense_embeddings_path: Optional[Path] = None,
    top_k_sparse: int = TOP_K_SPARSE_OR_CHAR,
    top_k_dense: int = TOP_K_DENSE,
    max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
    similarity_floor: float = SIMILARITY_FLOOR,
    cache_index: bool = True,
    resume: bool = True,
    num_workers: int = 1,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> Dict[str, Any]:
    """
    Run end-to-end Stage 1 blocking on input TSVs and emit:
      - candidate_pairs.tsv
      - candidate_provenance.tsv
      - blocking_summary.json

    Upgrades (Phases 2-4):
      - cache_index: save/load inverted index as pickle (avoids 15-min rebuild)
      - resume: skip already-processed S1 entities (checkpoint/resume)
      - num_workers: run query phase in parallel with ProcessPoolExecutor
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 2: Build or load the candidate index
    # -----------------------------------------------------------------------
    cache_path   = output_dir / INDEX_CACHE_NAME
    fingerprint  = _candidate_source_fingerprint(candidate_sources)
    blocker: Optional[MultiChannelBlocker] = None

    if cache_index and cache_path.exists():
        try:
            print(f"[CACHE] Loading index from {cache_path}...", flush=True)
            loaded_blocker = MultiChannelBlocker.load_index(cache_path)
            
            cached_fp = getattr(loaded_blocker, "_fingerprint", "")
            if cached_fp == fingerprint:
                print(f"[CACHE] Fingerprint match! Index loaded successfully.", flush=True)
                blocker = loaded_blocker
            else:
                print(f"[CACHE] Fingerprint mismatch (cached={cached_fp} current={fingerprint}). Discarding cache and rebuilding.", flush=True)
        except Exception as exc:
            print(f"[CACHE] Load failed ({exc}) — rebuilding index.", flush=True)

    if blocker is None:
        blocker = MultiChannelBlocker(
            top_k_sparse=top_k_sparse,
            top_k_dense=top_k_dense,
            max_candidates_per_entity=max_candidates_per_entity,
            similarity_floor=similarity_floor,
        )
        # 1. Index all candidate records (S2, S3)
        print(f"Indexing candidate records from {[str(p) for p in candidate_sources]}...", flush=True)
        cand_count = 0
        for row in read_tsv_records(candidate_sources):
            rec = BlockingRecord.from_row(
                entity_id=row["entity_id"],
                name=row["business_name"],
                address=row["business_address"],
                country=row["country"],
                norm_name=row.get("norm_name"),
                norm_address=row.get("norm_address"),
                canonical_country=row.get("canonical_country"),
                postal_code=row.get("postal_code"),
                street_number=row.get("street_number"),
                trailing_segment=row.get("trailing_segment"),
                is_address_missing=row.get("is_address_missing"),
            )
            blocker.index_candidate(rec)
            cand_count += 1
            if cand_count % 100_000 == 0:
                print(f"  Indexed {cand_count:,} candidate records...", flush=True)
        print(
            f"Finished indexing {cand_count:,} candidate records across "
            f"{len(blocker.country_partitions)} country partitions.",
            flush=True,
        )
        print("Precomputing IDF lookup tables...", flush=True)
        blocker.build_idf_tables()
        
        if cache_index:
            blocker.save_index(cache_path, fingerprint=fingerprint)

    # 1b. Initialize Dense Retrieval Index if provided
    dense_index: Optional[DenseRetrievalIndex] = None
    if dense_embeddings_path and Path(dense_embeddings_path).exists():
        print(f"Loading dense embeddings and creating FAISS index from {dense_embeddings_path}...", flush=True)
        dense_index = DenseRetrievalIndex(
            embeddings_path=Path(dense_embeddings_path),
            country_partitions=blocker.country_partitions,
            similarity_floor=similarity_floor,
        )
        print(f"  Dense index initialized with {len(dense_index.candidate_ids):,} candidates.", flush=True)

    # -----------------------------------------------------------------------
    # Phase 3: Checkpoint / Resume
    # -----------------------------------------------------------------------
    candidate_pairs_path = output_dir / "candidate_pairs.tsv"
    provenance_path      = output_dir / "candidate_provenance.tsv"
    checkpoint_path      = output_dir / "checkpoint.json"

    completed_s1: Set[str] = set()
    initial_pairs = 0
    initial_singletons = 0
    if resume:
        completed_s1, initial_pairs, initial_singletons = _load_completed_s1_ids(candidate_pairs_path)
        if completed_s1:
            print(
                f"[RESUME] Skipping {len(completed_s1):,} already-completed S1 entities "
                f"({initial_pairs:,} pairs, {initial_singletons:,} singletons).",
                flush=True,
            )

    resume_mode = len(completed_s1) > 0
    file_mode   = "a" if resume_mode else "w"

    # Audit tracking (initialized before any path branches)
    ground_truth: Optional[Dict[str, Set[str]]] = None
    gt_total_pairs    = 0
    gt_recovered_pairs = 0
    gt_entity_full_hit = 0
    gt_entity_any_hit  = 0
    gt_by_channel: Counter[str]  = Counter()
    gt_by_country: Counter[str]  = Counter()
    gt_country_totals: Counter[str] = Counter()

    if ground_truth_path and Path(ground_truth_path).exists():
        print(f"Loading ground truth for recall audit from {ground_truth_path}...", flush=True)
        ground_truth = load_ground_truth(Path(ground_truth_path))
        for mid_set in ground_truth.values():
            gt_total_pairs += len(mid_set)

    s1_count        = 0
    total_pairs     = initial_pairs
    singletons      = initial_singletons
    candidate_counts: List[int] = []
    channel_counts: Counter[str] = Counter()

    # -----------------------------------------------------------------------
    # Phase 4: Multiprocessing dispatch vs single-threaded fallback
    # -----------------------------------------------------------------------
    if num_workers > 1:
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            # With 'fork' context (Linux) or ThreadPoolExecutor (Windows), memory is shared.
            # We only need ~14GB base + ~0.5GB overhead per worker.
            min_needed_gb = 14.0 + (num_workers * 0.5)
            
            if total_ram_gb < min_needed_gb:
                print(
                    f"[WARNING] Parallel query mode with {num_workers} workers requires ~{min_needed_gb:.1f} GB RAM, "
                    f"but system has {total_ram_gb:.1f} GB total RAM. "
                    f"Reverting to robust single-process streaming mode to prevent OOM crash.",
                    flush=True,
                )
                num_workers = 1
        except Exception:
            pass

    if num_workers > 1:
        # Read all remaining S1 rows into memory once
        print(f"[PARALLEL] Loading remaining S1 rows into memory...", flush=True)
        all_s1_rows = [
            row for row in read_tsv_records(source1_paths)
            if row["entity_id"] not in completed_s1
        ]
        n_remaining = len(all_s1_rows)
        print(f"[PARALLEL] {n_remaining:,} S1 entities to process across {num_workers} workers.", flush=True)

        if n_remaining == 0:
            print("[PARALLEL] All S1 entities already complete — nothing to do.", flush=True)
        else:
            chunk_size = math.ceil(n_remaining / num_workers)
            chunks = [
                all_s1_rows[i : i + chunk_size]
                for i in range(0, n_remaining, chunk_size)
            ]

            tmp_dir = output_dir / "_tmp_worker_chunks"
            tmp_dir.mkdir(exist_ok=True)

            futures_map: Dict[Any, int] = {}
            global _shared_blocker
            _shared_blocker = blocker
            
            try:
                ctx = multiprocessing.get_context("fork")
                ExecutorClass = concurrent.futures.ProcessPoolExecutor
                kwargs = {"mp_context": ctx}
            except ValueError:
                ctx = None
                ExecutorClass = concurrent.futures.ThreadPoolExecutor
                kwargs = {}
                print("[INFO] Windows/spawn environment detected. Using ThreadPoolExecutor to share RAM safely.", flush=True)

            with ExecutorClass(max_workers=num_workers, **kwargs) as pool:
                for i, chunk in enumerate(chunks):
                    pp  = str(tmp_dir / f"pairs_{i}.tsv")
                    pvp = str(tmp_dir / f"prov_{i}.tsv")
                    fut = pool.submit(
                        _query_worker, chunk, str(cache_path),
                        top_k_sparse, top_k_dense, max_candidates_per_entity,
                        similarity_floor, pp, pvp, i,
                    )
                    futures_map[fut] = i

                for fut in concurrent.futures.as_completed(futures_map):
                    wi = futures_map[fut]
                    try:
                        res = fut.result()
                        total_pairs += res["pairs"]
                        singletons  += res["singletons"]
                        print(f"[MAIN] Worker {wi} finished: {res}", flush=True)
                    except Exception as exc:
                        print(f"[MAIN] Worker {wi} FAILED: {exc}", flush=True)
                        raise

            # Merge partial outputs in deterministic chunk order
            prov_fields = [
                "source1_entity_id", "candidate_entity_id", "candidate_source",
                "blocker_provenance", "blocker_count", "best_blocker_rank",
                "best_blocker_score", "country_partition",
            ]
            with open(candidate_pairs_path, file_mode, encoding="utf-8", newline="") as pf, \
                 open(provenance_path,       file_mode, encoding="utf-8", newline="") as pvf:
                if not resume_mode:
                    pf.write("source1_entity_id\tcandidate_entity_ids\n")
                    pvf.write("\t".join(prov_fields) + "\n")
                for i in range(len(chunks)):
                    pp  = tmp_dir / f"pairs_{i}.tsv"
                    pvp = tmp_dir / f"prov_{i}.tsv"
                    if pp.exists():
                        pf.write(pp.read_text(encoding="utf-8"))
                    if pvp.exists():
                        lines = pvp.read_text(encoding="utf-8").splitlines(keepends=True)
                        pvf.writelines(lines[1:])  # skip per-worker header

            shutil.rmtree(tmp_dir, ignore_errors=True)
            s1_count = n_remaining
            candidate_counts = [0] * n_remaining  # approximate; detailed stats not tracked in MP mode

    else:
        # Single-threaded path (default, ground-truth-audit capable)
        with open(candidate_pairs_path, file_mode, encoding="utf-8", newline="") as pairs_file, \
             open(provenance_path,      file_mode, encoding="utf-8", newline="") as prov_file:

            pairs_writer = csv.writer(pairs_file, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            prov_writer  = csv.writer(prov_file, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            if not resume_mode:
                pairs_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
                prov_writer.writerow([
                    "source1_entity_id", "candidate_entity_id", "candidate_source",
                    "blocker_provenance", "blocker_count", "best_blocker_rank",
                    "best_blocker_score", "country_partition",
                ])

            print(f"Generating candidate pairs for S1 records from {[str(p) for p in source1_paths]}...", flush=True)
            for row in read_tsv_records(source1_paths):
                # Phase 3: skip already-completed entities
                if row["entity_id"] in completed_s1:
                    continue

                s1_count += 1
                s1_rec = BlockingRecord.from_row(
                    entity_id=row["entity_id"],
                    name=row["business_name"],
                    address=row["business_address"],
                    country=row["country"],
                    norm_name=row.get("norm_name"),
                    norm_address=row.get("norm_address"),
                    canonical_country=row.get("canonical_country"),
                    postal_code=row.get("postal_code"),
                    street_number=row.get("street_number"),
                    trailing_segment=row.get("trailing_segment"),
                    is_address_missing=row.get("is_address_missing"),
                )

                dense_scores = (
                    dense_index.query(s1_rec.entity_id, s1_rec.canonical_country, top_k=blocker.top_k_dense)
                    if dense_index else None
                )
                candidates = blocker.generate_candidates_for_record(s1_rec, dense_scores=dense_scores)
                candidate_counts.append(len(candidates))

                if not candidates:
                    singletons += 1
                    pairs_writer.writerow([s1_rec.entity_id, ""])
                else:
                    total_pairs += len(candidates)
                    cand_ids = [c[1] for c in candidates]
                    pairs_writer.writerow([s1_rec.entity_id, ",".join(cand_ids)])
                    for cand_info in candidates:
                        prov_writer.writerow(cand_info)
                        for blk in cand_info[3].split(","):
                            channel_counts[blk] += 1

                # Recall audit
                if ground_truth and s1_rec.entity_id in ground_truth:
                    true_matches = ground_truth[s1_rec.entity_id]
                    c_country = s1_rec.canonical_country or "unknown"
                    gt_country_totals[c_country] += len(true_matches)
                    found_cands = {c[1] for c in candidates}
                    gt_hits = true_matches & found_cands
                    gt_recovered_pairs += len(gt_hits)
                    if gt_hits:
                        gt_entity_any_hit += 1
                    if gt_hits == true_matches and len(true_matches) > 0:
                        gt_entity_full_hit += 1
                    for c in candidates:
                        cid = c[1]
                        if cid in true_matches:
                            gt_by_country[c_country] += 1
                            for blk in c[3].split(","):
                                gt_by_channel[blk] += 1

                # Progress + checkpoint flush
                if s1_count % 25_000 == 0:
                    print(
                        f"  Processed {s1_count:,} new S1 entities "
                        f"(+{len(completed_s1):,} resumed) -> {total_pairs:,} total pairs...",
                        flush=True,
                    )
                if s1_count % checkpoint_interval == 0:
                    pairs_file.flush()
                    prov_file.flush()
                    _write_checkpoint(checkpoint_path, {
                        "completed_s1_count": s1_count + len(completed_s1),
                        "total_candidate_pairs": total_pairs,
                        "last_s1_entity_id": s1_rec.entity_id,
                        "last_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "source_fingerprint": fingerprint,
                    })

    candidate_counts.sort()
    n_s1 = len(candidate_counts)
    total_s1 = s1_count + len(completed_s1)
    mean_cands = total_pairs / total_s1 if total_s1 else 0.0
    median_cands = candidate_counts[n_s1 // 2] if n_s1 else 0
    p90_cands = candidate_counts[int(n_s1 * 0.90)] if n_s1 else 0
    p95_cands = candidate_counts[int(n_s1 * 0.95)] if n_s1 else 0
    max_cands = candidate_counts[-1] if n_s1 else 0

    summary: Dict[str, Any] = {
        "s1_count": total_s1,
        "total_source1_entities": total_s1,
        "total_candidate_pairs": total_pairs,
        "singletons_with_zero_candidates": singletons,
        "singleton_s1_count": singletons,
        "candidates_per_s1": {
            "mean": round(mean_cands, 2),
            "median": median_cands,
            "p90": p90_cands,
            "p95": p95_cands,
            "max": max_cands,
        },
        "mean_candidates_per_s1": round(mean_cands, 2),
        "median_candidates_per_s1": median_cands,
        "p90_candidates_per_s1": p90_cands,
        "p95_candidates_per_s1": p95_cands,
        "max_candidates_per_s1": max_cands,
        "channel_contributions": dict(channel_counts),
        "configuration": {
            "top_k_sparse": top_k_sparse,
            "top_k_dense": top_k_dense,
            "max_candidates_per_entity": max_candidates_per_entity,
            "similarity_floor": similarity_floor,
        },
    }

    if ground_truth:
        evaluated_s1 = sum(1 for m in ground_truth.values() if len(m) > 0)
        pair_recall = gt_recovered_pairs / gt_total_pairs if gt_total_pairs else 0.0
        entity_recall = gt_entity_full_hit / evaluated_s1 if evaluated_s1 else 0.0
        any_hit_rate = gt_entity_any_hit / evaluated_s1 if evaluated_s1 else 0.0

        country_recalls = {}
        for c, total_c in gt_country_totals.items():
            hit_c = gt_by_country.get(c, 0)
            country_recalls[c] = round(hit_c / total_c, 4) if total_c else 0.0

        channel_recalls = {}
        for blk, hit_cnt in gt_by_channel.items():
            channel_recalls[blk] = round(hit_cnt / gt_total_pairs, 4) if gt_total_pairs else 0.0

        summary["recall_audit"] = {
            "total_ground_truth_pairs": gt_total_pairs,
            "recovered_ground_truth_pairs": gt_recovered_pairs,
            "pair_recall": round(pair_recall, 4),
            "entity_full_recall": round(entity_recall, 4),
            "any_hit_rate": round(any_hit_rate, 4),
            "recall_by_country": country_recalls,
            "recall_by_channel": channel_recalls,
        }

    summary_path = output_dir / "blocking_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 70)
    print("STAGE 1 BLOCKING COMPLETE")
    print(f"S1 Entities: {n_s1:,} | Total Pairs: {total_pairs:,} | Singletons: {singletons:,}")
    print(f"Cands/S1 -> Mean: {mean_cands:.1f}, Median: {median_cands}, P90: {p90_cands}, P95: {p95_cands}, Max: {max_cands}")
    if ground_truth:
        audit = summary["recall_audit"]
        print(f"Recall Audit -> Pair Recall: {audit['pair_recall']:.2%}, "
              f"Entity Recall: {audit['entity_full_recall']:.2%}, "
              f"Any-Hit Rate: {audit['any_hit_rate']:.2%}")
    print("=" * 70)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Multi-Channel Candidate Generation")
    parser.add_argument(
        "--stage0_dir", "--stage0-dir",
        type=Path,
        default=None,
        help="Directory containing Stage 0 normalized TSV files (e.g. dataset/stage0_normalized)",
    )
    parser.add_argument(
        "--source1",
        type=Path,
        nargs="+",
        default=None,
        help="Path(s) to Source 1 TSV(s)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        default=None,
        help="Path(s) to Candidate Sources (S2, S3) TSV(s)",
    )
    parser.add_argument(
        "--output_dir", "--output-dir",
        type=Path,
        required=True,
        help="Directory to write candidate_pairs.tsv and provenance",
    )
    parser.add_argument(
        "--ground_truth", "--ground-truth",
        type=Path,
        default=None,
        help="Optional path to ground truth TSV for recall audit gate",
    )
    parser.add_argument(
        "--max_candidates", "--max-candidates",
        type=int,
        default=MAX_CANDIDATES_PER_ENTITY,
        help=f"Max candidates per S1 entity (default: {MAX_CANDIDATES_PER_ENTITY})",
    )
    parser.add_argument(
        "--top_k", "--top-k",
        type=int,
        default=None,
        help="Top-K for all channels (convenience flag setting both top-k-sparse and top-k-dense)",
    )
    parser.add_argument(
        "--top_k_sparse", "--top-k-sparse",
        type=int,
        default=TOP_K_SPARSE_OR_CHAR,
        help=f"Top-K for sparse/char/token channels (default: {TOP_K_SPARSE_OR_CHAR})",
    )
    parser.add_argument(
        "--top_k_dense", "--top-k-dense",
        type=int,
        default=TOP_K_DENSE,
        help=f"Top-K for dense channel (default: {TOP_K_DENSE})",
    )
    parser.add_argument(
        "--dense_embeddings", "--dense-embeddings",
        type=Path,
        default=None,
        help="Optional path to .npz file containing precomputed dense embeddings for Channel 5",
    )
    parser.add_argument(
        "--similarity_floor", "--similarity-floor",
        type=float,
        default=SIMILARITY_FLOOR,
        help=f"Minimum cosine similarity floor for candidate acceptance (default: {SIMILARITY_FLOOR})",
    )
    # Phase 2–4: new flags
    parser.add_argument(
        "--cache_index", "--cache-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save/load candidate inverted index as pickle cache (default: on). Use --no-cache-index to force rebuild.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output, skipping already-completed S1 entities (default: on). Use --no-resume to restart.",
    )
    parser.add_argument(
        "--num_workers", "--num-workers",
        type=int,
        default=1,
        help="Number of parallel ProcessPoolExecutor workers for the query phase (default: 1 = single-threaded).",
    )
    parser.add_argument(
        "--checkpoint_interval", "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help=f"Flush output and write checkpoint.json every N entities (default: {CHECKPOINT_INTERVAL}).",
    )
    args = parser.parse_args()

    source1_paths = args.source1
    candidate_paths = args.candidates

    if args.stage0_dir and (not source1_paths or not candidate_paths):
        stage0 = Path(args.stage0_dir)
        if not stage0.exists():
            raise FileNotFoundError(f"Stage 0 directory not found: {stage0}")
        all_tsvs = list(stage0.rglob("*.tsv"))
        s1_found = sorted([p for p in all_tsvs if "source1" in p.name.lower()])
        s23_found = sorted([p for p in all_tsvs if ("source2" in p.name.lower() or "source3" in p.name.lower())])
        if not s1_found:
            raise FileNotFoundError(f"No source1 TSV files found in {stage0}")
        if not s23_found:
            raise FileNotFoundError(f"No source2 or source3 TSV files found in {stage0}")
        source1_paths = source1_paths or s1_found
        candidate_paths = candidate_paths or s23_found

    if not source1_paths:
        parser.error("Either --source1 or --stage0_dir containing source1 files is required.")
    if not candidate_paths:
        parser.error("Either --candidates or --stage0_dir containing source2/source3 files is required.")

    top_k_sparse = args.top_k if args.top_k is not None else args.top_k_sparse
    top_k_dense  = args.top_k if args.top_k is not None else args.top_k_dense

    run_blocking(
        source1_paths=source1_paths,
        candidate_sources=candidate_paths,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        dense_embeddings_path=args.dense_embeddings,
        top_k_sparse=top_k_sparse,
        top_k_dense=top_k_dense,
        max_candidates_per_entity=args.max_candidates,
        similarity_floor=args.similarity_floor,
        cache_index=args.cache_index,
        resume=args.resume,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
    )
    return 0


if __name__ == "__main__":
    main()
