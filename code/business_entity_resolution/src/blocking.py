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
import csv
import json
import math
import re
import sys
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
MAX_NGRAM_DOC_COUNT = 50000
MIN_TOKEN_LEN = 3

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
    for n in (3, 4):
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
        raw_words = [w.lower() for w in _WORD_RE.findall(n_name or "")]
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
        self.candidate_records: Dict[str, BlockingRecord] = {}

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

    def index_candidate(self, record: BlockingRecord) -> None:
        """Add a candidate record (from S2 or S3) to all blocking indexes."""
        cid = record.entity_id
        country = record.canonical_country
        self.candidate_records[cid] = record
        self.num_candidates += 1
        self.all_candidate_ids.add(cid)
        self.country_partitions[country].add(cid)

        # 1. Exact & Structural Name Keys
        if record.norm_name:
            self.index_exact_name[record.norm_name][country].append(cid)
        if record.first_2_tokens:
            self.index_first_2_tokens[record.first_2_tokens][country].append(cid)
        for acr in record.acronyms:
            self.index_acronym[acr][country].append(cid)

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

        # 2. Character 3/4-Gram Inverted Index
        for gram in record.ngram_counts:
            self.index_ngrams[gram][country].append(cid)
            self.ngram_doc_counts[gram] += 1

        # 3. Token Inverted Index
        for tok in set(record.tokens):
            self.index_tokens[tok][country].append(cid)
            self.token_doc_counts[tok] += 1

        # 4. Address Structural Key (postal + street number, bypassed if missing address)
        if not record.is_address_missing and record.postal_code and record.street_number:
            self.index_address_structural[(record.postal_code, record.street_number)][country].append(cid)

    def _query_partitioned_index(
        self,
        index: Dict[Any, Dict[str, List[str]]],
        key: Any,
        s1_country: str,
        allow_cross_country_fallback: bool = True,
    ) -> List[str]:
        """
        Query an index respecting the country partitioning invariant:
        - If s1_country != "": matching partition + records with missing country ("")
        - If matching partition has 0 hits and allow_cross_country_fallback is True:
          softly search across all other partitions to recover spelling variants / cross-country matches!
        - If s1_country == "": global search across all partitions
        """
        sub = index.get(key)
        if not sub:
            return []

        if s1_country:
            hits = list(sub.get(s1_country, []))
            if "" in sub:
                hits.extend(sub[""])
            if not hits and allow_cross_country_fallback:
                # Soft fallback: if exact partition has NO hits for this key,
                # search across all other compatible partitions.
                # Invariant: Disjoint known markets (US vs India) are never crossed.
                # Open-set and unrecognized country labels are searched to recover typos and ISO3 variants.
                for c, partition_list in sub.items():
                    if c == s1_country or c == "":
                        continue
                    if s1_country in KNOWN_CANONICAL_COUNTRIES and c in KNOWN_CANONICAL_COUNTRIES:
                        continue
                    hits.extend(partition_list)
            return hits
        else:
            # S1 country is missing -> global fallback
            all_hits = []
            for partition_list in sub.values():
                all_hits.extend(partition_list)
            return all_hits

    def generate_candidates_for_record(
        self,
        s1: BlockingRecord,
        dense_scores: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate, union, score, and rank candidates for a single Source 1 record.
        Returns a list of dicts with keys:
          candidate_id, blocker_provenance, blocker_count, best_blocker_rank,
          best_blocker_score, country_partition
        """
        # Mapping: candidate_id -> {
        #   "blockers": set(),
        #   "best_score": float,
        #   "best_rank": int,
        #   "exact_match": bool,
        # }
        hits: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "blockers": set(),
                "best_score": 0.0,
                "best_rank": 999999,
                "exact_match": False,
            }
        )

        def record_hit(cand_id: str, blocker: str, score: float, rank: int, is_exact: bool = False):
            info = hits[cand_id]
            info["blockers"].add(blocker)
            if score > info["best_score"]:
                info["best_score"] = score
            if rank < info["best_rank"]:
                info["best_rank"] = rank
            if is_exact:
                info["exact_match"] = True

        country = s1.canonical_country

        # -------------------------------------------------------------
        # Channel 1: Exact Name, First-2-Tokens, Acronym & Composite Keys
        # -------------------------------------------------------------
        if s1.norm_name:
            exact_hits = self._query_partitioned_index(self.index_exact_name, s1.norm_name, country)
            for rank, cid in enumerate(exact_hits, 1):
                record_hit(cid, "exact_name", 1.0, rank, is_exact=True)

        if s1.first_2_tokens:
            for rank, cid in enumerate(self._query_partitioned_index(self.index_first_2_tokens, s1.first_2_tokens, country), 1):
                record_hit(cid, "first_2_tokens", 0.90, rank, is_exact=True)

        for acr in s1.acronyms:
            for rank, cid in enumerate(self._query_partitioned_index(self.index_acronym, acr, country), 1):
                record_hit(cid, "acronym_match", 0.85, rank, is_exact=True)

        if s1.norm_name and s1.postal_code:
            key = (s1.norm_name, s1.postal_code)
            for rank, cid in enumerate(self._query_partitioned_index(self.index_name_postal, key, country), 1):
                record_hit(cid, "exact_name_postal", 1.0, rank, is_exact=True)

        if s1.norm_name and s1.street_number:
            key = (s1.norm_name, s1.street_number)
            for rank, cid in enumerate(self._query_partitioned_index(self.index_name_street, key, country), 1):
                record_hit(cid, "exact_name_street", 0.95, rank, is_exact=True)

        if s1.norm_name and s1.trailing_segment:
            key = (s1.norm_name, s1.trailing_segment)
            for rank, cid in enumerate(self._query_partitioned_index(self.index_name_trailing, key, country), 1):
                record_hit(cid, "exact_name_trailing", 0.95, rank, is_exact=True)

        if s1.first_word and s1.postal_code and len(s1.first_word) >= 3:
            key = (s1.first_word, s1.postal_code)
            for rank, cid in enumerate(self._query_partitioned_index(self.index_lead_postal, key, country), 1):
                record_hit(cid, "name_lead_postal", 0.90, rank, is_exact=True)

        if s1.first_word and s1.street_number and s1.trailing_segment:
            key = (s1.first_word, s1.street_number, s1.trailing_segment)
            for rank, cid in enumerate(self._query_partitioned_index(self.index_lead_street_trailing, key, country), 1):
                record_hit(cid, "name_lead_street_trailing", 0.90, rank, is_exact=True)

        # -------------------------------------------------------------
        # Channel 2: Character 3/4-Gram Sub-Linear TF-IDF Retrieval
        # -------------------------------------------------------------
        if s1.ngram_counts:
            s1_gram_counts = s1.ngram_counts
            max_allowed_ngram_freq = max(50, int(self.num_candidates * self.max_ngram_doc_freq))
            max_allowed_ngram_count = min(self.max_ngram_doc_count, max_allowed_ngram_freq)

            s1_weights: Dict[str, float] = {}
            for gram, count in s1_gram_counts.items():
                doc_cnt = self.ngram_doc_counts.get(gram, 0)
                if doc_cnt == 0 or doc_cnt > max_allowed_ngram_count:
                    continue
                tf = 1.0 + math.log(count)
                idf = math.log(1.0 + (self.num_candidates - doc_cnt + 0.5) / (doc_cnt + 0.5))
                s1_weights[gram] = tf * idf

            if s1_weights:
                cand_scores: Counter[str] = Counter()
                for gram, w_s1 in s1_weights.items():
                    for cid in self._query_partitioned_index(self.index_ngrams, gram, country):
                        cand_rec = self.candidate_records.get(cid)
                        if not cand_rec:
                            continue
                        c_cnt = cand_rec.ngram_counts.get(gram, 1)
                        c_tf = 1.0 + math.log(c_cnt)
                        cand_scores[cid] += w_s1 * c_tf

                s1_norm = math.sqrt(sum(w * w for w in s1_weights.values()))
                if cand_scores and s1_norm > 0:
                    scored_cands = []
                    for cid, raw_score in cand_scores.most_common(self.top_k_sparse * 3):
                        cand_rec = self.candidate_records[cid]
                        cand_len = sum(cand_rec.ngram_counts.values())
                        norm_factor = s1_norm * math.sqrt(max(1, cand_len))
                        sim = min(1.0, raw_score / norm_factor)
                        if sim >= self.similarity_floor:
                            scored_cands.append((cid, sim))

                    scored_cands.sort(key=lambda x: -x[1])
                    for rank, (cid, sim) in enumerate(scored_cands[: self.top_k_sparse], 1):
                        record_hit(cid, "char_ngram", float(sim), rank, is_exact=False)

        # -------------------------------------------------------------
        # Channel 3: Token Inverted Index with Sub-Linear TF-IDF
        # -------------------------------------------------------------
        if s1.tokens:
            token_scores: Counter[str] = Counter()
            s1_token_counts = s1.token_counts
            max_allowed_freq = max(10, int(self.num_candidates * self.max_token_doc_freq))
            max_allowed_count = min(self.max_token_doc_count, max_allowed_freq)

            s1_token_weights: Dict[str, float] = {}
            for tok, count in s1_token_counts.items():
                doc_cnt = self.token_doc_counts.get(tok, 0)
                if doc_cnt == 0 or doc_cnt > max_allowed_count:
                    continue
                tf = 1.0 + math.log(count)
                idf = math.log(1.0 + (self.num_candidates - doc_cnt + 0.5) / (doc_cnt + 0.5))
                s1_token_weights[tok] = tf * idf

            if s1_token_weights:
                for tok, w_s1 in s1_token_weights.items():
                    cands = self._query_partitioned_index(self.index_tokens, tok, country)
                    for cid in cands:
                        cand_rec = self.candidate_records.get(cid)
                        c_tf = (1.0 + math.log(cand_rec.token_counts.get(tok, 1))) if cand_rec else 1.0
                        token_scores[cid] += w_s1 * c_tf

                total_s1_weight = sum(s1_token_weights.values())
                if token_scores and total_s1_weight > 0.0:
                    top_tokens = token_scores.most_common(self.top_k_sparse)
                    for rank, (cid, score_sum) in enumerate(top_tokens, 1):
                        normalized_score = min(1.0, score_sum / total_s1_weight)
                        record_hit(cid, "token_inverted", normalized_score, rank, is_exact=False)

        # -------------------------------------------------------------
        # Channel 4: Address Structural Key (postal + street number)
        # -------------------------------------------------------------
        if not s1.is_address_missing and s1.postal_code and s1.street_number:
            key = (s1.postal_code, s1.street_number)
            for rank, cid in enumerate(self._query_partitioned_index(self.index_address_structural, key, country), 1):
                cand = self.candidate_records.get(cid)
                if cand and not cand.is_address_missing:
                    record_hit(cid, "address_structural", 0.85, rank, is_exact=False)

        # -------------------------------------------------------------
        # Channel 5: Dense Retrieval Hook (if provided)
        # -------------------------------------------------------------
        if dense_scores:
            dense_sorted = sorted(
                (
                    (cid, score)
                    for cid, score in dense_scores.items()
                    if score >= self.similarity_floor
                ),
                key=lambda x: -x[1],
            )
            for rank, (cid, score) in enumerate(dense_sorted[: self.top_k_dense], 1):
                record_hit(cid, "dense_bge", float(score), rank, is_exact=False)

        # -------------------------------------------------------------
        # Union, Deterministic Priority Ranking & Capping
        # -------------------------------------------------------------
        # Deterministic combined priority (from docs/04_stage1_blocking.md §4.1):
        # Tier 1: Highest number of independent matching channels (-blocker_cnt)
        # Tier 2: Exact name or postal code match flag (-exact_val)
        # Tier 3: Maximum channel similarity score (-round(best_score, 4))
        # Tier 4: Best channel rank (best_rank)
        # Tier 5: Stable candidate ID tie-break (cid)
        ranked_candidates = []
        for cid, info in hits.items():
            exact_val = 1 if info["exact_match"] or (info["blockers"] & EXACT_BLOCKERS) else 0
            blocker_cnt = len(info["blockers"])
            best_score = info["best_score"]
            best_rank = info["best_rank"]
            provenance_str = ",".join(sorted(info["blockers"]))

            cand_rec = self.candidate_records.get(cid)
            cand_source = cand_rec.source if cand_rec else source_from_entity_id(cid)

            # Sort key (multi-tier priority sort per docs/04 spec)
            sort_key = (
                -blocker_cnt,
                -exact_val,
                -round(best_score, 4),
                best_rank,
                cid,
            )
            ranked_candidates.append(
                (
                    sort_key,
                    {
                        "source1_entity_id": s1.entity_id,
                        "candidate_entity_id": cid,
                        "candidate_source": cand_source,
                        "blocker_provenance": provenance_str,
                        "blocker_count": blocker_cnt,
                        "best_blocker_rank": best_rank,
                        "best_blocker_score": round(best_score, 4),
                        "country_partition": s1.canonical_country or "global",
                    },
                )
            )

        ranked_candidates.sort(key=lambda item: item[0])
        capped_candidates = [
            item[1] for item in ranked_candidates[: self.max_candidates_per_entity]
        ]
        return capped_candidates


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
) -> Dict[str, Any]:
    """
    Run end-to-end Stage 1 blocking on input TSVs and emit:
      - candidate_pairs.tsv
      - candidate_provenance.tsv
      - blocking_summary.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blocker = MultiChannelBlocker(
        top_k_sparse=top_k_sparse,
        top_k_dense=top_k_dense,
        max_candidates_per_entity=max_candidates_per_entity,
        similarity_floor=similarity_floor,
    )

    # 1. Index all candidate records (S2, S3)
    print(f"Indexing candidate records from {[str(p) for p in candidate_sources]}...")
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
        if cand_count % 200000 == 0:
            print(f"  Indexed {cand_count:,} candidate records...")

    print(f"Finished indexing {cand_count:,} candidate records across {len(blocker.country_partitions)} country partitions.")

    # 1b. Initialize Dense Retrieval Index if provided
    dense_index: Optional[DenseRetrievalIndex] = None
    if dense_embeddings_path and Path(dense_embeddings_path).exists():
        print(f"Loading dense embeddings and creating FAISS index from {dense_embeddings_path}...")
        dense_index = DenseRetrievalIndex(
            embeddings_path=Path(dense_embeddings_path),
            country_partitions=blocker.country_partitions,
            similarity_floor=similarity_floor,
        )
        print(f"  Dense index initialized with {len(dense_index.candidate_ids):,} candidates.")

    # 2. Open output file writers
    candidate_pairs_path = output_dir / "candidate_pairs.tsv"
    provenance_path = output_dir / "candidate_provenance.tsv"

    s1_count = 0
    total_pairs = 0
    singletons = 0
    candidate_counts: List[int] = []
    channel_counts: Counter[str] = Counter()

    # Audit tracking
    ground_truth: Optional[Dict[str, Set[str]]] = None
    gt_total_pairs = 0
    gt_recovered_pairs = 0
    gt_entity_full_hit = 0
    gt_entity_any_hit = 0
    gt_by_channel: Counter[str] = Counter()
    gt_by_country: Counter[str] = Counter()
    gt_country_totals: Counter[str] = Counter()

    if ground_truth_path and Path(ground_truth_path).exists():
        print(f"Loading ground truth for recall audit from {ground_truth_path}...")
        ground_truth = load_ground_truth(Path(ground_truth_path))
        for mid_set in ground_truth.values():
            gt_total_pairs += len(mid_set)

    with open(candidate_pairs_path, "w", encoding="utf-8", newline="") as pairs_file, \
         open(provenance_path, "w", encoding="utf-8", newline="") as prov_file:

        pairs_writer = csv.writer(pairs_file, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        pairs_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        prov_writer = csv.DictWriter(
            prov_file,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            fieldnames=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "blocker_provenance",
                "blocker_count",
                "best_blocker_rank",
                "best_blocker_score",
                "country_partition",
            ],
        )
        prov_writer.writeheader()

        # 3. Stream Source 1 records and generate candidates
        print(f"Generating candidate pairs for S1 records from {[str(p) for p in source1_paths]}...")
        for row in read_tsv_records(source1_paths):
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
                cand_ids = [c["candidate_entity_id"] for c in candidates]
                pairs_writer.writerow([s1_rec.entity_id, ",".join(cand_ids)])

                for cand_info in candidates:
                    prov_writer.writerow(cand_info)
                    for blk in cand_info["blocker_provenance"].split(","):
                        channel_counts[blk] += 1

            # Recall audit checking if ground truth is present
            if ground_truth and s1_rec.entity_id in ground_truth:
                true_matches = ground_truth[s1_rec.entity_id]
                c_country = s1_rec.canonical_country or "unknown"
                gt_country_totals[c_country] += len(true_matches)

                found_cands = {c["candidate_entity_id"] for c in candidates}
                hits = true_matches & found_cands
                gt_recovered_pairs += len(hits)

                if hits:
                    gt_entity_any_hit += 1
                if hits == true_matches and len(true_matches) > 0:
                    gt_entity_full_hit += 1

                for c in candidates:
                    cid = c["candidate_entity_id"]
                    if cid in true_matches:
                        gt_by_country[c_country] += 1
                        for blk in c["blocker_provenance"].split(","):
                            gt_by_channel[blk] += 1

            if s1_count % 50000 == 0:
                print(f"  Processed {s1_count:,} S1 entities -> {total_pairs:,} total candidate pairs...")

    candidate_counts.sort()
    n_s1 = len(candidate_counts)
    mean_cands = total_pairs / n_s1 if n_s1 else 0.0
    median_cands = candidate_counts[n_s1 // 2] if n_s1 else 0
    p90_cands = candidate_counts[int(n_s1 * 0.90)] if n_s1 else 0
    p95_cands = candidate_counts[int(n_s1 * 0.95)] if n_s1 else 0
    max_cands = candidate_counts[-1] if n_s1 else 0

    summary: Dict[str, Any] = {
        "s1_count": n_s1,
        "total_source1_entities": n_s1,
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
    top_k_dense = args.top_k if args.top_k is not None else args.top_k_dense

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
    )
    return 0


if __name__ == "__main__":
    main()
