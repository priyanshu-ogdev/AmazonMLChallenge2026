"""
Layer 1: Multi-Channel Candidate Generation and Blocking.

Implements the Stage 1 specification defined in docs/stage1_blocking.md:
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
    canonicalize_country,
    extract_postal_code,
    extract_structural_fields,
    normalize_address,
    normalize_name,
    source_from_entity_id,
)

# Configuration defaults matching docs/stage1_blocking.md
TOP_K_DENSE = 50
TOP_K_SPARSE_OR_CHAR = 50
MAX_CANDIDATES_PER_ENTITY = 100
SIMILARITY_FLOOR = 0.30
MAX_TOKEN_DOC_FREQ = 0.02
MAX_TOKEN_DOC_COUNT = 5000
MIN_TOKEN_LEN = 3

EXACT_BLOCKERS = {
    "exact_name",
    "exact_name_postal",
    "exact_name_street",
    "exact_name_trailing",
    "name_lead_postal",
    "name_lead_street_trailing",
}

_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Extract lowercased word tokens."""
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= MIN_TOKEN_LEN]


def _char_trigrams(text: str) -> Set[str]:
    """Extract character trigrams with edge padding."""
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return set()
    padded = f"  {compact}  "
    return {padded[i : i + 3] for i in range(max(0, len(padded) - 2))}


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
    postal_code: Optional[str]
    street_number: Optional[str]
    trailing_segment: Optional[str]
    tokens: List[str]
    trigrams: Set[str]
    first_word: Optional[str]

    @classmethod
    def from_row(
        cls,
        entity_id: str,
        name: str,
        address: str,
        country: str,
    ) -> BlockingRecord:
        n_name = normalize_name(name)
        n_addr = normalize_address(address)
        canon_c = canonicalize_country(country)
        struct = extract_structural_fields(address)
        postal = extract_postal_code(address)
        street_no = struct.get("street_number")
        trailing = struct.get("trailing_segment")
        tokens = _tokenize(n_name)
        trigrams = _char_trigrams(n_name) if len(n_name) >= 3 else set()
        first_word = tokens[0] if tokens else None

        return cls(
            entity_id=entity_id.strip(),
            source=source_from_entity_id(entity_id),
            raw_name=name,
            raw_address=address,
            raw_country=country,
            canonical_country=canon_c,
            norm_name=n_name,
            norm_address=n_addr,
            postal_code=postal,
            street_number=street_no,
            trailing_segment=trailing,
            tokens=tokens,
            trigrams=trigrams,
            first_word=first_word,
        )


def read_tsv_records(paths: Iterable[Path]) -> Iterator[Dict[str, str]]:
    """Yield entity rows from one or more raw challenge TSVs or Layer 0 normalized TSVs."""
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input TSV does not exist: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = set(reader.fieldnames or [])
            if "entity_id" not in fieldnames:
                raise ValueError(f"{path} is missing required 'entity_id' column")
            has_name = any(c in fieldnames for c in ("business_name", "raw_name", "norm_name"))
            has_addr = any(c in fieldnames for c in ("business_address", "raw_address", "norm_address"))
            if not has_name:
                raise ValueError(f"{path} is missing name column (expected business_name, raw_name, or norm_name)")
            if not has_addr:
                raise ValueError(f"{path} is missing address column (expected business_address, raw_address, or norm_address)")

            for row in reader:
                name = row.get("business_name") or row.get("raw_name") or row.get("norm_name") or ""
                addr = row.get("business_address") or row.get("raw_address") or row.get("norm_address") or ""
                country = row.get("country") or row.get("country_canonical") or ""
                yield {
                    "entity_id": row.get("entity_id", ""),
                    "business_name": name,
                    "business_address": addr,
                    "country": country,
                }


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
    ):
        self.top_k_sparse = top_k_sparse
        self.top_k_dense = top_k_dense
        self.max_candidates_per_entity = max_candidates_per_entity
        self.similarity_floor = similarity_floor
        self.max_token_doc_freq = max_token_doc_freq
        self.max_token_doc_count = max_token_doc_count

        # Total indexed candidate records
        self.num_candidates = 0
        self.candidate_records: Dict[str, BlockingRecord] = {}

        # Country partition tracking
        # canonical_country -> set of candidate entity IDs
        self.country_partitions: Dict[str, Set[str]] = defaultdict(set)
        self.all_candidate_ids: Set[str] = set()

        # Channel 1 & 2: Exact and Composite Key Inverted Indexes
        # Key tuple -> canonical_country -> list of candidate entity IDs
        self.index_exact_name: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_name_postal: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_name_street: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_name_trailing: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_lead_postal: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_lead_street_trailing: Dict[Tuple[str, str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Channel 5: Address Structural
        self.index_address_structural: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Channel 3: Token Inverted Index
        # token -> canonical_country -> list of candidate entity IDs
        self.index_tokens: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.token_doc_counts: Counter[str] = Counter()

        # Channel 4: Character 3-Gram Inverted Index
        # trigram -> canonical_country -> list of candidate entity IDs
        self.index_trigrams: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.trigram_doc_counts: Counter[str] = Counter()

    def index_candidate(self, record: BlockingRecord) -> None:
        """Add a candidate record (from S2 or S3) to all blocking indexes."""
        cid = record.entity_id
        country = record.canonical_country
        self.candidate_records[cid] = record
        self.num_candidates += 1
        self.all_candidate_ids.add(cid)
        self.country_partitions[country].add(cid)

        # 1. Exact Name Key
        if record.norm_name:
            self.index_exact_name[record.norm_name][country].append(cid)

        # 2. Composite Keys
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

        # 5. Address Structural Key (postal + street number)
        if record.postal_code and record.street_number:
            self.index_address_structural[(record.postal_code, record.street_number)][country].append(cid)

        # 3. Token Inverted Index
        seen_tokens = set(record.tokens)
        for tok in seen_tokens:
            self.index_tokens[tok][country].append(cid)
            self.token_doc_counts[tok] += 1

        # 4. Trigram Inverted Index
        for tri in record.trigrams:
            self.index_trigrams[tri][country].append(cid)
            self.trigram_doc_counts[tri] += 1

    def _query_partitioned_index(
        self,
        index: Dict[Any, Dict[str, List[str]]],
        key: Any,
        s1_country: str,
    ) -> List[str]:
        """
        Query an index respecting the country partitioning invariant:
        - If s1_country != "": matching partition + records with missing country ("")
        - If s1_country == "": global search across all partitions
        """
        sub = index.get(key)
        if not sub:
            return []

        if s1_country:
            hits = list(sub.get(s1_country, []))
            if "" in sub:
                hits.extend(sub[""])
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
        # Channel 1: Exact Name Key
        # -------------------------------------------------------------
        if s1.norm_name:
            exact_hits = self._query_partitioned_index(self.index_exact_name, s1.norm_name, country)
            for rank, cid in enumerate(exact_hits, 1):
                record_hit(cid, "exact_name", 1.0, rank, is_exact=True)

        # -------------------------------------------------------------
        # Channel 2: Composite Keys
        # -------------------------------------------------------------
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
        # Channel 5: Address Structural Key
        # -------------------------------------------------------------
        if s1.postal_code and s1.street_number:
            key = (s1.postal_code, s1.street_number)
            for rank, cid in enumerate(self._query_partitioned_index(self.index_address_structural, key, country), 1):
                record_hit(cid, "address_structural", 0.85, rank, is_exact=False)

        # -------------------------------------------------------------
        # Channel 3: Token Inverted Index
        # -------------------------------------------------------------
        if s1.tokens:
            # Score candidate entities by accumulated IDF of matched tokens
            token_scores: Counter[str] = Counter()
            s1_token_set = set(s1.tokens)
            max_allowed_freq = max(10, int(self.num_candidates * self.max_token_doc_freq))
            max_allowed_count = min(self.max_token_doc_count, max_allowed_freq)

            total_s1_weight = 0.0
            for tok in s1_token_set:
                doc_cnt = self.token_doc_counts.get(tok, 0)
                if doc_cnt == 0 or doc_cnt > max_allowed_count:
                    continue
                # Standard smoothed IDF
                idf = math.log(1.0 + (self.num_candidates - doc_cnt + 0.5) / (doc_cnt + 0.5))
                total_s1_weight += idf

                cands = self._query_partitioned_index(self.index_tokens, tok, country)
                for cid in cands:
                    token_scores[cid] += idf

            if token_scores and total_s1_weight > 0.0:
                top_tokens = token_scores.most_common(self.top_k_sparse)
                for rank, (cid, score_sum) in enumerate(top_tokens, 1):
                    normalized_score = min(1.0, score_sum / total_s1_weight)
                    record_hit(cid, "token_inverted", normalized_score, rank, is_exact=False)

        # -------------------------------------------------------------
        # Channel 4: Character 3-Gram Inverted Index
        # -------------------------------------------------------------
        if s1.trigrams:
            trigram_hits: Counter[str] = Counter()
            s1_tri_len = len(s1.trigrams)
            for tri in s1.trigrams:
                cands = self._query_partitioned_index(self.index_trigrams, tri, country)
                for cid in cands:
                    trigram_hits[cid] += 1

            if trigram_hits:
                # Rank top by Jaccard similarity approximation
                scored_trigrams = []
                for cid, common in trigram_hits.most_common(self.top_k_sparse * 2):
                    cand_rec = self.candidate_records.get(cid)
                    if not cand_rec:
                        continue
                    cand_tri_len = len(cand_rec.trigrams)
                    union_size = s1_tri_len + cand_tri_len - common
                    jaccard = common / union_size if union_size > 0 else 0.0
                    if jaccard >= 0.35:
                        scored_trigrams.append((cid, jaccard))

                scored_trigrams.sort(key=lambda item: -item[1])
                for rank, (cid, jaccard) in enumerate(scored_trigrams[: self.top_k_sparse], 1):
                    record_hit(cid, "char_ngram", float(jaccard), rank, is_exact=False)

        # -------------------------------------------------------------
        # Channel 7: Dense Retrieval Hook (if provided)
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
        # Deterministic combined priority (from docs/stage1_blocking.md):
        # 1. exact/composite evidence (highest priority);
        # 2. number of independent blockers;
        # 3. best channel score;
        # 4. best rank;
        # 5. stable candidate ID tie-break.
        ranked_candidates = []
        for cid, info in hits.items():
            exact_val = 1 if info["exact_match"] or (info["blockers"] & EXACT_BLOCKERS) else 0
            blocker_cnt = len(info["blockers"])
            best_score = info["best_score"]
            best_rank = info["best_rank"]
            provenance_str = ",".join(sorted(info["blockers"]))

            cand_rec = self.candidate_records.get(cid)
            cand_source = cand_rec.source if cand_rec else source_from_entity_id(cid)

            # Sort key (negated for descending)
            sort_key = (
                -exact_val,
                -blocker_cnt,
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

        if country in self.faiss_indexes:
            index, idxs = self.faiss_indexes[country]
            k = min(top_k, len(idxs))
            if k == 0:
                return {}
            distances, indices = index.search(emb.reshape(1, -1), k)
            scores: Dict[str, float] = {}
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and dist >= self.similarity_floor:
                    cid = self.candidate_ids[idxs[idx]]
                    scores[cid] = float(dist)
            return scores
        elif country in self.partition_indices:
            idxs = self.partition_indices[country]
            if not idxs:
                return {}
            sub_embs = self.partition_embs.get(country)
            if sub_embs is None:
                sub_embs = self.candidate_embs[idxs]
            dots = np.dot(sub_embs, emb)
            top_order = np.argsort(-dots)[:top_k]
            scores = {}
            for rank_i in top_order:
                score = float(dots[rank_i])
                if score >= self.similarity_floor:
                    scores[self.candidate_ids[idxs[rank_i]]] = score
            return scores
        return {}


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """Load ground truth mapping: source1_entity_id -> set of matched_entity_ids."""
    gt: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
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

        pairs_writer = csv.writer(pairs_file, delimiter="\t")
        pairs_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        prov_writer = csv.DictWriter(
            prov_file,
            delimiter="\t",
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
    max_cands = candidate_counts[-1] if n_s1 else 0

    summary: Dict[str, Any] = {
        "total_source1_entities": n_s1,
        "total_candidate_pairs": total_pairs,
        "singletons_with_zero_candidates": singletons,
        "candidates_per_s1": {
            "mean": round(mean_cands, 2),
            "median": median_cands,
            "p90": p90_cands,
            "max": max_cands,
        },
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
    print(f"Cands/S1 -> Mean: {mean_cands:.1f}, Median: {median_cands}, P90: {p90_cands}, Max: {max_cands}")
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
        "--source1",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to Source 1 TSV(s)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to Candidate Sources (S2, S3) TSV(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write candidate_pairs.tsv and provenance",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional path to ground truth TSV for recall audit gate",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=MAX_CANDIDATES_PER_ENTITY,
        help=f"Max candidates per S1 entity (default: {MAX_CANDIDATES_PER_ENTITY})",
    )
    parser.add_argument(
        "--top-k-sparse",
        type=int,
        default=TOP_K_SPARSE_OR_CHAR,
        help=f"Top-K for sparse/char/token channels (default: {TOP_K_SPARSE_OR_CHAR})",
    )
    parser.add_argument(
        "--top-k-dense",
        type=int,
        default=TOP_K_DENSE,
        help=f"Top-K for dense channel (default: {TOP_K_DENSE})",
    )
    parser.add_argument(
        "--dense-embeddings",
        type=Path,
        default=None,
        help="Optional path to .npz file containing precomputed dense embeddings for Channel 7",
    )
    args = parser.parse_args()

    run_blocking(
        source1_paths=args.source1,
        candidate_sources=args.candidates,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        dense_embeddings_path=args.dense_embeddings,
        top_k_sparse=args.top_k_sparse,
        top_k_dense=args.top_k_dense,
        max_candidates_per_entity=args.max_candidates,
    )


if __name__ == "__main__":
    main()
