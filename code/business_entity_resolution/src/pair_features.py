"""
Stage 2c: deterministic pair-feature extraction.

This module computes features for an already materialized candidate set. It
does not create candidates, use labels, or fit a learned model. Vectorizers
and other learned statistics belong in the Stage 3 fold-specific pipeline.
"""

from __future__ import annotations

import argparse
_shared_normalized = None
_shared_provenance = None
_shared_s1_max_score = None
import csv
import math
import os
import re
try:
    import psutil
except ImportError:
    psutil = None
try:
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:
    _rf_fuzz = None
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.normalize import (
    canonicalize_country,
    country_match_flag,
    extract_postal_code,
    normalize_address,
    normalize_name,
)

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"\d+")


def _tokens(value: str) -> Set[str]:
    return set(_TOKEN_RE.findall(value or ""))


def _numeric_tokens(value: str) -> Set[str]:
    return set(_NUMBER_RE.findall(value or ""))


def _token_sort_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if _rf_fuzz is not None:
        return float(_rf_fuzz.token_sort_ratio(left, right)) / 100.0
    return _safe_ratio(left, right)


def _token_set_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if _rf_fuzz is not None:
        return float(_rf_fuzz.token_set_ratio(left, right)) / 100.0
    return _safe_ratio(left, right)


def _fuzz_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if _rf_fuzz is not None:
        return float(_rf_fuzz.ratio(left, right)) / 100.0
    return _safe_ratio(left, right)


class PairFeatureRow(tuple):
    """Zero-overhead tuple representing a pair feature row with key/index dual access."""
    __slots__ = ()
    _FIELDS = (
        "source1_entity_id",
        "candidate_entity_id",
        "source1_country",
        "candidate_country",
        "source1_canonical_country",
        "candidate_canonical_country",
        "source_is_s3",
        "country_equal",
        "country_equal_missing",
        "left_country_missing",
        "right_country_missing",
        "name_both_missing",
        "address_both_missing",
        "name_exact",
        "address_exact",
        "name_jaccard",
        "name_overlap",
        "name_edit_similarity",
        "name_char_trigram_jaccard",
        "name_token_sort_ratio",
        "name_fuzz_ratio",
        "address_jaccard",
        "address_overlap",
        "address_edit_similarity",
        "address_char_trigram_jaccard",
        "address_token_set_ratio",
        "name_number_overlap",
        "address_number_overlap",
        "postal_equal",
        "postal_missing_either",
        "name_length_abs_diff",
        "address_length_abs_diff",
        "same_name_different_address",
        "same_address_different_name",
        "candidate_rank",
        "candidate_rank_missing",
        "rank_margin_from_best",
        "best_blocker_score",
        "best_blocker_score_missing",
        "best_blocker_score_diff",
        "best_blocker_score_diff_missing",
        "blocker_count",
        "candidate_count_for_s1",
        "has_blocker_provenance",
        "blocker_provenance",
    )
    _FIELD_MAP = {name: i for i, name in enumerate(_FIELDS)}
    _FIELD_MAP["candidate_count"] = _FIELD_MAP["candidate_count_for_s1"]
    _FIELD_MAP["has_provenance"] = _FIELD_MAP["has_blocker_provenance"]

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


try:
    from rapidfuzz.distance import Levenshtein as _rf_levenshtein

    def _safe_ratio(left: str, right: str) -> float:
        """Fast C-extension Levenshtein similarity via rapidfuzz."""
        return float(_rf_levenshtein.normalized_similarity(left or "", right or ""))
except ImportError:
    def _safe_ratio(left: str, right: str) -> float:
        """Normalized Levenshtein similarity fallback without an external dependency."""
        left, right = left or "", right or ""
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, 1):
            current = [i]
            for j, right_char in enumerate(right, 1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[j] + 1,
                        previous[j - 1] + (left_char != right_char),
                    )
                )
            previous = current
        return 1.0 - previous[-1] / max(len(left), len(right))


def _fast_jaccard_overlap(left: Set[str], right: Set[str]) -> Tuple[float, float]:
    """
    Zero-allocation Jaccard similarity and Overlap coefficient.
    Iterates over the smaller set without allocating any intermediate set objects on the heap.
    """
    len_l = len(left)
    len_r = len(right)
    if not len_l and not len_r:
        return 1.0, 1.0
    if not len_l or not len_r:
        return 0.0, 0.0
    smaller, larger = (left, right) if len_l <= len_r else (right, left)
    inter = 0
    for item in smaller:
        if item in larger:
            inter += 1
    union = len_l + len_r - inter
    jaccard = inter / union if union else 0.0
    min_len = len_l if len_l < len_r else len_r
    overlap = inter / min_len if min_len else 0.0
    return jaccard, overlap


def _jaccard(left: Set[str], right: Set[str]) -> float:
    return _fast_jaccard_overlap(left, right)[0]


def _overlap(left: Set[str], right: Set[str]) -> float:
    """Overlap coefficient, useful when one address is a partial observation."""
    return _fast_jaccard_overlap(left, right)[1]


def _char_trigrams(value: str) -> Set[str]:
    compact = re.sub(r"\s+", " ", value or "").strip()
    padded = f"  {compact}  "
    return {padded[i : i + 3] for i in range(max(0, len(padded) - 2))}


def _record_features(record: Dict[str, str]) -> Dict[str, object]:
    if "norm_name" in record and record["norm_name"] is not None:
        name = record["norm_name"]
    else:
        name = normalize_name(record.get("business_name") or record.get("raw_name", ""))

    if "norm_address" in record and record["norm_address"] is not None:
        address = record["norm_address"]
    else:
        address = normalize_address(record.get("business_address") or record.get("raw_address", ""))

    raw_country = record.get("country", "")
    canonical_country = record.get("canonical_country") or canonicalize_country(raw_country)
    raw_addr = record.get("raw_address") or record.get("business_address", "")
    postal = record.get("postal_code") or extract_postal_code(raw_addr, country=canonical_country)
    return {
        "entity_id": record["entity_id"],
        "country": raw_country,
        "canonical_country": canonical_country,
        "name": name,
        "address": address,
        "name_tokens": _tokens(name),
        "address_tokens": _tokens(address),
        "name_numbers": _numeric_tokens(name),
        "address_numbers": _numeric_tokens(address),
        "postal": postal,
        "name_trigrams": _char_trigrams(name),
        "address_trigrams": _char_trigrams(address),
    }


def pair_feature_row(
    left: Dict[str, object],
    right: Dict[str, object],
    provenance: Optional[str] = None,
    left_rank: Optional[float] = None,
    best_score: Optional[float] = None,
    blocker_count: Optional[int] = None,
    candidate_count: Optional[int] = None,
    score_margin_to_best: Optional[float] = None,
) -> Dict[str, object]:
    """Return one label-free feature row for a candidate pair."""
    name_left = left["name"]
    name_right = right["name"]
    address_left = left["address"]
    address_right = right["address"]
    name_tokens_left = left["name_tokens"]
    name_tokens_right = right["name_tokens"]
    address_tokens_left = left["address_tokens"]
    address_tokens_right = right["address_tokens"]
    name_numbers_left = left["name_numbers"]
    name_numbers_right = right["name_numbers"]
    address_numbers_left = left["address_numbers"]
    address_numbers_right = right["address_numbers"]

    # Use canonical country for the match flag so aliases (US/USA/us) and
    # France (France/FR) are correctly unified. country_match_flag() returns
    # None when either side is empty, so the GBM can use a missing branch.
    canon_left = left["canonical_country"]
    canon_right = right["canonical_country"]
    match_flag = country_match_flag(canon_left, canon_right)

    # Exact string short-circuits (bypasses Levenshtein and token ratios on identical strings)
    if name_left and name_left == name_right:
        name_exact = 1
        name_tok_jaccard = 1.0
        name_tok_overlap = 1.0
        name_ratio = 1.0
        name_tri_jaccard = 1.0
        name_sort_ratio = 1.0
        name_f_ratio = 1.0
    elif not name_left and not name_right:
        name_exact = 0
        name_tok_jaccard = 1.0
        name_tok_overlap = 1.0
        name_ratio = 1.0
        name_tri_jaccard = 1.0
        name_sort_ratio = 1.0
        name_f_ratio = 1.0
    else:
        name_exact = 0
        name_tok_jaccard, name_tok_overlap = _fast_jaccard_overlap(name_tokens_left, name_tokens_right)
        name_ratio = _safe_ratio(name_left, name_right)
        name_tri_jaccard, _ = _fast_jaccard_overlap(left["name_trigrams"], right["name_trigrams"])
        name_sort_ratio = _token_sort_ratio(name_left, name_right)
        name_f_ratio = _fuzz_ratio(name_left, name_right)

    if address_left and address_left == address_right:
        addr_exact = 1
        addr_tok_jaccard = 1.0
        addr_tok_overlap = 1.0
        addr_ratio = 1.0
        addr_tri_jaccard = 1.0
        addr_set_ratio = 1.0
    elif not address_left and not address_right:
        addr_exact = 0
        addr_tok_jaccard = 1.0
        addr_tok_overlap = 1.0
        addr_ratio = 1.0
        addr_tri_jaccard = 1.0
        addr_set_ratio = 1.0
    else:
        addr_exact = 0
        addr_tok_jaccard, addr_tok_overlap = _fast_jaccard_overlap(address_tokens_left, address_tokens_right)
        addr_ratio = _safe_ratio(address_left, address_right)
        addr_tri_jaccard, _ = _fast_jaccard_overlap(left["address_trigrams"], right["address_trigrams"])
        addr_set_ratio = _token_set_ratio(address_left, address_right)

    _, name_num_overlap = _fast_jaccard_overlap(name_numbers_left, name_numbers_right)
    _, addr_num_overlap = _fast_jaccard_overlap(address_numbers_left, address_numbers_right)

    row = (
        left["entity_id"],
        right["entity_id"],
        left["country"],
        right["country"],
        canon_left,
        canon_right,
        int(str(right["entity_id"]).startswith("S3-")),
        -1.0 if match_flag is None else (1.0 if match_flag else 0.0),
        int(match_flag is None),
        int(not left["country"]),
        int(not right["country"]),
        int(not name_left and not name_right),
        int(not address_left and not address_right),
        name_exact,
        addr_exact,
        name_tok_jaccard,
        name_tok_overlap,
        name_ratio,
        name_tri_jaccard,
        name_sort_ratio,
        name_f_ratio,
        addr_tok_jaccard,
        addr_tok_overlap,
        addr_ratio,
        addr_tri_jaccard,
        addr_set_ratio,
        name_num_overlap,
        addr_num_overlap,
        int(bool(left["postal"]) and bool(right["postal"]) and left["postal"] == right["postal"]),
        int(not left["postal"] or not right["postal"]),
        abs(len(name_left) - len(name_right)),
        abs(len(address_left) - len(address_right)),
        int(bool(name_left) and name_left == name_right and bool(address_left) and bool(address_right) and address_left != address_right),
        int(bool(address_left) and address_left == address_right and bool(name_left) and bool(name_right) and name_left != name_right),
        999.0 if left_rank is None else float(left_rank),
        int(left_rank is None),
        999.0 if left_rank is None else max(0.0, float(left_rank) - 1.0),
        -1.0 if best_score is None else float(best_score),
        int(best_score is None),
        0.0 if score_margin_to_best is None else float(score_margin_to_best),
        int(score_margin_to_best is None),
        0 if blocker_count is None else int(blocker_count),
        1 if candidate_count is None else int(candidate_count),
        int(bool(provenance)),
        provenance or "",
    )
    return PairFeatureRow(row)


def load_records(paths: Iterable[Path], needed_ids: Optional[Set[str]] = None) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for path in paths:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
        if "entity_id" not in frame.columns:
            raise ValueError(f"{path} is missing required column: entity_id")
        name_col = next((c for c in ("business_name", "raw_name", "norm_name") if c in frame.columns), None)
        addr_col = next((c for c in ("business_address", "raw_address", "norm_address") if c in frame.columns), None)
        country_col = next((c for c in ("country", "country_canonical") if c in frame.columns), None)
        if not name_col or not addr_col:
            raise ValueError(f"{path} is missing name/address columns: {list(frame.columns)}")

        has_stage0 = "norm_name" in frame.columns and "norm_address" in frame.columns
        eids = frame["entity_id"].values
        names = frame[name_col].values
        addrs = frame[addr_col].values
        raw_names = frame["raw_name"].values if "raw_name" in frame.columns else names
        raw_addrs = frame["raw_address"].values if "raw_address" in frame.columns else addrs
        countries = frame[country_col].values if country_col else np.array([""] * len(frame))
        can_countries = frame["country_canonical"].values if "country_canonical" in frame.columns else np.array([""] * len(frame))

        if has_stage0:
            norm_names = frame["norm_name"].values
            norm_addrs = frame["norm_address"].values
            postals = frame["postal_code"].values if "postal_code" in frame.columns else np.array([""] * len(frame))
            is_missings = frame["is_address_missing"].values if "is_address_missing" in frame.columns else np.array(["0"] * len(frame))
            for i in range(len(eids)):
                eid = eids[i]
                if needed_ids is not None and eid not in needed_ids:
                    continue
                records[eid] = {
                    "entity_id": eid,
                    "business_name": names[i],
                    "business_address": addrs[i],
                    "raw_name": raw_names[i],
                    "raw_address": raw_addrs[i],
                    "country": countries[i],
                    "canonical_country": can_countries[i],
                    "norm_name": norm_names[i],
                    "norm_address": norm_addrs[i],
                    "postal_code": postals[i],
                    "is_address_missing": is_missings[i],
                }
        else:
            for i in range(len(eids)):
                eid = eids[i]
                if needed_ids is not None and eid not in needed_ids:
                    continue
                records[eid] = {
                    "entity_id": eid,
                    "business_name": names[i],
                    "business_address": addrs[i],
                    "raw_name": raw_names[i],
                    "raw_address": raw_addrs[i],
                    "country": countries[i],
                    "canonical_country": can_countries[i],
                }
    return records


def _spawn_initializer(
    normalized: Dict[str, Dict[str, object]],
) -> None:
    """Initializer for spawn-based ProcessPoolExecutor workers."""
    global _shared_normalized
    _shared_normalized = normalized


def _feature_worker(
    chunk_items: List[Tuple[str, List[str], Dict[str, Tuple], Optional[float]]],
    normalized: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Tuple[Any, ...]]:
    global _shared_normalized
    norm_dict = normalized if normalized is not None else _shared_normalized

    worker_rows: List[Tuple[Any, ...]] = []
    for s1_id, cand_ids, s1_prov, max_s in chunk_items:
        cand_count = len(cand_ids)
        s1_norm = norm_dict[s1_id]
        for candidate_id in cand_ids:
            cand_norm = norm_dict.get(candidate_id)
            if cand_norm is None:
                continue
            blocker, rank, score, count = s1_prov.get(candidate_id, (None, None, None, None))
            margin = (max_s - score) if (max_s is not None and score is not None) else None
            worker_rows.append(
                pair_feature_row(
                    s1_norm,
                    cand_norm,
                    provenance=blocker,
                    left_rank=rank,
                    best_score=score,
                    blocker_count=count,
                    candidate_count=cand_count,
                    score_margin_to_best=margin,
                )
            )
    return worker_rows


def build_pair_features(
    records: Dict[str, Dict[str, str]],
    candidate_file: Path,
    output_file: Path,
    provenance_file: Optional[Path] = None,
    max_candidates_per_entity: int = 15,
) -> pd.DataFrame:
    """Build features for every candidate pair, failing on invalid references."""
    candidates = pd.read_csv(candidate_file, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    required = {"source1_entity_id", "candidate_entity_ids"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"{candidate_file} is missing required columns: {sorted(missing)}")

    active_ids: Set[str] = set()
    pairs_by_s1: Dict[str, List[str]] = {}

    s1_vals = candidates["source1_entity_id"].values
    cand_vals = candidates["candidate_entity_ids"].values

    for i in range(len(s1_vals)):
        source1_id = s1_vals[i].strip()
        if not source1_id:
            continue
        if source1_id not in records:
            raise ValueError(f"Unknown source1 entity ID: {source1_id}")
        active_ids.add(source1_id)
        
        seen_cand = set()
        cands_for_s1 = []
        raw_cands = [v.strip() for v in cand_vals[i].split(",") if v.strip()]
        for candidate_id in raw_cands[:max_candidates_per_entity]:
            if candidate_id not in seen_cand:
                seen_cand.add(candidate_id)
                if candidate_id not in records:
                    raise ValueError(f"Unknown candidate entity ID: {candidate_id}")
                active_ids.add(candidate_id)
                cands_for_s1.append(candidate_id)
        if cands_for_s1:
            pairs_by_s1[source1_id] = cands_for_s1

    del candidates

    # Fast streaming provenance parser (no pandas 6.7GB read, only loads top cands per S1)
    prov_by_s1: Dict[str, Dict[str, Tuple[str, Optional[float], Optional[float], Optional[int]]]] = {}
    if provenance_file and Path(provenance_file).exists():
        print(f"[STAGE 2c] Streaming provenance from {provenance_file} (capping top {max_candidates_per_entity} per S1)...", flush=True)
        with open(provenance_file, "r", encoding="utf-8") as f_prov:
            header_line = f_prov.readline()
            header = header_line.strip().split("\t")
            col_map = {col: idx for idx, col in enumerate(header)}
            s1_idx = col_map.get("source1_entity_id", 0)
            cid_idx = col_map.get("candidate_entity_id", 1)
            prov_idx = col_map.get("blocker_provenance", 3)
            rank_idx = col_map.get("best_blocker_rank", col_map.get("candidate_rank", 5))
            score_idx = col_map.get("best_blocker_score", 6)
            count_idx = col_map.get("blocker_count", 4)

            cur_s1 = None
            cur_s1_prefix = ""
            cur_dict = {}
            for line in f_prov:
                if cur_s1 is not None and len(cur_dict) >= max_candidates_per_entity:
                    if line.startswith(cur_s1_prefix):
                        continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(s1_idx, cid_idx):
                    continue
                s1 = parts[s1_idx]
                if s1 != cur_s1:
                    if cur_s1 is not None and cur_dict:
                        prov_by_s1[cur_s1] = cur_dict
                    cur_s1 = s1
                    cur_s1_prefix = s1 + "\t"
                    cur_dict = {}
                if len(cur_dict) < max_candidates_per_entity:
                    cid = parts[cid_idx]
                    prov = parts[prov_idx] if 0 <= prov_idx < len(parts) else ""
                    r = parts[rank_idx] if 0 <= rank_idx < len(parts) else ""
                    s = parts[score_idx] if 0 <= score_idx < len(parts) else ""
                    c = parts[count_idx] if 0 <= count_idx < len(parts) else ""
                    cur_dict[cid] = (
                        prov,
                        float(r) if r else None,
                        float(s) if s else None,
                        int(c) if c else None,
                    )
            if cur_s1 is not None and cur_dict:
                prov_by_s1[cur_s1] = cur_dict

    # Compute features only for active entities and free raw records memory immediately
    normalized = {}
    for entity_id in active_ids:
        rec = records.pop(entity_id, None)
        if rec is not None:
            normalized[entity_id] = _record_features(rec)
    records.clear()
    del records
    import gc
    gc.collect()

    # Pre-calculate max score per S1 entity for relative margin computation
    s1_max_score: Dict[str, float] = {}
    for s1_id, cand_ids in pairs_by_s1.items():
        s1_prov = prov_by_s1.get(s1_id, {})
        valid_scores = [
            s1_prov[cid][2] for cid in cand_ids
            if cid in s1_prov and s1_prov[cid][2] is not None
        ]
        if valid_scores:
            s1_max_score[s1_id] = max(valid_scores)

    # Package tasks with per-S1 provenance slice (zero global dict pickling overhead)
    chunk_size = 5000
    s1_keys = list(pairs_by_s1.keys())
    chunks = []
    for i in range(0, len(s1_keys), chunk_size):
        chunk_s1_keys = s1_keys[i:i + chunk_size]
        chunk_payload = [
            (
                s1,
                pairs_by_s1[s1],
                prov_by_s1.get(s1, {}),
                s1_max_score.get(s1),
            )
            for s1 in chunk_s1_keys
        ]
        chunks.append(chunk_payload)

    del pairs_by_s1
    del prov_by_s1
    del s1_max_score
    import gc
    gc.collect()

    num_workers = max(1, os.cpu_count() or 4)
    import sys as _sys
    _platform = _sys.platform

    # On Windows, ProcessPoolExecutor(spawn) pickles the multi-GB dictionary to every worker,
    # blowing past 25 GB RAM and triggering Windows pagefile OOM.
    # RapidFuzz releases the GIL in C++, so ThreadPoolExecutor achieves full multi-core throughput
    # with ZERO memory duplication (RAM < 2.5 GB).
    if _platform == "win32":
        from concurrent.futures import ThreadPoolExecutor
        ExecutorClass = ThreadPoolExecutor
        pool_kwargs = {}
        num_workers = min(num_workers, 12)
        print(f"[INFO] Windows detected. Using ThreadPoolExecutor with {num_workers} threads (zero memory duplication, RapidFuzz C++ parallel).", flush=True)
    else:
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        ExecutorClass = ProcessPoolExecutor
        pool_kwargs = {"mp_context": ctx}
        num_workers = min(num_workers, 14)

    feature_columns = list(PairFeatureRow._FIELDS)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(feature_columns)

        if num_workers <= 1 or not chunks:
            for chunk in chunks:
                chunk_rows = _feature_worker(chunk, normalized)
                writer.writerows(chunk_rows)
        else:
            with ExecutorClass(max_workers=num_workers, **pool_kwargs) as pool:
                futures = [
                    pool.submit(_feature_worker, chunk, normalized)
                    for chunk in chunks
                ]
                for fut in as_completed(futures):
                    chunk_rows = fut.result()
                    writer.writerows(chunk_rows)

    del normalized
    gc.collect()
    print(f"[STAGE 2c] Successfully wrote {output_file}.", flush=True)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2c pair features")
    parser.add_argument("--source1", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-sources", type=Path, nargs="+", default=None,
                        help="Candidate source TSV paths (e.g. source2.tsv source3.tsv)")
    parser.add_argument("--source2", type=Path, default=None, help="Candidate source 2 TSV")
    parser.add_argument("--source3", type=Path, default=None, help="Candidate source 3 TSV")
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Output TSV path (alias)")
    parser.add_argument("--output-file", type=Path, default=None, help="Output TSV path")
    parser.add_argument("--provenance-file", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output_file or args.output
    if not output_path:
        parser.error("Either --output-file or --output is required")

    candidate_sources = list(args.candidate_sources or [])
    if args.source2:
        candidate_sources.append(args.source2)
    if args.source3:
        candidate_sources.append(args.source3)
    if not candidate_sources:
        parser.error("Either --candidate-sources or --source2/--source3 is required")

    source1_paths = list(args.source1) if isinstance(args.source1, list) else [args.source1]
    all_input_paths = source1_paths + candidate_sources

    build_pair_features(
        load_records(all_input_paths),
        args.candidate_file,
        output_path,
        args.provenance_file,
    )


if __name__ == "__main__":
    main()
