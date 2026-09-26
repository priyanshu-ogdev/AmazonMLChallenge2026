"""
Stage 2c: deterministic pair-feature extraction.

This module computes features for an already materialized candidate set. It
does not create candidates, use labels, or fit a learned model. Vectorizers
and other learned statistics belong in the Stage 3 fold-specific pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
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


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _overlap(left: Set[str], right: Set[str]) -> float:
    """Overlap coefficient, useful when one address is a partial observation."""
    if not left and not right:
        return 1.0
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


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

    row: Dict[str, object] = {
        "source1_entity_id": left["entity_id"],
        "candidate_entity_id": right["entity_id"],
        # Raw country strings: DROP_CATEGORICAL in Stage 3 (not GBM inputs).
        "source1_country": left["country"],
        "candidate_country": right["country"],
        # Canonical country strings: also dropped by Stage 3, used for
        # fold stratification and per-country diagnostics only.
        "source1_canonical_country": canon_left,
        "candidate_canonical_country": canon_right,
        "source_is_s3": int(str(right["entity_id"]).startswith("S3-")),
        # country_equal uses canonicalized form: handles US/USA/us and
        # France/FR correctly. Value is 0.0/1.0; None is stored as -1.0 to allow
        # the GBM to learn a missing-country branch.
        "country_equal": (
            -1.0 if match_flag is None else (1.0 if match_flag else 0.0)
        ),
        "country_equal_missing": int(match_flag is None),
        "left_country_missing": int(not left["country"]),
        "right_country_missing": int(not right["country"]),
        "name_both_missing": int(not name_left and not name_right),
        "address_both_missing": int(not address_left and not address_right),
        "name_exact": int(bool(name_left) and name_left == name_right),
        "address_exact": int(bool(address_left) and address_left == address_right),
        "name_jaccard": _jaccard(name_tokens_left, name_tokens_right),
        "name_overlap": _overlap(name_tokens_left, name_tokens_right),
        "name_edit_similarity": _safe_ratio(name_left, name_right),
        "name_char_trigram_jaccard": _jaccard(
            left["name_trigrams"], right["name_trigrams"]
        ),
        "address_jaccard": _jaccard(address_tokens_left, address_tokens_right),
        "address_overlap": _overlap(address_tokens_left, address_tokens_right),
        "address_edit_similarity": _safe_ratio(address_left, address_right),
        "address_char_trigram_jaccard": _jaccard(
            left["address_trigrams"], right["address_trigrams"]
        ),
        "name_number_overlap": _overlap(name_numbers_left, name_numbers_right),
        "address_number_overlap": _overlap(
            address_numbers_left, address_numbers_right
        ),
        "postal_equal": int(
            bool(left["postal"])
            and bool(right["postal"])
            and left["postal"] == right["postal"]
        ),
        "postal_missing_either": int(not left["postal"] or not right["postal"]),
        "name_length_abs_diff": abs(len(name_left) - len(name_right)),
        "address_length_abs_diff": abs(len(address_left) - len(address_right)),
        "same_name_different_address": int(
            bool(name_left)
            and name_left == name_right
            and bool(address_left)
            and bool(address_right)
            and address_left != address_right
        ),
        "same_address_different_name": int(
            bool(address_left)
            and address_left == address_right
            and bool(name_left)
            and bool(name_right)
            and name_left != name_right
        ),
        # candidate_rank is monotonic-decreasing (-1 constraint). Missing values
        # use 999.0 (worse than any real rank) to prevent the constraint from treating
        # missing ranks as superior to rank 0.
        "candidate_rank": 999.0 if left_rank is None else float(left_rank),
        "candidate_rank_missing": int(left_rank is None),
        "rank_margin_from_best": 999.0 if left_rank is None else max(0.0, float(left_rank) - 1.0),
        "best_blocker_score": -1.0 if best_score is None else float(best_score),
        "best_blocker_score_missing": int(best_score is None),
        "best_blocker_score_diff": 0.0 if score_margin_to_best is None else float(score_margin_to_best),
        "best_blocker_score_diff_missing": int(score_margin_to_best is None),
        "blocker_count": 0 if blocker_count is None else int(blocker_count),
        "candidate_count_for_s1": 1 if candidate_count is None else int(candidate_count),
        "has_blocker_provenance": int(bool(provenance)),
        "blocker_provenance": provenance or "",
    }
    return row


def load_records(paths: Iterable[Path]) -> Dict[str, Dict[str, str]]:
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
                records[eids[i]] = {
                    "entity_id": eids[i],
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
                records[eids[i]] = {
                    "entity_id": eids[i],
                    "business_name": names[i],
                    "business_address": addrs[i],
                    "raw_name": raw_names[i],
                    "raw_address": raw_addrs[i],
                    "country": countries[i],
                    "canonical_country": can_countries[i],
                }
    return records


def _feature_worker(
    chunk_items: List[Tuple[str, List[str]]],
    normalized: Dict[str, Dict[str, object]],
    provenance: Dict[Tuple[str, str], Tuple[str, Optional[float], Optional[float], Optional[int]]],
    s1_max_score: Dict[str, float],
) -> List[Dict[str, object]]:
    worker_rows: List[Dict[str, object]] = []
    for s1_id, cand_ids in chunk_items:
        cand_count = len(cand_ids)
        max_s = s1_max_score.get(s1_id)
        for candidate_id in cand_ids:
            pair = (s1_id, candidate_id)
            blocker, rank, score, count = provenance.get(pair, (None, None, None, None))
            margin = (max_s - score) if (max_s is not None and score is not None) else None
            worker_rows.append(
                pair_feature_row(
                    normalized[s1_id],
                    normalized[candidate_id],
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
) -> pd.DataFrame:
    """Build features for every candidate pair, failing on invalid references."""
    candidates = pd.read_csv(candidate_file, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    required = {"source1_entity_id", "candidate_entity_ids"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"{candidate_file} is missing required columns: {sorted(missing)}")

    # Bug 4 fix: collect active entity IDs appearing in candidate pairs so we only normalize those.
    active_ids: Set[str] = set()
    seen: Set[Tuple[str, str]] = set()
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
        for candidate_id in (v.strip() for v in cand_vals[i].split(",") if v.strip()):
            pair = (source1_id, candidate_id)
            if pair in seen:
                continue
            seen.add(pair)
            if candidate_id not in records:
                raise ValueError(f"Unknown candidate entity ID: {candidate_id}")
            active_ids.add(candidate_id)
            pairs_by_s1.setdefault(source1_id, []).append(candidate_id)

    # Bug 3 fix: fast provenance lookup construction using column arrays.
    provenance: Dict[Tuple[str, str], Tuple[str, Optional[float], Optional[float], Optional[int]]] = {}
    if provenance_file:
        provenance_frame = pd.read_csv(
            provenance_file, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE
        )
        required_provenance = {
            "source1_entity_id",
            "candidate_entity_id",
            "blocker_provenance",
        }
        missing = required_provenance - set(provenance_frame.columns)
        if missing:
            raise ValueError(
                f"{provenance_file} is missing required columns: {sorted(missing)}"
            )
        s1_arr = provenance_frame["source1_entity_id"].values
        cid_arr = provenance_frame["candidate_entity_id"].values
        prov_arr = provenance_frame["blocker_provenance"].values
        rank_col = "best_blocker_rank" if "best_blocker_rank" in provenance_frame.columns else (
            "candidate_rank" if "candidate_rank" in provenance_frame.columns else None
        )
        rank_arr = provenance_frame[rank_col].values if rank_col else None
        score_arr = provenance_frame["best_blocker_score"].values if "best_blocker_score" in provenance_frame.columns else None
        count_arr = provenance_frame["blocker_count"].values if "blocker_count" in provenance_frame.columns else None

        for i in range(len(provenance_frame)):
            s1 = s1_arr[i]
            cid = cid_arr[i]
            prov = prov_arr[i]
            r = rank_arr[i] if rank_arr is not None else ""
            s = score_arr[i] if score_arr is not None else ""
            c = count_arr[i] if count_arr is not None else ""
            provenance[(s1, cid)] = (
                prov,
                float(r) if r else None,
                float(s) if s else None,
                int(c) if c else None,
            )

    # Bug 4 fix: compute features only for active entities.
    normalized = {entity_id: _record_features(records[entity_id]) for entity_id in active_ids}

    # Pre-calculate max score per S1 entity for relative margin computation
    s1_max_score: Dict[str, float] = {}
    for s1_id, cand_ids in pairs_by_s1.items():
        cand_scores = [
            provenance.get((s1_id, cid), (None, None, None, None))[2]
            for cid in cand_ids
        ]
        valid_scores = [s for s in cand_scores if s is not None]
        if valid_scores:
            s1_max_score[s1_id] = max(valid_scores)

    items = list(pairs_by_s1.items())
    num_workers = max(1, os.cpu_count() or 4)
    # Limit workers for pair features since each process duplicates the lookup dictionaries
    num_workers = min(num_workers, 6)

    chunk_size = math.ceil(len(items) / num_workers) if num_workers > 0 else 0
    if chunk_size == 0:
        chunks = []
    else:
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    rows: List[Dict[str, object]] = []
    if num_workers <= 1 or not chunks:
        rows = _feature_worker(items, normalized, provenance, s1_max_score)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(_feature_worker, chunk, normalized, provenance, s1_max_score)
                for chunk in chunks
            ]
            for fut in as_completed(futures):
                rows.extend(fut.result())

    feature_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        # Raw country strings (Stage 3 drops these as DROP_CATEGORICAL)
        "source1_country",
        "candidate_country",
        # Canonical country strings (Stage 3 drops these; used for diagnostics)
        "source1_canonical_country",
        "candidate_canonical_country",
        "source_is_s3",
        # country_equal: -1=missing, 0=mismatch, 1=match (canonical)
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
        "address_jaccard",
        "address_overlap",
        "address_edit_similarity",
        "address_char_trigram_jaccard",
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
    ]
    result = pd.DataFrame(rows, columns=feature_columns)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    return result


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
