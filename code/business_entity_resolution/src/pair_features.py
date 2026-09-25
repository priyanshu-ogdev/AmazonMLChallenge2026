"""
Stage 2c: deterministic pair-feature extraction.

This module computes features for an already materialized candidate set. It
does not create candidates, use labels, or fit a learned model. Vectorizers
and other learned statistics belong in the Stage 3 fold-specific pipeline.
"""

from __future__ import annotations

import argparse
import re
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


def _safe_ratio(left: str, right: str) -> float:
    """Normalized Levenshtein similarity without an external dependency."""
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
    name = normalize_name(record.get("business_name", ""))
    address = normalize_address(record.get("business_address", ""))
    raw_country = record.get("country", "")
    return {
        "entity_id": record["entity_id"],
        "country": raw_country,
        "canonical_country": canonicalize_country(raw_country),
        "name": name,
        "address": address,
        "name_tokens": _tokens(name),
        "address_tokens": _tokens(address),
        "name_numbers": _numeric_tokens(name),
        "address_numbers": _numeric_tokens(address),
        "postal": extract_postal_code(address),
        "name_trigrams": _char_trigrams(name),
        "address_trigrams": _char_trigrams(address),
    }


def pair_feature_row(
    left: Dict[str, object],
    right: Dict[str, object],
    provenance: Optional[str] = None,
    left_rank: Optional[float] = None,
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
        # France/FR correctly. Value is 0/1; None is stored as -1 to allow
        # the GBM to learn a missing-country branch.
        "country_equal": (
            -1 if match_flag is None else int(match_flag)
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
        "name_number_overlap": _jaccard(name_numbers_left, name_numbers_right),
        "address_number_overlap": _jaccard(
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
        "candidate_rank": -1.0 if left_rank is None else float(left_rank),
        "candidate_rank_missing": int(left_rank is None),
        "has_blocker_provenance": int(bool(provenance)),
        "blocker_provenance": provenance or "",
    }
    return row


def load_records(paths: Iterable[Path]) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for path in paths:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        required = {"entity_id", "business_name", "business_address", "country"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in frame.to_dict("records"):
            records[row["entity_id"]] = row
    return records


def build_pair_features(
    records: Dict[str, Dict[str, str]],
    candidate_file: Path,
    output_file: Path,
    provenance_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Build features for every candidate pair, failing on invalid references."""
    candidates = pd.read_csv(candidate_file, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "candidate_entity_ids"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"{candidate_file} is missing required columns: {sorted(missing)}")

    provenance: Dict[Tuple[str, str], Tuple[str, Optional[float]]] = {}
    if provenance_file:
        provenance_frame = pd.read_csv(
            provenance_file, sep="\t", dtype=str, keep_default_na=False
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
        for row in provenance_frame.to_dict("records"):
            rank = row.get("candidate_rank", "")
            provenance[(row["source1_entity_id"], row["candidate_entity_id"])] = (
                row["blocker_provenance"],
                float(rank) if rank else None,
            )

    normalized = {entity_id: _record_features(record) for entity_id, record in records.items()}
    rows: List[Dict[str, object]] = []
    seen: Set[Tuple[str, str]] = set()
    for candidate_row in candidates.to_dict("records"):
        source1_id = candidate_row["source1_entity_id"]
        if source1_id not in normalized:
            raise ValueError(f"Unknown source1 entity ID: {source1_id}")
        for candidate_id in (
            value.strip()
            for value in candidate_row["candidate_entity_ids"].split(",")
            if value.strip()
        ):
            pair = (source1_id, candidate_id)
            if pair in seen:
                continue
            seen.add(pair)
            if candidate_id not in normalized:
                raise ValueError(f"Unknown candidate entity ID: {candidate_id}")
            blocker, rank = provenance.get(pair, (None, None))
            rows.append(pair_feature_row(normalized[source1_id], normalized[candidate_id], blocker, rank))

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
        "has_blocker_provenance",
        "blocker_provenance",
    ]
    result = pd.DataFrame(rows, columns=feature_columns)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, sep="\t", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2c pair features")
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--provenance-file", type=Path, default=None)
    args = parser.parse_args()
    build_pair_features(
        load_records([args.source1, args.source2, args.source3]),
        args.candidate_file,
        args.output_file,
        args.provenance_file,
    )


if __name__ == "__main__":
    main()
