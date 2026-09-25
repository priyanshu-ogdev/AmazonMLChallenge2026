"""Stage 4: deterministic per-S1 decision and submission assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def load_source1_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "entity_id" not in frame:
        raise ValueError(f"{path} must contain entity_id")
    ids = frame["entity_id"].tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate entity_id values")
    return ids


def assemble_matching_results(
    scored: pd.DataFrame, source1_ids: Iterable[str], threshold: float
) -> pd.DataFrame:
    """Apply one saved threshold, retaining explicit empty rows for singletons."""
    required = {"source1_entity_id", "candidate_entity_id", "calibrated_score"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored table is missing columns: {sorted(missing)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    source1_ids = list(source1_ids)
    if len(source1_ids) != len(set(source1_ids)):
        raise ValueError("source1_ids contains duplicates")
    if scored[["source1_entity_id", "candidate_entity_id"]].duplicated().any():
        raise ValueError("scored table contains duplicate candidate pairs")

    try:
        numeric_scores = pd.to_numeric(scored["calibrated_score"])
    except (ValueError, TypeError) as err:
        raise ValueError(f"calibrated_score contains non-numeric values: {err}") from err
    if numeric_scores.isna().any() or not np.isfinite(numeric_scores).all():
        raise ValueError("calibrated_score contains missing or non-finite values")

    allowed = set(source1_ids)
    unknown = set(scored["source1_entity_id"]) - allowed
    if unknown:
        raise ValueError(f"scored table contains unknown S1 IDs: {sorted(unknown)[:5]}")

    matches: Dict[str, list[str]] = {entity_id: [] for entity_id in source1_ids}
    s1_col = scored["source1_entity_id"].values
    cand_col = scored["candidate_entity_id"].values
    score_vals = numeric_scores.values
    for s1, cand, score in zip(s1_col, cand_col, score_vals):
        if score >= threshold:
            matches[s1].append(str(cand))
    return pd.DataFrame(
        {
            "source1_entity_id": source1_ids,
            "matched_entity_ids": [",".join(matches[entity_id]) for entity_id in source1_ids],
        }
    )


def write_matching_results(
    scored_file: Path, source1_file: Path, metadata_file: Path, output_file: Path
) -> None:
    scored = pd.read_csv(scored_file, sep="\t", dtype=str, keep_default_na=False)
    with open(metadata_file, encoding="utf-8") as handle:
        metadata = json.load(handle)
    result = assemble_matching_results(
        scored,
        load_source1_ids(source1_file),
        float(metadata["threshold"]),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble Stage 4 matching results")
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_matching_results(args.scored, args.source1, args.metadata, args.output)


if __name__ == "__main__":
    main()
