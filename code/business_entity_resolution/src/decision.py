"""Stage 4: deterministic per-S1 decision and submission assembly."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def load_source1_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    if "entity_id" not in frame:
        raise ValueError(f"{path} must contain entity_id")
    ids = [str(x).strip() for x in frame["entity_id"] if str(x).strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate entity_id values")
    return ids


def assemble_matching_results(
    scored: pd.DataFrame,
    source1_ids: Iterable[str],
    threshold: float,
    injective: bool = True,
) -> pd.DataFrame:
    """Apply one saved threshold, retaining explicit empty rows for singletons.

    When injective=True (default, per docs/09_stage4_decision_and_singletons.md §3),
    enforces greedy 1-to-N bipartite matching: each S2/S3 candidate record can belong
    to at most one S1 entity, resolving competing claims in descending order of
    calibrated match probability.
    """
    required = {"source1_entity_id", "candidate_entity_id", "calibrated_score"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored table is missing columns: {sorted(missing)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    source1_ids = [str(x).strip() for x in source1_ids]
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

    # Filter to pairs above or at threshold
    mask = score_vals >= threshold
    if mask.any():
        passed_s1 = s1_col[mask]
        passed_cand = cand_col[mask]
        passed_scores = score_vals[mask]

        if injective:
            # Sort globally in descending order of calibrated probability (stable sort)
            order = np.argsort(-passed_scores, kind="stable")
            claimed_targets: set[str] = set()
            for idx in order:
                cand = str(passed_cand[idx]).strip()
                s1 = str(passed_s1[idx]).strip()
                if cand == s1:
                    continue
                if cand not in claimed_targets:
                    matches[s1].append(cand)
                    claimed_targets.add(cand)
        else:
            for s1_val, cand_val in zip(passed_s1, passed_cand):
                s1 = str(s1_val).strip()
                cand = str(cand_val).strip()
                if cand == s1:
                    continue
                if cand not in matches[s1]:
                    matches[s1].append(cand)

    return pd.DataFrame(
        {
            "source1_entity_id": source1_ids,
            "matched_entity_ids": [",".join(matches[entity_id]) for entity_id in source1_ids],
        }
    )


def write_matching_results(
    scored_file: Path,
    source1_file: Path,
    metadata_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    threshold: Optional[float] = None,
    injective: bool = True,
) -> None:
    if output_file is None:
        raise ValueError("output_file must be provided")

    scored = pd.read_csv(scored_file, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    if threshold is not None:
        effective_threshold = float(threshold)
    elif metadata_file is not None:
        with open(metadata_file, encoding="utf-8") as handle:
            metadata = json.load(handle)
        effective_threshold = float(metadata["threshold"])
    else:
        raise ValueError("Either metadata_file or threshold must be provided")

    result = assemble_matching_results(
        scored,
        load_source1_ids(source1_file),
        effective_threshold,
        injective=injective,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble Stage 4 matching results")
    parser.add_argument("--scored", type=Path, required=True, help="Path to scored candidates TSV")
    parser.add_argument("--source1", type=Path, required=True, help="Path to Source-1 TSV (to extract full S1 ID list)")
    parser.add_argument("--metadata", type=Path, default=None, help="Path to stage3_metadata.json (required unless --threshold is given)")
    parser.add_argument("--output", type=Path, required=True, help="Path to output matching_results.tsv")
    parser.add_argument("--threshold", type=float, default=None, help="Explicit decision threshold (overrides metadata threshold)")
    parser.add_argument(
        "--no-injective",
        dest="injective",
        action="store_false",
        default=True,
        help="Disable greedy 1-to-N injective bipartite matching (default: enabled)",
    )
    args = parser.parse_args()
    if args.metadata is None and args.threshold is None:
        parser.error("Either --metadata or --threshold must be provided")
    write_matching_results(
        args.scored,
        args.source1,
        metadata_file=args.metadata,
        output_file=args.output,
        threshold=args.threshold,
        injective=args.injective,
    )


if __name__ == "__main__":
    main()
