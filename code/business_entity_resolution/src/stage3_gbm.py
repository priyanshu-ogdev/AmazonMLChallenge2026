"""
Stage 3: entity-grouped GBM scoring, OOF calibration, and F0.5 thresholding.

The baseline deliberately uses XGBoost's regularized logistic objective.
Focal/beta-weighted objectives and DART are escalation experiments only and
are not enabled by this module's production path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.calibration import (
    apply_calibrator,
    apply_saved_calibrator,
    calibration_metrics,
    cross_fitted_metrics,
    fit_calibrator,
    serialize_calibrator,
)

RANDOM_SEED = 42
ID_COLUMNS = {"source1_entity_id", "candidate_entity_id", "label"}
DROP_CATEGORICAL = {
    "blocker_provenance",
    # Raw country strings: never a GBM predictor (would encode country identity).
    "source1_country",
    "candidate_country",
    # Canonical country strings: used for fold stratification only, not GBM input.
    "source1_canonical_country",
    "candidate_canonical_country",
}
DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": RANDOM_SEED,
}


def load_ground_truth(path: Path) -> Dict[str, set[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "matched_entity_ids"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    result = {}
    for row in frame.to_dict("records"):
        result[row["source1_entity_id"]] = {
            value.strip()
            for value in row["matched_entity_ids"].split(",")
            if value.strip()
        }
    return result


def attach_labels(
    features: pd.DataFrame, ground_truth: Dict[str, set[str]]
) -> pd.DataFrame:
    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")
    result = features.copy()
    result["label"] = [
        int(candidate_id in ground_truth.get(source1_id, set()))
        for source1_id, candidate_id in zip(
            result["source1_entity_id"], result["candidate_entity_id"]
        )
    ]
    return result


def merge_feature_file(features: pd.DataFrame, path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return features
    extra = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - set(extra.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    overlap = (set(features.columns) & set(extra.columns)) - required
    if overlap:
        raise ValueError(f"{path} would overwrite existing features: {sorted(overlap)}")
    return features.merge(
        extra, on=["source1_entity_id", "candidate_entity_id"], how="left", validate="one_to_one"
    )


def prepare_matrix(
    frame: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return numeric features; raw IDs and categorical provenance are excluded."""
    excluded = ID_COLUMNS | DROP_CATEGORICAL
    if feature_columns is None:
        candidates = [c for c in frame.columns if c not in excluded]
    else:
        candidates = list(feature_columns)
    missing = [c for c in candidates if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing requested feature columns: {missing}")
    matrix = frame[candidates].copy()
    for column in candidates:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    if matrix.isna().all(axis=0).any():
        bad = matrix.columns[matrix.isna().all(axis=0)].tolist()
        raise ValueError(f"Features are entirely non-numeric or missing: {bad}")
    return matrix.fillna(0.0), candidates


def _scale_pos_weight(y: np.ndarray) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Training fold must contain both positive and negative pairs")
    return negatives / positives


def macro_f05(
    source1_ids: Sequence[str],
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    all_source1_ids: Optional[Sequence[str]] = None,
) -> float:
    if len(source1_ids) != len(scores) or len(scores) != len(labels):
        raise ValueError("source1_ids, scores, and labels must have equal length")
    if not np.isfinite(np.asarray(scores, dtype=float)).all():
        raise ValueError("scores must contain only finite values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    grouped: Dict[str, List[Tuple[float, int]]] = {}
    for entity_id, score, label in zip(source1_ids, scores, labels):
        grouped.setdefault(entity_id, []).append((float(score), int(label)))
    if all_source1_ids is not None:
        for entity_id in all_source1_ids:
            grouped.setdefault(entity_id, [])
    entity_scores = []
    for pairs in grouped.values():
        predicted = {i for i, (score, _) in enumerate(pairs) if score >= threshold}
        actual = {i for i, (_, label) in enumerate(pairs) if label == 1}
        if not predicted and not actual:
            entity_scores.append(1.0)
            continue
        precision = len(predicted & actual) / len(predicted) if predicted else 0.0
        recall = len(predicted & actual) / len(actual) if actual else 0.0
        entity_scores.append(
            (1.25 * precision * recall / (0.25 * precision + recall))
            if precision + recall > 0
            else 0.0
        )
    return float(np.mean(entity_scores)) if entity_scores else 0.0


def choose_threshold(
    frame: pd.DataFrame,
    scores: np.ndarray,
    all_source1_ids: Optional[Sequence[str]] = None,
) -> Tuple[float, float]:
    if "source1_entity_id" not in frame or "label" not in frame:
        raise ValueError("threshold frame must contain source1_entity_id and label")
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("scores must match threshold frame length")
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    values = [
        macro_f05(
            frame["source1_entity_id"],
            scores,
            frame["label"],
            threshold,
            all_source1_ids=all_source1_ids,
        )
        for threshold in candidates
    ]
    index = int(np.argmax(values))
    return float(candidates[index]), float(values[index])


def score_candidates(
    feature_file: Path,
    artifact_dir: Path,
    qwen_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Score a candidate table with saved model, calibration, and threshold."""
    with open(artifact_dir / "stage3_metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    frame = pd.read_csv(feature_file, sep="\t", dtype=str, keep_default_na=False)
    frame = merge_feature_file(frame, qwen_file)
    matrix, _ = prepare_matrix(frame, metadata["feature_columns"])
    model = xgb.XGBClassifier()
    model.load_model(str(artifact_dir / "gbm.json"))
    raw = model.predict_proba(matrix)[:, 1]
    calibrated = apply_saved_calibrator(
        metadata["calibrator_parameters"], raw
    )
    result = frame[["source1_entity_id", "candidate_entity_id"]].copy()
    result["raw_score"] = raw
    result["calibrated_score"] = calibrated
    result["is_match"] = calibrated >= float(metadata["threshold"])
    return result


def train_oof(
    frame: pd.DataFrame,
    n_splits: int = 5,
    params: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Train grouped OOF models; no S1 entity appears in both train and valid."""
    matrix, columns = prepare_matrix(frame)
    X = matrix.to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int32)
    groups = frame["source1_entity_id"].to_numpy()
    # Prefer canonical country for stratification: it is stable across alias
    # variants (US/USA/us → 'us', France/FR → 'france') so it produces
    # balanced folds. Fall back to raw if canonical column is absent.
    if "source1_canonical_country" in frame.columns:
        countries = frame["source1_canonical_country"].astype(str)
    elif "source1_country" in frame.columns:
        countries = frame["source1_country"].astype(str)
    else:
        countries = pd.Series(["unknown"] * len(frame), index=frame.index)
    stratify = countries + ":" + frame["label"].astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    oof = np.zeros(len(frame), dtype=float)
    fold_info = []
    base = dict(DEFAULT_PARAMS)
    if params:
        base.update(params)
    try:
        splits = splitter.split(X, stratify, groups)
        split_list = list(splits)
    except ValueError:
        # Small data can lack enough examples in a country/label stratum.
        # Preserve entity grouping and class stratification rather than fail
        # or silently fall back to pair-level StratifiedKFold.
        split_list = list(splitter.split(X, y, groups))
    for fold, (train_idx, valid_idx) in enumerate(split_list):
        fold_params = dict(base)
        fold_params["scale_pos_weight"] = _scale_pos_weight(y[train_idx])
        model = xgb.XGBClassifier(
            n_estimators=1000,
            early_stopping_rounds=50,
            **fold_params,
        )
        model.fit(
            X[train_idx],
            y[train_idx],
            eval_set=[(X[valid_idx], y[valid_idx])],
            verbose=False,
        )
        oof[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]
        fold_info.append(
            {
                "fold": fold,
                "train_entities": int(len(set(groups[train_idx]))),
                "valid_entities": int(len(set(groups[valid_idx]))),
                "best_iteration": int(model.best_iteration or 0),
                "average_precision": float(
                    average_precision_score(y[valid_idx], oof[valid_idx])
                ),
            }
        )
    result = frame.copy()
    result["oof_score"] = oof
    return result, columns, {"folds": fold_info}


def fit_final(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    params: Optional[Dict] = None,
    n_estimators: int = 300,
) -> xgb.XGBClassifier:
    matrix, _ = prepare_matrix(frame, feature_columns)
    y = frame["label"].to_numpy(dtype=np.int32)
    final_params = dict(DEFAULT_PARAMS)
    if params:
        final_params.update(params)
    final_params["scale_pos_weight"] = _scale_pos_weight(y)
    model = xgb.XGBClassifier(n_estimators=n_estimators, **final_params)
    model.fit(matrix, y, verbose=False)
    return model


def run_training(
    feature_file: Path,
    ground_truth_file: Path,
    output_dir: Path,
    qwen_file: Optional[Path] = None,
) -> Dict:
    frame = pd.read_csv(feature_file, sep="\t", dtype=str, keep_default_na=False)
    frame = merge_feature_file(frame, qwen_file)
    # Explicitly assign ground_truth so it is available for choose_threshold's
    # all_source1_ids argument (includes singletons with empty match lists).
    ground_truth = load_ground_truth(ground_truth_file)
    labeled = attach_labels(frame, ground_truth)
    oof, columns, diagnostics = train_oof(labeled)
    calibrator = fit_calibrator(
        oof["oof_score"].to_numpy(), oof["label"].to_numpy()
    )
    calibrated = apply_calibrator(calibrator, oof["oof_score"].to_numpy())
    # Pass the full S1 universe (dict keys) so singleton entities with zero
    # candidates are included in the macro-F0.5 threshold search.
    threshold, f05 = choose_threshold(
        oof, calibrated, all_source1_ids=list(ground_truth)
    )
    model = fit_final(labeled, columns, n_estimators=max(50, int(np.mean(
        [fold["best_iteration"] for fold in diagnostics["folds"]]
    ) * 1.1)))
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "gbm.json"))
    oof.assign(calibrated_score=calibrated).to_csv(
        output_dir / "oof_predictions.tsv", sep="\t", index=False
    )
    metadata = {
        "feature_columns": columns,
        "calibrator": calibrator[0],
        "calibrator_parameters": serialize_calibrator(calibrator),
        "calibration_metrics": calibration_metrics(
            oof["label"].to_numpy(), calibrated
        ),
        "cross_fitted_calibration_metrics": cross_fitted_metrics(
            oof["oof_score"].to_numpy(),
            oof["label"].to_numpy(),
            oof["source1_entity_id"].to_numpy(),
        ),
        "oof_positive_count": int(labeled["label"].sum()),
        "threshold": threshold,
        "macro_f05": f05,
        "average_precision": float(
            average_precision_score(labeled["label"], oof["oof_score"])
        ),
        "params": DEFAULT_PARAMS,
        "final_n_estimators": int(model.get_params()["n_estimators"]),
        "diagnostics": diagnostics,
    }
    with open(output_dir / "stage3_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 3 grouped GBM")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-features", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_training(
        args.features, args.ground_truth, args.output_dir, args.qwen_features
    ), indent=2))


if __name__ == "__main__":
    main()
