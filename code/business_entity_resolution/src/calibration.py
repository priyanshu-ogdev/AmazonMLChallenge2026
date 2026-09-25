"""Leakage-safe score calibration for the Stage 3 OOF protocol.

The GBM must produce scores out of fold. This module fits a calibrator only
after that boundary, validates its inputs, reports reliability diagnostics, and
serializes parameters without pickle.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold


def _validate(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(scores) != len(labels) or not len(scores):
        raise ValueError("scores and labels must be non-empty and equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    if len(np.unique(labels)) < 2:
        raise ValueError("calibration requires both positive and negative labels")
    return np.clip(scores, 0.0, 1.0), labels


def fit_calibrator(scores: np.ndarray, labels: np.ndarray):
    """Select Platt or isotonic calibration using OOF scores only."""
    scores, labels = _validate(scores, labels)
    positives = int(labels.sum())
    if positives < 1000:
        model = LogisticRegression(max_iter=1000).fit(scores[:, None], labels)
        return "platt", model
    if positives > 10000:
        return "isotonic", IsotonicRegression(out_of_bounds="clip").fit(scores, labels)

    platt = LogisticRegression(max_iter=1000).fit(scores[:, None], labels)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
    platt_loss = log_loss(labels, platt.predict_proba(scores[:, None])[:, 1])
    isotonic_loss = log_loss(labels, isotonic.predict(scores))
    return ("platt", platt) if platt_loss <= isotonic_loss else ("isotonic", isotonic)


def apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    name, model = calibrator
    if name == "platt":
        result = model.predict_proba(np.clip(scores, 0.0, 1.0)[:, None])[:, 1]
    elif name == "isotonic":
        result = np.asarray(model.predict(np.clip(scores, 0.0, 1.0)), dtype=float)
    else:
        raise ValueError(f"Unsupported calibrator: {name}")
    return np.clip(result, 0.0, 1.0)


def serialize_calibrator(calibrator) -> Dict:
    name, model = calibrator
    if name == "platt":
        return {
            "name": name,
            "coef": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        }
    if name == "isotonic":
        return {
            "name": name,
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        }
    raise ValueError(f"Unsupported calibrator: {name}")


def apply_saved_calibrator(parameters: Dict, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    name = parameters.get("name")
    clipped = np.clip(scores, 0.0, 1.0)
    if name == "platt":
        logits = float(parameters["coef"]) * clipped + float(parameters["intercept"])
        result = 1.0 / (1.0 + np.exp(-logits))
    elif name == "isotonic":
        result = np.interp(
            clipped,
            np.asarray(parameters["x_thresholds"], dtype=float),
            np.asarray(parameters["y_thresholds"], dtype=float),
        )
    else:
        raise ValueError(f"Unsupported saved calibrator: {name}")
    return np.clip(result, 0.0, 1.0)


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> Dict:
    """Return pair-weighted reliability metrics for diagnostics."""
    labels = np.asarray(labels, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    _, labels = _validate(probabilities, labels)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    reliability = []
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        )
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        outcome = float(labels[mask].mean())
        ece += float(mask.mean()) * abs(confidence - outcome)
        reliability.append({"count": int(mask.sum()), "confidence": confidence, "outcome": outcome})
    return {
        "log_loss": float(log_loss(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": float(ece),
        "reliability": reliability,
    }


def cross_fitted_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Dict:
    """Estimate calibration quality on groups not used to fit each calibrator."""
    scores, labels = _validate(scores, labels)
    groups = np.asarray(groups).reshape(-1)
    if len(groups) != len(scores):
        raise ValueError("groups must have the same length as scores")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("not enough groups for requested calibration folds")
    predictions = np.empty(len(scores), dtype=float)
    for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(
        scores, labels, groups
    ):
        calibrator = fit_calibrator(scores[train_idx], labels[train_idx])
        predictions[valid_idx] = apply_calibrator(calibrator, scores[valid_idx])
    return calibration_metrics(labels, predictions)
