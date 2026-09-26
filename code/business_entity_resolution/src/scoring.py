"""
Stage 3: entity-grouped GBM scoring, OOF calibration, and F0.5 thresholding.

Implements the complete Layer 3 pipeline:
- Grouped-OOF training with nested StratifiedGroupKFold for leakage-free early stopping
- Monotonic constraints preserving feature directions and dynamic TF-IDF alignment
- Country-equal masking regularization (-1 sentinel matching prepare_matrix missing fill)
- Native support for standard GBDT ('gbtree') and DART tree dropout ('dart')
- Comparative cross-country diagnostic (--compare-dart) evaluating US <-> India generalization
- Leak-safe Platt and Isotonic probability calibration with JSON parameter serialization
- Macro-F0.5 threshold optimization rewarding true singleton non-matches
"""

from __future__ import annotations

import argparse
import json
import logging
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

logger = logging.getLogger(__name__)

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
    "booster": "gbtree",
    "eta": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": RANDOM_SEED,
}


def build_monotonic_constraints(columns: Sequence[str]) -> Tuple[int, ...]:
    """
    Build monotonic constraints matching feature column order.
    +1: higher value must not decrease predicted probability (similarities, blocker scores).
    -1: higher value must not increase predicted probability (contradictions, ranks, score diffs).
     0: unconstrained (country flags, missing indicators, source flags, candidate counts).

    Per specs:
    - Missing indicators (*_missing, *_missing_either) are strictly unconstrained (0).
    - country_equal is NEVER monotonic (0) to prevent shortcut learning.
    - best_blocker_score_diff is strictly negative (-1).
    - best_blocker_score is strictly positive (+1).
    """
    negative_features = {
        "same_name_different_address",
        "same_address_different_name",
        "name_length_abs_diff",
        "address_length_abs_diff",
        "best_blocker_score_diff",
        "candidate_rank",
        "rank_margin_from_best",
    }
    positive_features = {
        "bge_cosine",
        "qwen_cosine",
        "qwen_matcher_prob",
        "tfidf_cosine",
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
        "best_blocker_score",
        "blocker_count",
        "has_blocker_provenance",
    }
    constraints = []
    for col in columns:
        if (
            col.endswith("_missing")
            or col.endswith("_missing_either")
            or col.startswith("country_")
            or col.startswith("source_")
            or col in ("candidate_count_for_s1",)
        ):
            constraints.append(0)
        elif col in negative_features:
            constraints.append(-1)
        elif col in positive_features:
            constraints.append(1)
        elif any(col.startswith(p + "_") for p in negative_features):
            constraints.append(-1)
        elif any(col.startswith(p + "_") for p in positive_features):
            constraints.append(1)
        else:
            constraints.append(0)
    return tuple(constraints)


def apply_country_masking(
    matrix: pd.DataFrame,
    mask_rate: float = 0.15,
    random_state: Optional[np.random.RandomState] = None,
) -> pd.DataFrame:
    """
    Stochastically mask country_equal to -1 (missing) and country_equal_missing to 1
    during training on a fraction of rows.

    Prevents shortcut learning on US/India pairs and forces the tree splits to learn
    generalizable lexical/dense features that work on unseen countries (e.g. France).
    Never applied during evaluation or test inference.
    """
    if mask_rate <= 0.0 or "country_equal" not in matrix.columns:
        return matrix
    rng = random_state or np.random.RandomState(RANDOM_SEED)
    mask = rng.rand(len(matrix)) < mask_rate
    if not mask.any():
        return matrix
    masked = matrix.copy()
    masked.loc[mask, "country_equal"] = -1.0
    if "country_equal_missing" in masked.columns:
        masked.loc[mask, "country_equal_missing"] = 1.0
    return masked


def compute_fold_safe_tfidf_scores(
    pairs: pd.DataFrame,
    records: Dict[str, str],
    vectorizer: Optional[object] = None,
    entities_to_fit: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, object]:
    """
    Compute leakage-free character n-gram TF-IDF cosine similarity for candidate pairs.
    If vectorizer is None, fits strictly on entities_to_fit only (training fold boundary).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    if vectorizer is None:
        fit_texts = [records.get(eid, "") for eid in (entities_to_fit or set())]
        if not fit_texts:
            fit_texts = [""]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
        )
        vectorizer.fit(fit_texts)

    s1_ids = pairs["source1_entity_id"].tolist()
    cand_ids = pairs["candidate_entity_id"].tolist()

    unique_ids = list(set(s1_ids) | set(cand_ids))
    unique_texts = [records.get(eid, "") for eid in unique_ids]
    sparse_embs = vectorizer.transform(unique_texts)
    sparse_embs = normalize(sparse_embs, norm="l2")

    id_to_idx = {eid: idx for idx, eid in enumerate(unique_ids)}
    s1_indices = [id_to_idx[eid] for eid in s1_ids]
    cand_indices = [id_to_idx[eid] for eid in cand_ids]

    s1_mat = sparse_embs[s1_indices]
    cand_mat = sparse_embs[cand_indices]
    cosines = np.asarray(s1_mat.multiply(cand_mat).sum(axis=1)).reshape(-1)
    return cosines, vectorizer


def extract_feature_importances(
    model: xgb.XGBClassifier, feature_names: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Extract feature importance by gain and weight."""
    try:
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        feat_map = {f"f{i}": col for i, col in enumerate(feature_names)}
        clean_gain = {feat_map.get(k, k): float(v) for k, v in gain.items()}
        clean_weight = {feat_map.get(k, k): float(v) for k, v in weight.items()}
        sorted_gain = dict(sorted(clean_gain.items(), key=lambda item: item[1], reverse=True))
        sorted_weight = dict(sorted(clean_weight.items(), key=lambda item: item[1], reverse=True))
        return {"gain": sorted_gain, "weight": sorted_weight}
    except Exception as e:
        logger.warning(f"Failed to extract feature importances: {e}")
        return {"error": str(e)}


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
    merged = features.merge(
        extra, on=["source1_entity_id", "candidate_entity_id"], how="left", validate="one_to_one"
    )
    for col in extra.columns:
        if col.endswith("_missing") or col.endswith("_missing_either"):
            merged[col] = merged[col].fillna("1")
    return merged


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
        if column.endswith("_missing") or column.endswith("_missing_either"):
            matrix[column] = matrix[column].fillna(1.0)
        elif column in ("candidate_rank", "rank_margin_from_best"):
            matrix[column] = matrix[column].fillna(999.0)
        elif column in ("best_blocker_score", "country_equal"):
            matrix[column] = matrix[column].fillna(-1.0)
        else:
            matrix[column] = matrix[column].fillna(0.0)
    if matrix.isna().all(axis=0).any():
        bad = matrix.columns[matrix.isna().all(axis=0)].tolist()
        raise ValueError(f"Features are entirely non-numeric or missing: {bad}")
    return matrix, candidates


def _scale_pos_weight(y: np.ndarray) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Training fold must contain both positive and negative pairs")
    return negatives / positives


def compute_entity_f05(pred_count: int, actual_count: int, tp: int) -> float:
    """
    Compute F0.5 score for a single entity, accounting for true singletons and blocking misses.

    - True singleton (pred=0, actual=0): 1.0 (correctly identified as singleton)
    - Missed matches (pred=0, actual>0): 0.0 (blocking miss or threshold cut)
    - False positives on singleton (pred>0, actual=0): 0.0 (precision=0)
    - Non-zero candidates: standard F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if pred_count == 0 and actual_count == 0:
        return 1.0
    if pred_count == 0 or actual_count == 0:
        return 0.0
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / actual_count if actual_count else 0.0
    if precision + recall > 0.0:
        return (1.25 * precision * recall) / (0.25 * precision + recall)
    return 0.0


def macro_f05(
    source1_ids: Sequence[str],
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    all_source1_ids: Optional[Sequence[str]] = None,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
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
    if ground_truth is not None:
        for entity_id in ground_truth:
            grouped.setdefault(entity_id, [])
    elif all_source1_ids is not None:
        for entity_id in all_source1_ids:
            grouped.setdefault(entity_id, [])
    entity_scores = []
    for entity_id, pairs in grouped.items():
        predicted = {i for i, (score, _) in enumerate(pairs) if score >= threshold}
        # When ground_truth is provided, use the true ground-truth match count for the entity.
        # This prevents awarding 1.0 to entities that blocking completely failed to retrieve.
        if ground_truth is not None and entity_id in ground_truth:
            actual_count = len(ground_truth[entity_id])
        else:
            actual_count = sum(1 for _, label in pairs if label == 1)
        pred_count = len(predicted)
        tp = sum(1 for i in predicted if pairs[i][1] == 1)
        entity_scores.append(compute_entity_f05(pred_count, actual_count, tp))

    return float(np.mean(entity_scores)) if entity_scores else 0.0


def choose_threshold(
    frame: pd.DataFrame,
    scores: np.ndarray,
    all_source1_ids: Optional[Sequence[str]] = None,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[float, float]:
    if "source1_entity_id" not in frame or "label" not in frame:
        raise ValueError("threshold frame must contain source1_entity_id and label")
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("scores must match threshold frame length")
    # Grid search 0.05 to 0.95 with step 0.01 per parameters table + unique scores/quantiles + boundaries
    grid = np.arange(0.05, 0.96, 0.01)
    if len(scores) <= 1000:
        candidates = np.unique(np.r_[0.0, grid, scores, 1.0])
    else:
        score_samples = np.quantile(scores, np.linspace(0.01, 0.99, 100))
        candidates = np.unique(np.r_[0.0, grid, score_samples, 1.0])

    # Pre-group entities once for fast threshold scanning across candidates
    s1_col = frame["source1_entity_id"].values
    labels_col = frame["label"].values
    active_grouped: Dict[str, Tuple[List[float], List[int]]] = {}
    for eid, s, l in zip(s1_col, scores, labels_col):
        if eid not in active_grouped:
            active_grouped[eid] = ([], [])
        active_grouped[eid][0].append(float(s))
        active_grouped[eid][1].append(int(l))

    # Precompute actual counts and zero-pair singleton count
    if ground_truth is not None:
        all_eids = set(ground_truth.keys())
    elif all_source1_ids is not None:
        all_eids = set(all_source1_ids)
    else:
        all_eids = set(active_grouped.keys())

    zero_pair_eids = all_eids - set(active_grouped.keys())
    zero_pair_singleton_count = 0
    if ground_truth is not None:
        for eid in zero_pair_eids:
            if len(ground_truth[eid]) == 0:
                zero_pair_singleton_count += 1
    elif all_source1_ids is not None:
        zero_pair_singleton_count = len(zero_pair_eids)

    total_entities = len(all_eids) if all_eids else len(active_grouped)

    active_records = []
    for eid, (s_list, l_list) in active_grouped.items():
        if ground_truth is not None and eid in ground_truth:
            actual_count = len(ground_truth[eid])
        else:
            actual_count = sum(l_list)
        active_records.append((
            np.asarray(s_list, dtype=float),
            np.asarray(l_list, dtype=int),
            actual_count,
        ))

    def _eval_threshold(th: float) -> float:
        if total_entities == 0:
            return 0.0
        total_f05 = float(zero_pair_singleton_count)
        for s_arr, l_arr, actual_count in active_records:
            pred_mask = s_arr >= th
            pred_count = int(pred_mask.sum())
            tp = int(l_arr[pred_mask].sum())
            total_f05 += compute_entity_f05(pred_count, actual_count, tp)
        return total_f05 / total_entities

    values = [_eval_threshold(threshold) for threshold in candidates]
    index = int(np.argmax(values))
    return float(candidates[index]), float(values[index])


def score_candidates(
    feature_file: Path,
    artifact_dir: Path,
    qwen_file: Optional[Path] = None,
    bge_file: Optional[Path] = None,
    qwen_matcher_file: Optional[Path] = None,
    records: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Score a candidate table with saved model, calibration, and threshold."""
    meta_path = artifact_dir / "stage3_metadata.json"
    gbm_path = artifact_dir / "gbm.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing stage3_metadata.json in artifact directory: {artifact_dir}. "
            "Run Stage 3 training before scoring candidates."
        )
    if not gbm_path.exists():
        raise FileNotFoundError(
            f"Missing gbm.json model file in artifact directory: {artifact_dir}. "
            "Run Stage 3 training before scoring candidates."
        )

    with open(meta_path, encoding="utf-8") as handle:
        metadata = json.load(handle)

    required_keys = {"feature_columns", "calibrator_parameters", "threshold"}
    missing_keys = required_keys - set(metadata.keys())
    if missing_keys:
        raise ValueError(
            f"Corrupt stage3_metadata.json in {artifact_dir}: missing required keys {sorted(missing_keys)}"
        )

    frame = pd.read_csv(feature_file, sep="\t", dtype=str, keep_default_na=False)
    frame = merge_feature_file(frame, qwen_file)
    frame = merge_feature_file(frame, bge_file)
    # qwen_matcher_file is Stage 2b (stretch, docs/07_stage2b_...): pass None
    # whenever the held-out-country gate hasn't been run or has failed. The
    # GBM's feature_columns metadata from training already reflects whether
    # this column existed at fit time, so no special-casing is needed here --
    # prepare_matrix() below will raise clearly if the saved model expects a
    # column this call didn't provide, rather than silently scoring without it.
    frame = merge_feature_file(frame, qwen_matcher_file)

    tfidf_path = artifact_dir / "tfidf_vectorizer.joblib"
    if tfidf_path.exists():
        if records is None:
            raise ValueError(
                "The saved model includes a 'tfidf_cosine' feature but no entity "
                "records were provided to score_candidates(). Pass records= to "
                "recompute TF-IDF cosine similarities at inference time."
            )
        import joblib
        vectorizer = joblib.load(tfidf_path)
        tfidf_col, _ = compute_fold_safe_tfidf_scores(
            frame, records, vectorizer=vectorizer
        )
        frame["tfidf_cosine"] = tfidf_col

    matrix, _ = prepare_matrix(frame, metadata["feature_columns"])
    model_booster = metadata.get("booster") or metadata.get("params", {}).get("booster", "gbtree")
    model = xgb.XGBClassifier(booster=model_booster)
    model.load_model(str(gbm_path))
    actual_booster = model.get_params().get("booster")
    assert actual_booster == model_booster, (
        f"Expected loaded model booster to be '{model_booster}', got '{actual_booster}'"
    )
    raw = model.predict_proba(matrix)[:, 1]
    calibrated = apply_saved_calibrator(
        metadata["calibrator_parameters"], raw
    )
    threshold = float(metadata["threshold"])
    result = frame[["source1_entity_id", "candidate_entity_id"]].copy()
    result["raw_score"] = raw
    result["calibrated_score"] = calibrated
    result["is_match"] = calibrated >= threshold

    logger.info(
        "Scored %d candidate pairs with booster=%s, threshold=%.4f (matches=%d)",
        len(result),
        model_booster,
        threshold,
        int(result["is_match"].sum()),
    )
    return result


def evaluate_held_out_country_diagnostic(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    params: Optional[Dict] = None,
    country_mask_rate: float = 0.0,
    use_monotone_constraints: bool = False,
    n_estimators: int = 300,
    early_stopping_rounds: int = 30,
) -> Dict[str, object]:
    """
    Two-way held-out-country generalization check (e.g. US -> India and India -> US).
    Directly measures whether the GBM has overfit to domestic country noise patterns.
    """
    country_col = (
        "source1_canonical_country"
        if "source1_canonical_country" in frame.columns
        else "source1_country"
    )
    if country_col not in frame.columns:
        return {"status": "skipped", "reason": "No country column available"}

    counts = frame.groupby([country_col, "label"]).size().unstack(fill_value=0)
    valid_countries = [
        c
        for c in counts.index
        if c != "" and counts.loc[c, 0] > 0 and counts.loc[c, 1] > 0
    ]
    if len(valid_countries) < 2:
        return {
            "status": "skipped",
            "reason": f"Need at least 2 countries with pos and neg labels; found: {valid_countries}",
        }

    base_params = dict(DEFAULT_PARAMS)
    if params:
        base_params.update(params)
    if use_monotone_constraints:
        base_params["monotone_constraints"] = build_monotonic_constraints(feature_columns)

    diagnostic_results = {}
    cross_ap = []
    if len(valid_countries) > 2:
        logger.info(
            "Found %d qualifying countries for held-out diagnostic: %s. Evaluating all.",
            len(valid_countries),
            valid_countries,
        )
    for eval_country in valid_countries:
        train_mask = frame[country_col] != eval_country
        valid_mask = frame[country_col] == eval_country
        if not train_mask.any() or not valid_mask.any():
            continue

        train_df = frame[train_mask]
        valid_df = frame[valid_mask]

        train_mat, _ = prepare_matrix(train_df, feature_columns)
        valid_mat, _ = prepare_matrix(valid_df, feature_columns)
        if country_mask_rate > 0.0:
            train_mat = apply_country_masking(
                train_mat,
                mask_rate=country_mask_rate,
                random_state=np.random.RandomState(RANDOM_SEED),
            )
        y_train = train_df["label"].to_numpy(dtype=np.int32)
        y_valid = valid_df["label"].to_numpy(dtype=np.int32)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
            continue

        diag_params = dict(base_params)

        # Early stopping must be chosen on an inner holdout of the training country,
        # never on the validation country itself (which would cause model-selection leakage).
        train_groups = train_df["source1_entity_id"].to_numpy()
        diag_splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_SEED
        )
        inner_split_found = False
        try:
            strat_diag = y_train.astype(str)
            for inner_tr, inner_va in diag_splitter.split(train_mat, strat_diag, train_groups):
                if (
                    (y_train[inner_tr] == 1).any()
                    and (y_train[inner_tr] == 0).any()
                    and (y_train[inner_va] == 1).any()
                    and (y_train[inner_va] == 0).any()
                ):
                    inner_split_found = True
                    break
        except Exception:
            inner_split_found = False

        if inner_split_found:
            X_fit = train_mat.iloc[inner_tr]
            y_fit = y_train[inner_tr]
            X_es_val = train_mat.iloc[inner_va]
            y_es_val = y_train[inner_va]
            diag_params["scale_pos_weight"] = _scale_pos_weight(y_fit)
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                **diag_params,
            )
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_es_val, y_es_val)],
                verbose=False,
            )
        else:
            diag_params["scale_pos_weight"] = _scale_pos_weight(y_train)
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                **diag_params,
            )
            model.fit(train_mat, y_train, verbose=False)

        val_preds = model.predict_proba(valid_mat)[:, 1]
        ap = float(average_precision_score(y_valid, val_preds))
        cross_ap.append(ap)
        try:
            best_iter = int(model.best_iteration)
        except (AttributeError, TypeError, ValueError):
            best_iter = int(getattr(model, "n_estimators", 300) or 0)

        diagnostic_results[f"train_others_eval_{eval_country}"] = {
            "eval_country": str(eval_country),
            "train_entities": int(train_df["source1_entity_id"].nunique()),
            "valid_entities": int(valid_df["source1_entity_id"].nunique()),
            "average_precision": ap,
            "best_iteration": best_iter,
        }

    if cross_ap:
        diagnostic_results["mean_held_out_country_ap"] = float(np.mean(cross_ap))
    return diagnostic_results


def train_oof(
    frame: pd.DataFrame,
    n_splits: int = 5,
    params: Optional[Dict] = None,
    country_mask_rate: float = 0.15,
    use_monotone_constraints: bool = False,
    records: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Train grouped OOF models; no S1 entity appears in both train and valid."""
    matrix, columns = prepare_matrix(frame)
    y = frame["label"].to_numpy(dtype=np.int32)
    groups = frame["source1_entity_id"].to_numpy()
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
    if use_monotone_constraints and records is None:
        base["monotone_constraints"] = build_monotonic_constraints(columns)

    X_base = matrix.to_numpy(dtype=np.float32)
    try:
        splits = splitter.split(X_base, stratify, groups)
        split_list = list(splits)
    except ValueError:
        split_list = list(splitter.split(X_base, y, groups))

    for fold, (train_idx, valid_idx) in enumerate(split_list):
        fold_params = dict(base)
        fold_params["scale_pos_weight"] = _scale_pos_weight(y[train_idx])

        train_matrix = matrix.iloc[train_idx].copy()
        valid_matrix = matrix.iloc[valid_idx].copy()

        # Fold-safe TF-IDF: fit strictly on training fold entities
        if records is not None:
            train_eids = set(groups[train_idx]) | set(frame.iloc[train_idx]["candidate_entity_id"])
            train_tfidf, fold_vec = compute_fold_safe_tfidf_scores(
                frame.iloc[train_idx], records, entities_to_fit=train_eids
            )
            valid_tfidf, _ = compute_fold_safe_tfidf_scores(
                frame.iloc[valid_idx], records, vectorizer=fold_vec
            )
            train_matrix["tfidf_cosine"] = train_tfidf
            valid_matrix["tfidf_cosine"] = valid_tfidf
            fold_columns = list(columns) + ["tfidf_cosine"]
            if use_monotone_constraints:
                fold_params["monotone_constraints"] = build_monotonic_constraints(fold_columns)

        if country_mask_rate > 0.0:
            train_matrix = apply_country_masking(
                train_matrix,
                mask_rate=country_mask_rate,
                random_state=np.random.RandomState(RANDOM_SEED + fold),
            )

        X_train = train_matrix.to_numpy(dtype=np.float32)
        X_valid = valid_matrix.to_numpy(dtype=np.float32)

        # Model-selection leakage fix:
        # Carve an early-stopping holdout out of train_idx only (grouped by entity).
        # X_valid is strictly reserved for the final out-of-fold prediction.
        train_groups = groups[train_idx]
        train_y = y[train_idx]
        inner_splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_SEED + fold
        )
        inner_split_found = False
        try:
            strat_inner = train_y.astype(str)
            for inner_tr, inner_va in inner_splitter.split(X_train, strat_inner, train_groups):
                if (
                    (train_y[inner_tr] == 1).any()
                    and (train_y[inner_tr] == 0).any()
                    and (train_y[inner_va] == 1).any()
                    and (train_y[inner_va] == 0).any()
                ):
                    inner_split_found = True
                    break
        except Exception:
            inner_split_found = False

        if inner_split_found:
            X_fit = X_train[inner_tr]
            y_fit = train_y[inner_tr]
            X_es_val = X_train[inner_va]
            y_es_val = train_y[inner_va]
            fold_params["scale_pos_weight"] = _scale_pos_weight(y_fit)
            model = xgb.XGBClassifier(
                n_estimators=1000,
                early_stopping_rounds=50,
                **fold_params,
            )
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_es_val, y_es_val)],
                verbose=False,
            )
        else:
            fold_params["scale_pos_weight"] = _scale_pos_weight(train_y)
            model = xgb.XGBClassifier(
                n_estimators=300,
                **fold_params,
            )
            model.fit(X_train, train_y, verbose=False)

        try:
            best_iter = int(model.best_iteration)
        except (AttributeError, TypeError, ValueError):
            best_iter = int(getattr(model, "n_estimators", 300) or 0)

        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        fold_info.append(
            {
                "fold": fold,
                "train_entities": int(len(set(groups[train_idx]))),
                "valid_entities": int(len(set(groups[valid_idx]))),
                "best_iteration": best_iter,
                "average_precision": float(
                    average_precision_score(y[valid_idx], oof[valid_idx])
                ),
            }
        )
    result = frame.copy()
    result["oof_score"] = oof
    final_columns = list(columns) + (["tfidf_cosine"] if records is not None else [])
    return result, final_columns, {"folds": fold_info}


def fit_final(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    params: Optional[Dict] = None,
    n_estimators: int = 300,
    country_mask_rate: float = 0.15,
    use_monotone_constraints: bool = False,
    records: Optional[Dict[str, str]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[xgb.XGBClassifier, List[str]]:
    cols = list(feature_columns)
    if records is not None:
        import joblib
        all_eids = set(frame["source1_entity_id"]) | set(frame["candidate_entity_id"])
        tfidf_col, final_vec = compute_fold_safe_tfidf_scores(
            frame, records, entities_to_fit=all_eids
        )
        frame = frame.assign(tfidf_cosine=tfidf_col)
        if "tfidf_cosine" not in cols:
            cols.append("tfidf_cosine")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_vec, output_dir / "tfidf_vectorizer.joblib")

    matrix, _ = prepare_matrix(frame, cols)
    y = frame["label"].to_numpy(dtype=np.int32)
    final_params = dict(DEFAULT_PARAMS)
    if params:
        final_params.update(params)
    if use_monotone_constraints:
        final_params["monotone_constraints"] = build_monotonic_constraints(cols)
    final_params["scale_pos_weight"] = _scale_pos_weight(y)
    if country_mask_rate > 0.0:
        matrix = apply_country_masking(
            matrix,
            mask_rate=country_mask_rate,
            random_state=np.random.RandomState(RANDOM_SEED),
        )
    model = xgb.XGBClassifier(n_estimators=n_estimators, **final_params)
    model.fit(matrix, y, verbose=False)
    return model, cols


def run_training(
    feature_file: Path,
    ground_truth_file: Path,
    output_dir: Path,
    qwen_file: Optional[Path] = None,
    bge_file: Optional[Path] = None,
    qwen_matcher_file: Optional[Path] = None,
    source1_files: Optional[Sequence[Path]] = None,
    candidate_source_files: Optional[Sequence[Path]] = None,
    booster: str = "gbtree",
    eta: float = 0.03,
    country_mask_rate: float = 0.15,
    use_monotone_constraints: bool = False,
    compare_dart: bool = False,
) -> Dict:
    params = dict(DEFAULT_PARAMS)
    params["booster"] = booster
    params["eta"] = eta
    if booster == "dart":
        # Pass booster="dart" explicitly to ensure DART tree dropout (Rashmi & Gilad-Bachrach 2015)
        # is fully activated across all XGBoost versions.
        params["booster"] = "dart"
        params.update({
            "sample_type": "uniform",
            "normalize_type": "tree",
            "rate_drop": 0.1,
            "skip_drop": 0.5,
            "one_drop": 0,
        })

    frame = pd.read_csv(feature_file, sep="\t", dtype=str, keep_default_na=False)
    frame = merge_feature_file(frame, qwen_file)
    frame = merge_feature_file(frame, bge_file)
    # Stage 2b (stretch): only merged when the held-out-country gate passed
    # and the caller explicitly provides the file. Training with this column
    # absent is the normal v1 path; training with it present is what makes
    # its presence show up in metadata["feature_columns"] for score_candidates
    # to require consistently at inference time.
    frame = merge_feature_file(frame, qwen_matcher_file)

    # Optional records loading for fold-safe TF-IDF cosine feature
    records = None
    if source1_files and candidate_source_files:
        from src.bge_features import load_records as load_entity_records
        records = load_entity_records(list(source1_files) + list(candidate_source_files))

    ground_truth = load_ground_truth(ground_truth_file)
    labeled = attach_labels(frame, ground_truth)
    oof, columns, diagnostics = train_oof(
        labeled,
        params=params,
        country_mask_rate=country_mask_rate,
        use_monotone_constraints=use_monotone_constraints,
        records=records,
    )
    calibrator = fit_calibrator(
        oof["oof_score"].to_numpy(), oof["label"].to_numpy()
    )
    calibrated = apply_calibrator(calibrator, oof["oof_score"].to_numpy())
    threshold, f05 = choose_threshold(
        oof, calibrated, ground_truth=ground_truth
    )
    best_iters = [
        fold["best_iteration"]
        for fold in diagnostics["folds"]
        if fold.get("best_iteration", 0) > 0
    ]
    final_n_est = max(50, int(np.mean(best_iters) * 1.1)) if best_iters else 300

    model, final_columns = fit_final(
        labeled,
        columns,
        params=params,
        n_estimators=final_n_est,
        country_mask_rate=country_mask_rate,
        use_monotone_constraints=use_monotone_constraints,
        records=records,
        output_dir=output_dir,
    )
    actual_booster = model.get_params().get("booster")
    assert actual_booster == booster, (
        f"Expected fitted model booster to be '{booster}', got '{actual_booster}'"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "gbm.json"))
    oof.assign(calibrated_score=calibrated).to_csv(
        output_dir / "oof_predictions.tsv", sep="\t", index=False
    )

    # Held-out country cross-evaluation diagnostic
    if compare_dart:
        gbtree_params = dict(DEFAULT_PARAMS)
        gbtree_params["booster"] = "gbtree"
        gbtree_params["eta"] = eta

        dart_params = dict(DEFAULT_PARAMS)
        dart_params["booster"] = "dart"
        dart_params["eta"] = eta
        dart_params.update({
            "sample_type": "uniform",
            "normalize_type": "tree",
            "rate_drop": 0.1,
            "skip_drop": 0.5,
            "one_drop": 0,
        })

        gbtree_diag = evaluate_held_out_country_diagnostic(
            labeled,
            final_columns,
            params=gbtree_params,
            country_mask_rate=country_mask_rate,
            use_monotone_constraints=use_monotone_constraints,
        )
        dart_diag = evaluate_held_out_country_diagnostic(
            labeled,
            final_columns,
            params=dart_params,
            country_mask_rate=country_mask_rate,
            use_monotone_constraints=use_monotone_constraints,
        )

        gbtree_ap = float(gbtree_diag.get("mean_held_out_country_ap", 0.0))
        dart_ap = float(dart_diag.get("mean_held_out_country_ap", 0.0))
        winner = "dart" if dart_ap > gbtree_ap else "gbtree"
        diagnostics["dart_vs_gbtree_comparison"] = {
            "gbtree_mean_held_out_country_ap": gbtree_ap,
            "dart_mean_held_out_country_ap": dart_ap,
            "winning_booster": winner,
            "delta_ap": float(dart_ap - gbtree_ap),
            "gbtree_diagnostic": gbtree_diag,
            "dart_diagnostic": dart_diag,
        }
        diagnostics["held_out_country_cross_eval"] = dart_diag if booster == "dart" else gbtree_diag
    else:
        held_out_diag = evaluate_held_out_country_diagnostic(
            labeled,
            final_columns,
            params=params,
            country_mask_rate=country_mask_rate,
            use_monotone_constraints=use_monotone_constraints,
        )
        diagnostics["held_out_country_cross_eval"] = held_out_diag

    # Feature importance
    feature_imp = extract_feature_importances(model, final_columns)

    metadata = {
        "booster": actual_booster or booster,
        "feature_columns": final_columns,
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
        "params": params,
        "country_mask_rate": country_mask_rate,
        "use_monotone_constraints": use_monotone_constraints,
        "compare_dart": compare_dart,
        "final_n_estimators": int(model.get_params()["n_estimators"]),
        "feature_importance": feature_imp,
        "diagnostics": diagnostics,
    }
    with open(output_dir / "stage3_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 Grouped GBM Training and Candidate Scoring")
    parser.add_argument("--mode", choices=["train", "score"], default="train", help="Operation mode (default: train)")
    parser.add_argument("--features", type=Path, required=True, help="Input pair features TSV")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Path to ground truth TSV (required for train)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for model artifacts (train)")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Directory containing saved model artifacts (score)")
    parser.add_argument("--output-file", type=Path, default=None, help="Path for scored output TSV (score)")
    parser.add_argument("--qwen-features", type=Path, default=None)
    parser.add_argument("--bge-features", type=Path, default=None)
    parser.add_argument(
        "--qwen-matcher-features", type=Path, default=None,
        help="Stage 2b (stretch) generative-matcher features TSV. Omit entirely "
             "if the held-out-country gate hasn't passed -- do not pass a partial "
             "or untrusted file, since --use-monotone-constraints will happily "
             "train +1-constrained on whatever column this points to.",
    )
    parser.add_argument("--source1", type=Path, nargs="+", default=None)
    parser.add_argument("--candidate-sources", type=Path, nargs="+", default=None)
    parser.add_argument("--booster", type=str, choices=["gbtree", "dart"], default="gbtree")
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--country-mask-rate", type=float, default=0.15)
    parser.add_argument("--use-monotone-constraints", action="store_true")
    parser.add_argument(
        "--compare-dart", action="store_true",
        help="Run comparative cross-country diagnostic for DART vs standard GBDT, recording AUCPR delta in metadata.",
    )
    args = parser.parse_args()

    records = None
    if args.source1 and args.candidate_sources:
        from src.bge_features import load_records as load_entity_records
        records = load_entity_records(list(args.source1) + list(args.candidate_sources))

    if args.mode == "train":
        if args.ground_truth is None or args.output_dir is None:
            parser.error("--ground-truth and --output-dir are required when --mode train")
        print(json.dumps(run_training(
            feature_file=args.features,
            ground_truth_file=args.ground_truth,
            output_dir=args.output_dir,
            qwen_file=args.qwen_features,
            bge_file=args.bge_features,
            qwen_matcher_file=args.qwen_matcher_features,
            source1_files=args.source1,
            candidate_source_files=args.candidate_sources,
            booster=args.booster,
            eta=args.eta,
            country_mask_rate=args.country_mask_rate,
            use_monotone_constraints=args.use_monotone_constraints,
            compare_dart=args.compare_dart,
        ), indent=2))
    elif args.mode == "score":
        artifact_dir = args.artifact_dir or args.output_dir
        if artifact_dir is None or args.output_file is None:
            parser.error("--artifact-dir and --output-file are required when --mode score")
        scored = score_candidates(
            feature_file=args.features,
            artifact_dir=artifact_dir,
            qwen_file=args.qwen_features,
            bge_file=args.bge_features,
            qwen_matcher_file=args.qwen_matcher_features,
            records=records,
        )
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output_file, sep="\t", index=False)
        print(f"Scored {len(scored)} candidate pairs -> {args.output_file}")


if __name__ == "__main__":
    main()
