"""
Unit tests for Stage 3: Pair Scoring, Calibration, Monotonic Constraints,
Country Generalization Masking, and End-to-End Inference.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from src.scoring import (
    DEFAULT_PARAMS,
    apply_country_masking,
    attach_labels,
    build_monotonic_constraints,
    choose_threshold,
    evaluate_held_out_country_diagnostic,
    extract_feature_importances,
    fit_final,
    load_ground_truth,
    macro_f05,
    merge_feature_file,
    prepare_matrix,
    run_training,
    score_candidates,
    train_oof,
)
from src.decision import assemble_matching_results


class TestStage3Scoring(unittest.TestCase):
    """Test suite for Layer 3 GBM meta-layer and generalization mechanisms."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_monotonic_constraints(self):
        """Verify monotonic constraint vector matches column semantics."""
        cols = [
            "bge_cosine",
            "qwen_cosine",
            "qwen_matcher_prob",
            "name_exact",
            "address_edit_similarity",
            "same_name_different_address",
            "candidate_rank",
            "country_equal",
            "country_equal_missing",
            "source_is_s3",
            "postal_equal",
            "postal_missing_either",
            "rank_margin_from_best",
            "best_blocker_score",
            "best_blocker_score_missing",
            "best_blocker_score_diff",
            "best_blocker_score_diff_missing",
            "candidate_rank_missing",
            "candidate_count_for_s1",
            "blocker_count",
            "has_blocker_provenance",
        ]
        constraints = build_monotonic_constraints(cols)
        expected = (
            1,   # bge_cosine -> +1
            1,   # qwen_cosine -> +1
            1,   # qwen_matcher_prob -> +1 (Stage 2b generative-matcher probability)
            1,   # name_exact -> +1
            1,   # address_edit_similarity -> +1
            -1,  # same_name_different_address -> -1
            -1,  # candidate_rank -> -1 (lower rank number is better)
            0,   # country_equal -> 0 (NEVER monotonic, prevents shortcut)
            0,   # country_equal_missing -> 0
            0,   # source_is_s3 -> 0
            1,   # postal_equal -> +1
            0,   # postal_missing_either -> 0
            -1,  # rank_margin_from_best -> -1
            1,   # best_blocker_score -> +1
            0,   # best_blocker_score_missing -> 0 (indicator)
            -1,  # best_blocker_score_diff -> -1 (larger margin to best is worse)
            0,   # best_blocker_score_diff_missing -> 0 (indicator)
            0,   # candidate_rank_missing -> 0 (indicator)
            0,   # candidate_count_for_s1 -> 0
            1,   # blocker_count -> +1
            1,   # has_blocker_provenance -> +1
        )
        self.assertEqual(constraints, expected)

    def test_apply_country_masking(self):
        """Verify stochastic country masking replaces values with missing flags."""
        df = pd.DataFrame({
            "country_equal": [1.0, 1.0, 0.0, 1.0],
            "country_equal_missing": [0.0, 0.0, 0.0, 0.0],
            "name_jaccard": [0.8, 0.9, 0.5, 0.7],
        })
        # Rate 1.0 -> all masked
        masked = apply_country_masking(df, mask_rate=1.0)
        self.assertTrue((masked["country_equal"] == -1.0).all())
        self.assertTrue((masked["country_equal_missing"] == 1.0).all())
        # Other features untouched
        np.testing.assert_array_equal(masked["name_jaccard"], df["name_jaccard"])

        # Rate 0.0 -> none masked
        unmasked = apply_country_masking(df, mask_rate=0.0)
        np.testing.assert_array_equal(unmasked["country_equal"], df["country_equal"])

    def test_choose_threshold_with_grid(self):
        """Verify threshold selection scans probability grid and singletons."""
        frame = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "label": [1, 0, 1],
        })
        scores = np.array([0.92, 0.15, 0.88])
        all_s1 = ["S1-1", "S1-2", "S1-3", "S1-4"]  # S1-4 is a singleton with 0 candidates

        threshold, best_f05 = choose_threshold(frame, scores, all_source1_ids=all_s1)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertGreater(best_f05, 0.0)

    def test_macro_f05_blocking_miss_vs_singleton(self):
        """Verify macro_f05 correctly gives 0.0 to blocking misses and 1.0 to true singletons."""
        source1_ids = ["S1-hit"]
        scores = [0.9]
        labels = [1]

        # Ground truth:
        # S1-hit has 1 match (retrieved and predicted)
        # S1-miss has 1 match (NOT in candidates - blocking miss)
        # S1-singleton has 0 matches (true singleton)
        gt = {
            "S1-hit": {"S2-hit"},
            "S1-miss": {"S2-miss"},
            "S1-singleton": set(),
        }

        # With ground_truth:
        # S1-hit -> score 1.0 (tp=1, pred=1, act=1)
        # S1-miss -> score 0.0 (blocking miss, pred=0, act=1)
        # S1-singleton -> score 1.0 (true singleton, pred=0, act=0)
        # Overall macro_f05 = (1.0 + 0.0 + 1.0) / 3 = 2/3
        f05_with_gt = macro_f05(
            source1_ids=source1_ids,
            scores=scores,
            labels=labels,
            threshold=0.5,
            ground_truth=gt,
        )
        self.assertAlmostEqual(f05_with_gt, 2.0 / 3.0)

        # Contrast with legacy all_source1_ids where blocking miss was erroneously scored 1.0
        f05_legacy = macro_f05(
            source1_ids=source1_ids,
            scores=scores,
            labels=labels,
            threshold=0.5,
            all_source1_ids=list(gt.keys()),
        )
        self.assertAlmostEqual(f05_legacy, 1.0)

    def test_macro_f05_partial_blocking_recall(self):
        """Verify macro_f05 evaluates recall against true GT count, not retrieved count."""
        source1_ids = ["S1-1"]
        scores = [0.85]
        labels = [1]

        # S1-1 has 2 ground truth matches, but blocking only retrieved 1
        gt = {
            "S1-1": {"S2-1", "S2-2"}
        }

        # tp = 1, pred = 1, act = 2
        # precision = 1.0, recall = 0.5
        # F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6
        f05 = macro_f05(
            source1_ids=source1_ids,
            scores=scores,
            labels=labels,
            threshold=0.5,
            ground_truth=gt,
        )
        self.assertAlmostEqual(f05, 5.0 / 6.0)

    def test_choose_threshold_with_ground_truth(self):
        """Verify choose_threshold correctly optimizes threshold with ground_truth dictionary."""
        frame = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "label": [1, 0, 1],
        })
        scores = np.array([0.92, 0.15, 0.88])
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": {"S2-3"},
            "S1-miss": {"S2-miss"},  # blocking miss
            "S1-singleton": set(),   # singleton
        }

        threshold, best_f05 = choose_threshold(frame, scores, ground_truth=gt)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertGreater(best_f05, 0.0)
        self.assertLess(best_f05, 1.0)  # Cannot be 1.0 because S1-miss is 0.0

    def test_run_training_and_score_candidates_end_to_end(self):
        """End-to-end integration test of Stage 3 training and test scoring."""
        # 1. Prepare synthetic pair features
        pair_data = []
        for i in range(1, 11):
            s1_id = f"S1-{i}"
            country = "us" if i <= 5 else "india"
            # Positive pair
            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"S2-{i}",
                "source1_country": country.upper(),
                "candidate_country": country.upper(),
                "source1_canonical_country": country,
                "candidate_canonical_country": country,
                "source_is_s3": 0,
                "country_equal": 1,
                "country_equal_missing": 0,
                "left_country_missing": 0,
                "right_country_missing": 0,
                "name_both_missing": 0,
                "address_both_missing": 0,
                "name_exact": 1,
                "address_exact": 1,
                "name_jaccard": 0.95,
                "name_overlap": 1.0,
                "name_edit_similarity": 0.92,
                "name_char_trigram_jaccard": 0.90,
                "address_jaccard": 0.88,
                "address_overlap": 0.90,
                "address_edit_similarity": 0.85,
                "address_char_trigram_jaccard": 0.82,
                "name_number_overlap": 1.0,
                "address_number_overlap": 1.0,
                "postal_equal": 1,
                "postal_missing_either": 0,
                "name_length_abs_diff": 2,
                "address_length_abs_diff": 3,
                "same_name_different_address": 0,
                "same_address_different_name": 0,
                "candidate_rank": 1.0,
                "candidate_rank_missing": 0,
                "rank_margin_from_best": 0.0,
                "best_blocker_score": 0.98,
                "best_blocker_score_missing": 0,
                "best_blocker_score_diff": 0.0,
                "blocker_count": 3,
                "candidate_count_for_s1": 2,
                "has_blocker_provenance": 1,
                "blocker_provenance": "exact_name",
            })
            # Negative pair
            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"S3-{i}",
                "source1_country": country.upper(),
                "candidate_country": country.upper(),
                "source1_canonical_country": country,
                "candidate_canonical_country": country,
                "source_is_s3": 1,
                "country_equal": 1,
                "country_equal_missing": 0,
                "left_country_missing": 0,
                "right_country_missing": 0,
                "name_both_missing": 0,
                "address_both_missing": 0,
                "name_exact": 0,
                "address_exact": 0,
                "name_jaccard": 0.15,
                "name_overlap": 0.2,
                "name_edit_similarity": 0.20,
                "name_char_trigram_jaccard": 0.12,
                "address_jaccard": 0.10,
                "address_overlap": 0.15,
                "address_edit_similarity": 0.18,
                "address_char_trigram_jaccard": 0.11,
                "name_number_overlap": 0.0,
                "address_number_overlap": 0.0,
                "postal_equal": 0,
                "postal_missing_either": 0,
                "name_length_abs_diff": 15,
                "address_length_abs_diff": 25,
                "same_name_different_address": 0,
                "same_address_different_name": 0,
                "candidate_rank": 4.0,
                "candidate_rank_missing": 0,
                "rank_margin_from_best": 3.0,
                "best_blocker_score": 0.45,
                "best_blocker_score_missing": 0,
                "best_blocker_score_diff": 0.53,
                "blocker_count": 1,
                "candidate_count_for_s1": 2,
                "has_blocker_provenance": 1,
                "blocker_provenance": "token_inverted",
            })

        pair_file = self.output_dir / "features.tsv"
        pd.DataFrame(pair_data).to_csv(pair_file, sep="\t", index=False)

        # 2. Ground truth (matches are S2-i)
        gt_data = [
            {"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{i}"}
            for i in range(1, 11)
        ]
        # Include a singleton in ground truth
        gt_data.append({"source1_entity_id": "S1-999", "matched_entity_ids": ""})
        gt_file = self.output_dir / "ground_truth.tsv"
        pd.DataFrame(gt_data).to_csv(gt_file, sep="\t", index=False)

        # 3. Dense BGE feature table
        bge_data = [
            {"source1_entity_id": row["source1_entity_id"], "candidate_entity_id": row["candidate_entity_id"],
             "bge_cosine": 0.94 if row["candidate_entity_id"].startswith("S2") else 0.32}
            for row in pair_data
        ]
        bge_file = self.output_dir / "bge_features.tsv"
        pd.DataFrame(bge_data).to_csv(bge_file, sep="\t", index=False)

        # 4. Dense Qwen feature table
        qwen_data = [
            {"source1_entity_id": row["source1_entity_id"], "candidate_entity_id": row["candidate_entity_id"],
             "qwen_cosine": 0.91 if row["candidate_entity_id"].startswith("S2") else 0.28}
            for row in pair_data
        ]
        qwen_file = self.output_dir / "qwen_features.tsv"
        pd.DataFrame(qwen_data).to_csv(qwen_file, sep="\t", index=False)

        stage3_dir = self.output_dir / "stage3_out"

        # 5. Run Stage 3 training with monotonic constraints & country masking
        metadata = run_training(
            feature_file=pair_file,
            ground_truth_file=gt_file,
            output_dir=stage3_dir,
            qwen_file=qwen_file,
            bge_file=bge_file,
            booster="gbtree",
            eta=0.03,
            country_mask_rate=0.15,
            use_monotone_constraints=True,
        )

        self.assertIn("threshold", metadata)
        self.assertIn("macro_f05", metadata)
        self.assertIn("feature_importance", metadata)
        self.assertIn("held_out_country_cross_eval", metadata["diagnostics"])
        self.assertTrue((stage3_dir / "gbm.json").exists())
        self.assertTrue((stage3_dir / "oof_predictions.tsv").exists())
        self.assertTrue((stage3_dir / "stage3_metadata.json").exists())

        # 6. Test inference with score_candidates
        test_cands_file = self.output_dir / "test_features.tsv"
        pd.DataFrame(pair_data[:4]).to_csv(test_cands_file, sep="\t", index=False)
        test_bge_file = self.output_dir / "test_bge.tsv"
        pd.DataFrame(bge_data[:4]).to_csv(test_bge_file, sep="\t", index=False)
        test_qwen_file = self.output_dir / "test_qwen.tsv"
        pd.DataFrame(qwen_data[:4]).to_csv(test_qwen_file, sep="\t", index=False)

        scored = score_candidates(
            feature_file=test_cands_file,
            artifact_dir=stage3_dir,
            qwen_file=test_qwen_file,
            bge_file=test_bge_file,
        )

        self.assertEqual(len(scored), 4)
        self.assertIn("raw_score", scored.columns)
        self.assertIn("calibrated_score", scored.columns)
        self.assertIn("is_match", scored.columns)

        # 7. Feed scored candidates to Stage 4 decision assembly
        all_test_s1 = ["S1-1", "S1-2", "S1-singleton"]
        decision = assemble_matching_results(
            scored=scored,
            source1_ids=all_test_s1,
            threshold=float(metadata["threshold"]),
        )
        self.assertEqual(len(decision), 3)
        self.assertIn("source1_entity_id", decision.columns)
        self.assertIn("matched_entity_ids", decision.columns)
        singleton_row = decision[decision["source1_entity_id"] == "S1-singleton"].iloc[0]
        self.assertEqual(singleton_row["matched_entity_ids"], "")

    def test_run_training_dart_booster(self):
        """Verify DART booster escalation mode trains and persists metadata cleanly."""
        pair_data = []
        for i in range(1, 7):
            s1_id = f"S1-{i}"
            country = "us" if i <= 3 else "india"
            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"S2-{i}",
                "source1_country": country.upper(),
                "candidate_country": country.upper(),
                "source1_canonical_country": country,
                "candidate_canonical_country": country,
                "source_is_s3": 0,
                "country_equal": 1,
                "country_equal_missing": 0,
                "left_country_missing": 0,
                "right_country_missing": 0,
                "name_both_missing": 0,
                "address_both_missing": 0,
                "name_exact": 1,
                "address_exact": 1,
                "name_jaccard": 0.9,
                "name_overlap": 0.9,
                "name_edit_similarity": 0.9,
                "name_char_trigram_jaccard": 0.9,
                "address_jaccard": 0.9,
                "address_overlap": 0.9,
                "address_edit_similarity": 0.9,
                "address_char_trigram_jaccard": 0.9,
                "name_number_overlap": 1.0,
                "address_number_overlap": 1.0,
                "postal_equal": 1,
                "postal_missing_either": 0,
                "name_length_abs_diff": 0,
                "address_length_abs_diff": 0,
                "same_name_different_address": 0,
                "same_address_different_name": 0,
                "candidate_rank": 1.0,
                "candidate_rank_missing": 0,
                "rank_margin_from_best": 0.0,
                "best_blocker_score": 0.9,
                "best_blocker_score_missing": 0,
                "best_blocker_score_diff": 0.0,
                "blocker_count": 2,
                "candidate_count_for_s1": 2,
                "has_blocker_provenance": 1,
                "blocker_provenance": "exact",
            })
            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": f"S3-{i}",
                "source1_country": country.upper(),
                "candidate_country": country.upper(),
                "source1_canonical_country": country,
                "candidate_canonical_country": country,
                "source_is_s3": 1,
                "country_equal": 1,
                "country_equal_missing": 0,
                "left_country_missing": 0,
                "right_country_missing": 0,
                "name_both_missing": 0,
                "address_both_missing": 0,
                "name_exact": 0,
                "address_exact": 0,
                "name_jaccard": 0.1,
                "name_overlap": 0.1,
                "name_edit_similarity": 0.1,
                "name_char_trigram_jaccard": 0.1,
                "address_jaccard": 0.1,
                "address_overlap": 0.1,
                "address_edit_similarity": 0.1,
                "address_char_trigram_jaccard": 0.1,
                "name_number_overlap": 0.0,
                "address_number_overlap": 0.0,
                "postal_equal": 0,
                "postal_missing_either": 0,
                "name_length_abs_diff": 10,
                "address_length_abs_diff": 10,
                "same_name_different_address": 0,
                "same_address_different_name": 0,
                "candidate_rank": 5.0,
                "candidate_rank_missing": 0,
                "rank_margin_from_best": 4.0,
                "best_blocker_score": 0.2,
                "best_blocker_score_missing": 0,
                "best_blocker_score_diff": 0.7,
                "blocker_count": 1,
                "candidate_count_for_s1": 2,
                "has_blocker_provenance": 1,
                "blocker_provenance": "token",
            })

        pair_file = self.output_dir / "dart_features.tsv"
        pd.DataFrame(pair_data).to_csv(pair_file, sep="\t", index=False)
        gt_data = [
            {"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{i}"}
            for i in range(1, 7)
        ]
        gt_file = self.output_dir / "dart_gt.tsv"
        pd.DataFrame(gt_data).to_csv(gt_file, sep="\t", index=False)

        dart_out = self.output_dir / "dart_out"
        metadata = run_training(
            feature_file=pair_file,
            ground_truth_file=gt_file,
            output_dir=dart_out,
            booster="dart",
            eta=0.05,
            country_mask_rate=0.0,
            use_monotone_constraints=False,
        )
        self.assertEqual(metadata["params"]["booster"], "gbtree")  # XGBoost 3.x: dart uses gbtree+dropout
        self.assertEqual(metadata["params"]["sample_type"], "uniform")
        self.assertTrue((dart_out / "gbm.json").exists())

    def test_fold_safe_tfidf_training_and_inference(self):
        """Verify fold-safe TF-IDF cosine feature generation during training and inference."""
        from src.bge_features import load_records as load_entity_records

        s1_file = self.output_dir / "tfidf_s1.tsv"
        s2_file = self.output_dir / "tfidf_s2.tsv"
        pair_file = self.output_dir / "tfidf_pairs.tsv"
        gt_file = self.output_dir / "tfidf_gt.tsv"
        stage3_dir = self.output_dir / "tfidf_stage3"

        s1_df = pd.DataFrame([
            {"entity_id": f"S1-{i}", "business_name": f"Acme Store #{i}", "business_address": f"{100*i} Main St", "country": "US"}
            for i in range(1, 7)
        ])
        s2_df = pd.DataFrame([
            {"entity_id": f"S2-{i}", "business_name": f"Acme Corporation #{i}", "business_address": f"{100*i} Main Street", "country": "US"}
            for i in range(1, 7)
        ] + [
            {"entity_id": f"S2-neg-{i}", "business_name": f"Totally Unrelated #{i}", "business_address": f"{999*i} Broadway", "country": "US"}
            for i in range(1, 7)
        ])

        s1_df.to_csv(s1_file, sep="\t", index=False)
        s2_df.to_csv(s2_file, sep="\t", index=False)

        pair_data = []
        gt_data = []
        for i in range(1, 7):
            s1_id = f"S1-{i}"
            pos_id = f"S2-{i}"
            neg_id = f"S2-neg-{i}"
            gt_data.append({"source1_entity_id": s1_id, "matched_entity_ids": pos_id})

            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": pos_id,
                "source1_country": "US",
                "candidate_country": "US",
                "source1_canonical_country": "us",
                "candidate_canonical_country": "us",
                "source_is_s3": 0,
                "country_equal": 1,
                "country_equal_missing": 0,
                "left_country_missing": 0,
                "right_country_missing": 0,
                "name_both_missing": 0,
                "address_both_missing": 0,
                "name_exact": 0,
                "address_exact": 0,
                "name_jaccard": 0.8,
                "name_overlap": 0.8,
                "name_edit_similarity": 0.8,
                "name_char_trigram_jaccard": 0.8,
                "address_jaccard": 0.8,
                "address_overlap": 0.8,
                "address_edit_similarity": 0.8,
                "address_char_trigram_jaccard": 0.8,
                "name_number_overlap": 1.0,
                "address_number_overlap": 1.0,
                "postal_equal": 0,
                "postal_missing_either": 1,
                "name_length_abs_diff": 5,
                "address_length_abs_diff": 5,
                "same_name_different_address": 0,
                "same_address_different_name": 0,
                "candidate_rank": 1.0,
                "candidate_rank_missing": 0,
                "rank_margin_from_best": 0.0,
                "best_blocker_score": 0.9,
                "best_blocker_score_missing": 0,
                "best_blocker_score_diff": 0.0,
                "blocker_count": 2,
                "candidate_count_for_s1": 2,
                "has_blocker_provenance": 1,
                "blocker_provenance": "exact",
            })
            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": neg_id,
                "source1_country": "US",
                "candidate_country": "US",
                "source1_canonical_country": "us",
                "candidate_canonical_country": "us",
                "source_is_s3": 0,
                "country_equal": 1,
                "country_equal_missing": 0,
                "left_country_missing": 0,
                "right_country_missing": 0,
                "name_both_missing": 0,
                "address_both_missing": 0,
                "name_exact": 0,
                "address_exact": 0,
                "name_jaccard": 0.0,
                "name_overlap": 0.0,
                "name_edit_similarity": 0.1,
                "name_char_trigram_jaccard": 0.05,
                "address_jaccard": 0.0,
                "address_overlap": 0.0,
                "address_edit_similarity": 0.1,
                "address_char_trigram_jaccard": 0.05,
                "name_number_overlap": 0.0,
                "address_number_overlap": 0.0,
                "postal_equal": 0,
                "postal_missing_either": 1,
                "name_length_abs_diff": 15,
                "address_length_abs_diff": 15,
                "same_name_different_address": 0,
                "same_address_different_name": 0,
                "candidate_rank": 5.0,
                "candidate_rank_missing": 0,
                "rank_margin_from_best": 4.0,
                "best_blocker_score": 0.1,
                "best_blocker_score_missing": 0,
                "best_blocker_score_diff": 0.8,
                "blocker_count": 1,
                "candidate_count_for_s1": 2,
                "has_blocker_provenance": 1,
                "blocker_provenance": "token",
            })

        pd.DataFrame(pair_data).to_csv(pair_file, sep="\t", index=False)
        pd.DataFrame(gt_data).to_csv(gt_file, sep="\t", index=False)

        metadata = run_training(
            feature_file=pair_file,
            ground_truth_file=gt_file,
            output_dir=stage3_dir,
            source1_files=[s1_file],
            candidate_source_files=[s2_file],
            booster="gbtree",
            eta=0.03,
            country_mask_rate=0.0,
            use_monotone_constraints=True,
        )

        self.assertIn("tfidf_cosine", metadata["feature_columns"])
        self.assertTrue((stage3_dir / "tfidf_vectorizer.joblib").exists())

        # Test inference with records passed
        records = load_entity_records([s1_file, s2_file])
        scored = score_candidates(
            feature_file=pair_file,
            artifact_dir=stage3_dir,
            records=records,
        )
        self.assertEqual(len(scored), len(pair_data))
        self.assertIn("calibrated_score", scored.columns)

        # Test inference raises clearly when records=None but tfidf vectorizer exists
        with self.assertRaises(ValueError) as ctx:
            score_candidates(
                feature_file=pair_file,
                artifact_dir=stage3_dir,
                records=None,
            )
        self.assertIn("tfidf_cosine", str(ctx.exception))
        self.assertIn("records=", str(ctx.exception))

    def test_evaluate_held_out_country_diagnostic_two_way(self):
        """Verify two-way held-out-country diagnostic runs without early stopping leakage."""
        rows = []
        for i in range(1, 11):
            country = "us" if i <= 5 else "india"
            rows.append({
                "source1_entity_id": f"S1-{i}",
                "candidate_entity_id": f"S2-pos-{i}",
                "source1_canonical_country": country,
                "label": 1,
                "name_jaccard": 0.9,
                "address_jaccard": 0.85,
            })
            rows.append({
                "source1_entity_id": f"S1-{i}",
                "candidate_entity_id": f"S2-neg-{i}",
                "source1_canonical_country": country,
                "label": 0,
                "name_jaccard": 0.1,
                "address_jaccard": 0.05,
            })
        df = pd.DataFrame(rows)
        res = evaluate_held_out_country_diagnostic(df, ["name_jaccard", "address_jaccard"])
        self.assertIn("train_others_eval_us", res)
        self.assertIn("train_others_eval_india", res)
        self.assertIn("mean_held_out_country_ap", res)
        self.assertGreater(res["mean_held_out_country_ap"], 0.0)

    def test_qwen_matcher_features_integration(self):
        """Verify Stage 2b generative-matcher features integrate into training and scoring."""
        pair_file = self.output_dir / "matcher_pairs.tsv"
        gt_file = self.output_dir / "matcher_gt.tsv"
        matcher_file = self.output_dir / "matcher_feats.tsv"
        stage3_dir = self.output_dir / "matcher_stage3"

        pair_data = []
        gt_data = []
        matcher_data = []
        for i in range(1, 7):
            s1_id = f"S1-{i}"
            pos_id = f"S2-pos-{i}"
            neg_id = f"S2-neg-{i}"
            gt_data.append({"source1_entity_id": s1_id, "matched_entity_ids": pos_id})

            for cand_id, is_pos in [(pos_id, 1), (neg_id, 0)]:
                pair_data.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "source1_country": "US",
                    "candidate_country": "US",
                    "source1_canonical_country": "us",
                    "candidate_canonical_country": "us",
                    "source_is_s3": 0,
                    "country_equal": 1,
                    "country_equal_missing": 0,
                    "left_country_missing": 0,
                    "right_country_missing": 0,
                    "name_both_missing": 0,
                    "address_both_missing": 0,
                    "name_exact": is_pos,
                    "address_exact": is_pos,
                    "name_jaccard": 0.9 if is_pos else 0.1,
                    "name_overlap": 0.9 if is_pos else 0.1,
                    "name_edit_similarity": 0.9 if is_pos else 0.1,
                    "name_char_trigram_jaccard": 0.9 if is_pos else 0.1,
                    "address_jaccard": 0.9 if is_pos else 0.1,
                    "address_overlap": 0.9 if is_pos else 0.1,
                    "address_edit_similarity": 0.9 if is_pos else 0.1,
                    "address_char_trigram_jaccard": 0.9 if is_pos else 0.1,
                    "name_number_overlap": 1.0 if is_pos else 0.0,
                    "address_number_overlap": 1.0 if is_pos else 0.0,
                    "postal_equal": is_pos,
                    "postal_missing_either": 0,
                    "name_length_abs_diff": 0 if is_pos else 10,
                    "address_length_abs_diff": 0 if is_pos else 10,
                    "same_name_different_address": 0,
                    "same_address_different_name": 0,
                    "candidate_rank": 1.0 if is_pos else 2.0,
                    "candidate_rank_missing": 0,
                    "rank_margin_from_best": 0.0 if is_pos else 0.5,
                    "best_blocker_score": 0.95 if is_pos else 0.4,
                    "best_blocker_score_missing": 0,
                    "best_blocker_score_diff": 0.0 if is_pos else 0.55,
                    "blocker_count": 2 if is_pos else 1,
                    "candidate_count_for_s1": 2,
                    "has_blocker_provenance": 1,
                    "blocker_provenance": "exact" if is_pos else "token",
                })
                matcher_data.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "qwen_matcher_prob": 0.92 if is_pos else 0.08,
                    "qwen_matcher_prob_missing": 0,
                })

        pd.DataFrame(pair_data).to_csv(pair_file, sep="\t", index=False)
        pd.DataFrame(gt_data).to_csv(gt_file, sep="\t", index=False)
        pd.DataFrame(matcher_data).to_csv(matcher_file, sep="\t", index=False)

        metadata = run_training(
            feature_file=pair_file,
            ground_truth_file=gt_file,
            output_dir=stage3_dir,
            qwen_matcher_file=matcher_file,
            booster="gbtree",
            eta=0.03,
            country_mask_rate=0.0,
            use_monotone_constraints=True,
        )

        self.assertIn("qwen_matcher_prob", metadata["feature_columns"])
        self.assertTrue((stage3_dir / "gbm.json").exists())

        # Test inference with matcher file passed
        scored = score_candidates(
            feature_file=pair_file,
            artifact_dir=stage3_dir,
            qwen_matcher_file=matcher_file,
        )
        self.assertEqual(len(scored), len(pair_data))
        self.assertIn("calibrated_score", scored.columns)

        # Test inference without matcher file raises because required column is missing
        with self.assertRaises(ValueError) as ctx:
            score_candidates(
                feature_file=pair_file,
                artifact_dir=stage3_dir,
                qwen_matcher_file=None,
            )
        self.assertIn("Missing requested feature columns", str(ctx.exception))
        self.assertIn("qwen_matcher_prob", str(ctx.exception))

    def test_dart_booster_training(self):
        """Verify DART escalation mode in run_training executes and sets dropout parameters."""
        stage3_dir = self.output_dir / "stage3_dart_test"
        pair_file = self.output_dir / "pairs_dart.tsv"
        gt_file = self.output_dir / "gt_dart.tsv"

        pair_data = []
        gt_data = []
        for i in range(20):
            s1_id = f"S1-{i}"
            c1_id = f"S2-{i}"
            c2_id = f"S2-neg-{i}"
            gt_data.append({"source1_entity_id": s1_id, "matched_entity_ids": c1_id})

            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": c1_id,
                "country_equal": 1,
                "bge_cosine": 0.85,
                "name_exact": 1,
                "same_name_different_address": 0,
                "same_address_different_name": 0,
                "candidate_rank": 1.0,
                "candidate_rank_missing": 0,
                "rank_margin_from_best": 0.0,
                "best_blocker_score": 0.95,
                "best_blocker_score_missing": 0,
                "best_blocker_score_diff": 0.0,
                "blocker_count": 2,
                "candidate_count_for_s1": 2,
                "has_blocker_provenance": 1,
                "blocker_provenance": "exact",
            })
            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": c2_id,
                "country_equal": 1,
                "bge_cosine": 0.20,
                "name_exact": 0,
                "same_name_different_address": 0,
                "same_address_different_name": 0,
                "candidate_rank": 2.0,
                "candidate_rank_missing": 0,
                "rank_margin_from_best": 0.5,
                "best_blocker_score": 0.40,
                "best_blocker_score_missing": 0,
                "best_blocker_score_diff": 0.55,
                "blocker_count": 1,
                "candidate_count_for_s1": 2,
                "has_blocker_provenance": 1,
                "blocker_provenance": "token",
            })

        pd.DataFrame(pair_data).to_csv(pair_file, sep="\t", index=False)
        pd.DataFrame(gt_data).to_csv(gt_file, sep="\t", index=False)

        metadata = run_training(
            feature_file=pair_file,
            ground_truth_file=gt_file,
            output_dir=stage3_dir,
            booster="dart",
            eta=0.03,
            country_mask_rate=0.0,
            use_monotone_constraints=True,
        )

        self.assertTrue((stage3_dir / "gbm.json").exists())
        self.assertTrue((stage3_dir / "stage3_metadata.json").exists())
        self.assertTrue((stage3_dir / "oof_predictions.tsv").exists())
        self.assertEqual(metadata["params"]["booster"], "gbtree")
        self.assertEqual(metadata["params"]["sample_type"], "uniform")
        self.assertEqual(metadata["params"]["normalize_type"], "tree")
        self.assertEqual(metadata["params"]["rate_drop"], 0.10)
        self.assertEqual(metadata["params"]["skip_drop"], 0.50)

    def test_stage2c_to_gbm_matrix_synchronization(self):
        """Verify Stage 2c deterministic features feed cleanly into Stage 3 prepare_matrix."""
        from src.pair_features import build_pair_features

        records = {
            "S1-1": {
                "entity_id": "S1-1",
                "business_name": "Acme Widgets Inc",
                "business_address": "123 Main St, Springfield",
                "country": "US",
                "canonical_country": "us",
            },
            "S2-1": {
                "entity_id": "S2-1",
                "business_name": "Acme Widgets Incorporated",
                "business_address": "123 Main Street, Springfield",
                "country": "USA",
                "canonical_country": "us",
            },
        }
        cand_file = self.output_dir / "sync_candidates.tsv"
        pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1"}]).to_csv(
            cand_file, sep="\t", index=False
        )

        prov_file = self.output_dir / "sync_prov.tsv"
        pd.DataFrame([{
            "source1_entity_id": "S1-1",
            "candidate_entity_id": "S2-1",
            "blocker_provenance": "exact_name",
            "best_blocker_rank": "1",
            "best_blocker_score": "0.95",
            "blocker_count": "2",
        }]).to_csv(prov_file, sep="\t", index=False)

        out_feat = self.output_dir / "stage2c_pair_features.tsv"
        feat_df = build_pair_features(records, cand_file, out_feat, prov_file)
        self.assertEqual(len(feat_df), 1)

        # Feed directly into prepare_matrix
        matrix, cols = prepare_matrix(feat_df)
        self.assertEqual(len(cols), 35)

        # Assert ID and categorical columns are excluded
        excluded_expected = {
            "source1_entity_id", "candidate_entity_id",
            "source1_country", "candidate_country",
            "source1_canonical_country", "candidate_canonical_country",
            "blocker_provenance",
        }
        for col_name in excluded_expected:
            self.assertNotIn(col_name, cols)

        # Monotonic constraints
        constraints = build_monotonic_constraints(cols)
        self.assertEqual(len(constraints), len(cols))
        col_to_c = dict(zip(cols, constraints))
        self.assertEqual(col_to_c["name_exact"], 1)
        self.assertEqual(col_to_c["candidate_rank"], -1)
        self.assertEqual(col_to_c["country_equal"], 0)
        self.assertEqual(col_to_c["country_equal_missing"], 0)
        self.assertEqual(col_to_c["best_blocker_score"], 1)
        self.assertEqual(col_to_c["best_blocker_score_diff"], -1)
        self.assertEqual(col_to_c["best_blocker_score_missing"], 0)
        self.assertEqual(col_to_c["best_blocker_score_diff_missing"], 0)

    def test_merge_feature_file_missing_row_preserves_indicator(self):
        """Verify left-merging auxiliary features flags missing rows as 1.0 in missing indicator."""
        main_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "val": "1.0"},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2", "val": "2.0"},
        ])
        aux_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "bge_cosine": "0.90", "bge_cosine_missing": "0"},
        ])
        aux_path = self.output_dir / "aux_bge.tsv"
        aux_df.to_csv(aux_path, sep="\t", index=False)

        merged = merge_feature_file(main_df, aux_path)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.loc[0, "bge_cosine_missing"], "0")
        self.assertEqual(merged.loc[1, "bge_cosine_missing"], "1")

        matrix, cols = prepare_matrix(merged)
        self.assertEqual(matrix.loc[0, "bge_cosine_missing"], 0.0)
        self.assertEqual(matrix.loc[0, "bge_cosine"], 0.90)
        self.assertEqual(matrix.loc[1, "bge_cosine_missing"], 1.0)
        self.assertEqual(matrix.loc[1, "bge_cosine"], 0.0)

    def test_prepare_matrix_fill_semantics(self):
        """Verify prepare_matrix applies Stage 2c default fill values for missing data."""
        raw_df = pd.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "candidate_rank": [np.nan],
            "rank_margin_from_best": [np.nan],
            "best_blocker_score": [np.nan],
            "country_equal": [np.nan],
            "bge_cosine_missing": [np.nan],
            "name_jaccard": [np.nan],
        })
        matrix, cols = prepare_matrix(raw_df)
        self.assertEqual(matrix.loc[0, "candidate_rank"], 999.0)
        self.assertEqual(matrix.loc[0, "rank_margin_from_best"], 999.0)
        self.assertEqual(matrix.loc[0, "best_blocker_score"], -1.0)
        self.assertEqual(matrix.loc[0, "country_equal"], -1.0)
        self.assertEqual(matrix.loc[0, "bge_cosine_missing"], 1.0)
        self.assertEqual(matrix.loc[0, "name_jaccard"], 0.0)

    def test_compare_dart_diagnostic_execution(self):
        """Verify run_training with compare_dart=True executes both GBDT and DART diagnostics."""
        stage3_dir = self.output_dir / "stage3_compare_dart"
        pair_file = self.output_dir / "pairs_comp.tsv"
        gt_file = self.output_dir / "gt_comp.tsv"

        pair_data = []
        gt_data = []
        for i in range(20):
            s1_id = f"S1-{i}"
            c1_id = f"S2-{i}"
            c2_id = f"S2-neg-{i}"
            country = "us" if i < 10 else "india"
            gt_data.append({"source1_entity_id": s1_id, "matched_entity_ids": c1_id})

            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": c1_id,
                "source1_canonical_country": country,
                "candidate_canonical_country": country,
                "country_equal": 1,
                "name_exact": 1,
                "name_jaccard": 0.9,
                "address_edit_similarity": 0.85,
            })
            pair_data.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": c2_id,
                "source1_canonical_country": country,
                "candidate_canonical_country": country,
                "country_equal": 1,
                "name_exact": 0,
                "name_jaccard": 0.1,
                "address_edit_similarity": 0.2,
            })

        pd.DataFrame(pair_data).to_csv(pair_file, sep="\t", index=False)
        pd.DataFrame(gt_data).to_csv(gt_file, sep="\t", index=False)

        metadata = run_training(
            feature_file=pair_file,
            ground_truth_file=gt_file,
            output_dir=stage3_dir,
            booster="gbtree",
            eta=0.03,
            country_mask_rate=0.0,
            compare_dart=True,
        )

        self.assertIn("dart_vs_gbtree_comparison", metadata["diagnostics"])
        cmp = metadata["diagnostics"]["dart_vs_gbtree_comparison"]
        self.assertIn("gbtree_mean_held_out_country_ap", cmp)
        self.assertIn("dart_mean_held_out_country_ap", cmp)
        self.assertIn("winning_booster", cmp)


if __name__ == "__main__":
    unittest.main()
