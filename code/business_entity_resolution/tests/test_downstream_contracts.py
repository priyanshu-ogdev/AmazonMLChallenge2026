"""
Tests for downstream contracts and fixes: pair_features canonical country,
stage3_gbm categorical dropping and macro_f05 singletons, and stage4_decision.

Requires pandas and numpy (for target PC test execution).
"""

import sys
import unittest
from pathlib import Path

# Ensure src is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import numpy as np
    import pandas as pd
    from src.pair_features import _record_features, pair_feature_row
    try:
        from src.scoring import DROP_CATEGORICAL, prepare_matrix, macro_f05
    except ImportError:
        from src.stage3_gbm import DROP_CATEGORICAL, prepare_matrix, macro_f05
    try:
        from src.decision import assemble_matching_results, write_matching_results
    except ImportError:
        from src.stage4_decision import assemble_matching_results, write_matching_results
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


@unittest.skipUnless(HAS_DEPENDENCIES, "Requires pandas and numpy (installed on target execution PC)")
class TestDownstreamContracts(unittest.TestCase):
    """Verify pair feature contracts, canonical country fixes, and leakage boundaries."""

    def test_pair_features_canonical_country(self):
        """Verify pair features use canonical country comparison, avoiding alias bugs."""
        rec_us1 = {
            "entity_id": "S1-1",
            "business_name": "Acme Inc",
            "business_address": "123 Main St",
            "country": "US",
        }
        rec_us2 = {
            "entity_id": "S2-1",
            "business_name": "Acme Incorporated",
            "business_address": "123 Main Street",
            "country": "USA",
        }
        rec_fr = {
            "entity_id": "S2-2",
            "business_name": "Acme SAS",
            "business_address": "123 Rue Principale",
            "country": "France",
        }
        rec_nocountry = {
            "entity_id": "S2-3",
            "business_name": "Acme Co",
            "business_address": "123 Main St",
            "country": "",
        }

        f1 = _record_features(rec_us1)
        f2 = _record_features(rec_us2)
        f_fr = _record_features(rec_fr)
        f_no = _record_features(rec_nocountry)

        self.assertEqual(f1["canonical_country"], "us")
        self.assertEqual(f2["canonical_country"], "us")
        self.assertEqual(f_fr["canonical_country"], "france")
        self.assertEqual(f_no["canonical_country"], "")

        # US vs USA -> country_equal should be 1
        row_same = pair_feature_row(f1, f2)
        self.assertEqual(row_same["country_equal"], 1)
        self.assertEqual(row_same["country_equal_missing"], 0)

        # US vs France -> country_equal should be 0
        row_diff = pair_feature_row(f1, f_fr)
        self.assertEqual(row_diff["country_equal"], 0)
        self.assertEqual(row_diff["country_equal_missing"], 0)

        # US vs missing -> country_equal should be -1, country_equal_missing = 1
        row_missing = pair_feature_row(f1, f_no)
        self.assertEqual(row_missing["country_equal"], -1)
        self.assertEqual(row_missing["country_equal_missing"], 1)

    def test_stage3_drop_categorical(self):
        """Verify DROP_CATEGORICAL excludes both raw and canonical country strings from GBM matrix."""
        self.assertIn("source1_country", DROP_CATEGORICAL)
        self.assertIn("candidate_country", DROP_CATEGORICAL)
        self.assertIn("source1_canonical_country", DROP_CATEGORICAL)
        self.assertIn("candidate_canonical_country", DROP_CATEGORICAL)

        dummy = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2"],
            "source1_country": ["US", "India"],
            "candidate_country": ["US", "India"],
            "source1_canonical_country": ["us", "india"],
            "candidate_canonical_country": ["us", "india"],
            "country_equal": [1, 1],
            "name_exact": [1, 0],
            "name_jaccard": [0.9, 0.4],
        })
        matrix, cols = prepare_matrix(dummy)
        self.assertNotIn("source1_country", cols)
        self.assertNotIn("source1_canonical_country", cols)
        self.assertNotIn("candidate_canonical_country", cols)
        self.assertIn("country_equal", cols)
        self.assertIn("name_exact", cols)
        self.assertIn("name_jaccard", cols)

    def test_stage3_macro_f05_singletons(self):
        """Verify macro_f05 correctly awards 1.0 to true singletons with empty actual and predicted."""
        source1_ids = ["S1-1", "S1-2"]
        scores = [0.9, 0.2]
        labels = [1, 0]
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": set(),  # S1-3 has no candidates (true singleton)
        }

        score = macro_f05(
            source1_ids=source1_ids,
            scores=scores,
            labels=labels,
            threshold=0.5,
            ground_truth=gt,
        )
        self.assertAlmostEqual(score, 1.0)

    def test_stage3_macro_f05_blocking_miss_zero_credit(self):
        """Verify macro_f05 awards 0.0 to an entity with GT matches completely missed by blocking."""
        source1_ids = ["S1-1"]
        scores = [0.9]
        labels = [1]
        gt = {
            "S1-1": {"S2-1"},
            "S1-miss": {"S2-miss"},  # Blocking failed to retrieve S1-miss
        }

        score = macro_f05(
            source1_ids=source1_ids,
            scores=scores,
            labels=labels,
            threshold=0.5,
            ground_truth=gt,
        )
        # S1-1 scores 1.0, S1-miss scores 0.0 -> mean is 0.5
        self.assertAlmostEqual(score, 0.5)

    def test_stage4_decision_singletons(self):
        """Verify stage4 emits every S1 entity once, with empty match list for singletons."""
        scored = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2"],
            "calibrated_score": [0.85, 0.30],
        })
        source1_ids = ["S1-1", "S1-2", "S1-3"]
        results = assemble_matching_results(scored, source1_ids, threshold=0.5)

        self.assertEqual(len(results), 3)
        row1 = results[results["source1_entity_id"] == "S1-1"].iloc[0]
        self.assertEqual(row1["matched_entity_ids"], "S2-1")

        row2 = results[results["source1_entity_id"] == "S1-2"].iloc[0]
        self.assertEqual(row2["matched_entity_ids"], "")

    def test_provenance_signals_extraction(self):
        """Verify provenance signals (rank, score, count) are extracted and preserved for GBM."""
        rec1 = _record_features({"entity_id": "S1-1", "business_name": "Acme Inc", "business_address": "123 Main St", "country": "US"})
        rec2 = _record_features({"entity_id": "S2-1", "business_name": "Acme Corp", "business_address": "123 Main Street", "country": "US"})

        # Row with full provenance
        row_prov = pair_feature_row(
            rec1,
            rec2,
            provenance="exact_name;address_structural",
            left_rank=1.0,
            best_score=0.95,
            score_margin_to_best=0.15,
            blocker_count=2,
        )
        self.assertEqual(row_prov["candidate_rank"], 1.0)
        self.assertEqual(row_prov["candidate_rank_missing"], 0)
        self.assertEqual(row_prov["best_blocker_score"], 0.95)
        self.assertEqual(row_prov["best_blocker_score_missing"], 0)
        self.assertEqual(row_prov["best_blocker_score_diff"], 0.15)
        self.assertEqual(row_prov["best_blocker_score_diff_missing"], 0)
        self.assertEqual(row_prov["blocker_count"], 2)
        self.assertEqual(row_prov["has_blocker_provenance"], 1)

        # Row with missing provenance
        row_noprov = pair_feature_row(rec1, rec2)
        self.assertEqual(row_noprov["candidate_rank"], 999.0)
        self.assertEqual(row_noprov["candidate_rank_missing"], 1)
        self.assertEqual(row_noprov["best_blocker_score"], -1.0)
        self.assertEqual(row_noprov["best_blocker_score_missing"], 1)
        self.assertEqual(row_noprov["best_blocker_score_diff"], 0.0)
        self.assertEqual(row_noprov["best_blocker_score_diff_missing"], 1)
        self.assertEqual(row_noprov["blocker_count"], 0)
        self.assertEqual(row_noprov["has_blocker_provenance"], 0)

        # Check prepare_matrix keeps numeric provenance features and drops string blocker_provenance
        df = pd.DataFrame([row_prov])
        matrix, cols = prepare_matrix(df)
        self.assertIn("candidate_rank", cols)
        self.assertIn("best_blocker_score", cols)
        self.assertIn("best_blocker_score_diff", cols)
        self.assertIn("best_blocker_score_diff_missing", cols)
        self.assertIn("blocker_count", cols)
        self.assertNotIn("blocker_provenance", cols)

    def test_stage4_decision_threshold_boundary(self):
        """Verify score >= threshold is inclusive at the exact boundary."""
        scored = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-exact", "S2-below", "S2-above"],
            "calibrated_score": ["0.5000", "0.4999", 0.75],
        })
        results = assemble_matching_results(scored, ["S1-1", "S1-2"], threshold=0.50)
        row1 = results[results["source1_entity_id"] == "S1-1"].iloc[0]
        # Exactly 0.5000 must be included; 0.4999 must not
        self.assertEqual(row1["matched_entity_ids"], "S2-exact")
        row2 = results[results["source1_entity_id"] == "S1-2"].iloc[0]
        self.assertEqual(row2["matched_entity_ids"], "S2-above")

    def test_stage4_decision_invalid_inputs(self):
        """Verify stage4 decision raises loudly on invalid inputs."""
        valid_scored = pd.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "calibrated_score": [0.8],
        })
        # Duplicate S1 IDs
        with self.assertRaises(ValueError) as ctx:
            assemble_matching_results(valid_scored, ["S1-1", "S1-1"], threshold=0.5)
        self.assertIn("duplicates", str(ctx.exception))

        # Unknown S1 ID in scored table
        with self.assertRaises(ValueError) as ctx:
            assemble_matching_results(valid_scored, ["S1-2"], threshold=0.5)
        self.assertIn("unknown S1 IDs", str(ctx.exception))

        # Non-numeric calibrated_score
        bad_score_df = pd.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "calibrated_score": ["not_a_number"],
        })
        with self.assertRaises(ValueError) as ctx:
            assemble_matching_results(bad_score_df, ["S1-1"], threshold=0.5)
        self.assertIn("non-numeric", str(ctx.exception))

    def test_stage4_decision_injective_bipartite_matching(self):
        """Verify injective matching assigns shared candidates to highest score only."""
        scored = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-shared", "S2-shared"],
            "calibrated_score": [0.92, 0.78],
        })
        # Default injective=True: S1-1 gets S2-shared, S1-2 gets empty
        results_injective = assemble_matching_results(scored, ["S1-1", "S1-2"], threshold=0.50, injective=True)
        row1 = results_injective[results_injective["source1_entity_id"] == "S1-1"].iloc[0]
        row2 = results_injective[results_injective["source1_entity_id"] == "S1-2"].iloc[0]
        self.assertEqual(row1["matched_entity_ids"], "S2-shared")
        self.assertEqual(row2["matched_entity_ids"], "")

        # When injective=False: both S1-1 and S1-2 get S2-shared
        results_non_injective = assemble_matching_results(scored, ["S1-1", "S1-2"], threshold=0.50, injective=False)
        row1_non = results_non_injective[results_non_injective["source1_entity_id"] == "S1-1"].iloc[0]
        row2_non = results_non_injective[results_non_injective["source1_entity_id"] == "S1-2"].iloc[0]
        self.assertEqual(row1_non["matched_entity_ids"], "S2-shared")
        self.assertEqual(row2_non["matched_entity_ids"], "S2-shared")

    def test_stage4_decision_self_match_prevention(self):
        """Verify candidate pairs where candidate_entity_id == source1_entity_id are ignored."""
        scored = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1"],
            "candidate_entity_id": ["S1-1", "S2-valid"],
            "calibrated_score": [0.99, 0.85],
        })
        results = assemble_matching_results(scored, ["S1-1"], threshold=0.50, injective=True)
        self.assertEqual(results.iloc[0]["matched_entity_ids"], "S2-valid")

        results_non = assemble_matching_results(scored, ["S1-1"], threshold=0.50, injective=False)
        self.assertEqual(results_non.iloc[0]["matched_entity_ids"], "S2-valid")

    def test_stage4_decision_duplicate_pairs_validation(self):
        """Verify assemble_matching_results rejects input tables with duplicate candidate pairs."""
        scored = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1"],
            "candidate_entity_id": ["S2-cand", "S2-cand"],
            "calibrated_score": [0.90, 0.80],
        })
        with self.assertRaises(ValueError) as ctx:
            assemble_matching_results(scored, ["S1-1"], threshold=0.50)
        self.assertIn("duplicate candidate pairs", str(ctx.exception))

    def test_stage4_decision_threshold_override(self):
        """Verify write_matching_results operates with explicit threshold without metadata_file."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            scored_path = tmp / "scored.tsv"
            s1_path = tmp / "source1.tsv"
            out_path = tmp / "matching_results.tsv"

            pd.DataFrame({
                "source1_entity_id": ["S1-1", "S1-2"],
                "candidate_entity_id": ["S2-1", "S2-2"],
                "calibrated_score": [0.75, 0.45],
            }).to_csv(scored_path, sep="\t", index=False)

            pd.DataFrame({
                "entity_id": ["S1-1", "S1-2", "S1-3"],
            }).to_csv(s1_path, sep="\t", index=False)

            # Pass threshold=0.50 explicitly, metadata_file=None
            write_matching_results(
                scored_file=scored_path,
                source1_file=s1_path,
                metadata_file=None,
                output_file=out_path,
                threshold=0.50,
            )
            out_df = pd.read_csv(out_path, sep="\t", dtype=str, keep_default_na=False)
            self.assertEqual(len(out_df), 3)
            self.assertEqual(out_df[out_df["source1_entity_id"] == "S1-1"].iloc[0]["matched_entity_ids"], "S2-1")
            self.assertEqual(out_df[out_df["source1_entity_id"] == "S1-2"].iloc[0]["matched_entity_ids"], "")
            self.assertEqual(out_df[out_df["source1_entity_id"] == "S1-3"].iloc[0]["matched_entity_ids"], "")


if __name__ == "__main__":
    unittest.main()
