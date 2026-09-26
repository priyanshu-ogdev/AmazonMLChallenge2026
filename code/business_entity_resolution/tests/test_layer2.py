"""
Unit tests for Layer 2: Representations and Pair Features.

Covers:
- Stage 2a-i: BGE-M3 bi-encoder features & losses
- Stage 2a-ii: Qwen3-Embedding-0.6B auxiliary dense features
- Stage 2c: Deterministic pair features & provenance signals
- Data preparation: bidirectional gate & evaluation dataset construction
- Held-out evaluation metrics & go/no-go gate logic
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from src.pair_features import (
    _record_features,
    pair_feature_row,
    build_pair_features,
    load_records as pf_load_records,
)
from src.bge_features import load_records as bge_load_records, load_candidates
from src.qwen_features import format_entity_text, DEFAULT_INSTRUCTION
from src.qwen_matcher_features import (
    load_records as qm_load_records,
    load_candidates as qm_load_candidates,
    PROMPT_TEMPLATE as QM_PROMPT_TEMPLATE,
)

try:
    from src.eval_bi_encoder import (
        compute_retrieval_metrics,
        compute_margin_analysis,
        _evaluate_single_direction,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data_builder import (
    build_evaluation_data,
    build_positive_pairs,
    format_pairs_for_dataset,
)


class TestLayer2PairFeatures(unittest.TestCase):
    """Test Stage 2c deterministic pair feature generation and contracts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pair_features_end_to_end_with_provenance(self):
        """Verify build_pair_features correctly reads candidate_provenance with new signals."""
        s1_file = self.dir_path / "s1.tsv"
        s2_file = self.dir_path / "s2.tsv"
        s3_file = self.dir_path / "s3.tsv"
        cand_file = self.dir_path / "candidate_pairs.tsv"
        prov_file = self.dir_path / "candidate_provenance.tsv"
        out_file = self.dir_path / "pair_features.tsv"

        # Raw TSVs
        s1_df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Target Store #102", "business_address": "1000 Nicollet Mall, Minneapolis, MN 55403", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Infosys BPM", "business_address": "Electronics City, Bengaluru 560100", "country": "India"},
        ])
        s2_df = pd.DataFrame([
            {"entity_id": "S2-1", "business_name": "Target Corporation", "business_address": "1000 Nicollet Mall, Minneapolis, MN 55403", "country": "USA"},
        ])
        s3_df = pd.DataFrame([
            {"entity_id": "S3-1", "business_name": "Infosys B P M Ltd", "business_address": "Hosur Rd, Electronics City, Bangalore 560100", "country": "IN"},
        ])
        cand_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1"},
            {"source1_entity_id": "S1-2", "candidate_entity_ids": "S3-1"},
        ])
        prov_df = pd.DataFrame([
            {
                "source1_entity_id": "S1-1",
                "candidate_entity_id": "S2-1",
                "blocker_provenance": "exact_name_postal;address_structural",
                "best_blocker_rank": "1",
                "best_blocker_score": "0.95",
                "blocker_count": "2",
            },
            {
                "source1_entity_id": "S1-2",
                "candidate_entity_id": "S3-1",
                "blocker_provenance": "token_inverted",
                "best_blocker_rank": "3",
                "best_blocker_score": "0.78",
                "blocker_count": "1",
            },
        ])

        s1_df.to_csv(s1_file, sep="\t", index=False)
        s2_df.to_csv(s2_file, sep="\t", index=False)
        s3_df.to_csv(s3_file, sep="\t", index=False)
        cand_df.to_csv(cand_file, sep="\t", index=False)
        prov_df.to_csv(prov_file, sep="\t", index=False)

        records = pf_load_records([s1_file, s2_file, s3_file])
        features = build_pair_features(
            records=records,
            candidate_file=cand_file,
            output_file=out_file,
            provenance_file=prov_file,
        )

        self.assertEqual(len(features), 2)
        self.assertTrue(out_file.exists())

        # Check provenance signals for pair 1
        p1 = features[features["source1_entity_id"] == "S1-1"].iloc[0]
        self.assertEqual(p1["candidate_rank"], 1.0)
        self.assertEqual(p1["candidate_rank_missing"], 0)
        self.assertEqual(p1["best_blocker_score"], 0.95)
        self.assertEqual(p1["best_blocker_score_missing"], 0)
        self.assertEqual(p1["blocker_count"], 2)
        self.assertEqual(p1["country_equal"], 1)  # US vs USA alias folded
        self.assertEqual(p1["source_is_s3"], 0)

        # Check provenance signals for pair 2
        p2 = features[features["source1_entity_id"] == "S1-2"].iloc[0]
        self.assertEqual(p2["candidate_rank"], 3.0)
        self.assertEqual(p2["candidate_rank_missing"], 0)
        self.assertEqual(p2["best_blocker_score"], 0.78)
        self.assertEqual(p2["best_blocker_score_missing"], 0)
        self.assertEqual(p2["blocker_count"], 1)
        self.assertEqual(p2["country_equal"], 1)  # India vs IN alias folded
        self.assertEqual(p2["source_is_s3"], 1)

    def test_layer0_normalized_records_compatibility(self):
        """Verify pair_features transparently ingests Layer 0 normalized TSV schema."""
        s1_file = self.dir_path / "s1_norm.tsv"
        s2_file = self.dir_path / "s2_norm.tsv"
        cand_file = self.dir_path / "cand.tsv"
        out_file = self.dir_path / "features_norm.tsv"

        s1_norm = pd.DataFrame([{
            "entity_id": "S1-10",
            "raw_name": "Walmart Supercenter",
            "raw_address": "500 Terry Francois St",
            "norm_name": "walmart supercenter",
            "norm_address": "500 terry francois st",
            "encoder_text": "walmart supercenter 500 terry francois st",
            "country_canonical": "us",
        }])
        s2_norm = pd.DataFrame([{
            "entity_id": "S2-10",
            "raw_name": "Walmart",
            "raw_address": "500 Terry Francois Street",
            "norm_name": "walmart",
            "norm_address": "500 terry francois st",
            "encoder_text": "walmart 500 terry francois st",
            "country_canonical": "us",
        }])
        cand_df = pd.DataFrame([{"source1_entity_id": "S1-10", "candidate_entity_ids": "S2-10"}])

        s1_norm.to_csv(s1_file, sep="\t", index=False)
        s2_norm.to_csv(s2_file, sep="\t", index=False)
        cand_df.to_csv(cand_file, sep="\t", index=False)

        records = pf_load_records([s1_file, s2_file])
        self.assertIn("S1-10", records)
        self.assertEqual(records["S1-10"]["canonical_country"], "us")

        features = build_pair_features(records, cand_file, out_file)
        self.assertEqual(len(features), 1)
        row = features.iloc[0]
        self.assertEqual(row["country_equal"], 1)
        self.assertEqual(row["address_exact"], 1)


class TestLayer2DenseAndEmbeddingContracts(unittest.TestCase):
    """Test dense representation formatting, candidate deduplication, and loader invariants."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_qwen_prompt_formatting(self):
        """Verify Qwen format_entity_text treats both sides of the pair symmetrically."""
        text = "acme inc 123 main street"
        prompted = format_entity_text(text)
        expected = f"Instruct: {DEFAULT_INSTRUCTION}\nText: {text}"
        self.assertEqual(prompted, expected)

    def test_load_candidates_deduplication(self):
        """Verify load_candidates expands and deduplicates pairs accurately."""
        cand_path = self.dir_path / "test_cands.tsv"
        df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1, S3-1, S2-1"},  # S2-1 duplicated
            {"source1_entity_id": "S1-2", "candidate_entity_ids": "S2-2"},
        ])
        df.to_csv(cand_path, sep="\t", index=False)
        pairs = load_candidates(cand_path)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0], ("S1-1", "S2-1"))
        self.assertEqual(pairs[1], ("S1-1", "S3-1"))
        self.assertEqual(pairs[2], ("S1-2", "S2-2"))

    def test_bge_load_records_uses_encoder_text_when_present(self):
        """Verify bge_load_records reuses pre-computed encoder_text from Layer 0 without recomputing."""
        norm_path = self.dir_path / "norm.tsv"
        df = pd.DataFrame([{
            "entity_id": "S1-99",
            "business_name": "Raw Name Inc",
            "business_address": "Raw Address Blvd",
            "encoder_text": "clean name 123 address",
        }])
        df.to_csv(norm_path, sep="\t", index=False)
        recs = bge_load_records([norm_path])
        self.assertEqual(recs["S1-99"], "clean name 123 address")

    def test_empty_text_ids_detection_and_missing_flags(self):
        """Verify empty text identification logic mirrors the bge/qwen features contract."""
        records = {
            "S1-1": "valid text content",
            "S1-2": "",
            "S2-1": "another valid text",
            "S2-2": "   \n\t",  # whitespace only
        }
        candidate_pairs = [
            ("S1-1", "S2-1"),  # both non-empty -> missing flag 0
            ("S1-1", "S2-2"),  # candidate empty -> missing flag 1
            ("S1-2", "S2-1"),  # source1 empty -> missing flag 1
            ("S1-2", "S2-2"),  # both empty -> missing flag 1
        ]
        entity_ids = sorted({entity_id for pair in candidate_pairs for entity_id in pair})
        empty_text_ids = {
            entity_id for entity_id in entity_ids if not records[entity_id].strip()
        }
        self.assertEqual(empty_text_ids, {"S1-2", "S2-2"})

        missing_flags = [
            int(s1 in empty_text_ids or c2 in empty_text_ids)
            for s1, c2 in candidate_pairs
        ]
        self.assertEqual(missing_flags, [0, 1, 1, 1])

    def test_qwen_matcher_contracts(self):
        """Verify qwen_matcher_features record loading, candidate loading, and prompt formatting."""
        s1_file = self.dir_path / "qm_s1.tsv"
        s2_file = self.dir_path / "qm_s2.tsv"
        cand_file = self.dir_path / "qm_cands.tsv"

        s1_df = pd.DataFrame([{
            "entity_id": "S1-A",
            "business_name": "Target Store",
            "business_address": "1000 Nicollet Mall",
            "country": "US",
        }])
        s2_df = pd.DataFrame([{
            "entity_id": "S2-B",
            "business_name": "Target Corp",
            "business_address": "1000 Nicollet",
            "country": "USA",
        }])
        cand_df = pd.DataFrame([{
            "source1_entity_id": "S1-A",
            "candidate_entity_ids": "S2-B",
        }])

        s1_df.to_csv(s1_file, sep="\t", index=False)
        s2_df.to_csv(s2_file, sep="\t", index=False)
        cand_df.to_csv(cand_file, sep="\t", index=False)

        records = qm_load_records([s1_file, s2_file])
        self.assertEqual(records["S1-A"]["name"], "Target Store")
        self.assertEqual(records["S2-B"]["address"], "1000 Nicollet")

        pairs = qm_load_candidates(cand_file)
        self.assertEqual(pairs, [("S1-A", "S2-B")])

        prompt = QM_PROMPT_TEMPLATE.format(
            name1=records["S1-A"]["name"],
            address1=records["S1-A"]["address"],
            country1=records["S1-A"]["country"],
            name2=records["S2-B"]["name"],
            address2=records["S2-B"]["address"],
            country2=records["S2-B"]["country"],
        )
        self.assertIn("Task: Do the following two records refer to the same business entity?", prompt)
        self.assertIn("Target Store", prompt)
        self.assertIn("Target Corp", prompt)
        self.assertTrue(prompt.endswith("Match:"))


@unittest.skipUnless(HAS_TORCH, "Requires torch (installed separately)")
class TestLayer2EvaluationAndMetrics(unittest.TestCase):
    """Test retrieval metrics, margin analysis, and go/no-go gate decisions."""

    def test_compute_retrieval_metrics(self):
        """Verify Recall@K and MRR computation against exact known rankings."""
        # 2 queries, 3 corpus docs
        # Q0 relevant to C0; Q1 relevant to C1
        q_embs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        c_embs = np.array([
            [0.9, 0.1],  # C0: close to Q0
            [0.1, 0.9],  # C1: close to Q1
            [0.5, 0.5],  # C2: middle
        ], dtype=np.float32)

        query_ids = ["Q0", "Q1"]
        corpus_ids = ["C0", "C1", "C2"]
        relevant_docs = {"Q0": {"C0"}, "Q1": {"C1"}}

        metrics = compute_retrieval_metrics(
            query_embeddings=q_embs,
            corpus_embeddings=c_embs,
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            relevant_docs=relevant_docs,
            k_values=[1, 2, 5],
        )

        self.assertAlmostEqual(metrics["recall@1"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(metrics["precision@1"], 1.0)

    def test_compute_margin_analysis(self):
        """Verify score gap pass rate calculation."""
        q_embs = np.array([[1.0, 0.0]], dtype=np.float32)
        # Pos is C0 (dot=0.9), Neg is C1 (dot=0.6) -> margin = 0.30
        c_embs = np.array([[0.9, 0.0], [0.6, 0.0]], dtype=np.float32)
        margins = compute_margin_analysis(
            query_embeddings=q_embs,
            corpus_embeddings=c_embs,
            query_ids=["Q0"],
            corpus_ids=["C0", "C1"],
            relevant_docs={"Q0": {"C0"}},
            margin_thresholds=[0.10, 0.20, 0.35],
        )

        self.assertAlmostEqual(margins["mean_margin"], 0.30, places=4)
        self.assertAlmostEqual(margins["margin_pass@0.10"], 1.0)
        self.assertAlmostEqual(margins["margin_pass@0.20"], 1.0)
        self.assertAlmostEqual(margins["margin_pass@0.35"], 0.0)

    def test_evaluate_single_direction_gate_decision(self):
        """Verify go/no-go gate threshold enforcement."""
        class MockModel:
            def encode(self, texts, **kwargs):
                # Return dummy normalized unit vectors
                return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            with open(p / "eval_queries.json", "w") as f:
                json.dump({"Q1": "Query 1"}, f)
            with open(p / "eval_corpus.json", "w") as f:
                json.dump({"C1": "Pos 1", "C2": "Neg 1"}, f)
            with open(p / "eval_relevant.json", "w") as f:
                json.dump({"Q1": ["C1"]}, f)

            # Identical embeddings -> margin = 0.0 < 0.10 -> NO-GO expected on margin health
            res = _evaluate_single_direction(
                model=MockModel(),
                base_model=None,
                data_dir=str(p),
                prefix="",
                direction_name="test_dir",
            )
            self.assertEqual(res["decision"], "NO-GO")
            self.assertTrue(any("Margin pass" in r for r in res["reasons"]))


class TestLayer2DataBuilder(unittest.TestCase):
    """Test bi-encoder training pair extraction and IR evaluation dataset building."""

    def test_build_positive_pairs_without_negatives(self):
        gt = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1, S3-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-2"},
        ])
        s1_records = {
            "S1-1": {"encoder_text": "s1 one text", "country_canonical": "us"},
            "S1-2": {"encoder_text": "s1 two text", "country_canonical": "india"},
        }
        s2s3_records = {
            "S2-1": {"encoder_text": "s2 one text"},
            "S3-1": {"encoder_text": "s3 one text"},
            "S2-2": {"encoder_text": "s2 two text"},
        }
        pairs = build_positive_pairs(gt, s1_records, s2s3_records)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["anchor"], "s1 one text")
        self.assertEqual(pairs[0]["positive"], "s2 one text")
        self.assertNotIn("negatives", pairs[0])

    def test_build_positive_pairs_with_hard_negatives(self):
        gt = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        ])
        s1_records = {
            "S1-1": {"encoder_text": "s1 one text", "country_canonical": "us"},
        }
        s2s3_records = {
            "S2-1": {"encoder_text": "s2 one text"},
            "S2-cand-neg": {"encoder_text": "s2 candidate neg text"},
        }
        candidates_map = {
            "S1-1": ["S2-cand-neg", "S2-1"],
        }
        pairs = build_positive_pairs(
            gt, s1_records, s2s3_records,
            candidates_map=candidates_map,
            negatives_per_positive=1,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["anchor"], "s1 one text")
        self.assertEqual(pairs[0]["positive"], "s2 one text")
        self.assertIn("negatives", pairs[0])
        self.assertEqual(pairs[0]["negatives"], ["s2 candidate neg text"])

    def test_build_evaluation_data(self):
        eval_gt = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        ])
        s1_records = {"S1-1": {"encoder_text": "s1 text"}}
        s2s3_records = {"S2-1": {"encoder_text": "s2 text"}}
        neg_records = {"S2-neg": {"encoder_text": "s2 neg text"}}

        queries, corpus, relevant_docs = build_evaluation_data(
            eval_gt, s1_records, s2s3_records, neg_records,
        )
        self.assertEqual(queries, {"S1-1": "s1 text"})
        self.assertEqual(corpus["S2-1"], "s2 text")
        self.assertEqual(corpus["S2-neg"], "s2 neg text")
        self.assertEqual(relevant_docs, {"S1-1": {"S2-1"}})

    def test_hard_negatives_preserved_when_first_pair_has_none(self):
        """Verify negatives key is present and preserved even when pairs[0] has no candidates."""
        gt = pd.DataFrame([
            {"source1_entity_id": "S1-no-neg", "matched_entity_ids": "S2-1"},
            {"source1_entity_id": "S1-has-neg", "matched_entity_ids": "S2-2"},
        ])
        s1_records = {
            "S1-no-neg": {"encoder_text": "s1 no neg text", "country_canonical": "us"},
            "S1-has-neg": {"encoder_text": "s1 has neg text", "country_canonical": "us"},
        }
        s2s3_records = {
            "S2-1": {"encoder_text": "s2 one text"},
            "S2-2": {"encoder_text": "s2 two text"},
            "S2-cand": {"encoder_text": "s2 candidate text"},
        }
        candidates_map = {
            "S1-has-neg": ["S2-cand"],
        }
        pairs = build_positive_pairs(
            gt, s1_records, s2s3_records,
            candidates_map=candidates_map,
            negatives_per_positive=1,
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["negatives"], [])
        self.assertEqual(pairs[1]["negatives"], ["s2 candidate text"])

    def test_hard_negatives_disjointness_validation(self):
        """Verify hard negatives reject candidates matching anchor or true positive match text."""
        gt = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        ])
        s1_records = {
            "S1-1": {"encoder_text": "same business anchor", "country_canonical": "us"},
        }
        s2s3_records = {
            "S2-1": {"encoder_text": "same business positive"},
            # cand1 text matches anchor
            "S2-cand1": {"encoder_text": "same business anchor"},
            # cand2 text matches true positive
            "S2-cand2": {"encoder_text": "same business positive"},
            # cand3 is genuinely disjoint
            "S2-cand3": {"encoder_text": "completely distinct negative entity"},
        }
        candidates_map = {
            "S1-1": ["S2-cand1", "S2-cand2", "S2-cand3"],
        }
        pairs = build_positive_pairs(
            gt, s1_records, s2s3_records,
            candidates_map=candidates_map,
            negatives_per_positive=1,
        )
        self.assertEqual(len(pairs), 1)
        # Should skip cand1 (anchor text collision) and cand2 (pos text collision), selecting cand3
        self.assertEqual(pairs[0]["negatives"], ["completely distinct negative entity"])

    def test_format_pairs_for_dataset_columns(self):
        """Verify format_pairs_for_dataset produces flat 1D string columns for SentenceTransformers."""
        # 1. Pairs only (no negatives)
        pairs_no_negs = [
            {"anchor": "anc1", "positive": "pos1"},
            {"anchor": "anc2", "positive": "pos2"},
        ]
        data_no_negs = format_pairs_for_dataset(pairs_no_negs)
        self.assertEqual(set(data_no_negs.keys()), {"anchor", "positive"})
        self.assertEqual(data_no_negs["anchor"], ["anc1", "anc2"])
        self.assertEqual(data_no_negs["positive"], ["pos1", "pos2"])

        # 2. Triplet format: single negative column (flat strings)
        pairs_single_neg = [
            {"anchor": "anc1", "positive": "pos1", "negatives": []},
            {"anchor": "anc2", "positive": "pos2", "negatives": ["neg2_text"]},
        ]
        data_single_neg = format_pairs_for_dataset(pairs_single_neg)
        self.assertEqual(set(data_single_neg.keys()), {"anchor", "positive", "negative"})
        self.assertEqual(data_single_neg["negative"], ["", "neg2_text"])
        self.assertIsInstance(data_single_neg["negative"][0], str)
        self.assertIsInstance(data_single_neg["negative"][1], str)

        # 3. Multi-negatives format: negative_0, negative_1 (flat strings)
        pairs_multi_neg = [
            {"anchor": "anc1", "positive": "pos1", "negatives": ["neg1_a", "neg1_b"]},
            {"anchor": "anc2", "positive": "pos2", "negatives": ["neg2_a"]},
        ]
        data_multi_neg = format_pairs_for_dataset(pairs_multi_neg)
        self.assertEqual(
            set(data_multi_neg.keys()),
            {"anchor", "positive", "negative_0", "negative_1"}
        )
        self.assertEqual(data_multi_neg["negative_0"], ["neg1_a", "neg2_a"])
        self.assertEqual(data_multi_neg["negative_1"], ["neg1_b", ""])


    def test_dataconfig_and_trainingconfig_defaults(self):
        """Verify committed default hyperparameters for negative mining and distillation."""
        from src.config import DataConfig, TrainingConfig
        data_cfg = DataConfig()
        self.assertEqual(data_cfg.negatives_per_positive, 2)

        train_cfg = TrainingConfig()
        self.assertTrue(train_cfg.distill_anchor_positive_only)
        self.assertTrue(train_cfg.use_distillation)
        self.assertEqual(train_cfg.distillation_weight, 0.10)

    def test_distillation_cached_mnrl_feature_slicing(self):
        """Verify distill_anchor_positive_only limits self-distillation to anchor and positive."""
        from src.losses import DistillationCachedMNRL
        import torch

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.eval_calls = 0

            def __getitem__(self, idx):
                return self

            def forward(self, x):
                self.eval_calls += 1
                batch_size = next(iter(x.values())).shape[0]
                return {"sentence_embedding": torch.ones(batch_size, 8)}

        model = DummyModel()
        frozen = DummyModel()
        def mock_init(loss_self, *args, **kwargs):
            torch.nn.Module.__init__(loss_self)
            loss_self.model = model
            loss_self.frozen_model = frozen
            loss_self.mini_batch_size = 4
            loss_self.distill_anchor_positive_only = True

        from unittest.mock import patch
        with patch.object(DistillationCachedMNRL, "__init__", mock_init):
            loss_fn = DistillationCachedMNRL()

        # Batch with 4 columns: anchor, positive, neg1, neg2
        features = [
            {"input_ids": torch.zeros((4, 10), dtype=torch.long)},
            {"input_ids": torch.zeros((4, 10), dtype=torch.long)},
            {"input_ids": torch.zeros((4, 10), dtype=torch.long)},
            {"input_ids": torch.zeros((4, 10), dtype=torch.long)},
        ]
        distill_loss = loss_fn._compute_distillation_chunked(features)
        self.assertIsInstance(distill_loss, torch.Tensor)
        # With distill_anchor_positive_only=True, only 2 columns (anchor & positive) were forwarded
        self.assertEqual(frozen.eval_calls, 2)
        self.assertEqual(model.eval_calls, 2)


if __name__ == "__main__":
    unittest.main()



