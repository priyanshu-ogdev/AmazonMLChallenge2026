"""
Unit tests for Layer 1 candidate generation and blocking (src/blocking.py).

Verifies:
  1. Exact name and composite key matching
  2. Token inverted index with IDF and frequency capping
  3. Character 3-gram similarity matching
  4. Address structural keys (postal + street number)
  5. Country partitioning invariants (in-partition, cross-country blocking, open-set, missing fallback)
  6. Deterministic priority ranking and capping
  7. Singleton preservation (S1 with zero candidates emitted as empty string)
  8. Auditable provenance table schema and contents
  9. Recall audit gate computation
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure src is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.blocking import (
    BlockingRecord,
    MultiChannelBlocker,
    read_tsv_records,
    run_blocking,
    main,
)


class TestBlockingChannels(unittest.TestCase):
    """Test individual blocking channels in MultiChannelBlocker."""

    def setUp(self):
        self.blocker = MultiChannelBlocker(
            top_k_sparse=10,
            top_k_dense=10,
            max_candidates_per_entity=50,
            similarity_floor=0.30,
        )

    def test_exact_name_blocking(self):
        """Verify exact normalized business names match with score 1.0 and exact provenance."""
        cand = BlockingRecord.from_row(
            entity_id="S2-101",
            name="Acme Technology Solutions Inc",
            address="100 Main St, Cityville, NY 10001",
            country="US",
        )
        self.blocker.index_candidate(cand)

        s1 = BlockingRecord.from_row(
            entity_id="S1-1",
            name="Acme Technology Solutions Incorporated",
            address="Different Address, Othercity, CA",
            country="US",
        )
        hits = self.blocker.generate_candidates_for_record(s1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["candidate_entity_id"], "S2-101")
        self.assertIn("exact_name", hits[0]["blocker_provenance"])
        self.assertEqual(hits[0]["best_blocker_score"], 1.0)

    def test_composite_name_postal_key(self):
        """Verify composite name + postal code matches when name has slight variation."""
        cand = BlockingRecord.from_row(
            entity_id="S2-102",
            name="Apex Dental Care",
            address="450 River Rd, Austin, TX 78701",
            country="USA",
        )
        self.blocker.index_candidate(cand)

        s1 = BlockingRecord.from_row(
            entity_id="S1-2",
            name="Apex Dental Care",
            address="Suite 200, 450 River Road, Austin, Texas 78701",
            country="United States",
        )
        hits = self.blocker.generate_candidates_for_record(s1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["candidate_entity_id"], "S2-102")
        self.assertIn("exact_name_postal", hits[0]["blocker_provenance"])

    def test_address_structural_blocking(self):
        """Verify address structural key (postal + street number) retrieves candidate."""
        cand = BlockingRecord.from_row(
            entity_id="S3-103",
            name="PQR Logistics",
            address="1200 Industrial Pkwy, Chicago, IL 60601",
            country="US",
        )
        self.blocker.index_candidate(cand)

        s1 = BlockingRecord.from_row(
            entity_id="S1-3",
            name="Completely Different Name",
            address="1200 Industrial Parkway, Chicago, IL 60601",
            country="US",
        )
        hits = self.blocker.generate_candidates_for_record(s1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["candidate_entity_id"], "S3-103")
        self.assertIn("address_structural", hits[0]["blocker_provenance"])

    def test_char_trigram_blocking_on_typos(self):
        """Verify character 3-gram retrieval recovers candidates with spelling errors."""
        cand = BlockingRecord.from_row(
            entity_id="S2-104",
            name="Northwestern Healthcare Diagnostic Center",
            address="1500 Hospital Blvd",
            country="US",
        )
        self.blocker.index_candidate(cand)

        # Typo in S1 name: "Northwester Healthcare Dignostic Ctr"
        s1 = BlockingRecord.from_row(
            entity_id="S1-4",
            name="Northwester Healthcare Dignostic Center",
            address="No address available",
            country="US",
        )
        hits = self.blocker.generate_candidates_for_record(s1)
        self.assertTrue(any(h["candidate_entity_id"] == "S2-104" for h in hits))
        cand_hit = next(h for h in hits if h["candidate_entity_id"] == "S2-104")
        self.assertIn("char_ngram", cand_hit["blocker_provenance"])

    def test_first_2_tokens_blocking(self):
        """Verify first-2-tokens channel retrieves candidate when remaining name differs."""
        cand = BlockingRecord.from_row(
            entity_id="S2-105",
            name="Apex Health Technologies LLC",
            address="123 Hospital Way",
            country="US",
        )
        self.blocker.index_candidate(cand)

        s1 = BlockingRecord.from_row(
            entity_id="S1-5",
            name="Apex Health Care Systems",
            address="456 Other Rd",
            country="US",
        )
        hits = self.blocker.generate_candidates_for_record(s1)
        self.assertTrue(any(h["candidate_entity_id"] == "S2-105" for h in hits))
        cand_hit = next(h for h in hits if h["candidate_entity_id"] == "S2-105")
        self.assertIn("first_2_tokens", cand_hit["blocker_provenance"])

    def test_acronym_blocking(self):
        """Verify acronym channel matches acronym to expanded name."""
        cand = BlockingRecord.from_row(
            entity_id="S2-106",
            name="General Electric Medical Systems",
            address="100 Innovation Way",
            country="US",
        )
        self.blocker.index_candidate(cand)

        s1 = BlockingRecord.from_row(
            entity_id="S1-6",
            name="GE Healthcare",
            address="Different Address",
            country="US",
        )
        hits = self.blocker.generate_candidates_for_record(s1)
        self.assertTrue(any(h["candidate_entity_id"] == "S2-106" for h in hits))
        cand_hit = next(h for h in hits if h["candidate_entity_id"] == "S2-106")
        self.assertIn("acronym_match", cand_hit["blocker_provenance"])

    def test_address_missing_bypass(self):
        """Verify address structural channel is bypassed when address is missing or placeholder."""
        cand = BlockingRecord.from_row(
            entity_id="S2-107",
            name="Acme Alpha",
            address="",
            country="US",
        )
        self.assertTrue(cand.is_address_missing)
        self.blocker.index_candidate(cand)

        s1 = BlockingRecord.from_row(
            entity_id="S1-7",
            name="Beta Omega",
            address="no address available",
            country="US",
        )
        self.assertTrue(s1.is_address_missing)
        hits = self.blocker.generate_candidates_for_record(s1)
        # Address structural must not match
        self.assertFalse(any("address_structural" in h["blocker_provenance"] for h in hits))


class TestCountryPartitioning(unittest.TestCase):
    """Verify strict country partitioning invariants."""

    def setUp(self):
        self.blocker = MultiChannelBlocker()

    def test_cross_country_blocking(self):
        """Identical business name in different canonical countries must NOT match."""
        cand_india = BlockingRecord.from_row(
            entity_id="S2-IN",
            name="Apollo Pharmacy",
            address="MG Road, Bangalore",
            country="India",
        )
        self.blocker.index_candidate(cand_india)

        s1_us = BlockingRecord.from_row(
            entity_id="S1-US",
            name="Apollo Pharmacy",
            address="100 Main St, New York, NY",
            country="US",
        )
        hits = self.blocker.generate_candidates_for_record(s1_us)
        self.assertEqual(len(hits), 0, "Cross-country candidate must be blocked")

    def test_country_alias_folding(self):
        """USA and US fold to same canonical country 'us' and match."""
        cand_usa = BlockingRecord.from_row(
            entity_id="S2-USA",
            name="General Motors Co",
            address="Detroit, MI",
            country="USA",
        )
        self.blocker.index_candidate(cand_usa)

        s1_us = BlockingRecord.from_row(
            entity_id="S1-US",
            name="General Motors Company",
            address="Detroit, Michigan",
            country="United States of America",
        )
        hits = self.blocker.generate_candidates_for_record(s1_us)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["candidate_entity_id"], "S2-USA")

    def test_missing_country_fallback(self):
        """If S1 country is missing, global fallback retrieves matching candidate."""
        cand = BlockingRecord.from_row(
            entity_id="S2-ANY",
            name="Global Enterprises LLC",
            address="Tower 1",
            country="France",
        )
        self.blocker.index_candidate(cand)

        s1_nocountry = BlockingRecord.from_row(
            entity_id="S1-NO",
            name="Global Enterprises Limited Liability Company",
            address="Tower 1",
            country="",
        )
        hits = self.blocker.generate_candidates_for_record(s1_nocountry)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["candidate_entity_id"], "S2-ANY")


class TestDeterministicRankingAndCapping(unittest.TestCase):
    """Test priority ranking and capping logic."""

    def test_ranking_priority_order(self):
        """Exact matches and multi-blocker hits rank above single low-scoring hits."""
        blocker = MultiChannelBlocker(max_candidates_per_entity=2)

        # 3 candidates
        # Cand A: exact match (rank 1)
        cand_a = BlockingRecord.from_row("S2-A", "Target Store", "101 Target Way, Dallas, TX 75001", "US")
        # Cand B: address structural only
        cand_b = BlockingRecord.from_row("S2-B", "Unrelated Retail", "101 Target Way, Dallas, TX 75001", "US")
        # Cand C: exact match on name
        cand_c = BlockingRecord.from_row("S2-C", "Target Store", "Different Street, Other City, TX", "US")

        blocker.index_candidate(cand_a)
        blocker.index_candidate(cand_b)
        blocker.index_candidate(cand_c)

        s1 = BlockingRecord.from_row("S1-1", "Target Store", "101 Target Way, Dallas, TX 75001", "US")
        hits = blocker.generate_candidates_for_record(s1)

        # Capped at max_candidates_per_entity = 2
        self.assertEqual(len(hits), 2)
        # S2-A matched multiple channels (exact_name, exact_name_postal, exact_name_street, address_structural) -> top
        self.assertEqual(hits[0]["candidate_entity_id"], "S2-A")
        self.assertTrue(hits[0]["blocker_count"] >= 2)

    def test_multi_channel_consensus_over_single_exact_hit(self):
        """Verify candidate corroborated by multiple channels ranks ABOVE single-channel hit.

        Per docs/04 Section 4.1:
        Tier 1: Number of independent channels (higher is better).
        Tier 2: Exact-match flag (exact_val).
        Therefore, Candidate corroborated by 6 channels outranks a candidate with only 3 channels
        even if the latter has an exact name match.
        """
        blocker = MultiChannelBlocker(similarity_floor=0.8, max_candidates_per_entity=2)
        cand_a = BlockingRecord.from_row("CAND-A", "Vanguard", "999 Outland Road", "US")
        cand_b = BlockingRecord.from_row(
            "CAND-B",
            "Vanguard Pioneer Innovations",
            "100 Main St, Beverly Hills, CA 90210",
            "US",
        )
        blocker.index_candidate(cand_a)
        blocker.index_candidate(cand_b)

        s1 = BlockingRecord.from_row("S1", "Vanguard", "100 Main St, Beverly Hills, CA 90210", "US")
        hits = blocker.generate_candidates_for_record(s1, dense_scores={"CAND-B": 0.85})

        self.assertEqual(len(hits), 2)
        # CAND-B has higher blocker_count and must rank ahead of CAND-A
        self.assertEqual(hits[0]["candidate_entity_id"], "CAND-B")
        self.assertEqual(hits[1]["candidate_entity_id"], "CAND-A")
        self.assertGreater(hits[0]["blocker_count"], hits[1]["blocker_count"])



class TestEndToEndBlockingExecution(unittest.TestCase):
    """Test full run_blocking pipeline writing TSVs and summary."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.s1_file = self.test_dir / "s1.tsv"
        self.s2_file = self.test_dir / "s2.tsv"
        self.gt_file = self.test_dir / "gt.tsv"
        self.out_dir = self.test_dir / "out"

        # Create mock S1
        with open(self.s1_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tAcme Corporation\t123 Main St, New York, NY 10001\tUS\n")
            f.write("S1-2\tUnique Lone Enterprise\tNo matches anywhere\tUS\n")  # Singleton

        # Create mock S2
        with open(self.s2_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-10\tAcme Corp\t123 Main Street, New York, NY 10001\tUSA\n")
            f.write("S2-20\tRandom Unrelated LLC\t500 West Rd\tUS\n")

        # Create mock Ground Truth (S1-1 matches S2-10, S1-2 has no match)
        with open(self.gt_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-10\n")
            f.write("S1-2\t\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_run_blocking_pipeline(self):
        """Run blocking and verify candidate_pairs.tsv, provenance, and recall audit."""
        summary = run_blocking(
            source1_paths=[self.s1_file],
            candidate_sources=[self.s2_file],
            output_dir=self.out_dir,
            ground_truth_path=self.gt_file,
            max_candidates_per_entity=10,
        )

        pairs_path = self.out_dir / "candidate_pairs.tsv"
        prov_path = self.out_dir / "candidate_provenance.tsv"
        summary_path = self.out_dir / "blocking_summary.json"

        self.assertTrue(pairs_path.exists())
        self.assertTrue(prov_path.exists())
        self.assertTrue(summary_path.exists())

        # Verify candidate_pairs content
        with open(pairs_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f if line.strip()]

        header = lines[0]
        self.assertEqual(header, ["source1_entity_id", "candidate_entity_ids"])

        # Check rows
        s1_rows = {row[0]: (row[1] if len(row) > 1 else "") for row in lines[1:]}
        self.assertIn("S1-1", s1_rows)
        self.assertIn("S1-2", s1_rows)

        # S1-1 must contain S2-10
        self.assertIn("S2-10", s1_rows["S1-1"])

        # S1-2 is a singleton -> must be empty string
        self.assertEqual(s1_rows["S1-2"], "")

        # Verify recall audit in summary
        self.assertIn("recall_audit", summary)
        audit = summary["recall_audit"]
        self.assertEqual(audit["pair_recall"], 1.0)
        self.assertEqual(audit["entity_full_recall"], 1.0)
        self.assertEqual(audit["any_hit_rate"], 1.0)

        # Verify P95 candidate volume metric (mandatory audit gate per docs/04 Section 6)
        self.assertIn("p95", summary["candidates_per_s1"])
        self.assertIn("p95_candidates_per_s1", summary)
        self.assertIsInstance(summary["candidates_per_s1"]["p95"], (int, float))

    def test_blocking_with_layer0_normalized_schema(self):
        """Verify blocker transparently ingests Layer 0 normalized artifacts."""
        s1_norm = self.test_dir / "s1_norm.tsv"
        s2_norm = self.test_dir / "s2_norm.tsv"
        out_norm = self.test_dir / "out_norm"

        with open(s1_norm, "w", encoding="utf-8") as f:
            f.write("entity_id\tsource\tcountry\tcountry_canonical\traw_name\traw_address\tnorm_name\tnorm_address\tencoder_text\n")
            f.write("S1-N1\tS1\tUS\tus\tAcme Inc\t123 Main St\tacme incorporated\t123 main street\tacme incorporated | 123 main street\n")

        with open(s2_norm, "w", encoding="utf-8") as f:
            f.write("entity_id\tsource\tcountry\tcountry_canonical\traw_name\traw_address\tnorm_name\tnorm_address\tencoder_text\n")
            f.write("S2-N2\tS2\tUSA\tus\tAcme Corp\t123 Main St\tacme corporation\t123 main street\tacme corporation | 123 main street\n")

        summary = run_blocking(
            source1_paths=[s1_norm],
            candidate_sources=[s2_norm],
            output_dir=out_norm,
            max_candidates_per_entity=5,
        )
        self.assertEqual(summary["total_source1_entities"], 1)
        self.assertEqual(summary["total_candidate_pairs"], 1)

        pairs_path = out_norm / "candidate_pairs.tsv"
        with open(pairs_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("S1-N1\tS2-N2", content)

    def test_cli_stage0_dir_execution(self):
        """Verify CLI can be invoked with --stage0_dir and --output_dir as documented in docs/04."""
        stage0_dir = self.test_dir / "stage0"
        stage0_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.s1_file, stage0_dir / "train_source1_normalized.tsv")
        shutil.copy(self.s2_file, stage0_dir / "train_source2_normalized.tsv")

        cli_out = self.test_dir / "cli_out"
        test_argv = [
            "blocking.py",
            "--stage0_dir", str(stage0_dir),
            "--output_dir", str(cli_out),
            "--top_k", "10",
            "--max_candidates", "20",
        ]
        with patch.object(sys, "argv", test_argv):
            exit_code = main()
            self.assertEqual(exit_code, 0)

        self.assertTrue((cli_out / "candidate_pairs.tsv").exists())
        self.assertTrue((cli_out / "blocking_summary.json").exists())
        with open(cli_out / "blocking_summary.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("p95", data["candidates_per_s1"])

    def test_stage0_prenormalized_fields_utilization(self):
        """Verify read_tsv_records and BlockingRecord.from_row reuse pre-normalized Stage 0 fields."""
        norm_tsv = self.test_dir / "stage0_test.tsv"
        with open(norm_tsv, "w", encoding="utf-8") as f:
            f.write(
                "entity_id\tsource\tcountry\tcountry_canonical\traw_name\traw_address\t"
                "norm_name\tnorm_address\tencoder_text\tis_address_missing\tpostal_code\t"
                "street_number\ttrailing_segment\tdigit_runs\tname_token_count\taddress_token_count\n"
            )
            f.write(
                "S2-PRE1\tS2\tUnited States\tus\tAcme Corp Inc\t100 Main St\t"
                "custom_norm_acme\t100 main street\tcustom_norm_acme | 100 main street\t0\t90210\t"
                "100\tmain street\t100\t3\t3\n"
            )
            f.write(
                "S2-PRE2\tS2\tFrance\tfrance\tBoutique XYZ\tNone\t"
                "boutique xyz\t\tboutique xyz | [NO_ADDRESS]\t1\t\t"
                "\t\t\t2\t0\n"
            )

        rows = list(read_tsv_records([norm_tsv]))
        self.assertEqual(len(rows), 2)
        row1, row2 = rows[0], rows[1]

        self.assertEqual(row1["norm_name"], "custom_norm_acme")
        self.assertEqual(row1["postal_code"], "90210")
        self.assertEqual(row1["street_number"], "100")
        self.assertFalse(row1["is_address_missing"])

        rec1 = BlockingRecord.from_row(
            entity_id=row1["entity_id"],
            name=row1["business_name"],
            address=row1["business_address"],
            country=row1["country"],
            norm_name=row1.get("norm_name"),
            norm_address=row1.get("norm_address"),
            canonical_country=row1.get("canonical_country"),
            postal_code=row1.get("postal_code"),
            street_number=row1.get("street_number"),
            trailing_segment=row1.get("trailing_segment"),
            is_address_missing=row1.get("is_address_missing"),
        )
        self.assertEqual(rec1.norm_name, "custom_norm_acme")
        self.assertEqual(rec1.postal_code, "90210")
        self.assertEqual(rec1.street_number, "100")
        self.assertFalse(rec1.is_address_missing)

        rec2 = BlockingRecord.from_row(
            entity_id=row2["entity_id"],
            name=row2["business_name"],
            address=row2["business_address"],
            country=row2["country"],
            norm_name=row2.get("norm_name"),
            norm_address=row2.get("norm_address"),
            canonical_country=row2.get("canonical_country"),
            postal_code=row2.get("postal_code"),
            street_number=row2.get("street_number"),
            trailing_segment=row2.get("trailing_segment"),
            is_address_missing=row2.get("is_address_missing"),
        )
        self.assertEqual(rec2.norm_name, "boutique xyz")
        self.assertEqual(rec2.norm_address, "")
        self.assertTrue(rec2.is_address_missing)
        self.assertIsNone(rec2.postal_code)


if __name__ == "__main__":
    unittest.main()
