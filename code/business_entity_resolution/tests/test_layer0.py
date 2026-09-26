"""
Tests for Layer 0: Normalization, Country Handling, Structural Extraction,
and Stage 0 Artifact Contract.

Built with Python standard library unittest to run seamlessly in any
environment, including headless or minimal environments without third-party packages.
"""

import unittest
import sys
from pathlib import Path

# Ensure src is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.normalize import (
    normalize_name,
    normalize_address,
    normalize_entity,
    normalize_entity_record,
    canonicalize_country,
    country_match_flag,
    source_from_entity_id,
    extract_postal_code,
    extract_structural_fields,
    LEGAL_SUFFIX_MAP,
    ADDRESS_SUFFIX_MAP,
)

try:
    from src.data_builder import STAGE0_TSV_COLUMNS
except ImportError:
    STAGE0_TSV_COLUMNS = [
        "entity_id",
        "source",
        "country",
        "country_canonical",
        "raw_name",
        "raw_address",
        "norm_name",
        "norm_address",
        "encoder_text",
        "is_address_missing",
        "postal_code",
        "street_number",
        "trailing_segment",
        "digit_runs",
        "name_token_count",
        "address_token_count",
    ]


class TestLayer0Normalization(unittest.TestCase):
    """Unit tests for Stage 0 normalization and country-agnostic rules."""

    def test_unicode_and_accent_preservation(self):
        """Verify NFKC preserves French accents and Devanagari while standardizing typography."""
        french_name = "Société Générale SAS"
        norm_french = normalize_name(french_name)
        self.assertIn("société", norm_french)
        self.assertIn("générale", norm_french)
        self.assertIn("sas", norm_french)

        quoted = "“O’Connor & Sons—Retail”"
        norm_quoted = normalize_name(quoted)
        self.assertIn("o'connor", norm_quoted)
        self.assertIn("and", norm_quoted)
        self.assertIn("sons", norm_quoted)

    def test_legal_suffix_position_aware(self):
        """Legal suffix canonicalization applies only to the trailing 3 tokens."""
        mid_co = "Co Operation Services Inc"
        norm_co = normalize_name(mid_co)
        self.assertTrue(norm_co.startswith("co operation"))
        self.assertTrue(norm_co.endswith("incorporated"))

        pvt_ltd = "Shri Krishna Enterprises Pvt Ltd"
        norm_pvt = normalize_name(pvt_ltd)
        self.assertTrue(norm_pvt.endswith("private limited"))

        # Dotted abbreviations (L.L.C., L.L.P., S.A.R.L., S.A.S., P. Ltd.)
        self.assertEqual(normalize_name("Acme L.L.C."), "acme llc")
        self.assertEqual(normalize_name("Acme L.L.C"), "acme llc")
        self.assertEqual(normalize_name("Acme L. L. C."), "acme llc")
        self.assertEqual(normalize_name("Acme L.L.P."), "acme llp")
        self.assertEqual(normalize_name("Dupont S.A.R.L."), "dupont sarl")
        self.assertEqual(normalize_name("Dupont S.A.S."), "dupont sas")
        self.assertEqual(normalize_name("Dupont S.A."), "dupont sa")
        self.assertEqual(normalize_name("Shri Krishna Enterprises P. Ltd."), "shri krishna enterprises private limited")
        self.assertEqual(normalize_name("Shri Krishna Enterprises P Ltd"), "shri krishna enterprises private limited")

        # Enterprise vs Enterprises canonicalization symmetry
        self.assertEqual(normalize_name("Acme Enterprise"), "acme enterprises")
        self.assertEqual(normalize_name("Acme Enterprises"), "acme enterprises")

        fr_sarl = "Dupont et Freres SARL"
        norm_sarl = normalize_name(fr_sarl)
        self.assertTrue(norm_sarl.endswith("sarl"))

    def test_address_abbreviation_expansion(self):
        """Street types expanded; directions expanded only adjacent to street types."""
        addr1 = "123 MG Rd, Sector 4"
        norm1 = normalize_address(addr1)
        self.assertIn("road", norm1)
        self.assertIn("sector 4", norm1)

        # Direction N adjacent to street type (St / street)
        addr2 = "456 Main St N"
        norm2 = normalize_address(addr2)
        self.assertIn("north", norm2)
        self.assertIn("street", norm2)

        addr3 = "Plot 12, Block A, Salt Lake"
        norm3 = normalize_address(addr3)
        self.assertIn("block a", norm3)

        # Documented French abbreviations bd -> boulevard and r. -> rue
        self.assertEqual(normalize_address("123 Bd Voltaire"), "123 boulevard voltaire")
        self.assertEqual(normalize_address("12 R. de la Paix"), "12 rue de la paix")

    def test_landmark_removal_non_greedy(self):
        """Landmark patterns remove landmark phrase without deleting trailing address content."""
        addr = "797, Lake Town Block A, Kolkata, near SBI Bank, Howrah, West Bengal 700089"
        norm = normalize_address(addr)
        self.assertNotIn("near sbi bank", norm)
        self.assertIn("lake town block a", norm)
        self.assertIn("howrah", norm)
        self.assertIn("west bengal", norm)

    def test_postal_code_extraction(self):
        """Postal code extraction works across India PIN, US ZIP, and French postal codes."""
        self.assertEqual(extract_postal_code("New Delhi 110001"), "110001")
        self.assertEqual(extract_postal_code("Columbus, OH 43215"), "43215")
        self.assertEqual(extract_postal_code("Cleveland, OH 44114-2201"), "44114")
        self.assertEqual(extract_postal_code("29 Boulevard Haussmann, 75009 Paris"), "75009")
        self.assertIsNone(extract_postal_code("No postal code here"))

        # Country-aware extraction preventing cross-country misfires
        self.assertEqual(extract_postal_code("123456 Main Street, Columbus, OH 43215", country="US"), "43215")
        self.assertEqual(extract_postal_code("123456 Main Street, Columbus, OH 43215"), "43215")
        self.assertIsNone(extract_postal_code("Plot 12345, MG Road, Bangalore", country="India"))

    def test_structural_field_extraction(self):
        """Extracts street number, trailing segment, and digit runs accurately."""
        res = extract_structural_fields("1795 Westchester Drive, High Point, NC")
        self.assertEqual(res["street_number"], "1795")
        self.assertEqual(res["trailing_segment"], "NC")
        self.assertIn("1795", res["digit_runs"])

    def test_country_open_set_canonicalization(self):
        """Country canonicalization handles aliases and preserves unseen open-set countries."""
        self.assertEqual(canonicalize_country("US"), "us")
        self.assertEqual(canonicalize_country("USA"), "us")
        self.assertEqual(canonicalize_country("United States"), "us")
        self.assertEqual(canonicalize_country("India"), "india")
        self.assertEqual(canonicalize_country("Bharat"), "india")
        self.assertEqual(canonicalize_country("France"), "france")
        self.assertEqual(canonicalize_country("FR"), "france")
        # Unseen country passes through in lowercase without error
        self.assertEqual(canonicalize_country("Germany"), "germany")
        self.assertEqual(canonicalize_country("Brazil"), "brazil")
        self.assertEqual(canonicalize_country(""), "")

        # Match flag
        self.assertEqual(country_match_flag("US", "USA"), 1)
        self.assertEqual(country_match_flag("India", "France"), 0)
        self.assertEqual(country_match_flag("Germany", "germany"), 1)
        self.assertIsNone(country_match_flag("US", ""))
        self.assertIsNone(country_match_flag("", ""))

    def test_missing_address_contract(self):
        """Missing address produces explicit sentinel and empty normalized string."""
        rec = normalize_entity_record("Acme Corp", "", entity_id="S1-001", country="US")
        self.assertTrue(rec["is_address_missing"])
        self.assertEqual(rec["norm_address"], "")
        self.assertIn("[NO_ADDRESS]", rec["encoder_text"])
        self.assertEqual(rec["address_token_count"], 0)

        rec_nan = normalize_entity_record("Acme Corp", "NaN", entity_id="S1-002", country="US")
        self.assertTrue(rec_nan["is_address_missing"])
        self.assertEqual(rec_nan["norm_address"], "")
        self.assertIn("[NO_ADDRESS]", rec_nan["encoder_text"])

    def test_normalize_entity_record_complete_schema(self):
        """Verify all 16 canonical fields exist with proper types in normalize_entity_record."""
        rec = normalize_entity_record(
            name="Dupont SARL",
            address="10 Rue de la Paix, 75002 Paris",
            entity_id="S1-12345",
            country="France",
        )
        expected_keys = {
            "entity_id",
            "source",
            "country",
            "country_canonical",
            "raw_name",
            "raw_address",
            "norm_name",
            "norm_address",
            "encoder_text",
            "is_address_missing",
            "postal_code",
            "street_number",
            "trailing_segment",
            "digit_runs",
            "name_token_count",
            "address_token_count",
        }
        self.assertTrue(expected_keys.issubset(set(rec.keys())))
        self.assertEqual(rec["source"], "S1")
        self.assertEqual(rec["country_canonical"], "france")
        self.assertEqual(rec["postal_code"], "75002")
        self.assertEqual(rec["street_number"], "10")
        self.assertGreater(rec["name_token_count"], 0)
        self.assertGreater(rec["address_token_count"], 0)

    def test_stage0_tsv_columns_alignment(self):
        """Verify STAGE0_TSV_COLUMNS contains all normalized record keys for artifact persistence."""
        expected_tsv_cols = [
            "entity_id",
            "source",
            "country",
            "country_canonical",
            "raw_name",
            "raw_address",
            "norm_name",
            "norm_address",
            "encoder_text",
            "is_address_missing",
            "postal_code",
            "street_number",
            "trailing_segment",
            "digit_runs",
            "name_token_count",
            "address_token_count",
        ]
        self.assertEqual(STAGE0_TSV_COLUMNS, expected_tsv_cols)

    def test_source_from_entity_id(self):
        """Prefix derivation S1/S2/S3 is strict."""
        self.assertEqual(source_from_entity_id("S1-1002"), "S1")
        self.assertEqual(source_from_entity_id("S2-998877"), "S2")
        self.assertEqual(source_from_entity_id("S3-00012"), "S3")
        self.assertEqual(source_from_entity_id("INVALID-123"), "")
        self.assertEqual(source_from_entity_id(""), "")

    def test_tsv_unclosed_quote_resilience(self):
        """INV-5: Stray leading quote in business name must not swallow subsequent rows."""
        import csv
        import io
        tsv_data = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-000001\t\"6 Inch Sub Shop\t123 Main St\tUS\n"
            "S2-000002\tNormal Business Name\t456 Oak Ave\tUS\n"
        )
        reader = csv.DictReader(io.StringIO(tsv_data), delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["entity_id"], "S2-000001")
        self.assertEqual(rows[1]["entity_id"], "S2-000002")
        self.assertEqual(rows[1]["business_name"], "Normal Business Name")

    def test_street_number_s3_hash_prefix(self):
        """Street number extraction correctly handles S3-style '##' address prefixes."""
        struct = extract_structural_fields("##1234 Willow Oak Lane, Fl. 0, Saint Louis, Missouri 63108")
        self.assertEqual(struct.get("street_number"), "1234")
        self.assertEqual(struct.get("trailing_segment"), "Missouri 63108")

        rec = normalize_entity_record("Willow Oak Clinic", "##8 Willow Oak Lane, Saint Louis, MO 63108")
        self.assertEqual(rec["street_number"], "8")

    def test_postal_code_us_street_number_guard(self):
        """US postal extraction does not falsely treat a 5-digit leading street number as a ZIP code."""
        addr_no_zip = "12045 Main St, Springfield, IL"
        postal = extract_postal_code(addr_no_zip, country="us")
        self.assertIsNone(postal)

        addr_with_zip = "12045 Main St, Springfield, IL 62701"
        postal_real = extract_postal_code(addr_with_zip, country="us")
        self.assertEqual(postal_real, "62701")

    def test_is_missing_address_canonical_coverage(self):
        """Canonical missing address check detects all vendor placeholder strings."""
        from src.normalize import is_missing_address
        placeholders = [
            "", "   ", None, "nan", "NaN", "null", "NULL", "none",
            "no address", "no address available", "not available",
            "unknown", "n/a", "N/A", "missing", "Missing Address"
        ]
        for p in placeholders:
            self.assertTrue(is_missing_address(p), f"Failed to identify '{p}' as missing address")

        real_addresses = ["123 Main St", "PO Box 45", "MG Road Bangalore"]
        for a in real_addresses:
            self.assertFalse(is_missing_address(a), f"Falsely identified '{a}' as missing address")


if __name__ == "__main__":
    unittest.main()
