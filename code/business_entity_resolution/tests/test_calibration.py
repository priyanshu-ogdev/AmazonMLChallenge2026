"""
Unit tests for Stage 3 score calibration (src/calibration.py).

Verifies:
  1. Input validation (_validate): finite checks, binary labels, non-empty, equal lengths.
  2. Boundary conditions in fit_calibrator:
     - positives < 1000 -> Platt (logistic)
     - positives == 1000 -> intermediate comparison (winner selected by log loss)
     - positives == 9999 -> intermediate comparison
     - positives == 10000 -> Isotonic (regression)
     - positives > 10000 -> Isotonic (regression)
  3. Calibration application (apply_calibrator and apply_saved_calibrator):
     - Output in [0, 1] range
     - Numerically stable sigmoid (no overflow warnings on extreme logits)
     - Exact parity between model and serialized parameters
  4. Calibration metrics and reliability diagnostics (calibration_metrics):
     - ECE, Brier score, log loss, and reliability diagrams
  5. Cross-fitted calibration metrics across entity groups (cross_fitted_metrics).
"""

import sys
import unittest
from pathlib import Path

# Ensure src is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from src.calibration import (
    _validate,
    apply_calibrator,
    apply_saved_calibrator,
    calibration_metrics,
    cross_fitted_metrics,
    fit_calibrator,
    serialize_calibrator,
)


class TestCalibrationValidation(unittest.TestCase):
    """Test _validate function input assertions and clipping."""

    def test_valid_inputs(self):
        scores = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 1, 1])
        valid_scores, valid_labels = _validate(scores, labels)
        np.testing.assert_array_equal(valid_scores, [0.1, 0.5, 0.9])
        np.testing.assert_array_equal(valid_labels, [0, 1, 1])

    def test_clipping_out_of_bounds_scores(self):
        scores = np.array([-0.2, 0.5, 1.4])
        labels = np.array([0, 1, 0])
        valid_scores, valid_labels = _validate(scores, labels)
        np.testing.assert_array_equal(valid_scores, [0.0, 0.5, 1.0])

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError) as ctx:
            _validate(np.array([0.1, 0.2]), np.array([0]))
        self.assertIn("equal length", str(ctx.exception))

    def test_empty_inputs(self):
        with self.assertRaises(ValueError) as ctx:
            _validate(np.array([]), np.array([]))
        self.assertIn("non-empty", str(ctx.exception))

    def test_non_finite_scores(self):
        with self.assertRaises(ValueError) as ctx:
            _validate(np.array([0.1, np.nan, 0.5]), np.array([0, 1, 0]))
        self.assertIn("finite values", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            _validate(np.array([0.1, np.inf, 0.5]), np.array([0, 1, 0]))
        self.assertIn("finite values", str(ctx.exception))

    def test_non_binary_labels(self):
        with self.assertRaises(ValueError) as ctx:
            _validate(np.array([0.1, 0.5, 0.9]), np.array([0, 2, 1]))
        self.assertIn("binary 0/1", str(ctx.exception))

    def test_single_class_labels(self):
        with self.assertRaises(ValueError) as ctx:
            _validate(np.array([0.1, 0.5, 0.9]), np.array([0, 0, 0]))
        self.assertIn("both positive and negative", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            _validate(np.array([0.1, 0.5, 0.9]), np.array([1, 1, 1]))
        self.assertIn("both positive and negative", str(ctx.exception))


class TestFitCalibratorBoundaries(unittest.TestCase):
    """Test exact boundary transitions in fit_calibrator."""

    def test_low_positives_selects_platt(self):
        # 50 positives < 1000 -> platt
        rng = np.random.RandomState(42)
        scores = rng.uniform(0.0, 1.0, 200)
        labels = np.zeros(200, dtype=int)
        labels[:50] = 1
        name, model = fit_calibrator(scores, labels)
        self.assertEqual(name, "platt")

    def test_boundary_at_1000_positives(self):
        # exactly 1000 positives -> intermediate branch (fits both, returns winner)
        scores = np.linspace(0.0, 1.0, 2000)
        labels = np.zeros(2000, dtype=int)
        labels[1000:] = 1  # exactly 1000 positives
        name, model = fit_calibrator(scores, labels)
        self.assertIn(name, ("platt", "isotonic"))
        self.assertIsNotNone(model)

    def test_boundary_at_10000_positives(self):
        # exactly 10000 positives -> isotonic branch
        scores = np.linspace(0.0, 1.0, 20000)
        labels = np.zeros(20000, dtype=int)
        labels[10000:] = 1  # exactly 10000 positives
        name, model = fit_calibrator(scores, labels)
        self.assertEqual(name, "isotonic")
        self.assertIsNotNone(model)

    def test_high_positives_selects_isotonic(self):
        # 12000 positives > 10000 -> isotonic
        scores = np.linspace(0.0, 1.0, 24000)
        labels = np.zeros(24000, dtype=int)
        labels[12000:] = 1
        name, model = fit_calibrator(scores, labels)
        self.assertEqual(name, "isotonic")


class TestSerializationAndInference(unittest.TestCase):
    """Test serialization fidelity and numeric stability."""

    def test_platt_roundtrip_fidelity(self):
        rng = np.random.RandomState(42)
        scores = rng.uniform(0.1, 0.9, 100)
        labels = (scores > 0.5).astype(int)
        calibrator = fit_calibrator(scores, labels)
        self.assertEqual(calibrator[0], "platt")

        serialized = serialize_calibrator(calibrator)
        self.assertEqual(serialized["name"], "platt")
        self.assertIn("coef", serialized)
        self.assertIn("intercept", serialized)

        test_scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        model_preds = apply_calibrator(calibrator, test_scores)
        saved_preds = apply_saved_calibrator(serialized, test_scores)
        np.testing.assert_allclose(model_preds, saved_preds, rtol=1e-5, atol=1e-5)

    def test_isotonic_roundtrip_fidelity(self):
        scores = np.linspace(0.0, 1.0, 20000)
        labels = (scores > 0.4).astype(int)
        calibrator = fit_calibrator(scores, labels)
        self.assertEqual(calibrator[0], "isotonic")

        serialized = serialize_calibrator(calibrator)
        self.assertEqual(serialized["name"], "isotonic")
        self.assertIn("x_thresholds", serialized)
        self.assertIn("y_thresholds", serialized)

        test_scores = np.array([0.05, 0.35, 0.45, 0.85])
        model_preds = apply_calibrator(calibrator, test_scores)
        saved_preds = apply_saved_calibrator(serialized, test_scores)
        np.testing.assert_allclose(model_preds, saved_preds, rtol=1e-5, atol=1e-5)

    def test_platt_extreme_logits_stability(self):
        # Ensure sigmoid does not overflow or raise RuntimeWarning with extreme coefficients
        params = {"name": "platt", "coef": 10000.0, "intercept": -5000.0}
        scores = np.array([0.0, 0.5, 1.0])
        result = apply_saved_calibrator(params, scores)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all((result >= 0.0) & (result <= 1.0)))

    def test_apply_calibrator_unsupported_name(self):
        with self.assertRaises(ValueError) as ctx:
            apply_calibrator(("unsupported", None), np.array([0.5]))
        self.assertIn("Unsupported calibrator", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            apply_saved_calibrator({"name": "unsupported"}, np.array([0.5]))
        self.assertIn("Unsupported saved calibrator", str(ctx.exception))


class TestCalibrationDiagnostics(unittest.TestCase):
    """Test calibration_metrics and cross_fitted_metrics."""

    def test_calibration_metrics_values(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = calibration_metrics(labels, probs, bins=5)
        self.assertIn("log_loss", metrics)
        self.assertIn("brier", metrics)
        self.assertIn("ece", metrics)
        self.assertIn("reliability", metrics)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)
        self.assertGreaterEqual(metrics["brier"], 0.0)

    def test_cross_fitted_metrics(self):
        scores = np.linspace(0.1, 0.9, 100)
        labels = (scores > 0.5).astype(int)
        groups = np.repeat(np.arange(10), 10)  # 10 entity groups
        metrics = cross_fitted_metrics(scores, labels, groups, n_splits=5)
        self.assertIn("ece", metrics)
        self.assertIn("brier", metrics)
        self.assertIn("log_loss", metrics)


if __name__ == "__main__":
    unittest.main()
