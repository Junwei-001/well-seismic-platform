import numpy as np
import unittest
import torch

from src.losses import conservative_bce_tversky_loss
from src.thresholds import otsu_threshold, select_threshold, validate_threshold


class ThresholdTests(unittest.TestCase):
    def test_otsu_separates_bimodal_probabilities(self) -> None:
        probability = np.r_[np.full(1000, 0.1), np.full(1000, 0.9)].astype(np.float32)
        threshold = otsu_threshold(probability)
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLess(threshold, 0.9)

    def test_threshold_specs(self) -> None:
        probability = np.linspace(0, 1, 101, dtype=np.float32)
        self.assertEqual(select_threshold("profile", probability, 0.7).value, 0.7)
        self.assertEqual(select_threshold("0.8", probability, 0.5).value, 0.8)
        self.assertAlmostEqual(select_threshold("quantile:0.9", probability, 0.5).value, 0.9)

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_threshold(1.1)

    def test_conservative_loss_is_finite(self) -> None:
        logits = torch.zeros((1, 1, 4, 4, 4))
        target = torch.zeros_like(logits)
        target[..., 1, 1, 1] = 1
        self.assertTrue(torch.isfinite(conservative_bce_tversky_loss(logits, target)))
