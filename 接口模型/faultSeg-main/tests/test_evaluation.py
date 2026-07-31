import unittest

import numpy as np

from src.evaluation import ProbabilityHistogram, metrics_from_counts


class EvaluationTests(unittest.TestCase):
    def test_metrics_from_known_confusion_matrix(self) -> None:
        metrics = metrics_from_counts(tp=8, fp=2, fn=2, tn=8)
        self.assertAlmostEqual(metrics["precision"], 0.8)
        self.assertAlmostEqual(metrics["recall"], 0.8)
        self.assertAlmostEqual(metrics["dice"], 0.8)
        self.assertAlmostEqual(metrics["iou"], 2 / 3)

    def test_best_threshold_separates_probabilities(self) -> None:
        probability = np.array([0.05, 0.2, 0.75, 0.95], dtype=np.float32)
        truth = np.array([False, False, True, True])
        histogram = ProbabilityHistogram(bins=100)
        histogram.update(probability, truth)
        threshold, metrics = histogram.best("dice")
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.75)
        self.assertAlmostEqual(metrics["dice"], 1.0)

    def test_histograms_merge(self) -> None:
        first = ProbabilityHistogram(100)
        second = ProbabilityHistogram(100)
        first.update(np.array([0.9]), np.array([True]))
        second.update(np.array([0.1]), np.array([False]))
        first.merge(second)
        self.assertAlmostEqual(first.metrics_at(0.5)["dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
