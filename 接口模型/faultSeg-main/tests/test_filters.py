import unittest

import numpy as np

from src.filters import fault_enhancement_filter, normalized_trace_similarity
from src.volumes import detect_format


class FilterTests(unittest.TestCase):
    def test_similarity_is_high_for_equal_traces(self) -> None:
        trace = np.sin(np.linspace(0, 8, 32, dtype=np.float32))
        volume = np.broadcast_to(trace[:, None, None], (32, 5, 5)).copy()
        similarity = normalized_trace_similarity(volume, half_window=3)
        self.assertGreater(float(similarity[4:-4].mean()), 0.99)

    def test_fault_filter_preserves_shape_and_finite_values(self) -> None:
        random = np.random.default_rng(4)
        volume = random.normal(size=(24, 8, 8)).astype(np.float32)
        enhanced, similarity = fault_enhancement_filter(volume, half_window=3)
        self.assertEqual(enhanced.shape, volume.shape)
        self.assertEqual(similarity.shape, volume.shape)
        self.assertTrue(np.all(np.isfinite(enhanced)))

    def test_supported_extensions(self) -> None:
        self.assertEqual(detect_format("cube.sgy"), "sgy")
        self.assertEqual(detect_format("cube.cbvs"), "cbvs")
        self.assertEqual(detect_format("cube.tiff"), "tiff")
        self.assertEqual(detect_format("cube.dat"), "dat")


if __name__ == "__main__":
    unittest.main()

