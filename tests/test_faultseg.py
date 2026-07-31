from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from well_seismic.config import load_config
from well_seismic.faultseg import (
    FaultSegInputSpec,
    build_faultseg_volume,
    iter_faultseg_patches,
)


class _Reader:
    path = Path("synthetic.sgy")

    def __init__(self) -> None:
        inline, crossline = np.meshgrid(np.arange(10, 14), np.arange(20, 24), indexing="ij")
        self.geometry = SimpleNamespace(
            inline=inline.ravel(),
            crossline=crossline.ravel(),
            samples_per_trace=16,
            profile="standard_3d",
            confidence=0.98,
            issues=["inline_byte=189", "crossline_byte=21"],
        )

    def read_trace(self, trace_index: int, sample_slice: slice) -> np.ndarray:
        return (np.arange(16, dtype=np.float32) + trace_index * 100)[sample_slice]


class FaultSegFirstPreprocessingTests(unittest.TestCase):
    def test_config_loads_faultseg_as_a_first_class_contract(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs", {})
        spec = FaultSegInputSpec.from_config(config)
        self.assertEqual(spec.patch_size, (32, 32, 32))
        self.assertEqual(spec.overlap, (8, 8, 8))
        self.assertEqual(config["faultseg"]["tensor_order"], ["N", "C", "Z", "INLINE", "CROSSLINE"])

    def test_system_reader_is_reordered_to_faultseg_zyx(self):
        volume = build_faultseg_volume(
            _Reader(),
            sample_slice=slice(2, 10),
            inline_slice=slice(0, 4),
            crossline_slice=slice(0, 4),
        )
        self.assertEqual(volume.data.shape, (8, 4, 4))
        self.assertEqual(volume.tensor().shape, (1, 1, 8, 4, 4))
        self.assertAlmostEqual(float(volume.tensor().mean()), 0.0, places=5)
        self.assertAlmostEqual(float(volume.tensor().std()), 1.0, places=5)
        self.assertEqual(volume.provenance["model_order"], ["Z", "INLINE", "CROSSLINE"])

    def test_patches_follow_faultseg_overlap_and_tensor_contract(self):
        reader = _Reader()
        volume = build_faultseg_volume(
            reader,
            sample_slice=slice(0, 16),
            inline_slice=slice(0, 4),
            crossline_slice=slice(0, 4),
        )
        spec = FaultSegInputSpec(patch_size=(8, 4, 4), overlap=(4, 0, 0), patch_multiple=4)
        patches = list(iter_faultseg_patches(volume, spec))
        self.assertEqual([origin for origin, _ in patches], [(0, 0, 0), (4, 0, 0), (8, 0, 0)])
        self.assertTrue(all(tensor.shape == (1, 1, 8, 4, 4) for _, tensor in patches))
        self.assertTrue(all(tensor.dtype == np.float32 for _, tensor in patches))

    def test_faultseg_rejects_non_multiple_patch_dimensions(self):
        with self.assertRaises(ValueError):
            FaultSegInputSpec(patch_size=(30, 32, 32)).validated()


if __name__ == "__main__":
    unittest.main()
