import unittest

from src.inference import patch_starts


class PatchStartsTests(unittest.TestCase):
    def test_covers_tail(self) -> None:
        self.assertEqual(patch_starts(256, 128, 32), [0, 96, 128])

    def test_rejects_invalid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            patch_starts(64, 128, 32)
        with self.assertRaises(ValueError):
            patch_starts(256, 128, 128)
