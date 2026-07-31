from pathlib import Path
import unittest

from src.profiles import detect_profile, load_profile


class ProfileTests(unittest.TestCase):
    def test_detect_fault_enhancement_profile(self) -> None:
        self.assertEqual(detect_profile("9-3_Fault_enhancement_filter.cbvs"), "fault-enhanced")
        self.assertEqual(detect_profile("survey_fef.cbvs"), "fault-enhanced")
        self.assertEqual(detect_profile("Fault enhancement filter.cbvs"), "fault-enhanced")
        self.assertEqual(detect_profile("original_seismic.cbvs"), "field-raw")

    def test_profile_paths_are_portable_and_absolute(self) -> None:
        profile = load_profile("fault-enhanced", Path("example.cbvs"))
        self.assertEqual(profile.name, "fault-enhanced")
        self.assertTrue(profile.checkpoint.is_absolute())
        self.assertEqual(profile.checkpoint.name, "faultseg-best.pt")
        self.assertEqual(profile.threshold, 0.98)
