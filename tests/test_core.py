from __future__ import annotations

import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from well_seismic.catalog import build_catalog
from well_seismic.alignment import build_sonic_time_domain_alignment, estimate_static_shift, shift_trace
from well_seismic.depth_time import (
    NoDepthTimeTransform,
    ProvidedTimeDepthTransform,
    SonicIntegratedTimeDepthTransform,
)
from well_seismic.datasets import JsonlMultimodalDataset
from well_seismic.fusion import ConcatenateFusion, ConfidenceGatedFusion, WeightedFusion, build_fusion
from well_seismic.io.las import read_las
from well_seismic.io.segy import SegyReader
from well_seismic.knowledge import CurveKnowledgeBase, convert_unit, normalize_unit
from well_seismic.trajectory import minimum_curvature
from well_seismic.models import WellHead, WellLog
from well_seismic.registry import WellRegistry
from well_seismic.auto_input import build_automatic_manifest, build_explicit_paths_manifest
from well_seismic.io.adaptive_metadata import read_adaptive_metadata
from well_seismic.output_schema import sample_to_chinese


CONFIG = {
    "curve_knowledge": {
        "GR": {"aliases": {"exact": ["GR", "GAMMA"]}, "accepted_units": ["API"], "canonical_unit": "API", "range": {"hard_min": 0, "hard_max": 500}},
        "DT": {"aliases": {"exact": ["DT", "AC"]}, "accepted_units": ["us/ft", "us/m"], "canonical_unit": "us/m", "range": {"hard_min": 30, "hard_max": 2000}},
    },
    "unit_aliases": {"US/F": "us/ft", "US/M": "us/m"},
    "conversions": {"us/ft->us/m": {"scale": 3.280839895, "offset": 0}},
}


class KnowledgeTests(unittest.TestCase):
    def test_alias_and_unit_evidence(self):
        kb = CurveKnowledgeBase(CONFIG)
        info = kb.identify("AC", "US/F", "compressional sonic", np.array([70.0, 80.0]))
        self.assertEqual(info.standard_name, "DT")
        standardized, values, mask, issues = kb.standardize(info, np.array([70.0, 80.0]))
        self.assertTrue(np.allclose(values, [229.65879265, 262.4671916]))
        self.assertTrue(mask.all())
        self.assertFalse(issues)

    def test_unknown_curve_is_preserved(self):
        info = CurveKnowledgeBase(CONFIG).identify("VENDOR_X", "widgets", "", np.array([1.0]))
        self.assertTrue(info.standard_name.startswith("UNKNOWN__"))

    def test_unit_aliases_are_case_and_symbol_tolerant(self):
        aliases = {"MV": "mV", "(M/S)*(G/CM³)": "(m/s)*(g/cm3)"}
        self.assertEqual(normalize_unit("mv", aliases), "mV")
        self.assertEqual(normalize_unit("(m/s)*(g/cm³)", aliases), "(m/s)*(g/cm3)")

    def test_composite_unit_uses_atomic_conversion_rules(self):
        values, converted = convert_unit(
            np.array([2_500_000.0, 3_000_000.0]),
            "(m/s)*(kg/m3)",
            "(m/s)*(g/cm3)",
            {"conversions": {"kg/m3->g/cm3": {"scale": 0.001, "offset": 0}}},
        )
        self.assertTrue(converted)
        self.assertTrue(np.allclose(values, [2500.0, 3000.0]))


class TrajectoryTests(unittest.TestCase):
    def test_vertical_minimum_curvature(self):
        md = np.array([0.0, 100.0, 200.0])
        tvd, east, north = minimum_curvature(md, np.zeros(3), np.zeros(3))
        self.assertTrue(np.allclose(tvd, md))
        self.assertTrue(np.allclose(east, 0))
        self.assertTrue(np.allclose(north, 0))


class LasTests(unittest.TestCase):
    def test_las_parse_null_and_mapping(self):
        content = """~VERSION INFORMATION
VERS. 2.0
~WELL INFORMATION
WELL. TEST-1
NULL. -9999
~CURVE INFORMATION
DEPT.M : depth
GR.API : gamma
AC.US/F : sonic
~A
1000 50 70
1001 -9999 80
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.las"
            path.write_text(content, encoding="utf-8")
            log = read_las(path, CurveKnowledgeBase(CONFIG), {"null_values": [-9999]})
        self.assertEqual(log.well_name, "TEST-1")
        self.assertEqual(set(log.curves), {"GR", "DT"})
        self.assertTrue(np.isnan(log.curves["GR"][1]))


class SegyTests(unittest.TestCase):
    @staticmethod
    def _make_segy(path: Path):
        text = b"C 1 SYNTHETIC".ljust(3200, b" ")
        binary = bytearray(400)
        struct.pack_into(">H", binary, 16, 2000)
        struct.pack_into(">H", binary, 20, 4)
        struct.pack_into(">H", binary, 24, 5)
        struct.pack_into(">H", binary, 300, 0x0100)
        with path.open("wb") as handle:
            handle.write(text)
            handle.write(binary)
            for i in range(4):
                header = bytearray(240)
                struct.pack_into(">h", header, 70, 1)
                struct.pack_into(">i", header, 180, 1000 + i * 10)
                struct.pack_into(">i", header, 184, 2000 + i * 10)
                struct.pack_into(">i", header, 188, 10 + i // 2)
                struct.pack_into(">i", header, 192, 20 + i % 2)
                handle.write(header)
                handle.write(np.asarray([i, i + 1, i + 2, i + 3], dtype=">f4").tobytes())

    def test_geometry_and_trace_read(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.sgy"
            self._make_segy(path)
            reader = SegyReader(path, {"segy": {"profiles": {"standard_3d": {"inline": [189], "crossline": [193], "x": [181], "y": [185], "coordinate_scalar": 71}}, "geometry_sample_traces": 4}})
            geometry = reader.inspect()
            self.assertEqual(geometry.trace_count, 4)
            self.assertEqual(geometry.samples_per_trace, 4)
            self.assertTrue(np.allclose(reader.read_trace(2), [2, 3, 4, 5]))


class CatalogTests(unittest.TestCase):
    def test_hardlink_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            source = root / "a" / "x.las"
            source.write_text("x", encoding="utf-8")
            os.link(source, root / "b" / "x.las")
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text("", encoding="utf-8")
            manifest = {"root": ".", "deduplication": {"skip_duplicates": True}, "inputs": [
                {"role": "well_logs", "directory": "a", "patterns": ["*.las"]},
                {"role": "well_logs", "directory": "b", "patterns": ["*.las"]},
            ]}
            assets, duplicates = build_catalog(manifest, manifest_path)
            self.assertEqual(len(assets), 1)
            self.assertEqual(len(duplicates), 1)


class RegistryTests(unittest.TestCase):
    def test_split_files_join_through_aliases(self):
        registry = WellRegistry({"H47": ["H-47", "h47_new"]})
        registry.add_head(WellHead("H-47", 100.0, 200.0, source="wellhead.dat"))
        log = WellLog("h47_new", np.array([1.0]), {}, {}, {}, {}, "h47.las")
        registry.add_log(log)
        self.assertEqual(len(registry.entities), 1)
        entity = next(iter(registry.entities.values()))
        self.assertEqual(len(entity.heads), 1)
        self.assertEqual(len(entity.logs), 1)

    def test_time_depth_table_preserves_domain_units_and_confidence(self):
        registry = WellRegistry({})
        registry.add_time_depth(
            "H47",
            "checkshot.csv",
            np.array([1000.0, 2000.0]),
            np.array([0.5, 1.2]),
            depth_domain="md",
            depth_unit="ft",
            time_unit="s",
            confidence=0.87,
        )
        table = registry.resolve("H47").time_depth[0]
        self.assertEqual(table.depth_domain, "md")
        self.assertEqual(table.depth_unit, "ft")
        self.assertEqual(table.time_unit, "s")
        self.assertAlmostEqual(table.confidence, 0.87)


class DatasetTests(unittest.TestCase):
    def test_lazy_jsonl_and_numpy_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.jsonl"
            path.write_text('{"well_features":{"GR":50},"seismic_window":[1,2],"horizontal_confidence":0.8,"vertical_confidence":0.7}\n', encoding="utf-8")
            dataset = JsonlMultimodalDataset(path)
            well, seismic, quality = dataset.to_numpy(["GR", "RHOB"])
            self.assertEqual(len(dataset), 1)
            self.assertEqual(well.shape, (1, 2))
            self.assertEqual(seismic.shape, (1, 2))
            self.assertTrue(np.isnan(well[0, 1]))

    def test_chinese_output_is_read_back_as_internal_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "多模态样本.jsonl"
            chinese = sample_to_chinese({"well_name": "井A", "well_features": {"GR": 50}, "seismic_window": [1, 2]})
            import json
            path.write_text(json.dumps(chinese, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(JsonlMultimodalDataset(path)[0]["well_name"], "井A")


class AutomaticInputTests(unittest.TestCase):
    def test_only_coarse_folders_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "01_地震数据").mkdir()
            (root / "02_测井数据").mkdir()
            (root / "03_井相关数据").mkdir()
            manifest, inventory = build_automatic_manifest(root)
            self.assertEqual(len(manifest["inputs"]), 3)
            self.assertIn("井相关目录", inventory)

    def test_independent_absolute_directories_stay_in_place(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seismic = root / "任意地震目录"
            logs = root / "任意测井目录"
            metadata = root / "井位轨迹混合目录"
            seismic.mkdir()
            logs.mkdir()
            metadata.mkdir()
            second_seismic = root / "第二个地震目录"
            second_seismic.mkdir()
            manifest, inventory = build_explicit_paths_manifest([seismic, second_seismic], logs, metadata)
            self.assertEqual(manifest["schema_version"], "2.2-multi-paths")
            self.assertEqual(Path(manifest["inputs"][0]["directory"]), seismic.resolve())
            self.assertEqual(Path(manifest["inputs"][1]["directory"]), second_seismic.resolve())
            self.assertEqual(Path(manifest["inputs"][2]["directory"]), logs.resolve())
            self.assertEqual(len(inventory["地震数据路径"]), 2)

    def test_individual_file_is_accepted_as_a_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seismic = root / "volume.sgy"
            logs = root / "well.las"
            seismic.write_bytes(b"placeholder")
            logs.write_text("~V\n", encoding="utf-8")
            manifest, _ = build_explicit_paths_manifest(seismic, logs)
            self.assertEqual(manifest["inputs"][0]["path"], str(seismic.resolve()))
            self.assertEqual(manifest["inputs"][1]["path"], str(logs.resolve()))

    def test_combined_multiwell_metadata_file(self):
        content = "well_name x y kb md inclination azimuth\nA 10000 20000 100 0 0 0\nA 10000 20000 100 100 10 20\nB 11000 21000 120 0 0 0\nB 11000 21000 120 100 5 30\n"
        aliases = {
            "well_name": ["WELL_NAME"], "x": ["X"], "y": ["Y"], "kb": ["KB"],
            "md": ["MD"], "inclination": ["INCLINATION"], "azimuth": ["AZIMUTH"],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "井位轨迹合并表.dat"
            path.write_text(content, encoding="utf-8")
            result = read_adaptive_metadata(path, aliases)
        self.assertEqual(result.status, "已识别")
        self.assertEqual(len(result.heads), 2)
        self.assertEqual(len(result.trajectories), 2)

    def test_headered_curve_table_is_not_misclassified_as_trajectory(self):
        content = "measuredDepth p_ac den gr\n0 200 2.2 40\n100 210 2.3 45\n"
        aliases = {"md": ["MEASURED_DEPTH"], "tvd": ["TVD"]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "H47curves.dat"
            path.write_text(content, encoding="utf-8")
            result = read_adaptive_metadata(path, aliases)
        self.assertEqual(result.status, "不参与")
        self.assertFalse(result.trajectories)

    def test_headered_vertical_depth_table_is_registered_in_tvd_domain(self):
        content = "wellName verticalDepth p_td\nA 1000 800\nA 1100 900\n"
        aliases = {
            "well_name": ["WELLNAME"],
            "tvd": ["VERTICAL_DEPTH"],
            "depth": ["DEPTH", "VERTICAL_DEPTH"],
            "time": ["P_TD"],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "A_P_TD.dat"
            path.write_text(content, encoding="utf-8")
            result = read_adaptive_metadata(path, aliases)
        self.assertEqual(result.status, "已识别")
        self.assertEqual(result.time_depth_domain, "tvd")
        self.assertIn("A", result.time_depth)


class AlignmentAndFusionTests(unittest.TestCase):
    def test_depth_time_fallback_and_interpolation(self):
        depth = np.array([100.0, 150.0, 200.0])
        self.assertTrue(np.isnan(NoDepthTimeTransform().depth_to_time(depth)).all())
        result = ProvidedTimeDepthTransform(np.array([100.0, 200.0]), np.array([80.0, 180.0])).depth_to_time(depth)
        self.assertTrue(np.allclose(result, [80.0, 130.0, 180.0]))

    def test_sonic_velocity_integration_keeps_explicit_datum(self):
        transform = SonicIntegratedTimeDepthTransform(
            np.array([100.0, 200.0, 300.0]),
            np.array([2000.0, 2000.0, 2000.0]),
            reference_depth_m=0.0,
            datum_time_ms=10.0,
            replacement_velocity_m_s=2000.0,
            max_gap_m=150.0,
        )
        self.assertTrue(np.allclose(transform.depth_to_time(np.array([100.0, 200.0, 300.0])), [110.0, 210.0, 310.0]))

    def test_static_shift_recovers_later_event_and_polarity(self):
        synthetic = np.zeros(128, dtype=float)
        synthetic[40:45] = [0.2, 0.7, 1.0, 0.7, 0.2]
        seismic = -shift_trace(synthetic, 7)
        lag, polarity, correlation = estimate_static_shift(synthetic, seismic, 12)
        self.assertEqual(lag, 7)
        self.assertEqual(polarity, -1)
        self.assertGreater(correlation, 0.99)

    def test_sonic_well_tie_is_candidate_not_training_label_by_default(self):
        depth = np.arange(0.0, 1000.0, 1.0)
        density = np.full(depth.size, 2.2)
        density[350:650] = 2.5
        curves = {"DT": np.full(depth.size, 500.0), "RHOB": density}
        time_axis = np.arange(0.0, 1000.0, 2.0)
        initial = build_sonic_time_domain_alignment(
            depth,
            curves,
            np.zeros(time_axis.size),
            time_axis,
            {"replacement_velocity_m_s": 2000.0, "min_correlation": 0.2},
        )
        self.assertIsNotNone(initial)
        assert initial is not None and initial.synthetic_trace is not None
        seismic = shift_trace(initial.synthetic_trace, 5)
        tied = build_sonic_time_domain_alignment(
            depth,
            curves,
            seismic,
            time_axis,
            {"replacement_velocity_m_s": 2000.0, "min_correlation": 0.2, "max_static_shift_ms": 40.0},
        )
        self.assertIsNotNone(tied)
        assert tied is not None
        self.assertEqual(tied.status, "estimated_tie")
        self.assertAlmostEqual(tied.diagnostics["static_shift_ms"], 10.0)
        self.assertFalse(tied.training_eligible)

    def test_sonic_well_tie_rejects_mismatched_trace_and_time_axis(self):
        depth = np.arange(0.0, 100.0, 1.0)
        curves = {"DT": np.full(depth.size, 500.0), "RHOB": np.full(depth.size, 2.3)}
        with self.assertRaisesRegex(ValueError, "等长"):
            build_sonic_time_domain_alignment(
                depth,
                curves,
                np.zeros(40),
                np.arange(50, dtype=float),
            )

    def test_fusion_protocol(self):
        sample = {"well_features": {"GR": 50.0}, "seismic_window": [1.0, 2.0], "horizontal_confidence": 0.5, "vertical_confidence": 0.8}
        self.assertEqual(len(ConcatenateFusion().transform([sample])[0]["fused_features"]), 3)
        self.assertAlmostEqual(WeightedFusion().transform([sample])[0]["fusion_weight"], 0.4)

    def test_confidence_gated_fusion_and_factory(self):
        samples = [
            {"well_features": {"GR": 50.0, "RHOB": 2.4}, "well_mask": {"GR": True, "RHOB": True}, "seismic_window": [1.0, 2.0], "horizontal_confidence": 0.5, "vertical_confidence": 0.8},
            {"well_features": {"GR": 70.0}, "well_mask": {"GR": True, "RHOB": False}, "seismic_window": [2.0, 3.0], "horizontal_confidence": 1.0, "vertical_confidence": 0.9},
        ]
        fusion = build_fusion({"algorithm": "confidence_gated", "curve_order": ["GR", "RHOB"]})
        self.assertIsInstance(fusion, ConfidenceGatedFusion)
        result = fusion.fit_transform(samples)
        self.assertAlmostEqual(result[0]["fusion_weight"], 0.4)
        self.assertEqual(result[0]["fusion_metadata"]["curve_order"], ["GR", "RHOB"])
        restored = ConfidenceGatedFusion()
        restored.load_state_dict(fusion.state_dict())
        self.assertEqual(restored.curve_order, ["GR", "RHOB"])


if __name__ == "__main__":
    unittest.main()
