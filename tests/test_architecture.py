from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from well_seismic.alignment import build_spatial_aligner
from well_seismic.auto_input import build_explicit_paths_manifest
from well_seismic.fusion import ConfidenceGatedFusion, build_default_fusion_registry
from well_seismic.interpretation import build_default_interpretation_registry
from well_seismic.io.las import read_las
from well_seismic.knowledge import CurveKnowledgeBase
from well_seismic.modeling import build_default_registry
from well_seismic.llm.contracts import LLMDecision
from well_seismic.llm.providers import OpenAICompatibleChatProvider
from well_seismic.llm.resolver import DecisionResolver
from well_seismic.llm.settings import LLMSettings
from well_seismic.llm.transformation import create_transformation_draft


CURVE_CONFIG = {
    "curve_knowledge": {
        "GR": {
            "aliases": {"exact": ["GR"]},
            "accepted_units": ["API"],
            "canonical_unit": "API",
            "range": {"hard_min": 0, "hard_max": 500},
        },
    },
    "unit_aliases": {},
    "conversions": {},
}


class DataPreparationArchitectureTests(unittest.TestCase):
    def test_data_preparation_accepts_partial_modalities(self):
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "single.las"
            log_path.write_text("~V\n", encoding="utf-8")
            manifest, inventory = build_explicit_paths_manifest(
                [],
                [log_path],
                require_seismic=False,
                require_logs=False,
            )
        self.assertEqual(len(manifest["inputs"]), 1)
        self.assertEqual(manifest["inputs"][0]["role"], "well_logs")
        self.assertEqual(len(inventory["测井数据路径"]), 1)

    def test_duplicate_depth_is_collapsed_with_configured_mean(self):
        content = """~VERSION INFORMATION
VERS. 2.0
~WELL INFORMATION
WELL. DUPLICATE
~CURVE INFORMATION
DEPT.M : depth
GR.API : gamma
~A
1000 40
1000 60
1001 80
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.las"
            path.write_text(content, encoding="utf-8")
            log = read_las(
                path,
                CurveKnowledgeBase(CURVE_CONFIG),
                {"depth_duplicate": "mean", "sort_depth": True},
            )
        self.assertTrue(np.allclose(log.depth, [1000, 1001]))
        self.assertTrue(np.allclose(log.curves["GR"], [50, 80]))
        self.assertIn("duplicate_depth_samples_collapsed:mean", log.processing_steps)

    def test_wrapped_las_and_feet_depth_are_normalized(self):
        content = """~VERSION INFORMATION
VERS. 2.0
WRAP. YES
~WELL INFORMATION
WELL. WRAPPED
~CURVE INFORMATION
DEPT.FT : depth
GR.API : gamma
~A
1000
40 1001
60
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wrapped.las"
            path.write_text(content, encoding="utf-8")
            config = {
                **CURVE_CONFIG,
                "unit_aliases": {**CURVE_CONFIG["unit_aliases"], "FT": "ft", "M": "m"},
                "conversions": {**CURVE_CONFIG["conversions"], "ft->m": {"scale": 0.3048, "offset": 0}},
            }
            log = read_las(path, CurveKnowledgeBase(config), {})
        self.assertTrue(np.allclose(log.depth, [304.8, 305.1048]))
        self.assertTrue(np.allclose(log.curves["GR"], [40, 60]))
        self.assertIn("depth_unit_converted:ft->m", log.processing_steps)


class _FakeProvider:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self, choice: str, confidence: float = 0.95):
        self.choice = choice
        self.confidence = confidence

    def decide(self, request):
        return LLMDecision(
            request.decision_type,
            self.choice,
            self.confidence,
            "test",
            provider=self.provider_name,
            model=self.model,
            source_hash=request.source_hash,
        )


def _llm_settings() -> LLMSettings:
    return LLMSettings(
        enabled=True,
        provider="openai",
        api_mode="responses",
        base_url="https://api.openai.com/v1",
        model="fake-model",
        api_key="test-key",
        timeout_seconds=1,
        max_retries=0,
        max_output_tokens=128,
        min_confidence=0.82,
        max_calls_per_task=5,
        max_context_chars=2000,
        send_file_names=False,
        use_system_proxy=False,
        audit_decisions=True,
        allowed_decisions=("metadata_role", "curve_mapping", "issue_action"),
    )


class LLMFallbackTests(unittest.TestCase):
    def test_glm_chat_completion_parses_json_object(self):
        parsed = OpenAICompatibleChatProvider._structured_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"choice":"GR","confidence":0.96,"reason":"单位匹配","warnings":[]}'
                        }
                    }
                ]
            }
        )
        self.assertEqual(parsed["choice"], "GR")
        self.assertAlmostEqual(parsed["confidence"], 0.96)

    def test_candidate_outside_allow_list_is_rejected(self):
        resolver = DecisionResolver(_llm_settings(), _FakeProvider("INVENTED"), requested=True)
        decision = resolver.resolve("curve_mapping", "choose", ["GR", "保留原始曲线"], {"unit": "API"})
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.choice, "")

    def test_unknown_curve_can_only_map_to_offered_candidate(self):
        config = {
            "curve_knowledge": CURVE_CONFIG["curve_knowledge"],
            "unit_aliases": CURVE_CONFIG["unit_aliases"],
            "conversions": CURVE_CONFIG["conversions"],
        }
        content = """~VERSION INFORMATION
VERS. 2.0
~WELL INFORMATION
WELL. LLM-MAP
~CURVE INFORMATION
DEPT.M : depth
VENDOR_X.API : vendor gamma channel
~A
1000 40
1001 60
"""
        resolver = DecisionResolver(_llm_settings(), _FakeProvider("GR"), requested=True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unknown_curve.las"
            path.write_text(content, encoding="utf-8")
            log = read_las(path, CurveKnowledgeBase(config), {}, resolver)
        self.assertIn("GR", log.curves)
        self.assertTrue(any(step.startswith("llm_curve_mapping:VENDOR_X->GR") for step in log.processing_steps))
        self.assertTrue(resolver.records[0]["是否采纳"])

    def test_issue_recommendation_is_bounded_and_requires_later_confirmation(self):
        candidates = ["复核井名映射", "排除出高可信融合样本"]
        resolver = DecisionResolver(
            _llm_settings(),
            _FakeProvider("排除出高可信融合样本"),
            requested=True,
        )
        decision = resolver.resolve_issue_action(
            {
                "stage": "well_entity_alignment",
                "severity": "警告",
                "blocking": False,
                "title": "轨迹缺失",
                "message": "仅可按直井降级预览",
                "source": "well.csv",
            },
            candidates,
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertTrue(decision.accepted)
        self.assertIn(decision.choice, candidates)


class ExtensionArchitectureTests(unittest.TestCase):
    def test_safe_transformation_draft_compiles_and_tests_known_unit_rule(self):
        draft = create_transformation_draft(
            task_id="task",
            issue={
                "id": "issue",
                "stage": "log_preprocessing",
                "severity": "警告",
                "title": "单位转换",
                "message": "unit_conversion_unavailable:(m/s)*(kg/m3)->(m/s)*(g/cm3)",
            },
            config={
                "conversions": {
                    "(m/s)*(kg/m3)->(m/s)*(g/cm3)": {"scale": 0.001, "offset": 0},
                }
            },
            generator=None,
        )
        self.assertTrue(draft["valid"])
        self.assertEqual(draft["operations"][0]["op"], "unit_scale")
        self.assertIn("adapter.unit_scale", draft["generated_code"])

    def test_spatial_aligner_is_replaceable_factory_component(self):
        geometry = SimpleNamespace(
            x=np.asarray([0.0, 100.0, 200.0]),
            y=np.asarray([0.0, 100.0, 200.0]),
        )
        reader = SimpleNamespace(geometry=geometry)
        asset = SimpleNamespace(path="synthetic.sgy")
        aligner = build_spatial_aligner({"method": "nearest_trace"}).fit([(asset, reader)])
        match = aligner.match(110.0, 90.0)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.trace_index, 1)
        self.assertLess(match.distance, 15)

        neighborhood = build_spatial_aligner({"method": "nearest_trace", "neighbor_traces": 2}).fit([(asset, reader)])
        neighborhood_match = neighborhood.match(110.0, 90.0)
        self.assertIsNotNone(neighborhood_match)
        assert neighborhood_match is not None
        self.assertEqual(len(neighborhood_match.neighbor_trace_indices), 2)
        self.assertAlmostEqual(sum(neighborhood_match.interpolation_weights), 1.0)

    def test_default_model_registry_exposes_stable_components(self):
        registry = build_default_registry()
        specs = {spec.id: spec for spec in registry.list_specs()}
        self.assertIn("faultseg_3d", specs)
        self.assertEqual(
            specs["faultseg_3d"].metadata["tensor_order"],
            ["N", "C", "Z", "INLINE", "CROSSLINE"],
        )
        self.assertIn("seismic_baseline", specs)
        self.assertIn("well_seismic_alignment", specs)
        self.assertIn("confidence_gated_fusion", specs)
        self.assertEqual(registry.capabilities()["plugin_contract"]["entry_point_group"], "well_seismic.plugins")
        self.assertEqual(
            registry.create("confidence_gated_fusion").algorithm_name,
            "confidence_gated",
        )
        with self.assertRaises(RuntimeError):
            registry.create("seismic_baseline")

    def test_interpretation_tasks_are_bound_to_models_and_runners_dynamically(self):
        model_registry = build_default_registry()
        registry = build_default_interpretation_registry()
        capabilities = {
            item["id"]: item
            for item in registry.capabilities(model_registry.list_specs(), ["faultseg_3d"])
        }
        self.assertTrue(capabilities["fault"]["available"])
        self.assertEqual(capabilities["fault"]["runnable_model_ids"], ["faultseg_3d"])
        self.assertFalse(capabilities["reservoir"]["available"])
        self.assertIn("储层概率体", capabilities["reservoir"]["outputs"])
        self.assertEqual(
            registry.entry_point_group,
            "well_seismic.interpretation_tasks",
        )

    def test_fusion_registry_exposes_baselines_and_learnable_extension(self):
        registry = build_default_fusion_registry()
        capabilities = {item["id"]: item for item in registry.capabilities()}
        self.assertEqual(capabilities["confidence_gated"]["status"], "当前默认")
        self.assertTrue(capabilities["learnable"]["training_required"])
        self.assertIsInstance(
            registry.create("confidence_gated", options={"curve_order": ["GR"]}),
            ConfidenceGatedFusion,
        )
        with self.assertRaisesRegex(ValueError, "需要传入model"):
            registry.create("learnable")


if __name__ == "__main__":
    unittest.main()
