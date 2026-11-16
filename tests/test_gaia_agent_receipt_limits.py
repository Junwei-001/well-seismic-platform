"""Large owned result lists retain usable references within the model budget."""
from __future__ import annotations

import json
from unittest.mock import Mock, patch

from well_seismic.gaia_agent import _bounded_receipt, chat_with_huiyan_agent
from well_seismic.gaia_minimax import GaiaMiniMaxSettings

PROJECT = "hp_" + "a" * 32
SETTINGS = GaiaMiniMaxSettings("unit-provider", "MiniMax-M2.7", "unit-only-key")


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def versions_receipt(count=13):
    versions = []
    for index in range(count):
        versions.append({
            "revision_id": "dv_" + f"{index:032x}",
            "parent_revision_id": "dv_" + f"{index - 1:032x}" if index else None,
            "created_at": f"2026-11-16T00:{index % 60:02d}:00+00:00",
            "metadata": {"operation": "curve_mapping", "private_data": "hidden-array" * 500},
            "files": [{"path": "D:/private/raw.las", "values": [1.0] * 300}],
        })
    return {
        "execution_id": "hx_" + "a" * 32, "operation": "dvc_list", "status": "completed",
        "project_id": PROJECT, "result": {"versions": versions},
    }


def test_current_large_version_list_retains_all_revision_references():
    receipt = versions_receipt()
    assert len(encoded(receipt)) > 12000
    result = _bounded_receipt(receipt)
    summaries = result["result"]["versions"]
    assert len(encoded(result)) <= 12000
    assert {v["revision_id"] for v in summaries} == {v["revision_id"] for v in receipt["result"]["versions"]}
    assert result["result"]["versions_total"] == result["result"]["versions_count"] == 13
    assert result["result"]["versions_truncated"] is False
    assert all(v["operation"] == "curve_mapping" for v in summaries)
    assert "private_data" not in str(result) and "D:/private" not in str(result)
    assert "values" not in str(result) and "files" not in str(result)


def test_more_versions_remain_bounded_and_explicitly_truncated():
    receipt = versions_receipt(65)
    result = _bounded_receipt(receipt)
    summaries = result["result"]["versions"]
    assert len(encoded(result)) <= 12000
    assert 0 < len(summaries) < 65
    assert result["result"]["versions_total"] == 65
    assert result["result"]["versions_count"] == len(summaries)
    assert result["result"]["versions_truncated"] is True
    assert summaries[0]["revision_id"] == receipt["result"]["versions"][-1]["revision_id"]
    assert summaries[0]["parent_revision_id"] == receipt["result"]["versions"][-1]["parent_revision_id"]


def test_large_asset_list_keeps_scoped_ids_and_labels_without_paths_or_arrays():
    assets = [{
        "id": "ha_" + f"{index:032x}", "project_id": PROJECT, "role": "log",
        "name": "D:/private/well-" + str(index) + ".las",
        "sha256": "f" * 64, "parent_id": None if index == 0 else "ha_" + "0" * 32,
        "path": "D:/private/actual-source.las", "values": [2.3] * 500,
    } for index in range(70)]
    receipt = {"operation": "asset_list", "status": "completed", "project_id": PROJECT, "result": {"assets": assets}}
    result = _bounded_receipt(receipt)
    items = result["result"]["assets"]
    assert len(encoded(result)) <= 12000
    assert result["result"]["assets_total"] == 70
    assert result["result"]["assets_count"] == len(items)
    assert result["result"]["assets_truncated"] is True
    assert 0 < len(items) < 70
    assert items[0]["id"] == assets[0]["id"] and items[1]["parent_id"] == assets[0]["id"]
    assert items[0]["role"] == "log" and items[0]["sha256"] == "f" * 64
    assert items[0]["name"] == "well-0.las"
    assert "D:/private" not in str(result) and "values" not in str(result) and "\"path\"" not in json.dumps(result)


def test_large_project_list_keeps_public_project_references():
    projects = [{"id": "hp_" + f"{index:032x}", "name": "project" + str(index), "private": "x" * 1000} for index in range(50)]
    result = _bounded_receipt({"operation": "project_list", "status": "completed", "result": {"projects": projects}})
    assert 0 < len(result["result"]["projects"]) < 50
    assert result["result"]["projects_total"] == 50
    assert result["result"]["projects_truncated"] is True
    assert "private" not in json.dumps(result)
    assert len(encoded(result)) <= 12000


def test_optional_summaries_cannot_push_compact_result_over_byte_budget():
    receipt = versions_receipt(30)
    receipt["result"].update(summary="汉" * 2200, asset_contract="汉" * 2200)
    result = _bounded_receipt(receipt)
    assert result["result"]["versions_count"] > 0
    assert len(encoded(result)) <= 12000


def test_small_receipt_is_preserved():
    receipt = {"operation": "asset_list", "status": "completed", "result": {"assets": []}}
    assert _bounded_receipt(receipt) is receipt


def test_model_can_use_retained_revision_id_in_next_real_tool_dispatch(tmp_path):
    h = Mock()
    h.root = tmp_path
    h.catalog.return_value = {"actions": [{"name": name, "description": name, "parameters": {"type": "object"}} for name in ("dvc_list", "dvc_get")]}
    h.execute.side_effect = [versions_receipt(), {"operation": "dvc_get", "status": "completed", "result": {"verified": True}}]
    calls = []
    def provider(settings, payload):
        calls.append(payload)
        if len(calls) == 1:
            content = [{"type": "tool_use", "id": "list1", "name": "dvc_list", "input": {"project_id": PROJECT}}]
        elif len(calls) == 2:
            compact = json.loads(payload["messages"][-1]["content"][0]["content"])
            revision = compact["result"]["versions"][0]["revision_id"]
            content = [{"type": "tool_use", "id": "get1", "name": "dvc_get", "input": {"project_id": PROJECT, "revision_id": revision}}]
        else:
            content = [{"type": "text", "text": "已核对版本。"}]
        return {"model": SETTINGS.model, "content": content, "stop_reason": "tool_use" if len(calls) < 3 else "end_turn"}
    with patch("well_seismic.gaia_agent.request_minimax_message", side_effect=provider):
        result = chat_with_huiyan_agent(SETTINGS, subject="a" * 64, question="列出版本并检查最新版本", context={}, conversation_id=None, harness=h)
    assert h.execute.call_count == 2
    assert h.execute.call_args_list[1].args[2]["revision_id"] == "dv_" + f"{12:032x}"
    assert [item["operation"] for item in result["executions"]] == ["dvc_list", "dvc_get"]
