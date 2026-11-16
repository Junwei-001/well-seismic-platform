"""All owned references remain reachable when model receipts are paginated."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

from well_seismic.gaia_agent import _bounded_receipt
from well_seismic.huiyan_harness import HuiyanHarness

OWNER, OTHER = "a" * 64, "b" * 64


class VersionListFixture:
    def __init__(self):
        self.rows = [{
            "revision_id": "dv_" + f"{index:032x}",
            "parent_revision_id": "dv_" + f"{index - 1:032x}" if index else None,
            "created_at": (datetime(2026, 11, 16, tzinfo=timezone.utc) + timedelta(minutes=index)).isoformat(),
            "status": "archived",
            "metadata": {"operation": "curve_mapping", "asset_records": [{"role": "log", "name": "synthetic.las"}] * 30},
            "files": [{"path": "a/synthetic.las", "sha256": "f" * 64, "size": 300}],
        } for index in range(70)]

    def list_versions(self, subject, project_id):
        assert subject == OWNER
        return self.rows


@pytest.fixture
def populated(tmp_path):
    store = VersionListFixture()
    h = HuiyanHarness(tmp_path / "harness", config={}, native={}, dvc_store=store)
    projects = [h.execute(OWNER, "project_create", {"name": "项目" + "井" * 110 + str(index)})["project_id"] for index in range(70)]
    foreign = h.execute(OTHER, "project_create", {"name": "other owner"})["project_id"]
    project_id = projects[0]
    assets = []
    for index in range(70):
        source = h.root / f"asset-{index}.las"
        source.write_text(f"synthetic-{index}", encoding="utf-8")
        asset = h._register_asset(project_id, "log", source, "井" * 110 + f"-{index}.las")
        assets.append(asset["id"])
    return h, project_id, projects, assets, store.rows, foreign


@pytest.mark.parametrize("operation, field", [("project_list", "projects"), ("asset_list", "assets"), ("dvc_list", "versions")])
def test_every_owned_id_is_reachable_without_duplicates_through_model_pages(populated, operation, field):
    h, project_id, projects, assets, versions, _ = populated
    expected = {"projects": list(reversed(projects)), "assets": assets, "versions": [v["revision_id"] for v in reversed(versions)]}[field]
    offset, found, rounds = 0, [], 0
    while offset is not None:
        arguments = {"offset": offset, "limit": 32}
        if operation != "project_list":
            arguments["project_id"] = project_id
        receipt = h.execute(OWNER, operation, arguments)
        assert receipt["status"] == "completed"
        raw = receipt["result"]
        assert raw["total"] == 70 and raw["count"] == len(raw[field]) <= 32
        result = _bounded_receipt(receipt)
        assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 12000
        page = result["result"]
        assert page["total"] == 70 and page["offset"] == offset
        assert page["count"] == len(page[field]) > 0
        if field + "_total" in page:
            assert page[field + "_total"] == 70
            assert page[field + "_count"] == len(page[field])
        found.extend(row["revision_id" if field == "versions" else "id"] for row in page[field])
        next_offset = page["next_offset"]
        assert next_offset is None or offset < next_offset <= 70
        offset = next_offset
        rounds += 1
        assert rounds < 10
    assert found == expected
    assert len(found) == len(set(found)) == 70


@pytest.mark.parametrize("operation", ["project_list", "asset_list", "dvc_list"])
@pytest.mark.parametrize("bad_page", [{"offset": -1}, {"limit": 0}, {"limit": 33}])
def test_page_schema_rejects_invalid_bounds(populated, operation, bad_page):
    h, project, *_ = populated
    arguments = dict(bad_page)
    if operation != "project_list":
        arguments["project_id"] = project
    with pytest.raises(HTTPException) as error:
        h.execute(OWNER, operation, arguments)
    assert error.value.status_code == 422


def test_page_end_out_of_range_and_replay_contract(populated):
    h, project, *_ = populated
    end = h.execute(OWNER, "asset_list", {"project_id": project, "offset": 70})
    assert end["result"]["assets"] == [] and end["result"]["next_offset"] is None
    beyond = h.execute(OWNER, "asset_list", {"project_id": project, "offset": 71})
    assert beyond["status"] == "blocked"
    assert beyond["error"]["code"] == "native_contract_rejected"
    request_id = str(uuid4())
    args = {"project_id": project, "offset": 0, "limit": 10}
    first = h.execute(OWNER, "asset_list", args, request_id)
    assert h.execute(OWNER, "asset_list", args, request_id) == first
    with pytest.raises(HTTPException) as error:
        h.execute(OWNER, "asset_list", {**args, "offset": 10}, request_id)
    assert error.value.status_code == 409


def test_pagination_keeps_subject_and_project_isolation(populated):
    h, project, projects, _, _, foreign = populated
    for operation in ("asset_list", "dvc_list"):
        with pytest.raises(HTTPException) as error:
            h.execute(OTHER, operation, {"project_id": project, "offset": 0, "limit": 32})
        assert error.value.status_code == 404
    other = h.execute(OTHER, "project_list", {"offset": 0, "limit": 32})["result"]
    assert other["total"] == 1 and [p["id"] for p in other["projects"]] == [foreign]
    assert not set(projects) & {p["id"] for p in other["projects"]}
