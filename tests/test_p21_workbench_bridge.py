"""P21 readiness and archive labels must follow actual server evidence."""
from contextlib import contextmanager
from unittest.mock import Mock
import json

import pytest

from well_seismic.gaia_ui import owned_visualization_html


def test_point_document_does_not_get_optimistic_html_ready_signal():
    document = '<html><body><main id="workbench"><section data-point-viewer></section></main></body></html>'
    result = owned_visualization_html(document, "a" * 32)
    assert "huiyan:visualization-ready" not in result


@pytest.mark.parametrize("status,revision,version_status,expected", [
    ("completed", "dv_test", "archived", True),
    ("running", "dv_test", "archived", False),
    ("completed", None, "archived", False),
    ("completed", "dv_test", "blocked", False),
])
def test_archive_badge_requires_verified_version(monkeypatch, status, revision, version_status, expected):
    from well_seismic import api

    receipt = {"status": status, "revision_id": revision, "project_id": "owned-project"}
    database = Mock()
    database.execute.return_value.fetchall.return_value = [{"subject": "owned-user", "receipt": json.dumps(receipt)}]

    @contextmanager
    def db():
        yield database

    store = Mock()
    store.get.return_value = {"status": version_status}
    monkeypatch.setattr(api, "_huiyan_harness", Mock(db=db, dvc=store))
    assert api._point_task_archived("a" * 32) is expected
    assert database.execute.call_args.args[1] == ("a" * 32,)
    if status == "completed" and revision:
        store.get.assert_called_once_with("owned-user", "owned-project", revision)
    else:
        store.get.assert_not_called()
