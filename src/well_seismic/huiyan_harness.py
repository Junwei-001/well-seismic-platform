"""Owner-scoped, durable actions over the native Huiyan engine.

The model selects registered operations and opaque IDs, never local paths or
user identities. This adapter never substitutes fake inference for absent models.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

from .api_models import (
    InspectionRequest,
    PredictionRequest,
    PreprocessingRequest,
    RegistrationPreflightRequest,
)
from .gaia_tasks import authenticate
from .gaia_ui import owned_visualization_html, public_value, visualization_query
from .io.las import _read_text, read_las
from .knowledge import CurveKnowledgeBase
from .platform_mode import interface_only_enabled

SCHEMA = "well-seismic.harness.v1"
MAX_UPLOAD_BYTES = 120 * 1024 * 1024
ROLES = {
    "log": ("log_paths", {".las"}),
    "seismic": ("seismic_paths", {".sgy", ".segy"}),
    "well": ("well_paths", {".csv", ".xlsx", ".json"}),
    "trajectory": ("well_paths", {".csv", ".xlsx", ".dev", ".json"}),
    "survey": ("survey_paths", {".csv", ".xlsx", ".dev", ".json"}),
    "time_depth": ("time_depth_paths", {".csv", ".xlsx", ".txt", ".json"}),
    "interpretation": ("interpretation_paths", {".csv", ".json", ".txt"}),
}
SEMANTIC_FIELDS = set(InspectionRequest.model_fields) - {
    "seismic_paths",
    "survey_paths",
    "log_paths",
    "well_paths",
    "time_depth_paths",
    "interpretation_paths",
    "auxiliary_paths",
    "recursive",
    "use_llm_fallback",
    "survey_attestation",
}
SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        k: v
        for k, v in InspectionRequest.model_json_schema()["properties"].items()
        if k in SEMANTIC_FIELDS
    },
    "additionalProperties": False,
}


class HarnessBlocked(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}


def _object(properties: dict, required=()) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _string(description="") -> dict:
    return {
        "type": "string",
        "description": description,
        "minLength": 1,
        "maxLength": 512,
        "pattern": r"^[^\u0000-\u001f]*$",
    }


P, T, A, R = (
    _string(s)
    for s in ("当前账号的项目ID", "当前项目的任务ID", "当前项目的资产ID", "DVC版本ID")
)
MAX_LIST_PAGE = 32
LIST_PAGE_FIELDS = {
    "offset": {"type": "integer", "minimum": 0, "maximum": 2147483647, "description": "起始偏移；继续查询时使用上次返回的next_offset。"},
    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIST_PAGE, "description": "每页条数，默认32。"},
}
ACTION_SPECS = {
    "project_create": (
        "创建当前账号的慧眼项目，可导入已有盖亚数据检查任务。",
        _object({"name": _string(), "task_id": T}, ["name"]),
    ),
    "project_list": (
        "分页列出当前账号的慧眼项目，最新创建的在前。需要完整列表时继续使用next_offset。",
        _object(dict(LIST_PAGE_FIELDS)),
    ),
    "asset_list": (
        "按登记顺序分页列出项目资产；后续操作使用资产ID，翻页使用next_offset。",
        _object({"project_id": P, **LIST_PAGE_FIELDS}, ["project_id"]),
    ),
    "data_prepare": (
        "真实解析项目资产并创建不可变快照。坐标时深语义必须来自用户或数据，不得猜测。",
        _object(
            {
                "project_id": P,
                "asset_ids": {
                    "type": "array",
                    "items": A,
                    "maxItems": 100,
                    "uniqueItems": True,
                },
                "semantics": SEMANTIC_SCHEMA,
            },
            ["project_id"],
        ),
    ),
    "curve_preview": (
        "用原生LAS解析器读取曲线、单位、样点和处理证据。",
        _object(
            {
                "project_id": P,
                "asset_id": A,
                "curve": _string(),
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            ["project_id", "asset_id"],
        ),
    ),
    "curve_mapping": (
        "将LAS曲线名映射到知识库认可的同义名，如RHOB到DEN，生成新资产并验证数值不变。",
        _object(
            {
                "project_id": P,
                "asset_id": A,
                "source_curve": _string(),
                "target_curve": _string(),
            },
            ["project_id", "asset_id", "source_curve", "target_curve"],
        ),
    ),
    "registration": (
        "执行原生预检与井震标定，保留CRS及时深质量门禁。",
        _object({"project_id": P, "task_id": T}, ["project_id", "task_id"]),
    ),
    "sample_build": (
        "由同源快照与标定结果构建原生井震融合样本。",
        _object(
            {"project_id": P, "task_id": T, "registration_task_id": T},
            ["project_id", "task_id", "registration_task_id"],
        ),
    ),
    "prediction": (
        "执行已安装的原生预测模型；缺权重明确blocked，不生成替代预测。",
        _object(
            {
                "project_id": P,
                "task_id": T,
                "interpretation_task": _string(),
                "model_id": _string(),
                "registration_task_id": T,
            },
            ["project_id", "task_id", "interpretation_task", "model_id"],
        ),
    ),
    "horizon_attribute_analysis": (
        "执行原生层位或物性模型；必须指定原始模型，不以简单统计替代训练模型。",
        _object(
            {
                "project_id": P,
                "task_id": T,
                "interpretation_task": _string(),
                "model_id": _string(),
                "registration_task_id": T,
            },
            ["project_id", "task_id", "interpretation_task", "model_id"],
        ),
    ),
    "task_status": (
        "读取所属任务真实状态，并同步原执行的终态与DVC归档；归档或业务门禁失败明确返回blocked。",
        _object({"project_id": P, "task_id": T}, ["project_id", "task_id"]),
    ),
    "visualization": (
        "返回所属任务的原生预览和查看链接。",
        _object({"project_id": P, "task_id": T}, ["project_id", "task_id"]),
    ),
    "model_assets": ("查询原生模型能力与缺失资产合同。", _object({})),
    "dvc_status": ("查询真实DVC命令与版本存储可用性。", _object({})),
    "dvc_archive": (
        "将受控输入和所属任务结果归档到DVC。",
        _object(
            {
                "project_id": P,
                "asset_ids": {"type": "array", "items": A},
                "task_id": T,
                "parent_revision_id": R,
            },
            ["project_id"],
        ),
    ),
    "dvc_list": (
        "分页列出当前账号项目的DVC版本，最新版本在前；继续查询使用next_offset。",
        _object({"project_id": P, **LIST_PAGE_FIELDS}, ["project_id"]),
    ),
    "dvc_get": (
        "读取DVC版本哈希、文件和操作元数据。",
        _object({"project_id": P, "revision_id": R}, ["project_id", "revision_id"]),
    ),
    "dvc_compare": (
        "比较两个所属项目的DVC版本。",
        _object(
            {"project_id": P, "from_revision_id": R, "to_revision_id": R},
            ["project_id", "from_revision_id", "to_revision_id"],
        ),
    ),
    "dvc_restore": (
        "恢复DVC版本到新目录，不覆盖当前数据或封存证据。",
        _object({"project_id": P, "revision_id": R}, ["project_id", "revision_id"]),
    ),
}


class HuiyanHarness:
    def __init__(
        self,
        root: Path,
        *,
        config: dict,
        native: dict[str, Callable],
        legacy_root: Path | None = None,
        allowed_output_roots=(),
        dvc_store=None,
    ):
        self.root = Path(root).resolve()
        self.config, self.native = config, native
        self.legacy_root = Path(legacy_root).resolve() if legacy_root else None
        self.allowed_roots = [
            self.root,
            *(Path(p).resolve() for p in allowed_output_roots),
        ]
        if self.legacy_root:
            self.allowed_roots.append(self.legacy_root)
        self.dvc = dvc_store
        self.lock = threading.RLock()
        self.active_executions: set[str] = set()

    @contextmanager
    def db(self):
        self.root.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.root / "harness.sqlite3", timeout=30)
        con.row_factory = sqlite3.Row
        con.executescript("""
        CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,subject TEXT NOT NULL,name TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,role TEXT NOT NULL,name TEXT NOT NULL,path TEXT NOT NULL,sha256 TEXT NOT NULL,parent_id TEXT);
        CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY,project_id TEXT NOT NULL,subject TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS uploads(digest TEXT PRIMARY KEY,subject TEXT NOT NULL,project_id TEXT NOT NULL,receipt TEXT);
        CREATE TABLE IF NOT EXISTS executions(id TEXT PRIMARY KEY,subject TEXT NOT NULL,operation_id TEXT NOT NULL,digest TEXT NOT NULL,receipt TEXT NOT NULL,UNIQUE(subject,operation_id));
        """)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    @staticmethod
    def subject(value: str):
        if not re.fullmatch(r"[0-9a-f]{64}", value or ""):
            raise HTTPException(401, "Missing authenticated Gaia identity")

    def project(self, subject: str, project_id: str) -> dict:
        self.subject(subject)
        with self.db() as db:
            row = db.execute(
                "SELECT * FROM projects WHERE id=? AND subject=?", (project_id, subject)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Project not found")
        return dict(row)

    def asset(self, subject: str, project_id: str, asset_id: str) -> dict:
        self.project(subject, project_id)
        with self.db() as db:
            row = db.execute(
                "SELECT * FROM assets WHERE id=? AND project_id=?",
                (asset_id, project_id),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Asset not found")
        value = dict(row)
        path = Path(value["path"]).resolve()
        if (
            not any(path.is_relative_to(root) for root in self.allowed_roots)
            or not path.is_file()
        ):
            raise HarnessBlocked(
                "asset_unavailable", "受控文件不存在或已离开允许的数据根目录"
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
            raise HarnessBlocked(
                "asset_integrity_failed", "受控输入哈希变化，请重新上传为新资产"
            )
        return value

    def assets(self, subject: str, project_id: str) -> list[dict]:
        self.project(subject, project_id)
        with self.db() as db:
            rows = db.execute(
                "SELECT * FROM assets WHERE project_id=? ORDER BY rowid", (project_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def public_asset(asset: dict) -> dict:
        return {
            k: asset[k]
            for k in ("id", "project_id", "role", "name", "sha256", "parent_id")
        }

    @staticmethod
    def logical_asset_name(asset):
        return (
            "a/" + asset["id"].removeprefix("ha_") + Path(asset["name"]).suffix.lower()
        )

    def own_task(self, subject: str, project_id: str, task_id: str) -> dict:
        self.project(subject, project_id)
        with self.db() as db:
            row = db.execute(
                "SELECT task_id FROM tasks WHERE task_id=? AND project_id=? AND subject=?",
                (task_id, project_id, subject),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Task not found in this project")
        return self.native["task"](task_id)

    def assistant_context(self, subject: str, task_id: str | None) -> dict:
        """Resolve the selected task from trusted ownership records without writes."""
        self.subject(subject)
        if task_id is None:
            return {"binding": "unselected"}
        if not isinstance(task_id, str) or not re.fullmatch(r"[0-9a-f]{32}", task_id):
            raise HTTPException(404, "Task not found or not owned")
        database = self.root / "harness.sqlite3"
        if database.is_file():
            # Do not initialize tables, create projects or update task ownership
            # merely because the user opened a chat sidebar.
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
                db.execute("PRAGMA query_only=ON")
                row = db.execute(
                    "SELECT t.project_id,p.name FROM tasks t "
                    "JOIN projects p ON p.id=t.project_id "
                    "WHERE t.task_id=? AND t.subject=? AND p.subject=?",
                    (task_id, subject, subject),
                ).fetchone()
                if row:
                    return {
                        "binding": "bound", "task_id": task_id,
                        "project_id": row[0], "project_name": public_value(row[1]),
                    }
                if db.execute(
                    "SELECT 1 FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone():
                    raise HTTPException(404, "Task not found or not owned")
        if self.legacy_root is None:
            raise HTTPException(404, "Task not found or not owned")
        from .gaia_ui import require_owned_task

        require_owned_task(self.legacy_root, subject, task_id)
        return {
            "binding": "unbound", "task_id": task_id,
            "message": "当前任务尚未关联慧眼项目，请先明确创建或导入关联项目。",
        }

    def bind_task(self, subject: str, project_id: str, task_id: str):
        with self.db() as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row and (row["subject"] != subject or row["project_id"] != project_id):
                raise HTTPException(409, "Task already belongs to another project")
            db.execute(
                "INSERT OR IGNORE INTO tasks VALUES(?,?,?)",
                (task_id, project_id, subject),
            )

    def catalog(self) -> dict:
        actions = []
        for name, (description, parameters) in ACTION_SPECS.items():
            ready = {"status": "ready"}
            if (
                name in {"prediction", "horizon_attribute_analysis"}
                and interface_only_enabled()
            ):
                ready = {
                    "status": "blocked",
                    "reason": "interfaces_only：原始任务模型权重与运行时未安装",
                }
            elif name in {"registration", "sample_build"}:
                ready = {
                    "status": "conditional",
                    "reason": "需要同源地震、测井与时空基准，必须通过原生门禁",
                }
            elif name.startswith("dvc_"):
                ready = {
                    "status": "conditional",
                    "reason": "执行时检查DVC命令、版本和权限",
                }
            actions.append(
                {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                    "readiness": ready,
                }
            )
        return {
            "schema_version": SCHEMA,
            "actions": actions,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "asset_roles": {
                role: sorted(extensions) for role, (_, extensions) in ROLES.items()
            },
            "upload_endpoint": "/api/v1/gaia/harness/projects/{project_id}/assets/{role}?filename=...",
        }

    def _register_asset(self, project_id, role, path, name, parent_id=None):
        value = {
            "id": "ha_" + uuid.uuid4().hex,
            "project_id": project_id,
            "role": role,
            "name": name,
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parent_id": parent_id,
        }
        with self.db() as db:
            db.execute(
                "INSERT INTO assets VALUES(:id,:project_id,:role,:name,:path,:sha256,:parent_id)",
                value,
            )
        return value

    def upload(self, subject, project_id, role, filename, content: bytes):
        self.project(subject, project_id)
        if role not in ROLES or Path(filename).suffix.lower() not in ROLES[role][1]:
            raise HTTPException(422, "Unsupported asset role or file extension")
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Asset size must be between 1 byte and 120 MiB")
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not filename or re.search(r"[:\x00-\x1f]", filename):
            raise HTTPException(422, "Invalid filename")
        digest = hashlib.sha256(
            json.dumps(
                [
                    subject,
                    project_id,
                    role,
                    filename,
                    hashlib.sha256(content).hexdigest(),
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.lock:
            with self.db() as db:
                previous = db.execute(
                    "SELECT receipt FROM uploads WHERE digest=? AND subject=? AND project_id=?",
                    (digest, subject, project_id),
                ).fetchone()
                if previous:
                    if previous["receipt"]:
                        return json.loads(previous["receipt"])
                    raise HTTPException(
                        409,
                        "Upload already claimed; inspect owned assets before retrying",
                    )
                db.execute(
                    "INSERT INTO uploads VALUES(?,?,?,NULL)",
                    (digest, subject, project_id),
                )
            try:
                result = self._upload_new(subject, project_id, role, filename, content)
            except HTTPException:
                # Validation failed before registration; no successful mutation
                # can be replayed under this digest.
                with self.db() as db:
                    db.execute(
                        "DELETE FROM uploads WHERE digest=? AND receipt IS NULL",
                        (digest,),
                    )
                raise
            with self.db() as db:
                db.execute(
                    "UPDATE uploads SET receipt=? WHERE digest=?",
                    (json.dumps(result, ensure_ascii=False), digest),
                )
            return result

    def _upload_new(self, subject, project_id, role, filename, content):
        stored_name = (
            filename
            if len(filename) <= 80
            else filename[:50]
            + "_"
            + hashlib.sha256(filename.encode()).hexdigest()[:12]
            + Path(filename).suffix
        )
        path = self.root / "uploads" / uuid.uuid4().hex[:20] / stored_name
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_bytes(content)
        if role == "log":
            try:
                read_las(
                    path,
                    CurveKnowledgeBase(self.config),
                    self.config.get("preprocessing", {}),
                )
            except Exception as exc:
                path.unlink(missing_ok=True)
                raise HTTPException(422, "LAS could not be parsed") from exc
        asset = self._register_asset(project_id, role, path, filename)
        return {
            "schema_version": SCHEMA,
            "project_id": project_id,
            "asset": self.public_asset(asset),
            **self._auto_archive(subject, project_id, "asset_upload", [asset]),
        }

    def _store(self):
        if self.dvc is None:
            raise HarnessBlocked("dvc_unavailable", "DVC存储未配置，未创建任何版本")
        return self.dvc

    def _auto_archive(self, subject, project_id, operation, assets):
        try:
            receipt = self._store().archive(
                subject,
                project_id,
                {self.logical_asset_name(a): Path(a["path"]) for a in assets},
                {
                    "operation": operation,
                    "asset_ids": [a["id"] for a in assets],
                    "asset_records": [self.public_asset(a) for a in assets],
                },
            )
            return {"revision_id": receipt["revision_id"], "versioning": receipt}
        except Exception as exc:  # noqa: BLE001 - sanitize the registered-action boundary
            return {
                "versioning": {
                    "status": "blocked",
                    "code": getattr(exc, "code", "dvc_failed"),
                    "message": "数据操作已完成，但DVC归档未成功，不能视为版本闭环已完成。",
                }
            }

    def _validate_args(self, operation, args):
        spec = ACTION_SPECS.get(operation) if isinstance(operation, str) else None
        if spec is None or not isinstance(args, dict):
            raise HTTPException(422, "Unknown action or invalid arguments")
        errors = list(Draft202012Validator(spec[1]).iter_errors(args))
        if errors:
            # JSON Schema errors include instance data; never echo model-supplied
            # strings, file paths or tokens through a validation exception.
            raise HTTPException(
                422, "Action arguments do not match the registered schema"
            )

    def execute(self, subject, operation, arguments, operation_id=None):
        self.subject(subject)
        self._validate_args(operation, arguments)
        try:
            operation_id = (
                str(uuid.UUID(operation_id)) if operation_id else str(uuid.uuid4())
            )
        except ValueError as exc:
            raise HTTPException(422, "operation_id must be a UUID") from exc
        digest = hashlib.sha256(
            json.dumps(
                {"operation": operation, "arguments": arguments},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.lock:
            with self.db() as db:
                old = db.execute(
                    "SELECT * FROM executions WHERE subject=? AND operation_id=?",
                    (subject, operation_id),
                ).fetchone()
            if old:
                if old["digest"] != digest:
                    raise HTTPException(
                        409, "operation_id already used for different arguments"
                    )
                return self.get_run(subject, old["id"])
            receipt = {
                "schema_version": SCHEMA,
                "execution_id": "hx_" + uuid.uuid4().hex,
                "operation": operation,
                "operation_id": operation_id,
                "request_sha256": digest,
                "status": "running",
                "result": {},
            }
            if arguments.get("project_id"):
                self.project(subject, arguments["project_id"])
                receipt["project_id"] = arguments["project_id"]
            with self.db() as db:
                db.execute(
                    "INSERT INTO executions VALUES(?,?,?,?,?)",
                    (
                        receipt["execution_id"],
                        subject,
                        operation_id,
                        digest,
                        json.dumps(receipt),
                    ),
                )
            self.active_executions.add(receipt["execution_id"])
            try:
                result = self._dispatch(subject, operation, arguments)
                receipt["result"] = public_value(result)
                receipt["status"] = (
                    "running" if result.get("native_task_submitted") else "completed"
                )
                for key in ("task_id", "project_id", "revision_id"):
                    if result.get(key):
                        receipt[key] = result[key]
                if (result.get("versioning") or {}).get("status") == "blocked":
                    receipt.update(
                        status="blocked",
                        error={
                            "code": "versioning_incomplete",
                            "message": "业务操作已完成，但DVC归档未完成，请重试归档。",
                        },
                    )
            except HarnessBlocked as exc:
                receipt.update(
                    status="blocked",
                    error={"code": exc.code, "message": exc.message},
                    result=public_value(exc.details),
                )
                for key in ("task_id", "project_id", "revision_id"):
                    if receipt["result"].get(key):
                        receipt[key] = receipt["result"][key]
            except HTTPException as exc:
                if exc.status_code in {401, 403, 404}:
                    receipt.update(
                        status="failed",
                        error={
                            "code": "not_found",
                            "message": "资源不存在或不属于当前账号",
                        },
                    )
                    self._save(receipt)
                    self.active_executions.discard(receipt["execution_id"])
                    raise
                receipt.update(
                    status="blocked",
                    error={
                        "code": "native_contract_rejected",
                        "message": str(public_value(exc.detail))[:2500],
                    },
                )
            except Exception as exc:  # noqa: BLE001 - sanitize the registered-action boundary
                receipt.update(
                    status="failed",
                    error={
                        "code": getattr(exc, "code", "execution_failed"),
                        "message": "操作未完成，请检查输入、DVC或业务运行时，未将失败记为成功。",
                    },
                )
            self._save(receipt)
            self.active_executions.discard(receipt["execution_id"])
            return self.get_run(subject, receipt["execution_id"])

    def _save(self, receipt):
        with self.db() as db:
            db.execute(
                "UPDATE executions SET receipt=? WHERE id=?",
                (json.dumps(receipt, ensure_ascii=False), receipt["execution_id"]),
            )

    def get_run(self, subject, execution_id):
        self.subject(subject)
        # UI refresh and AI status queries share one terminal transition. The
        # existing RLock also permits execute() to replay/refresh while held.
        with self.lock:
            return self._get_run_locked(subject, execution_id)

    def _get_run_locked(self, subject, execution_id):
        with self.db() as db:
            row = db.execute(
                "SELECT receipt FROM executions WHERE id=? AND subject=?",
                (execution_id, subject),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Execution not found")
        receipt = json.loads(row[0])
        if (
            receipt["status"] == "running"
            and not receipt.get("task_id")
            and execution_id not in self.active_executions
        ):
            receipt.update(
                status="failed",
                error={
                    "code": "execution_interrupted",
                    "message": "服务中断时操作未生成完成回执；未自动重做可能已执行的操作。",
                },
            )
            self._save(receipt)
        if receipt.get("task_id") and receipt["status"] == "running":
            task = self.own_task(subject, receipt["project_id"], receipt["task_id"])
            native_status = task.get("status")
            receipt["result"]["native_status"] = native_status
            receipt["result"]["progress"] = task.get("progress", 0)
            if native_status == "completed":
                receipt["status"] = "completed"
                receipt["result"]["task"] = public_value(task)
                try:
                    version = self._archive_task(
                        subject, receipt["project_id"], receipt["task_id"], task
                    )
                    receipt["revision_id"] = version["revision_id"]
                    receipt["result"]["versioning"] = version
                except Exception as exc:  # noqa: BLE001 - sanitize the registered-action boundary
                    receipt["result"]["versioning"] = {
                        "status": "blocked",
                        "code": getattr(exc, "code", "dvc_failed"),
                    }
                    receipt.update(
                        status="blocked",
                        error={
                            "code": "versioning_incomplete",
                            "message": "原生任务已完成，但DVC归档未完成，请重试归档。",
                        },
                    )
                registration = (task.get("result") or {}).get("registration") or {}
                if registration.get("business_status") == "blocked":
                    receipt.update(
                        status="blocked",
                        error={
                            "code": "native_business_blocked",
                            "message": "原生标定生成了诊断产物，但未通过业务质量门禁，不能视为有效标定已完成。",
                        },
                    )
                    receipt["result"]["business_status"] = "blocked"
                    receipt["result"]["status_counts"] = registration.get(
                        "status_counts", {}
                    )
            elif native_status in {"failed", "cancelled", "interrupted"}:
                receipt.update(
                    status="failed",
                    error={
                        "code": "native_task_failed",
                        "message": str(
                            public_value(task.get("message") or "原生任务未完成")
                        )[:2000],
                    },
                )
            self._save(receipt)
        return receipt

    def _task_executions(self, subject, project_id, task_id):
        """Refresh only owned native submissions; never recursively query self."""
        self.own_task(subject, project_id, task_id)
        native_operations = {
            "data_prepare", "registration", "sample_build", "prediction",
            "horizon_attribute_analysis",
        }
        with self.lock:
            with self.db() as db:
                rows = db.execute(
                    "SELECT id,receipt FROM executions WHERE subject=? ORDER BY rowid",
                    (subject,),
                ).fetchall()
            summaries = []
            for row in rows:
                prior = json.loads(row["receipt"])
                if (
                    prior.get("project_id") != project_id
                    or prior.get("task_id") != task_id
                    or prior.get("operation") not in native_operations
                    or (prior.get("result") or {}).get("native_task_submitted") is not True
                    or row["id"] in self.active_executions
                ):
                    continue
                # get_run rechecks the owned task before refreshing a running
                # native result. A terminal receipt is reused without rearchive.
                receipt = self.get_run(subject, row["id"])
                summary = {
                    key: receipt[key] for key in (
                        "execution_id", "operation", "status", "revision_id", "error",
                    ) if key in receipt
                }
                versioning = (receipt.get("result") or {}).get("versioning")
                if isinstance(versioning, dict):
                    summary["versioning"] = {
                        key: versioning[key] for key in ("status", "code", "revision_id")
                        if key in versioning
                    }
                summaries.append(summary)
            return summaries

    def _archive_task(self, subject, project_id, task_id, task):
        path = self.root / "receipts" / (task_id + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                json.dumps(public_value(task), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        files = {"results/task.json": path}
        artifact_names = {}

        # Only native-declared output artifacts from an owned task and a trusted
        # data root may enter DVC. No arbitrary model-supplied file paths.
        def collect(value, prefix="result", inside_outputs=False):
            if isinstance(value, dict):
                for key, item in value.items():
                    collect(
                        item,
                        prefix + "/" + re.sub(r"[^A-Za-z0-9_-]", "_", str(key))[:60],
                        inside_outputs
                        or key in {"outputs", "output_files", "artifacts"},
                    )
            elif isinstance(value, list):
                for index, item in enumerate(value[:100]):
                    collect(item, prefix + "/" + str(index), inside_outputs)
            elif inside_outputs and isinstance(value, str) and len(files) < 100:
                candidate = Path(value)
                if not candidate.is_absolute():
                    return
                resolved = candidate.resolve()
                if resolved.is_file() and any(
                    resolved.is_relative_to(root) for root in self.allowed_roots
                ):
                    logical = (
                        "r/"
                        + hashlib.sha256(prefix.encode()).hexdigest()[:16]
                        + resolved.suffix[:10]
                    )
                    files[logical] = resolved
                    artifact_names[logical] = resolved.name

        collect(task.get("result") or {})
        return self._store().archive(
            subject,
            project_id,
            files,
            {
                "operation": "native_task_result",
                "task_id": task_id,
                "artifact_names": artifact_names,
            },
        )

    def _submit(self, subject, project_id, operation, payload):
        created = self.native[operation](payload)
        result = (
            created.model_dump(mode="json")
            if hasattr(created, "model_dump")
            else dict(created)
        )
        self.bind_task(subject, project_id, result["task_id"])
        return {**result, "project_id": project_id, "native_task_submitted": True}

    @staticmethod
    def _list_page(items: list[dict], arguments: dict, field: str, order: str) -> dict:
        offset, limit = arguments.get("offset", 0), arguments.get("limit", MAX_LIST_PAGE)
        total = len(items)
        if offset > total:
            raise HTTPException(422, "Page offset exceeds the collection size")
        page = items[offset:offset + limit]
        end = offset + len(page)
        return {
            field: page, "total": total, "count": len(page),
            "offset": offset, "limit": limit,
            "next_offset": end if end < total else None, "order": order,
        }

    def _dispatch(self, subject, op, args):
        project_id = args.get("project_id", "")
        if op == "project_create":
            project_id = "hp_" + uuid.uuid4().hex
            task_id, legacy_task = args.get("task_id"), None
            if task_id:
                from .gaia_ui import require_owned_task

                if self.legacy_root is None:
                    raise HTTPException(404, "Task not found")
                require_owned_task(self.legacy_root, subject, task_id)
                legacy_task = self.native["task"](task_id)
            with self.db() as db:
                db.execute(
                    "INSERT INTO projects VALUES(?,?,?)",
                    (project_id, subject, args["name"][:120]),
                )
            if legacy_task:
                self.bind_task(subject, project_id, task_id)
                for role, (field, _) in ROLES.items():
                    for value in (legacy_task.get("request") or {}).get(field, []):
                        path = Path(value).resolve()
                        if (
                            self.legacy_root
                            and path.is_relative_to(self.legacy_root)
                            and path.is_file()
                        ):
                            self._register_asset(project_id, role, path, path.name)
            return {"project_id": project_id, "name": args["name"][:120]}
        if op == "project_list":
            with self.db() as db:
                rows = db.execute(
                    "SELECT id,name FROM projects WHERE subject=? ORDER BY rowid DESC",
                    (subject,),
                ).fetchall()
            return self._list_page(
                [dict(row) for row in rows], args, "projects", "newest_first"
            )
        if op == "asset_list":
            return {
                "project_id": project_id,
                **self._list_page(
                    [self.public_asset(a) for a in self.assets(subject, project_id)],
                    args, "assets", "registry_order",
                ),
            }
        if op == "data_prepare":
            ids = args.get("asset_ids")
            if ids is None:
                ids = [a["id"] for a in self.assets(subject, project_id)]
            if not ids:
                raise HarnessBlocked("input_required", "请先上传项目数据")
            assets = [self.asset(subject, project_id, aid) for aid in ids]
            semantics = args.get("semantics", {})
            if set(semantics) - SEMANTIC_FIELDS:
                raise HTTPException(
                    422,
                    "Unsupported semantics; paths and credentials cannot be supplied",
                )
            raw = dict(semantics)
            for role, (field, _) in ROLES.items():
                raw.setdefault(field, []).extend(
                    a["path"] for a in assets if a["role"] == role
                )
            raw.update(recursive=False, use_llm_fallback=False)
            return self._submit(
                subject, project_id, op, InspectionRequest.model_validate(raw)
            )
        if op in {"curve_preview", "curve_mapping"}:
            asset = self.asset(subject, project_id, args["asset_id"])
            if asset["role"] != "log":
                raise HarnessBlocked("las_required", "当前资产不是LAS文件")
            log = read_las(
                Path(asset["path"]),
                CurveKnowledgeBase(self.config),
                self.config.get("preprocessing", {}),
            )
            if op == "curve_mapping":
                return self._map_curve(subject, project_id, asset, log, args)
            limit = max(1, min(500, args.get("limit", 100)))
            indices = np.linspace(
                0, max(0, len(log.depth) - 1), min(limit, len(log.depth)), dtype=int
            )
            names = list(log.curves)
            if args.get("curve"):
                canonical = (
                    CurveKnowledgeBase(self.config)
                    .identify(args["curve"])
                    .standard_name
                )
                names = [canonical] if canonical in log.curves else []
                if not names:
                    raise HarnessBlocked("curve_not_found", "指定曲线不存在")
            finite = lambda a: [float(v) if np.isfinite(v) else None for v in a]
            return {
                "project_id": project_id,
                "asset_id": asset["id"],
                "well_name": log.well_name,
                "samples": len(log.depth),
                "preview_sample_count": len(indices),
                "depth_m": finite(log.depth[indices]),
                "curves": {
                    n: {
                        "values": finite(log.curves[n][indices]),
                        "info": public_value(asdict(log.curve_info[n])),
                    }
                    for n in names
                },
                "issues": public_value(log.issues),
                "processing_steps": log.processing_steps,
            }
        if op in {"registration", "sample_build"}:
            self.own_task(subject, project_id, args["task_id"])
            preflight = self.native["registration_preflight"](
                RegistrationPreflightRequest(source_snapshot_id=args["task_id"])
            )
            raw = (
                preflight.model_dump(mode="json")
                if hasattr(preflight, "model_dump")
                else preflight
            )
            derived_id = raw["derived_snapshot_id"]
            self.bind_task(subject, project_id, derived_id)
            effective = PreprocessingRequest.model_validate(raw["effective_request"])
            effective.source_snapshot_id = derived_id
            effective.output_directory = str(
                self.root / "outputs" / uuid.uuid4().hex[:20]
            )
            if op == "sample_build":
                self.own_task(subject, project_id, args["registration_task_id"])
                effective.registration_task_id = args["registration_task_id"]
            return self._submit(subject, project_id, op, effective)
        if op in {"prediction", "horizon_attribute_analysis"}:
            self.own_task(subject, project_id, args["task_id"])
            if args.get("registration_task_id"):
                self.own_task(subject, project_id, args["registration_task_id"])
            if interface_only_enabled():
                raise HarnessBlocked(
                    "model_assets_missing",
                    "当前为无权重接口版，原始模型权重与运行时未安装，未执行预测。",
                    self._model_contract(),
                )
            payload = PredictionRequest(
                source_task_id=args["task_id"],
                model_id=args["model_id"],
                task_id=args["interpretation_task"],
                registration_task_id=args.get("registration_task_id"),
                output_directory=str(self.root / "outputs" / uuid.uuid4().hex[:20]),
            )
            return self._submit(subject, project_id, "prediction", payload)
        if op in {"task_status", "visualization"}:
            task = self.own_task(subject, project_id, args["task_id"])
            if op == "task_status":
                originals = self._task_executions(subject, project_id, args["task_id"])
                task = self.own_task(subject, project_id, args["task_id"])
                result = {
                    "project_id": project_id,
                    "task_id": args["task_id"],
                    "task": public_value(task),
                    "original_executions": originals,
                }
                if len(originals) == 1 and originals[0].get("revision_id"):
                    result["revision_id"] = originals[0]["revision_id"]
                rejected = next((r for r in originals if r["status"] in {"blocked", "failed"}), None)
                if rejected:
                    error = rejected.get("error") or {}
                    raise HarnessBlocked(
                        error.get("code", "native_task_incomplete"),
                        error.get("message", "原执行未通过业务或归档门禁，不能视为完整完成。"),
                        result,
                    )
                return result
            return {
                "project_id": project_id,
                "task_id": args["task_id"],
                "native_status": task.get("status"),
                "preview": public_value(
                    (task.get("result") or {}).get("visualization_preview") or {}
                ),
                "viewer_url": "/api/v1/gaia/harness/tasks/"
                + args["task_id"]
                + "/visualization",
                "scope": "原生任务可视化，数据与模型结果是否可用由原生页面判定",
            }
        if op == "model_assets":
            return {
                "capabilities": public_value(self.native["capabilities"]()),
                "asset_contract": self._model_contract(),
            }
        if op == "dvc_status":
            return self._store().status()
        if op == "dvc_archive":
            files = {}
            if args.get("task_id"):
                task = self.own_task(subject, project_id, args["task_id"])
                if task.get("status") != "completed":
                    raise HarnessBlocked("task_incomplete", "只可归档已完成任务的结果")
                path = self.root / "receipts" / (args["task_id"] + ".json")
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text(
                        json.dumps(public_value(task), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                files["results/" + path.name] = path
            ids = args.get("asset_ids")
            if ids is None:
                ids = [a["id"] for a in self.assets(subject, project_id)]
            asset_records = []
            for aid in ids:
                a = self.asset(subject, project_id, aid)
                files[self.logical_asset_name(a)] = Path(a["path"])
                asset_records.append(self.public_asset(a))
            if not files:
                raise HarnessBlocked("input_required", "没有可归档的数据或结果")
            return self._store().archive(
                subject,
                project_id,
                files,
                {
                    "operation": op,
                    "task_id": args.get("task_id"),
                    "asset_ids": ids,
                    "asset_records": asset_records,
                },
                parent_revision_id=args.get("parent_revision_id"),
            )
        if op == "dvc_list":
            return self._list_page(
                list(reversed(self._store().list_versions(subject, project_id))),
                args, "versions", "created_at_desc",
            )
        if op == "dvc_get":
            return self._store().get(subject, project_id, args["revision_id"])
        if op == "dvc_compare":
            return self._store().compare(
                subject, project_id, args["from_revision_id"], args["to_revision_id"]
            )
        if op == "dvc_restore":
            receipt = self._store().restore_owned(
                subject, project_id, args["revision_id"]
            )
            # Restored files remain in the store's verified directory. Copy only
            # known data types to a fresh controlled input directory for replay.
            assets = []
            version = self._store().get(subject, project_id, args["revision_id"])
            records = (version.get("metadata") or {}).get("asset_records") or []
            for logical, source in (
                self._store()
                .resolve_restored_files(subject, project_id, receipt["restore_id"])
                .items()
            ):
                suffix = source.suffix.lower()
                record = next(
                    (r for r in records if self.logical_asset_name(r) == logical), None
                )
                role = (
                    record["role"]
                    if record
                    else "log"
                    if suffix == ".las"
                    else "seismic"
                    if suffix in {".sgy", ".segy"}
                    else None
                )
                if role and (role not in ROLES or suffix not in ROLES[role][1]):
                    raise HarnessBlocked(
                        "restore_role_mismatch",
                        "版本资产角色与文件格式不一致，未登记为可执行输入",
                    )
                if role:
                    path = self.root / "restored" / uuid.uuid4().hex / source.name
                    path.parent.mkdir(parents=True, exist_ok=False)
                    path.write_bytes(source.read_bytes())
                    assets.append(
                        self.public_asset(
                            self._register_asset(project_id, role, path, path.name)
                        )
                    )
            return {
                **receipt,
                "restored_assets": assets,
                "replay_scope": "依据版本内封存角色恢复为新资产；不猜测未标注表格角色",
            }
        raise HTTPException(422, "Unsupported operation")

    @staticmethod
    def _model_contract():
        return {
            "mode": "interfaces_only" if interface_only_enabled() else "full_runtime",
            "missing_assets": [
                "原始版本checkpoint含SHA256、来源、许可和指标",
                "配套WellFuse src/wellfuse5090及scripts推理入口",
                "相容Python/CUDA依赖和输入输出配置",
            ],
            "registry": "interfaces/model_registry.json",
            "weights_manifest": "models/manifest.json",
            "runtime_contract": "runtime/wellfuse/README.md",
            "validation_boundary": "传统声波积分/合成地震/互相关标定可由原生算法执行，不等同于P13或LayerPulse模型已验收",
        }

    def _map_curve(self, subject, project_id, asset, log, args):
        source, target = args["source_curve"], args["target_curve"]
        if not all(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", n) for n in (source, target)
        ):
            raise HTTPException(422, "Invalid curve mnemonic")
        kb = CurveKnowledgeBase(self.config)
        canonical = kb.identify(source).standard_name
        if (
            canonical.startswith("UNKNOWN")
            or kb.identify(target).standard_name != canonical
        ):
            raise HarnessBlocked(
                "mapping_not_equivalent",
                "只允许知识库认可的同类曲线别名，单位转换或跨物性映射需单独合同",
            )
        if canonical not in log.curves:
            raise HarnessBlocked("curve_not_found", "原始曲线不存在")
        if not np.any(log.masks[canonical]):
            raise HarnessBlocked(
                "curve_values_unavailable",
                "该曲线没有通过原生单位或数值质量检查的样点；先解决源数据单位/质量问题，不能只改名声称修复",
            )
        original = Path(asset["path"])
        text, _ = _read_text(original)
        lines, section, count = [], "", 0
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("~"):
                section = line.strip()[1:].split()[0].upper()
            if section in {"C", "CURVE"} and re.match(
                r"^\s*" + re.escape(source) + r"\s*\.", line, re.IGNORECASE
            ):
                line = re.sub(
                    r"^(\s*)" + re.escape(source) + r"(?=\s*\.)",
                    lambda m: m[1] + target,
                    line,
                    count=1,
                    flags=re.IGNORECASE,
                )
                count += 1
            lines.append(line)
        if count != 1:
            raise HarnessBlocked("ambiguous_curve_column", "必须恰有一列匹配原始曲线名")
        if source.casefold() == target.casefold() or any(
            i.original_name.casefold() == target.casefold()
            for i in log.curve_info.values()
        ):
            raise HarnessBlocked("target_curve_exists", "目标曲线名已存在，不能覆盖")
        path = (
            self.root
            / "derived"
            / uuid.uuid4().hex[:20]
            / (original.stem + "_" + target + ".las")
        )
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_text("".join(lines), encoding="utf-8")
        mapped = read_las(path, kb, self.config.get("preprocessing", {}))
        valid = set(mapped.curves) == set(log.curves) and np.array_equal(
            mapped.depth, log.depth, equal_nan=True
        )
        valid = valid and all(
            np.array_equal(mapped.curves[c], log.curves[c], equal_nan=True)
            and np.array_equal(mapped.masks[c], log.masks[c])
            for c in log.curves
        )
        if not valid:
            path.unlink(missing_ok=True)
            raise HarnessBlocked(
                "mapping_validation_failed", "原生回读发现数据变化，派生文件未登记"
            )
        derived = self._register_asset(project_id, "log", path, path.name, asset["id"])
        return {
            "project_id": project_id,
            "asset": self.public_asset(derived),
            "source_asset_id": asset["id"],
            "mapping": {
                "from": source,
                "to": target,
                "canonical": canonical,
                "sample_count": len(log.depth),
                "values_unchanged": True,
            },
            **self._auto_archive(subject, project_id, "curve_mapping", [derived]),
        }


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)


def build_router(harness: HuiyanHarness):
    router = APIRouter(
        prefix="/api/v1/gaia/harness", tags=["Huiyan registered actions"]
    )

    @router.get("/actions")
    def actions(request: Request):
        authenticate(request)
        return harness.catalog()

    @router.post("/actions")
    def execute(payload: ActionRequest, request: Request):
        return harness.execute(
            authenticate(request),
            payload.operation,
            payload.arguments,
            str(payload.operation_id),
        )

    @router.get("/runs/{execution_id}")
    def execution(execution_id: str, request: Request):
        return harness.get_run(authenticate(request), execution_id)

    @router.post("/projects/{project_id}/assets/{role}", status_code=201)
    async def upload(project_id: str, role: str, request: Request, filename: str):
        subject = authenticate(request)
        await run_in_threadpool(harness.project, subject, project_id)
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Asset exceeds 120 MiB")
        # LAS parsing, hashing and DVC subprocesses must not block the ASGI loop.
        # Keep the authenticated identity and existing mutation locks unchanged.
        return await run_in_threadpool(
            lambda: harness.upload(subject, project_id, role, filename, bytes(content))
        )

    @router.get("/tasks/{task_id}/visualization")
    def visualization(task_id: str, request: Request):
        subject = authenticate(request)
        with harness.db() as db:
            row = db.execute(
                "SELECT project_id FROM tasks WHERE task_id=? AND subject=?",
                (task_id, subject),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Task not found")
        harness.own_task(subject, row[0], task_id)
        result = harness.native["visualization"](
            task_id=task_id, embed=True, **visualization_query(request)
        )
        document = owned_visualization_html(
            public_value(result.body.decode("utf-8")),
            task_id,
            status_code=result.status_code,
        )
        return HTMLResponse(
            document,
            status_code=result.status_code,
            headers={"Cache-Control": "private, no-store"},
        )

    return router
