from __future__ import annotations

import json
import threading
import time
from urllib.request import Request, urlopen

import uvicorn

from well_seismic.api import app


HOST = "127.0.0.1"
PORT = 8765
BASE = f"http://{HOST}:{PORT}"


def request_json(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_task(task_id: str, timeout_seconds: float = 300) -> dict:
    task = {}
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        task = request_json(f"/api/v1/tasks/{task_id}")
        if task["status"] in {"completed", "failed"}:
            return task
        time.sleep(0.5)
    return task


def main() -> None:
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="api-verification", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("FastAPI 服务未在规定时间内启动")

        health = request_json("/api/v1/health")
        capabilities = request_json("/api/v1/capabilities")
        demo = request_json("/api/v1/demo-paths")
        if not demo["available"]:
            raise RuntimeError("当前油田参考目录未找到")

        with urlopen(BASE + "/", timeout=15) as response:
            frontend = response.read().decode("utf-8")
        if '<div id="app"></div>' not in frontend:
            raise RuntimeError("FastAPI 未返回 Vue 构建页面")

        payload = {
            "seismic_paths": demo["seismic_paths"],
            "log_paths": demo["log_paths"],
            "well_paths": demo["well_paths"],
            "auxiliary_paths": [],
            "recursive": True,
            "lightweight": True,
        }
        created = request_json("/api/v1/data-preparation/tasks", payload)
        task_id = created["task_id"]
        task = wait_task(task_id)
        if task.get("status") != "completed":
            raise RuntimeError(json.dumps(task.get("error") or task, ensure_ascii=False))

        preprocessing_created = request_json("/api/v1/sample-building/tasks", payload)
        preprocessing_task = wait_task(preprocessing_created["task_id"])
        if preprocessing_task.get("status") != "completed":
            raise RuntimeError(json.dumps(preprocessing_task.get("error") or preprocessing_task, ensure_ascii=False))

        with urlopen(BASE + "/%E7%BB%9F%E4%B8%80%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96", timeout=15) as response:
            dashboard_status = response.status

        print(json.dumps({
            "健康检查": health,
            "Vue页面": "已由FastAPI同源返回",
            "统一数据可视化": f"HTTP {dashboard_status}",
            "工作流模块": len(capabilities["workflow"]),
            "模型组件": len(capabilities["models"]),
            "下游任务": len(capabilities["prediction_tasks"]),
            "任务ID": task_id,
            "任务状态": task["status"],
            "数据准备汇总": task["result"]["summary"],
            "预处理阶段": task["result"]["preparation"]["stages"],
            "样本构建任务": preprocessing_task["status"],
            "多模态匹配": preprocessing_task["result"]["matching"],
        }, ensure_ascii=False, indent=2))
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
