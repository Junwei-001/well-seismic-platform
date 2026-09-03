# 模型目录（当前为空）

本独立版不携带任何任务专属 checkpoint 或模型运行环境。`manifest.json` 保持
平台读取接口，但 `models` 列表为空；稳定模型 ID 与未来接入规则见
`../interfaces/model_registry.json`。

未来权重请按 `models/task-models/<model_id>/<version>/` 放置，并同时提交来源、
许可、输入输出合同和 SHA-256 清单。完成验证后再移除 `INTERFACE_ONLY` 标记并显式
切换 `WELLFUSE_MODEL_MODE=full_runtime`。
