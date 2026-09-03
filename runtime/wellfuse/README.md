# 运行时占位

本目录在独立接口版中只作为平台相对路径的稳定挂载点，不包含 Python
运行环境、任务脚本、预测产物或任何模型权重。启动时会通过
`WELLFUSE_MODEL_MODE=interfaces_only`（以及 `models/INTERFACE_ONLY` 标记）
阻止任务运行器加载外部资产。

后续如需接入模型，请将经过审查的扩展包放在 `models/task-models/<model_id>/`
或由 Python entry point 注册，并同步更新 `interfaces/model_registry.json`。
不要直接把训练目录、缓存或未校验的 checkpoint 复制到此目录。
