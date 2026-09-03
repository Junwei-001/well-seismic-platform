# 平台运行时挂载点

接口版不携带 Python/Conda/CUDA 离线环境。开发运行请使用本机 Python 安装
`.[web]` 依赖；生产模型运行时由独立扩展包提供。`runtime/wellfuse/` 仅保留相对
路径说明，不含任务脚本、缓存或权重。
