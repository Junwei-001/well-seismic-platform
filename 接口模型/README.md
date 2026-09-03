# 上游接口参考代码

此目录只放可选的可视化/推理适配参考源码，不放模型实体。后端保持原有相对路径名
（`接口模型/cigvis-main`、`接口模型/faultSeg-main`、`接口模型/seismic_surface_seg`）
以兼容现有导入；接口版启动时所有任务运行器均被统一门控，不会加载这些代码对应的
checkpoint。实际模型请放到 `models/task-models/` 并按接口清单登记。
