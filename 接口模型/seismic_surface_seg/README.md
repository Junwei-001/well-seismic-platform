# SurfaceSeg 接口参考（权重未接入）

这里仅保留 SurfaceSeg 的输入/输出适配参考代码与原始许可证。`models/` 下的
三个 checkpoint 没有随独立接口版抽离，因此本目录不可直接执行推理。

未来外部扩展需提供经过校验的 `segformer-base`、`segformer-refine` 和
`mask2former` 资产，并在 `models/task-models/seismic_surface_seg/` 登记 SHA-256
与版本后，才能切换到 `full_runtime`。
