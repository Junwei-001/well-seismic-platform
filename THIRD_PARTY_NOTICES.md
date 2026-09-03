# 第三方组件与许可（接口版）

本目录只保留接口/可视化参考源码，不分发任何第三方模型权重或运行环境。各子项目
仍受其目录内原始许可证约束；未来安装外部模型扩展时，必须重新核对对应权利范围。

| 组件 | 目录 | 许可证/状态 | 注意事项 |
| --- | --- | --- | --- |
| CIGVis | `接口模型/cigvis-main/cigvis-main` | 以目录内 `LICENSE` 为准 | 仅作为可视化适配参考，接口模式不自动加载 |
| FaultSeg / SurfaceSeg 适配参考 | `接口模型/faultSeg-main`、`接口模型/seismic_surface_seg` | 以各目录原始许可为准 | 未携带 checkpoint；不得据此宣称已集成模型 |
| pyproj / PROJ | Python 依赖 | 以发行版许可为准 | 用于坐标参考系解析和重投影 |

LayerPulse/NCS、Direct12B、FaultNet、P13/P17/P18 和 CUDA 环境均不在本接口版中。
本平台自研部分尚未自动指定许可证；公开发布前请由权利人补充根目录 `LICENSE`。
