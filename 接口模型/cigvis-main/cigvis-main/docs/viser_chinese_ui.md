# Viser 中文控制台

`cigvis.visernodes.Server` 默认使用中文控制台，切片轴显示为
`Inline（主测线）`、`Crossline（联络测线）` 和 `时间`。控件仅更改显示文案，
`x/y/z` 轴标识、色标名称、遮罩选项值和回调属性保持不变。

通过 `viserplot.create_server()` 创建的服务可以在第一次 `plot3D()` 前配置：

```python
from cigvis import viserplot

server = viserplot.create_server(port=8080)
server.configure_ui(
    language="zh-CN",
    axis_labels={
        "x": "Inline",
        "y": "Crossline",
        "z": "时间 / ms",
    },
    labels={
        "slices_folder": "切片导航",
    },
)
viserplot.plot3D(nodes, server=server, run_app=False)
```

`axis_labels` 也可使用三项列表或元组。若需原英文控制台：

```python
server.configure_ui(language="en")
```

应在 `plot3D()` 创建控件之前调用 `configure_ui()`；已经显示的控制台会在下一次
`plot3D()` 重建时应用新文案。方法返回 `server` 自身，原有公开 API 和 `_guix`、
`_guiy`、`_guiz`、`_guiclim` 等回调句柄保持兼容。
