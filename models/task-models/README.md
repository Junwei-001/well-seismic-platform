# 外部任务模型接入槽位

此目录预留给经过审核的外部模型包。当前为空，不应在这里直接放训练缓存、数据集、
未验收 checkpoint 或 CUDA 环境压缩包。推荐布局：

```text
<model_id>/<version>/checkpoint.<ext>
<model_id>/<version>/manifest.json
```

接入后请更新 `../manifest.json`、`../../interfaces/model_registry.json`，并保留
`model_id`、版本、权重 SHA-256、许可证和输入轴序。
