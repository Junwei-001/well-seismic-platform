# 接口与外部适配器

这里保存平台对模型的稳定边界，不保存模型实体。核心协议位于
`src/well_seismic/modeling/contracts.py`，任务—模型—适配器—运行器的登记关系见
`model_registry.json`。`接口模型/` 下的 CIGVis、FaultSeg 和 SurfaceSeg 代码仅作为
可选上游适配参考；在接口版启动模式中不会被加载执行。

外部扩展应通过以下入口点接入，避免修改 `api.py` 或 Vue 页面：

- `well_seismic.interpretation_tasks`：解释任务语义；
- `well_seismic.plugins`：`ModelSpec` 与模型插件；
- `well_seismic.input_adapters`：标准数据到模型输入的适配；
- `well_seismic.prediction_runners`：预测运行器；
- `well_seismic.fusion_strategies`：井震融合策略。

所有外部扩展必须显式声明版本、输入输出轴序、来源/许可、权重 SHA-256 和
`model_id`，并在隔离环境通过接口合同测试后再切换到 `full_runtime`。
