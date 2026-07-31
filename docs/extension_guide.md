# 下游解释与井震融合接入指南

## 目标边界

平台将扩展点拆成四层，新增代码时不要把任务逻辑重新写回 `api.py` 或前端页面：

1. **解释任务**：定义地质语义、标准输出和评价指标；
2. **模型规格**：声明模型属于哪个任务，以及模型输入输出；
3. **输入适配器与运行器**：把标准数据转换为模型张量并执行推理；
4. **井震融合策略**：形成可被一个或多个下游任务消费的统一表征。

```text
标准数据快照
  ├── 模型输入适配器 ──► 地震/测井/多模态张量
  ├── 井震对齐 ───────► 距离、偏移、置信度、不确定性
  └── 融合策略 ───────► 统一表征
                         │
                         ▼
                    下游任务模型
                         │
                         ▼
                  概率/分类/层位/目标体
```

## 新增下游解释代码

### 1. 注册任务语义

内置任务位于 `src/well_seismic/interpretation/registry.py`。新增平台内置任务时注册
`InterpretationTaskSpec`；外部包可使用入口点组 `well_seismic.interpretation_tasks`。

任务规格必须给出：

- 唯一 `id`；
- 中文名称、简短名称和用途说明；
- 标准输出；
- 必需数据模态；
- 评价指标；
- 显示顺序和契约版本。

### 2. 注册模型规格

在模型插件中注册 `ModelSpec`，并用元数据绑定任务：

```python
ModelSpec(
    id="my_reservoir_model",
    name="储层预测模型",
    category="井震多模态解释",
    status="可运行",
    description="...",
    inputs=("井震融合特征",),
    outputs=("储层概率体", "不确定性"),
    metadata={"prediction_task": "reservoir"},
)
```

任务和模型不是同一层：同一任务可以有多个模型，同一融合表征也可以被多个任务使用。

### 3. 注册输入适配器

在 `src/well_seismic/modeling/input_adapters.py` 实现并注册 `ModelInputAdapter`。外部包可通过
入口点组 `well_seismic.input_adapters` 暴露 `factory(config) -> adapter`。
适配器负责：

- 声明来源格式、轴序、dtype、patch、归一化和必需模态；
- 从标准读取器构建模型张量；
- 校验 shape、有限值、坐标和掩码；
- 写入来源、裁剪范围和配置版本。

不要让模型代码直接解析页面参数，也不要复用可视化产生的隐式转置数组。

### 4. 注册推理运行器

在独立模块实现运行器，再通过 `PredictionRunnerRegistry.register(model_id, runner)` 注册。外部包可
通过入口点组 `well_seismic.prediction_runners` 暴露运行器，入口点名称必须等于 `model_id`。
运行器接收公共参数和 `options` 扩展参数，输出至少包含：

- `model_id`、模型版本和运行设备；
- 输入轴序、shape、来源与适配记录；
- 推理参数；
- 可量化摘要；
- 输出文件映射。

后端会补充 `task_id` 和 `task_name`，并通过 `source_task_id` 保留任务血缘。

### 5. 验证接入

至少增加以下测试：

- 任务—模型绑定正确；
- 输入适配器对真实或最小合成样本兼容；
- 运行器输出契约稳定；
- 错误任务、错误模型、错误轴序和过小体块被明确拒绝；
- 前端构建后，新任务自动出现在任务页和模型中心。

## 新增井震融合方案

融合策略在 `src/well_seismic/fusion.py` 的 `FusionRegistry` 中注册。每个策略由
`FusionStrategySpec` 和工厂组成，配置仍由 `configs/fusion.yaml` 选择。
外部融合包可使用入口点组 `well_seismic.fusion_strategies`，入口对象实现
`register(fusion_registry)`。

当前可用层级：

| 策略 | 用途 | 状态 |
| --- | --- | --- |
| `identity` | 数据契约联调、消融 | 内置 |
| `concatenate` | 直接拼接对照 | 内置 |
| `weighted` | 对齐置信度加权 | 内置 |
| `confidence_gated` | 稳健标准化、掩码与置信度门控 | 当前默认 |
| `learnable` | 门控网络、交叉注意力或多模态 Transformer | 待注入模型 |

推荐按三阶段推进：

1. **稳健基线**：固定曲线顺序，保留缺失掩码，用水平/垂向置信度门控地震特征；
2. **可学习融合**：地震编码器、测井编码器、位置编码器与质量门控共同输入交叉注意力模块；
3. **任务验证**：共享融合骨干连接不同任务头，与地震单模态和直接拼接基线做同井外推对比。

可学习融合实现必须保留：

- `fit`、`transform`、`fit_transform`；
- `state_dict`、`load_state_dict`；
- 固定曲线顺序、归一化参数和契约版本；
- 水平/垂向置信度、缺失掩码和来源；
- 按井划分训练/验证/测试，避免同井样点泄漏。

## 代码落点

```text
src/well_seismic/
├── api.py                         HTTP路由与任务队列
├── api_models.py                  稳定请求/响应模型
├── platform_capabilities.py       平台能力文档组装
├── interpretation/
│   ├── contracts.py               下游任务规格
│   └── registry.py                任务注册与运行状态汇总
├── modeling/
│   ├── contracts.py               模型/融合插件协议
│   ├── registry.py                模型注册
│   └── input_adapters.py          模型输入适配
├── prediction.py                  推理运行器注册与实现
└── fusion.py                      融合策略、工厂和内置基线

frontend/src/
├── api.ts                         后端契约类型
├── domain/platform.ts             页面领域配置与显示规则
└── App.vue                        页面编排
```
