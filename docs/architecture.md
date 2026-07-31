# 系统架构与扩展边界

## 主工作流

```text
项目总览
  │
  ▼
数据准备
  ├── 文件登记与解析
  ├── 测井曲线清洗
  ├── 井实体合并
  ├── 井位、海拔与井轨迹对齐
  └── 地震几何重建
  │
  ▼
统一数据可视化
  └── 地震体、剖面、井轨迹、测井曲线和匹配结果均为场景图层
  │
  ▼
样本构建
  ├── 可替换井震空间对齐器
  ├── 可选时间—深度转换
  └── 多模态样本、掩码、置信度与来源
  │
  ▼
模型中心
  ├── 地震单模态Baseline
  ├── 测井编码器
  ├── 轨迹与位置编码器
  ├── 可学习井震对齐
  └── 置信度门控/可学习井震融合
  │
  ▼
预测解释 → 评估导出
```

数据检查不是独立页面，而是每个数据准备阶段的放行条件。问题必须包含严重性、来源、建议动作和是否阻断后续任务。

## Python分层

```text
src/well_seismic/
├── io/                 LAS、SEG-Y和井相关表格适配器
├── alignment/          可独立替换的井震空间对齐算法
├── workflow/           前后端共用的阶段、问题与放行结果
├── interpretation/     下游解释任务语义、输出契约与注册中心
├── modeling/           小模型协议、规格和插件注册中心
├── pipeline.py         读取、样本构建和输出的编排层
├── fusion.py           内置融合基线及可学习适配器
├── prediction.py       模型推理运行器与任务输出
├── datasets.py         JSONL、NumPy和可选PyTorch数据集适配
├── api_models.py       稳定HTTP请求/响应契约
├── platform_capabilities.py 平台能力与扩展点汇总
└── api.py              FastAPI路由、任务队列和静态前端托管
```

依赖方向：

```text
FastAPI / Vue
      │
      ▼
workflow契约 + modeling契约
      │
      ▼
pipeline编排
      │
      ├── io适配器
      ├── alignment空间对齐器
      ├── depth_time垂向转换器
      └── fusion融合算法
```

前端不直接依赖具体算法类，只读取稳定JSON契约。替换空间对齐器、融合算法或下游模型时，不需要修改数据输入和可视化页面。

下游任务清单由 `InterpretationTaskRegistry` 统一输出，前端不再维护第二份硬编码任务定义。模型通过
`ModelSpec.metadata["prediction_task"]` 绑定任务，输入适配器和运行器仍按模型 ID 独立注册。

## 稳定接口

### 空间对齐

`well_seismic.alignment`提供：

- `fit(seismic_sources)`：建立空间索引；
- `match(x, y, asset=None)`：返回地震资产、最近道、邻域道、插值权重和距离；
- `build_spatial_aligner(config)`：配置驱动工厂。

当前内置 KDTree 最近邻实现，可限定到最近三维地震体并返回配置数量的邻域道。后续增加可学习偏移、
CRS 重投影或坐标校正时，应新增实现并在工厂注册，不修改样本数据契约。

### 时间域井震标定

`well_seismic.alignment`还提供声波积分、合成地震和有界静态校正。垂向变换显式声明深度域，输出
`provided_tie`、`estimated_tie`、`vertical_initial`或`horizontal_only`状态、置信度和时间不确定度。
算法与门禁详见 [时间域井震标定方案](time_domain_well_tie.md)。

### 模型插件

`well_seismic.modeling.ModelPlugin`要求：

- `fit`
- `predict`
- `save`
- `load`
- `spec`

第三方包可通过Python入口点组`well_seismic.plugins`注册。核心工程不强制依赖PyTorch，因此传统算法、PyTorch、ONNX或其他框架可以并存。

任务语义可通过入口点组 `well_seismic.interpretation_tasks` 扩展。完整接入步骤见
[下游解释与井震融合接入指南](extension_guide.md)。

### 样本契约

模型统一读取：

- 地震窗口或三维子体；
- 标准测井曲线与缺失掩码；
- MD、TVD、TVDSS和XYZ；
- Inline、Crossline、地震道号及空间距离；
- 水平、垂向置信度；
- 井位、轨迹、LAS、SEG-Y及配置版本来源。

没有实测时间—深度关系时，系统可生成带来源和不确定度的声波积分候选，但默认不赋予训练资格；
只有水平距离、垂向质量、固定窗口和 CRS/单位核验全部通过时，样本才标记为可训练。

## 原始数据保护

- 原始SEG-Y、LAS和井数据只读；
- 大型SEG-Y不复制到项目目录；
- 可视化使用按需读取或降采样数据；
- 清洗、人工校正、样本和预测均写入新输出；
- 推断值不能覆盖用户明确配置；
- 低置信度井相关文件只进入待确认结果。
