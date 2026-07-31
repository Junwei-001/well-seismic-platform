# CIGVis 可视化集成

## 定位与数据边界

平台使用本地仓库 `接口模型/cigvis-main/cigvis-main` 中的
[CIGVis](https://github.com/JintaoLee-Roger/cigvis) 0.3.1，并在该本地副本中增加了
Viser 中文控件标签。CIGVis 仍只属于展示层：通用数据准备负责读取、几何检查和生成模型无关快照；
FaultSeg 适配器负责模型输入；可视化不得反向改变 SEG-Y 标准化或模型输入。

```text
原始 SEG-Y
  ├─ 数据准备 → 模型无关预览
  │    ├─ 三维预览 [Z, Inline, Crossline] → WebGL2 整体体绘制 / CIGVis Viser 切片
  │    └─ 二维预览 [Time, Trace]          → CIGVis Matplotlib
  └─ FaultSeg 输入适配与推理
       └─ 原始裁剪范围 + probability.npy/mask.npy
            → 有界 SEG-Y 背景裁剪
            → 同采样概率体
            → CIGVis create_slices + add_mask
```

因此，打开可视化不会改变原始数据，运行 FaultSeg 也不会覆盖通用预览。预测任务只在展示阶段把
其结果作为独立图层接回父数据任务。

## 浅色中文工作台

三维和二维场景统一放在浅色可视化工作台内。地震数据集以左侧常驻栏呈现，用户可以直接在二维
测线、三维体和预测结果之间切换，不再把一整段文件清单堆放在工作区顶部，也不需要先展开折叠面板。

- 三维数据默认进入“整体 3D”模式，使用浏览器原生 WebGL2 体绘制显示完整诊断体；顶部提供显眼的
  “整体 3D 默认 / 切片查看 CIGVis”切换按钮，切片模式仍由本地 CIGVis Viser 渲染。
- 整体 3D 背景地震默认使用 `seismic`，FaultSeg 及后续预测结果统一使用 `jet` 叠加，避免背景振幅与
  解释结果共用色标而难以辨认。
- 存在预测图层时，顶部显示“地震 + 预测 / 仅预测概率体”切换。该状态同时作用于整体 WebGL2
  和 CIGVis Viser 切片；“仅预测概率体”会真正关闭地震背景采样，而不是换成白色或降低背景对比度。
- 整体 3D 相机默认对准体数据中心，只允许拖动旋转；禁用滚轮缩放和平移，避免放大、缩小或拖动后
  体数据偏离窗口。CIGVis 切片模式同样锁定观察中心与相机距离，保留环绕旋转而抑制漂移。
- 顶部工具栏提供“适应窗口”“全屏”和“独立窗口”等常用入口。
- Viser 常用控件使用中文：切片位置、显示参数、振幅范围、色标、叠加层透明度、比例、截图和状态。
- CIGVis Viser 默认关闭切片边界与交线描边，避免 `seismic` 色标下出现蓝色外框干扰同相轴；
  切片位置仍由底部滑块和高级控件明确显示。
- 三维场景底部提供 `Inline`、`Crossline`、`时间 / 深度` 三条常驻滑块。拖动时显示真实坐标值，
  并经 `POST /api/v1/visualization/viser-slices` 更新服务端 CIGVis 切片句柄。
- 滑块输入采用短延迟合并，连续拖动不会为每个像素位置重复提交请求。Viser 自带的高级控制仍
  保留在场景内，方便精细调整振幅、色标和截图区域。
- Plotly 回退和二维剖面遵循相同的 `seismic` 背景、`jet` 预测叠加约定，背景振幅按对称范围显示。

测井工作台采用窄控制栏与窄长曲线画布，让纵向深度保留足够浏览空间；曲线、网格、刻度和坐标轴
使用专业测井配色与线型。预测解释的任务入口只保留在左侧导航的二级菜单中，任务首页不再重复
渲染同一组标签。

底部滑块移动的是当前已加载的有界场景，不会在每次拖动时重新读取整个 SEG-Y。

## FaultSeg 推理结果叠加

完成 `faultseg_3d` 推理后，可视化接口读取推理元数据中的原始 SEG-Y 路径、
`crop_start_zyx`、`crop_size_zyx`、源体尺寸以及 `probability.npy` / `mask.npy`。系统先按同一裁剪范围
构造地震背景，再用完全相同的采样索引从内存映射的概率体中取值，最后执行：

```python
background_nodes = viserplot.create_slices(
    background_il_xl_z,
    cmap="seismic",
)
viserplot.add_mask(
    background_nodes,
    probability_il_xl_z,
    clim=[threshold, 1.0],
    cmap="jet",
    alpha=0.62,
    excpt="min",
)
```

默认显示连续概率体，而不是只显示二值结果；小于推理阈值的最低端通过 `excpt="min"` 隐藏，
以便同时看清背景同相轴。二值 `mask.npy` 仍保留在结果契约中，供后续硬分割展示或导出使用。

切换到“仅预测概率体”时，整体三维把地震背景透明度置零，并适当提高概率体可见度；CIGVis
切片节点只合成 `add_mask` 图层，低于当前阈值的位置保持透明。切回“地震 + 预测”后恢复用户
当前选择的地震色标、概率阈值和切片位置，不重新读取 SEG-Y，也不修改概率数组。

在进入 `add_mask` 前会强制检查：

- 模型标识必须为 `faultseg_3d`，轴序必须为 `[Z, Inline, Crossline]`；
- 推理输入、概率体、二值体和裁剪范围的三维形状必须一致；
- 概率值必须位于 `[0, 1]`，二值体只能包含 `0/1`；
- 背景和概率层经过相同稀疏索引后形状必须完全相同。

任一条件不满足时拒绝叠加并返回明确错误，不对数组做猜测性转置或拉伸。

## 轴顺序契约

平台和 FaultSeg 结果统一保存为：

```text
[Z, Inline, Crossline]
```

CIGVis Viser / Plotly 的 line-first 输入为：

```text
[Inline, Crossline, Z]
```

背景体与每个叠加体都必须执行同一个转置：

```python
cigvis_volume = platform_volume_zyx.transpose(1, 2, 0)
```

对应关系固定为 `x = Inline`、`y = Crossline`、`z = 时间或深度样点`。井轨迹同样映射为
`[inline_index, crossline_index, sample_index]`。没有可信时深关系时，垂向只能标为“时间 / 深度”或
比例预览，不得把 TVD 直接解释为双程时间。CIGVis 的显示方向翻转只影响视角，不改变数组存储契约。

二维 SEG-Y 使用 `[Time, Trace]` 剖面路径，不要求数据形成完整 Inline/Crossline 网格，也不参与
FaultSeg 三维概率叠加。

## 大体积 SEG-Y 的有界按需缓存

FaultSeg 结果可视化使用进程内、线程安全的 `SegySliceCache`，不会一次性读取完整三维 SEG-Y：

1. 根据本次推理的 `crop_start_zyx + crop_size_zyx` 确定唯一空间范围；
2. 默认把可视化采样限制到最多 `[128, 96, 96]`，各轴按整数步长稀疏取样；
3. 仅查找所需 Inline/Crossline 组合并读取所需 Z 样点；
4. 用有限振幅的第 99 百分位做对称归一化，背景压缩为 `int8`；
5. 以同一组 Z/Inline/Crossline 索引读取内存映射的概率体和二值体。

默认缓存边界为：

| 项目 | 默认上限 |
|---|---:|
| LRU 条目数 | 8 |
| 缓存数组总量 | 64 MiB |
| 单次采样体素数 | 2,000,000 |
| 默认最大采样形状 `[Z, Inline, Crossline]` | `[128, 96, 96]` |

缓存键包含规范化文件路径、修改时间、文件大小、裁剪起点与尺寸、最大采样形状以及 SEG-Y 读取配置
签名。源文件或读取配置发生变化会自然产生新键；达到条目数或字节上限时按 LRU 淘汰。单个结果若
超过缓存字节限制仍可返回本次展示，但不会写入缓存。缓存仅驻留在当前 Python 进程，重启后清空。

这是一套面向交互检查的有界稀疏缓存，不是全分辨率解释体存储：它保证背景与 FaultSeg 结果同轴，
但不能替代原始 SEG-Y 或全分辨率预测文件。后续若需要任意区域连续漫游，应在此契约之上增加磁盘
分块金字塔或切片服务，而不是提高浏览器内单场景体积。

## 当前限制

- FaultSeg 叠加只支持能够解析出规则 Inline/Crossline 几何的三维 SEG-Y，不适用于二维测线。
- 当前按需读取范围来自一次推理的裁剪元数据，尚不支持在完整大体积上任意框选新区块连续漫游。
- 浏览器看到的是有界稀疏诊断体；定量解释和成果交付必须回到原分辨率 SEG-Y、概率体与二值体。
- 一个后端进程维护一个活动 Viser 场景，符合本机单用户使用方式；多用户部署需要会话级场景隔离。
- 没有可靠时深关系时，井轨迹垂向仅为比例预览，不能用于时间域定量校正。

## 后端和服务生命周期

- 三维默认由页面内 WebGL2 体绘制器显示整体诊断体；切换到切片模式后使用
  `cigvis.viserplot`，支持动态切片、振幅范围、色标、比例和截图。
- Viser 启动失败时自动回退到 `cigvis.plotlyplot`，保留正交切片和 FaultSeg 叠加，并关闭滚轮缩放。
- 二维测线使用 CIGVis Matplotlib 渲染，不依赖 Viser 服务。
- FastAPI 监听 `127.0.0.1:8000`；Viser 默认从 `8080` 启动独立 HTTP/WebSocket 服务。
- 切换场景时复用服务并重建节点；进程退出或回退时停止 Viser，避免残留端口与重复回调。

当前部署面向本机单用户。若通过 HTTPS 或远程服务器发布，应将 Viser HTTP 和 WebSocket 放到同一
反向代理域名下，避免浏览器混合内容、跨端口策略和防火墙问题。

## 安装与核验

`安装前后端依赖.bat` 会执行本地可编辑安装：

```powershell
python -m pip install -e ".\接口模型\cigvis-main\cigvis-main[viser]" --config-settings editable_mode=compat
```

启动脚本同时检查 `cigvis`、`viser`、`plotly` 和 `trimesh`。运行状态可从
`GET /api/v1/capabilities` 的 `visualization.engine` 查看；正式页面统一从
`GET /统一数据可视化` 进入。

排查错位时应依次核对 `axes`、`sourceShapeZYX`、`cropStartZYX`、`cropSizeZYX`、
`sampleIndicesZYX` 和最终背景/概率数组形状，不应先在前端通过 CSS 或图形缩放掩盖数据问题。
