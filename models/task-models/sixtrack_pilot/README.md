# 六赛道试点扩展包

把「师弟-军伟的比赛」项目里的模型按平台官方 entry point 协议接入。当前分支
仅在扩展包中声明 P21 和断层轻量模型两个入口；代码仓库默认保持接口模式，
不附带载荷，也不默认注册为可运行模型。
下表记录上游参考分支规划的 9 个 `model_id`、六个赛道，并不表示本分支已取得全部载荷；
缺少载荷的模型必须在发现阶段拒绝注册。运行范围以各模型卡和输入协议为准；
登记及载荷核验不能代替新区域工作流、科学质量、账号授权、许可或统一发布验收。
下文分阶段记录中的模型数量和“尚未登记”描述只代表当时状态。
参考分支的“逐位复现”是其原环境结论；本轮 Windows CPU 复算五折均完成，
fold 4 与随包历史数组的最大绝对差约 `1.94e-16`，不宣称本环境逐位相同。

| model_id | 任务 | 来源结论 | 定位 |
| --- | --- | --- | --- |
| `sixtrack_fault_local_logistic_audited_v2` | `fault` | P78 `VERIFIED_RETAIN_INCUMBENT`（没有新冠军，保留原模型） | **轻量基线 / 仅开发集验证** |
| `sixtrack_property_tabicl_phif` | `well_property` | P67 `VERIFIED_STABLE_CHAMPION`：冻结 TabICL | 开发集稳定冠军，档案绑定 |
| `sixtrack_property_tabicl_klogh` | `well_property` | 同上（KLOGH，单位 mD） | 同上 |
| `sixtrack_property_tabicl_sw` | `well_property` | 同上 | 同上 |
| `sixtrack_reconstruction_p21` | `volume_reconstruction` | 原始五折 10,240 个预测逐位复现 | 条件重建·仅开发档案 |
| `sixtrack_lithofacies_p76` | `lithofacies_classification` | P76 固定 TabICLv2 概率集成 | 原始三折 compact70 开发查询 |
| `sixtrack_sweetspot_p7_t3` | `sweetspot_productivity` | P7 Chronos2 与历史均值固定融合 | T3 原始四折观测索引预测 |
| `sixtrack_facies_f3_p81` | `seismic_facies_slices` | P81 固定协议恢复的稳定方法 | 原开发折二维切片 CPU 复现 |
| `sixtrack_facies_penobscot_p81_reference` | `seismic_facies_slices` | P81 保留操作参考 | 独立标签空间，原开发折二维切片 CPU 复现 |

## 提交渠道：本地试点面板或 API

上表是完整参考分支的九模型范围；本分支扩展包只暴露 P21 与断层轻量模型。
其余模型的源码保留供后续恢复，但缺载荷时不能作为可运行模型登记。
参考分支的本地试点面板曾接入六赛道受限协议，不能把该记录当成本分支的验收。
扩展源码更新后还需重新安装扩展包，才能刷新解释器中的 entry point 元数据。
`models/manifest.json` 与载荷齐全不能代替这个安装步骤；实际服务的可运行模型还受安装版本和部署配置约束。
fault 需要 `seismic_path` 加显式体块，物性需要 `options.query_family`，P21 需要 `options.fold_id`（0..4），
地震相需要 `options.fold_id`（0..4）及 `options.repeat_id`（0..2）。F3 与 Penobscot 不互换模型或标签。
原快照绑定工作台
会带着 `source_task_id` 提交，扩展不接受这一提交方式；未知快照可先被共享 API 以 409 拒绝，
进入模型校验器的快照请求以 422 拒绝。试点面板不等于授权新井工作流或正式发布。

## 放什么、不放什么

- 放：插件代码（`src/wellseismic_sixtrack_pilot/`）、载荷复制配方（`tools/materialize_payloads.py`）、
  模型运行时依赖清单（`requirements-model-runtime.txt`）。
- 同级的 `../<model_id>/<version>/manifest.json` 记录每个载荷的 SHA-256、比赛项目中的来源路径与
  commit、verifier 结论、许可、输入轴序、适用域和 claim boundary。
- 不内置研究训练源代码、训练数据或 SwinUNETR 重训。权重恢复工具只调用哈希锁定的原研究
  训练函数，输出到研究仓库之外，不做新算法、新搜索或冠军替换。载荷二进制
  （`payload/`）不进 git，由配方按哈希复制生成。

## 入口与各文件角色

- `plugin.py`：`well_seismic.plugins`（注册 `ModelSpec`，先校验载荷哈希和模型解释器）与
  `well_seismic.input_adapters`（提交时校验请求，非法请求由平台返回结构化 422）。
- `runner.py`：`well_seismic.prediction_runners`。经平台的 `managed_run` 启动隔离子进程，
  设 `CUDA_VISIBLE_DEVICES=""` 强制 CPU，并禁用 user site 和 HF 联网。
- `worker/fault_logistic.py`、`worker/property_tabicl.py`：只在模型解释器
  （`WELL_SEISMIC_SIXTRACK_PYTHON`）里运行，逐字复用比赛项目的特征、标准化和物理域变换。
- `catalog.py`：model_id、版本和请求校验，提交时与运行时各校验一次。
- `manifest.py`：载荷清单读取与 SHA-256 校验，只用标准库，平台进程与 worker 共用。

## 模型卡

### 断层：`sixtrack_fault_local_logistic_audited_v2`（轻量基线）

- **模型：** 逐 inline 切片，取 5 个局部特征（振幅、3×3 均值、3×3 标准差、两个 Sobel 分量），
  用 SGD logistic 输出断层概率。载荷是 portable 系数 JSON，不反序列化 joblib。
- **输入：** 一个叠后 3D SEG-Y，按 INLINE_3D/CROSSLINE_3D 道头寻址，原始 float32 振幅，
  不做归一化，轴序 `[Z, INLINE, CROSSLINE]`。这与 P78 参考做法（`predict_baseline_volume`）完全一致。
  - **必须**显式给出体块：`options.inline_range/crossline_range/sample_range`（闭区间），
    或 `crop_start/crop_size`。
  - 每轴最小 3 个样点，最多 134,217,728 体素。ST10010 全区约 2.45 亿体素，需要分块提交。
- **输出：**
  - `fault_probability.npy`
  - `fault_mask.npy`：阈值默认 0.4505760073661804，来自 validation_max_f1
  - `grid.json`
  - 平台另外生成 standard_result_manifest 和切片包。
- **质量（照实写）：**
  - audited_v2 在测试集上 precision≈0.008、F1≈0.016，掩码正类比例偏高，不能当生产断层解释用。
  - 训练与推理尺度不一致：系数在 z-score 后的 ST0202R08 inline patch 上训练（mean −3.03e-6，
    std 0.0279），而 P78 参考和本包都直接作用于 ST10010 原始振幅。0.4506 阈值是在 z-score 后的
    ST0202 验证集上选的，所以 P30 盒内约 79% 体素被判为断层。本包保持与 P78 参考逐位一致，
    没有自行加 z-score，因为那会偏离已验证的 operational reference；要不要改由用户决定。
- **平价：** P30 开发盒对 P78 参考概率的 max_abs_diff = 5.96e-8。
- **如何变强：** 神经断层模型（SwinUNETR）在 P78 排名第一，但没过阈值稳定门，权重也未保存。
  需要作为独立训练任务立项，本包不做。

### 物性：`sixtrack_property_tabicl_{phif,klogh,sw}`（P67 稳定 development 冠军）

- **模型：** 冻结的 TabICL v2 regressor（in-context，不训练权重，`n_estimators=2`），
  每个目标一个估计器，`random_state = seed + 目标序号`。
- **输入（档案绑定 dataset-bound）：**
  - `options.query_family`：取 15/9-19、15/9-F-1、15/9-F-11、15/9-F-12 之一。
  - 平台选出以该井族为验证组的 LOGO4 折。上下文用其余三个开发井族的 192 行，标准化只在上下文上拟合。
  - 预测该井族的 81 行，可再用 `options.query_well_ids` 缩小范围。
  - 15/9-F-15 冻结测试井永不进入。
  - 任何原始 SEG-Y、LAS、轨迹路径都会被拒绝。
- **特征（153 维）：** 81 个标准化的 ST0202 地震值（3×3×9 patch），36 个标准化的
  GR/RT/NPHI/RHOB 值（9×4，按掩码置零），加 36 个观测掩码。
- **输出：** `<target>_predictions.csv/.npz`，含物理单位预测、模型域原值、CPI 参考标签与误差指标。
  - KLOGH 在 log1p 域预测，再 expm1 还原。
  - PHIF/SW 裁剪到 [0,1]。
- **质量：**
  - P67 做了 LOGO4 × 3 seeds，composite 0.381249，是已确认各模型中最优；各折 KLOGH R² 为 0.27–0.67。
  - 参考标签是 CPI 解释值，不是岩心。
- **平价（fold0，seed 2693，CPU 对 P67 的 GPU/AMP）：** 模型域 max_abs_diff 如下，
  三个目标对 P67 OOF 的 R² 均 ≥ 0.9999。
  - PHIF：2.6e-4
  - KLOGH：2.1e-2
  - SW：7.1e-3

### 进阶能力：全量特征构建（当前默认不启用）

目前只能预测已封装开发档案里的井。**全量特征构建**是一项可选的进阶能力：它从原始 ST0202
SEG-Y + LAS（GR/RT/NPHI/RHOB）+ 井轨迹/时深关系现场生成 153 维特征，让模型能预测档案之外的新井。

- **现状：** 本轮没有实现，默认不启用。
- **何时告知：** 用户在对话里问"如何变强 / 提升精度 / 预测新井"时，应告知这个选项存在。
  它需要作为独立后续任务立项，包括移植井震标定与特征构建，并对新井重新做平价验证。
- **可发现位置：**
  - `GET /api/v1/capabilities` → `models[].metadata.advanced_options[id=full_feature_building]`
  - 各物性 `manifest.json` 的 `advanced_options`

## 安装与复现

```bash
# 平台 .venv 内以 editable 安装（entry point 生效）
uv pip install --python .venv/bin/python -e models/task-models/sixtrack_pilot
# 模型解释器（隔离 venv）
uv venv --python 3.11 ~/envs/well-seismic-models-py311
uv pip install --python ~/envs/well-seismic-models-py311/bin/python -r models/task-models/sixtrack_pilot/requirements-model-runtime.txt
# 按 manifest 复制并校验载荷（源文件只读）
.venv/bin/python models/task-models/sixtrack_pilot/tools/materialize_payloads.py \
    --competition-root <比赛项目根> --tabicl-checkpoint <HF 快照中的 tabicl-regressor-v2-20260212.ckpt>
```

`scripts/linux/bootstrap.sh` 会把以上三步串起来执行。改动原因请查 `git log -- models/task-models/`。

## 三维重建 P21（开发档案绑定运行）

`worker/reconstruction_p21.py` 移植固定三核的数值推理：坐标 Z 权重 4、GFM/PCA 特征权重 0.1、
地震权重 0/0.1/0.2、64 邻点、距离幂 1.5，各成员与原 PyKrige 基线按 0.75 混合后等权平均。
它只消费外部已准备好的训练/查询坐标、地震、16 维 latent、512 个训练标签和查询位置的
PyKrige 基线；严格拒绝查询标签、未知字段、非有限值和形状错误，不裁剪或修改科学参数。

模型解释器中可运行 `python -m wellseismic_sixtrack_pilot.worker.reconstruction_p21
--input <prepared.npz> --input-sha256 <sha256> --output-directory <新目录>`（同一行执行）。
输入 NPZ 的八个字段定义见模块 `INPUT_KEYS`；输出为三核成员与平均预测，以及输入/输出哈希回执。
已有结果目录不会覆盖。命令不访问 GPU，也不下载权重。

6 项测试包括对研究仓原始核函数的独立数值对照、查询标签隔离、输入约束及真实 CLI 哈希/不可覆盖
检查。另用 `tools/replay_reconstruction_p21.py` 从原始 Eclipse GRID、既有五折开发缓存和预训练
GFM 特征恢复真实 P21 输入：全部训练/查询索引哈希匹配，10,240 个预测与原归档逐位一致。
随后五折独立 CLI 运行也一致。该恢复不重建缺失的 `train.h5`，不读原始 INIT/PORO 标签；
查询参考值只用于推理后的误差审计。**这仍是开发集比对，不是新区域质量验收。**
现已增加载荷清单、解释任务、输入适配器和运行器 entry point，模型 ID 为
`sixtrack_reconstruction_p21`。本地登记在新安装的扩展环境中生效；正式服务尚未安装此版本。
新区域输入、完整预处理、授权 UI/API、许可及发布继续开放。

恢复命令为 `python tools/replay_reconstruction_p21.py --competition-root <研究仓> --stage3-root
<五折缓存目录> --feature-cache <原GFM缓存.npz> --eclipse-zip <原Eclipse.zip> --output-directory
<新目录>`（同一行执行）。它核对冻结研究摘要/预测、五折缓存、坐标索引和特征缓存哈希，
生成不含查询标签的五份 prepared NPZ 及 parity 回执。路径全部显式传入，不依赖旧工作树名。
[真实开发集恢复证据](../../../docs/evidence/sixtrack-reconstruction-p21-original-parity-20261002.json)记录
GRID、缓存、运行代码和载荷哈希；生成载荷保留在 `.cache/`，只追踪配方与身份。

`runner.run_reconstruction_prepared(ModelInputRequest(prepared_path,
options={"input_sha256": pinned_sha256}), output_directory=new_directory)` 已接通平台实际
`managed_run` 子进程接口。平台进程只做请求/哈希检查；数值推理仍在指定模型解释器运行。
该调用严格拒绝 GPU、体块字段和未知选项，前后核对输入/输出哈希；复用结果目录会失败。
`worker/reconstruction_request.py` 采用现有 `--request/--result` 协议，非法输入写结构化失败，
任务取消由平台运行时终止真实进程，取消异常不会变成普通失败或完成回执。

[平台运行接口证据](../../../docs/evidence/sixtrack-reconstruction-p21-managed-worker-20261002.json)
记录五折 10,240 个预测与原始复现逐位一致；8 项平台解释器测试还覆盖输入篡改、查询标签
拒绝和已启动进程的实际取消。数值解释器的原有 6 项测试继续通过。低层 prepared 入口仍供
数值审计；公开登记入口只允许选择五份按哈希封装的原始开发折。

P21 载荷先用上述恢复配方生成，再运行 `tools/materialize_payloads.py` 时附加
`--reconstruction-prepared-root <prepared目录>`。已有载荷复验通过时不会重新复制；源哈希不符
会失败。`development_folds.json` 记录原始行序、训练/查询索引哈希，五份无查询标签 NPZ
保留在忽略的 `payload/` 目录。提交示例为
`{"task_id":"volume_reconstruction","model_id":"sixtrack_reconstruction_p21","options":{"fold_id":0},"device":"cpu"}`。

[登记与实际 API 证据](../../../docs/evidence/sixtrack-reconstruction-p21-registration-api-20261002.json)
来自临时平台解释器：插件发现、五折真实 POST/查询/封存下载与逐位比对，以及非法请求的
结构化 422 检查通过；正式解释器和服务没有更新。当前三维重建未进入体积预测菜单，
授权网页流程、原生 GAIA 提交、新工区预处理和统一发布仍需继续实现。

### 从特征行现场准备 P21 输入

`worker/reconstruction_prepare.py` 移植了 P18 的训练端标准化、16 维 PCA、latent/坐标/地震
尺度变换，并用研究锁定的 PyKrige 1.7.3 现场拟合和预测基线。它接收七个 NPZ 字段：
`train_raw_features/query_raw_features`（三列地震属性 + 三列原始归一化几何），
`train_foundation/query_foundation`（同一预训练编码器的对齐特征），`train_target`（512 个训练
支撑标签），以及 `train_kriging_coordinate/query_kriging_coordinate`（原始训练折预处理后的
三列坐标）。查询标签和缓存基线字段均不接受。所有输入必须有明确来源并绑定 SHA-256。

PyKrige 使用的坐标尺度与 P21 核度量不同，调用方必须提供原始训练折的坐标变换，不能用
512 个支撑点的标准差代替。变差函数固定 `linear`、`nlags=4`。P5 Stage-3 将基线以 float32
归档，P21 再按 float64 读取；新实现保留这个精度步骤。PyKrige 方差单独保存，不能当作
整个 P21 集成预测的不确定性。

恢复配方加 `--include-feature-inputs` 可导出不含查询标签的五折特征输入。随后在具有上述
依赖的模型解释器中执行 `python -m wellseismic_sixtrack_pilot.worker.reconstruction_prepare
--input <feature_input.npz> --input-sha256 <sha256> --output-directory <新目录>`（同一行），
生成 `prepared.npz`、`pykrige_variance.npy` 和回执，再交给现有 P21 推理命令。

[现场准备证据](../../../docs/evidence/sixtrack-reconstruction-p21-live-preparation-20261002.json)
记录五折完整 CLI 链路：所有准备数组和 10,240 个最终预测逐位一致，12 项测试通过，包括
查询特征不能影响训练端尺度/PCA、查询标签拒绝、输入哈希与不可覆盖。此步骤仍从已对齐
特征行开始；原始地震/几何读取、GFM 特征提取、授权准备任务和新工区验证尚未接入。
已登记的开发档案模型继续使用固定载荷，正式模型解释器没有更新。

### 从原始 SEG-Y 恢复开发区域属性体

`tools/recover_reconstruction_seismic.py` 使用原始 Eclipse GRID 几何、地震索引和弱井震
时深表，读取原始 ST0202 SEG-Y 并输出 `[3,63,100,72]` 属性体、活动掩码和归一化坐标。
它不调用读取 INIT/PORO 的数据构建入口、不重建 `train.h5`，也不读取查询或冻结标签。
地震读取已移入 `worker/reconstruction_seismic.py`：保留原始最近时间样点、五样点 RMS
和中心垂向差分，原有上边界裁剪策略保留；首样点没有前驱时明确拒绝，避免负索引绕到末尾。

运行参数全部显式提供：`--competition-root`、`--eclipse-zip`、`--segy`、`--seismic-index`、
`--well-tie` 和 `--output-directory`；附加 `--feature-input-root <上述五折特征输入目录>`
可核对原始训练/查询点。输出目录必须全新，源文件只读，GRID/解析代码与输入输出身份以
SHA-256 记录；属性体和活动掩码可供后续 GFM 剖面构建使用。

[原始属性体恢复证据](../../../docs/evidence/sixtrack-reconstruction-p21-raw-seismic-20261002.json)
记录 134,222 个活动网格单元，以及五折训练/查询 12,800 行三项属性逐位一致。9 项测试
覆盖真实 SEG-Y、原始读取器独立对照、时间/几何错误和首样点拒绝。原弱对齐把 MD 与
TVD-like 网格深度混用，限制明确保留；现场 GFM 验证见下节，新工区及授权网页仍未验收。

### 本地预训练 GFM 现场 CPU 编码

`worker/reconstruction_gfm.py` 从上述无查询标签的属性体提取特征。它先核对本地 GFM
源码、配置、模型卡和预训练权重的固定 SHA-256，再加载冻结编码器；不下载、不训练。
三个地震通道分别进行活动区归一化、剖面缩放和无掩码编码，保留原研究的 trace/CLS
token、三组固定种子投影和每点 288 维特征。依赖列在 `requirements-gfm-runtime.txt`。

CLI 参数为 `--volume`、`--volume-sha256`、`--indices`、`--indices-sha256`、
`--source-root`、`--snapshot`、`--output-directory`。输入索引为独立整数 N×3 KJI NPY，
体积 NPZ 只接受 seismic、active、normalized_coordinates 三项。输出目录必须全新。
当前入口属于离线开发验证，尚未接入授权网页准备任务或更新正式模型解释器。

[现场编码证据](../../../docs/evidence/sixtrack-reconstruction-p21-gfm-cpu-20261002.json)
记录两个真实剖面、1,318 点与原研究 CPU 编码器逐位一致；全部 58 个剖面、12,583 点完成
预训练编码，并现场完成训练端变换、PyKrige 和五折 P21 推理。10,240 点的 CPU RMSE 为
0.0277339306，历史 GPU/bfloat16 RMSE 为 0.0277343744；逐点预测最大差值 0.000740469。
这是接近历史结果的开发区域验证，不能称为历史 GPU 特征逐位复现，也不能据此推广新工区。
查询标签仅在全部推理结束后用于误差评估，未进入编码或准备输入。现有登记载荷保持不变。

### 可取消的现场准备与预测任务

内部 `reconstruction_live.run_live_reconstruction` 将体积读取、冻结 GFM、训练端尺度与
PCA、现场 PyKrige 和 P21 串入同一隔离 CPU 子进程，沿用平台 `managed_run` 的任务取消
机制。调用方必须先核对源文件的项目授权；此函数尚未登记为公共 API 或网页入口。

输入是带 SHA-256 的体积 NPZ 与支撑 NPZ，以及固定源码/权重目录。支撑 NPZ 只允许
`train_indices_kji`、`query_indices_kji`、`train_target`、`train_kriging_coordinate`、
`query_kriging_coordinate`。训练为 512 个互异活动单元，查询为 1..2048 个互异活动单元，
二者不得重叠；不接受查询标签、缓存特征或缓存基线。原始六列特征直接从体积按索引读取。
Kriging 坐标仍须是显式原始训练折坐标，现场工作流不会猜补缺失的科学变换。

每次要求全新输出目录，回执记录输入、预训练资产、准备和预测的哈希与使用边界；输出含
准备数组、PyKrige 方差和 P21 预测。中断时不生成成功回执。13 项测试覆盖禁止字段、地址
错误、重叠与重复、源文件篡改，以及平台登记的真实子进程取消；取消测试只证明进程生命周期。
真实开发折链路的逐点对照见
[现场任务证据](../../../docs/evidence/sixtrack-reconstruction-p21-managed-live-20261002.json)。
原开发折 2,048 点完整运行成功，与上一轮 CPU 链路最大差值为 8.33e-17；GFM/PCA 与
PyKrige 输出逐位一致，原始地震/坐标尺度归约存在 float64 舍入差异，不声称全链路逐位一致。
新工区验证、产品授权接线和同版发布仍须独立验收。

模型中心的本地试点页面现可选择 P21 原始开发折 0..4，提交既有档案模型任务；页面按
实际装载目录显示赛道数，不把后端未加载的模型算作可用。该表单不接受新区域输入，亦未
接入上面的现场 GFM 准备入口。
[浏览器证据](../../../docs/evidence/sixtrack-reconstruction-p21-browser-20261002.json)
记录私有服务开发折 4 的 2,048 点真实推理、点击下载逐位匹配、刷新恢复和切换模型后保留
原任务身份。5 项前端合同检查和类型检查、验证构建通过；正式 dist 未替换。内嵌通用
查看器在该轮展示回执 JSON；多用户项目授权未在此入口验收。查询点预览的后续实现见下节。

### 三维重建查询点预览

P21 数值 worker 额外导出 `p21_query_points.npz`，只有标准化查询坐标与预测值两项；
原预测 NPZ 和预测数值保持不变。标准结果把它作为默认点图层，在离线 Canvas 中显示
彩色查询点，支持拖动旋转、三种平面投影和悬停查值；开发折查看器增加高度以容纳完整图。
图上坐标无量纲，不代表地图坐标或米制深度。只显示查询点，不插值为完整三维地质体。

[查询点预览证据](../../../docs/evidence/sixtrack-reconstruction-p21-query-points-20261002.json)
记录五折真实 API、新增 NPZ 的封存与默认查看器选择，以及私有浏览器 2,048 点显示、
旋转、投影、悬停和刷新恢复。查询坐标与预测分别逐位匹配原准备档案与预测档案；悬停
第 228 行的坐标/预测值另行核对。24 项后端检查和前端类型检查、验证构建通过。读取器
在加载前检查解压后成员尺寸、NPY 头的形状和类型，拒绝标签、超量点及非有限数据。
这项预览不证明新工区有效、完整体重建、原生 ResInsight 图层接入或多用户授权发布。

### 岩相 P76 开发冠军的可移植 CPU 推理

`worker/lithofacies_p76.py` 实现固定 TabICLv2 两路概率集成：e2 权重 0.25、e8 权重 0.75，
沿用原开发折 0..2、确认种子 3..5 和各路派生种子。九类概率按原固定类别代码 0..8 对齐，
不把缺失训练类别误当成不存在的输出类别。checkpoint SHA-256 固定为
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`，TabICL 为 2.1.1，
CPU 单线程；禁止下载，不执行下游优化器步骤。研究 runtime 默认仍是 XGBoost，最终
refit 未完成；本节记录可移植内核验证，后续本地目录登记见文末，研究侧默认保持不变。

输入 NPZ 仅允许 `train_well/train_seismic/train_labels/query_well/query_seismic`，其中
井窗口为 `[N,26,33]`，包含原档案的 13 个值通道与 13 个可见掩码通道；地震窗口为
`[N,3,3,33]`。35 个物理通道分别保留中心与均值，形成 70 维特征，不跨通道池化。
窗口已由原训练折预处理，不能把未对齐的新井数组当作该输入。训练/查询各至多 360 行，
查询标签不接受。输出含九类融合 logits、两路诊断 logits 和预测代码，不猜补岩相名称。

恢复命令为 `tools/recover_lithofacies_p76.py --competition-root <研究根> --output-directory
<新目录>`（同一行）。恢复配方先核对固定源档案 hash 与开发/guard/frozen-test 防火墙，
只读取训练标签和训练/查询窗口；导出三个开发折与身份回执。推理命令为
`python -m wellseismic_sixtrack_pilot.worker.lithofacies_p76 --input <fold_input.npz>
--input-sha256 <sha256> --checkpoint <固定本地权重> --fold 0 --repeat 3 --output-directory
<新目录>`（同一行）。独立 CLI 不受平台任务取消控制；内部可取消入口见下节。

[P76 CPU 证据](../../../docs/evidence/sixtrack-lithofacies-p76-portable-cpu-20261002.json)
记录九个完整单元、共 1,080 条开发查询预测，16 项合同检查及实际恢复→CLI 链路。
同环境原研究 CPU 对照的一次完整折复核逐位一致，另一初始对照曾出现 e2 最大 7.15e-7
差值，因此不保证逐位可重复。相比历史 GPU，3 个样本类别发生变化；CPU 平均九类
Macro-F1 0.1851737231，历史 GPU 0.1842126703。标签仅在所有推理结束后的评分中读取。
原研究归档会把分量 logits 减去行最大值，本入口保留其原始归一化 log 概率；对照证据
对两种诊断格式另行统一归一化，融合公式不变。这些观察不构成晋级、新工区验证或发布。

### 岩相 P76 隔离任务协议

内部 `lithofacies_runner.run_lithofacies_prepared` 接收平台 `ModelInputRequest`，显式传入
已授权的 checkpoint 路径，并用平台 `managed_run` 运行独立模型解释器。options 只能
包含 `input_sha256`、`fold_id`、`repeat_id`，拒绝 GPU、体积 crop、探索种子及额外标签。
源码入口为 `worker/lithofacies_request.py`；它复查输入与固定权重 hash，只在成功推理后
保存预测 NPZ、中文九类概率 CSV 和回执。调用方仍需接入项目授权；本节记录内部协议，后续本地 API 登记见文末。

原始输入、预训练权重、每项输出与推理代码身份进入结果 provenance；CSV 的代码及九类
概率与 NPZ 逐值对应。所有结果目录必须全新，旧结果不得覆盖。`worker_stage.json`
只是最近一次组件进度记录，不能据此认定进程仍活着；任务状态与取消结论以平台运行时
登记和进程退出为准。取消不会生成成功回执。

[隔离任务证据](../../../docs/evidence/sixtrack-lithofacies-p76-managed-worker-20261002.json)
记录三个完整开发折真实执行，以及在 e2 预训练模型实际推理完成后、确认进程存活时取消
剩余计算的实测。进程终止、平台确认静止、未生成成功回执；25 项定向测试通过，包括
结构化错误、禁止字段、旧目录保护、中文概率表与 NPZ 一致。该轮尚未接入平台目录、授权网页与原生
GAIA 提交；后续目录登记见下节，研究默认与正式服务未改。

### 岩相 P76 本地目录与预测 API 登记

`sixtrack_lithofacies_p76` 通过项目既有扩展入口登记模型、岩相任务、输入适配器和预测
runner。目录将其标为原开发折可运行、科学结论有条件；不表示最终 refit、许可审查或
多用户发布完成。独立模型环境执行 TabICL，平台进程不直接装载深度学习库。

物化脚本增加 `--lithofacies-input-root <恢复目录>` 和 `--lithofacies-checkpoint <固定权重>`，
复制前核对登记哈希。受 Git 管理的 manifest 与开发折索引绑定三个窗口 NPZ 和 checkpoint；
大模型及可恢复窗口不入 Git。索引只登记原开发折身份与分区，不含查询标签。

预测 API `POST /api/v1/prediction/tasks` 使用模型 ID `sixtrack_lithofacies_p76`、任务 ID
`lithofacies_classification`、`device: cpu` 和 `options: {fold_id: 0, repeat_id: 3}`。
开发折允许 0..2、确认重复允许 3..5；拒绝调用方哈希、布尔编号、原始井/地震、新区域、
快照 lineage、GPU 和体积裁剪参数。输入只能是登记索引选定的窗口，不接受调用方替换。
结果与下载回执记录模型版本、manifest 哈希、原分区身份、训练/查询行数及输出哈希。

[登记与 API 证据](../../../docs/evidence/sixtrack-lithofacies-p76-registration-api-20261002.json)
绑定本轮实际源文件，并记录三个开发折共 360 条查询的真实 API 执行、封存下载以及
CSV 九类概率与 NPZ 逐值一致。24 项目录/参数/物化/API 检查和 19 项目录/P21 回归检查
通过；本地目录现为四赛道六个模型，地震相与甜点尚未接入。检查使用私有解释器重新登记
插件；既有未提交 api.py 按哈希保留，正式服务、正式插件环境和前端 dist 均未更新。
该轮没有验证浏览器、多用户授权或原生 GAIA 提交，也未改变研究 XGBoost 默认。


### 岩相 P76 模型中心页面

本地试点页面新增岩相模型与原开发折 0..2、确认重复 3..5 的选择。表单只提交绑定档案的
编号，不继承断层路径、物性井族或体块参数。结果保持自己的任务模型身份，刷新恢复任务，
切换表单模型后仍显示原任务结果。当前装载赛道数以实际能力目录为准。

查看器使用中文标题与九类概率表，展示全部开发折查询行（最多 360 行）；类别代码是
离散标识，查询行不是井深，因此不使用通用连续曲线。概率与类别代码逐值对应封存 NPZ，
没有类别名称映射。表格可横向滚动查看九列概率，下载包含全部行与完整精度。

[岩相浏览器证据](../../../docs/evidence/sixtrack-lithofacies-p76-browser-20261002.json)
记录私有页面真实提交、开发折 0 的 132 行推理与显示、点击下载 hash 核对、首行逐值核对、
刷新恢复和切换表单后保持任务身份。6 项前端合同检查、13 项岩相 API 与 16 项 P21 回归
检查通过，类型检查与验证构建通过。正式 dist 未更新；用户项目授权、原生 GAIA 提交、
最终 refit 和新井数据仍需独立接入与验收。


### 甜点 T3 P7 的可移植查询内核

甜点研究真源仍保留 `F1_chronos2_train_blend`，不是 P79 exact-calendar 候选。此内核
仅移植 T3 的因果生产历史预测：固定 Chronos-2 权重输出与原历史均值按各开发折的原训练
选择权重融合（0.70、0.63、0.64、0.65），不重选权重、不读取查询标签、不执行最终 refit。
它不覆盖 T1/T2/T4/T5/T6/T7，也不输出空间甜点排名。

原 P7 输入是七通道的 30 条观测记录，记录间可有日期缺口；原数据代码也按后续 30 条
记录构建目标，不能把本入口说成连续 30 天预测。P79 的日历网格模型未晋级，本轮不替换
它。输出 `forecast_observation_steps` 明示观测索引，融合预测与历史均值是原油量字段
的数值尺度，不擅自换算流量单位。缺失协变量按原模型保留 NaN；全缺失油量及无穷值拒绝。

`tools/recover_sweetspot_p7.py` 从固定开发分区与授权 Volve 原始工作簿恢复查询窗口，
不调用原目标构建器；导出 `query_sequence`、`history_timestamps`、`sample_ids` 三项，
四折合计 1,111 条查询。原工作簿读取器会解析获准开发井的原始测量值；导出时只取各
cutoff 之前的 30 条记录，不声称原工作簿的后续原始行从未被解析。非开发井的数值行不读，
known holdout 与冻结测试文件不打开。恢复需要研究根路径与全新输出目录。

`worker/sweetspot_p7.py` 为独立 CLI，参数为 `--input <恢复NPZ>`、`--input-sha256 <hash>`、
`--snapshot <已固定本地权重目录>`、`--fold <0..3>`、`--output-directory <新目录>`。
模型与 config 必须匹配原 source lock；版本为 Chronos 2.2.0，CPU float32 单线程，
禁用下载并保持 `cross_learning=False`。原始 float64 历史均值与模型 float32 输入分别
保留。源文件、输入、权重和输出 hash 进入回执，旧输出目录不能覆盖。

该内核验证轮仅覆盖恢复与推理，后续目录/API 登记见文末；独立 CLI 不受平台任务取消控制，项目授权也未接线。恢复/推理配方与 CPU 对照见
[甜点查询内核证据](../../../docs/evidence/sixtrack-sweetspot-p7-portable-cpu-20261002.json)。


### 甜点 T3 P7 平台托管任务协议

内部 `sweetspot_runner.run_sweetspot_prepared` 接收 `ModelInputRequest`，只接受输入 SHA-256
与原开发折 0..3；模型目录由调用方授权后显式传入，不能调整融合权重、传未来协变量或
体积裁剪参数。它通过已有 `managed_run` 启动独立模型解释器，不在平台进程导入 Chronos。
甜点使用服务端环境变量 `WELL_SEISMIC_CHRONOS_PYTHON`，必须是绝对可执行文件路径；
缺失或错误时明确失败，不回退到其他赛道解释器。其他赛道沿用原解释器配置。

结构化 worker `worker/sweetspot_request.py` 在子进程复查历史输入与预训练资产，保存
NPZ、中文预测 CSV 和成功回执。CSV 的样本身份、融合预测、Chronos 观测索引均值与
原历史油量均值逐值对应 NPZ；父进程复核每项输出的哈希，旧结果目录不能覆盖。表中的
观测索引语义沿用上一节，不把该结果标成连续日历天、空间甜点图或新井验证。

worker 在预训练模型第一次真实前向计算完成后记录 `worker_stage.json`，不会改变
模型预测、分批方式或数值参数；另记录预测完成与输出保存阶段。阶段文件仅是历史记录，
进程是否存活与取消是否完成以平台运行时及真实退出状态为准。中断不生成成功回执。

[托管任务证据](../../../docs/evidence/sixtrack-sweetspot-p7-managed-worker-20261002.json)
记录四折共 1,111 查询的真实托管执行、中文表与 NPZ 一致、与上一轮独立 CPU 结果比较，
以及首次真实预训练前向完成后停止剩余查询的实测。进程返回 -15，运行时确认静止且没有
成功回执。47 项检查覆盖核心输入、任务协议、解释器配置拒绝及已有 P76/P21 回归。
该托管协议轮未公共登记；后续目录/API 登记见下节。项目授权或浏览器提交、部署仍待验收，研究默认不变。


### 甜点 T3 P7 本地目录与预测 API 登记

`sixtrack_sweetspot_p7_t3` 已通过既有扩展入口登记模型、`sweetspot_productivity` 任务、
输入适配器与 runner。元数据明确 T3 原开发折可运行、科学结论有条件；不代表其他甜点
目标、最终 refit、新井授权或正式服务可用。平台进程不导入 Chronos。

物化脚本新增 `--sweetspot-input-root <恢复目录>` 与 `--chronos-snapshot <固定本地目录>`。
受 Git 管理的 manifest 与开发折索引绑定四个查询 NPZ、Chronos weights/config、原查询
身份和原训练选择融合权重；大载荷不入 Git。复制前核对源 hash，错误源不得覆盖已有载荷。

`POST /api/v1/prediction/tasks` 使用模型 `sixtrack_sweetspot_p7_t3`、任务
`sweetspot_productivity`、`device: cpu`、`options: {fold_id: 0}`。只允许原折 0..3；
不接受 options 中的调用方输入哈希、融合权重、未来协变量，也不接原始井/地震、快照、
GPU 或裁剪参数。未知快照在共享 API 层先返回 409 且不排队，模型校验器另拒绝 lineage；
其余模型参数错误返回 422。回执绑定模型版本、manifest、原查询身份、权重和输出哈希。

[登记与 API 证据](../../../docs/evidence/sixtrack-sweetspot-p7-registration-api-20261002.json)
记录四折共 1,111 查询的真实 API 执行、封存下载、全部 NPZ 成员与托管 CPU 一致，以及
中文 CSV 身份/三列预测值与 NPZ 逐值一致。4 项实际 API、28 项目录/拒绝/物化和 19 项
目录/P21 回归检查通过。本地源目录现为五赛道七个模型，地震相未接入；私有解释器更新
插件元数据，正式环境未改。该轮没有浏览器、原生 GAIA、多用户项目授权或统一发布验收。


### 甜点 T3 P7 模型中心页面

本地试点页面新增 T3 模型与原开发折 0..3 的选择，当前装载赛道数以实际能力目录为准。
表单只提交绑定档案的开发折，不继承地震路径、体块裁剪、井族或岩相确认重复。必选模型、
开发折、确认重复和井族的选择提示使用中文；选定后清除必选错误，缺少开发折不发送请求。

结果使用中文五列表格，显示全部开发查询（最多 512 行），保留样本身份、融合预测、
Chronos 观测索引均值与历史油量均值。样本标识中的日期是输入截止点，不是预测日期；
30 条观测与 30 个预测步骤不保证连续日历天。独立查询不画成连续生产曲线，不作为空间
甜点图。数值沿用原油量字段尺度，不推断或换算流量单位；下载保留全部行与原精度。

[甜点浏览器证据](../../../docs/evidence/sixtrack-sweetspot-p7-browser-20261002.json)
记录私有页面真实完成开发折 0 的 99 条预测、中文必选提示及零请求拒绝、五列表显示、
点击下载 hash/首行/NPZ 全值核对、刷新恢复和切换模型后保留任务身份。7 项前端合同、
16 项甜点 API 及 29 项岩相/P21 回归检查通过，类型检查和验证构建通过。正式 dist 未改；
本轮不证明项目授权、原生 GAIA、新井数据、最终 refit 或全用户发布。

## 地震相 P81：权重恢复与独立推理

P81 原研究保存了逐样本预测，训练函数结束时释放了模型，没有保存训练后的 CNN、
SAM2 末层及融合/分割头权重。SAM2 预训练底座不足以复现这个模型，因此新增
`tools/recover_facies_p81.py`，调用原协议的训练函数并在最终预测时保存完整 `state_dict`。
恢复范围固定为 F3、Penobscot 各 5 个开发折、3 个原始种子；基线 40 次、融合 160 次
更新，原采样、train-only 标准化、损失和配置均保持不变。源码、开发数据、切分、
底座权重、上游版本和运行依赖由 `source-locks/facies-p81-recovery-v1.json` 绑定。

恢复是重新执行已验证协议，不是找回历史 GPU 权重字节，也不是最终全数据 refit。
输出必须位于研究仓库之外，已有结果不可覆盖；恢复记录绑定实际权重、无标签查询、
参考预测及历史预测的哈希。F3 保持 P81 稳定冠军方法；Penobscot 保持 P32/A0 方法的
operational reference 裁定，不把 P81 的均值第一改写成稳定冠军。

`worker/facies_architecture.py` 保留原 P12/P13 的原尺寸归一化、CNN 查询、SAM2
键值交叉注意力和 decoder 接线。`worker/facies_p81.py` 只加载完整训练权重，
不导入研究模块、不加载底座权重、不访问原始训练数据或标签。输入仅允许
`query_images: float32[N,1,128,128]` 与唯一 Unicode `sample_ids`，每次 1..16 张
原始开发查询；不接受标签、RGB、非有限值或其他尺寸。F3 输出 10 类，Penobscot
输出 8 类，各自保持原类别空间。输出为 logits、概率、类别码与原样本标识。

CPU 推理与原函数的 CPU 推理比较，GPU 推理与恢复时的 GPU 原函数输出比较。
两种设备之间的数值及分类变化另外记录，不能用单设备对照宣称 CPU/GPU 逐位等价。
GPU 恢复和推理必须先确认共享设备空闲，并只显式使用一个可见设备；其他人的进程不动。
独立推理在单独的验证环境运行，正式平台环境没有安装这一运行时。

[恢复与独立推理证据](../../../docs/evidence/sixtrack-facies-p81-portable-20261002.json)
记录真实逐单元对照结果。此处尚无平台模型登记、任务管理/取消、地震相页面、
新区输入预处理或正式发布证据，不能据此把模型中心的五个赛道计为六个。

本次验证恢复 30/30 个单元、执行 480 张开发查询，61 项检查通过。恢复预测按原归档
float16 softmax 规则比较，全部类别码一致；独立推理分别对原函数的同设备输出逐位一致。
按原归档标签、全部配置类别（无支持类别计零）复算，F3 的 CPU 平均 mIoU 为
0.318756（原归档 0.318752），Penobscot 为 0.173378（原归档 0.173409）。
CPU 与 GPU 的完整精度输出在 7,864,320 个像素中有 3,467 个类别码变化，
不宣称两种设备或归档 float16 与当前 float32 输出逐位一致。

### 地震相内部受管理任务

`facies_runner.run_facies_prepared` 接收原开发查询，并显式提供完整权重、恢复记录和
锁定 SAM2 上游源码路径。请求必须绑定数据集、原开发折 0..4、原种子重复 0..2，以及
查询、权重、恢复记录的三个哈希；父进程和子进程分别核对，跨折、类别空间不符、
篡改载荷、标签、GPU 请求、体块裁剪及其他参数均不接受。模型解释器单独由
`WELL_SEISMIC_FACIES_PYTHON` 指定，未配置或路径非法时直接失败，不借用其他模型环境。

任务由平台现有 `TaskRuntimeRegistry`/`managed_run` 管理，CPU 隔离执行。输出包括
原始 logits、概率、类别码、样本标识及切片索引 JSON，后者只表示 128×128 图像像素，
不虚构空间坐标或类别中文名。恢复记录中的训练折标准化、种子、标签版本和结果哈希
随运行回执保留。任务不训练模型，不读取研究源代码、原始训练数据或查询标签。

30 个真实受管理任务共 480 张查询，全部数组与此前独立 CPU 推理逐位一致。
取消测试在真实模型完成首批 2/16 张预测后触发，确认子进程原先仍在运行、随后以
`-15` 退出，注册表返回 `quiescent`，没有成功回执。阶段文件是历史进度，不能代替
当前进程状态或取消报告。这里尚未提供公开 API 取消入口或页面按钮。

配置真源随 wheel 放在 `wellseismic_sixtrack_pilot/source_locks/`；旧
`source-locks/facies-p81-recovery-v1.json` 作为同内容的软链接保留，原哈希未变。
独立 wheel 已在项目目录之外运行 F3/Penobscot 两个真实模型，对原 CPU 输出逐位一致；
这证明模型 worker 的安装包读取配置正常，不代表正式平台、模型登记或发布门已通过。

[受管理任务与取消证据](../../../docs/evidence/sixtrack-facies-p81-managed-worker-20261002.json)
封存 95 项检查及实际任务哈希。

### 地震相模型登记与预测接口

`sixtrack_facies_f3_p81` 与 `sixtrack_facies_penobscot_p81_reference` 分别登记模型、
输入适配器和推理入口，共用任务 `seismic_facies_slices`。F3 保持稳定方法身份；
Penobscot 保持操作参考身份，两者分别使用 10 类和 8 类，不合并标签空间。
每个模型的 `development_cells.json` 绑定原开发折 0..4、重复 0..2、原种子、
查询名单、完整训练权重和恢复记录。查询名单哈希与原 P81 归档逐项一致。

预测入口为 `POST /api/v1/prediction/tasks`，提供上述模型 ID、任务 ID、
`device: cpu` 和 `options: {fold_id: 0, repeat_id: 0}`。调用方只能选择开发折和
原重复，不能传入查询路径、权重、哈希、新区数据或查询标签。执行时重新检查模型、
数据集、版本和载荷，再核对输出切片名单；结果下载保留 logits、概率、类别码、
样本身份、切片索引和绑定回执。

模型解释器由 `WELL_SEISMIC_FACIES_PYTHON` 指定，SAM2 源码由
`WELL_SEISMIC_SAM2_SOURCE` 指定，必须是锁定提交且无已跟踪修改的独立上游仓库。
载荷物化增加 `--facies-recovered-root <恢复目录>`；权重与查询载荷不提交 Git。
本次安装与验证仅使用私有环境，未更新正式平台环境或服务。

部署前从平台根目录运行 `python3 scripts/check_sixtrack_inventory.py --require-all-runnable --probe-facies-runtime`。它使用当前配置的
P81 解释器，实际导入 torch、torchvision、SMP、timm、hydra、omegaconf、iopath，
核对原锁中的五项版本及 SAM2 提交和已跟踪改动，再导入真实 SAM2 构造入口。
任一缺失、版本不符、超时或入口失败均退出非零；禁用 GPU、在线模型下载及字节码写入，
不加载权重、不创建预测任务、不安装依赖。未加 `--probe-facies-runtime` 的清单检查只
证明模型登记和载荷，解释器存在也不保证依赖可用。此探测只覆盖 P81；成功不能替代
真实推理、其他赛道环境、用户授权或发布门。

2026-10-04 实测 geo-sixtrack-py311 解释器因缺少 hydra 被拒；已有 P81 专用环境通过。
该次通过的专用环境位于临时目录，并通过 `.pth` 读取研究环境中的主要依赖，不能把
临时路径或其局部 freeze 当作独立交付环境。
证据见 [P81 环境预检](../../../docs/evidence/facies-runtime-preflight-20261004/README.md)。

随后在全新私有环境完成独立离线安装，使用 `requirements-facies-runtime.lock` 固定
全部解析依赖和允许的发行文件 SHA，配合 CPython 3.11、Linux x86_64 和 cu130 包索引。
实际 worker 使用 CPU；为保持原模型锁，没有替换原 PyTorch CUDA 发行版本。
从新安装的 wheel 执行 F3/Penobscot 原开发折0、重复0，各16张查询，全部输出数组
与此前完成的 CPU 任务完全一致；主要模块实际路径均属于新环境，未借用研究或临时
环境的 site-packages。安装配方及包哈希见
[独立环境重建](../../../docs/evidence/facies-runtime-install-20261004/README.md)。
此证据使用本机既有缓存和固定 SAM2 源码，不等于空缓存机器安装、完整模型登记
部署、所有开发折复验、用户界面验收或依赖许可审查；正式服务配置尚未切换。

### 安装包模式的载荷目录与 worker

以非可编辑 wheel 安装扩展时，平台进程须配置
`WELL_SEISMIC_MODEL_PROJECT_ROOT=<平台项目绝对路径>`，该目录必须含
`models/task-models`。载荷仍由每个模型的 manifest 和 SHA 校验，不复制进 wheel；
模型输入元数据保留相对平台项目的路径，继续由平台原有源校验器约束。
源码开发模式可以使用现有项目布局；安装包模式不从当前目录或 site-packages 推断
项目位置。缺配置或显式配置错误时模型登记失败，不回退到其他目录。

安装包模式下，各模型解释器也必须安装同一 SHA 的扩展 wheel；启动 worker 时
不传入平台 site-packages，也不继承研究 PYTHONPATH。worker 仍使用显式输入及
原有 CPU、离线和取消管理合同；worker 源码 SHA 从所执行安装包的模块目录读取。
F3/Penobscot 的私有安装包登记与实际 API 任务验证见
[wheel 载荷目录核验](../../../docs/evidence/sixtrack-wheel-registration-20261004/README.md)。
此处不是正式配置变更；另外七个模型的安装包 worker 任务、完整平台安装、授权项目
和原生 GAIA 工作流仍需各自验证。

平台调用 worker 时现会核对实际解释器找到的扩展包：包内所有 `.py` 与 `.json`
必须逐项匹配父 runner 的包内容，新增、缺失、旧代码或不同配置均拒绝；核验在
导入扩展包和推理前执行。成功启动留下 `worker_package_identity.json`，运行溯源
包含实际包目录、内容 SHA 和文件清单。该记录只证明启动前文件核验，不表示任务
已经完成，不冻结整个环境或阻止运行中外部修改。其他依赖仍需各自版本和发布审查。
[真实旧包拒绝、匹配包推理及取消核验](../../../docs/evidence/sixtrack-worker-identity-20261004/README.md)
保留失败任务；正式服务未启用这项候选改动。

通用断层/物性结果及 P21 结果也传递 `verified_worker_package`，而非仅把核验记录
留在运行目录。旧任务历史不会因此改写，缺少公开字段的旧结果仍按原记录展示。

`requirements-model-runtime.lock` 和 `requirements-chronos-runtime.lock` 固定另两套
解释器的完整依赖及 SHA；模型环境使用原 torch 分发版本2.13.0，Chronos使用
原2.12.1+cu130，不混用包索引中的同公共版本替代品。新环境安装后主要模块均在
各自目录，无研究环境的 `.pth` 借用。实际安装包任务、索引选择失败及观察到的
数值差异见 [其余七模型安装核验](../../../docs/evidence/sixtrack-wheel-seven-20261004/README.md)。
这组记录含回执修复前后的两个制品阶段，不作为同一正式版本全量验收；P76分类
结果相同但logits未逐元素相同，不能把安装清单或依赖版本匹配说成全输出完全复现。

[登记与真实接口证据](../../../docs/evidence/sixtrack-facies-p81-registration-api-20261002.json)
记录两个数据集全部 30 个真实 API 任务、480 张查询和封存下载结果。
六赛道现均有模型载荷登记；公开取消、项目授权、新区数据能力、原生 GAIA 和统一发布
仍需继续验收。

### 地震相切片页面与查看器

模型中心的六赛道试点可分别选择 F3/Penobscot、原开发折与原重复，提交真实 CPU
任务。查看器逐张展示登记输入、类别码与最大类别概率，可切换全部 16 张原生
128×128 查询并读取鼠标处的像素类别和概率。类别图使用离散颜色，各类别像素数
随样本更新；概率只是模型最大 softmax 输出，不声称已校准可信度。输入灰度使用
单张切片的 1%—99% 分位显示，坐标只表示图像行、列，类别只显示原数据集代码。

显示前重新核对输入哈希、查询名单、类别空间及输出封存身份。六项查看器检查将
全部展示类别与概率逐像素对照原 NPZ，并验证错误查询/标签空间不能显示；连同
接口回归共 31 项后端检查通过。八项前端合同检查与验证构建通过。

私有浏览器已真实提交两个数据集各一组任务，共 32 张切片，检查样本切换、画布
颜色、鼠标概率、点击下载哈希、刷新恢复和切换选择后的任务身份。结果入口与
溯源导航使用中文。证据见
[地震相浏览器验收](../../../docs/evidence/sixtrack-facies-p81-browser-20261002.json)。
这证明原开发查询试点可运行与查看；双账号项目授权、原始上传、新区推断、原生
GAIA、公开取消与正式同版交付尚未通过。验证只使用私有服务与 `.build-verify`，
正式环境及服务未更新。

### 岩相查询输入导出（未安装的源码候选）

P76 worker新增`prepared_query_inputs_npz`及`prepared_input_contract`，仅保存原查询行的13测井值通道、13可见性通道、地震3×3×33窗口和查询序号。没有支持集标签或查询真值，不提供未知通道名、物理单位或完整原始轨迹。已有类别/概率输出保持；同解释器旧源码控制逐位一致，历史任务的微小浮点差异仍保留。三折360查询导出和真实CPU子进程证据见[岩相输入导出](../../../docs/evidence/p71-lithofacies-query-inputs-20261011/README.md)。[输入查看页离线候选](../../../docs/evidence/p71-lithofacies-input-view-20261011/README.md)已通过三折360行浏览器逐值与可见性核对；[私有运行接入](../../../docs/evidence/p71-lithofacies-input-runtime-20261011/README.md)已完成，authority与两worker显式加载冻结扩展包，未覆盖site-packages；[普通账号新版流程](../../../docs/evidence/p71-lithofacies-input-login-20261011/README.md)已完成fold0/repeat3单次CPU任务的五项封存产物、输入查看、实时与关闭后历史NPZ下载及隔离/撤销；AI五项元数据核对完成，但整段AI审计仍FAIL。其余配置与科学验收仍待，正式环境未发布。
