# 慧眼受控业务工具接口

该接口把现有数据准备、井震标定、融合样本、预测和可视化函数注册为 AI 可调用的操作。身份来自盖亚服务端认证，模型仅提交动作名和当前账号已有的项目、资产、任务、版本 ID；不允许传本机路径、命令、脚本、凭据或用户身份。

## 安装与启用前提

现有开发环境复用已配置的解释器，不在工作副本重复装包。首次配置该平台环境时，需要 `pyproject.toml` 的 `web` 与 `agent` extras；只安装 `.[web]` 或 `requirements-interface.txt` 不包含真实 DVC 和全部数据处理依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[web,agent]"
.\.venv\Scripts\python.exe -m dvc --version
```

`agent` 固定 `dvc==3.67.1`，并声明 LAS/SEG-Y、标定/样本的数据处理依赖。可选可视化适配器按 `visualization` extra 配置；这些 extras 均不附模型权重或训练运行时。通过认证代理调用 `dvc_status` 后，应检查实际 `available` 和 `version`；缺少命令时返回不可用，不能用普通聊天可回复代替 DVC 验证。

服务鉴权与 MiniMax 仍使用已有后端配置，详见 [盖亚任务接入的配置说明](02_Gaia任务接入.md#后端配置与安装前提) 和 [README 模型配置](../README.md#基础问答与-minimax-配置)。服务令牌、模型密钥及私有配置不进入浏览器或 Git。

## 入口与回执

| 入口 | 用途 |
|---|---|
| `GET /api/v1/gaia/harness/actions` | 动作 JSON Schema、真实就绪状态、上传大小与角色目录 |
| `POST /api/v1/gaia/harness/actions` | `{operation_id: UUID, operation, arguments}` |
| `GET /api/v1/gaia/harness/runs/{execution_id}` | 按当前账号查询持久化执行回执和原生任务状态 |
| `POST /api/v1/gaia/harness/projects/{project_id}/assets/{role}?filename=...` | 上传原始字节，最大 120 MiB；角色和扩展名以目录为真源 |
| `GET /api/v1/gaia/harness/tasks/{task_id}/visualization` | 按所属账号打开原生任务可视化 |

所有入口沿用服务端 Bearer 凭据及可信 `X-Gaia-Subject`。浏览器不应持有服务凭据，必须通过盖亚所属账号代理。回执使用 `well-seismic.harness.v1`，包括 `execution_id`、`operation_id`、`request_sha256`、状态、原生任务 ID、结果或结构化错误。请求摘要为 UTF-8 JSON 的 SHA256：`{operation,arguments}`，键排序、保留 Unicode、紧凑分隔符。

登录后的侧栏根据服务端 `subject + task_id` 只读解析当前任务所属项目，把已核实的项目 ID 置于任务摘要前部；不会根据项目列表顺序猜测，也不会因打开对话自动创建项目。跨账号、未知任务或不一致归属拒绝访问；已授权的旧 LAS 任务若未绑定 Harness 项目，明确返回未绑定，需要先创建或导入关联项目。权限、参数、冲突和服务不可用等预期错误保留安全的分类，不把它们统一显示成接口异常。

同一账号同一 `operation_id` 重试返回原执行；改动参数返回 409。跨账号项目、资产、任务和执行查询返回 404。原生队列 completed 不直接等于业务成功：标定 `business_status=blocked` 会传播为 blocked；原生成功但 DVC 失败也会明确版本闭环未完成。

`task_status` 在核对账号、项目和任务归属后，会刷新对应原生提交执行，复用页面刷新相同的终态和自动 DVC 归档路径。查询返回 `original_executions` 状态摘要与已有版本 ID；并发页面刷新和 AI 查询在同一 Harness 实例内串行收敛，重复查询复用原回执，不重跑原生任务或重复归档。归档失败、原生失败及业务门禁失败明确保留；没有 Harness 原执行的导入任务只报告原生状态，不伪造归档成功。失败归档不因再次查询而隐式重试，需显式发起归档操作。AI 同轮先提交、后查询时，会按当前账号刷新已收集的原执行回执，最终回答、执行卡与持久化请求使用相同的最新状态，避免仍显示首次提交的运行中快照。

上传按当前账号、工程、角色、原始文件名及内容 SHA256 去重。已有成功回执原样复用；已有资产但 `versioning.status=blocked` 时，用户在原项目重传同文件会重新检查原资产归属及哈希，只恢复其归档，不重写文件、不创建新资产。成功后更新上传缓存并复用同一版本；不会修改其他项目的资产或失败回执。请求中断后未形成回执的持久化上传声明仍明确返回 409，不盲目重做可能已完成的注册。

显式 `dvc_archive(task_id)` 与自动归档共用完整工件收集器，包含所属原生任务声明的 `outputs`、`output_files`、`artifacts` 文件及不可变任务摘要。结构化 artifact 的 `path` 是文件路径，名称、类型和哈希是描述字段。声明工件缺失、越界、无法定位或超量时整次归档失败，不能仅归档任务 JSON 后宣称结果完整。相同文件内容、元数据及父版本的归档复用已有版本；协调与完整性边界见 [DVC 版本说明](dvc-versions.md#归档去重与故障恢复)。

手动归档回执在 `result` 保留 `task_id`（仅资产归档时为 null）和 `source_execution_ids`。恢复成功后，新的 `task_status` 查询返回 `result.archival_recovery={execution_id, revision_id, status:'completed', source_execution_ids}`，当前查询的 `revision_id` 指向已核验的恢复版本；`original_executions` 和历史原生执行的失败回执不改写。只有来源执行匹配的 `versioning_incomplete` 可在新查询中解除；归档恢复不解除业务质量失败、其他原生失败或缺模型权重。AI 的大回执摘要保留有界的恢复关联与原执行状态，完整事实仍在所属执行记录中。

可视化的 Harness 入口和所属账号 UI 入口均将原生页面导航保留在当前任务路径，仅接受单值 `asset`（非负整数）、`block`、`layerpulse_output`（字母、数字、下划线或连字符）查询；不允许查询参数替换任务归属。

## 原生功能及边界

- `project_create/list`、`asset_list`：所属账号项目及受控资产。可通过 `project_create` 的 `task_id` 导入已有、属于当前账号的盖亚 LAS 检查任务。
- `data_prepare`：调用原有 `create_data_preparation_task`，保存原始输入并生成不可变快照。`well` 是井头资料，`trajectory` 是井轨迹，二者进入 `well_paths`；`survey` 是地震测区网格资料，不能混用。
- `curve_preview`：用 `read_las` 与 `CurveKnowledgeBase` 返回单位、处理证据和限量样点。
- `curve_mapping`：只允许知识库认可的同义曲线名，例如 RHOB→DEN。生成新 LAS，原生重新解析并验证深度、值和 masks 不变。原始数据不覆盖。未知单位导致全无效值时 blocked，不能用改名伪装单位修复。内部 canonical 仍是 RHOB，保证既有物理算法合同一致。
- `registration`：原生 `create_registration_preflight`→`create_registration_task`，保留 CRS、时深、质量和同源检查。
- `sample_build`：原生 `create_sample_building_task`，只使用所属账号的源快照与标定结果。
- `prediction`、`horizon_attribute_analysis`：原生预测入口。当前接口版缺原始任务权重和 WellFuse 运行时，明确返回 `model_assets_missing`，不生成随机权重或简化算法结果。
- `task_status`、`visualization`：真实任务结果与原生页面，不制造预测或指标。
- `dvc_status/archive/list/get/compare/restore`：调用真实 `DvcVersionStore`。上传、曲线映射与已完成原生任务自动归档；恢复到新目录并根据版本内封存角色登记新资产，允许重跑，不覆盖当前或封存文件。

数据、执行与所有权记录在忽略的运行目录中，服务重启后仍可读取完成历史。回执剥离本机绝对路径；DVC 逻辑文件名短且稳定，原名、角色、血缘和哈希保留在版本元数据中。

`project_list`、`asset_list`、`dvc_list` 共用可选分页参数 `offset`（从0开始）和 `limit`（1–32，默认32），返回全量 `total`、本页 `count`、`offset`、`limit`、`next_offset` 和 `order`。项目按创建先后倒序，资产按登记顺序，DVC版本按创建时间与版本 ID 倒序。需要完整清单时，用响应中的 `next_offset` 继续调用相同动作，直到它为 `null`；到达末尾返回空页，超出清单范围明确拒绝。身份仍由服务端认证决定，分页不能跨账号查询。

送给 AI 的单项回执超过 12,000 UTF-8 字节时会保留受限摘要。真实 ID、父版本/父资产和必要标签仍可用于下一次查询；`total` 保留全量总数，`count` 表示此次实际可见的引用数。字节预算可能使实际条数少于请求的 `limit`，此时 `next_offset` 从实际可见引用之后继续，不能自行按 `offset + limit` 跳页。摘要另有 `*_total`、`*_count`、`*_truncated`，不包含文件路径、原始数据数组或整份版本文件清单。只有读到 `next_offset=null` 才能称已遍历完整清单；如果工具预算耗尽，应明确已查询范围。完整回执保留在所属执行记录中，也可按版本 ID 查询详情。分页针对当时的清单；遍历期间若新增记录，应重新查询核对。

## 可复现验证

先完成上述依赖检查，并由盖亚普通登录账号使用实际受控入口。模块测试与隔离合成数据复现使用当前环境运行：

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_huiyan_harness.py tests/test_gaia_ui.py tests/test_gaia_tasks.py -q
.\.venv\Scripts\python.exe -B -m pytest tests/test_huiyan_archive_recovery.py tests/test_huiyan_task_status_completion.py -q
.\.venv\Scripts\python.exe -B scripts/verify_harness_native.py --output <新的报告路径>
.\.venv\Scripts\python.exe -B scripts/verify_harness_native.py --generate-only <新的合成数据目录>
```

另按 [DVC 说明](dvc-versions.md#运行与明确限制) 运行真实 CLI 用例。未安装 DVC 导致的 skipped 不算通过，单纯模块测试也不替代普通账号自然语言执行与页面验收。

原生验证脚本创建新隔离源码快照，仅复制源代码、配置、接口合同及无权重标记；不复制 `.env`、账号、旧任务或旧证据。它生成 601 点 LAS、3×3×701 SEG-Y、井头、带明确单位的轨迹和时深表，运行上传→曲线映射→数据准备→标定→融合样本→DVC 归档、比较、恢复与重跑。数据明确标注为合成，不是现场业务数据或模型效果评估。

2026-11-15 的隔离验证获得 1 口井、601/601 有效标定点，方法是原生 `provided_time_depth`，`business_status=usable` 且 1 口井融合就绪。P13 `checkpoint_executed=false`、`runtime_unavailable` 如实保留。预测步骤按缺原始资产 blocked，不能把此验证写成 LayerPulse、P13 或其他任务模型已经验收。

还需要团队提供 `interfaces/model_registry.json` 中对应模型的原版本 checkpoint（SHA256、来源、许可与指标）、配套 `src/wellfuse5090`/推理脚本、相容环境和真实业务样本。只有原生门禁通过并实际执行后，才能验收全核心模型流程。
