# 慧眼受控业务工具接口

该接口把现有数据准备、井震标定、融合样本、预测和可视化函数注册为 AI 可调用的操作。身份来自盖亚服务端认证，模型仅提交动作名和当前账号已有的项目、资产、任务、版本 ID；不允许传本机路径、命令、脚本、凭据或用户身份。

## 入口与回执

| 入口 | 用途 |
|---|---|
| `GET /api/v1/gaia/harness/actions` | 动作 JSON Schema、真实就绪状态、上传大小与角色目录 |
| `POST /api/v1/gaia/harness/actions` | `{operation_id: UUID, operation, arguments}` |
| `GET /api/v1/gaia/harness/runs/{execution_id}` | 按当前账号查询持久化执行回执和原生任务状态 |
| `POST /api/v1/gaia/harness/projects/{project_id}/assets/{role}?filename=...` | 上传原始字节，最大 120 MiB；角色和扩展名以目录为真源 |
| `GET /api/v1/gaia/harness/tasks/{task_id}/visualization` | 按所属账号打开原生任务可视化 |

所有入口沿用服务端 Bearer 凭据及可信 `X-Gaia-Subject`。浏览器不应持有服务凭据，必须通过盖亚所属账号代理。回执使用 `well-seismic.harness.v1`，包括 `execution_id`、`operation_id`、`request_sha256`、状态、原生任务 ID、结果或结构化错误。请求摘要为 UTF-8 JSON 的 SHA256：`{operation,arguments}`，键排序、保留 Unicode、紧凑分隔符。

同一账号同一 `operation_id` 重试返回原执行；改动参数返回 409。跨账号项目、资产、任务和执行查询返回 404。原生队列 completed 不直接等于业务成功：标定 `business_status=blocked` 会传播为 blocked；原生成功但 DVC 失败也会明确版本闭环未完成。

上传按当前账号、工程、角色、原始文件名及内容 SHA256 去重，重复上传返回原资产和版本回执。请求中断后不会静默生成重复资产；未完成的持久化上传声明明确返回 409。可视化的 Harness 入口和所属账号 UI 入口均将原生页面导航保留在当前任务路径，仅接受单值 `asset`（非负整数）、`block`、`layerpulse_output`（字母、数字、下划线或连字符）查询；不允许查询参数替换任务归属。

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

## 可复现验证

使用当前环境运行：

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_huiyan_harness.py tests/test_gaia_ui.py tests/test_gaia_tasks.py -q
.\.venv\Scripts\python.exe -B scripts/verify_harness_native.py --output <新的报告路径>
.\.venv\Scripts\python.exe -B scripts/verify_harness_native.py --generate-only <新的合成数据目录>
```

原生验证脚本创建新隔离源码快照，仅复制源代码、配置、接口合同及无权重标记；不复制 `.env`、账号、旧任务或旧证据。它生成 601 点 LAS、3×3×701 SEG-Y、井头、带明确单位的轨迹和时深表，运行上传→曲线映射→数据准备→标定→融合样本→DVC 归档、比较、恢复与重跑。数据明确标注为合成，不是现场业务数据或模型效果评估。

2026-11-15 的隔离验证获得 1 口井、601/601 有效标定点，方法是原生 `provided_time_depth`，`business_status=usable` 且 1 口井融合就绪。P13 `checkpoint_executed=false`、`runtime_unavailable` 如实保留。预测步骤按缺原始资产 blocked，不能把此验证写成 LayerPulse、P13 或其他任务模型已经验收。

还需要团队提供 `interfaces/model_registry.json` 中对应模型的原版本 checkpoint（SHA256、来源、许可与指标）、配套 `src/wellfuse5090`/推理脚本、相容环境和真实业务样本。只有原生门禁通过并实际执行后，才能验收全核心模型流程。
