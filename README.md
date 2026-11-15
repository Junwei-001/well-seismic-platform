# 地层慧眼平台｜独立接口版（无任务权重）

视频称谓：最新定稿视频（2026-11-12 12:14导出，录制于本次主线移植前）；其原文件和证据保持不变。

盖亚任务调用的配置、接口与本地验收步骤见 [Gaia 任务接入](docs/02_Gaia任务接入.md)。
当前已验证 LAS 数据准备调用链；模型预测仍遵守下面的接口版运行边界。

这是从当前 `WellFuse/platform` 工作树抽离出的独立平台骨架，版本标识为
`20260903`。平台的前端、接口路由、数据合同、任务语义和插件扩展点均已保留；
任务专属 checkpoint、CUDA/模型运行环境、历史训练产物、业务数据和缓存均未带入。

## 目录约定

```text
地层慧眼平台_独立版_无任务权重_20260903/
├─ frontend/                 Vue 前端（当前工作树原样复制，含 source + dist）
├─ src/well_seismic/         FastAPI、数据处理、合同与注册中心
├─ configs/                  与模型无关的输入/坐标/融合配置
├─ interfaces/               模型接口清单与上游可视化适配代码
├─ 接口模型/                 保持旧相对路径的上游适配源码（无权重）
├─ models/task-models/       未来外置任务模型的接入槽位（当前为空）
├─ runtime/                  平台相对路径挂载点（当前无运行时）
├─ data/                     本机 SEG-Y/LAS/轨迹数据入口（当前为空）
├─ model_outputs/            模型结果输出入口（当前为空；状态库在 runtime/state）
├─ scripts/                  通用脚本；启动使用 `启动接口平台.ps1`/`.bat`
└─ docs/                     抽离说明、接口扩展和前端保真记录
```

后端故意保持原有根级布局（`src`、`frontend/dist`、`configs`），因此
`python -m well_seismic.api` 的相对路径和同源前端资源不需要改写。

## 当前运行边界

启动脚本会设置 `WELLFUSE_MODEL_MODE=interfaces_only`。后端仍返回完整的
`/api/v1/capabilities` 和 `/api/v1/releases` 合同，但任务模型的
`runtime_status` 为 `adapter_required`、运行器列表为空，提交预测会得到结构化的
“模型不可运行”响应，不会尝试寻找或加载隐含权重。`models/INTERFACE_ONLY` 是同样的
fail-closed 标记，即使直接执行 `python -m well_seismic.api` 也会保持该模式。

允许运行的内容仅限于平台级数据合同、快照/元数据处理、接口查询和可视化壳；
不要把此目录误认为已完成模型部署包。

## 启动与检查（需要本机 Python）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[web]"
# 或：.\.venv\Scripts\python.exe -m pip install -r requirements-interface.txt
.\启动接口平台.ps1
```

也可以直接运行 `install_interface_dependencies.bat` 自动创建 `.venv` 并安装上述依赖。

启动脚本会依次尝试本目录 `.venv`、`runtime/py311`、系统的
`%LOCALAPPDATA%\WellSeismicPlatform\py311` 和 PATH 中的 Python，并检查
`fastapi + uvicorn + numpy + PyYAML + pyproj`；也可传入 `-PythonPath <python.exe>`。接口模式变量会写入子进程并
记录 PID，停止时只终止本目录登记的服务。双击 BAT 使用纯 ASCII 调度脚本，避免中文
路径编码问题。

浏览器访问 <http://127.0.0.1:725>。停止服务请运行
`.\停止接口平台.ps1`（也可双击 `停止接口平台.bat`）；不要只结束启动器窗口，避免后台
Python 进程继续占用端口。若只检查代码和目录合同：

```powershell
.\.venv\Scripts\python.exe tools\verify_platform_skeleton.py
```

前端原样回归测试仍位于 `frontend/tests`；有 Node.js 时在 `frontend` 目录执行
`npm ci` 和 `npm run test:unit`（动态编译类测试若提示缺少 `esbuild`，按开发环境补装
该工具即可）。不要重新构建后覆盖现有 `frontend/dist`，以免改变已封存的资源哈希；
需要构建时请先备份 dist。

如果直接双击后窗口提示“未找到完整的 Python Web 运行时”，说明所选 Python 缺少
`uvicorn` 或 `pyproj` 等依赖；在本目录重新创建 `.venv` 并执行上面的安装命令，或用
`-PythonPath` 指向已经同时包含 `fastapi、uvicorn、numpy、PyYAML、pyproj` 的解释器。

## 后续模型接入

模型接入顺序固定为：解释任务 → `ModelSpec` → `ModelInputAdapter` →
`PredictionRunner` →（可选）`FusionStrategy`。接口组、字段和稳定 ID 见
`interfaces/model_registry.json` 与 `docs/01_接口与扩展点.md`。未来扩展包应：

1. 放入 `models/task-models/<model_id>/<version>/`，不写入平台源码或前端；
2. 提供 SHA-256 清单、来源/许可和输入输出合同；
3. 以 entry point 注册适配器和运行器，并在隔离环境完成验证；
4. 明确设置 `WELLFUSE_MODEL_MODE=full_runtime` 后再启用。

当前抽离目录不包含任何 `.pt`、`.pth`、`.safetensors`、`.ckpt`、`.onnx`、SEG-Y、
LAS 或压缩模型环境文件。

## 慧眼 AI 侧栏接入盖亚

慧眼各业务页面右下角的“慧眼AI”按钮可打开页面内对话侧栏；原有完整对话页也保留。
浏览器只向本平台的 `/api/v1/assistant/chat` 发送问题。
切换任务会隔离旧会话与未完成请求；旧任务的迟到回复或错误不会进入新任务。
模型失败后使用本地规则回答时，页面会单独显示降级提示和安全的失败原因。
后端支持复用盖亚本机配置中的 MiniMax，也支持盖亚 Open API。模型不可用时，
继续使用原有的本地 LLM 或明确标记的规则回答。每个新对话生成会话 ID，连续消息复用该 ID；任务摘要与问题
在发往盖亚前会移除可识别的本地绝对路径。模型调用不会获得原始 SEG-Y/LAS 文件。

在被忽略的本地 `.env` 中设置以下变量（参照 `.env.example`），不要把密钥写入
前端、`configs/` 或 Git 仓库：

如果慧眼后端所在机器已有盖亚的 `data/cmd_config.json`，在 `.env` 中设置：

```text
WELL_SEISMIC_GAIA_CONFIG_PATH=<该文件的本机绝对路径>
```

后端按盖亚的 `provider` → `provider_source_id` → `provider_sources` 结构读取已启用的
MiniMax 模型及凭据，不复制密钥。默认跟随盖亚的默认 MiniMax provider；有多个 MiniMax
且默认模型不属于它们时，需显式设置 `WELL_SEISMIC_GAIA_PROVIDER_ID`。
盖亚配置中的密钥可使用 `$NAME` 或 `${NAME}` 引用；慧眼后端从自身进程环境读取，
不修改源配置。引用未设置、为空或无效时按配置不可用处理，不将引用字面值发送给模型接口。
协议与盖亚的 `minimax_token_plan` 适配器一致：Anthropic Messages API、Bearer 鉴权，
仅访问固定的 MiniMax 官方地址。回复来源显示“盖亚 MiniMax（复用配置）”；此方式调用
MiniMax 模型，不经过盖亚公网聊天服务，也不授予盖亚工具执行能力。
对话历史只保存在慧眼后端内存中，最多 64 个会话，空闲一小时过期，重启后清空。

如果使用远端盖亚服务，配置以下内容；完整的 Open API 配置优先于本机配置：

```text
WELL_SEISMIC_GAIA_CHAT_BASE_URL=https://gaia.yongan.site
WELL_SEISMIC_GAIA_CHAT_API_KEY=<Gaia Open API 中仅有 chat scope 的密钥>
WELL_SEISMIC_GAIA_PROVIDER_ID=minimax-token-plan/MiniMax-M2.7
```

示例 provider ID 来自 Gaia 代码中的 MiniMax 映射；部署时应先在盖亚管理界面核对
该 provider 仍已启用。修改 `.env` 后重启慧眼后端。侧栏配置状态由 `/api/v1/capabilities` 的 `assistant` 字段
返回，其中只包含密钥是否已配置，不包含密钥本身。可在慧眼页面连续发送两条消息
检查会话是否复用，并在盖亚侧核对使用的 provider。需要访问远端盖亚时必须使用
HTTPS；HTTP 仅允许本机回环地址。“模型已配置”只代表配置齐全，实际调用应检查聊天
响应的 `provider`、`model` 和回复来源。本机配置方式的 `model` 来自 MiniMax 实际响应。
每条 MiniMax 回复下方同时显示“接口返回模型”，它取自响应元数据，不采信回答正文中的
模型自述。顶部显示的是配置模型；这些 ID 不能独立证明服务商内部的模型部署版本。
对于“你是什么大模型”等独立型号查询，后端在本次真实接口调用成功后，用请求和响应
字段直接生成核对结果，并标记来源为“后端模型信息（本次调用已核对）”。普通聊天仍
展示模型生成的回答；型号查询的核对结果也会写入后续对话上下文。

盖亚发起的LAS数据检查及原生页面集成说明见[盖亚任务接入](docs/02_Gaia任务接入.md)。
当前配套Gaia基于团队主线`04b8555`，使用`my_plugins/gaia_huiyan`、
`/api/plug/huiyan`与现有mission/job ledger（producer `huiyan_las`），
从本人任务`job_ref`打开对应原生结果；没有恢复旧WorkflowRun API。

[任务1最新交付材料](docs/deliveries/task1-final-20261113/README.md)收录本次唯一新版视频、
当前截图、接入说明与独立测试证据。完整本地main.py/Core/PluginManager、正常注册账号和受保护iframe已实测，
8项注册、31项真实HTTP、9项浏览器操作分别通过；是隔离本地正常账号，不是用户生产账号或团队正式上线。
Gaia目标70项、启动工具12项、慧眼备用OpenAPI安全补正后52测试+15子测试分别留证，不相加为全库通过。

最新说明字幕版视频3分11.07秒（11章/12段字幕）、1920×1080/30fps，SHA-256
`966220b97c447512b5b0d20f5da482b07aec1b23c1e03fbbd98f9dfd56a30dcb`，
对应job`j_5605e8308565e088`与慧眼task`00371386dabb4b04aa7aeaea8b493772`。
视频先于备用OpenAPI重定向安全补丁录制；实际MiniMax、前端与LAS执行路径未变，
备用补丁另由离线伪凭据负控和回归验证，不声称录像覆盖备用在线调用或逐文件同一快照。
前轮视频与旧WorkflowRun截图保留原封存，不纳入本次新材料/提交白名单。

完整OpenAPI配置仍优先于本机MiniMax配置，鉴权模型调用拒绝重定向，避免Bearer被转发至其他目标；
错误经过既有安全脱敏路径。备用分支实际在线调用未作为本轮验收结论。
本次范围为合成LAS登记/检查、AI侧栏及页面集成，未执行模型预测或井震标定。
配套Gaia PR与材料线上链接：待本人提交后补充。用户自行提交，目前无暂存、commit、push、PR、合并或外发。
