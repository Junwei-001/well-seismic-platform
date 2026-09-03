# 地层慧眼平台｜独立接口版（无任务权重）

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
