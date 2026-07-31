# 井震数据处理与智能解释平台

这是一个可迁移的本地井震平台发布包，包含数据准备、测井/地震可视化、井震空间对齐、下游预测解释，以及 FaultSeg 三维断层分割和 Seismic Surface Seg 地层分割推理。平台只执行已集成模型的推理，不要求重新训练。

## 目录说明

- `src/well_seismic`：平台后端、预处理、适配器与任务编排
- `frontend`：Vue 3 可视化前端
- `configs`：平台及各模型的可迁移配置
- `接口模型/faultSeg-main`：FaultSeg 代码和推理权重
- `接口模型/seismic_surface_seg`：地层分割代码和推理权重
- `接口模型/cigvis-main/cigvis-main`：CIGVis 可视化库
- `data`：本机数据入口，真实数据默认不提交
- `model_outputs`、`输出结果`：运行结果目录，内容默认不提交
- `models/manifest.json`：模型文件大小和 SHA-256 清单

## 环境要求

- Windows 10/11 或主流 Linux
- Python 3.11
- Node.js 20
- Git 与 Git LFS
- 完整安装建议至少 16 GB 内存；运行 1.2 GB Mask2Former 权重建议使用独立 GPU

所有代码和模型路径均相对于仓库根目录解析，不依赖原电脑的 Conda 路径。项目数据、缓存、输出和 `.env` 密钥已从发布范围中隔离。

## Windows 一键安装

先安装 Python 3.11、Node.js 20、Git 和 Git LFS，然后双击 `安装环境.bat`。也可在 PowerShell 中执行：

```powershell
git lfs install
git lfs pull
.\scripts\setup.ps1
.\scripts\run.ps1
```

浏览器访问 <http://127.0.0.1:8000>。如果 PowerShell 限制脚本执行，可继续使用仓库内的 `.bat` 入口。

## Linux / macOS

```bash
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh
./scripts/run.sh
```

也可以先用 `conda env create -f environment.yml` 创建基础环境，再在该环境中执行安装脚本。

## 配置与数据

1. 将 `.env.example` 复制为 `.env`。GLM 助手不是核心推理必需项，默认关闭；只有启用时才填写服务端密钥。
2. 在平台中选择本机测井/地震数据的绝对路径，或把数据放入 `data`。数据不应提交到 GitHub。
3. 修改 `configs/*.yaml` 时优先使用相对仓库根目录的路径，不要写个人用户名、盘符或 Conda 环境绝对路径。
4. 用 `python tools/verify_release.py` 校验模型是否完整。

## Docker

确认 Git LFS 权重已经下载后运行：

```bash
docker compose up --build
```

服务地址为 <http://127.0.0.1:8000>。宿主机 `data` 会只读挂载到容器 `/data`，输出写回仓库输出目录。GPU 推理需要额外安装 NVIDIA Container Toolkit 并按部署环境扩展 Compose 配置。

## 上传 GitHub

SurfaceSeg 大权重必须使用 Git LFS。首次发布建议按以下顺序：

```bash
git init
git lfs install
git add .
git lfs ls-files
git commit -m "Initial portable platform release"
git branch -M main
git remote add origin https://github.com/<owner>/<repository>.git
git push -u origin main
```

克隆端必须执行 `git lfs pull`。更详细的模型说明见 `MODEL_WEIGHTS.md`。

## 验证

```bash
python tools/verify_release.py
pytest -q
cd frontend
npm ci
npm run build
```

CI 会验证 Python 测试、前端构建、仓库结构和运行时导入。由于 CI 默认不下载 1.2 GB 权重，CI 接受合法的 LFS pointer；本地完整自检仍会检查实际权重。

## 许可

第三方许可见 `THIRD_PARTY_NOTICES.md`。其中 FaultSeg 使用 CC BY-NC 4.0，仅限非商业用途。平台自研代码在公开发布前仍需由仓库所有者选择并添加根目录 `LICENSE`。

