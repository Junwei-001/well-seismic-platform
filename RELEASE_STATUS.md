# 发布状态

生成日期：2026-07-30

## 已完成

- 已从工作目录生成独立 GitHub 发布目录，未复制原始生产数据、缓存、虚拟环境和密钥。
- 已移除 SurfaceSeg 配置中绑定个人 Conda 环境的绝对路径。
- 已统一提供 Python 3.11、Node.js 20、Git LFS 的环境声明和跨平台安装/启动脚本。
- 已加入 Dockerfile、Docker Compose、GitHub Actions 和发布自检工具。
- 已登记并校验 4 个推理权重；SurfaceSeg 大权重由 Git LFS 管理。

## 验证结果

- Python：`82 passed`
- Vue/TypeScript：生产构建通过，npm 审计为 `0 vulnerabilities`
- API：`GET /api/v1/health` 返回 HTTP 200
- 模型：4 个权重的文件大小和 SHA-256 与清单一致
- Docker：`docker compose config` 解析通过
- 敏感信息：未发现真实 `.env`、GLM 密钥或原电脑用户目录硬编码

## 发布前由仓库所有者决定

- 公开仓库还是私有仓库
- 自研平台代码采用的根许可证
- GitHub LFS 额度是否足以承载约 1.29 GiB 发布包中的 SurfaceSeg 权重
- FaultSeg 的 CC BY-NC 4.0 非商业限制是否符合发布目的

