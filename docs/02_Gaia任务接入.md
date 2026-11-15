# 从盖亚调用慧眼：当前接法

本任务包含慧眼页面内 AI、盖亚调用 LAS 数据检查，以及盖亚内原生慧眼工作台和结果跳转。当前 Gaia 适配基于团队 `origin/master` `04b855577a68ea22f387ef3f7347139fad5fd0e4`；使用现有 mission/job ledger，**没有恢复已移除的 WorkflowRun API**。

配套 Gaia PR：`待本人提交后补充`。Gaia 端接口规格真源为 `docs/team/specs/HUIYAN_TASK1_NATIVE_INTEGRATION_SPEC_20261112.md`；当前真实完整实例及准确命令见[测试记录](deliveries/task1-final-20261113/docs/test-results.md)。本目录不附旧WorkflowRun启动方式、视频或截图。

## 当前架构与调用流程

1. Gaia 登录用户在 `/#/chatbox/huiyan` 上传 LAS；也可在 `/#/chatbox/huiyan/workbench` 内操作原生慧眼工作台。
2. 独立 Star 插件 `my_plugins/gaia_huiyan` 通过现有 `route_access.normal_route(scope="own")` 注册路由，身份取自认证后的 `g.username`。文件、operation_id、job 和结果记录绑定此身份，不接受浏览器指定 owner。
3. 插件保存输入和状态，以服务凭证及 `SHA256(username)` 调用慧眼受保护接口。慧眼实际执行已有数据准备、LAS 解析与封存；没有用模型文本或模拟结果代替执行。
4. 只有 run、task、输入摘要、报告摘要及数量合同校验通过才标记完成。`HuiyanProducer` 把真实状态投影到现有 `JobProjector`，producer 为 `huiyan_las`，公开任务标识为 `j_<16位小写十六进制>`。
5. 盖亚任务卡片/报告跳转 `/#/chatbox/huiyan/workbench?job_ref=<本人任务>`，页面先查询本人 job，再定位对应慧眼 task。刷新恢复选中任务；iframe 消息同时校验 origin 和 source。

本次 Gaia 改动集中在插件、mission producer 和 Dashboard 页面；不修改 AstrBot Core 或 `Chat.vue`，不改动原科学 MCP 的井属性/断层预测能力。慧眼前端默认使用新代理前缀，仅在明确的旧 pathname 下保留历史演示兼容。

## 当前接口

成功响应为 `{status:"ok",data:...}`；失败返回固定安全错误，不含服务路径、凭据或上游原文。

| Gaia 同源接口 | 用途 |
| --- | --- |
| `GET /api/plug/huiyan/status` | 连接配置是否齐全、LAS 检查范围；不返回配置值 |
| `POST /api/plug/huiyan/jobs` | multipart：非空 LAS `file`（不超过 20 MiB）和安全标识 `operation_id`（8–80 字符；前端生成 UUID） |
| `GET /api/plug/huiyan/jobs` | 当前用户最近 100 条任务 |
| `GET /api/plug/huiyan/jobs/<job_ref>` | 本人任务状态和已核验结果 |
| `GET /api/plug/huiyan/jobs/<job_ref>/report` | 本人已完成任务报告；未完成返回 409 |
| `GET /api/plug/huiyan/ui/`、`GET/POST .../ui/<path>` | 已登录用户的受控原生页面与 API 代理 |

慧眼后端合同仍为 `/api/v1/gaia/capabilities`、`POST/GET /api/v1/gaia/runs/<run_id>` 和 `/api/v1/gaia/ui/`。内部 `run_<16hex>` 是当前账号与 operation_id 派生的幂等键，不是旧 WorkflowRun 实例。相同账号/标识/内容复用任务，不同内容返回 409；其他账号无法读取任务、报告、原生结果、清单或 AI 会话。

状态文件由 `JobFiles` 单独写入，任务台账按 revision/digest 同步。恢复时使用原内部 run_id 查询未终止任务，不重跑已完成任务。当前总并发上限 2、未终止任务总数上限 16、每账号未结束任务上限 4；等待超过 120 秒明确失败并提示慧眼可能仍在执行。当前没有取消、重试或预测按钮，不把超时当成功。

## 后端配置与安装前提

| 服务 | 变量 | 含义 |
| --- | --- | --- |
| 慧眼 | `WELL_SEISMIC_GAIA_SERVICE_TOKEN_FILE` | 仓库外或 Git 忽略目录中的私有服务凭证文件 |
| Gaia | `GAIA_HUIYAN_SERVICE_TOKEN_FILE` | 同一服务凭证在 Gaia 主机上的文件路径 |
| Gaia | `GAIA_HUIYAN_BASE_URL` | 慧眼服务地址；HTTPS 或同机 loopback HTTP，不带 userinfo/path/query/fragment |

两端凭证文件内容一致，仅服务账号可读；不复用 MiniMax key、登录密码或 JWT secret，不把文件加入仓库。变量须进入实际后端进程，浏览器、Vite 配置和截图不保存密钥。网关不向慧眼转发用户 Cookie/Authorization，禁用携带服务凭证的重定向，并对修改请求校验同源。

Gaia 的正常插件发现目录是运行数据根下的 `data/plugins`。管理员须按已有插件部署流程安装/链接 `my_plugins/gaia_huiyan`，启用 `gaia-huiyan`，同时部署本次 mission producer 并启用 `gaia-missions`；仅把源码放在 `my_plugins` 不保证自动加载。任务台账未就绪时入口返回 503。两端前端均需构建，慧眼需使用支持 `/api/plug/huiyan/ui/` 的配套 dist。不得覆盖正在使用的运行数据目录。

MiniMax 仍由慧眼后端复用已有 Gaia 配置（见仓库 README）；`$NAME` / `${NAME}` 引用必须在慧眼服务进程环境中可解析，缺失、空或非法值拒绝使用，不回写源配置。任务服务凭证与模型凭证各自独立。

原生代理只开放 LAS 检查结果、测井预览、清单和 AI；原桌面路径、缓存、全局项目等接口不作为多用户接口公开。慧眼运行于 `interfaces_only`，本次没有启用 SEG-Y 处理、井震标定或预测模型。

## 完整本地实例验收与复现

Gaia工具`scripts/e2e/huiyan_full_instance.py`运行真实main.py/Core/PluginManager，自动加载必要的gaia-auth-plus、gaia-huiyan和gaia-missions。其独立数据根只允许当前checkout的_tmp专用后代目录，拒绝symlink/junction，fresh/resume/adopt明确区分；私有ACL仅当前用户、SYSTEM、Administrators。它不是生产部署工具，不替用户更改线上配置或复制生产账号数据库。

本轮本机Gaia为`http://localhost:3175/app/#/`、慧眼为`http://127.0.0.1:1726`。两个本地普通账号通过正常注册与验证码/登录，官方console邮件后端不向外发邮件。8项注册、31项HTTP和9项浏览器分别留证，没有fixture认证。HTTP脚本使用正常登录签发JWT，主动构造Bearer或正常Cookie名的请求头；浏览器另经页面真实登录接收原Secure/HttpOnly Cookie，未手动注入token并实测iframe。脚本HTTP不替代浏览器Cookie行为验证。对应编号见[测试记录](deliveries/task1-final-20261113/docs/test-results.md)，复现操作见[验收说明](deliveries/task1-final-20261113/docs/acceptance.md)。

录制另使用新job`j_5605e8308565e088`/task`00371386dabb4b04aa7aeaea8b493772`，不能拿HTTP或早一次浏览器编号代替。仅单文件合成LAS登记/检查与页面集成，不把validated解释为科学预测或井震标定成功，不声称团队正式上线。

任务1新版视频、录制后备用OpenAPI安全补正及独立审查边界见[最终交付说明](deliveries/task1-final-20261113/README.md)。前轮封存保留原处，不作为当前提交材料；线上PR链接待本人提交后补充。
