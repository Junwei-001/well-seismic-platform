# P21 开发折离线验收记录（2026-11-17）

## 范围与来源

- 接入基线：团队 `full-runtime-sixtrack-reference` 的 `e7a85c7fbefc29ac701a63a3be9a6e9dd1249859`。
- 外部资源包：`huiyan-sixtrack-assets-20261117.zip`，SHA-256 `036a602bca5ccf6d06458bb5cc4494768228521c1af2a8db2c7dfc04ab574455`。包内载荷清单 118 项，仅 21 项在本轮可用并通过哈希核验；缺少的 97 项不能标为已安装。
- 扩展包只暴露 P21 和断层轻量模型两个入口；代码仓库默认接口模式，不附带载荷。隔离验收环境在核验后另行安装私有载荷并临时启用运行时。实际完成盖亚访客全流程的是 P21。P21 限 Volve 已封存开发折 0–4 的条件重建，不接收任意新工区输入，也不提供完整三维体积预测。
- DeepSeek 测试配置只存在隔离盖亚实例的私有运行环境；源码、前端和本仓库没有 API 密钥。请求的 provider ID 为 `deepseek-chat`，测试 API 返回的模型字段为 `deepseek-flash`，不能据此声称服务商内部部署了某个未核实版本。

模型载荷不随 Git 提交。复现时先核对资源包 SHA-256，再将包内
`models/task-models/sixtrack_reconstruction_p21/p21-development-prepared-v1/payload/`
与 `models/task-models/sixtrack_fault_local_logistic_audited_v2/p78-retain-incumbent-v1/payload/`
复制到本仓库同名目录。不要复制历史 `model_outputs/` 当作本次任务产物。扩展插件需在
隔离模型解释器中登记 entry point，随后以各模型的 `manifest.json` 校验每个载荷的
大小和 SHA-256；缺项时运行器必须拒绝注册。该资源包不是可直接用于生产的新区域数据包。

## 已实测

| 环节 | 实测结果 |
|---|---|
| P21 CPU 原生 worker | 五个开发折全部运行；每折 2,048 个查询点，共 10,240 个。fold 4 与资源包历史数组的预测值最大绝对差约 `1.94e-16`，并非逐位相同。 |
| 自然语言工具调用 | 隔离访客在盖亚对 DeepSeek 发出宽任务，模型自主查询慧眼动作目录、项目列表，选择 P21 折 0 并提交原生任务。首次回复时任务仍在运行，正确报告“未完成”；随后同会话查询到完成状态与 DVC 版本。 |
| 最新代码的真实入口 | 访客经盖亚 `project_create → p21_reconstruction → visualization → dvc_get`，全新任务 `j_c6978ba9e3b99251` 原生状态 completed；DVC `dv_90f4ce0888cf0c948baae2dedab4992f` 为 `archived`，版本含任务摘要与四个唯一结果工件，共 5 个文件。无源代码 Git 版本历史（DVC `no_scm`）。 |
| 盖亚内页面 | 登录访客从已完成任务进入慧眼结果页；Edge 中显示 2,048 个查询点，可切到 X–Y 视角、重置三维视角，并在悬停时读取坐标和预测值。页面就绪标记和拥有者范围均由服务端控制。 |
| 回执与安全边界 | `visualization` 只在真实交互 HTML 通过检查后返回 `viewer_available=true`；模型摘要仍剥除内部 URL。其他账号无法凭任务 ID 读取所属页面。重复工件路径只归档一次，声明项数量上限不放松。 |
| DeepSeek 再核验 | 同一隔离访客会话通过 `huiyan_task_status` 与两次 `huiyan_execute` 回查上述全新任务；模型收到原生完成、DVC 已归档以及交互页可用的独立回执。该轮是回查，不是由模型发起本次全新执行。 |
| 回归测试 | `test_huiyan_harness.py`、`test_huiyan_archive_recovery.py`、`test_gaia_ui.py` 共 48 项：45 项在隔离工作区短临时目录通过，3 项真实 DVC 恢复测试在专用短路径通过。Windows 沙箱的默认长临时路径使 DVC 文件锁失败，默认网络隔离也会阻断 TestClient 的本地 socketpair；测试进程按各自边界隔离运行，未修改生产数据。 |

最初的自然语言宽任务发生在页面回执修订前。一次中间任务因整理隔离工作树时误删了未跟踪的本地预测文件，原生状态和 DVC 归档仍保留，但再次查看页面失败；已保留该失败回执。随后创建全新任务并重新完成执行、归档、DeepSeek 工具回查和浏览器交互验证。不同任务的证据不合并成一次未发生过的单轮调用。隔离运行目录保存回执与截图；不作为仓库内业务数据或训练样本提交。

## 仍待环境和资产

当前 Windows 主机没有非生产 Linux/WSL/Docker 环境。完整参考分支依赖 Linux 文件锁等系统调用，因此尚未在该分支原样部署。除 P21 与断层轻量模型之外的载荷缺失，断层模型也尚未完成盖亚访客端到端验证；不能宣称六赛道全部验收、模型科学效果已通过或可在生产使用。正式 3–5 分钟演示视频及与其一致的两页汇报材料，应在目标环境和最终流程确认后更新。

首次部署时使用服务端私有配置提供 DeepSeek 密钥与慧眼服务令牌，核对动作目录、真实推理、归档哈希、普通账号页面及跨账号拒绝。Windows 隔离测试若启用 DVC，可用绝对路径 `WELL_SEISMIC_DVC_ROOT` 指向受 ACL 保护的短目录，避免路径长度门禁；不得把服务令牌放在该根目录的可归档输入区。
