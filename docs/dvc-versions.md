# 慧眼数据版本与恢复（DVC）

`well_seismic.dvc_store.DvcVersionStore` 将可信服务端输入、参数文件和结果文件封存为独立的数据版本。每个新版本实际执行 `dvc init --no-scm`、`dvc add data` 和 `dvc status --json`，保存真实 `.dvc` 指针和内容寻址缓存；缺少 DVC 或命令失败会返回错误，不生成成功凭据。重复同内容请求可在验证完整性后复用既有版本，不冒称重新执行过归档命令。

## 安装与状态核对

DVC 由 `pyproject.toml` 的 `agent` extra 声明（`dvc==3.67.1`）。基础接口的 `.[web]`、`requirements-interface.txt` 或 BAT 安装并不包含它。现有开发环境复用项目解释器；首次配置完整任务1环境时执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[web,agent]"
.\.venv\Scripts\python.exe -m dvc --version
```

服务端与该命令须使用同一解释器。认证后的 `dvc_status` 动作返回实际 `available`、`version`、`mode` 和 `git_revision_history`；命令不可执行时 `available=false`，不生成成功归档回执。安装/状态可用仅是前提，还须验证 A/B 归档、比较和恢复。

## 认证和边界

- `subject` 必须来自服务端已认证身份，不能信任请求体中的用户标识。owner + project 的完整 SHA-256 决定私有目录，所有查询、比较、恢复和内部路径解析都重新限制到这个范围。
- `files` 的来源必须由业务层根据任务或数据集 ID 查找；构造 store 时显式配置 `allowed_source_roots`。禁止将用户任意绝对路径传入本模块。
- 拒绝符号链接、Windows junction/reparse point、路径穿越、大小写冲突、凭证字段和绝对路径元数据。源文件复制前后校验文件身份、大小、修改时间与 SHA-256；版本读取校验凭据、DVC 指针、配置、全部文件和未登记文件。
- 运行目录应由服务账户拥有；Windows ACL 应随私有数据目录部署。应用账号隔离不代替操作系统账号隔离。
- 公共凭据不包含本地绝对路径或 owner hash。`resolve_files` / `resolve_restored_files` 仅供可信服务端内部使用，其返回值不得作为 API 响应、日志或 LLM 上下文。

## 稳定接口

```python
store = DvcVersionStore(private_root, allowed_source_roots=[uploads, results])
store.status()  # available / version / mode / git_revision_history
version_a = store.archive(subject, project_id, {"inputs/well.las": source}, metadata)
version_b = store.archive(subject, project_id, files_b, metadata_b,
                          parent_revision_id=version_a["revision_id"])
store.list_versions(subject, project_id)
store.get(subject, project_id, version_a["revision_id"])
store.compare(subject, project_id, version_a["revision_id"], version_b["revision_id"])
restored = store.restore_owned(subject, project_id, version_a["revision_id"])
private_inputs = store.resolve_restored_files(subject, project_id, restored["restore_id"])
```

`archive` 的最少文件数为 1，默认至多 128 个、共 2 GiB；元数据必须是有限 JSON，最多 64 KiB。成功凭据含版本、父版本、文件 SHA-256、元数据、DVC 版本、DVC 指针与配置哈希。`compare` 按实际封存文件的哈希及元数据比较增加、删除、变化项目。

`restore_owned` 在新的恢复目录通过真实 `dvc checkout data.dvc` 从该版本缓存取回数据，再逐文件验证 SHA-256；缺失缓存、损坏对象或权限不符会失败。恢复不会覆盖现行输入或旧封存。业务层可以用内部 resolver 将恢复数据登记为新数据集、重新执行，再把新结果与旧版比较。单独恢复成功不代表模型计算已经重跑。

## 归档去重与故障恢复

Harness 在读取前检查工件数量与总字节限额，复用流式 `file_sha256`，再以文件逻辑名和摘要、公开元数据、父版本以及 owner/project 生成完整 64 位十六进制 `archive_key`。此字段由可信后端构造，不接受 AI 或浏览器指定。metadata 不含该字段的普通 store 调用仍新建独立版本。

每个 key 使用独立的跨进程 OS 文件锁，SQLite 只负责短事务读取或保存回执；DVC 执行和文件校验期间不持有全库写锁。不同 key 可以独立协调，进程退出后 OS 会释放锁。相同 key 的等待超时明确返回 `archive_in_progress`，不冒称完成。

含 key 的版本 ID 使用其前 32 位，读取时继续核验完整 key、元数据、父版本及内容，截断碰撞不能复用。`find_archive(subject, project_id, archive_key)` 仅解析并校验该确定性候选，不调用 `list_versions` 或哈希无关历史；旧版本损坏不会阻塞无关新版本，匹配版本损坏仍拒绝复用。发布使用独立临时目录，未完成的暂存不会占死该 key。若版本已发布但应用回执保存中断，重试查回并验证该版本，避免重复发布。

自动及手动任务归档共用完整工件清单；恢复到新目录后仍需逐文件核验。历史失败执行与新恢复版本通过明确的 execution/revision 关联表达，不能把首轮失败改写成首轮成功，详见 [Harness 回执契约](huiyan-harness.md#入口与回执)。

## 运行与明确限制

当前实现是本机 DVC 内容缓存和应用层不可变版本索引，显式使用 **no-SCM**，不会进行 Git add、Git commit 或 push。本轮数据/结果版本追踪验收覆盖本地归档、差异、恢复和完整性；它不宣称具备 Git 历史、远程 DVC push/pull、多机灾备或模型训练功能。团队共享 remote 属于后续部署配置，当前没有团队远程地址或凭据，不将本地成功表述为远程部署成功。

未来接入共享存储时，remote 及其凭证只由管理员在服务端私有配置中管理，不进入 AI 动作参数、前端或代码仓库。现有版本的 `.dvc/config` 已纳入封存哈希，不能直接对旧版本执行 `dvc remote add` 或改写配置；远程迁移需在新工作区执行并重新核对原始与恢复文件哈希，保留原封存。

Windows 未启用长路径时请选择较短的私有根目录和逻辑文件名；超过安全长度会返回 `path_limit`。内部布局不是接口契约，调用方必须使用 resolver。

测试命令（使用已安装 DVC 的项目 Python）：

```powershell
.venv/Scripts/python.exe -B -m pytest tests/test_dvc_store.py -q -p no:cacheprovider
.venv/Scripts/python.exe -B -m pytest tests/test_huiyan_archive_recovery.py -q -p no:cacheprovider
```

测试覆盖真实 A/B 版本、差异、DVC 缓存恢复、旧版不改写、跨身份拒绝，以及缓存缺失/损坏、指针/凭据/配置/文件篡改、越界来源、文件名、凭证元数据和失败不发布。真实 CLI 用例在未安装 DVC 时会标记 skipped，不算验收通过；Windows 账号未获创建 symlink 权限时，该测试单独 skipped，同时保留 reparse point 负控测试。
