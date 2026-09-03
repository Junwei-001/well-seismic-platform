# 本机任务状态与输出入口

接口版不携带历史任务、SQLite 数据库、预测体或验收产物。运行状态库默认写入
`runtime/state/platform_state.sqlite3`；本目录只作为未来任务输出入口，如需清理可按后端
`/api/v1/system/cache` 合同操作。
