import { isWorkflowResult, type BackgroundTask, type WorkflowResult } from "./api";

export interface GaiaTaskChain {
  selected: BackgroundTask;
  preparation: BackgroundTask;
  registration?: BackgroundTask;
  sample?: BackgroundTask;
  snapshotId: string;
}

function reference(values: unknown[], label: string): string {
  const supplied = values.filter(value => value !== undefined && value !== null && value !== "");
  if (!supplied.length || supplied.some(value => typeof value !== "string" || !/^[0-9a-f]{32}$/.test(value))) {
    throw new Error(`${label}缺少有效的原生任务引用`);
  }
  if (new Set(supplied).size !== 1) throw new Error(`${label}与原生结果来源不一致`);
  return supplied[0] as string;
}

function sourceSnapshot(task: BackgroundTask): string {
  const result = task.result as WorkflowResult;
  return reference([
    task.snapshot_id, task.request?.source_snapshot_id, result.source_snapshot_id,
    result.registration?.source_snapshot_id, result.data_snapshot?.snapshot_id,
    result.prepared_view?.source_snapshot_id,
  ], "源数据快照");
}

/** Resolve only existing owner-checked task resources; never manufacture results.
 * Stage the whole chain before App commits any coupled workflow state.
 */
export async function loadGaiaTaskChain(
  taskId: string,
  getTask: (id: string) => Promise<BackgroundTask>,
  current: () => boolean = () => true,
): Promise<GaiaTaskChain | null> {
  async function fetchTask(id: string, kind?: string): Promise<BackgroundTask> {
    reference([id], "任务");
    const task = await getTask(id);
    if (task.task_id !== id || (kind && task.task_type !== kind)) throw new Error("返回的原生任务与所选结果不匹配");
    if (task.status !== "completed" || !isWorkflowResult(task.result)) throw new Error("原生任务尚未完成或缺少可恢复结果");
    return task;
  }
  const selected = await fetchTask(taskId);
  if (!current()) return null;
  if (selected.task_type === "data_preparation") {
    const result = selected.result as WorkflowResult;
    const snapshotId = reference([selected.snapshot_id, result.data_snapshot?.snapshot_id, selected.task_id], "数据准备快照");
    return { selected, preparation: selected, snapshotId };
  }
  if (!["well_tie", "sample_building"].includes(selected.task_type || "")) throw new Error("当前原生任务类型尚不支持在此工作台恢复");
  const snapshotId = sourceSnapshot(selected);
  let registration = selected;
  if (selected.task_type === "sample_building") {
    const result = selected.result as WorkflowResult;
    if (!result.matching || !result.prepared_view) throw new Error("融合任务缺少原生 PreparedView 回执");
    const registrationId = reference([result.registration_task_id, selected.request?.registration_task_id], "标定任务");
    registration = await fetchTask(registrationId, "well_tie");
    if (!current()) return null;
    if (sourceSnapshot(registration) !== snapshotId) throw new Error("融合视图和标定任务不属于同一个源快照");
  }
  if (!(registration.result as WorkflowResult).registration) throw new Error("标定任务缺少原生标定结果");
  const preparation = await fetchTask(snapshotId, "data_preparation");
  if (!current()) return null;
  const preparationResult = preparation.result as WorkflowResult;
  if (reference([preparation.snapshot_id, preparationResult.data_snapshot?.snapshot_id, preparation.task_id], "数据准备快照") !== snapshotId) {
    throw new Error("数据准备结果与任务源快照不一致");
  }
  return { selected, preparation, registration, snapshotId,
    ...(selected.task_type === "sample_building" ? { sample: selected } : {}) };
}
