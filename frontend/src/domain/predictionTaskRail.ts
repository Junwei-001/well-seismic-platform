export interface PredictionEnvelope<TPrediction> {
  prediction: TPrediction;
  source_task_id?: string;
  registration_task_id?: string;
  prepared_view_task_id?: string;
}

export interface PredictionRunHistoryEntry<TPrediction> {
  executionTaskId: string;
  result: PredictionEnvelope<TPrediction>;
  updatedAt: number;
}

function parsedTimestamp(value: unknown): number {
  if (typeof value !== "string") return 0;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
}

/**
 * Build one durable result slot per interpretation task. Failed and running
 * attempts never replace the last usable result.
 */
export function collectLatestPredictionRuns<TPrediction extends { task_id: string }>(
  tasks: readonly unknown[],
  allowedTaskIds: readonly string[],
  acceptPrediction: (prediction: TPrediction) => boolean = () => true,
): Record<string, PredictionRunHistoryEntry<TPrediction>> {
  const allowed = new Set(allowedTaskIds);
  const history: Record<string, PredictionRunHistoryEntry<TPrediction>> = {};

  for (const candidate of tasks) {
    if (!candidate || typeof candidate !== "object") continue;
    const task = candidate as Record<string, unknown>;
    if (task.task_type !== "model_prediction" || task.status !== "completed") continue;
    if (typeof task.task_id !== "string" || !task.task_id) continue;
    if (!task.result || typeof task.result !== "object") continue;

    const result = task.result as PredictionEnvelope<TPrediction>;
    const prediction = result.prediction;
    if (!prediction || typeof prediction !== "object") continue;
    if (!allowed.has(prediction.task_id) || !acceptPrediction(prediction)) continue;

    const updatedAt = parsedTimestamp(task.updated_at) || parsedTimestamp(task.created_at);
    const current = history[prediction.task_id];
    if (current && current.updatedAt > updatedAt) continue;
    history[prediction.task_id] = {
      executionTaskId: task.task_id,
      result,
      updatedAt,
    };
  }

  return history;
}

/** Return a clamped scroll position that centers one tab inside its rail. */
export function centeredTaskScrollLeft(
  containerWidth: number,
  scrollWidth: number,
  itemLeft: number,
  itemWidth: number,
): number {
  const maximum = Math.max(0, scrollWidth - containerWidth);
  const centered = itemLeft - (containerWidth - itemWidth) / 2;
  return Math.min(maximum, Math.max(0, centered));
}

/** Return a clamped vertical scroll position that centers one directory item. */
export function centeredTaskScrollTop(
  containerHeight: number,
  scrollHeight: number,
  itemTop: number,
  itemHeight: number,
): number {
  const maximum = Math.max(0, scrollHeight - containerHeight);
  const centered = itemTop - (containerHeight - itemHeight) / 2;
  return Math.min(maximum, Math.max(0, centered));
}
