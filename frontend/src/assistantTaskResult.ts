import type { AssistantExecution } from "./api";

export interface AssistantTaskReference {
  projectId: string;
  taskId: string;
}

// IDs are navigation hints, never authority. The same-origin backend rechecks
// the current account and both resources before a native result is loaded.
export function validAssistantTaskReference(value: unknown): value is AssistantTaskReference {
  if (!value || typeof value !== "object") return false;
  const ref = value as Partial<AssistantTaskReference>;
  return typeof ref.projectId === "string" && /^hp_[0-9a-f]{32}$/.test(ref.projectId)
    && typeof ref.taskId === "string" && /^[0-9a-f]{32}$/.test(ref.taskId);
}

export function assistantTaskResultReference(receipt: AssistantExecution): AssistantTaskReference | null {
  if (receipt.status !== "completed"
    || !["data_prepare", "registration", "sample_build", "task_status", "visualization"].includes(receipt.operation)) return null;
  const reference = { projectId: receipt.project_id, taskId: receipt.task_id };
  if (!validAssistantTaskReference(reference)) return null;
  const result = receipt.result;
  if ((result?.project_id !== undefined && result.project_id !== reference.projectId)
    || (result?.task_id !== undefined && result.task_id !== reference.taskId)) return null;
  return reference;
}
