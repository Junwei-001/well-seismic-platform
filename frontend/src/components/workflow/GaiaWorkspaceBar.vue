<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { isLegacyGaiaEmbedded } from "../../gaiaEmbed";
import { validAssistantTaskReference, type AssistantTaskReference } from "../../assistantTaskResult";

const props = defineProps<{ restoreResult?: (value: { jobRef: string; taskId: string }) => Promise<boolean> }>();
const emit = defineEmits<{ completed: [value: { jobRef: string; taskId: string }]; loading: [] }>();
const file = ref<File | null>(null);
const busy = ref(false);
const error = ref("");
const status = ref("");
const taskId = ref("");
const executionReceipt = ref<any>(null);
const history = ref<Array<{ ref: string; status: string; created_at: number }>>([]);
const selectedRun = ref("");
const controller = new AbortController();
let generation = 0;
let pending: { file: File; id: string } | undefined;
const assistantResultRuns = new Map<string, { operationId: string; jobRef?: string }>();
const statusLabel = (value: string) => ({ submitted: "已提交", working: "执行中", completed: "已完成", failed: "未完成", interrupted: "执行中断" }[value] || "状态待确认");
const dateLabel = (value: number | string) => new Date(typeof value === "number" ? value * 1000 : value).toLocaleString();
const validRef = (id: string) => (isLegacyGaiaEmbedded ? /^run_[0-9a-f]{16}$/ : /^j_[0-9a-f]{16}$/).test(id);
function normalizeJob(value: any) {
  if (!isLegacyGaiaEmbedded) return value;
  const artifact = value.artifacts?.find((item: { kind: string }) => item.kind === "huiyan_preparation_report");
  return { ...value, ref: value.run_id, created_at: typeof value.created_at === "number" ? value.created_at / 1000 : value.created_at,
    status: value.status === "succeeded" ? "completed" : ["planning", "running", "queued", "pending", "created"].includes(value.status) ? "working" : value.status,
    task_id: artifact?.data?.task_id, report: artifact ? { ...artifact.data, validation_status: artifact.validation_status } : null };
}

async function gaia(path: string, init?: RequestInit) {
  const base = isLegacyGaiaEmbedded ? "/api/workflow-runs" : "/api/plug/huiyan/jobs";
  const response = await fetch(`${base}${path}`, { ...init, signal: controller.signal, credentials: "same-origin" });
  const payload = await response.json();
  if (!response.ok || payload.status !== "ok") throw new Error("盖亚请求失败，请检查登录和服务状态");
  return payload.data;
}

async function loadHistory() {
  const result = await gaia("");
  const items = result.items || [];
  if (!controller.signal.aborted) history.value = (isLegacyGaiaEmbedded ? items.filter((item: { scenario_id: string }) => item.scenario_id === "huiyan_las_preparation") : items).map(normalizeJob);
}

async function showNativeResult(value: { jobRef: string; taskId: string }, current: number): Promise<boolean> {
  if (props.restoreResult) {
    const restored = await props.restoreResult(value);
    if (current !== generation || controller.signal.aborted) return false;
    if (!restored) error.value = "无法恢复原生任务结果，请刷新后重试或检查任务来源";
    return restored;
  }
  // Preserve the existing event contract for standalone bridge consumers.
  emit("completed", value);
  return true;
}

async function followRun(id: string, current: number, expected?: AssistantTaskReference) {
  if (!validRef(id)) throw new Error("盖亚任务编号无效");
  selectedRun.value = id;
  taskId.value = "";
  executionReceipt.value = null;
  emit("loading");
  function announceSelection() {
    const url = new URL(window.location.href);
    url.searchParams.set(isLegacyGaiaEmbedded ? "run_id" : "job_ref", id);
    window.history.replaceState(null, "", url);
    window.parent.postMessage(isLegacyGaiaEmbedded ? { type: "huiyan:run", runId: id } : { type: "huiyan:job", jobRef: id }, window.location.origin);
  }
  // An AI receipt must match the owner-checked job before changing the parent URL.
  if (!expected) announceSelection();
  const deadline = Date.now() + 180_000;
  while (current === generation && !controller.signal.aborted) {
    const run = normalizeJob(await gaia(`/${id}`));
    if (current !== generation || controller.signal.aborted) return;
    if (run.ref !== id) throw new Error("任务返回结果不匹配，请重新打开任务");
    if (expected && run.kind !== "harness") throw new Error("任务返回结果不匹配，请重新打开任务");
    status.value = statusLabel(run.status);
    if (run.kind === "harness" && run.receipt && !["submitted", "working"].includes(run.status)) {
      if (expected && (run.operation !== "task_status" || run.receipt.operation !== "task_status"
        || run.project_id !== expected.projectId || run.receipt.project_id !== expected.projectId
        || run.task_id !== expected.taskId || run.receipt.task_id !== expected.taskId
        || run.receipt.result?.task?.task_id !== expected.taskId)) {
        throw new Error("任务返回结果不匹配，请重新打开任务");
      }
      executionReceipt.value = run.receipt;
      status.value = `${run.operation} · ${run.receipt.status}`;
      if (run.receipt.error) error.value = run.receipt.error.message;
      if (expected && run.receipt.result?.task?.status !== "completed") {
        assistantResultRuns.delete(`${expected.projectId}:${expected.taskId}`);
        throw new Error("原生任务尚未完成或执行失败，请刷新执行状态后再查看结果");
      }
      if (run.receipt.status === "completed" && /^[0-9a-f]{32}$/.test(run.task_id || "")) {
        if (expected) announceSelection();
        taskId.value = run.task_id;
        if (!await showNativeResult({ jobRef: id, taskId: taskId.value }, current)) return false;
      }
      await loadHistory();
      return Boolean(taskId.value);
    }
    if (run.status === "completed") {
      const report = run.report;
      if (!/^[0-9a-f]{32}$/.test(run.task_id || "") || report?.task_id !== run.task_id || report.validation_status !== "validated") throw new Error("缺少已核验的慧眼结果");
      taskId.value = run.task_id;
      if (!await showNativeResult({ jobRef: id, taskId: taskId.value }, current)) return false;
      await loadHistory();
      return Boolean(taskId.value);
    }
    if (!["submitted", "working"].includes(run.status)) throw new Error("任务未完成，请检查输入文件和后端服务后重试");
    if (Date.now() > deadline) throw new Error("等待结果超时，任务可能仍在执行；请从任务历史重新打开");
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

async function openRun(id: string) {
  const current = ++generation;
  busy.value = true;
  error.value = "";
  try { await followRun(id, current); }
  catch (cause) { if (current === generation && !controller.signal.aborted) error.value = cause instanceof Error ? cause.message : "读取失败"; }
  finally { if (current === generation && !controller.signal.aborted) busy.value = false; }
}

async function openAssistantResult(reference: AssistantTaskReference): Promise<boolean> {
  if (busy.value || controller.signal.aborted) return false;
  if (isLegacyGaiaEmbedded || !validAssistantTaskReference(reference)) {
    error.value = "当前入口或任务引用无效，请从盖亚慧眼工作区打开";
    return false;
  }
  // Copy only the checked IDs. No receipt URL, credentials or owner is forwarded.
  const expected = { projectId: reference.projectId, taskId: reference.taskId };
  const key = `${expected.projectId}:${expected.taskId}`;
  const request = assistantResultRuns.get(key) || { operationId: crypto.randomUUID() };
  assistantResultRuns.set(key, request);
  const current = ++generation;
  busy.value = true;
  error.value = "";
  status.value = "正在核对任务并打开结果…";
  try {
    if (!request.jobRef) {
      const response = await fetch("/api/plug/huiyan/actions", {
        method: "POST", credentials: "same-origin", signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ operation_id: request.operationId, operation: "task_status",
          arguments: { project_id: expected.projectId, task_id: expected.taskId } }),
      });
      const payload = await response.json();
      if (current !== generation || controller.signal.aborted) return false;
      if (!response.ok || payload.status !== "ok") throw new Error("无法访问当前任务，请检查登录状态和任务权限");
      if (!validRef(payload.data?.ref || "")) throw new Error("任务返回结果无效，请重试");
      request.jobRef = payload.data.ref;
    }
    if (current !== generation || controller.signal.aborted) return false;
    return Boolean(await followRun(request.jobRef!, current, expected));
  } catch {
    if (current === generation && !controller.signal.aborted) {
      error.value = "无法打开任务结果，请确认任务已完成且属于当前账号，稍后重试";
    }
    return false;
  } finally {
    if (current === generation && !controller.signal.aborted) busy.value = false;
  }
}
defineExpose({ openAssistantResult });

async function start() {
  if (!file.value || busy.value) return;
  if (!/\.las$/i.test(file.value.name) || file.value.size === 0 || file.value.size > 20 * 1024 * 1024) {
    error.value = "请选择非空 LAS 文件，最大 20MB";
    return;
  }
  const current = ++generation;
  busy.value = true;
  error.value = "";
  status.value = "正在上传 LAS…";
  try {
    if (pending?.file !== file.value) pending = { file: file.value, id: crypto.randomUUID() };
    const data = new FormData();
    data.append("file", file.value);
    data.append("operation_id", pending.id);
    let result;
    if (isLegacyGaiaEmbedded) {
      const uploaded = await gaia("/inputs", { method: "POST", body: data });
      result = await gaia("", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario_id: "huiyan_las_preparation", input_artifact_ids: [uploaded.artifact_id] }) });
    } else result = await gaia("", { method: "POST", body: data });
    const run = normalizeJob(result);
    if (current !== generation || controller.signal.aborted) return;
    if (!validRef(run.ref || "")) throw new Error("任务返回结果无效，请检查服务后重试");
    pending = undefined;
    await followRun(run.ref, current);
  } catch (cause) { if (current === generation && !controller.signal.aborted) error.value = cause instanceof Error ? cause.message : "运行失败"; }
  finally { if (current === generation && !controller.signal.aborted) busy.value = false; }
}

onMounted(async () => {
  try {
    await loadHistory();
    if (controller.signal.aborted) return;
    window.parent.postMessage({ type: "huiyan:ready" }, window.location.origin);
    const requested = new URLSearchParams(window.location.search).get(isLegacyGaiaEmbedded ? "run_id" : "job_ref");
    if (requested) await openRun(requested);
  } catch (cause) { error.value = cause instanceof Error ? cause.message : "盖亚工作区加载失败"; }
});
onBeforeUnmount(() => { generation++; controller.abort(); });
</script>

<template>
  <section class="gaia-workspace-bar" aria-label="盖亚慧眼工作区">
    <div class="gaia-workspace-controls">
      <strong>盖亚 · 慧眼工作区</strong>
      <label>上传 LAS <input type="file" accept=".las" :disabled="busy" @change="file = ($event.target as HTMLInputElement).files?.[0] || null" /></label>
      <button :disabled="busy || !file" @click="start">{{ busy ? "正在处理…" : "检查并查看结果" }}</button>
      <label>我的任务 <select v-model="selectedRun" :disabled="busy" @change="openRun(selectedRun)"><option value="" disabled>选择历史任务</option><option v-for="item in history" :key="item.ref" :value="item.ref">{{ dateLabel(item.created_at) }} · {{ statusLabel(item.status) }}</option></select></label>
    </div>
    <details v-if="executionReceipt"><summary>操作回传结果</summary><pre class="execution-result">{{ JSON.stringify(executionReceipt.result, null, 2) }}</pre></details>
    <p v-if="error" class="gaia-error" role="alert">{{ error }}</p>
    <p v-else role="status">{{ status || "在此上传数据、与慧眼 AI 对话，或打开当前账号的任务结果。" }}</p>
  </section>
</template>

<style scoped>
.execution-result { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 260px; overflow: auto; }
.gaia-workspace-bar { position: sticky; top: 0; z-index: 40; padding: 10px 18px; border-bottom: 1px solid #bdd4e0; background: #eff8fc; color: #173f59; font-size: 13px; }
.gaia-workspace-controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
label { display: flex; align-items: center; gap: 6px; }
input { max-width: 220px; }
select { max-width: 250px; }
button { background: #246a91; color: white; border: 0; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
p { margin: 8px 0 0; overflow-wrap: anywhere; }
.gaia-error { color: #b42318; }
</style>
