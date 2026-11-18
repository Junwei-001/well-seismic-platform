<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { isLegacyGaiaEmbedded } from "../../gaiaEmbed";
import { validAssistantTaskReference, type AssistantTaskReference } from "../../assistantTaskResult";

const props = defineProps<{ restoreResult?: (value: { jobRef: string; taskId: string }) => Promise<boolean> }>();
const emit = defineEmits<{ completed: [value: { jobRef: string; taskId: string }]; loading: [] }>();
const file = ref<File | null>(null);
const busy = ref(false);
const assetBusy = ref(false);
const projects = ref<Array<{ id: string; name: string }>>([]);
const selectedProject = ref("");
const newProjectName = ref("");
const fileRole = ref<"log" | "seismic">("log");
const assetStatus = ref("");
const assetError = ref("");
let pendingProject: { name: string; id: string } | undefined;

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
      status.value = statusLabel(run.receipt.status);
      if (run.receipt.error) error.value = "任务未完成，请核查输入后重试。";
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

// Project actions are owned by the logged-in Gaia account. Job receipts remain
// the authority; request identifiers are retained across uncertain retries.
async function projectAction(operation: "project_list" | "project_create", args: Record<string, unknown>, operationId: string) {
  const response = await fetch("/api/plug/huiyan/actions", {
    method: "POST", credentials: "same-origin", signal: controller.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operation_id: operationId, operation, arguments: args }),
  });
  const payload = await response.json();
  if (!response.ok || payload.status !== "ok" || !validRef(payload.data?.ref || "")) throw new Error("工区请求失败，请检查登录状态后重试。");
  const jobRef = payload.data.ref as string;
  const deadline = Date.now() + 30_000;
  while (!controller.signal.aborted && Date.now() < deadline) {
    const job = normalizeJob(await gaia(`/${jobRef}`));
    if (job.ref !== jobRef || job.operation !== operation) throw new Error("工区返回结果不匹配，请重试。");
    if (job.kind === "harness" && job.receipt && !["submitted", "working"].includes(job.status)) {
      if (job.receipt.operation !== operation || job.receipt.status !== "completed") throw new Error("工区操作未完成，请稍后重试。");
      return job.receipt.result || {};
    }
    if (!["submitted", "working"].includes(job.status)) throw new Error("工区操作未完成，请稍后重试。");
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  throw new Error("工区操作仍在处理中，请稍后重试。");
}

async function loadProjects() {
  if (isLegacyGaiaEmbedded) return;
  const found: Array<{ id: string; name: string }> = [];
  let offset = 0;
  for (let page = 0; page < 100; page++) {
    const result = await projectAction("project_list", { offset, limit: 32 }, crypto.randomUUID());
    if (!Array.isArray(result.projects)) throw new Error("无法读取工区列表，请重试。");
    found.push(...result.projects.filter((item: any) => /^hp_[0-9a-f]{32}$/.test(item?.id || "") && typeof item?.name === "string"));
    if (result.next_offset == null) break;
    if (!Number.isInteger(result.next_offset) || result.next_offset <= offset) throw new Error("无法读取工区列表，请重试。");
    offset = result.next_offset;
    if (page === 99) throw new Error("工区列表过长，请缩小查询范围。");
  }
  if (controller.signal.aborted) return;
  projects.value = found;
  if (!found.some((project) => project.id === selectedProject.value)) selectedProject.value = found[0]?.id || "";
}

async function refreshProjects() {
  if (assetBusy.value) return;
  assetBusy.value = true; assetError.value = "";
  try { await loadProjects(); }
  catch { if (!controller.signal.aborted) assetError.value = "工区列表暂不可用，请重试。"; }
  finally { if (!controller.signal.aborted) assetBusy.value = false; }
}

async function createProject() {
  const name = newProjectName.value.trim();
  if (!name || assetBusy.value) return;
  assetBusy.value = true; assetError.value = ""; assetStatus.value = "正在创建工区…";
  try {
    if (pendingProject?.name !== name) pendingProject = { name, id: crypto.randomUUID() };
    const result = await projectAction("project_create", { name }, pendingProject.id);
    if (!/^hp_[0-9a-f]{32}$/.test(result.project_id || "")) throw new Error("工区创建结果无效，请重试。");
    const id = result.project_id as string;
    pendingProject = undefined;
    projects.value = [{ id, name }, ...projects.value.filter((project) => project.id !== id)];
    selectedProject.value = id;
    newProjectName.value = "";
    assetStatus.value = "工区已创建";
    // A refresh failure must not turn a confirmed create into a duplicate on retry.
    try { await loadProjects(); selectedProject.value = id; }
    catch { if (!controller.signal.aborted) assetStatus.value = "工区已创建，列表刷新稍后重试"; }
  } catch { if (!controller.signal.aborted) assetError.value = "工区创建暂未完成，请重试。"; }
  finally { if (!controller.signal.aborted) assetBusy.value = false; }
}

function chooseFile(event: Event) {
  file.value = (event.target as HTMLInputElement).files?.[0] || null;
  if (file.value && /\.(sgy|segy)$/i.test(file.value.name)) fileRole.value = "seismic";
  else if (file.value && /\.las$/i.test(file.value.name)) fileRole.value = "log";
  assetError.value = ""; assetStatus.value = "";
}

async function uploadAsset() {
  const selected = file.value;
  if (!selected || !selectedProject.value || assetBusy.value) return;
  if (!projects.value.some((project) => project.id === selectedProject.value)) { assetError.value = "请选择当前账号的工区。"; return; }
  const validExtension = fileRole.value === "log" ? /\.las$/i : /\.(sgy|segy)$/i;
  if (!validExtension.test(selected.name) || selected.size === 0 || selected.size > 120 * 1024 * 1024) {
    assetError.value = "文件类型与数据角色不匹配，或文件为空、超过 120 MiB。"; return;
  }
  assetBusy.value = true; assetError.value = ""; assetStatus.value = "正在上传数据…";
  try {
    const project = selectedProject.value;
    const role = fileRole.value;
    const data = new FormData(); data.append("file", selected);
    const response = await fetch(`/api/plug/huiyan/projects/${encodeURIComponent(project)}/assets/${encodeURIComponent(role)}`, {
      method: "POST", credentials: "same-origin", signal: controller.signal, body: data,
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== "ok" || payload.data?.asset?.project_id !== project || payload.data?.asset?.role !== role) throw new Error("上传未完成，请检查文件和工区权限后重试。");
    const receipt = payload.data;
    const archived = receipt.versioning?.status === "archived" && typeof receipt.revision_id === "string" && receipt.versioning.revision_id === receipt.revision_id;
    assetStatus.value = archived ? "数据已上传并归档" : "数据已上传，归档未完成，可重新尝试归档";
    if (archived) file.value = null;
  } catch { if (!controller.signal.aborted) assetError.value = "上传未完成，请检查文件和工区权限后重试。"; }
  finally { if (!controller.signal.aborted) assetBusy.value = false; }
}

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
    if (!isLegacyGaiaEmbedded) loadProjects().catch(() => { if (!controller.signal.aborted) assetError.value = "工区列表暂不可用，请重试。"; });
    window.parent.postMessage({ type: "huiyan:ready" }, window.location.origin);
    const requested = new URLSearchParams(window.location.search).get(isLegacyGaiaEmbedded ? "run_id" : "job_ref");
    if (requested) await openRun(requested);
  } catch (cause) { error.value = cause instanceof Error ? cause.message : "盖亚工作区加载失败"; }
});
onBeforeUnmount(() => { generation++; controller.abort(); });
</script>

<template>
  <section class="gaia-workspace-bar" aria-label="附件与任务">
    <div v-if="!isLegacyGaiaEmbedded" class="asset-upload">
      <strong>添加数据</strong>
      <label>工区
        <select v-model="selectedProject" :disabled="assetBusy">
          <option value="" disabled>选择工区</option>
          <option v-for="project in projects" :key="project.id" :value="project.id">{{ project.name }}</option>
        </select>
      </label>
      <button type="button" class="secondary refresh-projects" :disabled="assetBusy" @click="refreshProjects">刷新工区</button>
      <div class="create-project"><input v-model="newProjectName" :disabled="assetBusy" maxlength="120" placeholder="新工区名称" aria-label="新工区名称" @keydown.enter.prevent="createProject" /><button type="button" class="secondary" :disabled="assetBusy || !newProjectName.trim()" @click="createProject">创建</button></div>
      <label>数据角色
        <select v-model="fileRole" :disabled="assetBusy"><option value="log">测井曲线 · LAS</option><option value="seismic">地震数据 · SEG-Y</option></select>
      </label>
      <label class="file-picker"><span>选择文件</span><input type="file" accept=".las,.sgy,.segy" :disabled="assetBusy || busy" @change="chooseFile" /></label>
      <small v-if="file" class="filename">{{ file.name }}</small>
      <button type="button" :disabled="assetBusy || busy || !file || !selectedProject" @click="uploadAsset">{{ assetBusy ? '处理中…' : '上传到工区' }}</button>
      <p v-if="assetError" class="gaia-error" role="alert">{{ assetError }}</p><p v-else-if="assetStatus" role="status">{{ assetStatus }}</p>
    </div>
    <div v-else class="asset-upload">
      <strong>添加测井数据</strong><label class="file-picker"><span>选择 LAS 文件</span><input type="file" accept=".las" :disabled="busy" @change="chooseFile" /></label><small v-if="file" class="filename">{{ file.name }}</small>
    </div>
    <button v-if="file && fileRole === 'log'" type="button" class="secondary" :disabled="busy || assetBusy" @click="start">{{ busy ? '正在处理…' : '检查并查看 LAS 结果' }}</button>
    <label v-if="history.length" class="history-select">我的任务
      <select v-model="selectedRun" :disabled="busy" @change="openRun(selectedRun)"><option value="" disabled>选择历史任务</option><option v-for="item in history" :key="item.ref" :value="item.ref">{{ dateLabel(item.created_at) }} · {{ statusLabel(item.status) }}</option></select>
    </label>
    <p v-if="error" class="gaia-error" role="alert">{{ error }}</p><p v-else-if="status" role="status">{{ status }}</p>
  </section>
</template>

<style scoped>
.gaia-workspace-bar,.asset-upload { display:grid; gap:9px; min-width:0; color:#344b5b; font-size:13px; }
.asset-upload strong { color:#17394e; font-size:14px; }
label { display:grid; gap:4px; font-weight:600; }
select,input:not([type=file]) { box-sizing:border-box; width:100%; min-width:0; padding:8px; color:inherit; font:inherit; background:#fff; border:1px solid #cedde2; border-radius:8px; }
.create-project { display:flex; gap:6px; }
.create-project input { flex:1; }
button,.file-picker span { display:inline-flex; align-items:center; justify-content:center; padding:8px 11px; color:#fff; font:inherit; font-weight:650; background:#176e78; border:1px solid #176e78; border-radius:8px; cursor:pointer; }
button.secondary,.file-picker span { color:#176e78; background:#f2faf8; border-color:#cde4e1; }
button:disabled { opacity:.5; cursor:not-allowed; }
.file-picker { position:relative; display:inline-flex; width:fit-content; }
.file-picker input { position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); }
.file-picker:focus-within span, button:focus-visible,select:focus-visible,input:focus-visible { outline:2px solid #297f98; outline-offset:2px; }
.filename { overflow-wrap:anywhere; color:#617789; }
p { margin:0; overflow-wrap:anywhere; }.gaia-error { color:#b42318; }
.history-select { padding-top:6px; border-top:1px solid #e6ecef; }
</style>
