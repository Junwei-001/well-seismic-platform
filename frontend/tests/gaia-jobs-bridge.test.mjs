import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";
import * as vue from "vue";

const read = (path) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
const bridge = read("components/workflow/GaiaWorkspaceBar.vue").match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1];
const J = "j_1111111111111111", R = "run_1111111111111111", TASK = "a".repeat(32);
function evaluate(source, imports, globals) {
  const module = { exports: {} };
  vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText, {
    module, exports: module.exports, require: (name) => imports[name],
    AbortController, URL, URLSearchParams, FormData, crypto, setTimeout, Error, ...globals,
  });
  return module.exports;
}
const taskReferences = evaluate(read("assistantTaskResult.ts"), {}, {});
function harness(legacy = false, fetchOverride, restoreResult) {
  const calls = [], events = [], messages = [];
  let mounted, unmount;
  const window = { location: { href: `http://localhost${legacy ? "/api/workflow-runs/huiyan/ui/" : "/api/plug/huiyan/ui/"}`, origin: "http://localhost", search: "" }, history: { replaceState() {} }, parent: { postMessage: (...args) => messages.push(args) } };
  const result = evaluate(bridge + "\nexports.api = { start, openRun, openAssistantResult, file, busy, error, history, normalizeJob, projectAction, loadProjects, createProject, uploadAsset, fileRole, projects, selectedProject, newProjectName, assetStatus, assetError };", {
    vue: { ...vue, onMounted: (fn) => { mounted = fn; }, onBeforeUnmount: (fn) => { unmount = fn; } }, "../../gaiaEmbed": { isLegacyGaiaEmbedded: legacy }, "../../assistantTaskResult": taskReferences,
  }, { window, defineProps: () => ({ restoreResult }), defineExpose: () => {}, defineEmits: () => (...args) => events.push(args), fetch: async (url, init) => {
    calls.push({ url, init });
    if (fetchOverride) return fetchOverride(url, init);
    const ref = legacy ? R : J;
    const data = url.endsWith(ref) ? (legacy ? { run_id: ref, status: "succeeded", artifacts: [{ kind: "huiyan_preparation_report", validation_status: "validated", data: { task_id: TASK } }] } : { ref, status: "completed", task_id: TASK, report: { task_id: TASK, validation_status: "validated" } }) : init.method === "POST" ? { ref, run_id: ref, artifact_id: "input" } : { items: [] };
    return { ok: true, json: async () => ({ status: "ok", data }) };
  } });
  return { ...result.api, calls, events, messages, mounted: () => mounted(), unmount: () => unmount() };
}
test("native API prefix defaults to plugin/jobs and selects historical mode only by exact pathname", () => {
  for (const [pathname, legacy, expected] of [["/", false, "/api/v1/health"], ["/api/plug/huiyan/ui/", false, "/api/plug/huiyan/ui/api/v1/health"], ["/api/workflow-runs/huiyan/ui/", true, "/api/workflow-runs/huiyan/ui/api/v1/health"]]) {
    const module = evaluate(read("gaiaEmbed.ts"), {}, { window: { location: { pathname } } });
    assert.equal(module.isLegacyGaiaEmbedded, legacy);
    assert.equal(module.nativeApiUrl("/api/v1/health"), expected);
  }
  const serverModule = evaluate(read("gaiaEmbed.ts"), {}, {});
  assert.equal(serverModule.isGaiaEmbedded, false);
  assert.equal(serverModule.nativeApiUrl("/api/v1/health"), "/api/v1/health");
});
test("new native bridge submits one idempotent job and restores its verified task", async () => {
  const h = harness();
  h.file.value = new File(["LAS"], "demo.las");
  await h.start();
  assert.equal(h.calls[0].url, "/api/plug/huiyan/jobs");
  assert.match(h.calls[0].init.body.get("operation_id"), /^[0-9a-f-]{36}$/);
  assert.equal(h.calls.filter((call) => call.init.method === "POST").length, 1);
  assert.equal(h.events.find(([kind]) => kind === "completed")[1].taskId, TASK);
  assert.equal(h.messages[0][0].type, "huiyan:job");
  assert.equal(h.messages[0][0].jobRef, J);
  assert.equal(h.busy.value, false);
});
test("explicit historical mode retains the already recorded local flow", async () => {
  const h = harness(true);
  h.file.value = new File(["LAS"], "demo.las");
  await h.start();
  assert.equal(h.calls[0].url, "/api/workflow-runs/inputs");
  assert.equal(h.calls[1].url, "/api/workflow-runs");
  assert.equal(h.events.find(([kind]) => kind === "completed")[1].taskId, TASK);
  assert.equal(h.messages[0][0].type, "huiyan:run");
  assert.equal(h.normalizeJob({ status: "planning", created_at: 1000 }).status, "working");
  assert.equal(h.normalizeJob({ created_at: 1000 }).created_at, 1);
});
test("invalid or unverified job responses never select a native result", async () => {
  const h = harness(false, async () => ({ ok: true, json: async () => ({ status: "ok", data: { ref: J, status: "completed", task_id: TASK, report: { task_id: "b".repeat(32), validation_status: "validated" } } }) }));
  await h.openRun(J);
  assert.equal(h.events.some(([kind]) => kind === "completed"), false);
  assert.match(h.error.value, /缺少已核验/);
});
test("backend failure details cannot be exposed in the native bridge", async () => {
  const h = harness(false, async () => ({ ok: false, status: 403, json: async () => ({ status: "error", message: "Bearer private-token" }) }));
  await h.openRun(J);
  assert.match(h.error.value, /请求失败/);
  assert.doesNotMatch(h.error.value, /private-token/);
});
test("an obsolete native job response does not overwrite a newer selection", async () => {
  let release;
  const old = new Promise((resolve) => { release = resolve; });
  const other = "j_2222222222222222";
  const h = harness(false, async (url) => ({ ok: true, json: async () => ({ status: "ok", data: url.endsWith(J) ? await old : url.endsWith(other) ? { ref: other, status: "completed", task_id: TASK, report: { task_id: TASK, validation_status: "validated" } } : { items: [] } }) }));
  const first = h.openRun(J);
  await h.openRun(other);
  release({ ref: J, status: "failed" });
  await first;
  assert.equal(h.error.value, "");
  assert.equal(h.events.filter(([kind]) => kind === "completed").length, 1);
});

const nativeApi = evaluate(read("api.ts"), { "./gaiaEmbed": { nativeApiUrl: value => value } }, {});
const { loadGaiaTaskChain } = evaluate(read("gaiaTaskRestore.ts"), { "./api": nativeApi }, {});
const PREP = "a".repeat(32), REG = "b".repeat(32), SAMPLE = "c".repeat(32);
function nativeTasks() {
  return {
    [PREP]: { task_id: PREP, task_type: "data_preparation", status: "completed", snapshot_id: PREP,
      request: { log_paths: ["synthetic.las"] }, result: { summary: { wells: 1 }, seismic: [], errors: [],
        preparation: { issues: [] }, data_snapshot: { snapshot_id: PREP } } },
    [REG]: { task_id: REG, task_type: "well_tie", status: "completed", snapshot_id: PREP,
      request: { source_snapshot_id: PREP }, result: { registration: { source_snapshot_id: PREP,
        can_build_multimodal_view: true, fusion_ready_well_count: 1, business_status: "usable" } } },
    [SAMPLE]: { task_id: SAMPLE, task_type: "sample_building", status: "completed", snapshot_id: PREP,
      request: { source_snapshot_id: PREP, registration_task_id: REG }, result: { source_snapshot_id: PREP,
        registration_task_id: REG, matching: { windows: 1 }, prepared_view: { source_snapshot_id: PREP, state: "ready" } } },
  };
}

test("registration and sample jobs select their native task without requiring a LAS report", async () => {
  for (const [operation, task_id] of [["registration", REG], ["sample_build", SAMPLE]]) {
    const h = harness(false, async url => ({ ok: true, json: async () => ({ status: "ok", data: url.endsWith(J)
      ? { ref: J, kind: "harness", operation, status: "completed", task_id, receipt: { status: "completed", task_id, result: {} } }
      : { items: [] } }) }));
    await h.openRun(J);
    assert.equal(h.events.find(([type]) => type === "completed")[1].taskId, task_id);
    assert.equal(h.error.value, "");
  }
});

test("native derivative restoration resolves and validates its full source chain", async () => {
  const tasks = nativeTasks(), calls = [];
  const load = async id => { calls.push(id); return tasks[id]; };
  const registration = await loadGaiaTaskChain(REG, load);
  assert.equal(registration.preparation, tasks[PREP]);
  assert.equal(registration.registration, tasks[REG]);
  assert.equal(registration.sample, undefined);
  calls.length = 0;
  const sample = await loadGaiaTaskChain(SAMPLE, load);
  assert.deepEqual(calls, [SAMPLE, REG, PREP]);
  assert.equal(sample.snapshotId, PREP);
  assert.equal(sample.sample, tasks[SAMPLE]);
  assert.equal(sample.registration, tasks[REG]);
});

test("foreign, mismatched or incomplete source chains never produce a restorable result", async () => {
  const tasks = nativeTasks();
  await assert.rejects(loadGaiaTaskChain(REG, async id => {
    if (id === PREP) throw new Error("source not owned");
    return tasks[id];
  }), /source not owned/);
  tasks[REG].result.registration.source_snapshot_id = "d".repeat(32);
  await assert.rejects(loadGaiaTaskChain(REG, async id => tasks[id]), /来源不一致/);
  const failed = nativeTasks(); failed[REG].status = "failed";
  await assert.rejects(loadGaiaTaskChain(SAMPLE, async id => failed[id]), /尚未完成/);
  await assert.rejects(loadGaiaTaskChain(REG, async () => nativeTasks()[PREP]), /不匹配/);
  let calls = 0;
  assert.equal(await loadGaiaTaskChain(SAMPLE, async id => { calls++; return nativeTasks()[id]; }, () => false), null);
  assert.equal(calls, 1, "obsolete selection must not continue restoring source tasks");
});

test("App restores native preparation, registration and PreparedView then opens their result stage", async () => {
  const app = read("App.vue");
  const start = app.indexOf("async function restoreGaiaResult(");
  const source = app.slice(start, app.indexOf("\nonMounted(", start));
  for (const [selected, expectedStages, screen] of [[PREP, [PREP], "pipeline"], [REG, [PREP, REG], "fusion"], [SAMPLE, [PREP, REG, SAMPLE], "fusion"]]) {
    const tasks = nativeTasks(), installed = [], views = [];
    const preparationScreen = vue.ref("input"), errorMessage = vue.ref("");
    const exposed = evaluate(source + "\nexports.restore = restoreGaiaResult;", {}, {
      gaiaSelectionEpoch: 1, componentUnmounted: false, loadGaiaTaskChain,
      getTask: async id => tasks[id], preparationScreen, errorMessage,
      pointVisualizationTaskId: vue.ref(""),
      restoreCompletedDataPreparationTask: task => installed.push(task.task_id),
      installCompletedWorkflowTask: task => installed.push(task.task_id),
      selectView: view => views.push(view),
    });
    await exposed.restore({ taskId: selected });
    assert.deepEqual(installed, expectedStages);
    assert.deepEqual(views, ["preparation"]);
    assert.equal(preparationScreen.value, screen);
    assert.equal(errorMessage.value, "");
  }
});


function appPersistenceHarness(writeError) {
  const app = read("App.vue");
  const section = (start, end) => app.slice(app.indexOf(start), app.indexOf(end, app.indexOf(start)));
  const source = section("function rememberSourceSnapshot(", "\nfunction verifiedAutomaticSegyContract(")
    + section("function taskRequestString(", "\nasync function reattachWorkflowTask(")
    + section("async function restoreGaiaResult(", "\nonMounted(");
  const writes = [], warnings = [], tasks = nativeTasks();
  const refs = Object.fromEntries([
    "taskId", "progress", "statusMessage", "dataSnapshotTaskId", "registrationTaskId", "registrationResult",
    "horizontalRegistrationTaskId", "horizontalRegistrationResult", "sampleBuildingTaskId", "sampleResult",
  ].map(key => [key, vue.ref("")]));
  const exported = evaluate(source + "\nexports.api = { restoreGaiaResult, installCompletedWorkflowTask, flush: () => activeSnapshotPersistenceQueue };", {}, {
    ...refs, activeProjectId: vue.ref("owned-project"),
    activeSnapshotPersistenceQueue: Promise.resolve(), activeSnapshotPersistencePendingCount: 0,
    latestRequestedActiveSnapshot: "", lastPersistedActiveSnapshot: "",
    LAST_SOURCE_SNAPSHOT_STORAGE_KEY: "snapshot", ACTIVE_SNAPSHOT_PERSISTENCE_WARNING_PREFIX: "snapshot write: ",
    window: { sessionStorage: { setItem() {} } },
    setActiveProjectSnapshot: async (...args) => { writes.push(args); if (writeError) throw writeError; },
    appendRestorationWarning: warning => warnings.push(warning), clearRestorationWarningsByPrefix() {},
    activeSnapshotPersistenceFailureReason: error => error.message,
    isWorkflowResult: nativeApi.isWorkflowResult,
    isHorizontalRegistrationTaskResult: nativeApi.isHorizontalRegistrationTaskResult,
    initializeVisualization() {},
    gaiaSelectionEpoch: 1, componentUnmounted: false, loadGaiaTaskChain,
    getTask: async id => tasks[id], preparationScreen: vue.ref("input"), errorMessage: vue.ref(""),
    pointVisualizationTaskId: vue.ref(""),
    restoreCompletedDataPreparationTask() {}, selectView() {},
  });
  return { ...exported.api, writes, warnings, tasks, refs };
}

test("embedded derivative restoration never writes the desktop active-snapshot pointer", async () => {
  for (const selected of [REG, SAMPLE]) {
    const h = appPersistenceHarness(new Error("Permission denied"));
    await h.restoreGaiaResult({ taskId: selected });
    await h.flush();
    assert.deepEqual(h.writes, [], "owner-checked result restoration must remain read-only");
    assert.deepEqual(h.warnings, []);
    assert.equal(h.refs.taskId.value, selected);
    assert.equal(h.refs.registrationTaskId.value, REG);
  }
});

test("desktop workflow installation still persists and reports real write failures", async () => {
  const h = appPersistenceHarness(new Error("Permission denied"));
  h.installCompletedWorkflowTask(h.tasks[REG], h.tasks[REG].result);
  await h.flush();
  assert.deepEqual(h.writes, [["owned-project", PREP]]);
  assert.equal(h.warnings.length, 1);
  assert.match(h.warnings[0], /Permission denied/);
});

const PROJECT = "hp_" + "d".repeat(32);
const reference = { projectId: PROJECT, taskId: REG };
const ok = data => ({ ok: true, json: async () => ({ status: "ok", data }) });
function statusJob(overrides = {}) {
  return { ref: J, kind: "harness", operation: "task_status", status: "completed", project_id: PROJECT, task_id: REG,
    receipt: { operation: "task_status", status: "completed", project_id: PROJECT, task_id: REG,
      result: { task: { task_id: REG, status: "completed" } } }, ...overrides };
}
test("only complete supported receipts with consistent opaque IDs can offer task navigation", () => {
  const receipt = { operation: "registration", status: "completed", project_id: PROJECT, task_id: REG, result: {} };
  assert.equal(taskReferences.assistantTaskResultReference(receipt).taskId, REG);
  for (const change of [ { status: "failed" }, { status: "running" }, { status: "blocked" }, { operation: "prediction" },
    { task_id: "../private" }, { project_id: "https://outside.invalid" }, { result: { task_id: SAMPLE } },
    { result: { project_id: "hp_" + "e".repeat(32) } } ]) {
    assert.equal(taskReferences.assistantTaskResultReference({ ...receipt, ...change }), null);
  }
});

test("AI result navigation registers a read-only current-account job and restores its checked native result", async () => {
  const h = harness(false, async url => ok(url === "/api/plug/huiyan/actions" ? { ref: J } : url.endsWith(J) ? statusJob() : { items: [statusJob()] }));
  assert.equal(await h.openAssistantResult({ ...reference, url: "https://outside.invalid", owner: "other" }), true);
  const post = h.calls.find(call => call.init.method === "POST");
  assert.equal(post.url, "/api/plug/huiyan/actions");
  assert.equal(post.init.credentials, "same-origin");
  const body = JSON.parse(post.init.body);
  assert.equal(body.operation, "task_status");
  assert.match(body.operation_id, /^[0-9a-f-]{36}$/);
  assert.deepEqual(body.arguments, { project_id: PROJECT, task_id: REG });
  assert.deepEqual(Object.keys(body).sort(), ["arguments", "operation", "operation_id"]);
  assert.equal(h.events.find(([event]) => event === "completed")[1].taskId, REG);
  assert.equal(h.messages[0][0].jobRef, J);
  assert.equal(h.messages[0][1], "http://localhost");
  assert.equal(h.history.value[0].ref, J);
  await h.openAssistantResult(reference);
  assert.equal(h.calls.filter(call => call.init.method === "POST").length, 1, "repeat views reuse the recorded job");
});

test("legacy mode and invalid references never issue actions or navigate", async () => {
  for (const [legacy, value] of [[true, reference], [false, { ...reference, taskId: "https://outside.invalid" }], [false, null]]) {
    const h = harness(legacy);
    assert.equal(await h.openAssistantResult(value), false);
    assert.equal(h.calls.length, 0);
    assert.equal(h.messages.length, 0);
  }
});

test("foreign-account or expired-login denial keeps private error details out of the result UI", async () => {
  const h = harness(false, async () => ({ ok: false, status: 403, json: async () => ({ status: "error", message: "Bearer FAKE_SECRET C:/private" }) }));
  assert.equal(await h.openAssistantResult(reference), false);
  assert.equal(h.events.some(([kind]) => kind === "completed"), false);
  assert.equal(h.messages.length, 0);
  assert.doesNotMatch(h.error.value, /FAKE_SECRET|private|Bearer/);
  assert.match(h.error.value, /当前账号/);
});

test("mismatched task, project, job kind and incomplete native result cannot be restored", async () => {
  const job = statusJob();
  for (const broken of [statusJob({ task_id: SAMPLE }), statusJob({ project_id: "hp_" + "e".repeat(32) }),
    statusJob({ kind: "las" }), statusJob({ receipt: { ...job.receipt, task_id: SAMPLE } }),
    statusJob({ receipt: { ...job.receipt, result: { task: { task_id: REG, status: "failed" } } } })]) {
    const h = harness(false, async url => ok(url === "/api/plug/huiyan/actions" ? { ref: J } : broken));
    assert.equal(await h.openAssistantResult(reference), false);
    assert.equal(h.events.some(([kind]) => kind === "completed"), false);
    assert.equal(h.messages.length, 0, "invalid result cannot change the parent selection");
  }
});

test("network retry preserves the operation identity and blocks duplicate clicks", async () => {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const h = harness(false, async (url, init) => {
    if (init.method === "POST") {
      if (h.calls.filter(call => call.init.method === "POST").length === 1) { await pending; throw new Error("private network error"); }
      return ok({ ref: J });
    }
    return ok(url.endsWith(J) ? statusJob() : { items: [] });
  });
  const first = h.openAssistantResult(reference);
  assert.equal(await h.openAssistantResult(reference), false);
  assert.equal(h.calls.length, 1);
  release();
  assert.equal(await first, false);
  assert.equal(await h.openAssistantResult(reference), true);
  const posts = h.calls.filter(call => call.init.method === "POST");
  assert.equal(JSON.parse(posts[0].init.body).operation_id, JSON.parse(posts[1].init.body).operation_id);
});

test("late action responses after a new selection or unmount cannot navigate or restore", async () => {
  for (const unmount of [false, true]) {
    let release;
    const pending = new Promise(resolve => { release = resolve; });
    const OTHER = "j_2222222222222222";
    const h = harness(false, async (url, init) => {
      if (init.method === "POST") return pending;
      return ok(url.endsWith(OTHER) ? statusJob({ ref: OTHER }) : { items: [] });
    });
    const old = h.openAssistantResult(reference);
    if (unmount) h.unmount();
    else await h.openRun(OTHER);
    release(ok({ ref: J }));
    assert.equal(await old, false);
    assert.equal(h.messages.some(([message]) => message.jobRef === J), false);
    assert.equal(h.events.filter(([kind]) => kind === "completed").length, unmount ? 0 : 1);
  }
});

test("App forwards both assistant views through the existing Gaia workspace bar", async () => {
  const app = read("App.vue");
  assert.equal((app.match(/@open-task-result="openAssistantTaskResult"/g) || []).length, 2);
  assert.match(app, /<GaiaWorkspaceBar[^>]+ref="gaiaWorkspaceBar"/);
  const start = app.indexOf("async function openAssistantTaskResult(");
  const source = app.slice(start, app.indexOf("\nlet gaiaSelectionEpoch", start));
  const calls = [], busy = vue.ref(false), drawer = vue.ref(true);
  const module = evaluate(source + "\nexports.open = openAssistantTaskResult;", {}, {
    beginAssistantResultNavigation: () => true, endAssistantResultNavigation() {},
    isGaiaEmbedded: true, isLegacyGaiaEmbedded: false, assistantResultOpening: busy, assistantResultError: vue.ref(""), assistantDrawerOpen: drawer,
    gaiaWorkspaceBar: vue.ref({ openAssistantResult: async value => { calls.push(value); return true; } }),
  });
  await module.open(reference);
  assert.equal(calls[0], reference);
  assert.equal(busy.value, false);
  assert.equal(drawer.value, true, "result navigation keeps assistant usable");
});

test("bridge waits for real native restoration and reports failed restoration without a success event", async () => {
  for (const success of [true, false]) {
    let release, entered;
    const restored = new Promise(resolve => { release = resolve; });
    const restorationStarted = new Promise(resolve => { entered = resolve; });
    const h = harness(false, async url => ok(url === "/api/plug/huiyan/actions" ? { ref: J } : url.endsWith(J) ? statusJob() : { items: [] }),
      async value => { assert.equal(value.taskId, REG); entered(); return restored; });
    let finished = false;
    const opened = h.openAssistantResult(reference).then(value => { finished = true; return value; });
    await restorationStarted;
    assert.equal(h.busy.value, true);
    assert.equal(finished, false);
    assert.equal(h.events.some(([kind]) => kind === "completed"), false);
    release(success);
    assert.equal(await opened, success);
    assert.equal(h.busy.value, false);
    if (!success) assert.match(h.error.value, /无法恢复原生任务结果/);
  }
});

test("failed result restoration keeps the assistant open and exposes a safe visible error", async () => {
  const app = read("App.vue"), start = app.indexOf("async function openAssistantTaskResult(");
  const source = app.slice(start, app.indexOf("\nlet gaiaSelectionEpoch", start));
  const busy = vue.ref(false), drawer = vue.ref(true), resultError = vue.ref("");
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const module = evaluate(source + "\nexports.open = openAssistantTaskResult;", {}, {
    beginAssistantResultNavigation: () => true, endAssistantResultNavigation() {},
    isGaiaEmbedded: true, isLegacyGaiaEmbedded: false, assistantResultOpening: busy,
    assistantResultError: resultError, assistantDrawerOpen: drawer,
    gaiaWorkspaceBar: vue.ref({ openAssistantResult: () => pending }),
  });
  const opening = module.open(reference);
  assert.equal(busy.value, true);
  assert.equal(drawer.value, true);
  release(false);
  await opening;
  assert.equal(drawer.value, true);
  assert.equal(busy.value, false);
  assert.match(resultError.value, /无法打开任务结果/);
  assert.match(read("components/assistant/FloatingAssistant.vue"), /v-if="resultError"[^>]+role="alert"/);
});

test("an obsolete failed restoration cannot overwrite a newer bridge selection", async () => {
  let release, entered, restores = 0;
  const pending = new Promise(resolve => { release = resolve; });
  const began = new Promise(resolve => { entered = resolve; });
  const OTHER = "j_2222222222222222";
  const h = harness(false, async url => ok(url === "/api/plug/huiyan/actions" ? { ref: J }
    : url.endsWith(J) ? statusJob() : url.endsWith(OTHER) ? statusJob({ ref: OTHER }) : { items: [] }),
    async () => { if (++restores === 1) { entered(); return pending; } return true; });
  const old = h.openAssistantResult(reference);
  await began;
  await h.openRun(OTHER);
  release(false);
  assert.equal(await old, false);
  assert.equal(h.error.value, "");
});


test("project creation uses an idempotent Gaia action and selects the receipt-owned project", async () => {
  let currentOperation = ""; const actionIds = [];
  const h = harness(false, async (url, init = {}) => {
    if (url === "/api/plug/huiyan/actions") {
      const body = JSON.parse(init.body); currentOperation = body.operation;
      actionIds.push(body.operation_id);
      if (currentOperation === "project_create" && actionIds.length === 1) throw new Error("connection interrupted");
      return ok({ ref: J });
    }
    assert.equal(url, `/api/plug/huiyan/jobs/${J}`);
    return ok({ ref: J, kind: "harness", operation: currentOperation, status: "completed",
      receipt: { operation: currentOperation, status: "completed", result: currentOperation === "project_create"
        ? { project_id: PROJECT, name: "Field A" } : { projects: [{ id: PROJECT, name: "Field A" }], next_offset: null } } });
  });
  h.newProjectName.value = "Field A";
  await h.createProject();
  assert.match(h.assetError.value, /暂未完成/);
  await h.createProject();
  assert.equal(actionIds[0], actionIds[1], "retry must reuse operation identity after transport failure");
  assert.equal(h.selectedProject.value, PROJECT);
  assert.equal(h.projects.value[0].name, "Field A");
  assert.equal(h.assetStatus.value, "工区已创建");
});

test("role-aware uploads go only to selected owner project, report partial archive honestly", async () => {
  const h = harness(false, async (url, init) => {
    assert.equal(url, `/api/plug/huiyan/projects/${PROJECT}/assets/seismic`);
    assert.equal(init.method, "POST"); assert.equal(init.credentials, "same-origin");
    assert.equal(init.body.get("file").name, "synthetic.sgy");
    return ok({ asset: { project_id: PROJECT, role: "seismic" }, versioning: { status: "blocked" } });
  });
  h.projects.value = [{ id: PROJECT, name: "Field A" }]; h.selectedProject.value = PROJECT;
  h.fileRole.value = "seismic"; h.file.value = new File(["synthetic"], "synthetic.sgy");
  await h.uploadAsset();
  assert.equal(h.assetStatus.value, "数据已上传，归档未完成，可重新尝试归档");
  assert.notEqual(h.file.value, null, "retain file so the server can safely recover archiving");
});

test("mismatched upload receipt and foreign project selection cannot claim success", async () => {
  const h = harness(false, async () => ok({ asset: { project_id: "hp_" + "f".repeat(32), role: "log" } }));
  h.projects.value = [{ id: PROJECT, name: "Field A" }]; h.selectedProject.value = PROJECT;
  h.file.value = new File(["synthetic"], "sample.las"); await h.uploadAsset();
  assert.match(h.assetError.value, /未完成/); assert.notEqual(h.file.value, null);
  h.selectedProject.value = "hp_" + "e".repeat(32); const priorCalls = h.calls.length;
  await h.uploadAsset(); assert.equal(h.calls.length, priorCalls);
});

test("completed P21 result opens the central visualization and retains both shell contexts", async () => {
  const app = read("App.vue"), start = app.indexOf("async function restoreGaiaResult(");
  const source = app.slice(start, app.indexOf("\nonMounted(", start));
  const point = vue.ref(""), ready = vue.ref(false), task = vue.ref(""), view = vue.ref("overview"), drawer = vue.ref(false);
  const location = { href: `http://localhost/api/plug/huiyan/ui/?job_ref=${J}&task_id=${TASK}#visualization`, origin: "http://localhost" };
  const urls = [];
  const module = evaluate(source + "\nexports.restore = restoreGaiaResult;", {}, {
    gaiaSelectionEpoch: 1, componentUnmounted: false,
    getTask: async id => ({ task_id: id, status: "completed", result: { prediction: { model_id: "sixtrack_reconstruction_p21" } } }),
    pointVisualizationTaskId: point, pointVisualizationReady: ready, taskId: task, activeView: view,
    sidebarDirectoryLevel: vue.ref(""), assistantDrawerOpen: drawer, errorMessage: vue.ref(""),
    window: { location, history: { replaceState: (...args) => urls.push(args[2]) } },
    loadGaiaTaskChain: () => { throw new Error("P21 must not require a preparation chain"); },
  });
  assert.equal(await module.restore({ taskId: TASK }), true);
  assert.equal(point.value, TASK); assert.equal(task.value, "", "P21 selection must not replace the preparation task identity");
  assert.equal(view.value, "visualization"); assert.equal(drawer.value, true);
  assert.equal(ready.value, false, "HTML navigation is not successful drawing");
  assert.equal(urls[0].searchParams.get("job_ref"), J);
  assert.equal(urls[0].hash, "#visualization");
  assert.match(app, /ref="pointVisualizationFrame"[^>]+:src="pointVisualizationUrl"/);
});

test("central canvas readiness requires exact origin, iframe and selected task", () => {
  const app = read("App.vue"), start = app.indexOf("function onPointVisualizationReady(");
  const source = app.slice(start, app.indexOf("\nconst assistantResultOpening", start));
  const ready = vue.ref(false), frameWindow = vue.markRaw({}), calls = [];
  const module = evaluate(source + "\nexports.receive=onPointVisualizationReady;", {}, {
    pointVisualizationReady: ready, pointVisualizationTaskId: vue.ref(TASK),
    pointVisualizationFrame: vue.ref({ contentWindow: frameWindow }), isGaiaEmbedded: true,
    window: { location: { origin: "http://localhost" }, parent: { postMessage: (...args) => calls.push(args) } },
  });
  const event = { origin: "http://localhost", source: frameWindow, data: { type: "huiyan:visualization-ready", taskId: TASK } };
  for (const wrong of [{ ...event, origin: "https://foreign.invalid" }, { ...event, source: {} },
    { ...event, data: { ...event.data, taskId: SAMPLE } }, { ...event, data: { type: "huiyan:ready", taskId: TASK } }]) {
    module.receive(wrong); assert.equal(ready.value, false); assert.equal(calls.length, 0);
  }
  module.receive(event); assert.equal(ready.value, true); assert.equal(calls.length, 1);
  assert.equal(calls[0][0].taskId, TASK); assert.equal(calls[0][1], "http://localhost");
});


test("new workflow identities clear a previously selected P21 view without changing workflow state", () => {
  const app = read("App.vue"), start=app.indexOf("watch(taskId, (next, previous) => {");
  const source=app.slice(start, app.indexOf("\nconst pointVisualizationUrl",start));
  const task=vue.ref(PREP), point=vue.ref(TASK), ready=vue.ref(true);
  evaluate(source, {}, {watch:vue.watch, taskId:task, pointVisualizationTaskId:point, pointVisualizationReady:ready});
  task.value=SAMPLE;
  assert.equal(point.value, ""); assert.equal(ready.value,false); assert.equal(task.value,SAMPLE);
  assert.equal((app.match(/<GaiaWorkspaceBar ref="gaiaWorkspaceBar"/g)||[]).length,1);
  assert.match(app, /<Teleport v-if="isGaiaEmbedded" defer/);
});
