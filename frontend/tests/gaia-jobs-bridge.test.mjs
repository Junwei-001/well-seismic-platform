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
function harness(legacy = false, fetchOverride) {
  const calls = [], events = [], messages = [];
  let mounted, unmount;
  const window = { location: { href: `http://localhost${legacy ? "/api/workflow-runs/huiyan/ui/" : "/api/plug/huiyan/ui/"}`, origin: "http://localhost", search: "" }, history: { replaceState() {} }, parent: { postMessage: (...args) => messages.push(args) } };
  const result = evaluate(bridge + "\nexports.api = { start, openRun, file, busy, error, history, normalizeJob };", {
    vue: { ...vue, onMounted: (fn) => { mounted = fn; }, onBeforeUnmount: (fn) => { unmount = fn; } }, "../../gaiaEmbed": { isLegacyGaiaEmbedded: legacy },
  }, { window, defineEmits: () => (...args) => events.push(args), fetch: async (url, init) => {
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
