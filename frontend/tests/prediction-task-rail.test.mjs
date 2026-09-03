import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { build } from "esbuild";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const themeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");
const temporaryDirectory = await mkdtemp(path.join(os.tmpdir(), "wellfuse-task-rail-"));
const bundlePath = path.join(temporaryDirectory, "prediction-task-rail.mjs");
await build({
  entryPoints: [fileURLToPath(new URL("../src/domain/predictionTaskRail.ts", import.meta.url))],
  outfile: bundlePath,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "es2022",
});
const {
  centeredTaskScrollLeft,
  centeredTaskScrollTop,
  collectLatestPredictionRuns,
} = await import(`${pathToFileURL(bundlePath).href}?${Date.now()}`);

test.after(async () => {
  await rm(temporaryDirectory, { recursive: true, force: true });
});

test("task rail centers first, middle, and last tabs without overscroll", () => {
  assert.equal(centeredTaskScrollLeft(500, 1400, 40, 120), 0);
  assert.equal(centeredTaskScrollLeft(500, 1400, 620, 200), 470);
  assert.equal(centeredTaskScrollLeft(500, 1400, 1280, 120), 900);
  assert.equal(centeredTaskScrollTop(420, 1200, 18, 58), 0);
  assert.equal(centeredTaskScrollTop(420, 1200, 590, 82), 421);
  assert.equal(centeredTaskScrollTop(420, 1200, 1120, 58), 780);
});

test("latest completed result is retained independently for every interpretation task", () => {
  const prediction = (taskId, marker) => ({ prediction: { task_id: taskId, marker } });
  const tasks = [
    { task_id: "run-a-1", task_type: "model_prediction", status: "completed", updated_at: "2026-08-20T10:00:00Z", result: prediction("horizon", "A1") },
    { task_id: "run-b-1", task_type: "model_prediction", status: "completed", updated_at: "2026-08-20T11:00:00Z", result: prediction("fault", "B1") },
    { task_id: "run-a-2", task_type: "model_prediction", status: "completed", updated_at: "2026-08-20T12:00:00Z", result: prediction("horizon", "A2") },
    { task_id: "run-b-2", task_type: "model_prediction", status: "failed", updated_at: "2026-08-20T13:00:00Z", result: null },
  ];
  const history = collectLatestPredictionRuns(tasks, ["horizon", "fault"]);

  assert.equal(history.horizon.executionTaskId, "run-a-2");
  assert.equal(history.horizon.result.prediction.marker, "A2");
  assert.equal(history.fault.executionTaskId, "run-b-1");
  assert.deepEqual(
    [history.horizon, history.fault, history.horizon].map((entry) => entry.result.prediction.marker),
    ["A2", "B1", "A2"],
  );
});

test("history collector restores the newest full-volume FaultSeg result", () => {
  const tasks = [
    {
      task_id: "legacy-full",
      task_type: "model_prediction",
      status: "completed",
      updated_at: "2026-08-20T13:00:00Z",
      result: { prediction: { task_id: "fault", scope: "full_volume" } },
    },
    {
      task_id: "representative-36",
      task_type: "model_prediction",
      status: "completed",
      updated_at: "2026-08-20T12:00:00Z",
      result: { prediction: { task_id: "fault", scope: "representative_grid_36" } },
    },
  ];
  const history = collectLatestPredictionRuns(
    tasks,
    ["fault"],
  );
  assert.equal(history.fault.executionTaskId, "legacy-full");
  assert.match(appSource, /旧抽样 · 需重跑/);
  assert.match(appSource, /!isCurrentFaultPrediction\(historyResult\)/);
});

test("prediction task directory is accessible, vertical, and replaces the top rail", () => {
  assert.match(appSource, /class="prediction-directory"/);
  assert.match(appSource, /返回一级目录/);
  assert.match(appSource, /role="tablist"/);
  assert.match(appSource, /aria-orientation="vertical"/);
  assert.match(appSource, /role="tab"/);
  assert.match(appSource, /:aria-selected="activePredictionTask === task\.id"/);
  assert.match(appSource, /centeredTaskScrollTop/);
  assert.match(appSource, /event\.propertyName !== "min-height"/);
  assert.match(appSource, /event\.key === "ArrowUp"/);
  assert.match(appSource, /event\.key === "ArrowDown"/);
  assert.match(appSource, /sidebarDirectoryLevel\.value = nextView === "prediction"[\s\S]*?\? "prediction"[\s\S]*?: nextView === "layerpulse"[\s\S]*?\? "layerpulse"[\s\S]*?: "primary"/);
  assert.doesNotMatch(appSource, /class="prediction-task-rail-shell"/);
  assert.match(themeSource, /\.prediction-directory-list[\s\S]*?overflow-y: auto/);
  assert.match(themeSource, /\.prediction-task-tab[\s\S]*?min-height: 58px/);
  assert.match(themeSource, /\.prediction-task-tab\.active[\s\S]*?min-height: 82px/);
});

test("both directory levels use prominent commercial typography and enlarged selection", () => {
  assert.match(themeSource, /--sidebar-width: 260px/);
  assert.match(themeSource, /\.directory-rail \.nav-item[\s\S]*?min-height: 54px/);
  assert.match(themeSource, /\.directory-rail \.nav-item\.active[\s\S]*?min-height: 76px/);
  assert.match(themeSource, /\.directory-rail \.nav-item\.active \.nav-label \{[^}]*font-size: 17px/);
  assert.match(themeSource, /\.prediction-task-tab > strong \{[^}]*font-size: 15px/);
  assert.match(themeSource, /\.prediction-task-tab\.active > strong \{[^}]*font-size: 18px/);
  assert.match(themeSource, /\.prediction-task-tab > small \{[^}]*font-size: 12px/);
  assert.match(themeSource, /@media \(prefers-reduced-motion: reduce\)[\s\S]*?\.prediction-directory-list \{ scroll-behavior: auto; \}[\s\S]*?\.prediction-task-tab \{ transition: none; \}/);
});

test("run completion becomes rerun and download follows the run action", () => {
  assert.match(appSource, /const selectedModelHasCompletedResult = computed/);
  assert.match(appSource, /const verb = selectedModelHasCompletedResult\.value \? "重新运行" : "运行"/);
  assert.match(appSource, /\{\{ predictionRunButtonLabel \}\}/);
  const runPosition = appSource.indexOf('class="sample-actions"');
  const downloadPosition = appSource.indexOf('class="result-action-details result-download-action prediction-control-download"');
  const progressPosition = appSource.indexOf('v-if="predictionBusyForActiveTask" class="task-progress"', runPosition);
  assert.ok(runPosition >= 0 && downloadPosition > runPosition && progressPosition > downloadPosition);
  assert.equal(appSource.match(/class="result-action-details result-download-action/g)?.length, 1);
});

test("all standard visualizations, including well sequences, can enter the linked viewer", () => {
  assert.doesNotMatch(appSource, /!isWellSequenceResult\.value/);
  assert.match(appSource, /predictionStandardBundle\.value\?\.visualization\.available === true/);
  assert.match(appSource, /predictionCanvasMode\.value = "result"/);
  assert.match(appSource, /predictionEmbeddedVisualizationUrl\.value \|\| predictionVisualizationUrl\.value/);
  assert.match(appSource, /visualizationUrlWithEmbed\(platformViewerUrl, 1\)/);
  assert.match(appSource, /predictionVisualizationUrl\.value\}&asset=0/);
  assert.doesNotMatch(appSource, /在右侧联动查看/);
  assert.doesNotMatch(appSource, /井曲线或层段成果已在当前页右侧与真实井轨迹联动/);
  assert.doesNotMatch(appSource, /结果已联动到当前页右侧 Viewer/);
  assert.match(themeSource, /\.prediction-workbench-layout \{[\s\S]*?grid-template-columns: minmax\(330px, 360px\) minmax\(540px, 1fr\)/);
  assert.match(themeSource, /\.prediction-live-stage \{[\s\S]*?grid-column: 2/);
  assert.match(themeSource, /\.prediction-live-stage \{[\s\S]*?height: calc\(100dvh - var\(--topbar-height\) - 36px\)[\s\S]*?min-height: 640px[\s\S]*?grid-template-rows: auto minmax\(0, 1fr\) auto/);
  assert.match(themeSource, /\.prediction-live-frame \{[^}]*min-height: 0;/);
  assert.doesNotMatch(themeSource, /\.prediction-live-frame \{[^}]*880px/);
});

test("acceptance only exposes a sealed standard visualization", () => {
  assert.match(
    appSource,
    /const standardResult = completed\s+&& hasArtifactEvidence\s+&& prediction\?\.standard_result_bundle\?\.visualization\?\.available === true;/,
  );
  assert.doesNotMatch(
    appSource,
    /const standardResult = completed && hasArtifactEvidence && Boolean\(prediction\);/,
  );
});

test("legacy well results explain that one rerun is required for linked display", () => {
  assert.match(appSource, /const wellSequenceLinkedViewerUnavailable = computed/);
  assert.match(appSource, /历史结果 · 需更新/);
  assert.match(appSource, /旧成果仍可下载；重新运行后会启用井位分类条联动/);
});
