import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { build } from "esbuild";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const temporaryDirectory = await mkdtemp(path.join(os.tmpdir(), "wellfuse-fusion-progress-"));
const bundlePath = path.join(temporaryDirectory, "fusion-progress.mjs");
await build({
  entryPoints: [fileURLToPath(new URL("../src/domain/fusionProgress.ts", import.meta.url))],
  outfile: bundlePath,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "es2022",
});
const { countDownRemainingSeconds, estimateProgressRemainingSeconds } = await import(
  `${pathToFileURL(bundlePath).href}?${Date.now()}`
);

test.after(async () => {
  await rm(temporaryDirectory, { recursive: true, force: true });
});

test("fusion ETA is withheld until enough progress exists and then uses observed pace", () => {
  assert.equal(estimateProgressRemainingSeconds(4, 50), null);
  assert.equal(estimateProgressRemainingSeconds(20, 20), 80);
  assert.equal(estimateProgressRemainingSeconds(60, 25), 180);
  assert.equal(estimateProgressRemainingSeconds(60, 100), null);
  assert.equal(countDownRemainingSeconds(180, 30), 150);
  assert.equal(countDownRemainingSeconds(180, 181), null);
});

test("both fusion progress surfaces use the reading spinner, elapsed time, and ETA copy", () => {
  assert.equal(
    appSource.match(/class="task-progress preparation-progress fusion-progress"/g)?.length,
    2,
  );
  assert.match(appSource, /class="reading-orbit"/);
  assert.match(appSource, /fusionElapsedSeconds/);
  assert.match(appSource, /fusionEtaSeconds/);
  assert.match(appSource, /完整标定与融合可能需要 10 分钟以上/);
  assert.match(appSource, /:data-ready="formalRegistrationReady"/);
  assert.match(appSource, /registrationRunning \? `运行中 · \$\{progress\}%`/);
  assert.match(appSource, /countDownRemainingSeconds\(/);
  assert.match(appSource, /task\.progress > fusionEtaSampleProgress\.value/);
  assert.match(
    appSource,
    /function startFusionClock[\s\S]*?fusionEtaBaselineSeconds\.value = null;[\s\S]*?fusionEtaComputedAt\.value = null;[\s\S]*?fusionEtaSampleProgress\.value = 0;/,
  );
});

test("non-FaultSeg predictions stay locked until a PreparedView is ready", () => {
  assert.match(
    appSource,
    /async function runPrediction\(\)[\s\S]*?if \(!preparedViewReady\.value && !isFaultSegModel\.value\)[\s\S]*?当前模型提交已锁定/,
  );
  assert.match(
    appSource,
    /:disabled="predictionBusy \|\| \(!isFaultSegModel && fusionInputsMutating\) \|\| !selectedPredictionWorkflowGateReady \|\| !predictionInputReady \|\| !selectedPredictionModel"/,
  );
  assert.match(appSource, /selectedPredictionWorkflowGateReady = computed\(\(\) =>[\s\S]*?isFaultSegModel\.value \? faultSegSnapshotSourceReady\.value : preparedViewReady\.value/);
  assert.match(appSource, /runIntent\.isFaultSegModel[\s\S]*?\? \{\}[\s\S]*?registration_task_id: registrationTaskId\.value \|\| undefined,[\s\S]*?prepared_view_task_id: sampleBuildingTaskId\.value \|\| undefined/);
  assert.doesNotMatch(appSource, /进入非融合(?:预测|下游)/);
  assert.doesNotMatch(appSource, /skipPreferredPreparedView|openPreferredPreparedViewDirectRoute|formal_nonfusion/);
});

test("fusion mutations are serialized and a PreparedView must match the active registration", () => {
  assert.match(
    appSource,
    /const preparedViewReady = computed[\s\S]*?sampleResult\.value\?\.registration_task_id === registrationTaskId\.value[\s\S]*?sampleResult\.value\?\.prepared_view\?\.source_snapshot_id === dataSnapshotTaskId\.value/,
  );
  assert.match(
    appSource,
    /const fusionWorkflowMutationRunning = computed[\s\S]*?registrationRunning\.value[\s\S]*?sampleRunning\.value[\s\S]*?geoPathCandidateRunning\.value[\s\S]*?geoPathAcceptanceRunning\.value[\s\S]*?predictionBusy\.value/,
  );
  for (const functionName of [
    "runWellTie",
    "runSampleBuilding",
    "runGeoPathCandidate",
    "acceptSelectedGeoPathWells",
  ]) {
    assert.match(
      appSource,
      new RegExp(`async function ${functionName}\\([^]*?if \\(fusionWorkflowMutationRunning\\.value\\)`),
    );
  }
  assert.match(
    appSource,
    /async function runPrediction\(\)[\s\S]*?if \(fusionInputsMutating\.value && !isFaultSegModel\.value\)/,
  );
  assert.match(
    appSource,
    /async function runSampleBuilding[\s\S]*?const completed = await waitForTask\(created\.task_id, \{[\s\S]*?persistent: true,[\s\S]*?onProgress: updateFusionActivity/,
  );
  assert.match(
    appSource,
    /async function reattachPredictionTask[\s\S]*?!isFaultVolumeModelId\(modelId\)[\s\S]*?interpretationTaskId !== "alignment"[\s\S]*?!preparedViewParentId[\s\S]*?已忽略旧版未绑定融合视图的在途预测/,
  );
});

test("prediction-page entry accepts PreparedView or the FaultSeg SourceSnapshot route", () => {
  assert.match(
    appSource,
    /const predictionEntryReady = computed\(\(\) =>[\s\S]*?preparedViewReady\.value \|\| faultSegPublicEntryAvailable\.value/,
  );
  assert.match(
    appSource,
    /function selectView[\s\S]*?nextView === "prediction" && !predictionEntryReady\.value[\s\S]*?activeView\.value = "preparation"/,
  );
  assert.match(
    appSource,
    /function syncViewFromHash[\s\S]*?nextView === "prediction" && !predictionEntryReady\.value[\s\S]*?#preparation/,
  );
});
