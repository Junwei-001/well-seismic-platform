import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const apiSource = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const domainSource = await readFile(new URL("../src/domain/layerPulse.ts", import.meta.url), "utf8");

function functionBody(name, nextName) {
  const start = appSource.indexOf(`function ${name}`);
  const end = appSource.indexOf(`function ${nextName}`, start + 1);
  assert.notEqual(start, -1, `${name} should exist`);
  assert.notEqual(end, -1, `${nextName} should follow ${name}`);
  return appSource.slice(start, end);
}

test("fusion completion, not data registration, opens the inference destination dialog", () => {
  const finalize = functionBody("finalizeDataPreparationTask", "reattachDataPreparationTask");
  assert.match(finalize, /await offerRuntimeContractReview\(result, dataSnapshotTaskId\.value \|\| completedTaskId\)/);
  assert.doesNotMatch(finalize, /openPostFusionInferenceDestination/);

  const confirm = functionBody("confirmRuntimeContractReview", "clearSourceContractDraft");
  assert.doesNotMatch(confirm, /openPostFusionInferenceDestination/);

  const context = functionBody("currentPostFusionInferenceContext", "postFusionInferenceContextIsCurrent");
  assert.match(context, /preparedViewReady\.value/);
  assert.match(context, /sampleResult\.value\?\.registration_task_id !== registrationId/);
  assert.match(context, /preparedView\?\.source_snapshot_id !== snapshotId/);

  const sample = functionBody("runSampleBuilding", "startDefaultWellSeismicWorkflow");
  assert.ok(
    sample.indexOf("sampleResult.value = completed")
      < sample.indexOf("openPostFusionInferenceDestination()"),
    "the chooser must open only after the new PreparedView is installed",
  );
  const reconnect = functionBody("reattachWorkflowTask", "recoverRememberedTaskReference");
  assert.match(reconnect, /task\.task_type === "sample_building"[\s\S]*?openPostFusionInferenceDestination\(\)/);
  const nextAction = functionBody("handlePreparationNextAction", "openReleaseRunner");
  assert.match(nextAction, /if \(preparedViewReady\.value\) openPostFusionInferenceDestination\(\);/);
  assert.match(appSource, /label: "选择推理方式"/);
});

test("LayerPulse submission binds the exact fusion lineage", () => {
  const run = functionBody("runLayerPulse", "selectLayerPulseOutput");
  assert.match(domainSource, /LAYER_PULSE_TASK_ID = "layerpulse"/);
  assert.match(domainSource, /LAYER_PULSE_MODEL_ID = "layerpulse_geochronograph_f3x200cf"/);
  assert.match(run, /task_id: LAYER_PULSE_TASK_ID/);
  assert.match(run, /model_id: LAYER_PULSE_MODEL_ID/);
  assert.match(run, /source_task_id: snapshotId/);
  assert.match(run, /const fusionContext = currentPostFusionInferenceContext\(\)/);
  assert.match(run, /registration_task_id: fusionContext\.registrationTaskId/);
  assert.match(run, /prepared_view_task_id: fusionContext\.preparedViewId/);
  assert.match(run, /seismic_path: source\.path/);
  assert.match(run, /device: "cuda"/);
  assert.match(run, /crop_size: \[128, 128, 128\]/);
  assert.match(run, /patch_size: \[128, 128, 128\]/);
  assert.match(run, /output_profile: "platform_preview"/);
  assert.match(run, /preview: true/);
  assert.doesNotMatch(run, /threshold:/);

  const genericReattach = functionBody("reattachPredictionTask", "waitForAcceptedRegistration");
  assert.ok(
    genericReattach.indexOf("isLayerPulsePredictionTask(task)")
      < genericReattach.indexOf("preparedViewParentId"),
    "LayerPulse must branch before the legacy PreparedView gate",
  );
});

test("LayerPulse state and refresh recovery stay independent from legacy prediction state", () => {
  assert.match(appSource, /LAST_LAYERPULSE_TASK_STORAGE_KEY = "strata_vision_last_layerpulse_task"/);
  assert.match(appSource, /const layerPulseTaskState = ref<LayerPulseTaskState>/);
  assert.match(appSource, /await restoreRememberedLayerPulseTask\(\)/);
  assert.match(appSource, /layerPulseTaskStateForCurrentSnapshot/);
  assert.match(appSource, /layerPulseSourceSnapshotId\.value !== snapshotId[\s\S]*?removeItem\(LAST_LAYERPULSE_TASK_STORAGE_KEY\)/);
  assert.match(appSource, /window\.sessionStorage\.setItem\(LAST_LAYERPULSE_TASK_STORAGE_KEY, created\.task_id\)/);
  assert.doesNotMatch(
    functionBody("runLayerPulse", "selectLayerPulseOutput"),
    /predictionResult\.value|predictionTaskId\.value|LAST_PREDICTION_TASK_STORAGE_KEY/,
  );
});

test("LayerPulse reconciles the durable project snapshot without interrupting active work", () => {
  const reconciliation = functionBody(
    "reconcileProjectActiveSnapshotForInference",
    "openPostFusionInferenceDestination",
  );
  assert.match(reconciliation, /const projectCatalog = await getProjects\(\)/);
  assert.match(reconciliation, /const activeSnapshotId = project\?\.active_snapshot_id \|\| ""/);
  assert.match(reconciliation, /activeSnapshotId === dataSnapshotTaskId\.value/);
  assert.match(reconciliation, /activeSnapshotReconciliationBlockedByRunningTask\(\)/);
  assert.match(reconciliation, /await restoreLatestDurableWorkflow\(\{ preferLatestSnapshot: true \}\)/);
  assert.match(reconciliation, /dataSnapshotTaskId\.value !== activeSnapshotId/);

  const chooser = functionBody("openPostFusionInferenceDestination", "choosePostFusionOriginal");
  assert.ok(
    chooser.indexOf('await reconcileProjectActiveSnapshotForInference("推理方式选择")')
      < chooser.indexOf("currentPostFusionInferenceContext()"),
    "the inference chooser must reconcile the durable snapshot before reading fusion lineage",
  );

  const run = functionBody("runLayerPulse", "selectLayerPulseOutput");
  assert.ok(
    run.indexOf('await reconcileProjectActiveSnapshotForInference("LayerPulse 统一解释")')
      < run.indexOf("currentPostFusionInferenceContext()"),
    "LayerPulse must reconcile the durable snapshot before binding submission ids",
  );

  const remembered = functionBody("restoreRememberedLayerPulseTask", "runLayerPulse");
  assert.match(remembered, /const preserveCurrentSnapshot = Boolean\(/);
  assert.match(remembered, /task\.status !== "queued"[\s\S]*?task\.status !== "running"/);
  assert.match(remembered, /restoreSourceSnapshot: !preserveCurrentSnapshot/);
  assert.match(remembered, /但未覆盖当前项目活动快照/);
});

test("support and visualization use registered receipts and the unified platform viewer", () => {
  assert.match(appSource, /source\.model_compatibility\?\.\[LAYER_PULSE_MODEL_ID\]/);
  assert.match(appSource, /snapshot_id: snapshotId/);
  assert.match(appSource, /未伪造 LayerPulse ready 状态/);
  assert.match(appSource, /supports_snapshot_wells === true/);
  assert.match(appSource, /当前预览桥接尚未消费 MD 与轨迹张量/);
  assert.match(apiSource, /task_catalog\?: Array</);
  assert.match(apiSource, /preview_artifact_key\?: string \| null/);
  assert.match(appSource, /const layerPulseBaseVisualizationUrl = computed\(\(\) => \{[\s\S]*return visualizationUrl\.value;/);
  assert.match(appSource, /standard_result_bundle\?\.visualization/);
  assert.match(appSource, /layerpulse_output=\$\{encodeURIComponent\(layerPulseSelectedOutputKey\.value\)\}/);
  assert.match(appSource, /统一数据可视化\?task_id=/);
});

test("LayerPulse joins each available output to standard downloads with historical fallbacks", () => {
  assert.match(apiSource, /primary_download_artifact_key\?: string \| null/);
  assert.match(apiSource, /download_artifact_keys\?: Array<\{/);
  assert.match(apiSource, /class_legend_artifact_key\?: string \| null/);
  assert.match(appSource, /const layerPulseOutputDownloads = computed<Record<string, LayerPulseOutputDownloads>>/);
  assert.match(appSource, /new Set\(layerPulseAvailableOutputKeys\.value\)/);
  assert.match(appSource, /prediction\.standard_result_bundle\?\.downloads\.artifacts \|\| \[\]/);
  assert.match(appSource, /entry\.primary_download_artifact_key/);
  assert.match(appSource, /entry\.class_legend_artifact_key/);
  assert.match(appSource, /artifactsByOutputKey\.get\(artifactKey\)/);
  assert.match(appSource, /layerpulse-exports\/\$\{encodedOutputKey\}\.sgy/);
  assert.match(appSource, /layerpulse-exports\/\$\{encodedOutputKey\}\.csv/);
  assert.match(appSource, /rawNpy: rawArtifact/);
  assert.match(appSource, /:output-downloads="layerPulseOutputDownloads"/);
});

test("LayerPulse renders as a first-level workbench and the post-fusion choice is mounted", () => {
  assert.match(appSource, /activeView === 'layerpulse'[\s\S]*?<LayerPulseWorkbench/);
  assert.match(appSource, /<PostFusionInferenceDialog/);
  assert.match(appSource, /:registration-task-id="postFusionInferenceContext\?\.registrationTaskId \|\| ''"/);
  assert.match(appSource, /@original="choosePostFusionOriginal"/);
  assert.match(appSource, /@layerpulse="choosePostFusionLayerPulse"/);
});
