import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const platformSource = await readFile(new URL("../src/domain/platform.ts", import.meta.url), "utf8");

test("legacy workflow routes resolve to the two merged workbenches", () => {
  assert.match(platformSource, /type PreparationScreen = "input" \| "pipeline" \| "fusion"/);
  assert.match(platformSource, /id: "preparation", label: "数据与融合"/);
  assert.match(platformSource, /id: "prediction", label: "单任务推理模型（共享井震融合基座）"/);
  assert.ok(
    platformSource.indexOf('id: "layerpulse"') < platformSource.indexOf('id: "prediction"'),
    "the single-task directory entry must follow the multimodal LayerPulse model",
  );
  assert.doesNotMatch(platformSource, /id: "samples", label:/);
  assert.doesNotMatch(platformSource, /id: "visualization", label:/);
  assert.doesNotMatch(platformSource, /id: "evaluation", label: "成果导出"/);
  assert.doesNotMatch(appSource, /selectView\('evaluation'\)[^\n]*验收与导出/);
  assert.match(appSource, /if \(view === "samples"\) return "preparation"/);
  assert.match(appSource, /if \(view === "visualization"\) return "prediction"/);
  assert.match(appSource, /activeView === 'preparation' && preparationScreen === 'fusion'/);
  assert.match(appSource, /const initialPreparationIntent:[\s\S]*?=== "samples"/);
  assert.match(appSource, /replayInitialViewIntent\(\);\s*\}\);/);
});

test("single-task inference uses its product name in breadcrumbs, landing path, and secondary directory", () => {
  assert.match(platformSource, /preparation: \{\s*eyebrow: "数据与融合"/);
  assert.match(platformSource, /prediction: \{\s*eyebrow: "单任务推理模型（共享井震融合基座）"/);
  assert.match(platformSource, /title: "单任务推理模型（共享井震融合基座）"/);
  assert.match(appSource, /<span>单任务推理目录<\/span><strong>共享井震融合基座<\/strong>/);
  assert.match(appSource, /selectView\('layerpulse'\)[^\n]*<span>02<\/span><strong>LayerPulse 多模态融合基础模型<\/strong>/);
  assert.match(appSource, /selectView\('prediction'\)[^\n]*<span>03<\/span><strong>单任务推理模型（共享井震融合基座）<\/strong>/);
  assert.match(appSource, /<strong>单任务推理模型（共享井震融合基座）<\/strong><small>配置 · 运行 · 联动查看<\/small>/);
  assert.doesNotMatch(platformSource, /预测与可视化/);
  assert.doesNotMatch(appSource, /预测与可视化/);
  assert.doesNotMatch(appSource, /工作流\s*0[12]/);
  assert.doesNotMatch(platformSource, /工作流\s*0[12]/);
});

test("prediction viewer keeps base and standard-result URLs separate", () => {
  assert.match(appSource, /predictionCanvasMode\.value === "base"\) return visualizationUrl\.value/);
  assert.match(appSource, /predictionEmbeddedVisualizationUrl\.value \|\| predictionVisualizationUrl\.value/);
  assert.match(appSource, /platform_viewer_url/);
  assert.match(appSource, /visualizationUrlWithEmbed\(platformViewerUrl, 1\)/);
  assert.match(appSource, /visualizationUrlWithEmbed\(platformViewerUrl, 0\)/);
  assert.match(appSource, /if \(isWellSequenceResult\.value\) return `\$\{predictionVisualizationUrl\.value\}&asset=0`/);
  assert.match(appSource, /class="prediction-live-frame"/);
  assert.match(appSource, /:src="predictionWorkbenchUrl"/);
  assert.match(appSource, /function compactModelSpecPresentationName/);
  assert.match(appSource, /compactModelSpecPresentationName\(model\)/);
});

test("prediction orchestration locks new submissions but permits read-only task switching", () => {
  assert.match(appSource, /const predictionBusy = computed\(\(\) => predictionOrchestrationRunning\.value \|\| predictionRunning\.value\)/);
  assert.doesNotMatch(appSource, /async function selectPredictionTask[\s\S]*?if \(predictionBusy\.value\) return;/);
  assert.doesNotMatch(appSource, /:disabled="predictionBusy"\s+@click="selectPredictionTask/);
  assert.match(appSource, /const anotherPredictionTaskRunning = computed/);
  assert.match(appSource, /const selectedEntry = predictionHistoryByTask\.value\[task\]/);
  assert.match(appSource, /if \(sourceTaskId\) \{[\s\S]*?restorePreparationForSnapshot\(sourceTaskId, \{ resetDownstream: false \}\)/);
  assert.match(appSource, /sourceTask\.task_type === "data_preparation"[\s\S]*?await restorePreparationForSnapshot\(sourceTaskId, \{ resetDownstream: false \}\)/);
  assert.match(appSource, /<fieldset[^>]+class="prediction-control-fieldset" :disabled="predictionBusy">/);
  assert.match(appSource, /model_id: runIntent\.modelId/);
  assert.match(appSource, /runIntent\.isFaultSegModel/);
  assert.match(appSource, /runIntent\.recommendedOptions/);
});

test("an offline startup keeps an explicit reconnect action", () => {
  assert.match(appSource, /function retryBackendConnection\(\)[\s\S]*?window\.location\.reload\(\)/);
  assert.match(appSource, /backendStatus === 'offline'[\s\S]*?class="service-retry"/);
});
