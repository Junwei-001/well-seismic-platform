import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const modelCenterSource = await readFile(
  new URL("../src/components/models/ModelCenter.vue", import.meta.url),
  "utf8",
);

function appViewBranch(viewId) {
  const escapedViewId = viewId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const branch = appSource.match(
    new RegExp(`<template\\s+v-else-if="activeView === '${escapedViewId}'">([\\s\\S]*?)<\\/template>`),
  );
  assert.ok(branch, `missing App view branch: ${viewId}`);
  return branch[1];
}

test("model center selects the canonical LayerPulse capability instead of iterating generic models", () => {
  assert.match(modelCenterSource, /LAYER_PULSE_MODEL_ID/);
  assert.match(modelCenterSource, /model\.id\s*===\s*LAYER_PULSE_MODEL_ID/);
  assert.doesNotMatch(modelCenterSource, /v-for="model in publicModels"|publicModels|direct12bParameterBreakdown/);
});

test("LayerPulse parameter, shared feature and head contracts drive the visible overview", () => {
  assert.match(modelCenterSource, /layerPulseModelContract\.parameterCount/);
  assert.match(modelCenterSource, /layerPulseModelContract\.fFinalChannels/);
  assert.match(modelCenterSource, /layerPulseModelContract\.headCount/);
  assert.match(modelCenterSource, /(?:参数|PARAMETER)/i);
  assert.match(modelCenterSource, /F_final/);
  assert.match(modelCenterSource, /(?:输出头|任务头|HEAD)/i);
});

test("all task heads are rendered from the canonical LayerPulse output catalog", () => {
  assert.match(modelCenterSource, /layerPulseOutputCatalog/);
  assert.match(modelCenterSource, /v-for="output\s+in\s+layerPulseOutputCatalog"/);
  assert.match(modelCenterSource, /output\.channels/);
  assert.match(modelCenterSource, /output\.(?:shortName|name)/);
});

test("legacy generic model, fusion strategy and downstream contract panels are removed", () => {
  for (const legacyMarker of [
    "模型组件",
    "井震融合方案",
    "下游解释任务契约",
    "模型插件入口",
    "fusion_strategies",
    "prediction_tasks",
  ]) {
    assert.equal(modelCenterSource.includes(legacyMarker), false, `${legacyMarker} remains in ModelCenter`);
  }
});

test("models route renders only the LayerPulse model center and not the release catalog", () => {
  const modelsBranch = appViewBranch("models");
  assert.match(modelsBranch, /<ModelCenter\s+:capabilities="capabilities"\s*\/>/);
  assert.doesNotMatch(modelsBranch, /<ReleaseCenter\b/);
});

test("model center removes the duplicate page introduction and model description", () => {
  assert.match(
    appSource,
    /v-if="activeView !== 'overview' && activeView !== 'models' && activeView !== 'assistant'"/,
  );
  assert.doesNotMatch(modelCenterSource, /class="lp-description"/);
});

test("the parameter visual is enlarged and the model center fills the available page height", () => {
  assert.match(modelCenterSource, /\.layerpulse-model-center\s*\{[\s\S]*?height:\s*100%/);
  assert.match(modelCenterSource, /\.lp-parameter-copy strong\s*\{[^}]*font-size:\s*72px/);
});

test("the architecture dashboard responds to smaller screens and respects reduced motion", () => {
  assert.match(modelCenterSource, /@media\s*\(max-width:\s*\d+px\)/);
  assert.match(modelCenterSource, /@media\s*\(prefers-reduced-motion:\s*reduce\)/);
  assert.match(modelCenterSource, /@keyframes/);
});

test("the architecture topology stays centered with bounded task-head height", () => {
  assert.match(modelCenterSource, /\.lp-model-flow\s*\{[\s\S]*?align-content:\s*center/);
  assert.match(modelCenterSource, /\.lp-head-network\s*\{[^}]*max-height:\s*144px/);
  assert.match(modelCenterSource, /\.lp-head-node\s*\{[^}]*align-content:\s*center/);
});

test("core metrics and architecture-stage typography remain prominently readable", () => {
  assert.match(modelCenterSource, /\.lp-metrics span\s*\{[^}]*font-size:\s*13px/);
  assert.match(modelCenterSource, /\.lp-input-stage strong\s*\{[^}]*font-size:\s*14px/);
  assert.match(modelCenterSource, /\.lp-backbone-stage > strong\s*\{[^}]*font-size:\s*26px/);
  assert.match(modelCenterSource, /\.lp-feature-stage > strong\s*\{[^}]*font-size:\s*28px/);
});
