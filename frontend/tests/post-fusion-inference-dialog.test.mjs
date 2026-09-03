import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const dialogSource = await readFile(
  new URL("../src/components/workflow/PostFusionInferenceDialog.vue", import.meta.url),
  "utf8",
);

test("post-fusion dialog is bound to the complete fusion lineage", () => {
  assert.match(dialogSource, /snapshotId: string/);
  assert.match(dialogSource, /registrationTaskId: string/);
  assert.match(dialogSource, /preparedViewId: string/);
  assert.match(dialogSource, /readyWellCount\?: number/);
  assert.match(dialogSource, /snapshotId: props\.snapshotId/);
  assert.match(dialogSource, /registrationTaskId: props\.registrationTaskId/);
  assert.match(dialogSource, /preparedViewId: props\.preparedViewId/);
  assert.match(dialogSource, /readyWellCount: normalizedReadyWellCount\.value/);
  assert.match(dialogSource, />SourceSnapshot</);
  assert.match(dialogSource, />PreparedView</);
});

test("post-fusion dialog offers the legacy model route and unified LayerPulse route", () => {
  assert.match(dialogSource, />融合标定完成/);
  assert.match(dialogSource, />单任务推理模型</);
  assert.match(dialogSource, />LayerPulse 多模态统一智能解释</);
  assert.match(dialogSource, /唯一共享 Backbone 一次生成构造、地层、沉积、属性和井震推理结果/);
  assert.match(dialogSource, /保留原有参数与成果流程/);
  assert.match(dialogSource, /单 checkpoint/);
  assert.match(dialogSource, /一次 forward/);
});

test("post-fusion choices emit the requested events with immutable lineage context", () => {
  assert.match(dialogSource, /\(event: "original", context: FusionInferenceContext\): void/);
  assert.match(dialogSource, /\(event: "layerpulse", context: FusionInferenceContext\): void/);
  assert.match(dialogSource, /\(event: "close"\): void/);
  assert.match(dialogSource, /emit\("original", inferenceContext\.value\)/);
  assert.match(dialogSource, /emit\("layerpulse", inferenceContext\.value\)/);
  assert.match(dialogSource, /emit\("close"\)/);
});

test("post-fusion dialog remains accessible, modal, responsive, and locally scoped", () => {
  assert.match(dialogSource, /<dialog[\s\S]*?class="post-fusion-inference-dialog"/);
  assert.match(dialogSource, /aria-labelledby="post-fusion-inference-title"/);
  assert.match(dialogSource, /aria-describedby="post-fusion-inference-description"/);
  assert.match(dialogSource, /@cancel\.prevent="requestClose"/);
  assert.match(dialogSource, /element\.showModal\(\)/);
  assert.match(dialogSource, /originalChoice\.value\?\.focus\(\)/);
  assert.match(dialogSource, /<style scoped>/);
  assert.match(dialogSource, /@media \(max-width: 720px\)/);
  assert.match(dialogSource, /@media \(prefers-reduced-motion: reduce\)/);
});
