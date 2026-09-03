import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const platformSource = await readFile(new URL("../src/domain/platform.ts", import.meta.url), "utf8");
const dialogSource = await readFile(new URL("../src/components/workflow/PostFusionInferenceDialog.vue", import.meta.url), "utf8");
const workbenchSource = await readFile(new URL("../src/components/layerpulse/LayerPulseWorkbench.vue", import.meta.url), "utf8");
const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");

test("LayerPulse is an independent primary workflow destination", () => {
  assert.match(platformSource, /\| "layerpulse"/);
  assert.match(platformSource, /id: "layerpulse", label: "LayerPulse 多模态融合基础模型", section: "workflow"/);
  assert.match(platformSource, /layerpulse: \{/);
  assert.match(platformSource, /title: "LayerPulse 多模态融合基础模型"/);
  assert.match(platformSource, /以相对地质时间为坐标、以构造连通域为传播拓扑、以井曲线为高分辨率语义锚点/);
});

test("post-fusion modal offers exactly the single-task and LayerPulse routes", () => {
  assert.match(dialogSource, /<dialog[\s\S]*?class="post-fusion-inference-dialog"/);
  assert.match(dialogSource, /@cancel\.prevent/);
  assert.match(dialogSource, /emit\("original", inferenceContext\.value\)/);
  assert.match(dialogSource, /emit\("layerpulse", inferenceContext\.value\)/);
  assert.match(dialogSource, />单任务推理模型</);
  assert.match(dialogSource, />LayerPulse 多模态统一智能解释</);
  assert.match(dialogSource, /registrationTaskId: props\.registrationTaskId/);
});

test("LayerPulse uses the same jumpable secondary-directory structure as single-task inference", () => {
  assert.match(appSource, /ref<"primary" \| "prediction" \| "layerpulse">\("primary"\)/);
  assert.match(appSource, /aria-label="LayerPulse 任务二级目录"/);
  assert.match(appSource, /<strong>返回一级目录<\/strong>/);
  assert.match(appSource, /v-for="\(output, index\) in layerPulseOutputCatalog"/);
  assert.match(appSource, /data-layerpulse-output-key/);
  assert.match(appSource, /@click="selectLayerPulseOutput\(output\.key\)"/);
  assert.match(appSource, /handleLayerPulseOutputTabKeydown/);
  assert.match(appSource, /layerPulseOutputTabStatusLabel\(output\.key\)/);
  assert.match(appSource, /activeView === 'prediction' \|\| activeView === 'layerpulse'/);
  assert.match(appSource, /activeView === 'visualization' \|\| activeView === 'prediction' \|\| activeView === 'layerpulse'/);
});

test("workbench keeps support gating internal and renders independent task state, active downloads, and the platform viewer", () => {
  assert.match(workbenchSource, /summarizeLayerPulseSupport\(matchingReceipt\.value\)/);
  assert.match(workbenchSource, /support\.value\.status !== "blocked"/);
  assert.doesNotMatch(workbenchSource, /当前区块支持能力/);
  assert.doesNotMatch(workbenchSource, /class="support-panel"/);
  assert.match(workbenchSource, /layerPulseTaskStatusLabels\[taskState\.status\]/);
  assert.match(workbenchSource, /taskState\.taskId/);
  assert.doesNotMatch(workbenchSource, /class="output-directory"/);
  assert.match(workbenchSource, /class="active-output-downloads"/);
  assert.match(workbenchSource, />下载 SEG-Y<\/a>/);
  assert.match(workbenchSource, />类别码表 CSV<\/a>/);
  assert.match(workbenchSource, />原始 NPY<\/a>/);
  assert.match(workbenchSource, /target="_blank"[\s\S]*?rel="noopener"/);
  assert.match(workbenchSource, /baseVisualizationUrl/);
  assert.match(workbenchSource, /resultVisualizationUrl/);
  assert.match(workbenchSource, /<iframe[\s\S]*?:src="viewerUrl"/);
  assert.doesNotMatch(workbenchSource, /<img\b/);
  assert.doesNotMatch(workbenchSource, /原平台可视化工作台/);
  assert.doesNotMatch(workbenchSource, /三维体/);
  assert.doesNotMatch(workbenchSource, /正交切片/);
  assert.doesNotMatch(workbenchSource, /图层切换/);
  assert.doesNotMatch(workbenchSource, /preview-coverage-note/);
  assert.doesNotMatch(workbenchSource, /确定性预览覆盖：当前结果仅覆盖本次选定子体，不代表完整工区推理；具体范围与锚点以工作台为准/);
  assert.doesNotMatch(workbenchSource, /视角、切片和图层交互由内嵌工作台提供/);
  assert.match(workbenchSource, /grid-template-rows: minmax\(0, 1fr\)/);
  assert.match(workbenchSource, /`\$\{activeOutputDisplayName\}可视化`/);
  assert.match(workbenchSource, /完整分类 logits/);
  assert.match(workbenchSource, /沿类别维直接 argmax/);
  assert.match(workbenchSource, /不使用 sigmoid、类别阈值或前端连通域清理/);
});

test("workbench states the unique-forward contract and does not present RGT display surfaces as another Head", () => {
  assert.match(appSource, /单 checkpoint/);
  assert.match(appSource, /唯一共享 Backbone/);
  assert.match(appSource, /一次 forward · \{\{ layerPulseModelContract\.headCount \}\} 项/);
  assert.match(appSource, /F_final \{\{ layerPulseModelContract\.fFinalChannels \}\}/);
  assert.match(appSource, /默认无时深表/);
  assert.match(workbenchSource, /class="task-actions"[\s\S]*?class="task-status-copy"/);
  assert.match(workbenchSource, /width: 100%; min-height: 58px/);
  assert.match(workbenchSource, /grid-template-columns: minmax\(330px, 360px\) minmax\(540px, 1fr\)/);
  assert.match(workbenchSource, /height: calc\(100dvh - var\(--topbar-height\) - 36px\); min-width: 0; min-height: 640px/);
  assert.doesNotMatch(workbenchSource, /880px/);
  assert.match(workbenchSource, /position: sticky; top: calc\(var\(--topbar-height\) \+ 14px\)/);
  assert.doesNotMatch(workbenchSource, /max-height: 680px/);
  assert.match(workbenchSource, /\.viewer-panel iframe \{ display: block; width: 100%; height: 100%; min-height: 0;/);
  assert.match(workbenchSource, /RGT 派生等时面（展示）/);
  assert.match(workbenchSource, /仅用于空间展示，不是独立层位 Head/);
  assert.match(workbenchSource, /模型仍为 11 个 Head，不包含独立层位 Head/);
});

test("both new surfaces keep their styles locally scoped and responsive", () => {
  assert.match(dialogSource, /<style scoped>/);
  assert.match(workbenchSource, /<style scoped>/);
  assert.match(dialogSource, /@media \(max-width: 720px\)/);
  assert.match(workbenchSource, /@media \(max-width: 820px\)/);
  assert.match(dialogSource, /prefers-reduced-motion/);
  assert.match(workbenchSource, /prefers-reduced-motion/);
});
