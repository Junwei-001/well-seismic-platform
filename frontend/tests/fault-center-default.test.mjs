import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const productThemeSource = await readFile(new URL("../src/product-theme.css", import.meta.url), "utf8");
const platformSource = await readFile(new URL("../src/domain/platform.ts", import.meta.url), "utf8");
const modelPresentationSource = await readFile(new URL("../src/domain/modelPresentation.ts", import.meta.url), "utf8");

test("FaultSeg single-task launch exposes only formal scopes", () => {
  assert.match(appSource, /FAULT_VOLUME_MODEL_IDS = new Set\(\["faultseg_3d", "faultnet_china_field"\]\)/);
  assert.match(appSource, /type FaultSegScope = "center_block_1" \| "full_volume"/);
  assert.match(appSource, /const faultSegScope = ref<FaultSegScope>\(normalizeFaultSegScope\(undefined\)\)/);
  assert.match(appSource, /const faultSegModelInputReady = computed\(\(\) => Boolean\([\s\S]*?faultSegFormalInputReady\.value,[\s\S]*?\)\)/);
  assert.doesNotMatch(appSource, /debug_crop/);
  assert.doesNotMatch(appSource, /adaptive_small_volume_candidate/);
  assert.doesNotMatch(appSource, /faultSegAdaptivePatch|faultSegAdaptiveCandidateReady/);
  assert.doesNotMatch(appSource, /faultSegDebugCropSize/);
  assert.doesNotMatch(appSource, /faultSegDebugPatchSize/);
  assert.doesNotMatch(appSource, /faultSegDebugThreshold/);
  assert.doesNotMatch(appSource, /小测区自适应|小测区实验候选|实验候选 · 可运行/);
  assert.match(appSource, /工区中心单块（默认）/);
  assert.match(appSource, /selectedPredictionModelId\.value === "faultseg_3d"[\s\S]*?\? "center_block_1"[\s\S]*?: "full_volume"/);
  assert.match(appSource, /v-if="selectedPredictionModelId === 'faultseg_3d'"[\s\S]*?value="center_block_1"/);
  assert.match(appSource, /三轴中心确定性截取 1 个完整 128³ 子体积/);
  assert.doesNotMatch(appSource, /value="representative_grid_128"/);
  assert.doesNotMatch(appSource, /128 个代表块（默认）/);
  assert.match(appSource, /value="full_volume"/);
  assert.match(appSource, /当前至少一个空间轴不足 128；单任务断层识别仅支持完整 128³ 中心块或全区重建，当前数据不可运行/);
  assert.match(appSource, /:disabled="predictionBusy[\s\S]*?!predictionInputReady/);
  assert.match(appSource, /约 20–30 分钟/);
  assert.match(appSource, /window\.confirm\("全区断层预测通常约需 20–30 分钟/);
  assert.doesNotMatch(appSource, /fault-seg-formal-scope|REPRESENTATIVE GRID · 128块快速预测/);
  assert.doesNotMatch(appSource, /surface-seg-formal-scope|正式推理范围/);
  assert.doesNotMatch(appSource, /model-runtime-notice|<strong>运行范围<\/strong>/);
  assert.doesNotMatch(appSource, /<span>预测范围<\/span>/);
  assert.match(appSource, /窗口重叠[\s\S]*?full_volume[\s\S]*?64 × 64 × 64[\s\S]*?0 × 0 × 0/);
  assert.match(appSource, /中心位置与 128³ 窗口固定；单块直接输出，不做重叠融合/);
  assert.match(appSource, /中心单块直接输出/);
  assert.match(appSource, /所有视图来自同一重叠融合概率体/);
  assert.doesNotMatch(appSource, /representative_grid_36/);
  assert.doesNotMatch(appSource, /36 个独立/);
  assert.match(platformSource, /默认预测工区三轴中心的单个 128³ 完整块/);
  assert.match(modelPresentationSource, /默认工区中心单个128³完整块/);
  assert.match(modelPresentationSource, /faultnet_china_field:[\s\S]*?逐窗 min-max/);
  assert.match(platformSource, /model_ids: \["faultseg_3d", "faultnet_china_field"\]/);
  assert.doesNotMatch(platformSource, /36 个独立断层代表块/);
  assert.doesNotMatch(modelPresentationSource, /36 个独立代表块/);
});

test("FaultSeg submits only the SourceSnapshot seismic parent", () => {
  assert.match(appSource, /faultSegSnapshotSeismicSources = computed/);
  assert.match(appSource, /faultSegSnapshotSeismicSources\.value\.length === 1/);
  assert.match(appSource, /dataSnapshotTaskId\.value[\s\S]*?selectedPredictionSource\.value\?\.path === faultSegSnapshotSeismicSources\.value\[0\]\?\.path/);
  assert.match(appSource, /if \(isFaultSegModel\.value && !faultSegSnapshotSourceReady\.value\)/);
  assert.match(appSource, /source_task_id: sourceTaskId \|\| undefined/);
  assert.match(appSource, /\.\.\.\(runIntent\.isFaultSegModel[\s\S]*?\? \{\}[\s\S]*?: \{[\s\S]*?registration_task_id:[\s\S]*?prepared_view_task_id:/);
  assert.match(appSource, /!isFaultVolumeModelId\(modelId\)[\s\S]*?&& !preparedViewParentId/);
  assert.match(appSource, /if \(!isFaultVolumeModelId\(result\.prediction\.model_id\)\)[\s\S]*?registrationTaskId\.value = result\.registration_task_id/);
});

test("restoration accepts only formal scopes while keeping old experimental results read-only", () => {
  assert.match(appSource, /function isFullVolumeFaultPrediction/);
  assert.match(appSource, /function isCenterBlockFaultPrediction/);
  assert.match(appSource, /function isRepresentativeGrid128FaultPrediction/);
  assert.match(appSource, /return isFullVolumeFaultPrediction\(result\)[\s\S]*?\|\| isCenterBlockFaultPrediction\(result\);/);
  assert.doesNotMatch(appSource, /isAdaptiveSmallVolumeFaultPrediction/);
  assert.match(appSource, /restoredFaultScope === "full_volume"/);
  assert.match(appSource, /restoredSpatialReceipt\?\.is_full_survey === true/);
  assert.match(appSource, /faultSegScope\.value = normalizeFaultSegScope\(result\.prediction\.inference\?\.faultseg_scope\)/);
  assert.match(appSource, /历史代表块结果仅供查看；请重新运行中心单块或全区识别/);
  assert.match(appSource, /工区中心 128³ 结果已就绪/);
  assert.match(appSource, /旧实验结果 · 只读/);
  assert.match(appSource, /历史非正式结果仅供查看/);
  assert.match(appSource, /历史128块结果 · 只读/);
  assert.doesNotMatch(appSource, /小测区自适应/);
  assert.doesNotMatch(appSource, /isHistoricalFullFaultPrediction/);
});

test("resource exhaustion remains a visible failed full-survey run", () => {
  assert.match(appSource, /function predictionFailureMessage\(error: unknown, faultSegFullVolume: boolean\)/);
  assert.match(appSource, /全工区断层识别因计算资源不足未完成/);
  assert.match(appSource, /局部中间产物不会被标记为完成成果/);
  assert.match(appSource, /<p v-if="errorMessage" class="error-message" role="alert">/);
  assert.doesNotMatch(appSource, /声明 ROI 的完整成果/);
});

test("prediction polling keeps the durable task running across transient timeouts", () => {
  const waitForPrediction = appSource.match(
    /async function waitForPrediction[\s\S]*?\n}\n\nasync function reattachPredictionTask/,
  )?.[0] || "";
  assert.match(waitForPrediction, /while \(!componentUnmounted\)/);
  assert.match(waitForPrediction, /isRetryableTaskStatusError\(error\)/);
  assert.match(waitForPrediction, /仅状态查询暂时未返回，正在重试/);
  assert.match(waitForPrediction, /这不代表模型断线/);
  assert.match(waitForPrediction, /continue;/);
  assert.doesNotMatch(waitForPrediction, /Date\.now\(\) < deadline/);
  assert.doesNotMatch(waitForPrediction, /模型推理超时/);
});

test("downstream task directory uses compact human-facing ordinals", () => {
  assert.match(appSource, /v-for="\(task, index\) in predictionTaskDefinitions"/);
  assert.match(appSource, /<span>\{\{ index \+ 1 \}\}<\/span>/);
  assert.doesNotMatch(appSource, /String\(task\.order\)\.padStart\(2, '0'\)/);
});

test("prominent information cards use light product surfaces", () => {
  assert.match(productThemeSource, /\.compact-model-flow > header \{[\s\S]*?background: linear-gradient\(112deg, #edf6ff/);
  assert.match(productThemeSource, /\.pipeline-next-step \{[\s\S]*?background: linear-gradient\(115deg, #f1f7fd/);
  assert.match(productThemeSource, /\.result-primary-visualization,[\s\S]*?background: linear-gradient\(112deg, #edf6ff/);
  assert.match(productThemeSource, /\.prediction-live-overlay \{[\s\S]*?background: rgb\(248 252 255 \/ 94%\)/);
});
