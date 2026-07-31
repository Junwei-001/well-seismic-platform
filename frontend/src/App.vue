<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import {
  activateTransformationDraft,
  confirmPreparationIssue,
  createDataPreparation,
  createSampleBuilding,
  createPrediction,
  demoPaths,
  getCapabilities,
  getTask,
  generateTransformationDraft,
  health,
  type BackgroundTask,
  type Capabilities,
  type DataPathsPayload,
  type DemoPaths,
  type PreparationIssue,
  type PreparationStage,
  type PredictionTaskCapability,
  type PredictionResult,
  type PredictionTaskResult,
  type TransformationDraft,
  type WellLogPreviewCurve,
  type WorkflowResult,
} from "./api";
import heroLaboratoryImage from "./assets/首屏B_明亮岩心实验室.png";
import FloatingAssistant from "./components/assistant/FloatingAssistant.vue";
import ModelCenter from "./components/models/ModelCenter.vue";
import {
  availableCurveForSlot,
  conventionalCoverage,
  conventionalCurveGroups,
  createDefaultPathGroups,
  fallbackInterpretationTasks,
  navigation,
  navigationIconPaths,
  viewMeta,
  type PathGroup,
  type PathGroupKey,
  type PredictionTaskKey,
  type PreparationScreen,
  type ViewKey,
} from "./domain/platform";

const LAST_TASK_STORAGE_KEY = "strata_vision_last_task";
const LAST_PREDICTION_TASK_STORAGE_KEY = "strata_vision_last_prediction_task";
const PATH_CONFIG_STORAGE_KEY = "strata_vision_path_config";
const TASK_POLL_TIMEOUT_MS = 30 * 60 * 1000;
let componentUnmounted = false;
const groups = ref<PathGroup[]>(createDefaultPathGroups());

const activeView = ref<ViewKey>("overview");
const backendStatus = ref<"checking" | "online" | "offline">("checking");
const backendVersion = ref("");
const capabilities = ref<Capabilities | null>(null);
const demo = ref<DemoPaths | null>(null);
const recursive = ref(true);
const lightweight = ref(true);
const useLlmFallback = ref(false);
const preparationRunning = ref(false);
const sampleRunning = ref(false);
const predictionRunning = ref(false);
const progress = ref(0);
const statusMessage = ref("等待登记数据");
const taskId = ref("");
const predictionTaskId = ref("");
const preparationScreen = ref<PreparationScreen>("input");
const preparationResult = ref<WorkflowResult | null>(null);
const sampleResult = ref<WorkflowResult | null>(null);
const predictionResult = ref<PredictionResult | null>(null);
const activePredictionTask = ref<PredictionTaskKey>("fault");
const selectedPredictionModelId = ref("faultseg_3d");
const predictionSeismicPath = ref("");
const predictionCropSize = ref<32 | 64 | 96 | 128>(32);
const predictionThreshold = ref(0.5);
const predictionDevice = ref<"auto" | "cpu" | "cuda">("auto");
const surfaceSegScope = ref<"smoke" | "full">("smoke");
const surfaceSegMaxInlines = ref(2);
const surfaceSegInlineCount = ref<number | "">("");
const surfaceSegAmplitudeMode = ref<"auto" | "training" | "robust">("auto");
const surfaceSegQueryThreshold = ref(0.35);
const surfaceSegMaskThreshold = ref(0.5);
const surfaceSegformerBatchSize = ref(2);
const surfaceMask2formerBatchSize = ref(1);
const errorMessage = ref("");
const issueFilter = ref("全部");
const confirmingIssueId = ref("");
const visualizationMode = ref<"seismic" | "logs">("seismic");
const visualizationStageElement = ref<HTMLElement | null>(null);
const visualizationFullscreen = ref(false);
const visualizationSourceTaskId = ref("");
const visualizationBaseTaskId = ref("");
const selectedSeismicAssetIndex = ref(0);
const predictionSourceTaskId = ref("");
const selectedWellLogId = ref("");
const visibleCurveIds = ref<string[]>([]);
const transformationDrafts = ref<Record<string, TransformationDraft>>({});
const generatingDraftIssueId = ref("");
const activatingDraftId = ref("");
const workflowResult = computed(() => sampleResult.value || preparationResult.value);
const predictionTaskDefinitions = computed<PredictionTaskCapability[]>(() =>
  capabilities.value?.prediction_tasks?.length
    ? capabilities.value.prediction_tasks
    : fallbackInterpretationTasks,
);
const activePredictionTaskSpec = computed(() =>
  predictionTaskDefinitions.value.find((task) => task.id === activePredictionTask.value)
  || predictionTaskDefinitions.value[0]
  || null,
);
const predictionSources = computed(() => workflowResult.value?.seismic || []);
const selectedPredictionSource = computed(() =>
  predictionSources.value.find((source) => source.path === predictionSeismicPath.value) || null,
);
const runnablePredictionModels = computed(() =>
  (capabilities.value?.models || []).filter((model) =>
    activePredictionTaskSpec.value?.runnable_model_ids.includes(model.id),
  ),
);
const selectedPredictionModel = computed(() =>
  runnablePredictionModels.value.find((model) => model.id === selectedPredictionModelId.value) || null,
);
const isSurfaceSegModel = computed(() => selectedPredictionModelId.value === "seismic_surface_seg");
const selectedModelAdapter = computed(() =>
  capabilities.value?.model_input_adapters.find((adapter) => adapter.model_id === selectedPredictionModelId.value) || null,
);
const selectedModelCompatibility = computed(() =>
  selectedPredictionSource.value?.model_compatibility?.[selectedPredictionModelId.value] || null,
);
const surfaceInlineFallbackReady = computed(() => {
  if (!isSurfaceSegModel.value || surfaceSegInlineCount.value === "") return false;
  const inlineCount = Number(surfaceSegInlineCount.value);
  const traceCount = selectedPredictionSource.value?.trace_count || 0;
  return Number.isInteger(inlineCount) && inlineCount > 0 && traceCount > inlineCount && traceCount % inlineCount === 0;
});
const predictionInputReady = computed(() => Boolean(
  selectedPredictionSource.value
  && selectedModelAdapter.value
  && (selectedModelCompatibility.value?.ready || surfaceInlineFallbackReady.value),
));
const predictionCompatibilityReason = computed(() =>
  surfaceInlineFallbackReady.value
    ? `将按显式 Inline 数 ${surfaceSegInlineCount.value} 重建 ${Number(selectedPredictionSource.value?.trace_count || 0) / Number(surfaceSegInlineCount.value)} 个 Crossline`
    : selectedModelCompatibility.value?.reason || "等待模型与数据匹配",
);
const preparation = computed(() => workflowResult.value?.preparation || null);
const wells = computed(() => workflowResult.value?.well_entities || workflowResult.value?.wells || []);
const currentMeta = computed(() => viewMeta[activeView.value]);
const registeredCount = computed(() =>
  groups.value.reduce((total, group) => total + group.paths.filter((path) => path.trim()).length, 0),
);
const filteredIssues = computed(() => {
  const issues = preparation.value?.issues || [];
  return issueFilter.value === "全部" ? issues : issues.filter((item) => item.stage === issueFilter.value);
});
const selectedStage = computed(() =>
  preparation.value?.stages.find((stage) => stage.id === issueFilter.value) || null,
);
const pendingConfirmationCount = computed(() =>
  preparation.value?.issues.filter((issue) => issue.confirmation_status === "待人工确认").length || 0,
);
const previewVolumes = computed(() => workflowResult.value?.visualization_preview?.volumes || []);
const seismicLinePreviews = computed(() => workflowResult.value?.visualization_preview?.lines2d || []);
const seismicPreviewCount = computed(() => previewVolumes.value.length + seismicLinePreviews.value.length);
const seismicInventory = computed(() =>
  workflowResult.value?.visualization_preview?.seismicInventory || workflowResult.value?.seismic || [],
);
const seismicWorkbenchAssets = computed(() => {
  const predictionOffset = visualizationSourceTaskId.value ? 1 : 0;
  const predictionAssets = visualizationSourceTaskId.value
    ? [{
        index: 0,
        kind: "预测叠加" as const,
        name: `${predictionResult.value?.task_name || activePredictionTaskSpec.value?.name || "模型"}预测叠加`,
        detail: selectedPredictionSource.value?.name || "预测体与背景地震同轴显示",
      }]
    : [];
  return [
    ...predictionAssets,
    ...previewVolumes.value.map((volume, index) => ({
      index: index + predictionOffset,
      kind: "三维体" as const,
      name: volume.name,
      detail: volume.path.replaceAll("\\", "/").split("/").at(-1) || volume.path,
    })),
    ...seismicLinePreviews.value.map((line, index) => ({
      index: previewVolumes.value.length + predictionOffset + index,
      kind: "二维测线" as const,
      name: line.name,
      detail: line.path.replaceAll("\\", "/").split("/").at(-1) || line.path,
    })),
  ];
});
const selectedSeismicAsset = computed(() =>
  seismicWorkbenchAssets.value.find((asset) => asset.index === selectedSeismicAssetIndex.value)
  || seismicWorkbenchAssets.value[0]
  || null,
);
const wellLogPreviews = computed(() => workflowResult.value?.visualization_preview?.wellLogs || []);
const selectedWellLog = computed(() =>
  wellLogPreviews.value.find((item) => item.id === selectedWellLogId.value) || wellLogPreviews.value[0] || null,
);
const groupedConventionalCurves = computed(() =>
  conventionalCurveGroups.map((group) => ({
    ...group,
    slots: group.slots.map((slot) => ({
      ...slot,
      curve: availableCurveForSlot(selectedWellLog.value, slot.curveId),
    })),
  })),
);
const activeConventionalCurveCount = computed(() =>
  groupedConventionalCurves.value.reduce(
    (count, group) => count + group.slots.filter((slot) => slot.curve && visibleCurveIds.value.includes(slot.curve.id)).length,
    0,
  ),
);
const depthTicks = computed(() => {
  const depth = selectedWellLog.value?.depth || [];
  if (!depth.length) return [];
  const tickCount = Math.min(9, depth.length);
  return Array.from({ length: tickCount }, (_, tickIndex) => {
    const index = tickCount === 1 ? 0 : Math.round((tickIndex / (tickCount - 1)) * (depth.length - 1));
    return {
      position: tickCount === 1 ? 0 : (tickIndex / (tickCount - 1)) * 100,
      value: depth[index],
    };
  });
});
const logPlotHeight = computed(() => {
  const sampleCount = selectedWellLog.value?.depth.length || 0;
  return Math.max(1680, Math.min(6200, Math.round(sampleCount * 1.7)));
});
const activeVisualizationTaskId = computed(() =>
  visualizationSourceTaskId.value || visualizationBaseTaskId.value || taskId.value,
);
const visualizationEndpoint = computed(() => activeVisualizationTaskId.value
  ? `${import.meta.env.DEV ? "http://127.0.0.1:8000" : ""}/统一数据可视化?task_id=${encodeURIComponent(activeVisualizationTaskId.value)}`
  : "");
const visualizationUrl = computed(() => visualizationEndpoint.value
  ? `${visualizationEndpoint.value}&asset=${selectedSeismicAssetIndex.value}&embed=1`
  : "");
const visualizationStandaloneUrl = computed(() => visualizationEndpoint.value
  ? `${visualizationEndpoint.value}&asset=${selectedSeismicAssetIndex.value}&embed=0`
  : "");
const predictionVisualizationUrl = computed(() => predictionTaskId.value
  ? `${import.meta.env.DEV ? "http://127.0.0.1:8000" : ""}/统一数据可视化?task_id=${encodeURIComponent(predictionTaskId.value)}&embed=1`
  : "");
const predictionOverviewUrl = computed(() =>
  predictionTaskId.value && predictionResult.value?.outputs.overview
    ? `${import.meta.env.DEV ? "http://127.0.0.1:8000" : ""}/api/v1/tasks/${encodeURIComponent(predictionTaskId.value)}/artifacts/overview`
    : "",
);
const predictionInputShape = computed(() =>
  predictionResult.value?.input.shape_zyx
  || predictionResult.value?.input.shape_ics
  || predictionResult.value?.segmentation?.shape_ics
  || [],
);
const predictionInputAxes = computed(() => predictionResult.value?.input.axes || []);
const predictionOutputEntries = computed(() =>
  Object.entries(predictionResult.value?.outputs || {}).filter((entry): entry is [string, string] => Boolean(entry[1])),
);
const mainBlockingIssue = computed(() => preparation.value?.issues.find((item) => item.blocking));
const nextAction = computed(() => {
  if (!preparation.value) return { label: "登记并准备数据", view: "preparation" as ViewKey };
  if (mainBlockingIssue.value) return { label: "处理阻断问题", view: "preparation" as ViewKey };
  if (!sampleResult.value?.matching) return { label: "构建井震样本", view: "samples" as ViewKey };
  return { label: "查看模型接口", view: "models" as ViewKey };
});

function selectView(view: ViewKey) {
  if (view === "samples") view = "preparation";
  activeView.value = view;
  window.location.hash = view;
  errorMessage.value = "";
  window.scrollTo({ top: 0, behavior: "auto" });
  void nextTick().then(() => {
    window.requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: "auto" }));
  });
}

function handleAssistantNavigation(target: string) {
  const view = target as ViewKey;
  if (view === "overview" || navigation.some((item) => item.id === view)) selectView(view);
}

function showPreparationInput() {
  preparationScreen.value = "input";
  window.scrollTo({ top: 0, behavior: "auto" });
}

function showPreparationPipeline(stage = "全部") {
  if (!preparation.value) return;
  preparationScreen.value = "pipeline";
  issueFilter.value = stage;
  void nextTick().then(() => window.scrollTo({ top: 0, behavior: "auto" }));
}

function addPath(group: PathGroup) {
  group.paths.push("");
}

function removePath(group: PathGroup, index: number) {
  group.paths.splice(index, 1);
}

function values(key: PathGroupKey): string[] {
  return groups.value
    .find((group) => group.key === key)!
    .paths.map((path) => path.trim())
    .filter(Boolean);
}

function commonPayload(): DataPathsPayload {
  return {
    seismic_paths: values("seismic"),
    log_paths: values("logs"),
    well_paths: values("wells"),
    auxiliary_paths: values("auxiliary"),
    recursive: recursive.value,
    lightweight: lightweight.value,
    use_llm_fallback: useLlmFallback.value,
  };
}

function savePathConfig(payload: DataPathsPayload) {
  window.sessionStorage.setItem(PATH_CONFIG_STORAGE_KEY, JSON.stringify(payload));
}

function restorePathConfig() {
  const saved = window.sessionStorage.getItem(PATH_CONFIG_STORAGE_KEY);
  if (!saved) return;
  try {
    const payload = JSON.parse(saved) as Partial<DataPathsPayload>;
    const mapping: Record<PathGroupKey, string[]> = {
      seismic: Array.isArray(payload.seismic_paths) ? payload.seismic_paths : [],
      logs: Array.isArray(payload.log_paths) ? payload.log_paths : [],
      wells: Array.isArray(payload.well_paths) ? payload.well_paths : [],
      auxiliary: Array.isArray(payload.auxiliary_paths) ? payload.auxiliary_paths : [],
    };
    groups.value.forEach((group) => {
      const restored = mapping[group.key].filter((path): path is string => typeof path === "string");
      group.paths = restored.length ? restored : group.optional ? [] : [""];
    });
    if (typeof payload.recursive === "boolean") recursive.value = payload.recursive;
    if (typeof payload.lightweight === "boolean") lightweight.value = payload.lightweight;
    if (typeof payload.use_llm_fallback === "boolean") useLlmFallback.value = payload.use_llm_fallback;
  } catch {
    window.sessionStorage.removeItem(PATH_CONFIG_STORAGE_KEY);
  }
}

function initializeVisualization(result: WorkflowResult) {
  const logs = result.visualization_preview?.wellLogs || [];
  const volumes = result.visualization_preview?.volumes || [];
  const lines = result.visualization_preview?.lines2d || [];
  const inventory = result.visualization_preview?.seismicInventory || result.seismic || [];
  if (logs.length) {
    selectedWellLogId.value = logs[0].id;
    visibleCurveIds.value = logs[0].curves.map((curve) => curve.id);
  }
  visualizationMode.value = volumes.length || lines.length || inventory.length ? "seismic" : "logs";
  visualizationSourceTaskId.value = "";
  visualizationBaseTaskId.value = taskId.value;
  selectedSeismicAssetIndex.value = 0;
  if (!predictionSeismicPath.value && result.seismic.length) {
    predictionSeismicPath.value = (
      result.seismic.find((source) => source.model_compatibility?.[selectedPredictionModelId.value]?.ready)
      || result.seismic.find((source) => source.trace_count > 0)
      || result.seismic[0]
    ).path;
  }
}

function resetSurfaceSegDefaults() {
  surfaceSegScope.value = "smoke";
  surfaceSegMaxInlines.value = 2;
  surfaceSegInlineCount.value = "";
  surfaceSegAmplitudeMode.value = "auto";
  surfaceSegQueryThreshold.value = 0.35;
  surfaceSegMaskThreshold.value = 0.5;
  surfaceSegformerBatchSize.value = 2;
  surfaceMask2formerBatchSize.value = 1;
  predictionDevice.value = "auto";
}

function selectCompatiblePredictionSource() {
  const sources = predictionSources.value;
  if (!sources.length) {
    predictionSeismicPath.value = "";
    return;
  }
  const current = sources.find((source) => source.path === predictionSeismicPath.value);
  if (current?.model_compatibility?.[selectedPredictionModelId.value]?.ready) return;
  predictionSeismicPath.value = (
    sources.find((source) => source.model_compatibility?.[selectedPredictionModelId.value]?.ready)
    || current
    || sources[0]
  ).path;
}

function handlePredictionModelChange() {
  if (isSurfaceSegModel.value) resetSurfaceSegDefaults();
  selectCompatiblePredictionSource();
}

function handlePredictionSourceChange() {
  if (isSurfaceSegModel.value) surfaceSegInlineCount.value = "";
}

function applyDemo() {
  if (!demo.value?.available) return;
  const mapping: Record<PathGroupKey, string[]> = {
    seismic: demo.value.seismic_paths,
    logs: demo.value.log_paths,
    wells: demo.value.well_paths,
    auxiliary: demo.value.auxiliary_paths,
  };
  groups.value.forEach((group) => {
    group.paths = [...mapping[group.key]];
  });
}

async function waitForTask(id: string): Promise<WorkflowResult> {
  const deadline = Date.now() + TASK_POLL_TIMEOUT_MS;
  while (!componentUnmounted && Date.now() < deadline) {
    const task: BackgroundTask = await getTask(id);
    progress.value = task.progress;
    statusMessage.value = task.message;
    if (task.status === "completed" && task.result && "summary" in task.result) return task.result as WorkflowResult;
    if (task.status === "failed") throw new Error(task.error?.message || "后端任务失败");
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
  if (componentUnmounted) throw new Error("页面已关闭，已停止轮询任务状态");
  throw new Error("任务运行超过30分钟，请检查后端日志或稍后重新打开任务");
}

async function runDataPreparation() {
  errorMessage.value = "";
  if (!registeredCount.value) {
    errorMessage.value = "至少需要登记一个文件或目录。";
    return;
  }
  preparationRunning.value = true;
  progress.value = 0;
  statusMessage.value = "正在提交数据准备任务";
  try {
    const payload = commonPayload();
    savePathConfig(payload);
    const created = await createDataPreparation(payload);
    taskId.value = created.task_id;
    preparationResult.value = await waitForTask(created.task_id);
    initializeVisualization(preparationResult.value);
    window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, created.task_id);
    sampleResult.value = null;
    const firstAttentionStage = preparationResult.value.preparation.issues.find(
      (issue) => issue.blocking || issue.severity === "警告",
    )?.stage;
    preparationScreen.value = "pipeline";
    issueFilter.value = firstAttentionStage || "全部";
    await nextTick();
    window.scrollTo({ top: 0, behavior: "auto" });
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "数据准备失败";
  } finally {
    preparationRunning.value = false;
  }
}

async function confirmIssueRecommendation(
  issue: PreparationIssue,
  decision: "确认采用" | "暂不采用",
) {
  if (!taskId.value || confirmingIssueId.value) return;
  confirmingIssueId.value = issue.id;
  errorMessage.value = "";
  try {
    const updated = await confirmPreparationIssue(
      taskId.value,
      issue.id,
      decision,
      issue.recommended_action,
    );
    Object.assign(issue, updated);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "处理建议确认失败";
  } finally {
    confirmingIssueId.value = "";
  }
}

function focusIssueReview() {
  const stage = preparation.value?.issues.find(
    (issue) => issue.blocking || issue.confirmation_status === "待人工确认",
  )?.stage;
  if (stage) issueFilter.value = stage;
  void nextTick().then(() => document.getElementById("issue-review")?.scrollIntoView({ behavior: "smooth", block: "start" }));
}

async function runSampleBuilding() {
  errorMessage.value = "";
  if (!values("seismic").length || !values("logs").length) {
    errorMessage.value = "井震样本构建至少需要地震数据和LAS测井数据。";
    return;
  }
  sampleRunning.value = true;
  progress.value = 0;
  statusMessage.value = "正在提交井震空间对齐任务";
  try {
    const payload = commonPayload();
    savePathConfig(payload);
    const created = await createSampleBuilding(payload);
    taskId.value = created.task_id;
    sampleResult.value = await waitForTask(created.task_id);
    initializeVisualization(sampleResult.value);
    window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, created.task_id);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "样本构建失败";
  } finally {
    sampleRunning.value = false;
  }
}

async function waitForPrediction(id: string): Promise<PredictionTaskResult> {
  const deadline = Date.now() + TASK_POLL_TIMEOUT_MS;
  while (!componentUnmounted && Date.now() < deadline) {
    const task = await getTask(id);
    progress.value = task.progress;
    statusMessage.value = task.message;
    if (task.status === "completed" && task.result && "prediction" in task.result) {
      return task.result as PredictionTaskResult;
    }
    if (task.status === "failed") throw new Error(task.error?.message || "模型推理失败");
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
  throw new Error("模型推理超时，请检查后端运行状态");
}

async function runPrediction() {
  errorMessage.value = "";
  if (!predictionSeismicPath.value) {
    errorMessage.value = "请先在数据准备中识别一个三维 SEG-Y 文件。";
    return;
  }
  const sourceTaskId = taskId.value;
  visualizationSourceTaskId.value = "";
  visualizationBaseTaskId.value = sourceTaskId;
  selectedSeismicAssetIndex.value = 0;
  predictionSourceTaskId.value = sourceTaskId;
  predictionTaskId.value = "";
  predictionResult.value = null;
  window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
  predictionRunning.value = true;
  progress.value = 0;
  statusMessage.value = `正在提交${activePredictionTaskSpec.value?.name || "模型"}推理任务`;
  try {
    const common = {
      task_id: activePredictionTask.value,
      model_id: selectedPredictionModelId.value,
      seismic_path: predictionSeismicPath.value,
      source_task_id: sourceTaskId || undefined,
      device: predictionDevice.value,
    };
    const created = isSurfaceSegModel.value
      ? await createPrediction({
          ...common,
          options: {
            amplitude_mode: surfaceSegAmplitudeMode.value,
            query_threshold: surfaceSegQueryThreshold.value,
            mask_threshold: surfaceSegMaskThreshold.value,
            segformer_batch_size: surfaceSegformerBatchSize.value,
            mask2former_batch_size: surfaceMask2formerBatchSize.value,
            num_visualizations: 5,
            write_mask_sgy: surfaceSegScope.value === "full",
            ...(surfaceSegScope.value === "smoke" ? { max_inlines: surfaceSegMaxInlines.value } : {}),
            ...(surfaceSegInlineCount.value === "" ? {} : { inline_count: Number(surfaceSegInlineCount.value) }),
          },
        })
      : await createPrediction({
          ...common,
          crop_size: [predictionCropSize.value, predictionCropSize.value, predictionCropSize.value],
          patch_size: [32, 32, 32],
          overlap: [8, 8, 8],
          threshold: predictionThreshold.value,
        });
    predictionTaskId.value = created.task_id;
    const completed = await waitForPrediction(created.task_id);
    predictionResult.value = completed.prediction;
    predictionSourceTaskId.value = completed.source_task_id || sourceTaskId;
    window.sessionStorage.setItem(LAST_PREDICTION_TASK_STORAGE_KEY, created.task_id);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "模型推理失败";
  } finally {
    predictionRunning.value = false;
  }
}

function navigationState(view: ViewKey): string {
  if (view === "overview") return "";
  if (view === "preparation") return preparation.value ? "已运行" : "未开始";
  if (view === "visualization") return seismicInventory.value.length || seismicPreviewCount.value || wellLogPreviews.value.length ? "当前任务" : "待数据";
  if (view === "samples") {
    if (sampleResult.value?.matching) return "已完成";
    return preparation.value?.gates.can_build_samples ? "可执行" : "待对齐";
  }
  if (view === "models") return sampleResult.value?.matching ? "可接入" : "接口";
  if (view === "prediction") return predictionResult.value ? "有结果" : "选择任务";
  if (view === "evaluation") return sampleResult.value?.matching ? "有结果" : "待结果";
  return "可配置";
}

function stageClass(stage: PreparationStage): string {
  if (stage.status === "阻断") return "blocked";
  if (stage.status === "需确认") return "warning";
  if (stage.status === "就绪") return "ready";
  return "waiting";
}

function stageShortName(stageId: string): string {
  return ({
    asset_registration: "资产登记",
    log_preprocessing: "测井标准化",
    well_entity_alignment: "井数据合并",
    seismic_geometry: "地震几何",
    spatial_alignment: "井震对齐",
    vertical_alignment: "时间域标定",
    sample_building: "样本构建",
  } as Record<string, string>)[stageId] || stageId;
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 ** 3) return `${(size / 1024 ** 2).toFixed(1)} MB`;
  return `${(size / 1024 ** 3).toFixed(2)} GB`;
}

function issueSource(issue: PreparationIssue): string {
  if (!issue.source) return "流程级问题";
  const normalized = issue.source.replaceAll("\\", "/");
  return normalized.split("/").at(-1) || issue.source;
}

function selectVisualizationMode(mode: "seismic" | "logs") {
  visualizationMode.value = mode;
}

function selectSeismicAsset(index: number) {
  selectedSeismicAssetIndex.value = index;
}

function showPredictionVisualization() {
  if (!predictionTaskId.value) return;
  visualizationBaseTaskId.value = predictionSourceTaskId.value || taskId.value;
  visualizationSourceTaskId.value = predictionTaskId.value;
  selectedSeismicAssetIndex.value = 0;
  visualizationMode.value = "seismic";
  selectView("visualization");
}

function showBaseVisualization() {
  visualizationSourceTaskId.value = "";
  selectedSeismicAssetIndex.value = 0;
}

function syncVisualizationFullscreen() {
  visualizationFullscreen.value = document.fullscreenElement === visualizationStageElement.value;
}

async function toggleVisualizationFullscreen() {
  const stage = visualizationStageElement.value;
  if (!stage) return;
  try {
    if (document.fullscreenElement === stage) await document.exitFullscreen();
    else {
      if (document.fullscreenElement) await document.exitFullscreen();
      await stage.requestFullscreen();
    }
  } catch {
    window.open(visualizationStandaloneUrl.value, "_blank", "noopener,noreferrer");
  }
}

function selectPredictionTask(task: PredictionTaskKey) {
  activePredictionTask.value = task;
  visualizationSourceTaskId.value = "";
  visualizationBaseTaskId.value = taskId.value;
  selectedSeismicAssetIndex.value = 0;
  predictionSourceTaskId.value = taskId.value;
  predictionTaskId.value = "";
  predictionResult.value = null;
  window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
  selectedPredictionModelId.value = runnablePredictionModels.value[0]?.id || "";
  handlePredictionModelChange();
}

function selectWellLog(id: string) {
  selectedWellLogId.value = id;
  const log = wellLogPreviews.value.find((item) => item.id === id);
  visibleCurveIds.value = log?.curves.map((curve) => curve.id) || [];
}

function toggleCurve(curveId: string) {
  visibleCurveIds.value = visibleCurveIds.value.includes(curveId)
    ? visibleCurveIds.value.filter((id) => id !== curveId)
    : [...visibleCurveIds.value, curveId];
}

function curvePath(curve: WellLogPreviewCurve): string {
  const values = curve.values.map((value) => {
    if (value === null || !Number.isFinite(value)) return null;
    if (curve.scale === "log") return value > 0 ? Math.log10(value) : null;
    return value;
  });
  const finite = values.filter((value): value is number => value !== null);
  const depths = selectedWellLog.value?.depth || [];
  const finiteDepths = depths.filter((value): value is number => value !== null && Number.isFinite(value));
  if (!finite.length || !finiteDepths.length) return "";
  const sorted = [...finite].sort((a, b) => a - b);
  const low = sorted[Math.floor((sorted.length - 1) * 0.03)];
  const high = sorted[Math.floor((sorted.length - 1) * 0.97)];
  const span = Math.max(high - low, 1e-9);
  const depthLow = Math.min(...finiteDepths);
  const depthSpan = Math.max(Math.max(...finiteDepths) - depthLow, 1e-9);
  let startSegment = true;
  return values.map((value, index) => {
      const depth = depths[index];
      if (value === null || depth === null || !Number.isFinite(depth)) {
        startSegment = true;
        return "";
      }
      const x = Math.max(3, Math.min(97, 3 + ((value - low) / span) * 94));
      const y = 18 + ((depth - depthLow) / depthSpan) * 604;
      const command = startSegment ? "M" : "L";
      startSegment = false;
      return `${command}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function curveRange(curve: WellLogPreviewCurve): string {
  const values = curve.values.filter((value): value is number => value !== null && Number.isFinite(value));
  if (!values.length) return "无有效值";
  const low = Math.min(...values);
  const high = Math.max(...values);
  return `${low.toPrecision(4)} — ${high.toPrecision(4)}`;
}

async function createTransformDraft(issue: PreparationIssue) {
  if (!taskId.value || generatingDraftIssueId.value) return;
  generatingDraftIssueId.value = issue.id;
  errorMessage.value = "";
  try {
    const draft = await generateTransformationDraft(taskId.value, issue.id);
    transformationDrafts.value = { ...transformationDrafts.value, [issue.id]: draft };
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "转换适配器生成失败";
  } finally {
    generatingDraftIssueId.value = "";
  }
}

async function enableTransformDraft(issue: PreparationIssue, draft: TransformationDraft) {
  if (activatingDraftId.value || !draft.valid) return;
  activatingDraftId.value = draft.id;
  try {
    const activated = await activateTransformationDraft(draft.id);
    transformationDrafts.value = { ...transformationDrafts.value, [issue.id]: activated };
    issue.confirmation_status = "已启用转换插件";
    issue.confirmed_action = activated.title;
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "转换适配器启用失败";
  } finally {
    activatingDraftId.value = "";
  }
}

function syncViewFromHash() {
  const hash = window.location.hash.replace("#", "") as ViewKey;
  if (hash === "overview" || navigation.some((item) => item.id === hash)) activeView.value = hash;
}

function predictionParentTaskId(task: BackgroundTask, result: PredictionTaskResult): string {
  const taskWithParent = task as BackgroundTask & {
    source_task_id?: string;
    parent_task_id?: string;
  };
  return result.source_task_id || taskWithParent.source_task_id || taskWithParent.parent_task_id || "";
}

async function restorePredictionSource(sourceTaskId: string) {
  if (!sourceTaskId) return;
  predictionSourceTaskId.value = sourceTaskId;
  visualizationBaseTaskId.value = sourceTaskId;
  if (taskId.value === sourceTaskId && workflowResult.value) return;
  try {
    const sourceTask = await getTask(sourceTaskId);
    if (
      sourceTask.status !== "completed"
      || !sourceTask.result
      || "prediction" in sourceTask.result
    ) return;
    taskId.value = sourceTask.task_id;
    preparationResult.value = sourceTask.result;
    sampleResult.value = sourceTask.task_type === "sample_building" ? sourceTask.result : null;
    initializeVisualization(sourceTask.result);
  } catch {
    // Keep the source id for the visualization endpoint even if the archived task summary is unavailable.
  }
}

async function restorePredictionTask(task: BackgroundTask) {
  if (task.status !== "completed" || !task.result || !("prediction" in task.result)) return;
  const result = task.result as PredictionTaskResult;
  predictionTaskId.value = task.task_id;
  predictionResult.value = result.prediction;
  activePredictionTask.value = result.prediction.task_id || "fault";
  selectedPredictionModelId.value = result.prediction.model_id;
  await restorePredictionSource(predictionParentTaskId(task, result));
}

onMounted(async () => {
  syncViewFromHash();
  restorePathConfig();
  window.addEventListener("hashchange", syncViewFromHash);
  document.addEventListener("fullscreenchange", syncVisualizationFullscreen);
  try {
    const service = await health();
    backendStatus.value = "online";
    backendVersion.value = service.version;
    const [demoResult, capabilitiesResult] = await Promise.allSettled([demoPaths(), getCapabilities()]);
    if (demoResult.status === "fulfilled") demo.value = demoResult.value;
    if (capabilitiesResult.status === "fulfilled") {
      capabilities.value = capabilitiesResult.value;
      if (capabilitiesResult.value.llm.available) useLlmFallback.value = true;
    }
    const rememberedTaskId = window.sessionStorage.getItem(LAST_TASK_STORAGE_KEY);
    if (rememberedTaskId) {
      try {
        const rememberedTask = await getTask(rememberedTaskId);
        if (rememberedTask.status === "completed" && rememberedTask.result) {
          if ("prediction" in rememberedTask.result) {
            await restorePredictionTask(rememberedTask);
          } else {
            taskId.value = rememberedTask.task_id;
            preparationResult.value = rememberedTask.result;
            if (rememberedTask.task_type === "sample_building") sampleResult.value = rememberedTask.result;
            initializeVisualization(rememberedTask.result);
            if (activeView.value === "preparation") preparationScreen.value = "pipeline";
          }
        }
      } catch {
        window.sessionStorage.removeItem(LAST_TASK_STORAGE_KEY);
      }
    }
    const rememberedPredictionTaskId = window.sessionStorage.getItem(LAST_PREDICTION_TASK_STORAGE_KEY);
    if (rememberedPredictionTaskId) {
      try {
        const rememberedPredictionTask = await getTask(rememberedPredictionTaskId);
        await restorePredictionTask(rememberedPredictionTask);
      } catch {
        window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
      }
    }
  } catch {
    backendStatus.value = "offline";
  }
});

onBeforeUnmount(() => {
  componentUnmounted = true;
  window.removeEventListener("hashchange", syncViewFromHash);
  document.removeEventListener("fullscreenchange", syncVisualizationFullscreen);
});
</script>

<template>
  <div :class="['app-shell', { 'landing-shell': activeView === 'overview' }]">
    <aside v-if="activeView !== 'overview'" class="sidebar">
      <button type="button" class="brand brand-button" title="返回首屏" @click="selectView('overview')">
        <svg class="brand-mark" viewBox="0 0 160 160" aria-hidden="true">
          <path class="brand-layer brand-layer-primary" d="M18 52C42 34 62 37 80 51c18 14 37 18 62-2" />
          <path class="brand-layer brand-layer-secondary" d="M18 82c24-17 45-14 63 0 18 14 37 17 61-3" />
          <path class="brand-layer brand-layer-accent" d="M18 110c25-15 47-11 65 2 18 12 37 14 59-3" />
          <path class="brand-well" d="M78 24v81c0 14 8 24 27 31" />
          <circle cx="105" cy="136" r="8" />
        </svg>
        <div><strong>地层慧眼</strong><span>地震—测井多模态大模型</span></div>
      </button>

      <nav class="navigation" aria-label="主导航">
        <template v-for="section in ['workflow', 'system']" :key="section">
          <p class="nav-section">
            {{ section === "workflow" ? "完整工作流" : "系统" }}
          </p>
          <div
            v-for="item in navigation.filter((nav) => nav.section === section)"
            :key="item.id"
            :class="['nav-entry', { 'has-submenu': item.id === 'prediction' }]"
          >
            <button
              type="button"
              :class="['nav-item', { active: activeView === item.id }]"
              :data-view="item.id"
              @click="selectView(item.id)"
            >
              <span class="nav-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24">
                  <path v-for="path in navigationIconPaths[item.id]" :key="path" :d="path" />
                </svg>
              </span>
              <span class="nav-label">{{ item.label }}</span>
              <span v-if="navigationState(item.id)" class="nav-state">{{ navigationState(item.id) }}</span>
              <span class="nav-chevron" aria-hidden="true">›</span>
            </button>
            <div v-if="item.id === 'prediction'" class="prediction-hover-menu">
              <span>选择预测任务</span>
              <button
                v-for="task in predictionTaskDefinitions"
                :key="task.id"
                type="button"
                :class="{ active: activeView === 'prediction' && activePredictionTask === task.id }"
                @click="selectPredictionTask(task.id); selectView('prediction')"
              >
                <strong>{{ task.short_name }}</strong><small>{{ task.description }}</small>
              </button>
            </div>
          </div>
        </template>
      </nav>

      <div class="service-state">
        <span :class="['status-light', backendStatus]"></span>
        <div>
          <strong>{{ backendStatus === "online" ? "后端已连接" : backendStatus === "offline" ? "后端未连接" : "正在连接" }}</strong>
          <span>{{ backendVersion ? `平台 ${backendVersion}` : "127.0.0.1:8000" }}</span>
        </div>
      </div>
    </aside>

    <main
      :class="[
        'main-content',
        {
          'overview-content': activeView === 'overview',
          'visualization-content': activeView === 'visualization',
        },
      ]"
    >
      <div v-if="activeView !== 'overview'" class="topbar">
        <div class="topbar-context"><span>地层慧眼</span><b>/</b><strong>{{ currentMeta.eyebrow }}</strong></div>
        <div class="topbar-account">
          <span :class="['topbar-status', backendStatus]"></span>
          <span>{{ backendStatus === "online" ? "本地工作区" : "服务未连接" }}</span>
          <b>⌄</b>
          <i aria-hidden="true">慧</i>
        </div>
      </div>
      <header v-if="activeView !== 'overview'" :class="['page-header', { 'immersive-page-header': activeView === 'visualization' }]">
        <div>
          <p class="eyebrow">{{ currentMeta.eyebrow }}</p>
          <h1>{{ currentMeta.title }}</h1>
          <p>{{ currentMeta.description }}</p>
        </div>
        <div class="header-actions">
          <button
            v-if="activeView === 'preparation' && demo?.available"
            type="button"
            class="secondary-button"
            :disabled="preparationRunning"
            @click="applyDemo"
          >
            填入参考数据
          </button>
        </div>
      </header>

      <template v-if="activeView === 'overview'">
        <section class="overview-hero" :style="{ backgroundImage: `url(${heroLaboratoryImage})` }">
          <header class="landing-header">
            <div class="landing-brand">
              <svg class="brand-mark" viewBox="0 0 160 160" aria-hidden="true">
                <path class="brand-layer brand-layer-primary" d="M18 52C42 34 62 37 80 51c18 14 37 18 62-2" />
                <path class="brand-layer brand-layer-secondary" d="M18 82c24-17 45-14 63 0 18 14 37 17 61-3" />
                <path class="brand-layer brand-layer-accent" d="M18 110c25-15 47-11 65 2 18 12 37 14 59-3" />
                <path class="brand-well" d="M78 24v81c0 14 8 24 27 31" />
                <circle cx="105" cy="136" r="8" />
              </svg>
              <div><strong>地层慧眼</strong><small>STRATA VISION</small></div>
            </div>
            <div class="landing-service"><span :class="['status-light', backendStatus]"></span>{{ backendStatus === 'online' ? '本地服务已连接' : backendStatus === 'offline' ? '服务未连接' : '正在连接服务' }}</div>
          </header>
          <div class="overview-copy">
            <p class="product-kicker"><span></span> 地震—测井多模态统一表征大模型</p>
            <h1>让油气甜点<br />有迹可循</h1>
            <p class="overview-summary">融合地震空间响应与测井精细表征，构建可解释的油气甜点识别能力。</p>
            <div class="product-hero-actions">
              <button type="button" class="hero-primary" @click="selectView(nextAction.view)">
                <span>开始使用</span><i aria-hidden="true">→</i>
              </button>
              <button type="button" class="hero-secondary" @click="selectView('models')">
                <span>了解方法</span><i aria-hidden="true">→</i>
              </button>
            </div>
            <div class="overview-trust-line">
              <span><b></b> 原始数据只读</span><span>来源全程记录</span><span>模型组件可替换</span>
            </div>
          </div>
          <div class="overview-dock">
            <button type="button" @click="selectView('preparation')"><span>DATA</span><strong>多源数据标准化</strong><small>版本兼容与知识映射</small><i>→</i></button>
            <button type="button" @click="selectView('preparation')"><span>VIEW</span><strong>数据视图</strong><small>井震匹配纳入数据准备</small><i>→</i></button>
            <button type="button" @click="selectView('models')"><span>FUSION</span><strong>统一表征</strong><small>单模态与多模态基线</small><i>→</i></button>
            <button type="button" @click="selectView('prediction')"><span>PREDICT</span><strong>下游解释</strong><small>断层 · 储层 · 甜点</small><i>→</i></button>
          </div>
        </section>
      </template>

      <template v-else-if="activeView === 'preparation'">
        <section class="preparation-navigation" aria-label="数据准备步骤">
          <button type="button" :class="{ active: preparationScreen === 'input' }" @click="showPreparationInput">
            <span>1</span><div><strong>登记数据源</strong><small>填写文件或目录的绝对路径</small></div>
            <em v-if="preparationResult">已完成</em>
          </button>
          <b>→</b>
          <button type="button" :disabled="!preparation" :class="{ active: preparationScreen === 'pipeline' }" @click="showPreparationPipeline()">
            <span>2</span><div><strong>预处理与对齐审核</strong><small>查看流程状态并确认处理建议</small></div>
            <em v-if="pendingConfirmationCount">{{ pendingConfirmationCount }} 项待确认</em>
          </button>
        </section>

        <section v-if="preparationScreen === 'input'" class="section-panel input-stage-panel">
          <div class="section-heading">
            <div><h2>输入数据源</h2><p>可以填写目录或单个文件；同类数据支持多个路径，中文路径不会被改名。</p></div>
            <span class="count-badge">已登记 {{ registeredCount }} 个路径</span>
          </div>

          <div class="path-groups">
            <section v-for="group in groups" :key="group.key" class="path-group">
              <div class="path-group-heading">
                <div><h3>{{ group.title }} <span v-if="group.optional">可选</span></h3><p>{{ group.hint }}</p></div>
                <button type="button" class="text-button" :disabled="preparationRunning" @click="addPath(group)">＋ 添加路径</button>
              </div>
              <div v-if="group.paths.length" class="path-list">
                <div v-for="(_, index) in group.paths" :key="index" class="path-row">
                  <span>{{ index + 1 }}</span>
                  <input v-model="group.paths[index]" type="text" :disabled="preparationRunning" placeholder="输入绝对路径，例如 D:\比赛数据\地震" />
                  <button type="button" class="remove-button" :disabled="preparationRunning" @click="removePath(group, index)">移除</button>
                </div>
              </div>
              <p v-else class="empty-path">本次任务跳过该类数据。</p>
            </section>
          </div>

          <div class="run-bar">
            <div class="switches">
              <label><input v-model="recursive" type="checkbox" :disabled="preparationRunning" />递归读取子目录</label>
              <label><input v-model="lightweight" type="checkbox" :disabled="preparationRunning" />SEG-Y优先读取文件头</label>
              <label class="llm-task-switch" :title="capabilities?.llm.data_policy">
                <input v-model="useLlmFallback" type="checkbox" :disabled="preparationRunning || !capabilities?.llm.available" />
                智能研判问题（推荐）
                <span>{{ capabilities?.llm.available ? "已配置" : "未配置" }}</span>
              </label>
            </div>
            <div class="run-actions">
              <small>执行完成后将自动进入流水线审核</small>
              <button type="button" class="primary-button" :disabled="preparationRunning || backendStatus !== 'online'" @click="runDataPreparation">
                {{ preparationRunning ? "正在准备数据…" : "执行数据准备并进入审核" }}
              </button>
            </div>
          </div>

          <div v-if="preparationRunning" class="task-progress">
            <div><span>{{ statusMessage }}</span><strong>{{ progress }}%</strong></div>
            <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
          </div>
          <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
        </section>

        <template v-else-if="preparation">
          <section class="section-panel pipeline-panel">
            <div class="section-heading pipeline-heading">
              <div>
                <span class="section-kicker">当前任务 · 预处理审核</span>
                <h2>预处理与对齐流水线</h2>
                <p>点击步骤查看证据和指标；黄色需要复核，红色会阻断后续对齐。</p>
              </div>
              <div class="pipeline-heading-actions">
                <div class="issue-summary">
                  <span class="danger">{{ preparation.summary.blocking }} 阻断</span>
                  <span class="warn">{{ preparation.summary.warnings }} 警告</span>
                  <span>{{ pendingConfirmationCount }} 待人工确认</span>
                </div>
                <button type="button" class="secondary-button compact" @click="showPreparationInput">返回修改输入</button>
              </div>
            </div>

            <ol class="pipeline-stepper" aria-label="预处理与对齐执行流程">
              <li v-for="(stage, index) in preparation.stages" :key="stage.id" :class="[stageClass(stage), { selected: issueFilter === stage.id }]">
              <button
                type="button"
                @click="issueFilter = stage.id"
              >
                <span>{{ stage.status === "就绪" ? "✓" : index + 1 }}</span>
                <strong>{{ stageShortName(stage.id) }}</strong>
                <em>{{ stage.issue_count ? `${stage.issue_count} 项需关注` : stage.status }}</em>
              </button>
              </li>
            </ol>

            <div v-if="selectedStage" class="pipeline-inspector">
              <div class="pipeline-inspector-copy"><span>当前步骤 · {{ selectedStage.status }}</span><strong>{{ selectedStage.name }}</strong><p>{{ selectedStage.description }}</p><button type="button" class="text-button" @click="focusIssueReview">查看本步骤问题 ↓</button></div>
              <dl><template v-for="(value, key) in selectedStage.metrics" :key="key"><div><dt>{{ key }}</dt><dd>{{ value }}</dd></div></template></dl>
            </div>
            <button v-else type="button" class="all-issues-button" @click="issueFilter = '全部'">查看全部阶段的问题与建议</button>
          </section>

          <section id="issue-review" class="section-panel issue-review-panel">
            <div class="section-heading">
              <div><h2>问题研判与转换实现</h2><p>确定性问题由知识库直接处理；未决问题可生成受控转换适配器，自动测试通过后由你启用。</p></div>
              <button type="button" class="secondary-button compact" @click="issueFilter = '全部'">显示全部问题</button>
            </div>
            <div v-if="filteredIssues.length" class="issue-list">
              <article v-for="issue in filteredIssues" :key="issue.id" :class="['issue-row', issue.severity]">
                <div class="issue-description">
                  <div class="issue-title-line"><span class="severity">{{ issue.severity }}</span><strong>{{ issue.title }}</strong></div>
                  <p>{{ issue.message }}</p>
                  <small v-if="(issue.affected_count || 0) > 1">影响 {{ issue.affected_count }} 个文件 · 首个来源：{{ issueSource(issue) }}</small>
                  <small v-else>来源：{{ issueSource(issue) }}</small>
                </div>
                <div class="recommendation-card">
                  <div class="recommendation-heading">
                    <span :class="['recommendation-source', { llm: issue.recommendation_source === 'LLM' }]">
                      {{ issue.recommendation_source === "LLM" ? "智能建议" : "规则建议" }}
                    </span>
                    <em v-if="issue.recommendation_confidence !== null">置信度 {{ Math.round(issue.recommendation_confidence * 100) }}%</em>
                    <em v-else>平台知识库</em>
                  </div>
                  <strong>{{ issue.recommended_action || "保留来源记录，无需处理" }}</strong>
                  <p>{{ issue.recommendation_reason }}</p>
                  <div v-if="issue.confirmation_status === '待人工确认'" class="confirmation-actions">
                    <button type="button" class="primary-button compact" :disabled="confirmingIssueId === issue.id" @click="confirmIssueRecommendation(issue, '确认采用')">
                      {{ confirmingIssueId === issue.id ? "正在确认…" : "确认采用此方案" }}
                    </button>
                    <button type="button" class="adapter-button" :disabled="generatingDraftIssueId === issue.id" @click="createTransformDraft(issue)">
                      {{ generatingDraftIssueId === issue.id ? "正在生成与测试…" : "生成转换适配器" }}
                    </button>
                    <button type="button" class="quiet-button" :disabled="confirmingIssueId === issue.id" @click="confirmIssueRecommendation(issue, '暂不采用')">暂不采用</button>
                  </div>
                  <div v-else :class="['confirmation-result', { accepted: issue.confirmation_status === '已确认采用' || issue.confirmation_status === '已启用转换插件' }]">
                    {{ issue.confirmation_status }}<span v-if="issue.confirmed_action"> · {{ issue.confirmed_action }}</span>
                  </div>
                  <div v-if="transformationDrafts[issue.id]" class="adapter-draft">
                    <div class="adapter-draft-head">
                      <div><span>{{ transformationDrafts[issue.id].provider === '平台规则编译器' ? '规则编译' : 'LLM 生成' }}</span><strong>{{ transformationDrafts[issue.id].title }}</strong></div>
                      <em :class="{ passed: transformationDrafts[issue.id].valid }">{{ transformationDrafts[issue.id].status }}</em>
                    </div>
                    <p>{{ transformationDrafts[issue.id].explanation }}</p>
                    <div class="adapter-tests">
                      <span v-for="test in transformationDrafts[issue.id].tests" :key="test.name" :class="{ passed: test.passed }">{{ test.passed ? '✓' : '!' }} {{ test.name }} · {{ test.details }}</span>
                    </div>
                    <details><summary>查看受控实现代码</summary><pre>{{ transformationDrafts[issue.id].generated_code }}</pre></details>
                    <div class="adapter-draft-actions">
                      <span>{{ transformationDrafts[issue.id].model }} · 原始文件不会被改写</span>
                      <button type="button" class="primary-button compact" :disabled="!transformationDrafts[issue.id].valid || activatingDraftId === transformationDrafts[issue.id].id || transformationDrafts[issue.id].status === '已启用'" @click="enableTransformDraft(issue, transformationDrafts[issue.id])">
                        {{ transformationDrafts[issue.id].status === '已启用' ? '已启用' : activatingDraftId === transformationDrafts[issue.id].id ? '正在启用…' : '确认启用插件' }}
                      </button>
                    </div>
                  </div>
                </div>
              </article>
            </div>
            <div v-else class="empty-inline">当前阶段没有需要处理的问题，可以继续下一步。</div>
            <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
          </section>

          <section v-if="workflowResult?.data_snapshot" class="section-panel data-snapshot-panel">
            <div class="section-heading">
              <div>
                <span class="section-kicker">通用数据契约 {{ workflowResult.data_snapshot.contract_version }}</span>
                <h2>模型无关数据快照</h2>
                <p>数据准备只负责形成统一、可追溯的数据基座；FaultSeg 等下游模型再通过各自适配器派生输入，不会反向改变地震读取和可视化。</p>
              </div>
              <span class="algorithm-badge">{{ workflowResult.data_snapshot.snapshot_id?.slice(0, 8) || '当前任务' }}</span>
            </div>
            <div class="snapshot-flow" aria-label="通用数据快照处理流程">
              <article><span>01</span><strong>源资产登记</strong><small>{{ workflowResult.data_snapshot.canonical_data.seismic_geometry.registered }} 个地震文件</small></article>
              <b>→</b>
              <article><span>02</span><strong>通用标准化</strong><small>{{ workflowResult.data_snapshot.canonical_data.seismic_geometry.readable }} 个已读几何</small></article>
              <b>→</b>
              <article><span>03</span><strong>下游数据视图</strong><small>可视化、井震样本及质量证据</small></article>
              <b>→</b>
              <article><span>04</span><strong>模型专属适配</strong><small>按任务与模型独立派生输入</small></article>
            </div>
            <p class="snapshot-policy">{{ workflowResult.data_snapshot.downstream_policy }}</p>
          </section>

          <section class="pipeline-next-step">
            <div>
              <span>下一步</span>
              <strong v-if="mainBlockingIssue">先处理 {{ preparation.summary.blocking }} 个阻断问题</strong>
              <strong v-else>数据已满足后续工作流的基础放行条件</strong>
              <p v-if="mainBlockingIssue">阻断问题会影响井位、地震几何或正式空间匹配。</p>
              <p v-else>可以先检查当前任务数据，再在数据准备层按需生成井震空间对齐视图。</p>
            </div>
            <div class="next-step-actions">
              <button v-if="mainBlockingIssue || pendingConfirmationCount" type="button" class="secondary-button" @click="focusIssueReview">审核待处理建议</button>
              <button type="button" class="secondary-button" :disabled="!seismicInventory.length && !wellLogPreviews.length" @click="selectView('visualization')">
                {{ seismicInventory.length || wellLogPreviews.length ? "查看当前任务数据" : "当前任务无可视化资产" }}
              </button>
              <button type="button" class="primary-button" :disabled="sampleRunning || !preparation.gates.can_build_samples" @click="runSampleBuilding">
                {{ sampleRunning ? "正在构建井震数据视图…" : sampleResult?.matching ? "重新构建井震数据视图" : "构建井震数据视图（可选）" }}
              </button>
            </div>
            <div v-if="sampleRunning" class="task-progress">
              <div><span>{{ statusMessage }}</span><strong>{{ progress }}%</strong></div>
              <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
            </div>
            <div v-if="sampleResult?.matching" class="data-view-result">
              <div class="data-view-result-head">
                <span>井震多模态数据视图</span>
                <strong>{{ sampleResult.matching.sample_count.toLocaleString() }} 条候选样本</strong>
              </div>
              <div class="data-view-result-metrics">
                <span><b>{{ (sampleResult.matching.valid_window_count ?? 0).toLocaleString() }}</b>有效时间窗</span>
                <span><b>{{ (sampleResult.matching.training_eligible_count ?? 0).toLocaleString() }}</b>可训练样本</span>
                <span><b>{{ sampleResult.matching.coordinate_reference_verified ? "已核验" : "待核验" }}</b>井震坐标参考</span>
              </div>
              <code>{{ sampleResult.matching.output_directory }}</code>
            </div>
          </section>

          <details class="section-panel asset-details">
            <summary>查看数据资产清单（{{ workflowResult?.assets.length || 0 }}）</summary>
            <p>大型SEG-Y保留在原位置，不复制到项目目录。</p>
            <div class="table-wrap">
              <table>
                <thead><tr><th>文件</th><th>角色</th><th>阶段</th><th>大小</th><th>原始路径</th></tr></thead>
                <tbody>
                  <tr v-for="asset in workflowResult?.assets" :key="asset.id">
                    <td>{{ asset.path.replaceAll("\\", "/").split("/").at(-1) }}</td>
                    <td>{{ asset.role }}</td><td>{{ asset.stage }}</td><td>{{ formatBytes(asset.size) }}</td><td class="path-cell">{{ asset.path }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </details>
        </template>
      </template>

      <template v-else-if="activeView === 'visualization'">
        <section v-if="!seismicInventory.length && !seismicPreviewCount && !wellLogPreviews.length" class="empty-workflow">
          <strong>当前任务没有可用于预览的数据</strong>
          <p>请先在“数据准备”中登记 SEG-Y 或 LAS；系统不会显示历史示例数据。</p>
          <button type="button" class="primary-button" @click="selectView('preparation')">进入数据准备</button>
        </section>
        <template v-else>
          <section class="visualization-switcher">
            <div class="visualization-task-summary">
              <span>{{ visualizationSourceTaskId ? `${predictionResult?.task_name || activePredictionTaskSpec?.name || '模型'}预测结果` : '当前任务可视化' }}</span>
              <strong>选择数据解释工作台</strong>
              <small>{{ seismicInventory.length }} 个地震资产 / {{ previewVolumes.length }} 个三维体 / {{ seismicLinePreviews.length }} 条二维测线 · {{ wellLogPreviews.length }} 口井 / LAS 预览</small>
            </div>
            <nav class="visualization-mode-cards" aria-label="可视化类型">
              <button type="button" data-visualization-mode="seismic" :aria-pressed="visualizationMode === 'seismic'" :disabled="!seismicInventory.length && !seismicPreviewCount" :class="['visualization-mode-card', { active: visualizationMode === 'seismic' }]" @click="selectVisualizationMode('seismic')">
                <span class="mode-card-icon seismic" aria-hidden="true">
                  <svg viewBox="0 0 28 28"><path d="m14 3 10 5.4v11.2L14 25 4 19.6V8.4L14 3Z"/><path d="m4.7 8.8 9.3 5.1 9.3-5.1M14 14v10.2"/><path d="M7.6 13.2c2.4-1.8 4.7 1.7 7.1 0s4.3-1.4 6.5.1"/></svg>
                </span>
                <span class="mode-card-copy"><small>CIGVis 空间解释场景</small><strong>三维地震与二维测线</strong><em>Viser 交互切片、二维剖面与井轨迹叠加</em></span>
                <b>{{ visualizationMode === 'seismic' ? '当前视图' : seismicInventory.length || seismicPreviewCount ? '进入' : '无数据' }} <i>→</i></b>
              </button>
              <button type="button" data-visualization-mode="logs" :aria-pressed="visualizationMode === 'logs'" :disabled="!wellLogPreviews.length" :class="['visualization-mode-card', { active: visualizationMode === 'logs' }]" @click="selectVisualizationMode('logs')">
                <span class="mode-card-icon logs" aria-hidden="true">
                  <svg viewBox="0 0 28 28"><path d="M5 3v22M10 3v22M15 3v22M20 3v22M25 3v22"/><path d="M2 7h24M2 14h24M2 21h24"/><path class="mode-curve-a" d="M7 3c6 4-4 7 5 11s-5 7 3 11"/><path class="mode-curve-b" d="M19 3c-5 4 4 7-2 11s4 7-3 11"/></svg>
                </span>
                <span class="mode-card-copy"><small>井筒精细表征</small><strong>常规九线测井</strong><em>岩性、三孔隙度与电阻率分组综合图</em></span>
                <b>{{ visualizationMode === 'logs' ? '当前视图' : wellLogPreviews.length ? '进入' : '无数据' }} <i>→</i></b>
              </button>
            </nav>
          </section>

          <div v-if="visualizationMode === 'seismic' && seismicPreviewCount" ref="visualizationStageElement" class="seismic-workbench-shell">
            <aside class="seismic-dataset-sidebar" aria-label="地震可视化数据集">
              <header>
                <span>SEISMIC DATASET</span>
                <strong>地震数据集</strong>
                <small>{{ seismicWorkbenchAssets.length }} 个可视化对象</small>
              </header>
              <div class="seismic-dataset-list">
                <button
                  v-for="asset in seismicWorkbenchAssets"
                  :key="`${asset.index}-${asset.name}`"
                  type="button"
                  :class="[{ active: selectedSeismicAsset?.index === asset.index }, { prediction: asset.kind === '预测叠加' }]"
                  :aria-pressed="selectedSeismicAsset?.index === asset.index"
                  @click="selectSeismicAsset(asset.index)"
                >
                  <span>{{ asset.kind === '预测叠加' ? 'AI' : asset.kind === '三维体' ? '3D' : '2D' }}</span>
                  <div><strong>{{ asset.name }}</strong><small :title="asset.detail">{{ asset.detail }}</small></div>
                  <b aria-hidden="true">›</b>
                </button>
              </div>
              <footer>
                <span>{{ seismicInventory.length }} 个已登记 SEG-Y</span>
                <small>点击左侧对象切换当前场景</small>
              </footer>
            </aside>

            <section class="visualization-stage cigvis-stage">
              <header class="visualization-stage-header">
                <div>
                  <p class="eyebrow">可视化阶段 02</p>
                  <h2>CIGVis 地震解释工作台</h2>
                  <span>当前数据：{{ selectedSeismicAsset?.name || '地震预览' }} · {{ selectedSeismicAsset?.kind }}</span>
                </div>
                <div class="visualization-stage-actions">
                  <span>
                    {{ capabilities?.visualization.engine?.web_engine || "CIGVis" }} · 图层契约 {{ capabilities?.visualization.contract_version || "1.0" }}
                    · {{ capabilities?.visualization.extension_points.length || 0 }} 个扩展点
                  </span>
                  <button v-if="visualizationSourceTaskId" type="button" class="secondary-button" @click="showBaseVisualization">查看基础数据</button>
                  <button type="button" class="secondary-button visualization-fullscreen-button" @click="toggleVisualizationFullscreen">
                    {{ visualizationFullscreen ? '退出全屏' : '全屏查看' }}
                  </button>
                  <a :href="visualizationStandaloneUrl" target="_blank" rel="noopener noreferrer" class="secondary-button link-button">独立窗口</a>
                </div>
              </header>
              <div class="visualization-frame">
                <iframe :key="visualizationUrl" title="当前任务 CIGVis 地震解释工作台" :src="visualizationUrl" allow="fullscreen" allowfullscreen></iframe>
              </div>
            </section>
          </div>

          <section v-else-if="visualizationMode === 'seismic'" class="section-panel seismic-preview-empty">
            <strong>地震文件已登记，但当前没有可交给 CIGVis 渲染的数据</strong>
            <p>系统会分别构建三维 Inline × Crossline 稀疏体和二维测线剖面；几何不完整或读取失败的文件仍保留在上方资产清单中，便于回到数据准备层复核。</p>
          </section>

          <section v-else-if="selectedWellLog" class="log-studio">
            <aside class="log-control-panel">
              <div class="log-panel-heading"><span>WELL LOG WORKBENCH</span><strong>测井曲线控制台</strong><p>选择井，并按地质用途控制曲线图层。</p></div>
              <label class="well-log-select-label">当前井 / LAS 文件
                <select :value="selectedWellLog.id" @change="selectWellLog(($event.target as HTMLSelectElement).value)">
                  <option v-for="log in wellLogPreviews" :key="log.id" :value="log.id">{{ log.name }} · {{ conventionalCoverage(log) }}</option>
                </select>
              </label>
              <div class="log-source-summary">
                <span>LAS {{ selectedWellLog.version }}</span><strong>{{ conventionalCoverage(selectedWellLog) }} 常规九线已识别</strong><small :title="selectedWellLog.source">{{ selectedWellLog.source }}</small>
              </div>
              <div class="curve-selector">
                <div class="curve-selector-heading"><strong>常规九线分组</strong><button type="button" @click="visibleCurveIds = selectedWellLog.curves.map((curve) => curve.id)">显示全部</button></div>
                <section v-for="group in groupedConventionalCurves" :key="group.id" class="curve-control-group">
                  <header><span>{{ group.order }}</span><div><strong>{{ group.title }}</strong><small>{{ group.subtitle }}</small></div></header>
                  <label v-for="slot in group.slots" :key="slot.curveId" :class="[{ active: slot.curve && visibleCurveIds.includes(slot.curve.id) }, { missing: !slot.curve }]">
                    <input v-if="slot.curve" type="checkbox" :checked="visibleCurveIds.includes(slot.curve.id)" @change="toggleCurve(slot.curve.id)" />
                    <span v-else class="missing-checkbox">—</span>
                    <i :style="{ background: slot.color }"></i>
                    <span><strong>{{ slot.displayId }} <em>{{ slot.name }}</em></strong><small v-if="slot.curve">{{ slot.curve.unit || '单位未知' }} · {{ slot.curve.validCount.toLocaleString() }} 点</small><small v-else>当前井未提供</small></span>
                  </label>
                </section>
              </div>
            </aside>

            <div class="log-track-workspace">
              <header class="log-workspace-heading"><div><span>CONVENTIONAL NINE-CURVE COMPOSITE</span><h2>{{ selectedWellLog.wellName }} · 常规九线综合测井图</h2></div><p>深度 {{ selectedWellLog.depth[0] ?? '—' }} — {{ selectedWellLog.depth.at(-1) ?? '—' }} {{ selectedWellLog.depthUnit }}<br />每条曲线按自身有效量程归一化；电阻率采用对数尺度</p></header>
              <div class="log-plot-note"><span>浏览方式</span><p>三类曲线共享同一深度轴。当前按深度展开显示，请拖动图板右侧纵向滚动条浏览完整井段；灰色图头表示曲线缺失或已隐藏。</p></div>
              <div v-if="activeConventionalCurveCount" class="conventional-log-board" :style="{ '--log-plot-height': `${logPlotHeight}px` }">
                <aside class="grouped-depth-axis">
                  <header><span>深度</span><strong>DEPTH</strong><small>{{ selectedWellLog.depthUnit }}</small></header>
                  <div class="grouped-depth-scale">
                    <span v-for="tick in depthTicks" :key="`${tick.position}-${tick.value}`" :style="{ top: `${tick.position}%` }"><b>{{ tick.value ?? '—' }}</b></span>
                  </div>
                </aside>
                <article v-for="group in groupedConventionalCurves" :key="group.id" :class="['conventional-group-track', group.id]">
                  <header class="group-track-heading"><span>{{ group.order }}</span><div><strong>{{ group.title }}</strong><small>{{ group.subtitle }}</small></div></header>
                  <div class="group-curve-scales">
                    <div v-for="slot in group.slots" :key="slot.curveId" :class="{ inactive: !slot.curve || !visibleCurveIds.includes(slot.curve.id) }">
                      <span :style="{ color: slot.color }">{{ slot.displayId }}</span>
                      <small>{{ slot.curve?.unit || '—' }}</small>
                      <b>{{ slot.curve ? curveRange(slot.curve) : '未提供' }}</b>
                    </div>
                  </div>
                  <div class="group-curve-plot">
                    <svg viewBox="0 0 100 640" preserveAspectRatio="none" role="img" :aria-label="`${group.title}综合曲线`">
                      <path class="major-track-grid" d="M0 18h100M0 93.5h100M0 169h100M0 244.5h100M0 320h100M0 395.5h100M0 471h100M0 546.5h100M0 622h100" />
                      <path class="minor-track-grid" d="M10 18v604M20 18v604M30 18v604M40 18v604M50 18v604M60 18v604M70 18v604M80 18v604M90 18v604" />
                      <path v-for="slot in group.slots.filter((item) => item.curve && visibleCurveIds.includes(item.curve.id))" :key="slot.curveId" :class="['curve-data-line', `curve-${slot.curveId.toLowerCase()}`]" :d="curvePath(slot.curve!)" :stroke="slot.color" />
                    </svg>
                    <div v-if="!group.slots.some((slot) => slot.curve && visibleCurveIds.includes(slot.curve.id))" class="empty-group-track">本组曲线未提供或已隐藏</div>
                  </div>
                </article>
              </div>
              <div v-else class="empty-inline">请至少启用一条已识别的常规测井曲线。</div>
            </div>
          </section>

        </template>
      </template>

      <template v-else-if="activeView === 'samples'">
        <section class="section-panel sample-builder">
          <div class="section-heading">
            <div><h2>井震时空标定与样本构建</h2><p>水平定位、时间域标定与训练放行相互独立；候选窗口不会自动变成训练标签。</p></div>
            <span class="algorithm-badge">nearest_3d · k=9 · sonic tie</span>
          </div>
          <div class="contract-flow">
            <span>井实体与轨迹</span><b>→</b><span>最近三维地震体</span><b>→</b><span>9道距离加权参考</span><b>→</b>
            <span>实测时深表优先 / 声波合成候选</span><b>→</b><span>固定32样点窗口</span><b>→</b><span>质量门禁</span>
          </div>
          <div class="gate-grid">
            <article :class="{ ready: preparation?.gates.can_build_samples }"><span>水平候选</span><strong>{{ preparation?.gates.can_build_samples ? "可执行" : "未就绪" }}</strong><p>需要可定位井与带XY的三维地震几何</p></article>
            <article :class="{ ready: Boolean(sampleResult?.matching?.valid_window_count) }"><span>时间域标定</span><strong>{{ sampleResult?.matching ? `${(sampleResult.matching.valid_window_count ?? 0).toLocaleString()} 个有效窗口` : "等待执行" }}</strong><p>实测时深表优先；DT/VP合成记录只作为候选</p></article>
            <article :class="{ ready: Boolean(sampleResult?.matching?.training_eligible_count) }"><span>训练放行</span><strong>{{ sampleResult?.matching?.training_eligible_count ? `${sampleResult.matching.training_eligible_count.toLocaleString()} 个可训练` : "尚未放行" }}</strong><p>同时要求时深质量、水平置信度与CRS/单位已核验</p></article>
          </div>
          <div class="sample-actions">
            <button type="button" class="primary-button" :disabled="sampleRunning || !preparation?.gates.can_build_samples" @click="runSampleBuilding">
              {{ sampleRunning ? "正在对齐与构建…" : "启动井震空间对齐与样本构建" }}
            </button>
            <p v-if="!preparation?.gates.can_build_samples">请先完成数据准备并处理井位或地震坐标阻断问题。</p>
          </div>
          <div v-if="sampleRunning" class="task-progress">
            <div><span>{{ statusMessage }}</span><strong>{{ progress }}%</strong></div>
            <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
          </div>
          <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
        </section>
        <section v-if="sampleResult?.matching" class="summary-grid">
          <article><span>多模态样本</span><strong>{{ sampleResult.matching.sample_count.toLocaleString() }}</strong><p>包含水平候选与垂向状态</p></article>
          <article><span>有效时间窗</span><strong>{{ (sampleResult.matching.valid_window_count ?? 0).toLocaleString() }}</strong><p>严格固定为32个地震样点</p></article>
          <article><span>可训练样本</span><strong>{{ (sampleResult.matching.training_eligible_count ?? 0).toLocaleString() }}</strong><p>候选标定默认不放行</p></article>
          <article><span>坐标参考</span><strong>{{ sampleResult.matching.coordinate_reference_verified ? "已核验" : "待核验" }}</strong><p>CRS与水平单位需显式确认</p></article>
          <article class="full-stat"><span>输出目录</span><code>{{ sampleResult.matching.output_directory }}</code><p>原始数据没有被覆盖</p></article>
        </section>
      </template>

      <template v-else-if="activeView === 'models'">
        <ModelCenter :capabilities="capabilities" />
      </template>

      <template v-else-if="activeView === 'prediction'">
        <section v-if="runnablePredictionModels.length" class="section-panel faultseg-runner">
          <div class="section-heading">
            <div>
              <h2>{{ activePredictionTaskSpec?.name }}</h2>
              <p>{{ activePredictionTaskSpec?.description }} 任务、模型、输入适配器和推理运行器分别注册，可独立替换。</p>
            </div>
            <span class="algorithm-badge">正式推理接口</span>
          </div>
          <div class="gate-grid">
            <label>
              <span>推理模型</span>
              <select v-model="selectedPredictionModelId" class="form-select" @change="handlePredictionModelChange">
                <option v-for="model in runnablePredictionModels" :key="model.id" :value="model.id">{{ model.name }}</option>
              </select>
            </label>
            <label>
              <span>三维地震文件</span>
              <select v-model="predictionSeismicPath" class="form-select" @change="handlePredictionSourceChange">
                <option value="">请选择 SEG-Y</option>
                <option v-for="source in predictionSources" :key="source.path" :value="source.path">
                  {{ source.name }} · {{ source.samples_per_trace }} × {{ source.trace_count }} 道
                </option>
              </select>
            </label>
            <label v-if="!isSurfaceSegModel">
              <span>中心测试体块</span>
              <select v-model.number="predictionCropSize" class="form-select">
                <option :value="32">32³（快速验证）</option>
                <option :value="64">64³</option>
                <option :value="96">96³</option>
                <option :value="128">128³</option>
              </select>
            </label>
            <label>
              <span>计算设备</span>
              <select v-model="predictionDevice" class="form-select">
                <option value="auto">自动选择</option>
                <option value="cpu">CPU</option>
                <option value="cuda">GPU / CUDA</option>
              </select>
            </label>
            <label v-if="!isSurfaceSegModel">
              <span>判别阈值 {{ predictionThreshold.toFixed(2) }}</span>
              <input v-model.number="predictionThreshold" type="range" min="0" max="1" step="0.01" class="form-range" />
            </label>
            <template v-else>
              <label>
                <span>推理范围</span>
                <select v-model="surfaceSegScope" class="form-select">
                  <option value="smoke">验证模式（前若干 Inline）</option>
                  <option value="full">完整三维 SEG-Y（耗时与内存较大）</option>
                </select>
              </label>
              <label v-if="surfaceSegScope === 'smoke'">
                <span>验证 Inline 数</span>
                <input v-model.number="surfaceSegMaxInlines" type="number" min="1" max="32" step="1" class="form-select" />
              </label>
              <label>
                <span>Inline 数覆盖（高级，可选）</span>
                <input v-model.number="surfaceSegInlineCount" type="number" min="1" step="1" :placeholder="selectedModelCompatibility?.native_inline_count ? `平台已自动识别 ${selectedModelCompatibility.native_inline_count}` : '仅平台无法识别有序网格时填写'" class="form-select" />
              </label>
              <label>
                <span>幅值适配</span>
                <select v-model="surfaceSegAmplitudeMode" class="form-select">
                  <option value="auto">自动判断训练尺度 / 鲁棒分位</option>
                  <option value="robust">1%–99% 鲁棒分位</option>
                  <option value="training">固定训练幅值 [-1215, 1930]</option>
                </select>
              </label>
              <label>
                <span>实例查询阈值 {{ surfaceSegQueryThreshold.toFixed(2) }}</span>
                <input v-model.number="surfaceSegQueryThreshold" type="range" min="0" max="1" step="0.01" class="form-range" />
              </label>
              <label>
                <span>掩码阈值 {{ surfaceSegMaskThreshold.toFixed(2) }}</span>
                <input v-model.number="surfaceSegMaskThreshold" type="range" min="0" max="1" step="0.01" class="form-range" />
              </label>
              <label>
                <span>SegFormer 批量</span>
                <select v-model.number="surfaceSegformerBatchSize" class="form-select">
                  <option :value="1">1（最低显存）</option>
                  <option :value="2">2（推荐）</option>
                  <option :value="4">4</option>
                  <option :value="8">8</option>
                </select>
              </label>
              <label>
                <span>Mask2Former 批量</span>
                <select v-model.number="surfaceMask2formerBatchSize" class="form-select">
                  <option :value="1">1（8 GB 显存推荐）</option>
                  <option :value="2">2</option>
                  <option :value="4">4</option>
                </select>
              </label>
            </template>
          </div>
          <div v-if="isSurfaceSegModel" class="model-runtime-notice">
            <strong>逐 Inline 二维实例分割</strong>
            <p>模型会把每个 Inline 的 [Sample, Crossline] 剖面缩放到 512×512，依次运行 SegFormer Base、SegFormer Refine 与 Mask2Former。标签按单张 Inline 由浅到深编号，暂不代表跨 Inline 连续的同一层位面。</p>
          </div>
          <section class="model-data-lineage" aria-label="数据到模型输入状态">
            <article :class="{ ready: Boolean(selectedPredictionSource) }">
              <span>01</span><div><strong>源数据</strong><small>{{ selectedPredictionSource?.name || '尚未选择 SEG-Y' }}</small></div>
            </article>
            <b>→</b>
            <article :class="{ ready: selectedPredictionSource?.dimension === '三维地震体' }">
              <span>02</span><div><strong>三维几何</strong><small>{{ selectedPredictionSource?.shape_zyx?.join(' × ') || '等待 Inline/Crossline 重建' }}</small></div>
            </article>
            <b>→</b>
            <article :class="{ ready: previewVolumes.some((volume) => volume.path === predictionSeismicPath) }">
              <span>03</span><div><strong>可视化预览</strong><small>{{ previewVolumes.some((volume) => volume.path === predictionSeismicPath) ? '稀疏体已生成' : '当前数据任务未生成预览' }}</small></div>
            </article>
            <b>→</b>
            <article :class="{ ready: predictionInputReady }">
              <span>04</span><div><strong>模型输入适配</strong><small>{{ predictionCompatibilityReason }}</small></div>
            </article>
          </section>
          <div class="sample-actions">
            <button type="button" class="primary-button" :disabled="predictionRunning || !predictionInputReady || !selectedPredictionModelId" @click="runPrediction">
              {{ predictionRunning ? `正在运行${selectedPredictionModel?.name || "模型"}…` : `启动${selectedPredictionModel?.name || "模型"}推理` }}
            </button>
            <p>{{ selectedModelAdapter ? `输入轴序 ${selectedModelAdapter.tensor_axes.join(" / ")}；${selectedModelAdapter.normalization}。` : "模型输入参数由适配器声明。" }}</p>
          </div>
          <div v-if="predictionRunning" class="task-progress">
            <div><span>{{ statusMessage }}</span><strong>{{ progress }}%</strong></div>
            <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
          </div>
          <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
          <div v-if="predictionResult && predictionVisualizationUrl" class="prediction-visualization-entry">
            <div>
              <span>{{ predictionResult.model_id }} · CIGVIS</span>
              <strong>{{ predictionResult.task_name }}结果已可叠加查看</strong>
              <p v-if="predictionResult.segmentation">以 seismic 地震振幅为背景，用离散色表查看逐 Inline 地层实例标签。</p>
              <p v-else>以原始地震振幅为背景，查看概率体与阈值结果。</p>
            </div>
            <button type="button" class="primary-button" @click="showPredictionVisualization">在 CIGVis 查看结果</button>
          </div>
          <div v-if="predictionResult" class="summary-grid">
            <article><span>{{ predictionResult.segmentation ? "输入剖面体" : "输入体块" }}</span><strong>{{ predictionInputShape.join(' × ') }}</strong><p>{{ predictionInputAxes.join(' / ') }}</p></article>
            <template v-if="predictionResult.probability">
              <article><span>输出概率范围</span><strong>{{ predictionResult.probability.min.toFixed(4) }}–{{ predictionResult.probability.max.toFixed(4) }}</strong><p>均值 {{ predictionResult.probability.mean.toFixed(4) }}</p></article>
              <article><span>阳性体素比例</span><strong>{{ (predictionResult.probability.positive_fraction * 100).toFixed(2) }}%</strong><p>阈值 {{ Number(predictionResult.inference.threshold ?? 0.5).toFixed(2) }}</p></article>
            </template>
            <template v-else-if="predictionResult.segmentation">
              <article><span>地层实例标签</span><strong>{{ predictionResult.segmentation.label_range.join('–') }}</strong><p>{{ predictionResult.segmentation.instance_count }} 个逐剖面标签</p></article>
              <article><span>平均置信度</span><strong>{{ Number(predictionResult.segmentation.confidence_mean ?? 0).toFixed(4) }}</strong><p>连续分值仅表示模型响应，不是校准概率</p></article>
            </template>
            <article><span>运行设备</span><strong>{{ predictionResult.device }}</strong><p>结果已写入本地输出目录</p></article>
          </div>
          <figure v-if="predictionOverviewUrl" class="prediction-overview">
            <figcaption><strong>代表性 Inline 分割总览</strong><span>地震剖面与地层实例标签对照</span></figcaption>
            <img :src="predictionOverviewUrl" alt="地层分割代表性 Inline 总览" />
          </figure>
          <div v-if="predictionResult" class="output-list">
            <article v-for="[name, path] in predictionOutputEntries" :key="name"><span>{{ name }}</span><code>{{ path }}</code></article>
          </div>
        </section>
        <section v-else class="section-panel prediction-task-empty">
          <div class="section-heading">
            <div>
              <h2>{{ activePredictionTaskSpec?.name }}</h2>
              <p>{{ activePredictionTaskSpec?.description }} 当前尚未注册可运行模型；模型接入后会根据任务绑定自动出现在这里。</p>
            </div>
            <span class="algorithm-badge">等待模型接入</span>
          </div>
          <div class="model-data-lineage compact-lineage">
            <article class="ready"><span>01</span><div><strong>共享数据层</strong><small>复用 SEG-Y / LAS / 井轨迹标准化结果</small></div></article>
            <b>→</b>
            <article><span>02</span><div><strong>任务模型</strong><small>{{ activePredictionTaskSpec?.required_modalities.join(" / ") || "等待声明输入模态" }}</small></div></article>
            <b>→</b>
            <article><span>03</span><div><strong>专属输出</strong><small>{{ activePredictionTaskSpec?.outputs.join(" / ") }}</small></div></article>
          </div>
        </section>
      </template>

      <template v-else-if="activeView === 'evaluation'">
        <section class="section-panel">
          <div class="section-heading"><div><h2>可交付结果</h2><p>没有真实模型结果时不展示虚构指标。</p></div></div>
          <div class="deliverable-grid">
            <article><span>01</span><div><strong>数据质量报告</strong><p>读取、清洗、单位统一、井实体合并与问题来源记录</p></div><b>数据准备后</b></article>
            <article><span>02</span><div><strong>井震对齐报告</strong><p>空间距离、匹配置信度、有效掩码与算法版本</p></div><b>样本构建后</b></article>
            <article><span>03</span><div><strong>模型评估报告</strong><p>按任务接入指标、阈值、版本及测试集范围</p></div><b>模型接入后</b></article>
            <article><span>04</span><div><strong>解释成果导出</strong><p>概率体、分割体、井旁结果及可追溯索引</p></div><b>推理完成后</b></article>
          </div>
          <div v-if="sampleResult?.matching" class="output-list">
            <article v-for="(path, name) in sampleResult.matching.output_files" :key="name">
              <span>{{ name }}</span><code>{{ path }}</code>
            </article>
          </div>
          <div v-else class="empty-inline">完成样本构建后，这里将展示质量报告和样本输出；接入模型后再增加指标、预测体和版本对比。</div>
        </section>
      </template>

      <template v-else-if="activeView === 'settings'">
        <section class="section-panel llm-settings-panel">
          <div class="section-heading">
            <div>
              <h2>LLM受控判断接口</h2>
              <p>规则与知识库优先；未决问题可生成白名单转换适配器，自动测试后由人工启用，同时支持带任务上下文的对话诊断。</p>
            </div>
            <span :class="['llm-status-pill', { ready: capabilities?.llm.available }]">
              {{ capabilities?.llm.available ? "已就绪" : capabilities?.llm.configured ? "未启用" : "待配置" }}
            </span>
          </div>
          <div class="llm-config-overview">
            <article><span>提供方 / 模式</span><strong>{{ capabilities?.llm.provider || "—" }} · {{ capabilities?.llm.api_mode || "—" }}</strong></article>
            <article><span>模型</span><strong>{{ capabilities?.llm.model || "未配置" }}</strong></article>
            <article><span>采纳阈值</span><strong>{{ capabilities ? `${Math.round(capabilities.llm.min_confidence * 100)}%` : "—" }}</strong></article>
            <article><span>单任务调用上限</span><strong>{{ capabilities?.llm.max_calls_per_task ?? "—" }}</strong></article>
          </div>
          <div class="llm-policy-row">
            <div><span>触发策略</span><p>{{ capabilities?.llm.trigger_policy }}</p></div>
            <div><span>数据边界</span><p>{{ capabilities?.llm.data_policy }}</p></div>
          </div>
          <div v-if="capabilities?.llm.missing.length" class="llm-missing">
            <span>后端还缺少</span><code>{{ capabilities.llm.missing.join(" · ") }}</code>
          </div>
        </section>
        <section class="section-panel">
          <div class="section-heading"><div><h2>知识库与算法配置</h2><p>配置与实现分离，比赛数据变化时优先增加映射或厂商配置，不修改核心代码。</p></div></div>
          <div class="settings-grid">
            <article v-for="library in capabilities?.configuration_libraries" :key="library.id">
              <span>可独立维护</span><h2>{{ library.name }}</h2><code>{{ library.file }}</code>
            </article>
          </div>
        </section>
      </template>
    </main>

    <FloatingAssistant
      :visible="activeView !== 'overview'"
      :task-id="taskId"
      :context-label="currentMeta.title"
      :llm-available="Boolean(capabilities?.llm.available)"
      :llm-model="capabilities?.llm.model"
      @navigate="handleAssistantNavigation"
    />
  </div>
</template>
