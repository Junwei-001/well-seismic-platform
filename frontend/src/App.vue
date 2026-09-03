<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
  acceptRegistrationCandidate,
  activateTransformationDraft,
  autofillPreparationIssues,
  clearSystemCache,
  classifyRegistrationPreflightFailure,
  confirmPreparationIssue,
  confirmRuntimeContract,
  createDataPreparation,
  createHorizontalRegistration,
  createRegistration,
  createSampleBuilding,
  createPrediction,
  demoPaths,
  getCapabilities,
  getProjects,
  getProjectSnapshots,
  getReleaseCatalog,
  getSnapshotDetail,
  getSystemCache,
  getTask,
  generateTransformationDraft,
  health,
  isHorizontalRegistrationTaskResult,
  isWorkflowResult,
  preflightRegistration,
  setActiveProjectSnapshot,
  ApiRequestError,
  type BackgroundTask,
  type Capabilities,
  type DataPathsPayload,
  type DemoPaths,
  type HorizontalRegistrationTaskResult,
  type PreparationIssue,
  type PreparationStage,
  type PredictionTaskCapability,
  type PredictionResult,
  type PredictionTaskResult,
  type RegistrationConsumptionStatus,
  type ReleaseCatalogResponse,
  type RuntimeContractConfirmationResponse,
  type RuntimeContractReview,
  type RuntimeContractReviewField,
  type RuntimeContractValue,
  type SeismicAssetSummary,
  type SnapshotDetailResponse,
  type SourceSnapshotSummary,
  type SourceContractCandidate,
  type StandardResultArtifact,
  type SurveyAttestationPayload,
  type SystemCacheStatus,
  type TransformationDraft,
  type WellLogPreviewCurve,
  type WorkflowResult,
} from "./api";
import heroInterpretationCenterImage from "./assets/首屏_井震智能解释中心.jpg";
import FloatingAssistant from "./components/assistant/FloatingAssistant.vue";
import LayerPulseWorkbench from "./components/layerpulse/LayerPulseWorkbench.vue";
import ModelCenter from "./components/models/ModelCenter.vue";
import PostFusionInferenceDialog from "./components/workflow/PostFusionInferenceDialog.vue";
import {
  availableCurveForSlot,
  conventionalCoverage,
  conventionalCurveGroups,
  createDefaultPathGroups,
  fallbackInterpretationTasks,
  isWellPropertyCompletionModelId,
  isPrimaryPredictionTaskId,
  navigation,
  navigationIconPaths,
  primaryPredictionTasks,
  viewMeta,
  type PathGroup,
  type PathGroupKey,
  type PredictionTaskKey,
  type PreparationScreen,
  type ViewKey,
} from "./domain/platform";
import {
  modelPresentationName,
  modelSpecPresentationName,
  publicModelIdentifier,
  publicModelText,
  scientificStatusLabel,
} from "./domain/modelPresentation";
import {
  centeredTaskScrollTop,
  collectLatestPredictionRuns,
  type PredictionRunHistoryEntry,
} from "./domain/predictionTaskRail";
import {
  countDownRemainingSeconds,
  estimateProgressRemainingSeconds,
} from "./domain/fusionProgress";
import { buildSourceDataStatistics } from "./domain/sourceDataStatistics";
import {
  LAYER_PULSE_MODEL_ID,
  LAYER_PULSE_TASK_ID,
  createIdleLayerPulseTaskState,
  isLayerPulseTaskActive,
  layerPulseModelContract,
  layerPulseOutputCatalog,
  summarizeLayerPulseSupport,
  type LayerPulseSupportCheck,
  type LayerPulseSupportReceipt,
  type LayerPulseTaskState,
} from "./domain/layerPulse";

const LAST_TASK_STORAGE_KEY = "strata_vision_last_task";
const LAST_SOURCE_SNAPSHOT_STORAGE_KEY = "strata_vision_last_source_snapshot";
const RUNTIME_CONTRACT_SUBMISSION_STORAGE_PREFIX = "strata_vision_runtime_contract_submission";
const LAST_PREDICTION_TASK_STORAGE_KEY = "strata_vision_last_prediction_task";
const LAST_LAYERPULSE_TASK_STORAGE_KEY = "strata_vision_last_layerpulse_task";
const LAST_GEOPATH_CANDIDATE_TASK_STORAGE_KEY = "strata_vision_last_geopath_candidate_task";
const LAST_GEOPATH_ACCEPTED_REGISTRATION_STORAGE_KEY = "strata_vision_last_geopath_accepted_registration";
const PATH_CONFIG_STORAGE_KEY = "strata_vision_path_config";
const FRESH_SESSION_STORAGE_KEY = "strata_vision_fresh_session";
const CACHE_FLASH_STORAGE_KEY = "strata_vision_cache_flash";
const ACTIVE_SNAPSHOT_PERSISTENCE_WARNING_PREFIX = "当前SourceSnapshot未能写入平台状态库：";
const GEOPATH_MINIMUM_AUTOMATIC_GEOMETRY_CONFIDENCE = 0.75;
const GEOPATH_MINIMUM_AUTOMATIC_HEADER_CONFIDENCE = 0.9;
const TASK_POLL_TIMEOUT_MS = 30 * 60 * 1000;
type CalibrationOutcome =
  | { kind: "formal"; taskId: string; result: WorkflowResult }
  | { kind: "fusion_blocked"; taskId: string; result: WorkflowResult }
  | { kind: "horizontal_only"; taskId: string; result: HorizontalRegistrationTaskResult }
  | { kind: "blocked_integrity"; detail: string }
  | { kind: "failed"; detail: string };
interface LayerPulseDownloadLink {
  url: string;
  filename: string;
}
interface LayerPulseOutputDownloads {
  segy: LayerPulseDownloadLink;
  rawNpy: LayerPulseDownloadLink;
  classLegendCsv?: LayerPulseDownloadLink;
}
interface AcceptanceArtifact {
  name: string;
  path: string;
  directory: boolean;
}
interface AcceptanceGroup {
  taskId: string;
  label: string;
  family: string;
  executionStatus: BackgroundTask["status"];
  visualizationStatus: "source" | "accepted" | "candidate" | "evidence_only" | "unavailable";
  scientificStatus: string;
  acceptanceStatus: "accepted" | "candidate" | "source_evidence" | "not_evaluated" | "failed";
  artifacts: AcceptanceArtifact[];
  sequenceArtifacts: AcceptanceArtifact[];
  warnings: string[];
  canVisualize: boolean;
  standardResult: boolean;
}
interface SequenceSeries {
  id: string;
  min: number;
  max: number;
  values: Array<number | null>;
}
interface SequencePreview {
  depthName: string;
  depths: number[];
  series: SequenceSeries[];
}
interface FractureDisplaySegment {
  level: "low" | "medium" | "high";
  label: string;
  topMd: number;
  bottomMd: number;
  thicknessM: number;
  sampleCount: number;
  y: number;
  height: number;
}
interface FractureIntervalPreview {
  depthMin: number;
  depthMax: number;
  segments: FractureDisplaySegment[];
}
interface PlanViewStation {
  x: number;
  y: number;
  traceX: number | null;
  traceY: number | null;
  covered: boolean;
}
interface PlanViewWell {
  name: string;
  geometryMode: string;
  horizontalStatus: string;
  stations: PlanViewStation[];
}
interface PlanViewPreview {
  horizontalCrs: string;
  polygons: Array<Array<{ x: number; y: number }>>;
  wells: PlanViewWell[];
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
}
let componentUnmounted = false;
let preparationClockTimer: number | null = null;
let fusionClockTimer: number | null = null;
let lastPersistedActiveSnapshot = "";
let latestRequestedActiveSnapshot = "";
let activeSnapshotPersistencePendingCount = 0;
let activeSnapshotPersistenceQueue: Promise<void> = Promise.resolve();
let activeSnapshotReconciliationPromise: Promise<boolean> | null = null;
const groups = ref<PathGroup[]>(createDefaultPathGroups());

const activeView = ref<ViewKey>("overview");
const backendStatus = ref<"checking" | "online" | "offline">("checking");
const backendVersion = ref("");
const capabilities = ref<Capabilities | null>(null);
const CONFIGURATION_LIBRARY_POSITIONS: Record<string, { x: number; y: number }> = {
  curve_knowledge: { x: 15, y: 16 },
  units: { x: 37, y: 10 },
  well_schema: { x: 64, y: 10 },
  vertical_datum: { x: 85, y: 17 },
  segy_profiles: { x: 88, y: 42 },
  preprocessing: { x: 86, y: 70 },
  matching: { x: 69, y: 89 },
  fusion: { x: 43, y: 90 },
  fault_models: { x: 20, y: 83 },
  surface_seg: { x: 12, y: 58 },
  llm: { x: 17, y: 34 },
};
const configurationLibraryNodes = computed(() => {
  const libraries = capabilities.value?.configuration_libraries ?? [];
  const count = Math.max(libraries.length, 1);
  return libraries.map((library, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / count;
    const fallbackPosition = {
      x: 50 + Math.cos(angle) * 38,
      y: 50 + Math.sin(angle) * 38,
    };
    const { x, y } = CONFIGURATION_LIBRARY_POSITIONS[library.id] ?? fallbackPosition;
    return {
      ...library,
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(3)),
      style: {
        "--knowledge-x": `${x.toFixed(3)}%`,
        "--knowledge-y": `${y.toFixed(3)}%`,
        "--knowledge-drift-delay": `${-((index % 7) * 0.53)}s`,
        "--knowledge-glow-delay": `${-((index % 5) * 0.41)}s`,
      } as Record<string, string>,
    };
  });
});
const releaseCatalog = ref<ReleaseCatalogResponse | null>(null);
const releaseCatalogLoading = ref(true);
const releaseCatalogError = ref("");
const cacheStatus = ref<SystemCacheStatus | null>(null);
const cacheStatusLoading = ref(true);
const cacheStatusError = ref("");
const cacheClearing = ref(false);
const cacheMessage = ref("");
const demo = ref<DemoPaths | null>(null);
const recursive = ref(true);
const lightweight = ref(true);
const useLlmFallback = ref(false);
const horizontalCrsId = ref("");
const wellSourceCrsId = ref("");
const seismicSourceCrsId = ref("");
const horizontalUnit = ref<"m" | "ft" | "unknown">("unknown");
const horizontalAxisOrder = ref<"XY" | "YX" | "unknown">("unknown");
const coordinateReferenceVerified = ref(false);
const seismicSrdElevation = ref<number | "">("");
const verticalCrsId = ref("LOCAL_MSL_UNSPECIFIED");
const seismicReplacementVelocity = ref<number | "">("");
const seismicTimeDomain = ref<"TWT" | "OWT" | "unknown">("unknown");
const seismicCorrectionState = ref<"corrected_to_srd" | "uncorrected" | "unknown">("unknown");
const segyGeometryProfile = ref("");
const segyInlineByte = ref<number | "">("");
const segyCrosslineByte = ref<number | "">("");
const segyXByte = ref<number | "">("");
const segyYByte = ref<number | "">("");
const segyCoordinateScalarByte = ref<number | "">("");
const wellCoordinateSourceUnit = ref<"m" | "ft" | "unknown">("unknown");
const wellVerticalDatumSourceUnit = ref<"m" | "ft" | "unknown">("unknown");
const lasTwtSourceUnit = ref<"ms" | "s" | "us" | "unknown">("unknown");
const sealedSnapshotSegyContract = ref<Partial<DataPathsPayload> | null>(null);
const sealedSnapshotSegyGeometryAuthority = ref<"explicit" | "verified_automatic" | null>(null);
const surveyAttestation = ref<SurveyAttestationPayload | null>(null);
const autoPopulatedSurveyContractFields = ref<string[]>([]);
const preparationRunning = ref(false);
type PreparationActivityPhase =
  | "idle"
  | "submitting"
  | "validating"
  | "cataloging"
  | "reading"
  | "hashing"
  | "summarizing"
  | "caching"
  | "autofill"
  | "reconnecting"
  | "completed"
  | "failed";
interface PreparationRunTiming {
  taskId: string;
  startedAt: string;
  finishedAt: string;
  durationSeconds: number | null;
  finishedSource: "completed_at" | "updated_at";
}
const preparationProgress = ref(0);
const preparationStatusMessage = ref("等待登记数据");
const preparationActivityPhase = ref<PreparationActivityPhase>("idle");
const preparationStartedAt = ref<number | null>(null);
const preparationReadingStartedAt = ref<number | null>(null);
const preparationClockNow = ref(0);
const preparationWorkDone = ref(0);
const preparationWorkTotal = ref(0);
const preparationCurrentItem = ref("");
const preparationCurrentItemSizeBytes = ref<number | null>(null);
const preparationCurrentItemStartedAt = ref<number | null>(null);
const preparationSubworkDone = ref(0);
const preparationSubworkTotal = ref(0);
const preparationWorkUnit = ref<"assets" | "bytes" | null>(null);
const preparationSubworkUnit = ref<"traces" | "bytes" | null>(null);
const preparationEtaBaselineSeconds = ref<number | null>(null);
const preparationEtaComputedAt = ref<number | null>(null);
const preparationEtaSampleSubworkDone = ref(0);
const preparationHistoricalEstimateSeconds = ref<number | null>(null);
const preparationHistoricalEstimateSamples = ref(0);
const preparationHistoricalEstimateConfidence = ref<"low" | "medium" | null>(null);
const preparationRunTiming = ref<PreparationRunTiming | null>(null);
const runtimeContractDialog = ref<HTMLDialogElement | null>(null);
const runtimeContractReview = ref<RuntimeContractReview | null>(null);
const runtimeContractDraft = ref<Record<string, RuntimeContractValue>>({});
const runtimeContractSourceSnapshotId = ref("");
const runtimeContractSubmitting = ref(false);
const runtimeContractError = ref("");
const runtimeContractSubmission = ref<{
  fingerprint: string;
  attestation: SurveyAttestationPayload;
  confirmation?: RuntimeContractConfirmationResponse;
} | null>(null);
interface PostFusionInferenceContext {
  snapshotId: string;
  registrationTaskId: string;
  preparedViewId: string;
  readyWellCount: number;
}
const postFusionInferenceOpen = ref(false);
const postFusionInferenceContext = ref<PostFusionInferenceContext | null>(null);
const batchApplyingRecommendations = ref(false);
const sampleRunning = ref(false);
const registrationRunning = ref(false);
type FusionActivityPhase = "registration" | "prepared_view" | "reconnecting";
const fusionActivityPhase = ref<FusionActivityPhase>("registration");
const fusionStartedAt = ref<number | null>(null);
const fusionClockNow = ref(0);
const fusionEtaBaselineSeconds = ref<number | null>(null);
const fusionEtaComputedAt = ref<number | null>(null);
const fusionEtaSampleProgress = ref(0);
const horizontalRegistrationTaskId = ref("");
const predictionRunning = ref(false);
const predictionOrchestrationRunning = ref(false);
const predictionTaskSwitching = ref(false);
const geoPathCandidateRunning = ref(false);
const geoPathAcceptanceRunning = ref(false);
const progress = ref(0);
const statusMessage = ref("等待登记数据");
const taskId = ref("");
const activeProjectId = ref("local-default");
const dataSnapshotTaskId = ref("");
const registrationTaskId = ref("");
const sampleBuildingTaskId = ref("");
const predictionTaskId = ref("");
const geoPathCandidateTaskId = ref("");
const geoPathAcceptedRegistrationTaskId = ref("");
const preparationScreen = ref<PreparationScreen>("input");
const preparationResult = ref<WorkflowResult | null>(null);
const registrationResult = ref<WorkflowResult | null>(null);
const horizontalRegistrationResult = ref<HorizontalRegistrationTaskResult | null>(null);
const sampleResult = ref<WorkflowResult | null>(null);
const predictionResult = ref<PredictionResult | null>(null);
const layerPulseResult = ref<PredictionResult | null>(null);
const layerPulseExecutionTaskId = ref("");
const layerPulseSourceSnapshotId = ref("");
const layerPulseTaskState = ref<LayerPulseTaskState>(createIdleLayerPulseTaskState());
const layerPulseSelectedOutputKey = ref("fault_logits");
const layerPulseCanvasMode = ref<"base" | "result">("base");
const predictionHistoryByTask = ref<Record<string, PredictionRunHistoryEntry<PredictionResult>>>({});
const predictionHistorySnapshotId = ref("");
const predictionTaskRailElement = ref<HTMLElement | null>(null);
const layerPulseTaskRailElement = ref<HTMLElement | null>(null);
const sidebarDirectoryLevel = ref<"primary" | "prediction" | "layerpulse">("primary");
let predictionTaskSelectionSequence = 0;
const geoPathCandidateResult = ref<PredictionResult | null>(null);
const geoPathAcceptedRegistrationResult = ref<WorkflowResult | null>(null);
const geoPathSelectedWellIds = ref<string[]>([]);
const geoPathAcceptanceConfirmed = ref(false);
const geoPathReviewNote = ref("");
const preparationTargetTaskId = ref<PredictionTaskKey>("");
const preparationTargetModelId = ref("");
const preparationScopeNotice = ref("");
// Data preparation starts task-neutral. A geological task only narrows the
// preparation contract after the user explicitly selects it.
const activePredictionTask = ref<PredictionTaskKey>("");
const runningPredictionTask = ref<PredictionTaskKey>("");
const runningPredictionExecutionTaskId = ref("");
const predictionConnectionState = ref<"idle" | "online" | "retrying">("idle");
const predictionLastHeartbeatAt = ref<number | null>(null);
const selectedPredictionModelId = ref("");
const predictionSeismicPath = ref("");
const predictionCropSize = ref<32 | 64 | 96 | 128>(32);
const predictionThreshold = ref(0.5);
const predictionDevice = ref<"auto" | "cpu" | "cuda">("auto");
const FAULT_VOLUME_MODEL_IDS = new Set(["faultseg_3d", "faultnet_china_field"]);
function isFaultVolumeModelId(value: unknown): boolean {
  return FAULT_VOLUME_MODEL_IDS.has(String(value || ""));
}
type FaultSegScope = "center_block_1" | "full_volume";
function normalizeFaultSegScope(value: unknown): FaultSegScope {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "full_volume") return "full_volume";
  return "center_block_1";
}
const faultSegScope = ref<FaultSegScope>(normalizeFaultSegScope(undefined));
const surfaceSegScope = ref<"smoke" | "full">("full");
const surfaceSegMaxInlines = ref(2);
const surfaceSegInlineCount = ref<number | "">("");
const surfaceSegAmplitudeMode = ref<"auto" | "training" | "robust">("auto");
const surfaceSegQueryThreshold = ref(0.35);
const surfaceSegMaskThreshold = ref(0.5);
const surfaceSegformerBatchSize = ref(2);
const surfaceMask2formerBatchSize = ref(1);
const f3FaciesMode = ref<"auto_roi" | "manual_roi" | "single_trace">("auto_roi");
const f3TStart = ref(850);
const f3TCount = ref(256);
const f3InlineStart = ref(1000);
const f3InlineCount = ref(16);
const f3CrosslineStart = ref(300);
const f3CrosslineCount = ref(16);
const errorMessage = ref("");
const registrationPreparationRequired = ref(false);
const restorationWarning = ref("");
const restorationWarningMessages: string[] = [];
const issueFilter = ref("全部");
const confirmingIssueId = ref("");
const acceptanceSnapshotDetail = ref<SnapshotDetailResponse | null>(null);
const acceptanceSnapshotCatalog = ref<SourceSnapshotSummary[]>([]);
const selectedAcceptanceSnapshotId = ref("");
const loadedAcceptanceSnapshotId = ref("");
const acceptanceLoading = ref(false);
const acceptanceError = ref("");
const selectedAcceptanceTaskId = ref("");
const selectedSequenceArtifact = ref("");
const sequencePreview = ref<SequencePreview | null>(null);
const fractureIntervalPreview = ref<FractureIntervalPreview | null>(null);
const sequenceLoading = ref(false);
const sequenceError = ref("");
const sequencePreviewLimited = ref(false);
const planViewPreview = ref<PlanViewPreview | null>(null);
const planViewLoading = ref(false);
const planViewError = ref("");
let acceptanceLoadSequence = 0;
let sequenceLoadSequence = 0;
let sequenceAbortController: AbortController | null = null;
let planViewLoadSequence = 0;
let planViewAbortController: AbortController | null = null;
const visualizationMode = ref<"seismic" | "logs">("seismic");
const predictionCanvasMode = ref<"auto" | "base" | "result">("auto");
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
// Source assets and canonical snapshot data always come from data preparation.
// Derived registration/sample results are fallbacks only for legacy task
// records where the preparation task cannot be recovered.
const workflowResult = computed(() => preparationResult.value || sampleResult.value || registrationResult.value);
const p13RegistrationPresentation = computed(() => {
  const registration = registrationResult.value?.registration;
  if (!registration) return null;

  const attempted = registration.p13_runtime_attempted === true;
  const executed = registration.p13_checkpoint_executed === true;
  const eligibleCount = Math.max(0, Number(registration.p13_eligible_well_count || 0));
  const executedCount = Math.max(0, Number(registration.p13_executed_well_count || 0));
  const acceptedCount = Math.max(0, Number(registration.p13_raw_candidate_accepted_count || 0));
  const registeredCount = Math.max(0, Number(registration.registered_well_count || 0));
  const fusionFeatureCount = Math.max(0, Number(registration.fusion_feature_well_count || 0));
  const sonicMethodCounts = Object.entries(registration.method_counts || {})
    .filter(([method]) => method.includes("sonic_integrated"));
  const sonicIntegratedCount = sonicMethodCounts.reduce(
    (total, [, count]) => total + Math.max(0, Number(count || 0)),
    0,
  );
  const sonicStaticTieCount = Math.max(
    0,
    Number(registration.method_counts?.sonic_integrated_static_tie || 0),
  );
  const physicalTrackLabel = sonicStaticTieCount > 0
    ? "DT/AC 积分 + 地震约束"
    : sonicIntegratedCount > 0
      ? "DT/AC 积分候选"
      : "物理回退";
  const geoPathHumanAccepted = registration.candidate_status === "human_accepted_for_experimental_downstream_use";
  const rejectedCount = Math.max(0, executedCount - acceptedCount);
  const rejectionReasons = (registration.p13_rejection_reasons || []).map(formatP13RejectionReason);

  let executionLabel = "未执行";
  if (executed) executionLabel = `${executedCount || eligibleCount || 1} 口已实际执行`;
  else if (attempted) executionLabel = "已尝试但未执行成功";
  else if (registration.p13_runtime_status === "not_eligible") executionLabel = "无满足输入合同的井";
  else if (!registration.p13_runtime_status) executionLabel = "历史结果未记录执行证据";

  const acceptanceLabel = executed
    ? `${acceptedCount} / ${executedCount || eligibleCount || 1} 口通过`
    : "无已执行候选";

  const primaryTrackLabel = registeredCount > 0
    ? physicalTrackLabel
    : "无可用物理主轨";
  const featureLabel = fusionFeatureCount > 0
    ? `${fusionFeatureCount} 口已绑定`
    : acceptedCount > 0
      ? "raw 候选未绑定"
      : "无";

  let detail = registration.registration_source_policy || "";
  if (geoPathHumanAccepted) {
    detail = `已逐井人工接受 ${registeredCount} 口轨迹感知实验候选；该候选不改写物理主轨，冻结概率约束与物理轨仍作为父级先验保留。当前不确定度未校准。`;
  } else if (registration.p13_runtime_status === "executed_product_invalid_physics_fallback") {
    detail = "冻结概率标定模型已实际执行，但候选产品合同校验失败；未覆盖当前物理回退主轨。";
  } else if (executed && rejectedCount > 0) {
    const reason = rejectionReasons.length ? rejectionReasons.join("；") : "未通过物理一致性门";
    detail = `冻结概率标定模型已实际执行；${rejectedCount} 口原始候选被拒绝（${reason}），未覆盖当前主轨。`;
  } else if (acceptedCount > 0 && fusionFeatureCount > 0) {
    detail = `${acceptedCount} 口概率标定候选通过物理门，其中 ${fusionFeatureCount} 口已形成父身份绑定的独立消费特征；物理主轨仍为${physicalTrackLabel}。`;
  } else if (acceptedCount > 0) {
    detail = `${acceptedCount} 口概率标定候选通过初步物理门，但未形成可消费的独立融合特征；不会被显示为主轨或融合成果。`;
  } else if (registration.p13_runtime_status === "execution_failed_physics_fallback") {
    detail = "冻结概率标定模型运行发生异常；当前可用结果来自物理回退。";
  } else if (registration.p13_runtime_status === "runtime_unavailable") {
    detail = "冻结概率标定模型的 GPU 运行环境当前不可用；当前可用结果来自物理回退。";
  } else if (registration.p13_runtime_status === "runner_missing") {
    detail = "未找到冻结概率标定模型的在线运行器；当前可用结果来自物理回退。";
  } else if (registration.p13_runtime_status === "not_eligible") {
    detail = `当前井不满足冻结概率标定模型的完整轨迹与声波输入合同；可用结果来自${physicalTrackLabel}。`;
  } else if (sonicStaticTieCount > 0) {
    detail = `${sonicStaticTieCount} 口井使用 AC/DT 积分先验并通过地震约束；无独立时深表带来的不确定性仍保留在逐井标定记录中。`;
  } else if (sonicIntegratedCount > 0) {
    detail = `${sonicIntegratedCount} 口井已形成 AC/DT 积分候选，但尚未通过地震约束或高置信融合门；不会把候选标成已接受成果。`;
  }

  return {
    executed,
    acceptedCount,
    rejectedCount,
    executionLabel,
    acceptanceLabel,
    primaryTrackLabel,
    featureLabel,
    rejectionLabel: rejectionReasons.join("；") || "无",
    detail,
  };
});
const registrationTimeAxisPresentation = computed(() => {
  const execution = registrationResult.value?.registration?.execution_contract;
  const timeDomain = String(
    execution?.time_domain
    ?? execution?.timeDomain
    ?? "",
  ).trim();
  const timeAxisReady = execution?.time_axis_ready ?? execution?.timeAxisReady;
  if (timeAxisReady === true && timeDomain.toUpperCase() === "TWT") {
    return {
      provenTwt: true,
      nativeTimeUnknown: false,
      resultLabel: "口井有已证明的 TWT 时间主轨",
      completionLabel: "TWT 时间主轨",
      currentLabel: "当前主轨",
      limitation: "",
    };
  }
  if (timeDomain.toLowerCase() === "native_time_unknown" || timeAxisReady === false) {
    return {
      provenTwt: false,
      nativeTimeUnknown: true,
      resultLabel: "口井有原生 SEG-Y 时间候选",
      completionLabel: "原生 SEG-Y 时间候选",
      currentLabel: "当前候选",
      limitation: "原生时间轴域尚未证明，未授予 fusion 资格。",
    };
  }
  return {
    provenTwt: false,
    nativeTimeUnknown: false,
    resultLabel: "口井有井震时间候选",
    completionLabel: "井震时间候选",
    currentLabel: "当前候选",
    limitation: "任务结果未登记可证明的 TWT 时间轴，按候选展示且不授予 fusion 资格。",
  };
});
const predictionTaskDefinitions = computed<PredictionTaskCapability[]>(() =>
  primaryPredictionTasks(
    capabilities.value?.prediction_tasks?.length
      ? capabilities.value.prediction_tasks.filter(
          (task) => task.active !== false && (
            task.show_in_prediction_menu === true
            || (task.show_in_prediction_menu === undefined && isPrimaryPredictionTaskId(task.id))
          ),
        )
      : fallbackInterpretationTasks,
  ),
);
const activePredictionTaskSpec = computed(() =>
  predictionTaskDefinitions.value.find((task) => task.id === activePredictionTask.value)
  || null,
);
const isWellPropertyTask = computed(() => activePredictionTask.value === "well_property");
const preparationDraftTaskSpec = computed(() =>
  predictionTaskDefinitions.value.find((task) => task.id === preparationTargetTaskId.value)
  || null,
);
const preparationDraftModels = computed(() => {
  const task = preparationDraftTaskSpec.value;
  if (!task) return [];
  const modelIds = new Set([...task.model_ids, ...task.runnable_model_ids]);
  return (capabilities.value?.models || []).filter((model) => modelIds.has(model.id));
});
const predictionSources = computed(() => workflowResult.value?.seismic || []);
const selectedPredictionSource = computed(() =>
  predictionSources.value.find((source) => source.path === predictionSeismicPath.value) || null,
);
function isLayerPulseSeismicCandidate(source: SeismicAssetSummary): boolean {
  const dimension = String(source.dimension || "").trim().toLowerCase();
  const shape = source.shape_zyx || [];
  return source.trace_count > 0
    && source.samples_per_trace > 1
    && (
      dimension.includes("3d")
      || dimension.includes("三维")
      || (source.inline_count > 1 && source.crossline_count > 1)
      || (shape.length === 3 && shape.every((value) => Number(value) > 1))
    );
}
const layerPulseSeismicSources = computed(() => (
  preparationResult.value?.seismic || []
).filter(isLayerPulseSeismicCandidate));
const layerPulseSeismicAsset = computed(() => {
  const sources = layerPulseSeismicSources.value;
  const current = sources.find((source) => source.path === predictionSeismicPath.value) || null;
  if (current?.model_compatibility?.[LAYER_PULSE_MODEL_ID]?.ready) return current;
  return sources.find((source) => source.model_compatibility?.[LAYER_PULSE_MODEL_ID]?.ready)
    || (current?.model_compatibility?.[LAYER_PULSE_MODEL_ID] ? current : null)
    || sources.find((source) => Boolean(source.model_compatibility?.[LAYER_PULSE_MODEL_ID]))
    || current
    || sources[0]
    || null;
});
const layerPulseModelCapability = computed(() => (
  (capabilities.value?.models || []).find((model) => model.id === LAYER_PULSE_MODEL_ID) || null
));
const layerPulseSupportReceipt = computed<LayerPulseSupportReceipt | null>(() => {
  const snapshotId = dataSnapshotTaskId.value;
  if (!snapshotId) return null;
  const result = preparationResult.value;
  const reportedSnapshotId = result?.data_snapshot?.snapshot_id || "";
  const source = layerPulseSeismicAsset.value;
  const compatibility = source?.model_compatibility?.[LAYER_PULSE_MODEL_ID];
  const checks: LayerPulseSupportCheck[] = [];
  const warnings: string[] = [];

  if (!result) {
    checks.push({
      id: "snapshot_binding",
      label: "SourceSnapshot 绑定",
      status: "blocked",
      detail: "当前页面尚未恢复该封存快照的数据登记结果。",
      required: true,
      evidence: snapshotId,
    });
  } else if (reportedSnapshotId && reportedSnapshotId !== snapshotId) {
    checks.push({
      id: "snapshot_binding",
      label: "SourceSnapshot 绑定",
      status: "blocked",
      detail: "数据登记结果与当前 SourceSnapshot 不一致，已阻止跨快照复用。",
      required: true,
      evidence: `${reportedSnapshotId} != ${snapshotId}`,
    });
  } else {
    checks.push({
      id: "snapshot_binding",
      label: "SourceSnapshot 绑定",
      status: reportedSnapshotId ? "ready" : "degraded",
      detail: reportedSnapshotId
        ? "支持能力与当前封存快照精确绑定。"
        : "旧登记结果未回填快照字段，保留当前平台快照引用并交由服务端复核。",
      required: true,
      evidence: snapshotId,
    });
    if (!reportedSnapshotId) warnings.push("登记结果缺少内嵌 snapshot_id，未将其提升为完整支持。");
  }

  checks.push({
    id: "prepared_view_lineage",
    label: "融合标定与 PreparedView",
    status: preparedViewReady.value ? "ready" : "blocked",
    detail: preparedViewReady.value
      ? "SourceSnapshot、Registration 与 PreparedView 三重血缘一致，可绑定统一推理。"
      : "必须先完成当前快照的精细标定与融合视图。",
    required: true,
    evidence: preparedViewReady.value
      ? `${registrationTaskId.value} · ${sampleBuildingTaskId.value}`
      : snapshotId,
  });

  if (!source) {
    checks.push({
      id: "seismic_3d",
      label: "三维地震体",
      status: "blocked",
      detail: "当前封存快照没有可解释的三维地震体。",
      required: true,
    });
  } else if (!compatibility) {
    checks.push({
      id: "model_compatibility",
      label: "LayerPulse 输入适配",
      status: "degraded",
      detail: "三维地震可读，但登记结果没有返回该模型的兼容性收据；运行时仍需服务端复核。",
      required: true,
      evidence: source.path,
    });
    warnings.push("未伪造 LayerPulse ready 状态：model_compatibility 收据缺失。");
  } else {
    checks.push({
      id: "model_compatibility",
      label: "LayerPulse 输入适配",
      status: compatibility.ready ? "ready" : "blocked",
      detail: compatibility.reason || (compatibility.ready ? "模型输入适配已通过。" : "模型输入适配未通过。"),
      required: true,
      evidence: [compatibility.adapter, source.path].filter(Boolean).join(" · "),
    });
    if (!compatibility.ready && compatibility.reason) warnings.push(compatibility.reason);
  }

  const wellCount = result?.well_entities?.length
    || result?.wells?.length
    || result?.summary.wells
    || 0;
  const snapshotWellsSupported = layerPulseModelCapability.value?.metadata?.supports_snapshot_wells === true;
  checks.push({
    id: "well_assets",
    label: "井曲线与轨迹（可选）",
    status: wellCount > 0 && snapshotWellsSupported && preparedViewReady.value ? "ready" : "degraded",
    detail: wellCount > 0
      ? snapshotWellsSupported && preparedViewReady.value
        ? `已登记 ${wellCount} 口井，MD、曲线缺失掩码与轨迹可送入 Backbone。`
        : snapshotWellsSupported
          ? `已登记 ${wellCount} 口井；完成当前融合视图后可送入 Backbone。`
          : `已登记 ${wellCount} 口井，但当前预览桥接尚未消费 MD 与轨迹张量；本次按纯地震降级模式运行。`
      : "未登记井数据，本次按纯地震降级模式运行。",
    required: false,
    evidence: `${wellCount} wells`,
  });
  if (!wellCount || !snapshotWellsSupported || !preparedViewReady.value) {
    warnings.push("井语义锚点未进入当前 forward，井震匹配输出仅作为无井模式候选。");
  }
  checks.push({
    id: "time_depth_optional",
    label: "无时深表 forward",
    status: "ready",
    detail: "时深表不是最终 forward 必需输入，本次请求不会发送时深监督资产。",
    required: false,
    evidence: LAYER_PULSE_MODEL_ID,
  });

  const requiredBlocked = checks.some((check) => check.required && check.status === "blocked");
  const degraded = checks.some((check) => check.status === "degraded");
  return {
    contract_version: "layerpulse.platform-support.v1",
    snapshot_id: snapshotId,
    model_id: LAYER_PULSE_MODEL_ID,
    status: requiredBlocked ? "blocked" : degraded ? "degraded" : "ready",
    checks,
    warnings,
    dataset_label: source?.name || "当前封存快照",
  };
});
const runnablePredictionModels = computed(() =>
  (capabilities.value?.models || []).filter((model) =>
    activePredictionTaskSpec.value?.runnable_model_ids.includes(model.id),
  ),
);
const selectedPredictionModel = computed(() =>
  runnablePredictionModels.value.find((model) => model.id === selectedPredictionModelId.value) || null,
);
const faultSegSnapshotSeismicSources = computed(() => predictionSources.value.filter((source) => {
  const dimension = String(source.dimension || "").trim().toLowerCase();
  const shape = source.shape_zyx || [];
  return source.trace_count > 0
    && source.samples_per_trace > 1
    && (
      dimension.includes("3d")
      || dimension.includes("三维")
      || (source.inline_count > 1 && source.crossline_count > 1)
      || (shape.length === 3 && shape.every((value) => Number(value) > 1))
    );
}));
const faultSegSnapshotSourceReady = computed(() => Boolean(
  dataSnapshotTaskId.value
  && faultSegSnapshotSeismicSources.value.length === 1
  && selectedPredictionSource.value?.path === faultSegSnapshotSeismicSources.value[0]?.path,
));
const faultSegSnapshotSourceReason = computed(() => {
  if (!dataSnapshotTaskId.value) return "请先完成数据准备并封存 SourceSnapshot。";
  if (!faultSegSnapshotSeismicSources.value.length) return "当前 SourceSnapshot 未识别出可运行的三维 SEG-Y。";
  if (faultSegSnapshotSeismicSources.value.length > 1) {
    return `当前 SourceSnapshot 含 ${faultSegSnapshotSeismicSources.value.length} 个三维 SEG-Y；断层识别任务要求唯一输入体，请在数据准备中明确一个正式地震体。`;
  }
  if (selectedPredictionSource.value?.path !== faultSegSnapshotSeismicSources.value[0]?.path) {
    return "请选择 SourceSnapshot 中唯一的三维 SEG-Y。";
  }
  return "SourceSnapshot 唯一三维 SEG-Y 已就绪。";
});
const faultSegPublicEntryAvailable = computed(() => Boolean(
  faultSegSnapshotSourceReady.value
  && (capabilities.value?.models || []).some((model) => isFaultVolumeModelId(model.id)),
));
function compactModelSpecPresentationName(model: Parameters<typeof modelSpecPresentationName>[0]): string {
  const fullName = modelSpecPresentationName(model);
  const taskName = fullName.includes("（")
    ? fullName.slice(0, fullName.indexOf("（"))
    : fullName.split("｜", 1)[0];
  return `${taskName} · ${scientificStatusLabel(model.scientific_status)}`;
}
const SURFACE_SEG_MODEL_ID = "seismic_surface_seg";
const LEGACY_HORIZON_MODEL_ID = "wellfuse_horizon_p17";
const selectedPredictionModelName = computed(() => selectedPredictionModel.value
  ? compactModelSpecPresentationName(selectedPredictionModel.value)
  : "模型");
const selectedModelDataFlow = computed(() =>
  capabilities.value?.model_data_flows?.find(
    (flow) => flow.model_id === selectedPredictionModelId.value,
  ) || null,
);
const GEOPATH_TIE_MODEL_ID = "wellfuse_align_geopath_tie_v1";
const geoPathCandidateModel = computed(() =>
  capabilities.value?.models.find((model) => model.id === GEOPATH_TIE_MODEL_ID) || null,
);
const geoPathCandidateDataFlow = computed(() =>
  capabilities.value?.model_data_flows?.find((flow) => flow.model_id === GEOPATH_TIE_MODEL_ID) || null,
);
const releaseUiRunnableModelIds = computed(() => [
  ...new Set([
    ...predictionTaskDefinitions.value.flatMap((task) => task.runnable_model_ids),
    GEOPATH_TIE_MODEL_ID,
  ]),
]);
const geoPathCandidateRelease = computed(() =>
  releaseCatalog.value?.releases.find((release) => release.model_id === GEOPATH_TIE_MODEL_ID) || null,
);
const geoPathCandidateReview = computed(() => geoPathCandidateResult.value?.candidate_review || null);
interface GeoPathCandidateWellPresentation {
  id: string;
  geometry: string;
  acceptedFraction: number | null;
  apertureEligibleFraction: number | null;
  repairStatus: string;
  repairReason: string;
  acceptanceEligible: boolean;
}
const geoPathCandidateWells = computed<GeoPathCandidateWellPresentation[]>(() => {
  const review = geoPathCandidateReview.value;
  if (!review) return [];
  const records = review.wells || [];
  const diagnostics = geoPathCandidateResult.value?.diagnostics || [];
  return review.well_ids.map((wellId) => {
    const record = records.find((item) =>
      [item.well_id, item.well_uid, item.well_name]
        .filter(Boolean)
        .some((identity) => String(identity).toLowerCase() === wellId.toLowerCase()),
    );
    const diagnostic = diagnostics.find((item) =>
      [item.well_id, item.well_uid, item.well_name]
        .filter(Boolean)
        .some((identity) => String(identity).toLowerCase() === wellId.toLowerCase()),
    );
    const acceptedFraction = record?.accepted_fraction ?? diagnostic?.accepted_fraction;
    const apertureFraction = record?.aperture_eligible_fraction ?? diagnostic?.aperture_eligible_fraction;
    return {
      id: wellId,
      geometry: String(record?.geometry ?? diagnostic?.geometry ?? "未分类"),
      acceptedFraction: typeof acceptedFraction === "number" ? acceptedFraction : null,
      apertureEligibleFraction: typeof apertureFraction === "number" ? apertureFraction : null,
      repairStatus: String(record?.repair_status ?? diagnostic?.repair_status ?? "awaiting_review_contract"),
      repairReason: String(record?.repair_reason ?? diagnostic?.repair_reason ?? "后端尚未提供逐井修复判据"),
      // Fail closed: a missing field is not equivalent to a safe candidate.
      acceptanceEligible: record?.acceptance_eligible === true,
    };
  });
});
const geoPathSelectedEligible = computed(() => {
  const eligibleIds = new Set(
    geoPathCandidateWells.value.filter((well) => well.acceptanceEligible).map((well) => well.id),
  );
  return geoPathSelectedWellIds.value.length > 0
    && geoPathSelectedWellIds.value.every((wellId) => eligibleIds.has(wellId));
});
const geoPathCandidateAlreadyAccepted = computed(() => Boolean(
  geoPathCandidateTaskId.value
  && geoPathAcceptedRegistrationResult.value?.registration?.candidate_prediction_task_id === geoPathCandidateTaskId.value,
));
const geoPathAcceptanceReady = computed(() => Boolean(
  geoPathCandidateTaskId.value
  && geoPathCandidateReview.value?.candidate_manifest_sha256
  && geoPathSelectedEligible.value
  && geoPathAcceptanceConfirmed.value
  && !geoPathCandidateAlreadyAccepted.value,
));
const modalityLabels: Record<string, string> = {
  seismic: "三维地震 SEG-Y",
  well_log: "测井曲线",
  trajectory: "完整井轨迹",
  registration: "Registration V3",
  registration_control: "可选标定控制",
  well_facies_context: "可选井旁相约束",
};
const sourceModeLabels: Record<string, string> = {
  sealed_snapshot: "封存 SourceSnapshot",
  explicit_raw: "显式原始井文件",
  registered_dataset: "登记数据集",
};
const formatModality = (value: string) => modalityLabels[value] || value.replaceAll("_", " ");
const formatSourceMode = (value: string) => sourceModeLabels[value] || value.replaceAll("_", " ");
const selectedModelRequiredModalities = computed(() =>
  selectedModelDataFlow.value?.required_modalities || [],
);
const selectedModelOptionalModalities = computed(() =>
  selectedModelDataFlow.value?.optional_modalities || [],
);
const SNAPSHOT_ONLY_DOWNSTREAM_WELL_TASK_IDS = new Set([
  "well_property",
  "fluid_interpretation",
  "facies_1d",
  "fracture_development",
]);
const isSnapshotOnlyDownstreamWellTask = computed(() =>
  SNAPSHOT_ONLY_DOWNSTREAM_WELL_TASK_IDS.has(activePredictionTask.value),
);
const selectedModelSourceModes = computed(() =>
  isSnapshotOnlyDownstreamWellTask.value
    ? ["sealed_snapshot"]
    : selectedModelDataFlow.value?.source_modes || ["sealed_snapshot"],
);
const sealedSnapshotWellCount = computed(() => {
  const result = preparationResult.value;
  const canonicalCount = Number(result?.data_snapshot?.canonical_data?.well_entities?.count || 0);
  const summaryCount = Number(result?.summary.wells || 0);
  const entityCount = result?.well_entities?.length || result?.wells.length || 0;
  return Math.max(0, canonicalCount, summaryCount, entityCount);
});
const sealedSnapshotWellLogCount = computed(() => {
  const result = preparationResult.value;
  const canonicalCount = Number(result?.data_snapshot?.canonical_data?.well_logs?.count || 0);
  const summaryCount = Number(result?.summary.log_files || 0);
  const entityCount = (result?.well_entities || result?.wells || [])
    .filter((well) => well.log_count > 0)
    .length;
  return Math.max(0, canonicalCount, summaryCount, entityCount);
});
const sealedSnapshotWellInputReady = computed(() => Boolean(
  dataSnapshotTaskId.value && sealedSnapshotWellLogCount.value > 0,
));
// A sealed seismic volume plus sealed well logs is enough to start the platform's
// default well-seismic route.  Model selection comes after this route has made a
// formal-calibration vs. spatial-QC decision, not before it.
const sealedWellSeismicWorkflowReady = computed(() => Boolean(
  dataSnapshotTaskId.value
  && predictionSources.value.length > 0
  && sealedSnapshotWellInputReady.value,
));
const formalRegistrationReady = computed(() => Boolean(
  registrationTaskId.value
  && registrationResult.value?.registration?.can_build_multimodal_view === true
  && Number(
    registrationResult.value.registration.downstream_fusion_ready_well_count
    ?? registrationResult.value.registration.fusion_ready_well_count
    ?? 0,
  ) > 0,
));
const formalRegistrationFusionBlocked = computed(() => Boolean(
  registrationTaskId.value
  && registrationResult.value?.registration
  && !formalRegistrationReady.value,
));
const preparedViewReady = computed(() => Boolean(
  sampleBuildingTaskId.value
  && sampleResult.value?.registration_task_id === registrationTaskId.value
  && sampleResult.value?.prepared_view?.source_snapshot_id === dataSnapshotTaskId.value
  && sampleResult.value?.matching
  && sampleResult.value.prepared_view?.state === "ready"
  && sampleResult.value.prepared_view.manifest_path
  && sampleResult.value.prepared_view.manifest_sha256
  && sampleResult.value.prepared_view.view_sha256,
));
const sealedSnapshotWellSourceCountLabel = computed(() => sealedSnapshotWellCount.value
  ? `${sealedSnapshotWellCount.value} 口井`
  : `${sealedSnapshotWellLogCount.value} 个测井文件`,
);
const selectedSourceMode = computed(() => "sealed_snapshot");
const snapshotWellSourceReason = computed(() => {
  if (!dataSnapshotTaskId.value) {
    return "尚未恢复封存快照；完成数据准备后，系统会自动启用快照中的井文件。";
  }
  if (!sealedSnapshotWellLogCount.value) {
    return `快照 ${dataSnapshotTaskId.value.slice(0, 8)} 没有可用测井文件；请返回数据准备，将井文件纳入当前封存快照。`;
  }
  return `快照 ${dataSnapshotTaskId.value.slice(0, 8)} 已封存 ${sealedSnapshotWellSourceCountLabel.value}；系统自动解析 LAS、井口与共享轨迹，无需逐井选择。`;
});
const selectedModelRequiresSeismic = computed(() => {
  if (selectedModelDataFlow.value) {
    return selectedModelDataFlow.value.required_modalities.includes("seismic");
  }
  return Boolean(
    selectedPredictionModel.value
    && selectedPredictionModel.value.metadata?.requires_seismic !== false,
  );
});
const usesSealedSnapshotWellInput = computed(() =>
  isSnapshotOnlyDownstreamWellTask.value,
);
const resultModelDataFlow = computed(() =>
  capabilities.value?.model_data_flows?.find(
    (flow) => flow.model_id === predictionResult.value?.model_id,
  ) || null,
);
const predictionOutputContract = computed(() => {
  const resultContract = predictionResult.value?.result_contract;
  if (typeof resultContract === "string") return resultContract;
  const outputContract = predictionResult.value?.output_contract;
  return resultContract?.id
    || resultContract?.schema_version
    || (typeof outputContract === "string" ? outputContract : outputContract?.contract_version)
    || resultModelDataFlow.value?.output_contract
    || "";
});
const isFluidInterpretationResult = computed(() =>
  predictionResult.value?.model_id === "wellfuse_fluid_interpretation_fast",
);
const predictionTargetSemantics = computed(() => {
  if (isFluidInterpretationResult.value) {
    return "连续 MD 确定流体层段；概率仅在进程内参与判别，不落盘或公开下载";
  }
  if (predictionResult.value?.units) return predictionResult.value.units;
  const outputContract = predictionResult.value?.output_contract;
  if (outputContract && typeof outputContract !== "string") {
    const semantics = String(outputContract.primary_semantics || "").toLowerCase();
    if (semantics.includes("regression") || semantics.includes("continuous")) {
      return "连续值回归";
    }
    const decisionRule = String(outputContract.primary_decision_rule || "").toLowerCase();
    if (decisionRule.includes("argmax") || semantics.includes("class")) {
      return "多类概率与 Argmax";
    }
  }
  return "分类概率";
});
const predictionOutputAxes = computed(() => {
  const resultContract = predictionResult.value?.result_contract;
  if (resultContract && typeof resultContract !== "string") {
    const axes = resultContract.output_axes || resultContract.axes;
    if (axes?.length) return axes;
  }
  if (predictionResult.value?.output_axes?.length) return predictionResult.value.output_axes;
  return predictionResult.value?.input?.axes || [];
});
const WELL_SEQUENCE_OUTPUT_CONTRACTS = new Set([
  "well-seismic.well-property-curves.v1",
  "well-seismic.fluid-classes.v1",
  "well-seismic.well-facies-sequence.v1",
  "well-seismic.fracture-development.v1",
]);
const isWellSequenceResult = computed(() => Boolean(
  predictionResult.value && (
    WELL_SEQUENCE_OUTPUT_CONTRACTS.has(predictionOutputContract.value)
    || (
      predictionOutputAxes.value.length === 1
      && ["MD", "DEPTH"].includes(String(predictionOutputAxes.value[0]).toUpperCase())
    )
  ),
));
const selectedModelRegistrationPolicy = computed(() => {
  if (selectedModelDataFlow.value) return selectedModelDataFlow.value.registration_policy;
  if (activePredictionTask.value === "horizon") return "none";
  if (selectedPredictionModel.value?.metadata?.requires_registration === true) return "required";
  return "none";
});
const selectedModelRequiresRegistration = computed(() =>
  selectedModelRegistrationPolicy.value === "required",
);
const selectedModelRequiresPreparedView = computed(() =>
  selectedModelDataFlow.value?.prepared_view_policy === "required",
);
const selectedModelPrefersPreparedView = computed(() =>
  selectedModelDataFlow.value?.prepared_view_policy === "preferred",
);
const selectedModelConsumesPreferredPreparedView = computed(() =>
  selectedModelPrefersPreparedView.value
  && selectedModelDataFlow.value?.prepared_view_consumed === true,
);
const selectedModelPreparedViewPolicyLabel = computed(() => {
  if (isFaultSegModel.value) return "不使用 PreparedView";
  return ({
    none: "不使用 PreparedView",
    optional: "PreparedView 可选",
    preferred: "优先使用 PreparedView",
    required: "PreparedView 必需",
  })[selectedModelDataFlow.value?.prepared_view_policy || "optional"];
});
const isFaultSegModel = computed(() => isFaultVolumeModelId(selectedPredictionModelId.value));
const selectedPredictionWorkflowGateReady = computed(() =>
  isFaultSegModel.value ? faultSegSnapshotSourceReady.value : preparedViewReady.value,
);
const predictionEntryReady = computed(() =>
  preparedViewReady.value || faultSegPublicEntryAvailable.value,
);
const isSurfaceSegModel = computed(() => selectedPredictionModelId.value === SURFACE_SEG_MODEL_ID);
const isHorizonModel = computed(() => selectedPredictionModelId.value === LEGACY_HORIZON_MODEL_ID);
const isF3FaciesModel = computed(() => selectedPredictionModelId.value === "wellfuse_facies_3d_f3_fast");
const isWellFuseGeobodyModel = computed(() =>
  ["wellfuse_channel_p17", "wellfuse_karst_p17"].includes(selectedPredictionModelId.value),
);
const selectedModelAdapter = computed(() =>
  capabilities.value?.model_input_adapters.find((adapter) => adapter.model_id === selectedPredictionModelId.value) || null,
);
const selectedModelCompatibility = computed(() =>
  selectedPredictionSource.value?.model_compatibility?.[selectedPredictionModelId.value] || null,
);
const faultSegFormalInputReady = computed(() => Boolean(
  isFaultSegModel.value && selectedModelCompatibility.value?.ready === true,
));
const faultSegModelInputReady = computed(() => Boolean(
  faultSegFormalInputReady.value,
));
const surfaceInlineFallbackReady = computed(() => {
  if (!isSurfaceSegModel.value || surfaceSegInlineCount.value === "") return false;
  const inlineCount = Number(surfaceSegInlineCount.value);
  const traceCount = selectedPredictionSource.value?.trace_count || 0;
  return Number.isInteger(inlineCount) && inlineCount > 0 && traceCount > inlineCount && traceCount % inlineCount === 0;
});
const surfaceResolvedHeaderFallbackReady = computed(() => {
  if (!isSurfaceSegModel.value) return false;
  const compatibility = selectedModelCompatibility.value;
  const shape = compatibility?.shape_ics || [];
  return compatibility?.geometry_mode === "nonstandard_headers"
    && Number(compatibility.duplicate_trace_count || 0) === 0
    && shape.length === 3
    && Number(shape[0]) > 1
    && Number(shape[1]) > 1
    && Number(shape[2]) > 1;
});
const predictionInputReady = computed(() => Boolean(
  usesSealedSnapshotWellInput.value
    ? selectedModelAdapter.value && sealedSnapshotWellInputReady.value
    : selectedPredictionSource.value
      && selectedModelAdapter.value
      && (
        (isFaultSegModel.value && faultSegModelInputReady.value)
        || selectedModelCompatibility.value?.ready
        || surfaceResolvedHeaderFallbackReady.value
        || surfaceInlineFallbackReady.value
      ),
));
const predictionCompatibilityReason = computed(() =>
  usesSealedSnapshotWellInput.value
    ? snapshotWellSourceReason.value
    : isFaultSegModel.value && !faultSegFormalInputReady.value
    ? "当前至少一个空间轴不足 128；单任务断层识别仅支持完整 128³ 中心块或全区重建，当前数据不可运行。"
    : surfaceResolvedHeaderFallbackReady.value
    ? "平台将复用已识别的三维 Inline/Crossline 道头，保留真实稀疏网格运行"
    : surfaceInlineFallbackReady.value
    ? `将按显式 Inline 数 ${surfaceSegInlineCount.value} 重建 ${Number(selectedPredictionSource.value?.trace_count || 0) / Number(surfaceSegInlineCount.value)} 个 Crossline`
    : publicModelText(selectedModelCompatibility.value?.reason, "等待模型与数据匹配"),
);
const preparation = computed(() => workflowResult.value?.preparation || null);
const runtimeContractReviewPending = computed(() => Boolean(
  preparationResult.value?.preparation.runtime_contract_review?.required
  && preparationResult.value.preparation.runtime_contract_review.fields.length,
));
const runtimeContractAttestationText = computed(() => {
  const source = runtimeContractDraft.value.seismic_srd_elevation_m;
  if (source === "" || source === null || source === undefined) {
    return "请先填写 SRD 高程";
  }
  const raw = Number(source);
  if (!Number.isFinite(raw) || Math.abs(raw) > 10_000) {
    return "请先填写 -10000 至 10000 m 范围内的 SRD 高程";
  }
  const normalized = Math.round(raw * 100) / 100;
  return surveyAttestationDeclaration(normalized);
});
const preparationTimeDepthPolicy = computed(() => {
  const policy = preparation.value?.task_readiness?.time_depth_policy;
  return policy && typeof policy === "object" ? policy : null;
});
const preparationRegistrationEntryPolicy = computed(() => {
  const policy = preparation.value?.task_readiness?.registration_entry_policy;
  return policy && typeof policy === "object" ? policy : null;
});
const acousticFineCalibrationCandidateCount = computed(() => {
  const acousticCount = Math.max(
    0,
    Number(preparationTimeDepthPolicy.value?.acoustic_candidate_well_count || 0),
  );
  const entryPolicy = preparationRegistrationEntryPolicy.value;
  const eligibleReceiptCount = (entryPolicy?.native_relative_well_receipts || [])
    .filter((receipt) => receipt.eligible === true)
    .length;
  const eligibleCount = eligibleReceiptCount || Math.max(
    0,
    Number(entryPolicy?.native_relative_candidate_well_count || 0),
  );
  return Math.min(acousticCount, eligibleCount);
});
const acousticFineCalibrationCandidateReady = computed(() => {
  const policy = preparationTimeDepthPolicy.value;
  const entryPolicy = preparationRegistrationEntryPolicy.value;
  return Boolean(
    policy
    && entryPolicy?.native_relative_registration_ready === true
    && policy.provided_control_required !== true
    && Number(policy.provided_control_well_count || 0) === 0
    && acousticFineCalibrationCandidateCount.value > 0
    && policy.missing_provided_control_blocks_current_task !== true
    && policy.model_forbids_time_depth_supervision !== true,
  );
});
const FORMAL_REGISTRATION_AUDIT_STAGES = new Set([
  "vertical_datum_normalization",
  "seismic_time_reference",
  "vertical_alignment",
]);
const SUPPORTED_VERTICAL_CONTRACT_FIELDS = new Set([
  "vertical_crs_id",
  "seismic_srd_elevation_m",
  "seismic_time_domain",
  "seismic_correction_state",
]);
const hasExplicitPreparationScope = computed(() => Boolean(
  preparation.value?.task_readiness?.task_id,
));
const currentScopeModelId = computed(() =>
  hasExplicitPreparationScope.value
    ? preparation.value?.task_readiness?.model_id || ""
    : "",
);
const currentScopeDataFlow = computed(() =>
  capabilities.value?.model_data_flows?.find((flow) => flow.model_id === currentScopeModelId.value)
  || null,
);
const currentScopeModelName = computed(() => {
  const model = capabilities.value?.models.find((item) => item.id === currentScopeModelId.value);
  return modelPresentationName(
    currentScopeModelId.value,
    model?.name || currentScopeModelId.value || "当前模型",
    model?.scientific_status,
  );
});
const currentScopeModelShortName = computed(() => currentScopeModelName.value);
const currentScopeRequiresRegistration = computed(() =>
  currentScopeDataFlow.value?.registration_policy === "required",
);
const currentScopeRequiresPreparedView = computed(() =>
  currentScopeDataFlow.value?.prepared_view_policy === "required",
);
const currentScopePrefersPreparedView = computed(() =>
  currentScopeDataFlow.value?.prepared_view_policy === "preferred",
);
function stageRequiredForCurrentRun(stageId: string): boolean {
  if (!hasExplicitPreparationScope.value) return false;
  const declared = preparation.value?.task_readiness?.required_stages;
  if (Array.isArray(declared)) return declared.includes(stageId);
  return preparation.value?.stages.find((stage) => stage.id === stageId)?.status !== "本任务不需要";
}

function issueRequiredForCurrentRun(issue: PreparationIssue): boolean {
  return hasExplicitPreparationScope.value && issue.required_for_task !== false;
}

const AUTO_AUDIT_STATUSES = new Set([
  "无需确认",
  "本任务不需要",
  "LLM已补全",
  "LLM已补全并复检",
  "系统已自动处理",
  "已启用转换插件",
  "已确认采用",
]);

function issueNeedsCurrentAttention(issue: PreparationIssue): boolean {
  return issueRequiredForCurrentRun(issue)
    && issue.attention_required === true
    && !AUTO_AUDIT_STATUSES.has(issue.confirmation_status);
}

const currentAttentionIssues = computed(() =>
  preparation.value?.issues.filter(issueNeedsCurrentAttention) || [],
);
const auditIssues = computed(() =>
  preparation.value?.issues.filter((issue) => !issueNeedsCurrentAttention(issue)) || [],
);

const effectiveBlockingIssues = computed(() =>
  currentAttentionIssues.value.filter((issue) => issue.blocking),
);
const deferredRegistrationIssues = computed(() =>
  auditIssues.value.filter((issue) =>
    issue.original_blocking === true
    && FORMAL_REGISTRATION_AUDIT_STAGES.has(issue.stage)
    && issue.required_for_task === false,
  ),
);
const effectiveSurveyInputRequiredCount = computed(() =>
  currentAttentionIssues.value.filter((issue) => issue.resolution_mode === "survey_input").length,
);
const effectiveBlockingCount = computed(() => effectiveBlockingIssues.value.length);
const hasDeferredRegistration = computed(() => deferredRegistrationIssues.value.length > 0);
const preparationInventoryIssueCount = computed(() => preparation.value?.issues.length || 0);
const preparationTargetTaskSpec = computed(() => {
  const targetTaskId = preparation.value?.task_readiness?.task_id;
  if (!targetTaskId) return null;
  return predictionTaskDefinitions.value.find((task) => task.id === targetTaskId) || null;
});
const preparationTargetTaskName = computed(() =>
  preparationTargetTaskSpec.value?.name
  || preparation.value?.task_readiness?.task_id
  || "通用数据准备",
);
const preparationRequiredModalitiesLabel = computed(() =>
  preparation.value?.task_readiness?.required_modalities?.join("、") || "该任务声明的输入",
);
const unusedPreparationStageCount = computed(() =>
  preparation.value?.stages.filter((stage) => stage.status === "本任务不需要").length || 0,
);
const predictionBusy = computed(() => predictionOrchestrationRunning.value || predictionRunning.value);
const predictionBusyForActiveTask = computed(() => Boolean(
  predictionBusy.value
  && runningPredictionTask.value
  && runningPredictionTask.value === activePredictionTask.value,
));
const anotherPredictionTaskRunning = computed(() => Boolean(
  predictionBusy.value
  && runningPredictionTask.value
  && runningPredictionTask.value !== activePredictionTask.value,
));
const runningPredictionTaskSpec = computed(() =>
  predictionTaskDefinitions.value.find((task) => task.id === runningPredictionTask.value) || null,
);
const predictionConnectionDetail = computed(() => {
  if (!predictionBusy.value || !runningPredictionExecutionTaskId.value) return "";
  const taskLabel = `任务 ${runningPredictionExecutionTaskId.value.slice(0, 8)}`;
  if (predictionConnectionState.value === "retrying") {
    return `${taskLabel}仍在后台运行；仅状态查询暂时重试，不代表模型断线`;
  }
  if (predictionLastHeartbeatAt.value) {
    const heartbeat = new Date(predictionLastHeartbeatAt.value).toLocaleTimeString("zh-CN", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    return `${taskLabel} · 状态连接正常 · 最近心跳 ${heartbeat}`;
  }
  return `${taskLabel} · 正在建立状态连接`;
});
const fusionInputsMutating = computed(() =>
  registrationRunning.value
  || sampleRunning.value
  || geoPathCandidateRunning.value
  || geoPathAcceptanceRunning.value,
);
const fusionWorkflowMutationRunning = computed(() =>
  fusionInputsMutating.value || predictionBusy.value,
);
const wells = computed(() => workflowResult.value?.well_entities || workflowResult.value?.wells || []);
const currentMeta = computed(() => viewMeta[activeView.value]);
const HIDDEN_REGISTRATION_PATH_GROUPS = new Set<PathGroupKey>([
  "interpretations",
  "survey",
  "timeDepth",
]);
const visiblePathGroups = computed(() =>
  groups.value.filter((group) => !HIDDEN_REGISTRATION_PATH_GROUPS.has(group.key)),
);
const registeredCount = computed(() =>
  visiblePathGroups.value.reduce(
    (total, group) => total + group.paths.filter((path) => path.trim()).length,
    0,
  ),
);
const segyExplicitHeaderByteCount = computed(() => [
  segyInlineByte.value,
  segyCrosslineByte.value,
  segyXByte.value,
  segyYByte.value,
  segyCoordinateScalarByte.value,
].filter((value) => value !== "").length);
const segySourceContractLabel = computed(() => {
  if (segyGeometryProfile.value.trim()) return `已显式选择 ${segyGeometryProfile.value.trim()}`;
  if (segyExplicitHeaderByteCount.value === 5) return "5项道头字节已显式填写";
  if (segyExplicitHeaderByteCount.value > 0) return `仅填写 ${segyExplicitHeaderByteCount.value}/5 项，将阻断几何封存`;
  return "自动识别；低置信度时将阻断";
});
const horizontalSourceContractReady = computed(() => Boolean(
  horizontalCrsId.value.trim()
  && horizontalUnit.value === "m"
  && horizontalAxisOrder.value !== "unknown"
  && coordinateReferenceVerified.value,
));
const verticalSourceContractReady = computed(() => Boolean(
  verticalCrsId.value.trim()
  && verticalCrsId.value !== "LOCAL_MSL_UNSPECIFIED"
  && seismicSrdElevation.value !== ""
  && Number.isFinite(Number(seismicSrdElevation.value)),
));
const timeSourceContractReady = computed(() => (
  seismicTimeDomain.value === "TWT"
  && seismicCorrectionState.value === "corrected_to_srd"
));
const preparationElapsedSeconds = computed(() => {
  if (preparationStartedAt.value === null) return 0;
  return Math.max(0, Math.floor((preparationClockNow.value - preparationStartedAt.value) / 1000));
});
const preparationEtaSeconds = computed(() => {
  if (
    !["reading", "hashing"].includes(preparationActivityPhase.value)
    || preparationEtaBaselineSeconds.value === null
    || preparationEtaComputedAt.value === null
  ) return null;
  const elapsedSinceEstimate = Math.max(
    0,
    (preparationClockNow.value - preparationEtaComputedAt.value) / 1000,
  );
  const remaining = preparationEtaBaselineSeconds.value - elapsedSinceEstimate;
  return remaining >= 5 ? Math.ceil(remaining) : null;
});
const preparationHistoricalRemainingSeconds = computed(() => {
  if (preparationHistoricalEstimateSeconds.value === null) return null;
  const remaining = preparationHistoricalEstimateSeconds.value - preparationElapsedSeconds.value;
  return remaining >= 5 ? Math.ceil(remaining) : null;
});
const fusionElapsedSeconds = computed(() => {
  if (fusionStartedAt.value === null) return 0;
  return Math.max(0, Math.floor((fusionClockNow.value - fusionStartedAt.value) / 1000));
});
const fusionEtaSeconds = computed(() => countDownRemainingSeconds(
  fusionEtaBaselineSeconds.value,
  fusionEtaComputedAt.value === null
    ? 0
    : (fusionClockNow.value - fusionEtaComputedAt.value) / 1000,
));
const fusionActivityTitle = computed(() => {
  if (fusionActivityPhase.value === "reconnecting") return "正在恢复融合任务进度";
  if (fusionActivityPhase.value === "prepared_view") return "正在构建井震融合视图";
  return "正在进行井震精细标定";
});
const fusionEstimateLabel = computed(() => {
  if (fusionEtaSeconds.value !== null) {
    return `按当前阶段进度粗估剩余约 ${formatDuration(fusionEtaSeconds.value)}；完整标定与融合可能需要 10 分钟以上`;
  }
  if (progress.value >= 100) return "当前阶段正在收尾；完成后平台会核验融合视图是否真正就绪";
  return "正在采集速度与进度，随后显示预计剩余时间；完整标定与融合可能需要 10 分钟以上";
});
const preparationActivityTitle = computed(() => ({
  reading: "正在读取数据",
  hashing: "正在校验文件身份",
  summarizing: "正在汇总与校验",
  caching: "正在保存复用缓存",
  autofill: "正在受控补全",
  reconnecting: "正在恢复进度连接",
  completed: "数据读取已完成",
  failed: "数据读取未完成",
} as Partial<Record<PreparationActivityPhase, string>>)[preparationActivityPhase.value] || "正在准备数据");
const preparationCurrentItemName = computed(() => {
  const normalized = preparationCurrentItem.value.replaceAll("\\", "/");
  return normalized.split("/").filter(Boolean).at(-1) || preparationCurrentItem.value;
});
const preparationCurrentItemAction = computed(() => (
  preparationActivityPhase.value === "hashing" ? "当前校验" : "当前读取"
));
const preparationEstimateLabel = computed(() => {
  if (preparationActivityPhase.value === "reading") {
    if (!preparationWorkTotal.value) return "正在确认待读取文件数量";
    if (preparationWorkDone.value >= preparationWorkTotal.value) return "文件读取完成，正在进入汇总";
    if (preparationSubworkUnit.value === "traces" && preparationEtaSeconds.value !== null) {
      return `当前 SEG-Y 文件预计剩余约 ${formatDuration(preparationEtaSeconds.value)}；之后还会处理其余资产并校验文件身份`;
    }
    if (
      preparationHistoricalEstimateConfidence.value === "medium"
      && preparationHistoricalRemainingSeconds.value !== null
    ) {
      return `同配置历史总耗时剩余约 ${formatDuration(preparationHistoricalRemainingSeconds.value)}（${preparationHistoricalEstimateSamples.value} 次中位数，仅作粗略参考）`;
    }
    if (
      preparationHistoricalEstimateConfidence.value === "medium"
      &&
      preparationHistoricalEstimateSeconds.value !== null
      && preparationElapsedSeconds.value >= preparationHistoricalEstimateSeconds.value
    ) return "已超过历史任务的常见耗时，当前大文件仍在读取";
    if (preparationEtaBaselineSeconds.value !== null) return "当前文件仍在读取，正在重新估算";
    if (preparationSubworkTotal.value) return "扫描首批 SEG-Y 道头后开始估算剩余时间";
    return "普通文件读取较快；遇到 SEG-Y 后会按真实道头吞吐显示当前文件预计时间";
  }
  if (preparationActivityPhase.value === "hashing") {
    if (preparationSubworkUnit.value === "bytes" && preparationEtaSeconds.value !== null) {
      return `完整文件身份校验预计剩余约 ${formatDuration(preparationEtaSeconds.value)} · 按当前磁盘吞吐估算`;
    }
    if (
      preparationHistoricalEstimateConfidence.value === "medium"
      &&
      preparationHistoricalEstimateSeconds.value !== null
      && preparationElapsedSeconds.value >= preparationHistoricalEstimateSeconds.value
    ) return "已超过历史任务的常见耗时，仍在校验大文件身份";
    return preparationSubworkDone.value
      ? "当前磁盘吞吐发生波动，正在重新估算"
      : "读取首个校验块后开始估算剩余时间";
  }
  if (preparationActivityPhase.value === "summarizing") {
    return useLlmFallback.value && capabilities.value?.llm.available
      ? "读取与校验已完成，正在生成质量报告和 Kimi 受控建议；网络阶段暂不显示固定倒计时"
      : "读取与校验已完成，正在生成质量报告、预览并封存快照";
  }
  if (preparationActivityPhase.value === "caching") return "数据已封存，正在保存可跨进程复用的 SEG-Y 几何缓存";
  if (preparationActivityPhase.value === "autofill") return "数据读取已完成，正在执行 Kimi 受控修复";
  if (preparationActivityPhase.value === "reconnecting") return "正在恢复任务状态连接，后台读取不会被取消";
  if (
    preparationHistoricalEstimateConfidence.value === "medium"
    && preparationHistoricalEstimateSeconds.value !== null
  ) {
    return `同配置历史总耗时约 ${formatDuration(preparationHistoricalEstimateSeconds.value)}（${preparationHistoricalEstimateSamples.value} 次中位数，仅作粗略参考）`;
  }
  return "预计时间将在开始读取文件后显示";
});
const geoPathGeometryFormReady = computed(() => Boolean(
  segyGeometryProfile.value.trim() && segyExplicitHeaderByteCount.value === 5,
));
const geoPathSnapshotGeometryReady = computed(() => {
  const contract = sealedSnapshotSegyContract.value;
  if (!contract?.segy_geometry_profile?.trim()) return false;
  return [
    contract.segy_inline_byte,
    contract.segy_crossline_byte,
    contract.segy_x_byte,
    contract.segy_y_byte,
    contract.segy_coordinate_scalar_byte,
  ].every((value) => typeof value === "number" && Number.isFinite(value));
});
const geoPathSnapshotGeometryEvidenceLabel = computed(() => (
  sealedSnapshotSegyGeometryAuthority.value === "verified_automatic"
    ? "当前 SourceSnapshot 的高置信自动几何收据已通过资产 SHA、解析选项和几何指纹绑定；运行时仍会复检后再消费。"
    : "当前 SourceSnapshot 已显式封存 Profile 与五个道头字节。候选完成后仍需逐井审核与显式确认。"
));
const filteredIssues = computed(() => {
  const issues = currentAttentionIssues.value;
  const stageIssues = issueFilter.value !== "全部"
    ? issues.filter((item) => item.stage === issueFilter.value)
    : issues;
  return stageIssues;
});
const selectedStage = computed(() =>
  preparation.value?.stages.find((stage) => stage.id === issueFilter.value) || null,
);
const pendingConfirmationCount = computed(() =>
  currentAttentionIssues.value.filter((issue) =>
    issue.confirmation_status === "待人工确认" || issue.confirmation_status === "需一次集中补充",
  ).length || 0,
);
const autofillEligibleCount = computed(() =>
  currentAttentionIssues.value.filter((issue) =>
    issue.confirmation_status === "待人工确认" && !issue.blocking,
  ).length || 0,
);
const seismicVerticalContractIssues = computed(() =>
  (preparation.value?.issues || []).filter((issue) =>
    issue.confirmation_group === "seismic_vertical_contract"
    || ["vertical_datum_normalization", "seismic_time_reference"].includes(issue.stage),
  ),
);
const seismicVerticalContractCandidates = computed<SourceContractCandidate[]>(() => {
  const byField = new Map<string, SourceContractCandidate>();
  const rank = (candidate: SourceContractCandidate) => {
    const statusScore = ({ verified: 4, candidate: 3, review_required: 3, insufficient: 1, conflict: 0 } as Record<string, number>)[candidate.status || ""] ?? 2;
    return statusScore * 10 + (candidate.auto_applied ? 2 : 0) + Number(candidate.confidence || 0);
  };
  for (const issue of seismicVerticalContractIssues.value) {
    for (const candidate of issue.contract_candidates || []) {
      if (!candidate?.field) continue;
      const existing = byField.get(candidate.field);
      if (!existing || rank(candidate) > rank(existing)) byField.set(candidate.field, candidate);
    }
  }
  for (const candidate of preparation.value?.survey_contract_candidate?.candidates || []) {
    if (!candidate?.field) continue;
    const existing = byField.get(candidate.field);
    if (!existing || rank(candidate) > rank(existing)) byField.set(candidate.field, candidate);
  }
  for (const [field, value] of Object.entries(preparation.value?.request_patch || {})) {
    if (byField.has(field) || value === undefined) continue;
    byField.set(field, {
      field,
      value,
      status: "verified",
      evidence: "结构化工区合同候选",
      requires_human_confirmation: preparation.value?.survey_contract_candidate?.confirmation_required === true,
      auto_applied: false,
    });
  }
  return [...byField.values()];
});
const adoptableSeismicVerticalCandidates = computed(() =>
  seismicVerticalContractCandidates.value.filter((candidate) => contractCandidateCanPopulate(candidate)),
);
const correctedToSrdContractCandidate = computed(() => {
  const patchValue = preparation.value?.request_patch?.seismic_correction_state;
  if (patchValue !== "corrected_to_srd") return null;
  return adoptableSeismicVerticalCandidates.value.find((candidate) =>
    candidate.field === "seismic_correction_state"
    && candidate.value === "corrected_to_srd",
  ) || null;
});
const pendingSeismicVerticalCandidates = computed(() =>
  adoptableSeismicVerticalCandidates.value.filter((candidate) =>
    !autoPopulatedSurveyContractFields.value.includes(candidate.field)
    && contractCandidateNeedsHumanConfirmation(candidate),
  ),
);
const pendingSeismicVerticalCandidateCount = computed(() =>
  pendingSeismicVerticalCandidates.value.length,
);
const unresolvedSurveyContractFields = computed(() => {
  const populated = new Set(autoPopulatedSurveyContractFields.value);
  return (preparation.value?.survey_contract_candidate?.unresolved_fields || []).filter((field) => {
    if (populated.has(field)) return false;
    if (field === "seismic_correction_state" && surveyAttestationMatchesDraft()) return false;
    return true;
  });
});
const evidenceReviewRequiresAction = computed(() => Boolean(
  effectiveBlockingCount.value
  || pendingConfirmationCount.value
  || pendingSeismicVerticalCandidateCount.value,
));
const evidenceReviewStatusLabel = computed(() => {
  if (effectiveBlockingCount.value) return `${effectiveBlockingCount.value} 项当前阻断需处理`;
  const pendingCount = Math.max(
    pendingConfirmationCount.value,
    pendingSeismicVerticalCandidateCount.value,
  );
  if (pendingCount) return `${pendingCount} 项证据待确认`;
  return "当前无需处理";
});
// Registration carries the newest well/time alignment, while preparation is
// the durable authority for source LAS previews.  Keeping these sources
// separate lets the log-only workbench remain usable even when a seismic
// preview is absent or a derived result omits its source assets.
const wellGeometrySummary = computed(() =>
  registrationResult.value?.visualization_preview?.wellGeometrySummary
  || sampleResult.value?.visualization_preview?.wellGeometrySummary
  || preparationResult.value?.visualization_preview?.wellGeometrySummary
  || workflowResult.value?.visualization_preview?.wellGeometrySummary
  || null,
);
const wellTimeAlignmentSummary = computed(() =>
  registrationResult.value?.visualization_preview?.wellTimeAlignmentSummary
  || sampleResult.value?.visualization_preview?.wellTimeAlignmentSummary
  || preparationResult.value?.visualization_preview?.wellTimeAlignmentSummary
  || workflowResult.value?.visualization_preview?.wellTimeAlignmentSummary
  || null,
);
const previewTimeAxisLabel = computed(() => {
  const summary = wellTimeAlignmentSummary.value;
  if (summary?.provenTwt === true) return "可审计 TWT";
  const domain = String(summary?.timeAxisDomain || "").toUpperCase();
  if (domain === "TWT") return "TWT 候选（基准未核验）";
  if (domain === "OWT") return "OWT 时间候选";
  if (domain === "UNAVAILABLE") return "未解析时间轴";
  if (domain === "UNKNOWN_TIME" || domain === "MIXED") return "原生 SEG-Y 时间候选";
  return registrationTimeAxisPresentation.value.provenTwt
    ? "可审计 TWT"
    : "井震时间候选";
});
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
        name: publicModelText(`${predictionResult.value?.task_name || activePredictionTaskSpec.value?.name || "模型"}预测叠加`),
        detail: publicModelText(selectedPredictionSource.value?.name, "预测体与背景地震同轴显示"),
      }]
    : [];
  return [
    ...predictionAssets,
    ...previewVolumes.value.map((volume, index) => ({
      index: index + predictionOffset,
      kind: "三维体" as const,
      name: publicModelText(volume.name),
      detail: publicModelText(volume.path.replaceAll("\\", "/").split("/").at(-1) || volume.path),
    })),
    ...seismicLinePreviews.value.map((line, index) => ({
      index: previewVolumes.value.length + predictionOffset + index,
      kind: "二维测线" as const,
      name: publicModelText(line.name),
      detail: publicModelText(line.path.replaceAll("\\", "/").split("/").at(-1) || line.path),
    })),
  ];
});
const selectedSeismicAsset = computed(() =>
  seismicWorkbenchAssets.value.find((asset) => asset.index === selectedSeismicAssetIndex.value)
  || seismicWorkbenchAssets.value[0]
  || null,
);
const wellLogPreviews = computed(() => {
  const candidates = [
    preparationResult.value?.visualization_preview?.wellLogs,
    sampleResult.value?.visualization_preview?.wellLogs,
    registrationResult.value?.visualization_preview?.wellLogs,
    workflowResult.value?.visualization_preview?.wellLogs,
  ];
  return candidates.find((logs) => logs?.length) || [];
});
const sourceDataStatistics = computed(() => buildSourceDataStatistics(
  preparationResult.value || workflowResult.value,
));
const primarySeismicStatistics = computed(() => sourceDataStatistics.value.seismic.primaryVolume);
const seismicGridCoveragePercent = computed(() => {
  const coverage = primarySeismicStatistics.value?.gridCoverage;
  if (coverage === null || coverage === undefined || !Number.isFinite(coverage)) return null;
  return Math.min(100, Math.max(0, coverage * 100));
});
const wellGeometrySegments = computed(() => {
  const distribution = sourceDataStatistics.value.wells.wellTypeDistribution;
  return [
    { id: "vertical", label: "直井", value: distribution.vertical },
    { id: "deviated", label: "斜井", value: distribution.deviated },
    { id: "horizontal", label: "水平井", value: distribution.horizontal },
    { id: "unknown", label: "未分类", value: distribution.unknown },
  ].filter((item) => item.value > 0);
});
const wellGeometryTotal = computed(() => wellGeometrySegments.value.reduce(
  (total, item) => total + item.value,
  0,
));
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
  ? `${import.meta.env.DEV ? "http://127.0.0.1:725" : ""}/统一数据可视化?task_id=${encodeURIComponent(activeVisualizationTaskId.value)}`
  : "");
const visualizationUrl = computed(() => visualizationEndpoint.value
  ? `${visualizationEndpoint.value}&asset=${selectedSeismicAssetIndex.value}&embed=1`
  : "");
const visualizationStandaloneUrl = computed(() => visualizationEndpoint.value
  ? `${visualizationEndpoint.value}&asset=${selectedSeismicAssetIndex.value}&embed=0`
  : "");
const layerPulseResultForCurrentSnapshot = computed(() => (
  layerPulseResult.value
  && layerPulseSourceSnapshotId.value
  && layerPulseSourceSnapshotId.value === dataSnapshotTaskId.value
    ? layerPulseResult.value
    : null
));
const layerPulseTaskStateForCurrentSnapshot = computed<LayerPulseTaskState>(() => (
  !layerPulseSourceSnapshotId.value
  || layerPulseSourceSnapshotId.value === dataSnapshotTaskId.value
    ? layerPulseTaskState.value
    : createIdleLayerPulseTaskState()
));
const layerPulseTaskCatalog = computed(() => (
  Array.isArray(layerPulseResultForCurrentSnapshot.value?.task_catalog)
    ? layerPulseResultForCurrentSnapshot.value.task_catalog
    : []
));
const layerPulseAvailableOutputKeys = computed(() => {
  const outputs = layerPulseResultForCurrentSnapshot.value?.outputs || {};
  return layerPulseTaskCatalog.value.flatMap((entry) => {
    const artifactKey = typeof entry.artifact_key === "string" ? entry.artifact_key : "";
    return entry.output_key && artifactKey && outputs[artifactKey] && entry.finite !== false
      ? [entry.output_key]
      : [];
  });
});
const layerPulseOutputDownloads = computed<Record<string, LayerPulseOutputDownloads>>(() => {
  const prediction = layerPulseResultForCurrentSnapshot.value;
  const executionTaskId = layerPulseExecutionTaskId.value;
  if (!prediction || !executionTaskId) return {};

  const availableOutputKeys = new Set(layerPulseAvailableOutputKeys.value);
  const artifacts = prediction.standard_result_bundle?.downloads.artifacts || [];
  const artifactsByOutputKey = new Map<string, StandardResultArtifact[]>();
  for (const artifact of artifacts) {
    const existing = artifactsByOutputKey.get(artifact.output_key) || [];
    existing.push(artifact);
    artifactsByOutputKey.set(artifact.output_key, existing);
  }
  const artifactForKey = (artifactKey: string): StandardResultArtifact | undefined => (
    (artifactsByOutputKey.get(artifactKey) || [])[0]
  );
  const artifactFormat = (artifact: StandardResultArtifact): string => (
    String(artifact.format || artifact.filename.split(".").pop() || "").trim().toLowerCase()
  );
  const isSegy = (artifact: StandardResultArtifact): boolean => (
    artifactFormat(artifact) === "sgy" || artifactFormat(artifact) === "segy"
  );
  const isCsv = (artifact: StandardResultArtifact): boolean => artifactFormat(artifact) === "csv";
  const publicArtifactLink = (artifact: StandardResultArtifact): LayerPulseDownloadLink => ({
    url: backendPublicUrl(artifact.download_url),
    filename: artifact.filename,
  });
  const taskBaseUrl = `/api/v1/tasks/${encodeURIComponent(executionTaskId)}`;
  const downloads: Record<string, LayerPulseOutputDownloads> = {};

  for (const entry of layerPulseTaskCatalog.value) {
    const outputKey = String(entry.output_key || "").trim();
    if (!outputKey || !availableOutputKeys.has(outputKey)) continue;

    const primaryKey = String(entry.primary_download_artifact_key || "").trim();
    const declarations = Array.isArray(entry.download_artifact_keys)
      ? entry.download_artifact_keys
      : [];
    const declaredSegyKeys = declarations
      .filter((item) => ["sgy", "segy"].includes(String(item?.format || "").trim().toLowerCase()))
      .map((item) => String(item?.artifact_key || "").trim())
      .filter(Boolean);
    const primaryArtifact = primaryKey ? artifactForKey(primaryKey) : undefined;
    const segyArtifact = primaryArtifact && isSegy(primaryArtifact)
      ? primaryArtifact
      : declaredSegyKeys
        .map(artifactForKey)
        .find((artifact): artifact is StandardResultArtifact => Boolean(artifact && isSegy(artifact)));

    const rawArtifactKey = String(entry.artifact_key || "").trim();
    const rawArtifact = rawArtifactKey ? artifactForKey(rawArtifactKey) : undefined;
    const classLegendKey = String(entry.class_legend_artifact_key || "").trim();
    const declaredCsvKeys = declarations
      .filter((item) => String(item?.format || "").trim().toLowerCase() === "csv")
      .map((item) => String(item?.artifact_key || "").trim())
      .filter(Boolean);
    const classLegendArtifact = entry.kind === "classification"
      ? [classLegendKey, ...declaredCsvKeys]
        .filter(Boolean)
        .map(artifactForKey)
        .find((artifact): artifact is StandardResultArtifact => Boolean(artifact && isCsv(artifact)))
      : undefined;
    const encodedOutputKey = encodeURIComponent(outputKey);

    downloads[outputKey] = {
      segy: segyArtifact
        ? publicArtifactLink(segyArtifact)
        : {
          url: backendPublicUrl(`${taskBaseUrl}/layerpulse-exports/${encodedOutputKey}.sgy`),
          filename: `${outputKey}.sgy`,
        },
      rawNpy: rawArtifact
        ? publicArtifactLink(rawArtifact)
        : {
          url: backendPublicUrl(`${taskBaseUrl}/artifacts/${encodeURIComponent(rawArtifactKey)}`),
          filename: `${outputKey}.npy`,
        },
      ...(entry.kind === "classification"
        ? {
          classLegendCsv: classLegendArtifact
            ? publicArtifactLink(classLegendArtifact)
            : {
              url: backendPublicUrl(`${taskBaseUrl}/layerpulse-exports/${encodedOutputKey}.csv`),
              filename: `${outputKey}_classes.csv`,
            },
        }
        : {}),
    };
  }
  return downloads;
});
const layerPulseBaseVisualizationUrl = computed(() => {
  // Keep LayerPulse on the same interactive viewer as the platform's
  // original tasks.  The child-generated PNG remains a downloadable audit
  // artifact, never the primary workbench after inference completes.
  return visualizationUrl.value;
});
const layerPulseResultVisualizationUrl = computed(() => {
  const result = layerPulseResultForCurrentSnapshot.value;
  if (
    !result
    || !layerPulseExecutionTaskId.value
    || !layerPulseAvailableOutputKeys.value.includes(layerPulseSelectedOutputKey.value)
  ) return "";
  const registeredPath = result.standard_result_bundle?.visualization.platform_viewer_url;
  const basePath = registeredPath
    || `/统一数据可视化?task_id=${encodeURIComponent(layerPulseExecutionTaskId.value)}`;
  const separator = basePath.includes("?") ? "&" : "?";
  return visualizationUrlWithEmbed(
    `${basePath}${separator}layerpulse_output=${encodeURIComponent(layerPulseSelectedOutputKey.value)}`,
    1,
  );
});
const layerPulseStandaloneResultUrl = computed(() => {
  if (!layerPulseResultVisualizationUrl.value) return "";
  return visualizationUrlWithEmbed(layerPulseResultVisualizationUrl.value, 0);
});
const predictionVisualizationUrl = computed(() => predictionTaskId.value
  ? `${import.meta.env.DEV ? "http://127.0.0.1:725" : ""}/统一数据可视化?task_id=${encodeURIComponent(predictionTaskId.value)}&embed=1`
  : "");
const isFaultSegResult = computed(() => isFaultVolumeModelId(predictionResult.value?.model_id));
const predictionStandardBundle = computed(() => predictionResult.value?.standard_result_bundle || null);
const predictionEmbeddedVisualizationUrl = computed(() => {
  if (!predictionTaskId.value || predictionStandardBundle.value?.visualization.available !== true) return "";
  // Well-side 1-D results stay in the current workbench and open the seismic
  // volume asset so prediction tracks remain attached to their real wells.
  if (isWellSequenceResult.value) return `${predictionVisualizationUrl.value}&asset=0`;
  const platformViewerUrl = predictionStandardBundle.value.visualization.platform_viewer_url;
  if (platformViewerUrl) return visualizationUrlWithEmbed(platformViewerUrl, 1);
  const entryUrl = predictionStandardBundle.value.visualization.entry_url;
  return backendPublicUrl(
    entryUrl || `/api/v1/tasks/${encodeURIComponent(predictionTaskId.value)}/standard-results/visualization`,
  );
});
const predictionStandardVisualizationUrl = computed(() => {
  if (!predictionTaskId.value) return "";
  const platformViewerUrl = predictionStandardBundle.value?.visualization.platform_viewer_url;
  if (platformViewerUrl) return visualizationUrlWithEmbed(platformViewerUrl, 0);
  const path = predictionStandardBundle.value?.visualization.entry_url
    || `/api/v1/tasks/${encodeURIComponent(predictionTaskId.value)}/standard-results/visualization`;
  return backendPublicUrl(path);
});
const predictionWorkbenchUrl = computed(() => {
  const resultUrl = predictionResultRenderable.value
    ? predictionEmbeddedVisualizationUrl.value || predictionVisualizationUrl.value
    : "";
  if (predictionCanvasMode.value === "base") return visualizationUrl.value;
  if (predictionCanvasMode.value === "result") return resultUrl || visualizationUrl.value;
  return predictionResult.value && predictionTaskId.value && resultUrl ? resultUrl : visualizationUrl.value;
});
const predictionWorkbenchShowingResult = computed(() =>
  predictionResultRenderable.value
  && predictionCanvasMode.value !== "base"
  && Boolean(predictionResult.value && predictionTaskId.value),
);
const predictionWorkbenchStandaloneUrl = computed(() => {
  if (predictionCanvasMode.value !== "base" && predictionResultRenderable.value) {
    return predictionStandardVisualizationUrl.value || visualizationUrlWithEmbed(predictionVisualizationUrl.value, 0);
  }
  return visualizationStandaloneUrl.value;
});
const predictionWorkbenchPhase = computed(() => {
  if (registrationRunning.value) return "井震标定";
  if (sampleRunning.value) return "融合视图";
  if (predictionBusyForActiveTask.value) return "模型推理";
  if (predictionResult.value && predictionTaskId.value) {
    if (isFaultSegResult.value && !isCurrentFaultPrediction(predictionResult.value)) {
      return "旧抽样待重跑";
    }
    if (isCenterBlockFaultPrediction(predictionResult.value)) return "中心单块结果就绪";
    if (
      isWellSequenceResult.value
      && predictionStandardBundle.value?.visualization.available !== true
      && !predictionDisplayAccepted.value
      && !predictionCandidateRenderable.value
    ) return "历史结果待更新";
    return "结果就绪";
  }
  return predictionWorkbenchUrl.value ? "数据预览" : "等待数据";
});
const selectedModelHasCompletedResult = computed(() => {
  const current = predictionResult.value;
  const historical = predictionHistoryByTask.value[activePredictionTask.value]?.result.prediction;
  return Boolean(
    (current?.task_id === activePredictionTask.value
      && current.model_id === selectedPredictionModelId.value)
    || (historical?.task_id === activePredictionTask.value
      && historical.model_id === selectedPredictionModelId.value),
  );
});
const predictionRunButtonLabel = computed(() => {
  if (predictionBusy.value) return `${predictionWorkbenchPhase.value}…`;
  const verb = selectedModelHasCompletedResult.value ? "重新运行" : "运行";
  return isFaultSegModel.value
    ? faultSegScope.value === "full_volume"
      ? `${verb}全区断层识别`
      : `${verb}中心单块预测`
    : `${verb}${selectedPredictionModelName.value}`;
});
const predictionStandardManifestUrl = computed(() => {
  if (!predictionTaskId.value) return "";
  const path = predictionStandardBundle.value?.downloads.manifest_url
    || `/api/v1/tasks/${encodeURIComponent(predictionTaskId.value)}/standard-results/manifest`;
  return backendPublicUrl(path);
});
const predictionStandardDownloads = computed(() => predictionStandardBundle.value?.downloads.artifacts || []);
const predictionPrimaryStandardDownloads = computed(() => predictionStandardDownloads.value);
const predictionStandardVisualizationAssetCount = computed(() =>
  predictionStandardBundle.value?.visualization.assets.length || 0,
);
const faultResultIsFullSurvey = computed(() => {
  return isFullVolumeFaultPrediction(predictionResult.value);
});
const faultResultIsCenterBlock = computed(() =>
  isCenterBlockFaultPrediction(predictionResult.value),
);
const faultResultIsRepresentative128 = computed(() =>
  isRepresentativeGrid128FaultPrediction(predictionResult.value),
);
const faultResultNeedsSupportedScopeRerun = computed(() =>
  isFaultSegResult.value && !isCurrentFaultPrediction(predictionResult.value),
);
const predictionSpatialScopeLabel = computed(() => {
  const receipt = predictionResult.value?.standard_spatial_export;
  if (!receipt) return "";
  if (receipt.is_full_survey === true || receipt.scope === "full_survey") {
    return "全工区完整空间成果";
  }
  if (isFaultSegResult.value) return "未形成全工区完整成果";
  if (receipt.is_complete_for_declared_roi === true || receipt.coverage === "complete_for_declared_roi") {
    return "声明范围内成果（非全工区）";
  }
  return "有界预览或非完整空间成果";
});
const predictionSpatialScopeDetail = computed(() => {
  const receipt = predictionResult.value?.standard_spatial_export;
  if (!receipt) return "";
  const roi = receipt.roi;
  const roiShape = roi?.roi_shape || receipt.shape || [];
  const sourceShape = roi?.source_shape || [];
  const roiStart = roi?.roi_start || [];
  const shapeLabel = roiShape.length ? roiShape.join(" × ") : "范围未声明";
  if (receipt.is_full_survey === true || receipt.scope === "full_survey") {
    return `${shapeLabel}；覆盖整个封存地震网格`;
  }
  if (isFaultSegResult.value) {
    const sourceLabel = sourceShape.length ? ` / 源网格 ${sourceShape.join(" × ")}` : "";
    if (faultResultIsCenterBlock.value) {
      const startLabel = roiStart.length ? `，起点 ${roiStart.join(" / ")}` : "";
      return `${shapeLabel}${sourceLabel}${startLabel}；完整覆盖工区中心单块，但不代表全工区连续成果`;
    }
    return `${shapeLabel}${sourceLabel}；该历史非完整产物不得标记为完整范围或全工区成果`;
  }
  if (sourceShape.length) {
    const startLabel = roiStart.length ? `，起点 ${roiStart.join(" / ")}` : "";
    return `${shapeLabel} / 源网格 ${sourceShape.join(" × ")}${startLabel}；不得作为全工区结果申报`;
  }
  return `${shapeLabel}；以标准 Manifest 的 ROI 与重建合同为准`;
});
const faultResultScopeLabel = computed(() => {
  if (faultResultIsFullSurvey.value) return "全工区连续概率体";
  if (faultResultIsCenterBlock.value) return "工区三轴中心单个 128³ 块";
  if (faultResultIsRepresentative128.value) return "历史 128 个代表块 · 只读";
  return "历史非完整结果 · 不作为完成成果";
});
const predictionOverviewUrl = computed(() =>
  predictionTaskId.value && predictionResult.value?.outputs.overview
    ? `${import.meta.env.DEV ? "http://127.0.0.1:725" : ""}/api/v1/tasks/${encodeURIComponent(predictionTaskId.value)}/artifacts/overview`
    : "",
);
const predictionInputShape = computed(() =>
  predictionResult.value?.facies?.shape_t_inline_xline
  || predictionResult.value?.input.shape_zyx
  || predictionResult.value?.input.shape_ics
  || predictionResult.value?.segmentation?.shape_ics
  || [],
);
const predictionInputAxes = computed(() => predictionResult.value?.input.axes || []);
const predictionDisplayAcceptance = computed(() => predictionResult.value?.display_acceptance_decision || null);
const predictionDisplayAccepted = computed(() => predictionDisplayAcceptance.value?.display_status === "accepted");
const predictionSourceIdentity = computed(() =>
  predictionResult.value?.provenance?.prediction_source_identity || null,
);
const predictionSourceLabel = computed(() => {
  const identity = predictionSourceIdentity.value;
  if (!identity) return predictionSourceTaskId.value ? "数据准备快照（消费来源未证明）" : "历史结果未登记数据来源";
  if (identity.kind === "sealed_snapshot_wells") return "数据准备快照 · 井资产";
  if (identity.kind === "seismic_file") return "数据准备快照 · SEG-Y";
  if (identity.kind === "raw_well_files") return "显式井文件";
  if (identity.kind === "registered_well_dataset") return "平台登记数据集";
  return identity.kind || "未登记来源";
});
const predictionSourceDetail = computed(() => {
  const identity = predictionSourceIdentity.value;
  if (!identity) return predictionSourceTaskId.value || "无来源证明";
  if (identity.dataset) {
    const wells = identity.well_ids?.length ? ` · ${identity.well_ids.join(", ")}` : "";
    return `${identity.dataset}${wells}`;
  }
  if (identity.raw_well_paths?.length) return `${identity.raw_well_paths.length} 个井文件`;
  if (identity.raw_well_root) return identity.raw_well_root;
  if (identity.path) return identity.path.split(/[\\/]/).pop() || identity.path;
  return identity.source_snapshot_id || identity.source_snapshot_fingerprint?.slice(0, 12) || "来源已登记";
});
const predictionRegistrationStatus = computed<RegistrationConsumptionStatus | null>(() => {
  if (!predictionResult.value) return null;
  return predictionResult.value.registration_consumption?.status
    || predictionResult.value.registration_usage
    || predictionResult.value.provenance?.registration_usage
    || (registrationTaskId.value ? "unattested" : "not_requested");
});
const registrationConsumptionLabels: Record<RegistrationConsumptionStatus, string> = {
  not_requested: "未请求标定",
  available_not_used: "标定可用，模型未消费",
  lineage_only: "仅用于血缘，不作数值输入",
  used: "已核验模型实际消费",
  unattested: "未获得可验证消费证明",
};
const predictionRegistrationLabel = computed(() =>
  predictionRegistrationStatus.value
    ? registrationConsumptionLabels[predictionRegistrationStatus.value]
    : "尚未运行",
);
const predictionRegistrationDetail = computed(() => {
  const decision = predictionResult.value?.registration_consumption;
  if (!decision) return registrationTaskId.value ? "历史运行没有消费收据" : "本次模型输入未请求标定产品";
  if (decision.status === "used") {
    const rows = Number(decision.evidence?.joined_row_count || 0);
    return rows > 0 ? `消费收据已核验 · ${rows.toLocaleString()} 个联结点` : "消费收据已核验";
  }
  if (decision.status === "available_not_used") return "标定产品已提供给运行器，但没有进入模型特征";
  if (decision.status === "lineage_only") return "保留标定谱系，仅作来源追踪或候选控制";
  if (decision.status === "unattested") return decision.issues.join("；") || "运行器未提交完整消费证据";
  return "本次模型输入未请求标定产品";
});
const predictionPreparedViewStatus = computed(() =>
  predictionResult.value?.prepared_view_consumption?.status
  || predictionResult.value?.provenance?.prepared_view_usage
  || "not_requested",
);
const predictionCandidateRenderable = computed(() =>
  predictionResult.value?.candidate_visualization_decision?.renderable === true,
);
const predictionResultRenderable = computed(() =>
  Boolean(
    predictionStandardBundle.value?.visualization.available === true
    || predictionDisplayAccepted.value
    || predictionCandidateRenderable.value,
  ),
);
const wellSequenceLinkedViewerUnavailable = computed(() => Boolean(
  predictionResult.value
  && predictionTaskId.value
  && isWellSequenceResult.value
  && !predictionResultRenderable.value,
));
const isLegacyHorizonResult = computed(() =>
  predictionResult.value?.model_id === LEGACY_HORIZON_MODEL_ID,
);
const isF3FaciesCandidateResult = computed(() =>
  predictionCandidateRenderable.value && predictionResult.value?.model_id === "wellfuse_facies_3d_f3_fast",
);
const predictionOutputEntries = computed(() =>
  Object.entries(predictionResult.value?.outputs || {}).filter((entry): entry is [string, string] => Boolean(entry[1])),
);
const registrationOutputEntries = computed(() =>
  Object.entries(registrationResult.value?.registration?.output_files || {}).filter((entry): entry is [string, string] => Boolean(entry[1])),
);
const preparedViewOutputEntries = computed(() =>
  Object.entries(sampleResult.value?.matching?.output_files || {}).filter((entry): entry is [string, string] => Boolean(entry[1])),
);
const preparedViewDownloadableCount = computed(() =>
  preparedViewOutputEntries.value.filter(([name, path]) => !isDirectoryOutput(name, path)).length,
);
type FlowStageState = "completed" | "current" | "waiting" | "not_required" | "not_consumed";
interface FlowStagePresentation {
  id: "snapshot" | "registration" | "prepared" | "prediction" | "artifacts";
  order: string;
  name: string;
  state: FlowStageState;
  stateLabel: string;
  detail: string;
}
interface FlowStageArtifact {
  taskId: string;
  name: string;
  path: string;
  label: string;
  directory: boolean;
}
const flowStateLabels: Record<FlowStageState, string> = {
  completed: "已完成",
  current: "当前步骤",
  waiting: "等待上游",
  not_required: "本模型不需要",
  not_consumed: "当前模型未消费",
};
const selectedModelFlowStages = computed<FlowStagePresentation[]>(() => {
  const sourceMode = selectedSourceMode.value;
  const registrationPolicy = selectedModelRegistrationPolicy.value;
  const preparedPolicy = selectedModelDataFlow.value?.prepared_view_policy || "optional";
  const registrationStatus = predictionRegistrationStatus.value;
  const preparedStatus = predictionPreparedViewStatus.value;
  const snapshotReady = Boolean(dataSnapshotTaskId.value);
  const registrationReady = formalRegistrationReady.value;
  const preparedReady = preparedViewReady.value;
  const predictionReady = Boolean(predictionResult.value);
  const artifactReady = predictionOutputEntries.value.length > 0;

  let snapshotState: FlowStageState = "waiting";
  let snapshotDetail = "等待数据准备封存工区资产";
  if (sourceMode === "explicit_raw") {
    snapshotState = "not_required";
    snapshotDetail = "该运行器明确允许显式原始井文件入口";
  } else if (sourceMode === "registered_dataset") {
    snapshotState = "not_required";
    snapshotDetail = "该运行器消费已登记并冻结的数据集";
  } else if (snapshotReady) {
    snapshotState = "completed";
    snapshotDetail = `已绑定 ${dataSnapshotTaskId.value}`;
  } else {
    snapshotState = "current";
  }

  let registrationState: FlowStageState = "waiting";
  let registrationDetail = "等待 SourceSnapshot 后执行井震标定";
  if (isFaultSegModel.value) {
    registrationState = "not_required";
    registrationDetail = "断层识别仅消费 SourceSnapshot 中唯一三维 SEG-Y，不提交 Registration";
  } else if (registrationStatus === "used") {
    registrationState = "completed";
    registrationDetail = "运行收据已核验数值消费";
  } else if (["available_not_used", "lineage_only"].includes(registrationStatus || "")) {
    registrationState = "not_consumed";
    registrationDetail = registrationStatus === "lineage_only" ? "仅保留谱系，没有进入模型数值特征" : "产品可用，但运行器没有消费";
  } else if (registrationPolicy === "none") {
    registrationState = "not_required";
    registrationDetail = "模型级 DataFlowSpec 声明 registration_policy=none";
  } else if (registrationPolicy === "optional_control") {
    registrationState = registrationReady ? "not_consumed" : "not_required";
    registrationDetail = registrationReady ? "候选控制可用；是否消费以运行收据为准" : "模型可不消费该控制，但平台仍会先完成统一融合门禁";
  } else if (registrationReady) {
    registrationState = "completed";
    registrationDetail = `已绑定 ${registrationTaskId.value}`;
  } else {
    registrationState = snapshotReady ? "current" : "waiting";
  }

  let preparedState: FlowStageState = "waiting";
  let preparedDetail = "等待标定后构建统一融合视图";
  if (isFaultSegModel.value) {
    preparedState = "not_required";
    preparedDetail = "断层识别不携带或消费 PreparedView";
  } else if (preparedStatus === "used") {
    preparedState = "completed";
    preparedDetail = "PreparedView 哈希与消费收据已核验";
  } else if (preparedStatus === "available_not_used") {
    preparedState = "not_consumed";
    preparedDetail = "PreparedView 已提供，但未进入本次模型特征";
  } else if (preparedPolicy === "preferred") {
    if (preparedReady) {
      preparedState = "completed";
      preparedDetail = `已优先绑定 ${sampleBuildingTaskId.value}`;
    } else if (predictionReady) {
      preparedState = "not_consumed";
      preparedDetail = "这是未绑定融合视图的历史结果；当前版本重新运行前必须先完成融合";
    } else {
      preparedState = registrationState === "waiting" ? "waiting" : "current";
      preparedDetail = "平台统一要求先构建融合视图，再开放预测";
    }
  } else if (selectedModelDataFlow.value?.prepared_view_consumed === false) {
    preparedState = preparedReady ? "not_consumed" : registrationState === "waiting" ? "waiting" : "current";
    preparedDetail = preparedReady ? "平台融合门禁已完成；当前模型合同明确不消费" : "等待平台统一融合门禁；模型合同仍保留不消费声明";
  } else if (preparedPolicy === "required") {
    preparedState = preparedReady ? "completed" : registrationState === "completed" ? "current" : "waiting";
    preparedDetail = preparedReady ? `已绑定 ${sampleBuildingTaskId.value}` : preparedDetail;
  } else {
    preparedState = preparedReady ? "completed" : registrationState === "waiting" ? "waiting" : "current";
    preparedDetail = preparedReady ? "平台融合门禁已完成" : "当前模型不消费融合特征，但预测仍需等待融合完成";
  }

  const predictionPrerequisitesReady = isFaultSegModel.value
    ? faultSegSnapshotSourceReady.value
    : preparedReady && (registrationPolicy !== "required" || registrationReady);
  const predictionState: FlowStageState = predictionReady
    ? "completed"
    : predictionInputReady.value && predictionPrerequisitesReady
      ? "current"
      : "waiting";
  const artifactState: FlowStageState = artifactReady
    ? "completed"
    : predictionReady
      ? "current"
      : "waiting";
  const stages: Array<Omit<FlowStagePresentation, "stateLabel">> = [
    { id: "snapshot", order: "01", name: "SourceSnapshot", state: snapshotState, detail: snapshotDetail },
    { id: "registration", order: "02", name: "Registration", state: registrationState, detail: registrationDetail },
    { id: "prepared", order: "03", name: "PreparedView", state: preparedState, detail: preparedDetail },
    {
      id: "prediction",
      order: "04",
      name: "Prediction",
      state: predictionState,
      detail: predictionReady
        ? `${modelPresentationName(
            predictionResult.value?.model_id,
            predictionResult.value?.model_name,
            predictionResult.value?.scientific_status,
          )} 已完成`
        : !predictionPrerequisitesReady
          ? isFaultSegModel.value
            ? faultSegSnapshotSourceReason.value
            : "等待本模型声明的标定或融合视图前置条件"
          : predictionCompatibilityReason.value,
    },
    {
      id: "artifacts",
      order: "05",
      name: "Artifacts",
      state: artifactState,
      detail: artifactReady ? `${predictionOutputEntries.value.length} 项成果已登记` : predictionReady ? "等待成果登记" : "推理完成后生成可追溯成果",
    },
  ];
  return stages.map((stage) => ({ ...stage, stateLabel: flowStateLabels[stage.state] }));
});
const mainBlockingIssue = computed(() => effectiveBlockingIssues.value[0]);
const currentPreparationGateReady = computed(() => {
  if (!preparation.value) return false;
  if (!hasExplicitPreparationScope.value) return true;
  return effectiveBlockingCount.value === 0
    && preparation.value.gates.can_run_selected_task !== false
    && preparation.value.task_readiness?.ready !== false;
});
const nextAction = computed(() => {
  if (!preparation.value) return { label: "登记并准备数据", view: "preparation" as ViewKey };
  if (!hasExplicitPreparationScope.value || !currentScopeModelId.value) {
    if (preparedViewReady.value) return { label: "选择推理方式", view: "prediction" as ViewKey };
    if (predictionEntryReady.value) return { label: "选择下游模型", view: "prediction" as ViewKey };
    return { label: "进入标定融合", view: "samples" as ViewKey };
  }
  if (mainBlockingIssue.value) return { label: "处理阻断问题", view: "preparation" as ViewKey };
  if (!currentPreparationGateReady.value) return { label: "检查当前模型输入", view: "preparation" as ViewKey };
  if (
    !preparedViewReady.value
    && !(isFaultVolumeModelId(currentScopeModelId.value) && faultSegSnapshotSourceReady.value)
  ) return { label: "进入标定融合", view: "samples" as ViewKey };
  if (!activePredictionTaskSpec.value) return { label: "选择预测任务", view: "prediction" as ViewKey };
  return {
    label: `${selectedModelHasCompletedResult.value ? "重新运行" : "运行"}${activePredictionTaskSpec.value?.short_name || "下游预测"}`,
    view: "prediction" as ViewKey,
  };
});

async function refreshSystemCacheStatus() {
  cacheStatusLoading.value = true;
  cacheStatusError.value = "";
  try {
    cacheStatus.value = await getSystemCache();
  } catch (error) {
    cacheStatus.value = null;
    cacheStatusError.value = error instanceof Error ? error.message : "无法读取缓存状态";
  } finally {
    cacheStatusLoading.value = false;
  }
}

function formatP13RejectionReason(reason: string): string {
  const labels: Record<string, string> = {
    time_scale_domain_and_physics_mismatch: "时间尺度/域与声波物理约束不一致",
  };
  return labels[reason] || reason.replaceAll("_", " ");
}

function canonicalView(view: ViewKey): ViewKey {
  if (view === "samples") return "preparation";
  if (view === "visualization") return "prediction";
  return view;
}

function selectView(view: ViewKey) {
  if (view === "samples") preparationScreen.value = "fusion";
  if (view === "visualization") predictionCanvasMode.value = "base";
  const nextView = canonicalView(view);
  if (nextView === "prediction" && !predictionEntryReady.value) {
    sidebarDirectoryLevel.value = "primary";
    preparationScreen.value = isFaultSegModel.value ? "input" : "fusion";
    activeView.value = "preparation";
    window.location.hash = "preparation";
    statusMessage.value = isFaultSegModel.value
      ? "断层识别等待 SourceSnapshot 唯一三维 SEG-Y"
      : "预测入口将在井震精细标定与融合视图完成后开放";
    errorMessage.value = isFaultSegModel.value
      ? faultSegSnapshotSourceReason.value
      : "请先完成井震融合；当前模型不提供跳过融合的预测路径。";
    return;
  }
  activeView.value = nextView;
  sidebarDirectoryLevel.value = nextView === "prediction"
    ? "prediction"
    : nextView === "layerpulse"
      ? "layerpulse"
      : "primary";
  window.location.hash = nextView;
  errorMessage.value = "";
  if (nextView === "settings") void refreshSystemCacheStatus();
  if (nextView === "evaluation") void refreshAcceptanceSnapshot();
  window.scrollTo({ top: 0, behavior: "auto" });
  void nextTick().then(() => {
    window.requestAnimationFrame(() => {
      window.scrollTo({ top: 0, behavior: "auto" });
      window.requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: "auto" }));
    });
  });
}

function retryBackendConnection() {
  window.location.reload();
}

function handleAssistantNavigation(target: string) {
  const view = target as ViewKey;
  if (view === "overview" || view === "samples" || view === "visualization" || navigation.some((item) => item.id === view)) selectView(view);
}

function openWellLogWorkspace() {
  visualizationMode.value = "logs";
  activeView.value = "visualization";
  window.history.pushState(null, "", "#visualization-logs");
  errorMessage.value = "";
  window.scrollTo({ top: 0, behavior: "auto" });
}

function showPreparationInput() {
  // The streamlined input screen always starts a new model-agnostic survey.
  // Historical task scope remains attached to the sealed result itself.
  preparationTargetTaskId.value = "";
  preparationTargetModelId.value = "";
  preparationScreen.value = "input";
  window.scrollTo({ top: 0, behavior: "auto" });
}

function showPreparationSourceStep() {
  preparationScreen.value = "input";
  window.scrollTo({ top: 0, behavior: "auto" });
}

function openRegistrationPreparation() {
  const reason = errorMessage.value;
  selectView("preparation");
  showPreparationInput();
  errorMessage.value = reason;
  statusMessage.value = "原始文件路径已保留；请重新识别、补全并封存数据合同";
}

function showPreparationPipeline(stage = "全部") {
  if (!preparation.value) return;
  preparationScreen.value = "pipeline";
  issueFilter.value = stage;
  void nextTick().then(() => window.scrollTo({ top: 0, behavior: "auto" }));
}

function showPreparationFusion() {
  if (!dataSnapshotTaskId.value) return;
  preparationScreen.value = "fusion";
  void nextTick().then(() => window.scrollTo({ top: 0, behavior: "auto" }));
}

function openAdvancedDataContract() {
  selectView("preparation");
  preparationScreen.value = "input";
  void nextTick().then(() => {
    const details = document.getElementById("source-contract-advanced") as HTMLDetailsElement | null;
    if (!details) return;
    details.open = true;
    details.scrollIntoView({ behavior: "smooth", block: "center" });
  });
}

function contractCandidateCanPopulate(candidate: SourceContractCandidate): boolean {
  if (!SUPPORTED_VERTICAL_CONTRACT_FIELDS.has(candidate.field)) return false;
  const status = String(candidate.status || "").toLowerCase();
  if (
    !candidate.auto_applied
    && !["verified", "candidate", "review_required", "llm_suggestion"].includes(status)
  ) return false;
  if (["insufficient", "conflict"].includes(status)) return false;
  // Kimi may suggest low-risk draft values, but it may never manufacture or
  // sign the physical corrected-to-SRD declaration.
  if (
    status === "llm_suggestion"
    && candidate.field === "seismic_correction_state"
  ) return false;
  if (candidate.field === "vertical_crs_id") {
    const value = String(candidate.value ?? "").trim();
    return Boolean(value && value !== "LOCAL_MSL_UNSPECIFIED");
  }
  if (candidate.field === "seismic_srd_elevation_m") {
    return typeof candidate.value === "number" && Number.isFinite(candidate.value);
  }
  if (candidate.field === "seismic_time_domain") return ["TWT", "OWT"].includes(String(candidate.value));
  if (candidate.field === "seismic_correction_state") {
    return ["corrected_to_srd", "uncorrected"].includes(String(candidate.value));
  }
  return false;
}

function contractCandidateNeedsHumanConfirmation(candidate: SourceContractCandidate): boolean {
  if (candidate.auto_applied === true) return false;
  const status = String(candidate.status || "").toLowerCase();
  if (status === "llm_suggestion") return candidate.field !== "seismic_correction_state";
  return candidate.field === "seismic_correction_state"
    && candidate.value === "corrected_to_srd"
    || candidate.requires_human_confirmation === true;
}

function contractCandidateFieldLabel(field: string): string {
  return ({
    vertical_crs_id: "测区垂向 CRS",
    seismic_srd_elevation_m: "SRD 高程",
    seismic_time_domain: "地震时间域",
    seismic_correction_state: "时间校正状态",
  } as Record<string, string>)[field] || field;
}

function contractCandidateValueLabel(candidate: SourceContractCandidate): string {
  if (candidate.status === "insufficient" || candidate.value === null || candidate.value === "unknown") {
    return "原始证据不足";
  }
  const suffix = candidate.field === "seismic_srd_elevation_m" ? " m MSL" : "";
  return `${String(candidate.value)}${suffix}`;
}

function contractCandidateStatusLabel(candidate: SourceContractCandidate): string {
  if (autoPopulatedSurveyContractFields.value.includes(candidate.field)) return "已自动写入草稿";
  if (candidate.status === "llm_suggestion") return "Kimi建议 · 需一次确认";
  if (contractCandidateNeedsHumanConfirmation(candidate)) return "需一次物理声明";
  if (candidate.auto_applied) return "规则已核验";
  if (candidate.status === "insufficient") return "不可推测";
  if (candidate.status === "conflict") return "证据冲突";
  if (candidate.status === "review_required") return "规则候选";
  if (candidate.status === "verified") return "证据已核验";
  return "候选";
}

function contractCandidateEvidenceLabel(candidate: SourceContractCandidate): string {
  const evidence = Array.isArray(candidate.evidence)
    ? candidate.evidence.filter(Boolean).join("；")
    : String(candidate.evidence || "").trim();
  return evidence || candidate.source || "未返回可引用的原文证据";
}

function populateSeismicVerticalContractField(field: string, value: unknown) {
  if (field === "vertical_crs_id") verticalCrsId.value = String(value);
  if (field === "seismic_srd_elevation_m") seismicSrdElevation.value = Number(value);
  if (field === "seismic_time_domain" && ["TWT", "OWT"].includes(String(value))) {
    seismicTimeDomain.value = String(value) as typeof seismicTimeDomain.value;
  }
  if (field === "seismic_correction_state" && value === "uncorrected") {
    seismicCorrectionState.value = "uncorrected";
  }
}

function surveyAttestationMatchesDraft(): boolean {
  return surveyAttestation.value !== null
    && verticalCrsId.value !== "LOCAL_MSL_UNSPECIFIED"
    && verticalCrsId.value.toUpperCase().includes("MSL")
    && seismicSrdElevation.value !== ""
    && Number.isFinite(Number(seismicSrdElevation.value))
    && Number(surveyAttestation.value.declared_srd_elevation_m) === Number(seismicSrdElevation.value)
    && (surveyAttestation.value.vertical_reference ?? "MSL") === "MSL"
    && (surveyAttestation.value.time_domain ?? "TWT") === "TWT"
    && (surveyAttestation.value.correction_state ?? "corrected_to_srd") === "corrected_to_srd"
    && seismicTimeDomain.value === "TWT"
    && seismicCorrectionState.value === "corrected_to_srd";
}

function surveyAttestationDeclaration(srdElevationM: number): string {
  const rendered = Object.is(srdElevationM, -0) || srdElevationM === 0
    ? "0"
    : String(srdElevationM);
  return "本人确认：本次数据准备中登记的全部 SEG-Y 地震数据，其处理基准面（SRD）"
    + `为平均海平面（MSL）${rendered} m，地震时间域为 TWT，且均已校正到该 SRD。`;
}

function autoPopulateSafeSeismicVerticalContract(): number {
  const patch = preparation.value?.request_patch || {};
  const populated: string[] = [];
  for (const [field, value] of Object.entries(patch)) {
    if (field === "seismic_correction_state" && value === "corrected_to_srd") continue;
    const candidate: SourceContractCandidate = { field, value, status: "verified" };
    if (!contractCandidateCanPopulate(candidate)) continue;
    populateSeismicVerticalContractField(field, value);
    populated.push(field);
  }
  if (!surveyAttestationMatchesDraft()) surveyAttestation.value = null;
  autoPopulatedSurveyContractFields.value = Array.from(new Set([
    ...autoPopulatedSurveyContractFields.value,
    ...populated,
  ]));
  if (populated.length) {
    preparationScopeNotice.value = `Kimi与本地规则已自动判断并写入 ${populated.length} 项安全合同候选；不会自动重跑数据准备。`;
    statusMessage.value = preparationScopeNotice.value;
  }
  return populated.length;
}

function pendingContractValue(
  field: string,
  fallback: string | number,
  candidates: SourceContractCandidate[],
): string | number {
  return candidates.find((candidate) => candidate.field === field)?.value as string | number
    ?? fallback;
}

function correctedToSrdAttestationReady(candidates: SourceContractCandidate[] = []): boolean {
  const nextVerticalCrs = String(pendingContractValue("vertical_crs_id", verticalCrsId.value, candidates));
  const nextSrd = pendingContractValue(
    "seismic_srd_elevation_m",
    seismicSrdElevation.value === "" ? Number.NaN : Number(seismicSrdElevation.value),
    candidates,
  );
  const nextTimeDomain = String(pendingContractValue("seismic_time_domain", seismicTimeDomain.value, candidates));
  return nextVerticalCrs !== "LOCAL_MSL_UNSPECIFIED"
    && nextVerticalCrs.toUpperCase().includes("MSL")
    && Number.isFinite(Number(nextSrd))
    && nextTimeDomain === "TWT";
}

function writeCorrectedToSrdAttestation() {
  const srdElevationM = Number(seismicSrdElevation.value);
  if (!Number.isFinite(srdElevationM)) return;
  seismicCorrectionState.value = "corrected_to_srd";
  surveyAttestation.value = {
    contract_version: "well-seismic.survey-attestation.v1",
    declared_srd_elevation_m: srdElevationM,
    vertical_reference: "MSL",
    time_domain: "TWT",
    correction_state: "corrected_to_srd",
    source: "human_user",
    confirmation_channel: "user_ui",
    confirmed_at: new Date().toISOString(),
  };
}

function confirmCorrectedToSrdAttestation(): boolean {
  const ready = correctedToSrdAttestationReady();
  if (!ready) {
    errorMessage.value = "只有垂向基准已明确为MSL、SRD高程有限且时间域为TWT时，才能确认corrected_to_srd；其余安全候选已保留在草稿中。";
    return false;
  }
  const declaration = surveyAttestationDeclaration(Number(seismicSrdElevation.value));
  const confirmed = window.confirm(
    `${declaration}\n\n此声明将与全部安全候选一起写入下一次数据准备请求，并绑定到SEG-Y完整内容哈希；不会自动重跑或修改当前快照。`,
  );
  if (!confirmed) return false;
  writeCorrectedToSrdAttestation();
  autoPopulatedSurveyContractFields.value = Array.from(new Set([
    ...autoPopulatedSurveyContractFields.value,
    "seismic_correction_state",
  ]));
  return true;
}

function applySeismicVerticalContractCandidates() {
  autoPopulateSafeSeismicVerticalContract();
  const pending = [...pendingSeismicVerticalCandidates.value];
  if (!pending.length) return;
  const includesCorrectedToSrd = pending.some((candidate) =>
    candidate.field === "seismic_correction_state"
    && candidate.value === "corrected_to_srd"
    && candidate.status !== "llm_suggestion",
  );
  if (includesCorrectedToSrd && !correctedToSrdAttestationReady(pending)) {
    errorMessage.value = "合并候选仍不足以证明MSL、SRD=0 m与TWT三项前提，暂不能签署corrected_to_srd；Kimi不会代签物理声明。";
    return;
  }
  const decisions = pending
    .map((candidate) => `${contractCandidateFieldLabel(candidate.field)} = ${contractCandidateValueLabel(candidate)}`)
    .join("\n");
  const declaration = surveyAttestationDeclaration(Number(pendingContractValue(
    "seismic_srd_elevation_m",
    seismicSrdElevation.value,
    pending,
  )));
  const prompt = includesCorrectedToSrd
    ? `确认一次性采用以下候选？\n\n${decisions}\n\n${declaration}\n\n候选与声明只写入下一次准备草稿并绑定SEG-Y哈希；不会自动重跑。`
    : `确认一次性采用以下Kimi/规则候选？\n\n${decisions}\n\n这些值只写入下一次准备草稿；不会自动重跑或修改当前快照。`;
  if (!window.confirm(prompt)) return;
  for (const candidate of pending) {
    if (candidate.field === "seismic_correction_state") continue;
    populateSeismicVerticalContractField(candidate.field, candidate.value);
  }
  if (includesCorrectedToSrd) writeCorrectedToSrdAttestation();
  autoPopulatedSurveyContractFields.value = Array.from(new Set([
    ...autoPopulatedSurveyContractFields.value,
    ...pending.map((candidate) => candidate.field),
  ]));
  preparationScopeNotice.value = includesCorrectedToSrd
    ? "全部安全候选、Kimi建议与一次物理声明已写入下一次准备草稿；不会自动重跑15GB数据。"
    : "Kimi/规则候选已通过一次合并确认写入下一次准备草稿；不会自动重跑数据准备。";
  statusMessage.value = preparationScopeNotice.value;
  errorMessage.value = "";
  try {
    savePathConfig(commonPayload());
  } catch {
    appendRestorationWarning("浏览器未能保存自动判断后的数据合同草稿");
  }
}

function handleSeismicCorrectionStateChange(event: Event) {
  const value = (event.target as HTMLSelectElement).value;
  if (value === "corrected_to_srd") {
    if (!confirmCorrectedToSrdAttestation()) return;
  } else {
    seismicCorrectionState.value = value as typeof seismicCorrectionState.value;
    surveyAttestation.value = null;
  }
}

function requestSeismicVerticalCandidateRefresh() {
  if (capabilities.value?.llm.available) useLlmFallback.value = true;
  preparationScopeNotice.value = capabilities.value?.llm.available
    ? "已启用 Kimi 受控判读；请重新执行数据准备以提取可审计的合同候选。"
    : "当前未配置 Kimi；请重新执行数据准备，由本地规则提取可审计候选。";
  openAdvancedDataContract();
}

function addPath(group: PathGroup) {
  clearSourceContractDraft();
  group.paths.push("");
}

function removePath(group: PathGroup, index: number) {
  clearSourceContractDraft();
  group.paths.splice(index, 1);
}

function handleManualPathDraftChange() {
  // Programmatic restoration writes refs directly and does not emit input
  // events.  This handler therefore only invalidates contracts after an
  // operator actually changes the registered asset set in the DOM.
  clearSourceContractDraft();
}

function values(key: PathGroupKey): string[] {
  return groups.value
    .find((group) => group.key === key)!
    .paths.map((path) => path.trim())
    .filter(Boolean);
}

const SURVEY_AUXILIARY_PATH_HINTS = [
  "survey",
  "geometry",
  "coordinate",
  "coords",
  "grid",
  "inline_crossline",
  "ilxl",
  "测区",
  "坐标",
  "网格",
  "线道",
];

function isLikelySurveyAuxiliaryPath(path: string): boolean {
  const normalized = path.trim().replaceAll("\\", "/").toLocaleLowerCase();
  return SURVEY_AUXILIARY_PATH_HINTS.some((hint) => normalized.includes(hint));
}

function commonPayload(): DataPathsPayload {
  const auxiliaryPaths = values("auxiliary");
  const surveyPaths = Array.from(new Set([
    ...values("survey"),
    ...auxiliaryPaths.filter(isLikelySurveyAuxiliaryPath),
  ]));
  return {
    seismic_paths: values("seismic"),
    survey_paths: surveyPaths,
    log_paths: values("logs"),
    well_paths: values("wells"),
    time_depth_paths: values("timeDepth"),
    interpretation_paths: [],
    auxiliary_paths: auxiliaryPaths,
    recursive: recursive.value,
    lightweight: lightweight.value,
    use_llm_fallback: useLlmFallback.value,
    horizontal_crs_id: horizontalCrsId.value.trim() || undefined,
    well_source_crs_id: wellSourceCrsId.value.trim() || undefined,
    seismic_source_crs_id: seismicSourceCrsId.value.trim() || undefined,
    ...(horizontalUnit.value === "unknown" ? {} : { horizontal_unit: horizontalUnit.value }),
    ...(horizontalAxisOrder.value === "unknown" ? {} : { horizontal_axis_order: horizontalAxisOrder.value }),
    ...(coordinateReferenceVerified.value ? { coordinate_reference_verified: true } : {}),
    ...(seismicSrdElevation.value === "" ? {} : { seismic_srd_elevation_m: Number(seismicSrdElevation.value) }),
    ...(verticalCrsId.value.trim() && verticalCrsId.value.trim() !== "LOCAL_MSL_UNSPECIFIED"
      ? { vertical_crs_id: verticalCrsId.value.trim() }
      : {}),
    ...(seismicReplacementVelocity.value === "" ? {} : { seismic_replacement_velocity_mps: Number(seismicReplacementVelocity.value) }),
    ...(seismicTimeDomain.value === "unknown" ? {} : { seismic_time_domain: seismicTimeDomain.value }),
    ...(seismicCorrectionState.value === "unknown" ? {} : { seismic_correction_state: seismicCorrectionState.value }),
    ...(segyGeometryProfile.value.trim() ? { segy_geometry_profile: segyGeometryProfile.value.trim() } : {}),
    ...(segyInlineByte.value === "" ? {} : { segy_inline_byte: Number(segyInlineByte.value) }),
    ...(segyCrosslineByte.value === "" ? {} : { segy_crossline_byte: Number(segyCrosslineByte.value) }),
    ...(segyXByte.value === "" ? {} : { segy_x_byte: Number(segyXByte.value) }),
    ...(segyYByte.value === "" ? {} : { segy_y_byte: Number(segyYByte.value) }),
    ...(segyCoordinateScalarByte.value === "" ? {} : { segy_coordinate_scalar_byte: Number(segyCoordinateScalarByte.value) }),
    ...(wellCoordinateSourceUnit.value === "unknown" ? {} : { well_coordinate_source_unit: wellCoordinateSourceUnit.value }),
    ...(wellVerticalDatumSourceUnit.value === "unknown"
      ? {}
      : { well_vertical_datum_source_unit: wellVerticalDatumSourceUnit.value }),
    ...(lasTwtSourceUnit.value === "unknown" ? {} : { las_twt_source_unit: lasTwtSourceUnit.value }),
    target_task_id: undefined,
    target_model_id: undefined,
    target_scope_explicit: false,
    ...(surveyAttestationMatchesDraft() && surveyAttestation.value
      ? { survey_attestation: surveyAttestation.value }
      : {}),
  };
}

function restoreSourceContractFields(
  payload: Record<string, unknown> | Partial<DataPathsPayload>,
) {
  const source = payload as Partial<DataPathsPayload>;
  horizontalCrsId.value = typeof source.horizontal_crs_id === "string" ? source.horizontal_crs_id : "";
  wellSourceCrsId.value = typeof source.well_source_crs_id === "string" ? source.well_source_crs_id : "";
  seismicSourceCrsId.value = typeof source.seismic_source_crs_id === "string" ? source.seismic_source_crs_id : "";
  horizontalUnit.value = ["m", "ft", "unknown"].includes(String(source.horizontal_unit))
    ? source.horizontal_unit as typeof horizontalUnit.value
    : "unknown";
  horizontalAxisOrder.value = ["XY", "YX", "unknown"].includes(String(source.horizontal_axis_order))
    ? source.horizontal_axis_order as typeof horizontalAxisOrder.value
    : "unknown";
  coordinateReferenceVerified.value = source.coordinate_reference_verified === true;
  seismicSrdElevation.value = typeof source.seismic_srd_elevation_m === "number"
    ? source.seismic_srd_elevation_m
    : "";
  verticalCrsId.value = typeof source.vertical_crs_id === "string"
    ? source.vertical_crs_id
    : "LOCAL_MSL_UNSPECIFIED";
  seismicReplacementVelocity.value = typeof source.seismic_replacement_velocity_mps === "number"
    ? source.seismic_replacement_velocity_mps
    : "";
  seismicTimeDomain.value = ["TWT", "OWT", "unknown"].includes(String(source.seismic_time_domain))
    ? source.seismic_time_domain as typeof seismicTimeDomain.value
    : "unknown";
  seismicCorrectionState.value = ["corrected_to_srd", "uncorrected", "unknown"].includes(String(source.seismic_correction_state))
    ? source.seismic_correction_state as typeof seismicCorrectionState.value
    : "unknown";
  segyGeometryProfile.value = typeof source.segy_geometry_profile === "string" ? source.segy_geometry_profile : "";
  segyInlineByte.value = typeof source.segy_inline_byte === "number" ? source.segy_inline_byte : "";
  segyCrosslineByte.value = typeof source.segy_crossline_byte === "number" ? source.segy_crossline_byte : "";
  segyXByte.value = typeof source.segy_x_byte === "number" ? source.segy_x_byte : "";
  segyYByte.value = typeof source.segy_y_byte === "number" ? source.segy_y_byte : "";
  segyCoordinateScalarByte.value = typeof source.segy_coordinate_scalar_byte === "number" ? source.segy_coordinate_scalar_byte : "";
  wellCoordinateSourceUnit.value = ["m", "ft", "unknown"].includes(String(source.well_coordinate_source_unit))
    ? source.well_coordinate_source_unit as typeof wellCoordinateSourceUnit.value
    : "unknown";
  wellVerticalDatumSourceUnit.value = ["m", "ft", "unknown"].includes(String(source.well_vertical_datum_source_unit))
    ? source.well_vertical_datum_source_unit as typeof wellVerticalDatumSourceUnit.value
    : "unknown";
  lasTwtSourceUnit.value = ["ms", "s", "us", "unknown"].includes(String(source.las_twt_source_unit))
    ? source.las_twt_source_unit as typeof lasTwtSourceUnit.value
    : "unknown";
}

function runtimeContractControlValue(field: RuntimeContractReviewField): string | number {
  const value = runtimeContractDraft.value[field.key];
  return typeof value === "boolean" ? String(value) : value ?? "";
}

function updateRuntimeContractValue(field: RuntimeContractReviewField, event: Event) {
  const raw = (event.target as HTMLInputElement | HTMLSelectElement).value;
  runtimeContractSubmission.value = null;
  if (field.control === "number") {
    if (!raw.trim()) {
      runtimeContractDraft.value[field.key] = "";
      return;
    }
    const parsed = Number(raw);
    runtimeContractDraft.value[field.key] = Number.isFinite(parsed) ? parsed : raw;
    return;
  }
  const matchingChoice = field.choices?.find((choice) => String(choice.value) === raw);
  runtimeContractDraft.value[field.key] = matchingChoice?.value ?? raw;
}

async function showRuntimeContractDialog() {
  await nextTick();
  const dialog = runtimeContractDialog.value;
  if (dialog && !dialog.open) {
    dialog.showModal();
    window.requestAnimationFrame(() => {
      dialog.querySelector<HTMLElement>("input, select")?.focus();
    });
  }
}

function currentPostFusionInferenceContext(): PostFusionInferenceContext | null {
  const snapshotId = dataSnapshotTaskId.value;
  const registrationId = registrationTaskId.value;
  const preparedViewId = sampleBuildingTaskId.value;
  const preparedView = sampleResult.value?.prepared_view;
  if (
    !preparedViewReady.value
    || !snapshotId
    || !registrationId
    || !preparedViewId
    || sampleResult.value?.registration_task_id !== registrationId
    || preparedView?.source_snapshot_id !== snapshotId
  ) return null;
  return {
    snapshotId,
    registrationTaskId: registrationId,
    preparedViewId,
    readyWellCount: preparedView.gates?.registration_fusion_ready_well_ids?.length || 0,
  };
}

function postFusionInferenceContextIsCurrent(context: PostFusionInferenceContext): boolean {
  const current = currentPostFusionInferenceContext();
  return Boolean(
    current
    && context.snapshotId === current.snapshotId
    && context.registrationTaskId === current.registrationTaskId
    && context.preparedViewId === current.preparedViewId
  );
}

function activeSnapshotReconciliationBlockedByRunningTask(): boolean {
  return preparationRunning.value
    || runtimeContractSubmitting.value
    || batchApplyingRecommendations.value
    || fusionWorkflowMutationRunning.value
    || isLayerPulseTaskActive(layerPulseTaskState.value.status)
    || activeSnapshotPersistencePendingCount > 0;
}

async function reconcileProjectActiveSnapshotForInference(destinationLabel: string): Promise<boolean> {
  if (activeSnapshotReconciliationPromise) return activeSnapshotReconciliationPromise;
  const reconciliation = (async () => {
    try {
      const projectCatalog = await getProjects();
      const project = projectCatalog.projects.find((item) => item.project_id === activeProjectId.value)
        || projectCatalog.projects.find((item) => item.project_id === "local-default")
        || [...projectCatalog.projects].sort((left, right) =>
          Date.parse(right.updated_at || "") - Date.parse(left.updated_at || ""),
        )[0];
      const activeSnapshotId = project?.active_snapshot_id || "";
      if (!project || !activeSnapshotId || activeSnapshotId === dataSnapshotTaskId.value) return true;

      const pageSnapshotId = dataSnapshotTaskId.value;
      if (activeSnapshotReconciliationBlockedByRunningTask()) {
        const message = `当前任务仍在运行，已保留页面快照 ${pageSnapshotId.slice(0, 8) || "未绑定"}；任务结束后重新打开${destinationLabel}，平台将自动对齐活动快照 ${activeSnapshotId.slice(0, 8)}`;
        statusMessage.value = message;
        errorMessage.value = message;
        appendRestorationWarning(message);
        return false;
      }

      await restoreLatestDurableWorkflow({ preferLatestSnapshot: true });
      if (dataSnapshotTaskId.value !== activeSnapshotId) {
        const message = `${destinationLabel}未能对齐项目活动快照 ${activeSnapshotId.slice(0, 8)}，已阻止使用旧快照提交模型`;
        statusMessage.value = message;
        errorMessage.value = message;
        appendRestorationWarning(message);
        return false;
      }
      const message = `已从页面快照 ${pageSnapshotId.slice(0, 8) || "未绑定"} 自动对齐项目活动快照 ${activeSnapshotId.slice(0, 8)}，${destinationLabel}使用最新融合合同`;
      statusMessage.value = message;
      errorMessage.value = "";
      return true;
    } catch (error) {
      const reason = error instanceof Error ? error.message : "未知错误";
      const message = `${destinationLabel}无法核验项目活动快照：${reason}`;
      statusMessage.value = message;
      errorMessage.value = message;
      appendRestorationWarning(message);
      return false;
    }
  })();
  activeSnapshotReconciliationPromise = reconciliation;
  try {
    return await reconciliation;
  } finally {
    if (activeSnapshotReconciliationPromise === reconciliation) {
      activeSnapshotReconciliationPromise = null;
    }
  }
}

async function openPostFusionInferenceDestination() {
  if (!await reconcileProjectActiveSnapshotForInference("推理方式选择")) return;
  const context = currentPostFusionInferenceContext();
  if (!context) return;
  const snapshotId = context.snapshotId;
  if (
    layerPulseSourceSnapshotId.value
    && layerPulseSourceSnapshotId.value !== snapshotId
  ) {
    layerPulseExecutionTaskId.value = "";
    layerPulseSourceSnapshotId.value = "";
    layerPulseResult.value = null;
    layerPulseTaskState.value = createIdleLayerPulseTaskState();
    layerPulseSelectedOutputKey.value = "fault_logits";
    layerPulseCanvasMode.value = "base";
    try {
      window.sessionStorage.removeItem(LAST_LAYERPULSE_TASK_STORAGE_KEY);
    } catch {
      appendRestorationWarning("浏览器未能清除上一 SourceSnapshot 的 LayerPulse 任务引用");
    }
  }
  postFusionInferenceContext.value = context;
  postFusionInferenceOpen.value = true;
}

function choosePostFusionOriginal(context: PostFusionInferenceContext) {
  if (!postFusionInferenceContextIsCurrent(context)) return;
  postFusionInferenceOpen.value = false;
  postFusionInferenceContext.value = null;
  selectView("prediction");
  statusMessage.value = "融合视图已绑定，请选择单项解释任务和推理模型。";
}

function choosePostFusionLayerPulse(context: PostFusionInferenceContext) {
  if (!postFusionInferenceContextIsCurrent(context)) return;
  postFusionInferenceOpen.value = false;
  postFusionInferenceContext.value = null;
  selectView("layerpulse");
  statusMessage.value = "融合视图已绑定 LayerPulse，可用唯一共享 Backbone 一次生成全部解释结果。";
}

function closePostFusionInferenceDestination() {
  postFusionInferenceOpen.value = false;
  postFusionInferenceContext.value = null;
}

async function offerRuntimeContractReview(result: WorkflowResult, sourceSnapshotId: string) {
  const review = result.preparation.runtime_contract_review;
  if (!review?.required || !review.fields.length) return;
  const sameSnapshot = runtimeContractSourceSnapshotId.value === sourceSnapshotId
    && runtimeContractReview.value?.contract_version === review.contract_version;
  runtimeContractReview.value = review;
  if (!sameSnapshot) {
    runtimeContractDraft.value = {
      ...review.values,
      ...Object.fromEntries(review.fields.map((field) => [field.key, field.value])),
    };
    runtimeContractSubmission.value = null;
  }
  runtimeContractSourceSnapshotId.value = sourceSnapshotId;
  runtimeContractError.value = "";
  await showRuntimeContractDialog();
}

function reopenRuntimeContractReview() {
  if (runtimeContractReview.value && runtimeContractSourceSnapshotId.value) {
    runtimeContractError.value = "";
    void showRuntimeContractDialog();
    return;
  }
  if (preparationResult.value && dataSnapshotTaskId.value) {
    void offerRuntimeContractReview(preparationResult.value, dataSnapshotTaskId.value);
  }
}

function returnFromRuntimeContractReview() {
  if (runtimeContractSubmitting.value) return;
  runtimeContractDialog.value?.close();
  runtimeContractError.value = "";
  preparationScreen.value = "input";
}

function runtimeContractSubmissionStorageKey(sourceSnapshotId: string): string {
  return `${RUNTIME_CONTRACT_SUBMISSION_STORAGE_PREFIX}:${sourceSnapshotId}`;
}

function loadStoredRuntimeContractSubmission(
  sourceSnapshotId: string,
  fingerprint: string,
): typeof runtimeContractSubmission.value {
  try {
    const raw = window.localStorage.getItem(
      runtimeContractSubmissionStorageKey(sourceSnapshotId),
    );
    if (!raw) return null;
    const parsed = JSON.parse(raw) as {
      fingerprint?: unknown;
      attestation?: Partial<SurveyAttestationPayload>;
    };
    const attestation = parsed.attestation;
    if (
      parsed.fingerprint !== fingerprint
      || attestation?.contract_version !== "well-seismic.survey-attestation.v1"
      || attestation.source !== "human_user"
      || attestation.confirmation_channel !== "user_ui"
      || typeof attestation.declaration_text !== "string"
      || typeof attestation.confirmed_at !== "string"
      || !Number.isFinite(Number(attestation.declared_srd_elevation_m))
    ) return null;
    return {
      fingerprint,
      attestation: attestation as SurveyAttestationPayload,
    };
  } catch {
    return null;
  }
}

function storeRuntimeContractSubmission(
  sourceSnapshotId: string,
  submission: NonNullable<typeof runtimeContractSubmission.value>,
) {
  try {
    window.localStorage.setItem(
      runtimeContractSubmissionStorageKey(sourceSnapshotId),
      JSON.stringify({
        fingerprint: submission.fingerprint,
        attestation: submission.attestation,
      }),
    );
  } catch {
    // The in-memory copy still keeps retries idempotent for this page session.
  }
}

function clearStoredRuntimeContractSubmission(sourceSnapshotId: string) {
  try {
    window.localStorage.removeItem(
      runtimeContractSubmissionStorageKey(sourceSnapshotId),
    );
  } catch {
    // A stale retry record is fingerprint-bound and cannot alter another review.
  }
}

async function confirmRuntimeContractReview() {
  const review = runtimeContractReview.value;
  const sourceSnapshotId = runtimeContractSourceSnapshotId.value;
  if (!review || !sourceSnapshotId || runtimeContractSubmitting.value) return;
  const emptyField = review.fields.find((field) => {
    const value = runtimeContractDraft.value[field.key];
    return value === ""
      || value === null
      || value === undefined
      || (field.control === "number" && (typeof value !== "number" || !Number.isFinite(value)));
  });
  if (emptyField) {
    runtimeContractError.value = `请填写${emptyField.label}`;
    return;
  }
  const replacementVelocity = Number(
    runtimeContractDraft.value.seismic_replacement_velocity_mps,
  );
  if (!Number.isFinite(replacementVelocity) || replacementVelocity <= 0) {
    runtimeContractError.value = "替换速度必须大于 0";
    return;
  }
  const rawSrdElevationM = Number(
    runtimeContractDraft.value.seismic_srd_elevation_m,
  );
  if (!Number.isFinite(rawSrdElevationM) || Math.abs(rawSrdElevationM) > 10_000) {
    runtimeContractError.value = "SRD 高程必须位于 -10000 至 10000 m";
    return;
  }
  const srdElevationM = Math.round(rawSrdElevationM * 100) / 100;
  runtimeContractDraft.value.seismic_srd_elevation_m = srdElevationM;
  const fingerprint = JSON.stringify({
    sourceSnapshotId,
    values: Object.entries(runtimeContractDraft.value).sort(([left], [right]) => (
      left.localeCompare(right)
    )),
  });

  runtimeContractSubmitting.value = true;
  runtimeContractError.value = "";
  try {
    let submission = runtimeContractSubmission.value;
    if (!submission || submission.fingerprint !== fingerprint) {
      submission = loadStoredRuntimeContractSubmission(sourceSnapshotId, fingerprint);
    }
    if (!submission || submission.fingerprint !== fingerprint) {
      submission = {
        fingerprint,
        attestation: {
          contract_version: "well-seismic.survey-attestation.v1",
          declared_srd_elevation_m: srdElevationM,
          vertical_reference: "MSL",
          time_domain: "TWT",
          correction_state: "corrected_to_srd",
          declaration_text: surveyAttestationDeclaration(srdElevationM),
          source: "human_user",
          confirmation_channel: "user_ui",
          confirmed_at: new Date().toISOString(),
        },
      };
      storeRuntimeContractSubmission(sourceSnapshotId, submission);
    }
    runtimeContractSubmission.value = submission;
    if (!submission.confirmation) {
      submission.confirmation = await confirmRuntimeContract(
        sourceSnapshotId,
        runtimeContractDraft.value,
        submission.attestation,
      );
    }
    const confirmed = submission.confirmation;
    const derivedTask = await getTask(confirmed.derived_snapshot_id);
    if (
      derivedTask.task_type !== "data_preparation"
      || derivedTask.status !== "completed"
      || !isWorkflowResult(derivedTask.result)
    ) {
      throw new Error("运行参数已提交，但派生快照尚未完整落地，请重试确认");
    }
    restoreCompletedDataPreparationTask(derivedTask, {
      showPipeline: false,
      persistPathConfig: false,
      persistActiveSnapshot: true,
    });
    try {
      savePathConfig(confirmed.effective_request);
    } catch {
      appendRestorationWarning("运行参数已封存，但浏览器未能保存页面引用；平台任务记录不受影响");
    }
    runtimeContractDialog.value?.close();
    runtimeContractReview.value = null;
    runtimeContractDraft.value = {};
    runtimeContractSubmission.value = null;
    clearStoredRuntimeContractSubmission(sourceSnapshotId);
    const canEnterFusion = sealedWellSeismicWorkflowReady.value;
    preparationScreen.value = canEnterFusion ? "fusion" : "pipeline";
    statusMessage.value = canEnterFusion
      ? "运行参数已确认，井震融合可以继续"
      : "运行参数已确认，数据快照可以继续使用";
    preparationStatusMessage.value = statusMessage.value;
  } catch (error) {
    runtimeContractError.value = error instanceof Error ? error.message : "运行参数确认失败";
  } finally {
    runtimeContractSubmitting.value = false;
  }
}

function clearSourceContractDraft() {
  // A path-set replacement is a new survey draft.  Never carry Beijing 1954,
  // SEG-Y byte locations, or a vertical datum declaration into a WGS84 draft.
  restoreSourceContractFields({});
  sealedSnapshotSegyContract.value = null;
  sealedSnapshotSegyGeometryAuthority.value = null;
  surveyAttestation.value = null;
  autoPopulatedSurveyContractFields.value = [];
}

function rememberSourceSnapshot(
  snapshotId: string,
  projectId = activeProjectId.value,
  options: { persistActiveSnapshot?: boolean } = {},
) {
  if (!snapshotId) return;
  if (projectId) activeProjectId.value = projectId;
  try {
    window.sessionStorage.setItem(LAST_SOURCE_SNAPSHOT_STORAGE_KEY, snapshotId);
  } catch {
    appendRestorationWarning("浏览器未能保存当前SourceSnapshot；刷新后将从任务记录恢复");
  }
  if (options.persistActiveSnapshot === false) return;
  const persistenceProjectId = activeProjectId.value;
  const persistenceKey = `${persistenceProjectId}:${snapshotId}`;
  latestRequestedActiveSnapshot = persistenceKey;
  if (!persistenceProjectId) return;
  if (
    persistenceKey === lastPersistedActiveSnapshot
    && activeSnapshotPersistencePendingCount === 0
  ) return;
  activeSnapshotPersistencePendingCount += 1;
  activeSnapshotPersistenceQueue = activeSnapshotPersistenceQueue
    .catch(() => undefined)
    .then(async () => {
      // Coalesce selections that changed before their queued write began. A
      // write already in flight still completes before the final selection,
      // so an older response can never overwrite the newest pointer.
      try {
        if (
          persistenceKey !== latestRequestedActiveSnapshot
          || persistenceKey === lastPersistedActiveSnapshot
        ) return;
        await setActiveProjectSnapshot(persistenceProjectId, snapshotId);
        lastPersistedActiveSnapshot = persistenceKey;
        clearRestorationWarningsByPrefix(ACTIVE_SNAPSHOT_PERSISTENCE_WARNING_PREFIX);
      } catch (error) {
        if (persistenceKey !== latestRequestedActiveSnapshot) return;
        const reason = activeSnapshotPersistenceFailureReason(error);
        appendRestorationWarning(`当前SourceSnapshot未能写入平台状态库：${reason}`);
      } finally {
        activeSnapshotPersistencePendingCount = Math.max(
          0,
          activeSnapshotPersistencePendingCount - 1,
        );
      }
    });
}

function verifiedAutomaticSegyContract(result: WorkflowResult): Partial<DataPathsPayload> | null {
  const snapshot = result.data_snapshot;
  if (
    snapshot?.contract_version !== "well-seismic.source-snapshot.v3"
    || snapshot.state !== "sealed"
    || !/^[0-9a-f]{64}$/i.test(String(snapshot.snapshot_sha256 || ""))
    || result.seismic.length !== 1
  ) return null;
  const seismic = result.seismic[0];
  if (
    !Number.isFinite(seismic.confidence)
    || seismic.confidence < GEOPATH_MINIMUM_AUTOMATIC_GEOMETRY_CONFIDENCE
  ) return null;
  const matchingAssets = result.assets.filter((asset) =>
    asset.role.toLocaleLowerCase() === "seismic" && asset.path === seismic.path,
  );
  if (matchingAssets.length !== 1) return null;
  const asset = matchingAssets[0];
  const identity = asset.geometry_identity;
  if (
    !/^[0-9a-f]{64}$/i.test(String(asset.sha256 || ""))
    || !/^[0-9a-f]{64}$/i.test(String(asset.geometry_fingerprint || ""))
    || !/^[0-9a-f]{64}$/i.test(String(asset.asset_options_sha256 || ""))
    || !identity?.profile?.trim()
    || identity.geometry_fingerprint !== asset.geometry_fingerprint
  ) return null;

  const headerBytes = new Map<string, number>();
  const headerConfidence = new Map<string, number>();
  const scalarBytes: number[] = [];
  const gridConfidences: number[] = [];
  for (const issue of seismic.issues) {
    const header = issue.match(/^(inline|crossline|x|y)_byte=(\d+):confidence=(\d+(?:\.\d+)?)$/);
    if (header) {
      if (headerBytes.has(header[1])) return null;
      const byte = Number(header[2]);
      const confidence = Number(header[3]);
      if (
        !Number.isInteger(byte)
        || byte < 1
        || byte > 237
        || !Number.isFinite(confidence)
        || confidence < GEOPATH_MINIMUM_AUTOMATIC_HEADER_CONFIDENCE
      ) return null;
      headerBytes.set(header[1], byte);
      headerConfidence.set(header[1], confidence);
      continue;
    }
    const scalar = issue.match(/^coordinate_scalar_byte=(\d+):configured$/);
    if (scalar) {
      scalarBytes.push(Number(scalar[1]));
      continue;
    }
    const grid = issue.match(/^inline_crossline_grid=.*(?:^|,)confidence:(\d+(?:\.\d+)?)$/);
    if (grid) gridConfidences.push(Number(grid[1]));
  }
  const fields = ["inline", "crossline", "x", "y"];
  if (
    fields.some((field) => !headerBytes.has(field))
    || new Set(headerBytes.values()).size !== fields.length
    || scalarBytes.length !== 1
    || !Number.isInteger(scalarBytes[0])
    || scalarBytes[0] < 1
    || scalarBytes[0] > 239
    || gridConfidences.length !== 1
  ) return null;
  const expectedConfidence = Math.min(
    fields.reduce((total, field) => total + Number(headerConfidence.get(field)), 0) / fields.length,
    gridConfidences[0],
  );
  if (Math.abs(expectedConfidence - seismic.confidence) > 0.002) return null;
  const scalarRange = new Set([scalarBytes[0], scalarBytes[0] + 1]);
  if (fields.some((field) => {
    const start = Number(headerBytes.get(field));
    return [start, start + 1, start + 2, start + 3].some((byte) => scalarRange.has(byte));
  })) return null;
  return {
    segy_geometry_profile: identity.profile.trim(),
    segy_inline_byte: headerBytes.get("inline"),
    segy_crossline_byte: headerBytes.get("crossline"),
    segy_x_byte: headerBytes.get("x"),
    segy_y_byte: headerBytes.get("y"),
    segy_coordinate_scalar_byte: scalarBytes[0],
  };
}

function rememberSealedSourceContract(
  payload: Record<string, unknown> | Partial<DataPathsPayload> | undefined,
  result?: WorkflowResult | null,
) {
  if (!payload) {
    sealedSnapshotSegyContract.value = null;
    sealedSnapshotSegyGeometryAuthority.value = null;
    return;
  }
  const source = payload as Partial<DataPathsPayload>;
  restoreSourceContractFields(source);
  const fields: Array<keyof DataPathsPayload> = [
    "segy_geometry_profile",
    "segy_inline_byte",
    "segy_crossline_byte",
    "segy_x_byte",
    "segy_y_byte",
    "segy_coordinate_scalar_byte",
  ];
  const populated = fields.filter((field) => source[field] !== null && source[field] !== undefined);
  if (populated.length === fields.length) {
    sealedSnapshotSegyContract.value = {
      segy_geometry_profile: source.segy_geometry_profile,
      segy_inline_byte: source.segy_inline_byte,
      segy_crossline_byte: source.segy_crossline_byte,
      segy_x_byte: source.segy_x_byte,
      segy_y_byte: source.segy_y_byte,
      segy_coordinate_scalar_byte: source.segy_coordinate_scalar_byte,
    };
    sealedSnapshotSegyGeometryAuthority.value = "explicit";
    return;
  }
  // A partial explicit declaration is a conflict, never permission for
  // automatic evidence to fill the remaining fields.
  const automatic = populated.length === 0 && result
    ? verifiedAutomaticSegyContract(result)
    : null;
  sealedSnapshotSegyContract.value = automatic;
  sealedSnapshotSegyGeometryAuthority.value = automatic ? "verified_automatic" : null;
}

function restorePreparationTarget(payload: Record<string, unknown> | Partial<DataPathsPayload> | undefined) {
  if (!payload) return;
  const source = payload as Partial<DataPathsPayload>;
  preparationTargetTaskId.value = source.target_scope_explicit === true
    && typeof source.target_task_id === "string"
    ? source.target_task_id
    : "";
  preparationTargetModelId.value = preparationTargetTaskId.value
    && typeof source.target_model_id === "string"
    ? source.target_model_id
    : "";
  normalizePreparationTarget();
}

function restoreSurveyAttestation(payload: Record<string, unknown> | Partial<DataPathsPayload>) {
  const source = payload as Partial<DataPathsPayload>;
  const attestation = source.survey_attestation;
  surveyAttestation.value = attestation
    && attestation.contract_version === "well-seismic.survey-attestation.v1"
    && Number.isFinite(Number(attestation.declared_srd_elevation_m))
    && (attestation.vertical_reference ?? "MSL") === "MSL"
    && (attestation.time_domain ?? "TWT") === "TWT"
    && (attestation.correction_state ?? "corrected_to_srd") === "corrected_to_srd"
    && attestation.source === "human_user"
    && attestation.confirmation_channel === "user_ui"
    ? { ...attestation }
    : null;
}

function normalizePreparationTarget() {
  const targetTaskId = preparationTargetTaskId.value;
  if (!targetTaskId) {
    preparationTargetModelId.value = "";
    return;
  }
  if (!capabilities.value) return;
  if (!predictionTaskDefinitions.value.some((task) => task.id === targetTaskId)) {
    preparationTargetTaskId.value = "";
    preparationTargetModelId.value = "";
    preparationScopeNotice.value = `历史准备范围「${targetTaskId}」当前不在可选任务中；新运行已切换为通用井震检查。`;
    return;
  }
  const targetModelId = preparationTargetModelId.value;
  if (targetModelId && !preparationDraftModels.value.some((model) => model.id === targetModelId)) {
    preparationTargetModelId.value = recommendedPreparationModelId(targetTaskId);
    preparationScopeNotice.value = `历史目标模型「${targetModelId}」不属于当前任务；已切换为当前任务的推荐可运行模型。`;
  }
}

function savePathConfig(payload: DataPathsPayload) {
  window.sessionStorage.setItem(PATH_CONFIG_STORAGE_KEY, JSON.stringify(payload));
}

function activeSnapshotPersistenceFailureReason(error: unknown): string {
  if (
    error instanceof ApiRequestError
    && error.status === 404
    && error.message.trim().toLowerCase() === "not found"
  ) {
    return "后端版本不匹配：当前服务缺少活动快照持久化接口，请重启平台后端";
  }
  return error instanceof Error ? error.message : "未知错误";
}

function clearRestorationWarningsByPrefix(prefix: string) {
  let changed = false;
  for (let index = restorationWarningMessages.length - 1; index >= 0; index -= 1) {
    if (!restorationWarningMessages[index].startsWith(prefix)) continue;
    restorationWarningMessages.splice(index, 1);
    changed = true;
  }
  if (changed) restorationWarning.value = restorationWarningMessages.join("；");
}

function appendRestorationWarning(message: string) {
  const normalized = message.trim();
  if (!normalized || restorationWarningMessages.includes(normalized)) return;
  restorationWarningMessages.push(normalized);
  restorationWarning.value = restorationWarningMessages.join("；");
}

function applyPathPayload(payload: Record<string, unknown> | Partial<DataPathsPayload>) {
  const source = payload as Partial<DataPathsPayload>;
  clearSourceContractDraft();
  const mapping: Record<PathGroupKey, string[]> = {
    seismic: Array.isArray(source.seismic_paths) ? source.seismic_paths : [],
    survey: Array.isArray(source.survey_paths) ? source.survey_paths : [],
    logs: Array.isArray(source.log_paths) ? source.log_paths : [],
    wells: Array.isArray(source.well_paths) ? source.well_paths : [],
    timeDepth: Array.isArray(source.time_depth_paths) ? source.time_depth_paths : [],
    interpretations: Array.isArray(source.interpretation_paths) ? source.interpretation_paths : [],
    auxiliary: Array.isArray(source.auxiliary_paths) ? source.auxiliary_paths : [],
  };
  groups.value.forEach((group) => {
    const restored = mapping[group.key].filter((path): path is string => typeof path === "string");
    group.paths = restored.length ? restored : group.optional ? [] : [""];
  });
  if (typeof source.recursive === "boolean") recursive.value = source.recursive;
  if (typeof source.lightweight === "boolean") lightweight.value = source.lightweight;
  if (typeof source.use_llm_fallback === "boolean") {
    useLlmFallback.value = source.use_llm_fallback && capabilities.value?.llm.available !== false;
  }
  restoreSourceContractFields(source);
  restorePreparationTarget(source);
  restoreSurveyAttestation(source);
}

function restorePathConfig(): boolean {
  const saved = window.sessionStorage.getItem(PATH_CONFIG_STORAGE_KEY);
  if (!saved) return false;
  try {
    const payload = JSON.parse(saved) as Partial<DataPathsPayload>;
    applyPathPayload(payload);
    return true;
  } catch {
    window.sessionStorage.removeItem(PATH_CONFIG_STORAGE_KEY);
    return false;
  }
}

interface CompletedPreparationRestoreState {
  taskId: string;
  snapshotId: string;
  projectId: string;
  message: string;
  request: Partial<DataPathsPayload>;
  result: WorkflowResult;
}

interface CompletedPreparationRestoreOptions {
  snapshotId?: string;
  resetDownstream?: boolean;
  persistPathConfig?: boolean;
  rememberTask?: boolean;
  showPipeline?: boolean;
  persistActiveSnapshot?: boolean;
  offerRuntimeReview?: boolean;
}

const PREPARATION_PATH_KEYS: Array<keyof DataPathsPayload> = [
  "seismic_paths",
  "survey_paths",
  "log_paths",
  "well_paths",
  "time_depth_paths",
  "interpretation_paths",
  "auxiliary_paths",
];

function stageCompletedDataPreparationTask(
  task: BackgroundTask,
  snapshotId?: string,
): CompletedPreparationRestoreState {
  if (
    task.task_type !== "data_preparation"
    || task.status !== "completed"
    || !task.result
    || "prediction" in task.result
    || !("summary" in task.result)
  ) {
    throw new Error("历史任务不是可恢复的已完成数据准备任务");
  }
  if (!task.request || typeof task.request !== "object") {
    throw new Error("历史数据准备任务缺少输入合同，无法安全恢复路径");
  }
  if (!PREPARATION_PATH_KEYS.some((key) => Object.hasOwn(task.request!, key))) {
    throw new Error("历史数据准备任务没有登记路径字段，已阻止结果与空白输入混用");
  }
  const result = task.result as WorkflowResult;
  if (!result.preparation || !Array.isArray(result.seismic) || !Array.isArray(result.errors)) {
    throw new Error("历史数据准备结果结构不完整，无法恢复");
  }
  return {
    taskId: task.task_id,
    snapshotId: snapshotId || task.snapshot_id || result.data_snapshot?.snapshot_id || task.task_id,
    projectId: task.project_id || result.data_snapshot?.project_id || activeProjectId.value,
    message: task.message,
    request: { ...task.request } as Partial<DataPathsPayload>,
    result,
  };
}

function applyCompletedDataPreparationState(
  state: CompletedPreparationRestoreState,
  options: CompletedPreparationRestoreOptions = {},
) {
  const {
    resetDownstream = true,
    persistPathConfig = true,
    rememberTask = true,
    showPipeline = true,
    persistActiveSnapshot = false,
  } = options;

  // The staged task has already been validated. Commit all coupled refs in one
  // synchronous block so a completed result is never shown against another draft.
  applyPathPayload(state.request);
  rememberSealedSourceContract(state.request, state.result);
  taskId.value = state.taskId;
  dataSnapshotTaskId.value = state.snapshotId;
  rememberSourceSnapshot(state.snapshotId, state.projectId, { persistActiveSnapshot });
  preparationResult.value = state.result;
  preparationProgress.value = 100;
  preparationActivityPhase.value = "completed";
  preparationStatusMessage.value = state.message || "数据准备已完成";
  statusMessage.value = state.message || "数据准备已完成";

  if (resetDownstream) {
    registrationTaskId.value = "";
    registrationResult.value = null;
    horizontalRegistrationTaskId.value = "";
    horizontalRegistrationResult.value = null;
    sampleBuildingTaskId.value = "";
    sampleResult.value = null;
    predictionSourceTaskId.value = "";
    predictionTaskId.value = "";
    predictionResult.value = null;
    if (predictionHistorySnapshotId.value !== state.snapshotId) {
      predictionHistoryByTask.value = {};
      predictionHistorySnapshotId.value = state.snapshotId;
    }
    clearGeoPathCandidateState();
  }

  initializeVisualization(state.result);
  autoPopulatedSurveyContractFields.value = [];
  autoPopulateSafeSeismicVerticalContract();
  const firstAttentionStage = state.result.preparation.issues.find(
    (issue) => issueNeedsCurrentAttention(issue),
  )?.stage;
  issueFilter.value = firstAttentionStage || "全部";
  if (showPipeline) preparationScreen.value = "pipeline";

  if (persistPathConfig) {
    try {
      savePathConfig(commonPayload());
    } catch {
      appendRestorationWarning("浏览器未能保存已恢复的输入合同；本次页面仍可继续使用");
    }
  }
  if (rememberTask) {
    try {
      window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, state.taskId);
    } catch {
      appendRestorationWarning("浏览器未能保存后台任务引用；下次刷新可能需要再次从平台状态库恢复");
    }
  }
}

function restoreCompletedDataPreparationTask(
  task: BackgroundTask,
  options: CompletedPreparationRestoreOptions = {},
) {
  capturePreparationRunTiming(task);
  const state = stageCompletedDataPreparationTask(task, options.snapshotId);
  applyCompletedDataPreparationState(state, options);
  if (
    options.offerRuntimeReview !== false
    && state.result.preparation.runtime_contract_review?.required
  ) {
    void offerRuntimeContractReview(state.result, state.snapshotId);
  }
}

function initializeVisualization(result?: WorkflowResult | null) {
  // Registration derivatives may omit every source asset. Prefer the immutable
  // preparation result, and tolerate legacy/incomplete workflow envelopes.
  const source = preparationResult.value || result;
  const logs = wellLogPreviews.value;
  const volumes = source?.visualization_preview?.volumes || [];
  const lines = source?.visualization_preview?.lines2d || [];
  const seismic = Array.isArray(source?.seismic) ? source.seismic : [];
  if (logs.length) {
    selectedWellLogId.value = logs[0].id;
    visibleCurveIds.value = logs[0].curves.map((curve) => curve.id);
  }
  visualizationMode.value = volumes.length || lines.length ? "seismic" : logs.length ? "logs" : "seismic";
  visualizationSourceTaskId.value = "";
  visualizationBaseTaskId.value = taskId.value;
  selectedSeismicAssetIndex.value = 0;
  if (!predictionSeismicPath.value && seismic.length) {
    predictionSeismicPath.value = (
      seismic.find((item) => item.model_compatibility?.[selectedPredictionModelId.value]?.ready)
      || seismic.find((item) => item.trace_count > 0)
      || seismic[0]
    ).path;
  }
}

function resetFaultSegDefaults() {
  if (faultSegSnapshotSeismicSources.value.length === 1) {
    predictionSeismicPath.value = faultSegSnapshotSeismicSources.value[0].path;
  }
  faultSegScope.value = selectedPredictionModelId.value === "faultseg_3d"
    ? "center_block_1"
    : "full_volume";
  predictionDevice.value = "auto";
}

function resetSurfaceSegDefaults() {
  surfaceSegScope.value = "full";
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
  if (!selectedModelRequiresSeismic.value) return;
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
  if (predictionResult.value?.model_id !== selectedPredictionModelId.value) {
    predictionTaskId.value = "";
    predictionResult.value = null;
    visualizationSourceTaskId.value = "";
    window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
  }
  if (isFaultSegModel.value) resetFaultSegDefaults();
  if (isSurfaceSegModel.value) resetSurfaceSegDefaults();
  if (isF3FaciesModel.value) predictionDevice.value = "auto";
  selectCompatiblePredictionSource();
}

function handlePredictionSourceChange() {
  if (isSurfaceSegModel.value) surfaceSegInlineCount.value = "";
  if (isFaultSegModel.value) resetFaultSegDefaults();
}

function applyDemo() {
  if (!demo.value?.available) return;
  const mapping: Record<PathGroupKey, string[]> = {
    seismic: demo.value.seismic_paths,
    survey: demo.value.survey_paths || [],
    logs: demo.value.log_paths,
    wells: demo.value.well_paths,
    timeDepth: demo.value.time_depth_paths || [],
    interpretations: demo.value.interpretation_paths || [],
    auxiliary: demo.value.auxiliary_paths,
  };
  clearSourceContractDraft();
  groups.value.forEach((group) => {
    group.paths = [...mapping[group.key]];
  });
  const contract = demo.value.contract || demo.value.source_contract;
  if (contract) restoreSourceContractFields(contract);
}

function stopPreparationClock() {
  if (preparationClockTimer !== null) {
    window.clearInterval(preparationClockTimer);
    preparationClockTimer = null;
  }
  if (preparationStartedAt.value !== null) preparationClockNow.value = performance.now();
}

function stopFusionClock() {
  if (fusionClockTimer !== null) {
    window.clearInterval(fusionClockTimer);
    fusionClockTimer = null;
  }
  if (fusionStartedAt.value !== null) fusionClockNow.value = performance.now();
}

function startFusionClock(phase: Exclude<FusionActivityPhase, "reconnecting">, createdAt?: string) {
  stopFusionClock();
  const now = performance.now();
  const backendStartedAt = Date.parse(createdAt || "");
  const priorElapsed = Number.isFinite(backendStartedAt)
    ? Math.max(0, Date.now() - backendStartedAt)
    : 0;
  fusionActivityPhase.value = phase;
  fusionStartedAt.value = now - priorElapsed;
  fusionClockNow.value = now;
  fusionEtaBaselineSeconds.value = null;
  fusionEtaComputedAt.value = null;
  fusionEtaSampleProgress.value = 0;
  fusionClockTimer = window.setInterval(() => {
    fusionClockNow.value = performance.now();
  }, 500);
}

function updateFusionActivity(task: BackgroundTask) {
  const phase: Exclude<FusionActivityPhase, "reconnecting"> = task.task_type === "sample_building"
    ? "prepared_view"
    : "registration";
  if (fusionStartedAt.value === null || fusionActivityPhase.value !== phase) {
    startFusionClock(phase, task.created_at);
  }
  fusionClockNow.value = performance.now();
  progress.value = task.progress;
  statusMessage.value = task.message;
  if (task.progress > fusionEtaSampleProgress.value && task.progress < 100) {
    const estimate = estimateProgressRemainingSeconds(fusionElapsedSeconds.value, task.progress);
    if (estimate !== null) {
      fusionEtaBaselineSeconds.value = estimate;
      fusionEtaComputedAt.value = fusionClockNow.value;
      fusionEtaSampleProgress.value = task.progress;
    }
  }
}

function capturePreparationRunTiming(task: BackgroundTask) {
  if (task.task_type !== "data_preparation" || task.status !== "completed") return;
  const finishedAt = task.completed_at || task.updated_at || "";
  if (!task.created_at || !finishedAt) {
    preparationRunTiming.value = null;
    return;
  }
  const explicitDuration = Number(task.preparation_duration_seconds);
  const calculatedDuration = (
    Date.parse(finishedAt) - Date.parse(task.created_at)
  ) / 1000;
  const durationSeconds = Number.isFinite(explicitDuration) && explicitDuration >= 0
    ? explicitDuration
    : Number.isFinite(calculatedDuration) && calculatedDuration >= 0
      ? calculatedDuration
      : null;
  preparationRunTiming.value = {
    taskId: task.task_id,
    startedAt: task.created_at,
    finishedAt,
    durationSeconds,
    finishedSource: task.completed_at ? "completed_at" : "updated_at",
  };
}

function startPreparationClock(createdAt?: string) {
  stopPreparationClock();
  const now = performance.now();
  const backendStartedAt = Date.parse(createdAt || "");
  const priorElapsed = Number.isFinite(backendStartedAt)
    ? Math.max(0, Date.now() - backendStartedAt)
    : 0;
  preparationStartedAt.value = now - priorElapsed;
  preparationReadingStartedAt.value = null;
  preparationClockNow.value = now;
  preparationWorkDone.value = 0;
  preparationWorkTotal.value = 0;
  preparationCurrentItem.value = "";
  preparationCurrentItemSizeBytes.value = null;
  preparationCurrentItemStartedAt.value = null;
  preparationSubworkDone.value = 0;
  preparationSubworkTotal.value = 0;
  preparationWorkUnit.value = null;
  preparationSubworkUnit.value = null;
  preparationEtaBaselineSeconds.value = null;
  preparationEtaComputedAt.value = null;
  preparationEtaSampleSubworkDone.value = 0;
  preparationHistoricalEstimateSeconds.value = null;
  preparationHistoricalEstimateSamples.value = 0;
  preparationHistoricalEstimateConfidence.value = null;
  preparationRunTiming.value = null;
  preparationClockTimer = window.setInterval(() => {
    preparationClockNow.value = performance.now();
  }, 500);
}

function updatePreparationActivity(task: BackgroundTask) {
  const now = performance.now();
  capturePreparationRunTiming(task);
  preparationClockNow.value = now;
  preparationProgress.value = task.progress;
  preparationStatusMessage.value = task.message;
  progress.value = task.progress;
  statusMessage.value = task.message;
  if (task.preparation_estimate && task.preparation_estimate.duration_seconds > 0) {
    preparationHistoricalEstimateSeconds.value = task.preparation_estimate.duration_seconds;
    preparationHistoricalEstimateSamples.value = task.preparation_estimate.samples;
    preparationHistoricalEstimateConfidence.value = task.preparation_estimate.confidence;
  }
  const detail = task.progress_detail;
  if (!detail) return;
  const previousPhase = preparationActivityPhase.value;
  let incomingPhase = previousPhase;
  if ([
    "submitting",
    "validating",
    "cataloging",
    "reading",
    "hashing",
    "summarizing",
    "caching",
    "completed",
    "failed",
  ].includes(String(detail.phase))) {
    incomingPhase = detail.phase as PreparationActivityPhase;
    preparationActivityPhase.value = incomingPhase;
  }
  if (incomingPhase !== previousPhase) {
    preparationReadingStartedAt.value = null;
    preparationCurrentItem.value = "";
    preparationCurrentItemSizeBytes.value = null;
    preparationCurrentItemStartedAt.value = null;
    preparationEtaBaselineSeconds.value = null;
    preparationEtaComputedAt.value = null;
    preparationEtaSampleSubworkDone.value = 0;
    preparationWorkDone.value = 0;
    preparationWorkTotal.value = 0;
    preparationSubworkDone.value = 0;
    preparationSubworkTotal.value = 0;
    preparationWorkUnit.value = null;
    preparationSubworkUnit.value = null;
  }
  if (detail.phase !== "reading" && detail.phase !== "hashing") return;
  const total = Math.max(0, Number(detail.work_total) || 0);
  const done = Math.min(Math.max(0, Number(detail.work_done) || 0), total);
  const subworkTotal = Math.max(0, Number(detail.subwork_total) || 0);
  const subworkDone = Math.min(Math.max(0, Number(detail.subwork_done) || 0), subworkTotal);
  const currentItem = typeof detail.current_item === "string" ? detail.current_item : "";
  if (currentItem !== preparationCurrentItem.value) {
    preparationCurrentItem.value = currentItem;
    preparationCurrentItemSizeBytes.value = null;
    preparationCurrentItemStartedAt.value = null;
    preparationEtaBaselineSeconds.value = null;
    preparationEtaComputedAt.value = null;
    preparationEtaSampleSubworkDone.value = 0;
    preparationSubworkDone.value = 0;
    preparationSubworkTotal.value = 0;
  }
  preparationWorkTotal.value = total;
  preparationWorkDone.value = done;
  preparationCurrentItemSizeBytes.value = (
    typeof detail.current_item_size_bytes === "number"
    && Number.isFinite(detail.current_item_size_bytes)
    && detail.current_item_size_bytes >= 0
  ) ? detail.current_item_size_bytes : null;
  preparationSubworkTotal.value = subworkTotal;
  preparationSubworkDone.value = subworkDone;
  preparationWorkUnit.value = detail.unit === "assets" || detail.unit === "bytes" ? detail.unit : null;
  preparationSubworkUnit.value = detail.subunit === "traces" || detail.subunit === "bytes"
    ? detail.subunit
    : null;
  if (preparationReadingStartedAt.value === null) {
    const backendStartedAt = typeof detail.started_at === "string" ? Date.parse(detail.started_at) : Number.NaN;
    const priorElapsed = Number.isFinite(backendStartedAt)
      ? Math.max(0, Date.now() - backendStartedAt)
      : 0;
    preparationReadingStartedAt.value = now - priorElapsed;
  }
  if (preparationCurrentItemStartedAt.value === null && currentItem) {
    const backendCurrentStartedAt = typeof detail.current_started_at === "string"
      ? Date.parse(detail.current_started_at)
      : Number.NaN;
    const priorCurrentElapsed = Number.isFinite(backendCurrentStartedAt)
      ? Math.max(0, Date.now() - backendCurrentStartedAt)
      : 0;
    preparationCurrentItemStartedAt.value = now - priorCurrentElapsed;
  }
  if (
    subworkDone > preparationEtaSampleSubworkDone.value
    && subworkDone > 0
    && subworkTotal > subworkDone
    && preparationCurrentItemStartedAt.value !== null
  ) {
    const currentElapsedSeconds = Math.max(1, (now - preparationCurrentItemStartedAt.value) / 1000);
    const estimatedRemaining = currentElapsedSeconds / subworkDone * (subworkTotal - subworkDone);
    if (Number.isFinite(estimatedRemaining) && estimatedRemaining > 0) {
      preparationEtaBaselineSeconds.value = Math.max(10, estimatedRemaining);
      preparationEtaComputedAt.value = now;
      preparationEtaSampleSubworkDone.value = subworkDone;
    }
  }
}

interface WaitForTaskOptions {
  onProgress?: (task: BackgroundTask) => void;
  onRetry?: (attempt: number) => void;
  persistent?: boolean;
}

type WorkflowTaskResult = WorkflowResult | HorizontalRegistrationTaskResult;

function isRetryableTaskStatusError(error: unknown): boolean {
  if (!(error instanceof ApiRequestError)) return false;
  if (error.kind === "network" || error.kind === "timeout") return true;
  if (error.kind !== "http") return false;
  return error.status === 408
    || error.status === 425
    || error.status === 429
    || (error.status >= 500 && error.status <= 599);
}

async function waitForTask(id: string, options: WaitForTaskOptions = {}): Promise<WorkflowTaskResult> {
  const deadline = Date.now() + TASK_POLL_TIMEOUT_MS;
  let consecutiveFailures = 0;
  while (!componentUnmounted && (options.persistent || Date.now() < deadline)) {
    let task: BackgroundTask;
    try {
      task = await getTask(id);
      consecutiveFailures = 0;
      predictionConnectionState.value = "online";
      predictionLastHeartbeatAt.value = Date.now();
    } catch (error) {
      if (error instanceof ApiRequestError && error.status === 404) {
        throw new ApiRequestError(404, "后台任务记录不存在，无法继续恢复读取进度");
      }
      if (!isRetryableTaskStatusError(error)) throw error;
      consecutiveFailures += 1;
      if (consecutiveFailures > 4 && !options.persistent) {
        const reason = error instanceof Error ? error.message : "未知连接错误";
        throw new Error(`任务状态连接连续中断，后台任务可能仍在运行：${reason}`);
      }
      options.onRetry?.(consecutiveFailures);
      if (!options.onRetry) statusMessage.value = `任务状态连接中断，正在重试（${consecutiveFailures}/4）`;
      await new Promise((resolve) => window.setTimeout(
        resolve,
        Math.min(options.persistent ? 10_000 : 3_000, 750 * consecutiveFailures),
      ));
      continue;
    }
    if (options.onProgress) options.onProgress(task);
    else {
      progress.value = task.progress;
      statusMessage.value = task.message;
    }
    if (task.status === "completed") {
      if (isWorkflowResult(task.result) || isHorizontalRegistrationTaskResult(task.result)) return task.result;
      throw new Error("任务已完成，但后端没有返回可识别的工作流结果");
    }
    if (task.status === "failed") throw new Error(task.error?.message || "后端任务失败");
    if (task.status === "cancelled") throw new Error(task.error?.message || "后台任务已取消");
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
  if (componentUnmounted) throw new Error("页面已关闭，已停止轮询任务状态");
  throw new Error("任务运行超过30分钟，请检查后端日志或稍后重新打开任务");
}

async function finalizeDataPreparationTask(
  completedTaskId: string,
  result: WorkflowResult,
  payload: Record<string, unknown> | Partial<DataPathsPayload>,
  shouldAutofill: boolean,
) {
  applyCompletedDataPreparationState({
    taskId: completedTaskId,
    snapshotId: result.data_snapshot?.snapshot_id || completedTaskId,
    projectId: result.data_snapshot?.project_id || activeProjectId.value,
    message: "数据准备已完成",
    request: { ...payload } as Partial<DataPathsPayload>,
    result,
  }, { showPipeline: false, persistActiveSnapshot: true });
  if (shouldAutofill) {
    preparationActivityPhase.value = "autofill";
    preparationProgress.value = 100;
    preparationStatusMessage.value = "文件读取与本地校验完成，正在执行 Kimi 受控修复";
    try {
      const autofill = await autofillPreparationIssues(completedTaskId);
      Object.assign(result.preparation, autofill.preparation);
      autoPopulateSafeSeismicVerticalContract();
      try {
        savePathConfig(commonPayload());
      } catch {
        appendRestorationWarning("浏览器未能保存Kimi判断后的数据合同草稿");
      }
      statusMessage.value = autofill.survey_input_required_count
        ? `Kimi 受控修复与规则复检已处理 ${autofill.autofilled_count} 项；${autofill.survey_input_required_count} 项缺少原始证据，已合并为工区级补充`
        : `Kimi 受控修复与规则复检已处理 ${autofill.autofilled_count} 项`;
      preparationStatusMessage.value = statusMessage.value;
    } catch (autofillError) {
      const reason = autofillError instanceof Error ? autofillError.message : "未知错误";
      statusMessage.value = `数据准备已完成；Kimi 受控自动修复暂未完成：${reason}`;
      preparationStatusMessage.value = statusMessage.value;
    }
  }
  preparationActivityPhase.value = "completed";
  preparationProgress.value = 100;
  preparationStatusMessage.value = statusMessage.value;
  const firstAttentionStage = result.preparation.issues.find(
    (issue) => issueNeedsCurrentAttention(issue),
  )?.stage;
  preparationScreen.value = "pipeline";
  issueFilter.value = firstAttentionStage || "全部";
  await nextTick();
  window.scrollTo({ top: 0, behavior: "auto" });
  await offerRuntimeContractReview(result, dataSnapshotTaskId.value || completedTaskId);
}

async function reattachDataPreparationTask(task: BackgroundTask) {
  if (preparationRunning.value) return;
  let payload = (task.request || {}) as Partial<DataPathsPayload>;
  let payloadHydrated = Boolean(task.request);
  preparationRunning.value = true;
  try {
    preparationProgress.value = task.progress;
    preparationStatusMessage.value = "正在重新连接后台数据读取任务";
    preparationActivityPhase.value = "reconnecting";
    taskId.value = task.task_id;
    dataSnapshotTaskId.value = task.task_id;
    sampleBuildingTaskId.value = "";
    if (payloadHydrated) {
      applyPathPayload(payload);
      try {
        savePathConfig(payload as DataPathsPayload);
      } catch {
        appendRestorationWarning("浏览器未能保存正在运行任务的输入合同");
      }
    }
    try {
      window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, task.task_id);
    } catch {
      appendRestorationWarning("浏览器未能保存正在运行的后台任务引用");
    }
    startPreparationClock(task.created_at);
    updatePreparationActivity(task);
    const result = await waitForTask(task.task_id, {
      persistent: true,
      onProgress: (task) => {
        if (!payloadHydrated && task.request) {
          payload = task.request as Partial<DataPathsPayload>;
          payloadHydrated = true;
          applyPathPayload(payload);
          try {
            savePathConfig(payload as DataPathsPayload);
          } catch {
            appendRestorationWarning("浏览器未能保存后台补回的输入合同");
          }
        }
        updatePreparationActivity(task);
      },
      onRetry: (attempt) => {
        preparationActivityPhase.value = "reconnecting";
        preparationStatusMessage.value = `后台任务仍在运行，正在恢复状态连接（第 ${attempt} 次）`;
      },
    });
    if (!isWorkflowResult(result)) throw new Error("数据准备任务返回了错误的结果类型");
    await finalizeDataPreparationTask(
      task.task_id,
      result,
      payload,
      payload.use_llm_fallback === true && capabilities.value?.llm.available === true,
    );
  } catch (error) {
    if (!componentUnmounted) {
      preparationActivityPhase.value = "failed";
      errorMessage.value = error instanceof Error ? error.message : "无法恢复数据准备任务";
      preparationStatusMessage.value = errorMessage.value;
      if (error instanceof ApiRequestError && error.status === 404) {
        window.sessionStorage.removeItem(LAST_TASK_STORAGE_KEY);
      }
    }
  } finally {
    stopPreparationClock();
    preparationRunning.value = false;
  }
}

type ReconnectableWorkflowTaskType = "well_tie" | "horizontal_registration" | "sample_building";

function isReconnectableWorkflowTask(task: BackgroundTask): task is BackgroundTask & {
  task_type: ReconnectableWorkflowTaskType;
} {
  return task.task_type === "well_tie"
    || task.task_type === "horizontal_registration"
    || task.task_type === "sample_building";
}

function taskRequestString(task: BackgroundTask, key: string): string {
  const value = task.request?.[key];
  return typeof value === "string" ? value : "";
}

function workflowTaskSourceSnapshotId(task: BackgroundTask): string {
  if (task.snapshot_id) return task.snapshot_id;
  const requested = taskRequestString(task, "source_snapshot_id");
  if (requested) return requested;
  if (isHorizontalRegistrationTaskResult(task.result)) {
    return task.result.horizontal_registration.source_snapshot_id;
  }
  if (isWorkflowResult(task.result)) {
    return task.result.registration?.source_snapshot_id
      || task.result.source_snapshot_id
      || task.result.data_snapshot?.snapshot_id
      || "";
  }
  return "";
}

function installCompletedWorkflowTask(
  task: BackgroundTask,
  result: WorkflowTaskResult,
  options: { persistActiveSnapshot?: boolean } = {},
) {
  taskId.value = task.task_id;
  progress.value = 100;
  statusMessage.value = task.message || "后台任务已完成";
  const sourceSnapshotId = workflowTaskSourceSnapshotId({ ...task, result });
  if (sourceSnapshotId) {
    dataSnapshotTaskId.value = sourceSnapshotId;
    rememberSourceSnapshot(sourceSnapshotId, task.project_id || activeProjectId.value, options);
  }

  if (task.task_type === "horizontal_registration") {
    if (!isHorizontalRegistrationTaskResult(result)) throw new Error("水平配准任务返回了错误的结果类型");
    horizontalRegistrationTaskId.value = task.task_id;
    horizontalRegistrationResult.value = result;
    registrationTaskId.value = "";
    registrationResult.value = null;
    sampleBuildingTaskId.value = "";
    sampleResult.value = null;
    initializeVisualization();
    return;
  }

  if (!isWorkflowResult(result)) throw new Error("后台任务返回了错误的结果类型");
  if (task.task_type === "well_tie") {
    registrationTaskId.value = task.task_id;
    registrationResult.value = result;
    horizontalRegistrationTaskId.value = "";
    horizontalRegistrationResult.value = null;
    sampleBuildingTaskId.value = "";
    sampleResult.value = null;
  } else if (task.task_type === "sample_building") {
    sampleBuildingTaskId.value = task.task_id;
    sampleResult.value = result;
    registrationTaskId.value = result.registration_task_id
      || taskRequestString(task, "registration_task_id")
      || registrationTaskId.value;
    horizontalRegistrationTaskId.value = "";
    horizontalRegistrationResult.value = null;
  } else {
    throw new Error("当前任务不是可恢复的井震工作流任务");
  }
  initializeVisualization(result);
}

async function reattachWorkflowTask(task: BackgroundTask & { task_type: ReconnectableWorkflowTaskType }) {
  if (registrationRunning.value || sampleRunning.value) return;
  const sourceSnapshotId = workflowTaskSourceSnapshotId(task);
  if (sourceSnapshotId) {
    dataSnapshotTaskId.value = sourceSnapshotId;
    rememberSourceSnapshot(sourceSnapshotId, task.project_id || activeProjectId.value);
    if (!preparationResult.value) {
      try {
        await restorePreparationForSnapshot(sourceSnapshotId);
      } catch (error) {
        const reason = error instanceof Error ? error.message : "未知错误";
        appendRestorationWarning(`源数据状态恢复不完整，后台任务仍将继续连接：${reason}`);
      }
    }
  }

  taskId.value = task.task_id;
  progress.value = task.progress;
  statusMessage.value = task.message || "正在重新连接后台任务";
  if (task.task_type === "sample_building") {
    sampleRunning.value = true;
    sampleBuildingTaskId.value = task.task_id;
    sampleResult.value = null;
    const relatedRegistrationTaskId = taskRequestString(task, "registration_task_id");
    if (relatedRegistrationTaskId && !registrationResult.value) {
      try {
        const related = await getTask(relatedRegistrationTaskId);
        if (related.status === "completed" && isWorkflowResult(related.result)) {
          registrationTaskId.value = related.task_id;
          registrationResult.value = related.result;
        }
      } catch (error) {
        const reason = error instanceof Error ? error.message : "未知错误";
        appendRestorationWarning(`PreparedView 的标定父任务恢复不完整：${reason}`);
      }
    }
  } else {
    registrationRunning.value = true;
    sampleBuildingTaskId.value = "";
    sampleResult.value = null;
    if (task.task_type === "horizontal_registration") {
      horizontalRegistrationTaskId.value = task.task_id;
      horizontalRegistrationResult.value = null;
      registrationTaskId.value = "";
      registrationResult.value = null;
    } else {
      registrationTaskId.value = task.task_id;
      registrationResult.value = null;
      horizontalRegistrationTaskId.value = "";
      horizontalRegistrationResult.value = null;
    }
  }
  startFusionClock(task.task_type === "sample_building" ? "prepared_view" : "registration", task.created_at);
  try {
    window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, task.task_id);
  } catch {
    appendRestorationWarning("浏览器未能保存正在运行的井震任务引用");
  }

  try {
    const result = await waitForTask(task.task_id, {
      persistent: true,
      onProgress: updateFusionActivity,
      onRetry: (attempt) => {
        fusionActivityPhase.value = "reconnecting";
        statusMessage.value = `融合任务仍在运行，正在恢复状态连接（第 ${attempt} 次）`;
      },
    });
    installCompletedWorkflowTask({
      ...task,
      progress: 100,
      message: statusMessage.value || "后台任务已完成",
    }, result);
    if (task.task_type === "sample_building") {
      await nextTick();
      openPostFusionInferenceDestination();
    }
  } catch (error) {
    if (!componentUnmounted) {
      errorMessage.value = error instanceof Error ? error.message : "后台任务恢复失败";
      statusMessage.value = errorMessage.value;
    }
  } finally {
    stopFusionClock();
    if (task.task_type === "sample_building") sampleRunning.value = false;
    else registrationRunning.value = false;
  }
}

async function recoverRememberedTaskReference(rememberedTaskId: string) {
  preparationRunning.value = true;
  let attempt = 0;
  try {
    preparationProgress.value = 0;
    preparationStatusMessage.value = "暂时无法读取后台任务，正在重新连接";
    preparationActivityPhase.value = "reconnecting";
    startPreparationClock();
    while (!componentUnmounted) {
      let recoveredTask: BackgroundTask;
      try {
        recoveredTask = await getTask(rememberedTaskId);
      } catch (error) {
        if (error instanceof ApiRequestError && error.status === 404) {
          window.sessionStorage.removeItem(LAST_TASK_STORAGE_KEY);
          throw new Error("之前的数据准备任务记录已不存在，已解除输入锁定");
        }
        if (!isRetryableTaskStatusError(error)) throw error;
        attempt += 1;
        preparationStatusMessage.value = `后台任务仍可能运行，正在恢复连接（第 ${attempt} 次）`;
        await new Promise((resolve) => window.setTimeout(resolve, Math.min(10_000, 750 * attempt)));
        continue;
      }
      stopPreparationClock();
      preparationRunning.value = false;
      if (recoveredTask.task_type === "data_preparation") {
        if (recoveredTask.status === "completed" && recoveredTask.result && "summary" in recoveredTask.result) {
          restoreCompletedDataPreparationTask(recoveredTask);
          await nextTick();
          window.scrollTo({ top: 0, behavior: "auto" });
        } else if (recoveredTask.status === "queued" || recoveredTask.status === "running") {
          await reattachDataPreparationTask(recoveredTask);
        } else {
          errorMessage.value = recoveredTask.error?.message
            || recoveredTask.message
            || "之前的数据准备任务无法恢复，请重新运行";
          window.sessionStorage.removeItem(LAST_TASK_STORAGE_KEY);
        }
      } else if (
        isReconnectableWorkflowTask(recoveredTask)
        && (recoveredTask.status === "queued" || recoveredTask.status === "running")
      ) {
        await reattachWorkflowTask(recoveredTask);
      } else if (
        isReconnectableWorkflowTask(recoveredTask)
        && recoveredTask.status === "completed"
        && (isWorkflowResult(recoveredTask.result) || isHorizontalRegistrationTaskResult(recoveredTask.result))
      ) {
        const sourceSnapshotId = workflowTaskSourceSnapshotId(recoveredTask);
        if (!preparationResult.value && sourceSnapshotId) {
          try {
            await restorePreparationForSnapshot(sourceSnapshotId);
          } catch (error) {
            const reason = error instanceof Error ? error.message : "未知错误";
            appendRestorationWarning(`源数据状态恢复不完整，已完成任务仍将继续落地：${reason}`);
          }
        }
        installCompletedWorkflowTask(recoveredTask, recoveredTask.result);
      } else {
        await restoreLatestDurableWorkflow();
      }
      return;
    }
  } catch (error) {
    if (!componentUnmounted) {
      preparationActivityPhase.value = "failed";
      errorMessage.value = error instanceof Error ? error.message : "无法恢复后台任务";
      preparationStatusMessage.value = errorMessage.value;
    }
  } finally {
    stopPreparationClock();
    preparationRunning.value = false;
  }
}

async function runDataPreparation() {
  errorMessage.value = "";
  registrationPreparationRequired.value = false;
  if (!registeredCount.value) {
    errorMessage.value = "至少需要登记一个文件或目录。";
    return;
  }
  preparationRunning.value = true;
  preparationProgress.value = 0;
  preparationStatusMessage.value = "正在提交数据准备任务";
  preparationActivityPhase.value = "submitting";
  startPreparationClock();
  progress.value = 0;
  statusMessage.value = "正在提交数据准备任务";
  try {
    const payload = commonPayload();
    try {
      window.sessionStorage.removeItem(FRESH_SESSION_STORAGE_KEY);
      savePathConfig(payload);
    } catch {
      appendRestorationWarning("浏览器未能保存本次输入；后台任务仍会继续，当前页面可正常跟踪");
    }
    const created = await createDataPreparation(payload);
    taskId.value = created.task_id;
    dataSnapshotTaskId.value = created.task_id;
    sampleBuildingTaskId.value = "";
    try {
      window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, created.task_id);
    } catch {
      appendRestorationWarning("后台任务已创建，但浏览器未能保存任务引用；请勿重复提交本次任务");
    }
    const result = await waitForTask(created.task_id, {
      persistent: true,
      onProgress: updatePreparationActivity,
      onRetry: (attempt) => {
        preparationActivityPhase.value = "reconnecting";
        preparationStatusMessage.value = `后台任务仍在运行，正在恢复状态连接（第 ${attempt} 次）`;
      },
    });
    if (!isWorkflowResult(result)) throw new Error("数据准备任务返回了错误的结果类型");
    await finalizeDataPreparationTask(created.task_id, result, payload, useLlmFallback.value);
  } catch (error) {
    preparationActivityPhase.value = "failed";
    errorMessage.value = error instanceof Error ? error.message : "数据准备失败";
  } finally {
    stopPreparationClock();
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

async function applyAllSafeRecommendations() {
  if (!taskId.value || batchApplyingRecommendations.value) return;
  batchApplyingRecommendations.value = true;
  errorMessage.value = "";
  try {
    const updated = await autofillPreparationIssues(taskId.value);
    if (preparation.value) Object.assign(preparation.value, updated.preparation);
    statusMessage.value = updated.survey_input_required_count
      ? `已自动补全 ${updated.autofilled_count} 项；${updated.survey_input_required_count} 项缺少可推断证据，保留为一次集中补充`
      : `已自动补全并通过规则校验 ${updated.autofilled_count} 项`;
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "一键处理失败";
  } finally {
    batchApplyingRecommendations.value = false;
  }
}

function openIssueReviewDrawer() {
  void nextTick().then(() => {
    const drawer = document.getElementById("issue-review");
    if (drawer instanceof HTMLDetailsElement) drawer.open = true;
    drawer?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function focusIssueReview() {
  const stage = preparation.value?.issues.find(
    (issue) => issueRequiredForCurrentRun(issue) && (
      issue.blocking
      || issue.confirmation_status === "待人工确认"
      || issue.confirmation_status === "需一次集中补充"
    ),
  )?.stage;
  if (stage) issueFilter.value = stage;
  openIssueReviewDrawer();
}

function showAllIssues() {
  issueFilter.value = "全部";
  openIssueReviewDrawer();
}

async function runWellTie(): Promise<CalibrationOutcome> {
  errorMessage.value = "";
  registrationPreparationRequired.value = false;
  if (fusionWorkflowMutationRunning.value) {
    errorMessage.value = "井震标定、融合视图、轨迹候选与预测任务需依次运行；请等待当前任务结束。";
    return { kind: "failed", detail: errorMessage.value };
  }
  let sourceSnapshotId = dataSnapshotTaskId.value
    || preparationResult.value?.data_snapshot?.snapshot_id
    || "";
  let submittedRegistrationTaskId = "";
  let submittedHorizontalRegistrationTaskId = "";
  if (!sourceSnapshotId) {
    errorMessage.value = "请先完成数据准备，再启动井震标定。";
    return { kind: "failed", detail: errorMessage.value };
  }
  registrationRunning.value = true;
  startFusionClock("registration");
  clearGeoPathCandidateState();
  progress.value = 0;
  statusMessage.value = acousticFineCalibrationCandidateReady.value
    ? `已识别 ${acousticFineCalibrationCandidateCount.value} 口井的 DT/AC 声波候选，正在核验并启动精细标定`
    : "正在自动评估封存快照中的井震标定证据";
  try {
    sourceSnapshotId = await adoptNewestSameSourceSnapshot(sourceSnapshotId);
    let preflight;
    try {
      preflight = await preflightRegistration(sourceSnapshotId);
    } catch (error) {
      const failureKind = classifyRegistrationPreflightFailure(error);
      if (failureKind === "blocked_integrity") {
        const detail = error instanceof Error ? error.message : "数据快照完整性校验失败";
        statusMessage.value = "SourceSnapshot 已阻止复用，请重新准备数据";
        registrationPreparationRequired.value = true;
        selectView("preparation");
        showPreparationInput();
        errorMessage.value = "数据解析或坐标合同已经升级，旧快照不能直接用于标定。请重新执行数据准备生成新快照；原始文件未丢失。";
        return { kind: "blocked_integrity", detail };
      }
      if (failureKind === "needs_preparation") {
        const detail = error instanceof Error ? error.message : "当前快照的源数据质量不足";
        statusMessage.value = "当前快照需要补充或重新解析源数据后再标定";
        registrationPreparationRequired.value = true;
        errorMessage.value = `当前快照暂不能进入井震标定：${detail}。原始文件路径已保留，可返回数据准备重新识别并封存。`;
        return { kind: "failed", detail };
      }
      if (failureKind !== "horizontal_only") throw error;

      statusMessage.value = "精细标定的最低证据不足，正在执行仅几何空间配准";
      const created = await createHorizontalRegistration({ source_snapshot_id: sourceSnapshotId });
      submittedHorizontalRegistrationTaskId = created.task_id;
      horizontalRegistrationTaskId.value = created.task_id;
      registrationTaskId.value = "";
      registrationResult.value = null;
      sampleBuildingTaskId.value = "";
      sampleResult.value = null;
      taskId.value = created.task_id;
      window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, created.task_id);
      const completed = await waitForTask(created.task_id, {
        persistent: true,
        onProgress: updateFusionActivity,
        onRetry: (attempt) => {
          fusionActivityPhase.value = "reconnecting";
          statusMessage.value = `标定任务仍在运行，正在恢复状态连接（第 ${attempt} 次）`;
        },
      });
      if (!isHorizontalRegistrationTaskResult(completed)) {
        throw new Error("水平配准任务返回了错误的结果类型");
      }
      horizontalRegistrationResult.value = completed;
      rememberSourceSnapshot(sourceSnapshotId);
      initializeVisualization();
      const horizontal = horizontalRegistrationResult.value.horizontal_registration;
      const summary = horizontal?.summary;
      statusMessage.value = [
        `仅几何空间配准已完成：${summary?.fully_covered_well_count ?? 0}/${summary?.well_count ?? 0} 口井完全覆盖，${summary?.covered_station_count ?? 0}/${summary?.station_count ?? 0} 个轨迹站命中地震网格。`,
        "未形成可审计的 TWT 或 PreparedView；预测入口保持锁定，请完善标定证据后重新融合。",
      ].join("");
      return { kind: "horizontal_only", taskId: created.task_id, result: horizontalRegistrationResult.value };
    }

    const payload = {
      ...preflight.effective_request,
      source_snapshot_id: preflight.derived_snapshot_id,
      registration_task_id: undefined,
    };
    dataSnapshotTaskId.value = preflight.derived_snapshot_id;
    rememberSourceSnapshot(preflight.derived_snapshot_id);
    applyPathPayload(preflight.effective_request);
    rememberSealedSourceContract(preflight.effective_request, preparationResult.value);
    savePathConfig(payload);
    const created = await createRegistration(payload);
    submittedRegistrationTaskId = created.task_id;
    registrationTaskId.value = created.task_id;
    horizontalRegistrationTaskId.value = "";
    horizontalRegistrationResult.value = null;
    sampleBuildingTaskId.value = "";
    sampleResult.value = null;
    taskId.value = created.task_id;
    window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, created.task_id);
    const completed = await waitForTask(created.task_id, {
      persistent: true,
      onProgress: updateFusionActivity,
      onRetry: (attempt) => {
        fusionActivityPhase.value = "reconnecting";
        statusMessage.value = `标定任务仍在运行，正在恢复状态连接（第 ${attempt} 次）`;
      },
    });
    if (!isWorkflowResult(completed)) throw new Error("井震标定任务返回了错误的结果类型");
    registrationResult.value = completed;
    initializeVisualization(registrationResult.value);
    const registration = registrationResult.value.registration;
    const registered = registration?.registered_well_count || 0;
    const timeAxisPresentation = registrationTimeAxisPresentation.value;
    if (!registered) {
      const blockedByDatum = Number(registration?.method_counts?.blocked_vertical_datum || 0);
      const p13Detail = p13RegistrationPresentation.value?.detail;
      const fallbackDetail = blockedByDatum
        ? `${blockedByDatum} 口井缺少可配对的真实轨迹或工区垂向基准；系统没有把 MD 假设成 TVD。`
        : timeAxisPresentation.provenTwt
          ? "物理回退也没有生成可用的 TWT 时间主轨。"
          : "物理回退也没有生成可用的原生 SEG-Y 时间候选。";
      errorMessage.value = [p13Detail, fallbackDetail].filter(Boolean).join(" ");
      return { kind: "failed", detail: errorMessage.value };
    }
    const blockedByDatum = Number(registration?.method_counts?.blocked_vertical_datum || 0);
    const primaryTrack = p13RegistrationPresentation.value?.primaryTrackLabel || "合法回退";
    const p13Detail = p13RegistrationPresentation.value?.detail;
    const blockedDetail = blockedByDatum
      ? `另有 ${blockedByDatum} 口因轨迹/基准证据不足被安全隔离。`
      : "";
    statusMessage.value = [
      `已生成 ${registered} 口井的${timeAxisPresentation.completionLabel}；${timeAxisPresentation.currentLabel}：${primaryTrack}。`,
      timeAxisPresentation.limitation,
      p13Detail,
      blockedDetail,
    ].filter(Boolean).join(" ");
    if (
      registration?.can_build_multimodal_view !== true
      || Number(
        registration?.downstream_fusion_ready_well_count
        ?? registration?.fusion_ready_well_count
        ?? 0,
      ) <= 0
    ) {
      statusMessage.value = [
        "井震精细标定已完成，但没有通过全部物理门的 fusion-ready 井。",
        "平台不会构建或宣称 PreparedView，预测入口保持锁定；请修复物理或轨迹合同后重新融合。",
      ].join("");
      return { kind: "fusion_blocked", taskId: created.task_id, result: registrationResult.value };
    }
    return { kind: "formal", taskId: created.task_id, result: registrationResult.value };
  } catch (error) {
    if (
      submittedRegistrationTaskId
      && registrationTaskId.value === submittedRegistrationTaskId
    ) {
      registrationTaskId.value = "";
      registrationResult.value = null;
      sampleBuildingTaskId.value = "";
      sampleResult.value = null;
    }
    if (
      submittedHorizontalRegistrationTaskId
      && horizontalRegistrationTaskId.value === submittedHorizontalRegistrationTaskId
    ) {
      horizontalRegistrationTaskId.value = "";
      horizontalRegistrationResult.value = null;
    }
    const detail = error instanceof Error ? error.message : "井震标定失败";
    const failureKind = classifyRegistrationPreflightFailure(error);
    if (
      failureKind === "blocked_integrity"
      || (
        detail.includes("source data snapshot integrity verification failed")
        || detail.includes("source snapshot semantic contract changed")
      )
    ) {
      statusMessage.value = "请用当前规则重新读取并封存数据快照";
      registrationPreparationRequired.value = true;
      selectView("preparation");
      showPreparationInput();
      errorMessage.value = "数据解析或坐标合同已经升级，旧快照仍按原规则封存，不能直接重新解释。请重新执行一次数据准备生成新快照；这不表示原始文件丢失。";
    } else if (failureKind === "needs_preparation") {
      statusMessage.value = "当前快照需要补充或重新解析源数据后再标定";
      registrationPreparationRequired.value = true;
      errorMessage.value = `当前快照暂不能进入井震标定：${detail}。原始文件路径已保留，可返回数据准备重新识别并封存。`;
    } else {
      errorMessage.value = detail;
    }
    return {
      kind: failureKind === "blocked_integrity"
        || detail.includes("source data snapshot integrity verification failed")
        || detail.includes("source snapshot semantic contract changed")
        ? "blocked_integrity"
        : "failed",
      detail,
    };
  } finally {
    stopFusionClock();
    registrationRunning.value = false;
  }
}

async function runSampleBuilding() {
  errorMessage.value = "";
  if (fusionWorkflowMutationRunning.value) {
    errorMessage.value = "井震标定、融合视图、轨迹候选与预测任务需依次运行；请等待当前任务结束。";
    return;
  }
  if (!dataSnapshotTaskId.value) {
    errorMessage.value = "请先完成数据准备，再启动井震标定。";
    return;
  }
  if (formalRegistrationFusionBlocked.value) {
    errorMessage.value = "精细标定没有形成 fusion-ready 消费产品，不能构建 PreparedView。预测入口保持锁定；请修复当前快照的物理或轨迹合同后重新准备。";
    return;
  }
  if (!registrationTaskId.value) {
    const outcome = await runWellTie();
    if (outcome.kind !== "formal") {
      if (outcome.kind === "horizontal_only") {
        statusMessage.value = "仅完成几何空间配准，不能构建 TWT 融合视图；预测入口保持锁定。";
      }
      return;
    }
  }
  sampleRunning.value = true;
  startFusionClock("prepared_view");
  progress.value = 0;
  statusMessage.value = "正在提交井震空间对齐任务";
  try {
    const payload = {
      ...commonPayload(),
      source_snapshot_id: dataSnapshotTaskId.value,
      registration_task_id: registrationTaskId.value,
    };
    savePathConfig(payload);
    const created = await createSampleBuilding(payload);
    sampleBuildingTaskId.value = created.task_id;
    taskId.value = created.task_id;
    window.sessionStorage.setItem(LAST_TASK_STORAGE_KEY, created.task_id);
    const completed = await waitForTask(created.task_id, {
      persistent: true,
      onProgress: updateFusionActivity,
      onRetry: (attempt) => {
        fusionActivityPhase.value = "reconnecting";
        statusMessage.value = `融合视图任务仍在运行，正在恢复状态连接（第 ${attempt} 次）`;
      },
    });
    if (!isWorkflowResult(completed)) throw new Error("样本构建任务返回了错误的结果类型");
    sampleResult.value = completed;
    initializeVisualization(sampleResult.value);
    await nextTick();
    openPostFusionInferenceDestination();
  } catch (error) {
    sampleBuildingTaskId.value = "";
    sampleResult.value = null;
    errorMessage.value = error instanceof Error ? error.message : "样本构建失败";
  } finally {
    stopFusionClock();
    sampleRunning.value = false;
  }
}

async function startDefaultWellSeismicWorkflow() {
  if (!sealedWellSeismicWorkflowReady.value) {
    selectView("preparation");
    showPreparationInput();
    errorMessage.value = "默认井震流程需要同一封存快照中的三维地震与可用井/测井资产。";
    return;
  }
  selectView("samples");
  if (horizontalRegistrationTaskId.value && !registrationTaskId.value) {
    statusMessage.value = "当前快照仅完成空间 QC，未形成融合资格；预测入口保持锁定，请完善数据合同后重新融合。";
    return;
  }
  if (formalRegistrationFusionBlocked.value) {
    statusMessage.value = "精细标定已完成但没有 fusion-ready 消费产品；预测入口保持锁定，请修复物理或轨迹合同后重新融合。";
    return;
  }
  if (!registrationTaskId.value) {
    const outcome = await runWellTie();
    if (outcome.kind !== "formal") return;
  }
  if (!preparedViewReady.value) await runSampleBuilding();
  if (preparedViewReady.value) {
    statusMessage.value = "融合视图已就绪，可以进入预测工作台。";
    showPreparationFusion();
  }
}

function selectablePredictionModelId(taskId: string, modelId: string): string {
  if (taskId !== "well_property" || !isWellPropertyCompletionModelId(modelId)) return modelId;
  const task = predictionTaskDefinitions.value.find((item) => item.id === taskId);
  return task?.runnable_model_ids[0] || "";
}

function isLayerPulsePredictionTask(task: BackgroundTask): boolean {
  if (task.task_type !== "model_prediction") return false;
  const requestMatches = taskRequestString(task, "task_id") === LAYER_PULSE_TASK_ID
    && taskRequestString(task, "model_id") === LAYER_PULSE_MODEL_ID;
  const prediction = task.result && "prediction" in task.result
    ? task.result.prediction
    : null;
  return requestMatches || Boolean(
    prediction?.task_id === LAYER_PULSE_TASK_ID
    && prediction.model_id === LAYER_PULSE_MODEL_ID,
  );
}

function layerPulseSourceTaskId(task: BackgroundTask): string {
  if (task.result && "prediction" in task.result && task.result.source_task_id) {
    return task.result.source_task_id;
  }
  return taskRequestString(task, "source_task_id") || task.snapshot_id || "";
}

function availableLayerPulseOutputKeys(prediction: PredictionResult): string[] {
  if (!Array.isArray(prediction.task_catalog)) return [];
  return prediction.task_catalog.flatMap((entry) => {
    const artifactKey = typeof entry.artifact_key === "string" ? entry.artifact_key : "";
    return entry.output_key
      && artifactKey
      && prediction.outputs[artifactKey]
      && entry.finite !== false
      ? [entry.output_key]
      : [];
  });
}

async function ensureLayerPulseSourceSnapshot(sourceSnapshotId: string) {
  if (!sourceSnapshotId || (
    dataSnapshotTaskId.value === sourceSnapshotId
    && preparationResult.value
  )) return;
  try {
    await restorePreparationForSnapshot(sourceSnapshotId, {
      resetDownstream: false,
      offerRuntimeReview: false,
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : "未知错误";
    appendRestorationWarning(`LayerPulse 来源快照 ${sourceSnapshotId.slice(0, 8)} 恢复不完整：${reason}`);
  }
}

function installCompletedLayerPulseTask(task: BackgroundTask, result: PredictionTaskResult) {
  const prediction = result.prediction;
  if (
    prediction.task_id !== LAYER_PULSE_TASK_ID
    || prediction.model_id !== LAYER_PULSE_MODEL_ID
  ) throw new Error("后台任务不是当前 LayerPulse 单 checkpoint 结果");
  const sourceSnapshotId = result.source_task_id || layerPulseSourceTaskId(task);
  const availableOutputKeys = availableLayerPulseOutputKeys(prediction);
  layerPulseExecutionTaskId.value = task.task_id;
  layerPulseSourceSnapshotId.value = sourceSnapshotId;
  layerPulseResult.value = prediction;
  layerPulseTaskState.value = {
    status: "completed",
    taskId: task.task_id,
    progress: 100,
    message: task.message || "LayerPulse 11 项统一解释已完成",
    availableOutputKeys,
  };
  if (
    availableOutputKeys.length
    && !availableOutputKeys.includes(layerPulseSelectedOutputKey.value)
  ) layerPulseSelectedOutputKey.value = availableOutputKeys[0];
  layerPulseCanvasMode.value = availableOutputKeys.length ? "result" : "base";
}

function installTerminalLayerPulseTask(task: BackgroundTask) {
  const cancelled = task.status === "cancelled" || task.status === "superseded";
  const message = task.error?.message
    || task.message
    || (cancelled ? "LayerPulse 推理已取消" : "LayerPulse 推理失败");
  layerPulseExecutionTaskId.value = task.task_id;
  layerPulseSourceSnapshotId.value = layerPulseSourceTaskId(task);
  layerPulseResult.value = null;
  layerPulseTaskState.value = {
    status: cancelled ? "cancelled" : "failed",
    taskId: task.task_id,
    progress: task.progress,
    message,
    error: message,
    availableOutputKeys: [],
  };
  layerPulseCanvasMode.value = "base";
}

async function waitForLayerPulsePrediction(id: string): Promise<PredictionTaskResult> {
  let consecutiveFailures = 0;
  while (!componentUnmounted) {
    let task: BackgroundTask;
    try {
      task = await getTask(id);
      consecutiveFailures = 0;
    } catch (error) {
      if (error instanceof ApiRequestError && error.status === 404) {
        layerPulseTaskState.value = {
          status: "failed",
          taskId: id,
          progress: layerPulseTaskState.value.progress,
          message: "LayerPulse 后台任务记录不存在",
          error: "后台任务记录不存在，无法恢复运行状态。",
          availableOutputKeys: [],
        };
        throw error;
      }
      if (!isRetryableTaskStatusError(error)) throw error;
      consecutiveFailures += 1;
      layerPulseTaskState.value = {
        ...layerPulseTaskState.value,
        status: isLayerPulseTaskActive(layerPulseTaskState.value.status)
          ? layerPulseTaskState.value.status
          : "running",
        taskId: id,
        message: `LayerPulse 任务仍在 GPU 运行，状态查询重试 ${consecutiveFailures} 次`,
      };
      await new Promise((resolve) => window.setTimeout(
        resolve,
        Math.min(10_000, 750 * consecutiveFailures),
      ));
      continue;
    }
    if (!isLayerPulsePredictionTask(task)) {
      throw new Error("恢复引用指向了非 LayerPulse 任务，已停止接线");
    }
    if (task.status === "queued" || task.status === "running") {
      layerPulseTaskState.value = {
        status: task.status,
        taskId: task.task_id,
        progress: task.progress,
        message: task.message || (task.status === "queued" ? "等待 GPU" : "统一推理中"),
        availableOutputKeys: [],
      };
    } else if (task.status === "completed") {
      if (task.result && "prediction" in task.result) return task.result as PredictionTaskResult;
      throw new Error("LayerPulse 推理已结束，但没有返回可识别的预测结果");
    } else {
      installTerminalLayerPulseTask(task);
      throw new Error(layerPulseTaskState.value.error || "LayerPulse 推理未完成");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
  throw new Error("页面已关闭，已停止轮询 LayerPulse 任务状态");
}

async function reattachLayerPulseTask(task: BackgroundTask) {
  if (
    !isLayerPulsePredictionTask(task)
    || (task.status !== "queued" && task.status !== "running")
  ) return;
  if (
    layerPulseTaskState.value.taskId === task.task_id
    && isLayerPulseTaskActive(layerPulseTaskState.value.status)
  ) return;
  const sourceSnapshotId = layerPulseSourceTaskId(task);
  layerPulseExecutionTaskId.value = task.task_id;
  layerPulseSourceSnapshotId.value = sourceSnapshotId;
  layerPulseResult.value = null;
  layerPulseTaskState.value = {
    status: task.status,
    taskId: task.task_id,
    progress: task.progress,
    message: task.message || "正在重新连接 LayerPulse 统一推理",
    availableOutputKeys: [],
  };
  try {
    window.sessionStorage.setItem(LAST_LAYERPULSE_TASK_STORAGE_KEY, task.task_id);
  } catch {
    appendRestorationWarning("浏览器未能保存正在运行的 LayerPulse 任务引用");
  }
  await ensureLayerPulseSourceSnapshot(sourceSnapshotId);
  try {
    const completed = await waitForLayerPulsePrediction(task.task_id);
    installCompletedLayerPulseTask(task, completed);
  } catch (error) {
    if (
      !componentUnmounted
      && layerPulseTaskState.value.status !== "failed"
      && layerPulseTaskState.value.status !== "cancelled"
    ) {
      const message = error instanceof Error ? error.message : "LayerPulse 推理恢复失败";
      layerPulseTaskState.value = {
        status: "failed",
        taskId: task.task_id,
        progress: layerPulseTaskState.value.progress,
        message,
        error: message,
        availableOutputKeys: [],
      };
    }
  }
}

async function restoreLayerPulseTask(
  task: BackgroundTask,
  options: { restoreSourceSnapshot?: boolean } = {},
) {
  if (!isLayerPulsePredictionTask(task)) {
    throw new Error("会话中的 LayerPulse 引用指向了其他模型任务");
  }
  const sourceSnapshotId = layerPulseSourceTaskId(task);
  layerPulseSourceSnapshotId.value = sourceSnapshotId;
  if (task.status === "queued" || task.status === "running") {
    void reattachLayerPulseTask(task);
    return;
  }
  if (options.restoreSourceSnapshot !== false) {
    await ensureLayerPulseSourceSnapshot(sourceSnapshotId);
  }
  if (task.status === "completed" && task.result && "prediction" in task.result) {
    installCompletedLayerPulseTask(task, task.result as PredictionTaskResult);
    return;
  }
  installTerminalLayerPulseTask(task);
}

async function restoreRememberedLayerPulseTask() {
  let rememberedTaskId = "";
  try {
    rememberedTaskId = window.sessionStorage.getItem(LAST_LAYERPULSE_TASK_STORAGE_KEY) || "";
  } catch {
    appendRestorationWarning("浏览器会话存储不可用，LayerPulse 任务无法自动恢复");
    return;
  }
  if (!rememberedTaskId) return;
  try {
    const task = await getTask(rememberedTaskId);
    const rememberedSourceSnapshotId = layerPulseSourceTaskId(task);
    const preserveCurrentSnapshot = Boolean(
      dataSnapshotTaskId.value
      && rememberedSourceSnapshotId
      && rememberedSourceSnapshotId !== dataSnapshotTaskId.value
      && task.status !== "queued"
      && task.status !== "running",
    );
    await restoreLayerPulseTask(task, {
      restoreSourceSnapshot: !preserveCurrentSnapshot,
    });
    if (preserveCurrentSnapshot) {
      appendRestorationWarning(
        `已恢复快照 ${rememberedSourceSnapshotId.slice(0, 8)} 的历史 LayerPulse 成果引用，但未覆盖当前项目活动快照 ${dataSnapshotTaskId.value.slice(0, 8)}`,
      );
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "LayerPulse 历史任务恢复失败";
    layerPulseTaskState.value = {
      status: "failed",
      taskId: rememberedTaskId,
      progress: 0,
      message,
      error: message,
      availableOutputKeys: [],
    };
  }
}

async function runLayerPulse() {
  if (isLayerPulseTaskActive(layerPulseTaskState.value.status)) return;
  if (!await reconcileProjectActiveSnapshotForInference("LayerPulse 统一解释")) return;
  const fusionContext = currentPostFusionInferenceContext();
  if (!fusionContext) {
    errorMessage.value = "LayerPulse 统一解释需要先完成当前快照的精细标定与融合视图。";
    selectView("preparation");
    showPreparationFusion();
    return;
  }
  const snapshotId = dataSnapshotTaskId.value;
  const source = layerPulseSeismicAsset.value;
  const receipt = layerPulseSupportReceipt.value;
  const support = summarizeLayerPulseSupport(receipt);
  if (
    !snapshotId
    || !source
    || !receipt
    || receipt.snapshot_id !== snapshotId
    || support.status === "blocked"
  ) return;

  layerPulseExecutionTaskId.value = "";
  layerPulseSourceSnapshotId.value = snapshotId;
  layerPulseResult.value = null;
  layerPulseCanvasMode.value = "base";
  layerPulseTaskState.value = {
    status: "queued",
    taskId: "",
    progress: 0,
    message: "正在提交 LayerPulse 单 checkpoint 统一推理",
    availableOutputKeys: [],
  };
  try {
    const created = await createPrediction({
      task_id: LAYER_PULSE_TASK_ID,
      model_id: LAYER_PULSE_MODEL_ID,
      source_task_id: snapshotId,
      registration_task_id: fusionContext.registrationTaskId,
      prepared_view_task_id: fusionContext.preparedViewId,
      seismic_path: source.path,
      device: "cuda",
      crop_size: [128, 128, 128],
      patch_size: [128, 128, 128],
      options: {
        output_profile: "platform_preview",
        preview: true,
      },
    });
    layerPulseExecutionTaskId.value = created.task_id;
    layerPulseTaskState.value = {
      ...layerPulseTaskState.value,
      taskId: created.task_id,
      message: "LayerPulse 已提交，等待 GPU 或正在统一推理",
    };
    try {
      window.sessionStorage.setItem(LAST_LAYERPULSE_TASK_STORAGE_KEY, created.task_id);
    } catch {
      appendRestorationWarning("浏览器未能保存 LayerPulse 任务引用；当前运行不受影响");
    }
    const completed = await waitForLayerPulsePrediction(created.task_id);
    installCompletedLayerPulseTask({
      task_id: created.task_id,
      task_type: "model_prediction",
      status: "completed",
      progress: 100,
      message: "LayerPulse 11 项统一解释已完成",
      request: {
        task_id: LAYER_PULSE_TASK_ID,
        model_id: LAYER_PULSE_MODEL_ID,
        source_task_id: snapshotId,
        registration_task_id: fusionContext.registrationTaskId,
        prepared_view_task_id: fusionContext.preparedViewId,
      },
      result: completed,
      error: null,
    }, completed);
  } catch (error) {
    if (componentUnmounted) return;
    if (
      layerPulseTaskState.value.status !== "failed"
      && layerPulseTaskState.value.status !== "cancelled"
    ) {
      const message = error instanceof Error ? error.message : "LayerPulse 推理失败";
      layerPulseTaskState.value = {
        status: "failed",
        taskId: layerPulseExecutionTaskId.value,
        progress: layerPulseTaskState.value.progress,
        message,
        error: message,
        availableOutputKeys: [],
      };
    }
  }
}

function selectLayerPulseOutput(outputKey: string) {
  layerPulseSelectedOutputKey.value = outputKey;
  if (layerPulseAvailableOutputKeys.value.includes(outputKey)) {
    layerPulseCanvasMode.value = "result";
  }
  centerLayerPulseOutputTab(outputKey);
}

function layerPulseOutputTabStatusLabel(outputKey: string): string {
  const state = layerPulseTaskStateForCurrentSnapshot.value;
  if ((state.availableOutputKeys || []).includes(outputKey)) return "有结果";
  if (state.status === "queued" || state.status === "running") return "等待中";
  if (state.status === "completed") return "未登记";
  if (state.status === "failed") return "未完成";
  if (state.status === "cancelled") return "已取消";
  return "可运行";
}

function centerLayerPulseOutputTab(outputKey: string, behavior: ScrollBehavior = "smooth") {
  void nextTick(() => {
    window.requestAnimationFrame(() => {
      const rail = layerPulseTaskRailElement.value;
      const button = Array.from(rail?.querySelectorAll<HTMLButtonElement>("[data-layerpulse-output-key]") || [])
        .find((candidate) => candidate.dataset.layerpulseOutputKey === outputKey);
      if (!rail || !button) return;
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      rail.scrollTo({
        top: centeredTaskScrollTop(
          rail.clientHeight,
          rail.scrollHeight,
          button.offsetTop,
          button.offsetHeight,
        ),
        behavior: reducedMotion ? "auto" : behavior,
      });
    });
  });
}

function handleLayerPulseOutputTabTransitionEnd(event: TransitionEvent, outputKey: string) {
  if (layerPulseSelectedOutputKey.value !== outputKey || event.propertyName !== "min-height") return;
  centerLayerPulseOutputTab(outputKey);
}

async function handleLayerPulseOutputTabKeydown(event: KeyboardEvent, outputKey: string) {
  const index = layerPulseOutputCatalog.findIndex((output) => output.key === outputKey);
  let nextIndex: number | null = null;
  if (event.key === "ArrowDown") nextIndex = Math.min(layerPulseOutputCatalog.length - 1, index + 1);
  if (event.key === "ArrowUp") nextIndex = Math.max(0, index - 1);
  if (event.key === "Home") nextIndex = 0;
  if (event.key === "End") nextIndex = layerPulseOutputCatalog.length - 1;
  if (nextIndex === null || nextIndex === index) return;
  event.preventDefault();
  const nextOutput = layerPulseOutputCatalog[nextIndex];
  if (!nextOutput) return;
  selectLayerPulseOutput(nextOutput.key);
  await nextTick();
  const button = Array.from(
    layerPulseTaskRailElement.value?.querySelectorAll<HTMLButtonElement>("[data-layerpulse-output-key]") || [],
  ).find((candidate) => candidate.dataset.layerpulseOutputKey === nextOutput.key);
  button?.focus({ preventScroll: true });
}

function openLayerPulseStandalone(url: string) {
  if (!url) return;
  window.open(url, "_blank", "noopener,noreferrer");
}

async function waitForPrediction(id: string): Promise<PredictionTaskResult> {
  let consecutiveFailures = 0;
  while (!componentUnmounted) {
    let task: BackgroundTask;
    try {
      task = await getTask(id);
      consecutiveFailures = 0;
    } catch (error) {
      if (error instanceof ApiRequestError && error.status === 404) {
        throw new ApiRequestError(404, "后台推理任务记录不存在，无法继续恢复运行状态");
      }
      if (!isRetryableTaskStatusError(error)) throw error;
      consecutiveFailures += 1;
      predictionConnectionState.value = "retrying";
      statusMessage.value = `后台任务仍在运行；仅状态查询暂时未返回，正在重试（第 ${consecutiveFailures} 次），这不代表模型断线`;
      await new Promise((resolve) => window.setTimeout(
        resolve,
        Math.min(10_000, 750 * consecutiveFailures),
      ));
      continue;
    }
    progress.value = task.progress;
    statusMessage.value = publicModelText(task.message);
    if (task.status === "completed") {
      if (task.result && "prediction" in task.result) return task.result as PredictionTaskResult;
      throw new Error("模型推理已结束，但后端没有返回可识别的预测结果");
    }
    if (task.status === "failed") throw new Error(publicModelText(task.error?.message, "模型推理失败"));
    if (task.status === "cancelled") throw new Error(publicModelText(task.error?.message, "模型推理已取消"));
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
  throw new Error("页面已关闭，已停止轮询模型推理状态");
}

async function reattachPredictionTask(task: BackgroundTask) {
  if (
    task.task_type !== "model_prediction"
    || (task.status !== "queued" && task.status !== "running")
  ) return;
  if (isLayerPulsePredictionTask(task)) {
    await reattachLayerPulseTask(task);
    return;
  }
  const modelId = taskRequestString(task, "model_id");
  const interpretationTaskId = taskRequestString(task, "task_id");
  const sourceTaskId = taskRequestString(task, "source_task_id");
  const registrationParentId = taskRequestString(task, "registration_task_id");
  const preparedViewParentId = taskRequestString(task, "prepared_view_task_id");
  const isGeoPathCandidate = modelId === GEOPATH_TIE_MODEL_ID;
  const requestOptions = (
    task.request?.options && typeof task.request.options === "object"
      ? task.request.options
      : {}
  ) as Record<string, unknown>;
  const reattachedFaultScope = normalizeFaultSegScope(requestOptions.faultseg_scope);
  if (
    sourceTaskId
    && !isFaultVolumeModelId(modelId)
    && interpretationTaskId !== "alignment"
    && !preparedViewParentId
  ) {
    appendRestorationWarning("已忽略旧版未绑定融合视图的在途预测；完成当前快照的融合后可重新提交。");
    try {
      if (window.sessionStorage.getItem(LAST_PREDICTION_TASK_STORAGE_KEY) === task.task_id) {
        window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
      }
    } catch {
      appendRestorationWarning("浏览器未能清理旧版非融合预测任务引用");
    }
    return;
  }
  if (isGeoPathCandidate ? geoPathCandidateRunning.value : predictionRunning.value) return;

  // Always refresh the snapshot task index while reconnecting a prediction.
  // A restored preparation task can already populate preparationResult without
  // having indexed its completed prediction children, which made earlier 1D
  // results disappear from the task rail while another prediction was running.
  if (sourceTaskId) {
    try {
      await restorePreparationForSnapshot(sourceTaskId, { resetDownstream: false });
    } catch (error) {
      const reason = error instanceof Error ? error.message : "未知错误";
      appendRestorationWarning(`预测任务的源数据状态恢复不完整，后台推理仍将继续连接：${reason}`);
    }
  }
  progress.value = task.progress;
  statusMessage.value = publicModelText(task.message, "正在重新连接模型推理任务");
  if (sourceTaskId) {
    predictionSourceTaskId.value = sourceTaskId;
    visualizationBaseTaskId.value = sourceTaskId;
  }
  if (!isFaultVolumeModelId(modelId)) {
    if (registrationParentId) registrationTaskId.value = registrationParentId;
    if (preparedViewParentId) sampleBuildingTaskId.value = preparedViewParentId;
  }

  if (isGeoPathCandidate) {
    geoPathCandidateTaskId.value = task.task_id;
    geoPathCandidateResult.value = null;
    geoPathCandidateRunning.value = true;
    try {
      window.sessionStorage.setItem(LAST_GEOPATH_CANDIDATE_TASK_STORAGE_KEY, task.task_id);
    } catch {
      appendRestorationWarning("浏览器未能保存正在运行的轨迹感知实验候选任务引用");
    }
  } else {
    predictionTaskId.value = task.task_id;
    predictionResult.value = null;
    runningPredictionTask.value = interpretationTaskId as PredictionTaskKey;
    runningPredictionExecutionTaskId.value = task.task_id;
    predictionConnectionState.value = "online";
    predictionLastHeartbeatAt.value = Date.now();
    predictionRunning.value = true;
    if (interpretationTaskId && predictionTaskDefinitions.value.some((item) => item.id === interpretationTaskId)) {
      activePredictionTask.value = interpretationTaskId as PredictionTaskKey;
    }
    if (modelId) {
      selectedPredictionModelId.value = interpretationTaskId === "horizon"
        ? SURFACE_SEG_MODEL_ID
        : selectablePredictionModelId(interpretationTaskId, modelId);
    }
    if (isFaultVolumeModelId(modelId)) faultSegScope.value = reattachedFaultScope;
    try {
      window.sessionStorage.setItem(LAST_PREDICTION_TASK_STORAGE_KEY, task.task_id);
    } catch {
      appendRestorationWarning("浏览器未能保存正在运行的预测任务引用");
    }
  }

  try {
    const completed = await waitForPrediction(task.task_id);
    if (isGeoPathCandidate) {
      geoPathCandidateResult.value = completed.prediction;
      statusMessage.value = "轨迹感知实验候选已生成，等待逐井人工审核";
    } else {
      if (!isFaultVolumeModelId(modelId)) {
        sampleBuildingTaskId.value = completed.prepared_view_task_id || sampleBuildingTaskId.value;
      }
      rememberPredictionHistory(task.task_id, completed);
      if (activePredictionTask.value === interpretationTaskId) {
        predictionTaskId.value = task.task_id;
        predictionResult.value = completed.prediction;
        predictionSourceTaskId.value = completed.source_task_id || sourceTaskId;
        predictionCanvasMode.value = "result";
        if (predictionSourceTaskId.value) await restorePredictionSource(predictionSourceTaskId.value);
      }
    }
  } catch (error) {
    if (!componentUnmounted) {
      const message = predictionFailureMessage(
        error,
        isFaultVolumeModelId(modelId) && reattachedFaultScope === "full_volume",
      );
      if (isGeoPathCandidate || activePredictionTask.value === interpretationTaskId) {
        errorMessage.value = message;
        statusMessage.value = message;
      }
    }
  } finally {
    if (isGeoPathCandidate) geoPathCandidateRunning.value = false;
    else {
      predictionRunning.value = false;
      if (runningPredictionExecutionTaskId.value === task.task_id) {
        runningPredictionTask.value = "";
        runningPredictionExecutionTaskId.value = "";
        predictionConnectionState.value = "idle";
      }
    }
  }
}

async function waitForAcceptedRegistration(id: string): Promise<WorkflowResult> {
  const deadline = Date.now() + TASK_POLL_TIMEOUT_MS;
  while (!componentUnmounted && Date.now() < deadline) {
    const task = await getTask(id);
    progress.value = task.progress;
    statusMessage.value = task.message;
    if (task.status === "completed" && task.result && "registration" in task.result) {
      return task.result as WorkflowResult;
    }
    if (task.status === "failed") throw new Error(task.error?.message || "轨迹感知实验候选审核失败");
    if (task.status === "cancelled") throw new Error(task.error?.message || "轨迹感知实验候选审核已取消");
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
  throw new Error("轨迹感知实验候选审核超时，请检查后端任务状态");
}

function clearGeoPathCandidateState() {
  geoPathCandidateTaskId.value = "";
  geoPathCandidateResult.value = null;
  geoPathSelectedWellIds.value = [];
  geoPathAcceptanceConfirmed.value = false;
  geoPathReviewNote.value = "";
  geoPathAcceptedRegistrationTaskId.value = "";
  geoPathAcceptedRegistrationResult.value = null;
  try {
    window.sessionStorage.removeItem(LAST_GEOPATH_CANDIDATE_TASK_STORAGE_KEY);
    window.sessionStorage.removeItem(LAST_GEOPATH_ACCEPTED_REGISTRATION_STORAGE_KEY);
  } catch {
    appendRestorationWarning("浏览器未能清除旧的轨迹感知实验候选会话引用");
  }
}

function installAcceptedGeoPathRegistration(taskIdValue: string, result: WorkflowResult) {
  const source = preparationResult.value || registrationResult.value || sampleResult.value;
  // The acceptance result is intentionally small. Preserve the sealed source
  // inventory in the UI while replacing only the active Registration V3.
  registrationResult.value = {
    ...(source || {}),
    ...result,
    registration: {
      ...(source?.registration || {}),
      ...result.registration,
    },
  } as WorkflowResult;
  geoPathAcceptedRegistrationResult.value = result;
  geoPathAcceptedRegistrationTaskId.value = taskIdValue;
  registrationTaskId.value = taskIdValue;
  horizontalRegistrationTaskId.value = "";
  horizontalRegistrationResult.value = null;
  sampleBuildingTaskId.value = "";
  sampleResult.value = null;
}

async function runGeoPathCandidate() {
  errorMessage.value = "";
  if (fusionWorkflowMutationRunning.value) {
    errorMessage.value = "井震标定、融合视图、轨迹候选与预测任务需依次运行；请等待当前任务结束。";
    return;
  }
  const sourceTaskId = dataSnapshotTaskId.value
    || preparationResult.value?.data_snapshot?.snapshot_id
    || "";
  const seismicPath = predictionSeismicPath.value || predictionSources.value[0]?.path || "";
  if (!geoPathCandidateModel.value || !geoPathCandidateDataFlow.value?.runnable) {
    errorMessage.value = "当前后端能力表尚未加载可运行的轨迹感知井震校正模型。";
    return;
  }
  if (!geoPathSnapshotGeometryReady.value) {
    errorMessage.value = "轨迹感知井震校正需要显式封存的 SEG-Y 几何，或与当前资产 SHA/几何指纹绑定的高置信自动几何收据；当前快照两者均未通过。";
    openAdvancedDataContract();
    return;
  }
  if (!sourceTaskId || !registrationTaskId.value || !seismicPath) {
    errorMessage.value = "请先完成 SourceSnapshot 与基础 Registration V3，并选择三维 SEG-Y。";
    return;
  }
  geoPathCandidateRunning.value = true;
  geoPathCandidateResult.value = null;
  geoPathSelectedWellIds.value = [];
  geoPathAcceptanceConfirmed.value = false;
  geoPathAcceptedRegistrationResult.value = null;
  progress.value = 0;
  statusMessage.value = "正在提交轨迹感知井震校正实验候选";
  try {
    const created = await createPrediction({
      task_id: "alignment",
      model_id: GEOPATH_TIE_MODEL_ID,
      seismic_path: seismicPath,
      source_task_id: sourceTaskId,
      registration_task_id: registrationTaskId.value,
      device: predictionDevice.value,
    });
    geoPathCandidateTaskId.value = created.task_id;
    window.sessionStorage.setItem(LAST_GEOPATH_CANDIDATE_TASK_STORAGE_KEY, created.task_id);
    const completed = await waitForPrediction(created.task_id);
    geoPathCandidateResult.value = completed.prediction;
    statusMessage.value = "轨迹感知实验候选已生成，等待逐井人工审核";
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "轨迹感知实验候选运行失败";
  } finally {
    geoPathCandidateRunning.value = false;
  }
}

async function acceptSelectedGeoPathWells() {
  const review = geoPathCandidateReview.value;
  if (!review || !geoPathAcceptanceReady.value) return;
  errorMessage.value = "";
  if (fusionWorkflowMutationRunning.value) {
    errorMessage.value = "井震标定、融合视图、轨迹候选与预测任务需依次运行；请等待当前任务结束。";
    return;
  }
  geoPathAcceptanceRunning.value = true;
  progress.value = 0;
  statusMessage.value = "正在核验候选哈希、逐井选择与修复门禁";
  try {
    const created = await acceptRegistrationCandidate(geoPathCandidateTaskId.value, {
      accepted_well_ids: [...geoPathSelectedWellIds.value],
      expected_candidate_manifest_sha256: review.candidate_manifest_sha256,
      confirmation: "ACCEPT_GEOPATH_CANDIDATE",
      review_note: geoPathReviewNote.value.trim(),
    });
    const accepted = await waitForAcceptedRegistration(created.task_id);
    installAcceptedGeoPathRegistration(created.task_id, accepted);
    window.sessionStorage.setItem(
      LAST_GEOPATH_ACCEPTED_REGISTRATION_STORAGE_KEY,
      created.task_id,
    );
    statusMessage.value = `${geoPathSelectedWellIds.value.length} 口井已生成新的实验性 Registration V3；请重新构建 PreparedView`;
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "轨迹感知实验候选审核失败";
  } finally {
    geoPathAcceptanceRunning.value = false;
  }
}

function predictionFailureMessage(error: unknown, faultSegFullVolume: boolean): string {
  const detail = publicModelText(error instanceof Error ? error.message : "模型推理失败");
  if (!faultSegFullVolume) return detail;
  const resourceLimited = /(out of memory|oom|cuda memory|allocation|资源不足|显存不足|内存不足|磁盘空间不足|存储空间不足)/i.test(detail);
  if (resourceLimited) {
    return `全工区断层识别因计算资源不足未完成：${detail}。局部中间产物不会被标记为完成成果；请释放内存、显存或磁盘空间后重新运行。`;
  }
  return `全工区断层识别未完成：${detail}。未生成可标记为完成的全工区概率体。`;
}

async function runPrediction() {
  if (predictionBusy.value) return;
  errorMessage.value = "";
  if (fusionInputsMutating.value && !isFaultSegModel.value) {
    errorMessage.value = "井震标定、融合视图或轨迹候选任务正在运行；请等待当前任务结束后再预测。";
    return;
  }
  if (!selectedPredictionModel.value) {
    errorMessage.value = "当前选择不是可启动模型；历史四层位成果仅供只读查看，请使用地层实例分割运行新任务。";
    return;
  }
  if (!preparedViewReady.value && !isFaultSegModel.value) {
    statusMessage.value = "预测等待井震融合视图就绪";
    errorMessage.value = "当前模型提交已锁定：请先完成井震精细标定与融合视图。";
    return;
  }
  if (isFaultSegModel.value && !faultSegSnapshotSourceReady.value) {
    statusMessage.value = "断层识别等待 SourceSnapshot 唯一三维 SEG-Y";
    errorMessage.value = faultSegSnapshotSourceReason.value;
    return;
  }
  if (selectedModelRequiresSeismic.value && !predictionSeismicPath.value) {
    errorMessage.value = "请先在数据准备中识别一个三维 SEG-Y 文件。";
    return;
  }
  if (usesSealedSnapshotWellInput.value && !sealedSnapshotWellInputReady.value) {
    errorMessage.value = snapshotWellSourceReason.value;
    return;
  }
  if (
    isFaultSegModel.value
    && faultSegScope.value === "full_volume"
    && !window.confirm("全区断层预测通常约需 20–30 分钟，数据盘繁忙时可能更久。任务会遍历全部重叠窗口并生成大型概率体，确认继续吗？")
  ) return;
  const runIntent = {
    taskId: activePredictionTask.value,
    modelId: selectedPredictionModelId.value,
    seismicPath: predictionSeismicPath.value,
    device: predictionDevice.value,
    requiresSeismic: selectedModelRequiresSeismic.value,
    usesSealedSnapshotWellInput: usesSealedSnapshotWellInput.value,
    isFaultSegModel: isFaultSegModel.value,
    faultSegScope: faultSegScope.value,
    isSurfaceSegModel: isSurfaceSegModel.value,
    isF3FaciesModel: isF3FaciesModel.value,
    isHorizonModel: isHorizonModel.value,
    isWellFuseGeobodyModel: isWellFuseGeobodyModel.value,
    recommendedOptions: { ...(selectedModelCompatibility.value?.recommended_options || {}) },
    cropSize: predictionCropSize.value,
    threshold: predictionThreshold.value,
    surfaceSegScope: surfaceSegScope.value,
    surfaceSegMaxInlines: surfaceSegMaxInlines.value,
    surfaceSegInlineCount: surfaceSegInlineCount.value,
    surfaceSegAmplitudeMode: surfaceSegAmplitudeMode.value,
    surfaceSegQueryThreshold: surfaceSegQueryThreshold.value,
    surfaceSegMaskThreshold: surfaceSegMaskThreshold.value,
    surfaceSegformerBatchSize: surfaceSegformerBatchSize.value,
    surfaceMask2formerBatchSize: surfaceMask2formerBatchSize.value,
    f3FaciesMode: f3FaciesMode.value,
    f3TStart: f3TStart.value,
    f3TCount: f3TCount.value,
    f3InlineStart: f3InlineStart.value,
    f3InlineCount: f3InlineCount.value,
    f3CrosslineStart: f3CrosslineStart.value,
    f3CrosslineCount: f3CrosslineCount.value,
  };
  runningPredictionTask.value = runIntent.taskId;
  runningPredictionExecutionTaskId.value = "";
  predictionConnectionState.value = "idle";
  predictionLastHeartbeatAt.value = null;
  predictionOrchestrationRunning.value = true;
  predictionCanvasMode.value = "auto";
  statusMessage.value = "正在准备运行环境";
  try {
  // Predictions must remain attached to the immutable data snapshot.  A
  // registration/sample task may be the most recent UI task, but it is not the
  // seismic source and cannot reconstruct the original visualization lineage.
  const sourceTaskId = dataSnapshotTaskId.value
    || preparationResult.value?.data_snapshot?.snapshot_id
    || "";
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
      task_id: runIntent.taskId,
      model_id: runIntent.modelId,
      ...(runIntent.requiresSeismic ? { seismic_path: runIntent.seismicPath } : {}),
      source_task_id: sourceTaskId || undefined,
      ...(runIntent.isFaultSegModel
        ? {}
        : {
            registration_task_id: registrationTaskId.value || undefined,
            prepared_view_task_id: sampleBuildingTaskId.value || undefined,
          }),
      device: runIntent.device,
    };
    const created = runIntent.usesSealedSnapshotWellInput
      ? await createPrediction({
          ...common,
          options: { batch_size: 4 },
        })
      : runIntent.isFaultSegModel
      ? await createPrediction({
          ...common,
          options: {
            faultseg_scope: runIntent.faultSegScope,
          },
        })
      : runIntent.isSurfaceSegModel
      ? await createPrediction({
          ...common,
          options: {
              ...runIntent.recommendedOptions,
              amplitude_mode: runIntent.surfaceSegAmplitudeMode,
              query_threshold: runIntent.surfaceSegQueryThreshold,
              mask_threshold: runIntent.surfaceSegMaskThreshold,
              segformer_batch_size: runIntent.surfaceSegformerBatchSize,
              mask2former_batch_size: runIntent.surfaceMask2formerBatchSize,
              num_visualizations: 5,
              write_mask_sgy: runIntent.surfaceSegScope === "full",
              ...(runIntent.surfaceSegScope === "smoke" ? { max_inlines: runIntent.surfaceSegMaxInlines } : {}),
              ...(runIntent.surfaceSegInlineCount === "" ? {} : { inline_count: Number(runIntent.surfaceSegInlineCount) }),
          },
        })
      : runIntent.isF3FaciesModel
        ? await createPrediction({
            ...common,
            options: {
              ...runIntent.recommendedOptions,
              ...(runIntent.f3FaciesMode === "manual_roi"
                ? {
                    t_start: runIntent.f3TStart,
                    t_count: runIntent.f3TCount,
                    inline_start: runIntent.f3InlineStart,
                    inline_count: runIntent.f3InlineCount,
                    crossline_start: runIntent.f3CrosslineStart,
                    crossline_count: runIntent.f3CrosslineCount,
                    batch_size: 4,
                  }
                : runIntent.f3FaciesMode === "single_trace"
                  ? { inline_count: 1, crossline_count: 1, batch_size: 1 }
                  : { batch_size: 4 }),
            },
          })
      : runIntent.isHorizonModel
        ? await createPrediction({
            ...common,
            options: {
              ...runIntent.recommendedOptions,
            },
          })
      : runIntent.isWellFuseGeobodyModel
        ? await createPrediction({
            ...common,
            patch_size: [64, 96, 96],
            overlap: [32, 48, 48],
            threshold: runIntent.threshold,
            options: {
              minimum_voxels: 256,
              ...runIntent.recommendedOptions,
            },
          })
        : await createPrediction({
          ...common,
          crop_size: [runIntent.cropSize, runIntent.cropSize, runIntent.cropSize],
          patch_size: [32, 32, 32],
          overlap: [8, 8, 8],
          threshold: runIntent.threshold,
        });
    runningPredictionExecutionTaskId.value = created.task_id;
    predictionConnectionState.value = "online";
    predictionLastHeartbeatAt.value = Date.now();
    if (activePredictionTask.value === runIntent.taskId) predictionTaskId.value = created.task_id;
    window.sessionStorage.setItem(LAST_PREDICTION_TASK_STORAGE_KEY, created.task_id);
    const completed = await waitForPrediction(created.task_id);
    if (!runIntent.isFaultSegModel) {
      sampleBuildingTaskId.value = completed.prepared_view_task_id || sampleBuildingTaskId.value;
    }
    rememberPredictionHistory(created.task_id, completed);
    if (activePredictionTask.value === runIntent.taskId) {
      predictionTaskId.value = created.task_id;
      predictionResult.value = completed.prediction;
      predictionSourceTaskId.value = completed.source_task_id || sourceTaskId;
      predictionCanvasMode.value = "result";
    }
  } catch (error) {
    const message = predictionFailureMessage(
      error,
      runIntent.isFaultSegModel && runIntent.faultSegScope === "full_volume",
    );
    if (activePredictionTask.value === runIntent.taskId) {
      errorMessage.value = message;
      statusMessage.value = message;
    }
  } finally {
    predictionRunning.value = false;
  }
  } finally {
    predictionOrchestrationRunning.value = false;
    if (runningPredictionTask.value === runIntent.taskId) {
      runningPredictionTask.value = "";
      runningPredictionExecutionTaskId.value = "";
      predictionConnectionState.value = "idle";
    }
  }
}

function navigationState(view: ViewKey): string {
  if (view === "overview") return "";
  if (view === "preparation") {
    if (preparedViewReady.value) return "已融合";
    if (registrationRunning.value || sampleRunning.value) return "处理中";
    return preparation.value ? "可融合" : "未开始";
  }
  if (view === "visualization") return seismicInventory.value.length || seismicPreviewCount.value || wellLogPreviews.value.length ? "当前任务" : "待数据";
  if (view === "samples") {
    if (preparedViewReady.value) return "已完成";
    if (horizontalRegistrationTaskId.value && !registrationTaskId.value) return "仅XY完成";
    return preparation.value?.gates.can_build_samples ? "可执行" : "待对齐";
  }
  if (view === "models") return preparedViewReady.value ? "可接入" : "接口";
  if (view === "layerpulse") {
    if (isLayerPulseTaskActive(layerPulseTaskStateForCurrentSnapshot.value.status)) return "运行中";
    if (layerPulseTaskStateForCurrentSnapshot.value.status === "completed") return "有结果";
    if (layerPulseTaskStateForCurrentSnapshot.value.status === "failed") return "运行失败";
    if (layerPulseTaskStateForCurrentSnapshot.value.status === "cancelled") return "已取消";
    const support = summarizeLayerPulseSupport(layerPulseSupportReceipt.value);
    if (support.status === "blocked") return dataSnapshotTaskId.value ? "需核验" : "待数据";
    return support.status === "degraded" ? "降级可用" : "可运行";
  }
  if (view === "prediction") {
    if (!predictionEntryReady.value) return "待融合";
    if (faultResultNeedsSupportedScopeRerun.value) return "需重跑";
    return predictionBusy.value ? "运行中" : predictionResult.value ? "有结果" : "选择任务";
  }
  if (view === "assistant") return capabilities.value?.llm.available ? "已连接" : "本地模式";
  if (view === "evaluation") return preparedViewReady.value ? "有结果" : "待结果";
  return "可配置";
}

function stageCompletedByDownstream(stageId: string): boolean {
  if (stageId === "vertical_alignment") {
    return formalRegistrationReady.value || preparedViewReady.value;
  }
  if (stageId === "sample_building") return preparedViewReady.value;
  return false;
}

function stageDisplayReady(stage: PreparationStage): boolean {
  return stage.status === "就绪" || stageCompletedByDownstream(stage.id);
}

function stageClass(stage: PreparationStage): string {
  if (stageDisplayReady(stage)) return "ready";
  if (!hasExplicitPreparationScope.value) {
    if (stage.status === "需确认") return "warning";
    if (stage.status === "阻断") return "blocked";
    return "deferred";
  }
  if (!stageRequiredForCurrentRun(stage.id)) return "deferred";
  if (stage.status === "阻断") return "blocked";
  if (stage.status === "需确认") return "warning";
  if (stage.status === "本任务不需要") return "not-used";
  return "waiting";
}

function stageStatusLabel(stage: PreparationStage): string {
  if (stageCompletedByDownstream(stage.id)) {
    return stage.id === "sample_building" ? "PreparedView 已就绪" : "标定已完成";
  }
  if (!hasExplicitPreparationScope.value) {
    if (stage.status === "就绪") return stage.issue_count ? `${stage.issue_count} 项审计记录` : "已核验";
    if (stage.status === "未就绪" || stage.status === "阻断") return "未就绪";
    if (stage.status === "需确认") return "待复核";
    if (stage.status === "本任务不需要") return "本轮未执行";
    return stage.status || "状态未记录";
  }
  if (!stageRequiredForCurrentRun(stage.id)) {
    return "未参与当前任务";
  }
  return stage.status === "本任务不需要" ? "不影响当前任务" : stage.status;
}

function stageDescription(stage: PreparationStage): string {
  if (stageCompletedByDownstream(stage.id)) {
    return stage.id === "sample_building"
      ? "融合样本已通过 PreparedView 门禁，并与当前 SourceSnapshot、Registration 血缘一致。"
      : "当前 SourceSnapshot 已生成可用于融合的正式井震标定产品。";
  }
  if (!hasExplicitPreparationScope.value) {
    if (FORMAL_REGISTRATION_AUDIT_STAGES.has(stage.id)) {
      return "该缺口会影响平台统一的井震融合门禁；即使模型不消费融合特征，也必须先形成可用 PreparedView 才能预测。";
    }
    return "这是通用数据盘点结果。选择具体任务与模型并重新准备后，平台才会按该模型的实际输入合同计算门禁。";
  }
  if (stageRequiredForCurrentRun(stage.id)) return stage.description;
  if (FORMAL_REGISTRATION_AUDIT_STAGES.has(stage.id)) {
    return `该项影响平台统一融合门禁；${currentScopeModelName.value} 可以不消费融合特征，但在 PreparedView 就绪前仍不能启动预测。`;
  }
  return `「${currentScopeModelName.value || preparationTargetTaskName.value}」不依赖本步骤；这不是失败，也不会删除已登记数据。`;
}

function issueStatusLabel(issue: PreparationIssue): string {
  if (!issueNeedsCurrentAttention(issue)) return AUTO_AUDIT_STATUSES.has(issue.confirmation_status)
    ? issue.confirmation_status
    : "审计记录 · 不影响当前执行";
  return issue.confirmation_status === "本任务不需要" ? "不影响当前任务" : issue.confirmation_status;
}

function stageShortName(stageId: string): string {
  return ({
    asset_registration: "资产登记",
    log_preprocessing: "测井标准化",
    well_entity_alignment: "井数据合并",
    seismic_geometry: "地震几何",
    vertical_datum_normalization: "垂向基准统一",
    seismic_time_reference: "时间基准统一",
    spatial_alignment: "井震对齐",
    vertical_alignment: "时间域标定",
    sample_building: "样本构建",
  } as Record<string, string>)[stageId] || stageId;
}

function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(1, Math.round(totalSeconds));
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (minutes < 60) return remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分钟`;
  const hours = Math.floor(minutes / 60);
  const minuteRemainder = minutes % 60;
  return minuteRemainder ? `${hours} 小时 ${minuteRemainder} 分` : `${hours} 小时`;
}

function formatTaskTimestamp(value: string): string {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) return "时间未记录";
  return timestamp.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 ** 3) return `${(size / 1024 ** 2).toFixed(1)} MB`;
  return `${(size / 1024 ** 3).toFixed(2)} GB`;
}

async function clearPlatformCaches() {
  if (cacheClearing.value) return;
  const confirmed = window.confirm(
    "确定停止全部任务、清空可重建缓存并重新开始？\n\n所有排队或运行中的任务都会被取消，且不会自动恢复；当前页面的路径草稿、任务选择和恢复引用会被清空。原始数据、模型权重、任务记录、封存快照、已完成的预测结果和报告不会被删除。",
  );
  if (!confirmed) return;
  cacheClearing.value = true;
  cacheMessage.value = "正在清理可重建缓存…";
  try {
    const result = await clearSystemCache();
    const taskSummary = result.tasks_cancelled > 0
      ? `已停止 ${result.tasks_cancelled} 个排队或运行任务；`
      : "";
    const summary = `${taskSummary}已清理 ${result.files_removed} 个缓存文件、${result.memory_entries_removed} 个内存条目，释放 ${formatBytes(result.bytes_reclaimed)}`;
    if (result.errors.length) {
      cacheMessage.value = `${summary}；仍有 ${result.errors.length} 项未能清理，请重试后再重新开始`;
      cacheClearing.value = false;
      return;
    }
    const keys = Array.from({ length: window.sessionStorage.length }, (_, index) =>
      window.sessionStorage.key(index),
    ).filter((key): key is string => Boolean(key));
    keys.forEach((key) => {
      if (key.startsWith("strata_vision_")) {
        window.sessionStorage.removeItem(key);
      }
    });
    window.sessionStorage.setItem(FRESH_SESSION_STORAGE_KEY, "1");
    window.sessionStorage.setItem(CACHE_FLASH_STORAGE_KEY, summary);
    window.location.reload();
  } catch (error) {
    cacheMessage.value = error instanceof Error ? `缓存清理失败：${error.message}` : "缓存清理失败";
    cacheClearing.value = false;
  }
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

function showBaseVisualization() {
  visualizationSourceTaskId.value = "";
  selectedSeismicAssetIndex.value = 0;
}

function backendPublicUrl(path: string): string {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  const prefix = import.meta.env.DEV ? "http://127.0.0.1:725" : "";
  return `${prefix}${path.startsWith("/") ? path : `/${path}`}`;
}

function visualizationUrlWithEmbed(path: string, embed: 0 | 1): string {
  const publicUrl = backendPublicUrl(path);
  if (!publicUrl) return "";
  const hashIndex = publicUrl.indexOf("#");
  const fragment = hashIndex >= 0 ? publicUrl.slice(hashIndex) : "";
  const base = hashIndex >= 0 ? publicUrl.slice(0, hashIndex) : publicUrl;
  const updated = /([?&])embed=[^&]*/i.test(base)
    ? base.replace(/([?&])embed=[^&]*/i, `$1embed=${embed}`)
    : `${base}${base.includes("?") ? "&" : "?"}embed=${embed}`;
  return `${updated}${fragment}`;
}

function taskArtifactUrl(taskIdValue: string, name: string): string {
  if (!taskIdValue || !name) return "";
  return backendPublicUrl(`/api/v1/tasks/${encodeURIComponent(taskIdValue)}/artifacts/${encodeURIComponent(name)}`);
}

function isDirectoryOutput(name: string, path: string): boolean {
  const normalizedName = name.trim().toLowerCase();
  return /(?:^|_)(?:directory|dir)$/.test(normalizedName) || /[\\/]$/.test(path.trim());
}

function artifactDisplayName(name: string, path: string): string {
  if (isDirectoryOutput(name, path)) return "目录成果";
  const normalized = path.trim().replaceAll("\\", "/");
  return publicModelText(normalized.split("/").filter(Boolean).at(-1) || name);
}

function taskOutputEntries(task: BackgroundTask): AcceptanceArtifact[] {
  if (task.status !== "completed") return [];
  const result = task.result;
  if (!result) return [];
  let outputs: Record<string, string | null | undefined> = {};
  if ("prediction" in result) outputs = result.prediction.outputs || {};
  else if (isHorizontalRegistrationTaskResult(result)) outputs = result.horizontal_registration.output_files;
  else if (result.matching?.output_files) outputs = result.matching.output_files;
  else if (result.registration?.output_files) outputs = result.registration.output_files;
  if (task.task_type === "data_preparation" && isWorkflowResult(result)) {
    const snapshotManifestPath = result.data_snapshot?.snapshot_manifest_path;
    if (snapshotManifestPath) outputs = { snapshot_manifest: snapshotManifestPath, ...outputs };
  }
  return Object.entries(outputs).flatMap(([name, path]) => {
    if (!path) return [];
    return [{ name, path, directory: isDirectoryOutput(name, path) }];
  });
}

function acceptanceTaskLabel(task: BackgroundTask): string {
  const result = task.result;
  if (result && "prediction" in result) return modelPresentationName(
    result.prediction.model_id,
    result.prediction.model_name || result.prediction.task_name,
    result.prediction.scientific_status,
  );
  if (task.task_type === "data_preparation") return "源快照与输入 QC";
  if (task.task_type === "well_tie" || task.task_type === "horizontal_registration") return "井震标定";
  if (task.task_type === "sample_building") return "PreparedView 融合样本";
  return task.task_type || "任务";
}

function acceptanceTaskFamily(task: BackgroundTask): string {
  const result = task.result;
  if (result && "prediction" in result) return result.prediction.task_id || "prediction";
  if (task.task_type === "data_preparation") return "source_snapshot";
  if (task.task_type === "well_tie" || task.task_type === "horizontal_registration") return "registration";
  return task.task_type || "workflow";
}

function compactAcceptanceWarnings(warnings: string[]): string[] {
  const unique = [...new Set(warnings.map((warning) => String(warning).trim()).filter(Boolean))];
  const physicalRange: number[] = [];
  const fractureDepthRange: number[] = [];
  const retained: string[] = [];
  for (const warning of unique) {
    const physicalMatch = warning.match(/^([\d.]+)% of observed log values are outside broad physical plausibility ranges\.?$/i);
    if (physicalMatch) {
      physicalRange.push(Number(physicalMatch[1]));
      continue;
    }
    const fractureDepthMatch = warning.match(/^([\d.]+)% of original MD samples are outside the fracture model training range; relative MD was mapped into the training interval and inference was not blocked\.?$/i);
    if (fractureDepthMatch) {
      fractureDepthRange.push(Number(fractureDepthMatch[1]));
      continue;
    }
    retained.push(warning);
  }
  const percentRange = (values: number[]) => {
    const finite = values.filter(Number.isFinite).sort((left, right) => left - right);
    if (!finite.length) return "";
    const minimum = finite[0].toFixed(3);
    const maximum = finite.at(-1)?.toFixed(3) || minimum;
    return minimum === maximum ? `${minimum}%` : `${minimum}%–${maximum}%`;
  };
  if (physicalRange.length) {
    retained.push(`${physicalRange.length} 口井存在超出宽松物理合理范围的观测值（${percentRange(physicalRange)}）；逐井明细保留在模型回执中。`);
  }
  if (fractureDepthRange.length) {
    retained.push(`${fractureDepthRange.length} 口井的 MD 超出裂缝模型训练范围（${percentRange(fractureDepthRange)}），已做可审计相对深度映射；结果仅用于候选排序。`);
  }
  return retained;
}

const acceptanceGroups = computed<AcceptanceGroup[]>(() => {
  const tasks = acceptanceSnapshotDetail.value?.tasks || [];
  return [...tasks]
    .sort((left, right) => taskUpdatedAt(right) - taskUpdatedAt(left))
    .map((task) => {
      const completed = task.status === "completed";
      const artifacts = taskOutputEntries(task);
      const hasArtifactEvidence = artifacts.length > 0;
      const result = task.result;
      const prediction = result && "prediction" in result ? result.prediction : null;
      const accepted = completed
        && hasArtifactEvidence
        && prediction?.display_acceptance_decision?.display_status === "accepted";
      const sourceEvidence = completed
        && hasArtifactEvidence
        && task.task_type === "data_preparation";
      const standardResult = completed
        && hasArtifactEvidence
        && prediction?.standard_result_bundle?.visualization?.available === true;
      const fractureResult = Boolean(
        prediction
        && (
          /fracture/i.test(prediction.task_id || "")
          || /fracture/i.test(prediction.model_id || "")
        ),
      );
      const sequenceArtifacts = completed && prediction
        ? artifacts.filter((artifact) =>
          !artifact.directory
          && /\.csv$/i.test(artifact.path)
          && (
            fractureResult
              ? /fracture_intervals_csv/i.test(artifact.name)
              : /prediction_csv/i.test(artifact.name)
          ),
        )
        : [];
      const candidate = completed
        && hasArtifactEvidence
        && !accepted
        && (
          prediction?.candidate_visualization_decision?.renderable === true
          || sequenceArtifacts.length > 0
          || standardResult
        );
      const canVisualize = completed && hasArtifactEvidence && Boolean(
        sourceEvidence
        || task.task_type === "well_tie"
        || task.task_type === "horizontal_registration"
        || task.task_type === "sample_building"
        || accepted
        || prediction?.candidate_visualization_decision?.renderable === true
        || standardResult,
      );
      const acceptanceStatus: AcceptanceGroup["acceptanceStatus"] = task.status === "failed"
        ? "failed"
        : !completed
          ? "not_evaluated"
          : accepted
            ? "accepted"
            : candidate
              ? "candidate"
              : sourceEvidence
                ? "source_evidence"
                : "not_evaluated";
      const visualizationStatus: AcceptanceGroup["visualizationStatus"] = accepted
        ? "accepted"
        : candidate || sequenceArtifacts.length
          ? "candidate"
          : sourceEvidence && canVisualize
            ? "source"
            : artifacts.length
              ? "evidence_only"
              : "unavailable";
      return {
        taskId: task.task_id,
        label: acceptanceTaskLabel(task),
        family: acceptanceTaskFamily(task),
        executionStatus: task.status,
        visualizationStatus,
        scientificStatus: prediction?.scientific_status
          || prediction?.candidate_visualization_decision?.scientific_status
          || prediction?.artifact_bundle?.scientific_status
          || (sourceEvidence ? "source_evidence" : "not_declared"),
        acceptanceStatus,
        artifacts,
        sequenceArtifacts,
        warnings: compactAcceptanceWarnings([
          ...(prediction?.warnings || []),
          ...(prediction?.artifact_bundle?.warnings || []),
        ]),
        canVisualize,
        standardResult,
      };
    });
});
const selectedAcceptanceGroup = computed(() =>
  acceptanceGroups.value.find((group) => group.taskId === selectedAcceptanceTaskId.value)
  || acceptanceGroups.value[0]
  || null,
);
const selectedAcceptanceIsFracture = computed(() => Boolean(
  selectedAcceptanceGroup.value
  && (
    /fracture/i.test(selectedAcceptanceGroup.value.family)
    || selectedAcceptanceGroup.value.label.includes("裂缝")
  )
));
const selectedPlanViewArtifact = computed(() =>
  selectedAcceptanceGroup.value?.artifacts.find((artifact) =>
    artifact.name === "horizontal_registration_plan_view"
    || /horizontal_registration_plan_view\.json$/i.test(artifact.path),
  ) || null,
);
const acceptanceSnapshotSourceLabel = computed(() => {
  const preparationTask = acceptanceSnapshotDetail.value?.tasks.find((task) => task.task_type === "data_preparation");
  const request = preparationTask?.request || {};
  const seismicPaths = Array.isArray(request.seismic_paths) ? request.seismic_paths : [];
  const sourcePath = seismicPaths.find((path): path is string => typeof path === "string" && Boolean(path.trim()));
  if (!sourcePath) return "来源文件未命名";
  return sourcePath.split(/[\\/]/).filter(Boolean).at(-1) || sourcePath;
});
const acceptanceVisualizationUrl = computed(() => {
  const group = selectedAcceptanceGroup.value;
  if (!group?.canVisualize || selectedPlanViewArtifact.value) return "";
  if (group.standardResult) {
    return backendPublicUrl(`/api/v1/tasks/${encodeURIComponent(group.taskId)}/standard-results/visualization`);
  }
  return backendPublicUrl(`/统一数据可视化?task_id=${encodeURIComponent(group.taskId)}&embed=1`);
});

function acceptanceStatusLabel(status: AcceptanceGroup["acceptanceStatus"]): string {
  return ({
    accepted: "已验收",
    candidate: "候选 · 未验收",
    source_evidence: "来源证据",
    not_evaluated: "未做定量验收",
    failed: "执行失败",
  } as Record<AcceptanceGroup["acceptanceStatus"], string>)[status];
}

function visualizationStatusLabel(status: AcceptanceGroup["visualizationStatus"]): string {
  return ({
    source: "源数据可视化",
    accepted: "验收视图",
    candidate: "候选视图",
    evidence_only: "仅证据下载",
    unavailable: "无可渲染结果",
  } as Record<AcceptanceGroup["visualizationStatus"], string>)[status];
}

function selectAcceptanceGroup(taskIdValue: string) {
  const group = acceptanceGroups.value.find((candidate) => candidate.taskId === taskIdValue) || null;
  sequenceLoadSequence += 1;
  sequenceAbortController?.abort();
  sequenceAbortController = null;
  planViewLoadSequence += 1;
  planViewAbortController?.abort();
  planViewAbortController = null;
  selectedAcceptanceTaskId.value = group?.taskId || "";
  selectedSequenceArtifact.value = group?.sequenceArtifacts[0]?.name || "";
  sequencePreview.value = null;
  fractureIntervalPreview.value = null;
  sequenceError.value = "";
  sequencePreviewLimited.value = false;
  sequenceLoading.value = false;
  planViewPreview.value = null;
  planViewError.value = "";
  planViewLoading.value = false;
  if (group && /fracture/i.test(group.family) && selectedSequenceArtifact.value) {
    void nextTick().then(loadSequencePreview);
  }
}

function acceptanceSnapshotOptionLabel(snapshot: SourceSnapshotSummary): string {
  const createdAt = snapshot.created_at ? new Date(snapshot.created_at) : null;
  const timestamp = createdAt && Number.isFinite(createdAt.getTime())
    ? createdAt.toLocaleString("zh-CN", { hour12: false })
    : "时间未知";
  const displayName = snapshot.display_name?.trim() || "未命名测区";
  return `${displayName} · ${timestamp} · ${snapshot.snapshot_id.slice(0, 8)}`;
}

async function loadAcceptanceSnapshot(snapshotId: string) {
  if (!snapshotId) {
    acceptanceSnapshotDetail.value = null;
    acceptanceError.value = "尚未选择封存快照。";
    return;
  }
  const loadSequence = ++acceptanceLoadSequence;
  const previousLoadedSnapshotId = loadedAcceptanceSnapshotId.value;
  acceptanceLoading.value = true;
  acceptanceError.value = "";
  try {
    const detail = await getSnapshotDetail(snapshotId);
    if (loadSequence !== acceptanceLoadSequence) return;
    selectedAcceptanceSnapshotId.value = detail.snapshot.snapshot_id;
    loadedAcceptanceSnapshotId.value = detail.snapshot.snapshot_id;
    acceptanceSnapshotDetail.value = detail;
    const selectedExists = acceptanceGroups.value.some((group) => group.taskId === selectedAcceptanceTaskId.value);
    if (!selectedExists) selectAcceptanceGroup(acceptanceGroups.value[0]?.taskId || "");
  } catch (error) {
    if (loadSequence !== acceptanceLoadSequence) return;
    selectedAcceptanceSnapshotId.value = previousLoadedSnapshotId;
    acceptanceError.value = error instanceof Error ? error.message : "无法读取验收任务清单";
  } finally {
    if (loadSequence === acceptanceLoadSequence) acceptanceLoading.value = false;
  }
}

async function loadSelectedAcceptanceSnapshot() {
  await loadAcceptanceSnapshot(selectedAcceptanceSnapshotId.value);
}

async function refreshAcceptanceSnapshot() {
  const loadSequence = ++acceptanceLoadSequence;
  const previousLoadedSnapshotId = loadedAcceptanceSnapshotId.value;
  acceptanceLoading.value = true;
  acceptanceError.value = "";
  try {
    const projectCatalog = await getProjects();
    const project = projectCatalog.projects.find((item) => item.project_id === "local-default")
      || [...projectCatalog.projects].sort((left, right) =>
        Date.parse(right.updated_at || "") - Date.parse(left.updated_at || ""),
      )[0];
    if (!project) throw new Error("尚未创建可读取的项目");
    const snapshotCatalog = await getProjectSnapshots(project.project_id);
    if (loadSequence !== acceptanceLoadSequence) return;
    acceptanceSnapshotCatalog.value = [...snapshotCatalog.snapshots]
      .filter((snapshot) => snapshot.state === "sealed")
      .sort((left, right) =>
        Date.parse(right.updated_at || right.created_at || "")
        - Date.parse(left.updated_at || left.created_at || ""),
      );
    const catalogIds = new Set(acceptanceSnapshotCatalog.value.map((snapshot) => snapshot.snapshot_id));
    const snapshotId = catalogIds.has(selectedAcceptanceSnapshotId.value)
      ? selectedAcceptanceSnapshotId.value
      : catalogIds.has(dataSnapshotTaskId.value)
        ? dataSnapshotTaskId.value
        : acceptanceSnapshotCatalog.value[0]?.snapshot_id || "";
    if (!snapshotId) {
      acceptanceSnapshotDetail.value = null;
      acceptanceError.value = "当前项目尚无封存快照，请先完成数据准备。";
      return;
    }
    const detail = await getSnapshotDetail(snapshotId);
    if (loadSequence !== acceptanceLoadSequence) return;
    selectedAcceptanceSnapshotId.value = detail.snapshot.snapshot_id;
    loadedAcceptanceSnapshotId.value = detail.snapshot.snapshot_id;
    acceptanceSnapshotDetail.value = detail;
    const selectedExists = acceptanceGroups.value.some((group) => group.taskId === selectedAcceptanceTaskId.value);
    if (!selectedExists) selectAcceptanceGroup(acceptanceGroups.value[0]?.taskId || "");
  } catch (error) {
    if (loadSequence !== acceptanceLoadSequence) return;
    selectedAcceptanceSnapshotId.value = previousLoadedSnapshotId;
    acceptanceError.value = error instanceof Error ? error.message : "无法读取验收任务清单";
  } finally {
    if (loadSequence === acceptanceLoadSequence) acceptanceLoading.value = false;
  }
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') { current += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === "," && !quoted) { cells.push(current); current = ""; }
    else current += character;
  }
  cells.push(current);
  return cells;
}

function parseFiniteTextNumber(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value !== "string" || !value.trim()) return null;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function parseSequenceCsv(text: string): SequencePreview {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 3) throw new Error("CSV 没有足够的序列行");
  const headers = parseCsvLine(lines[0]).map((header) => header.trim());
  const rows = lines.slice(1).map(parseCsvLine);
  const depthIndex = headers.findIndex((header) => /^(md|md_m|depth|depth_m|tvd|tvd_m|tvdss|tvdss_m)$/i.test(header));
  if (depthIndex < 0) throw new Error("CSV 未声明 MD/TVD/TVDSS/depth 列，浏览器不会猜测深度轴");
  const numericColumns = headers.map((header, index) => ({ header, index })).filter(({ header, index }) => {
    if (index === depthIndex || /(^|_)(well|id|name|source)(_|$)/i.test(header)) return false;
    return rows.reduce((count, row) => count + (parseFiniteTextNumber(row[index]) !== null ? 1 : 0), 0) >= 2;
  }).slice(0, 6);
  if (!numericColumns.length) throw new Error("CSV 未包含可绘制的数值结果列");
  const parsed = rows.map((row) => ({
    depth: parseFiniteTextNumber(row[depthIndex]),
    values: numericColumns.map(({ index }) => {
      return parseFiniteTextNumber(row[index]);
    }),
  })).filter((row): row is { depth: number; values: Array<number | null> } => row.depth !== null);
  if (parsed.length < 2) throw new Error("CSV 深度轴有效点不足");
  const stride = Math.max(1, Math.ceil(parsed.length / 480));
  const sampled = parsed.filter((_, index) => index % stride === 0 || index === parsed.length - 1);
  return {
    depthName: headers[depthIndex],
    depths: sampled.map((row) => row.depth),
    series: numericColumns.map(({ header }, seriesIndex) => {
      const values = sampled.map((row) => row.values[seriesIndex]);
      const finite = values.filter((value): value is number => value !== null);
      return { id: header, min: Math.min(...finite), max: Math.max(...finite), values };
    }),
  };
}

function parseFractureIntervalCsv(text: string): FractureIntervalPreview {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) throw new Error("裂缝层段 CSV 没有结果行");
  const headers = parseCsvLine(lines[0]).map((header) => header.trim());
  const indexByName = new Map(headers.map((header, index) => [header.toLowerCase(), index]));
  const required = ["top_md_m", "bottom_md_m", "fracture_level_code", "fracture_level"];
  if (required.some((name) => !indexByName.has(name))) {
    throw new Error("裂缝层段 CSV 缺少 top/bottom MD 或确定性发育级别");
  }
  const cell = (row: string[], name: string) => row[indexByName.get(name) ?? -1] || "";
  const labels: Record<FractureDisplaySegment["level"], string> = {
    low: "相对较弱",
    medium: "相对中等",
    high: "相对较强",
  };
  const parsed: Array<Omit<FractureDisplaySegment, "y" | "height">> = lines.slice(1).flatMap(
    (line): Array<Omit<FractureDisplaySegment, "y" | "height">> => {
    const row = parseCsvLine(line);
    const topMd = parseFiniteTextNumber(cell(row, "top_md_m"));
    const bottomMd = parseFiniteTextNumber(cell(row, "bottom_md_m"));
    const rawCode = parseFiniteTextNumber(cell(row, "fracture_level_code"));
    const rawLevel = cell(row, "fracture_level").trim().toLowerCase();
    const level = rawLevel === "low" || rawCode === 0
      ? "low"
      : rawLevel === "high" || rawCode === 2
        ? "high"
        : rawLevel === "medium" || rawCode === 1
          ? "medium"
          : null;
    if (topMd === null || bottomMd === null || bottomMd < topMd || level === null) return [];
    const thickness = parseFiniteTextNumber(cell(row, "thickness_m"));
    const samples = parseFiniteTextNumber(cell(row, "sample_count"));
      return [{
        level,
        label: cell(row, "fracture_level_zh").trim() || labels[level],
        topMd,
        bottomMd,
        thicknessM: thickness === null ? bottomMd - topMd : thickness,
        sampleCount: samples === null ? 0 : Math.max(0, Math.round(samples)),
      }];
    },
  ).sort((left, right) => left.topMd - right.topMd || left.bottomMd - right.bottomMd);
  if (!parsed.length) throw new Error("裂缝层段 CSV 没有可展示的有效深度段");
  if (parsed.some((segment, index) => index > 0 && segment.topMd < parsed[index - 1].bottomMd - 1e-6)) {
    throw new Error("裂缝层段 CSV 存在重叠深度段");
  }
  const depthMin = Math.min(...parsed.map((segment) => segment.topMd));
  const depthMax = Math.max(...parsed.map((segment) => segment.bottomMd));
  const depthSpan = depthMax - depthMin || 1;
  return {
    depthMin,
    depthMax,
    segments: parsed.map((segment) => ({
      ...segment,
      y: 10 + ((segment.topMd - depthMin) / depthSpan) * 620,
      height: Math.max(2, ((segment.bottomMd - segment.topMd) / depthSpan) * 620),
    })),
  };
}

async function readResponsePrefix(response: Response, maximumBytes: number): Promise<{ text: string; limited: boolean }> {
  const reader = response.body?.getReader();
  const contentLength = Number(response.headers.get("content-length") || "");
  const contentRange = response.headers.get("content-range") || "";
  const rangeMatch = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(contentRange.trim());
  const rangeIncomplete = Boolean(
    rangeMatch
    && rangeMatch[3] !== "*"
    && Number(rangeMatch[2]) + 1 < Number(rangeMatch[3]),
  );
  if (!reader) {
    const text = await response.text();
    const encodedSize = new TextEncoder().encode(text).byteLength;
    if (encodedSize > maximumBytes || rangeIncomplete) {
      return { text: text.slice(0, maximumBytes), limited: true };
    }
    return { text, limited: false };
  }
  const chunks: Uint8Array[] = [];
  let total = 0;
  let limited = rangeIncomplete || (Number.isFinite(contentLength) && contentLength > maximumBytes);
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    const remaining = maximumBytes - total;
    if (value.byteLength > remaining) {
      if (remaining > 0) chunks.push(value.slice(0, remaining));
      total = maximumBytes;
      limited = true;
      await reader.cancel();
      break;
    }
    chunks.push(value);
    total += value.byteLength;
    if (total >= maximumBytes) {
      const declaredComplete = !rangeIncomplete
        && Number.isFinite(contentLength)
        && contentLength <= maximumBytes;
      if (!declaredComplete) {
        limited = true;
        await reader.cancel();
        break;
      }
    }
  }
  const payload = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    payload.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { text: new TextDecoder("utf-8").decode(payload), limited };
}

async function loadSequencePreview() {
  const group = selectedAcceptanceGroup.value;
  const artifact = group?.sequenceArtifacts.find((item) => item.name === selectedSequenceArtifact.value);
  if (!group || !artifact) return;
  sequenceAbortController?.abort();
  const controller = new AbortController();
  sequenceAbortController = controller;
  const loadSequence = ++sequenceLoadSequence;
  const snapshotId = loadedAcceptanceSnapshotId.value;
  const taskIdValue = group.taskId;
  sequenceLoading.value = true;
  sequenceError.value = "";
  sequencePreviewLimited.value = false;
  sequencePreview.value = null;
  fractureIntervalPreview.value = null;
  try {
    const response = await fetch(taskArtifactUrl(taskIdValue, artifact.name), {
      headers: { Range: "bytes=0-2097151" },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`证据读取失败（HTTP ${response.status}）`);
    const prefix = await readResponsePrefix(response, 2 * 1024 * 1024);
    if (
      loadSequence !== sequenceLoadSequence
      || snapshotId !== loadedAcceptanceSnapshotId.value
      || taskIdValue !== selectedAcceptanceGroup.value?.taskId
    ) return;
    if (selectedAcceptanceIsFracture.value) {
      fractureIntervalPreview.value = parseFractureIntervalCsv(prefix.text);
    } else {
      sequencePreview.value = parseSequenceCsv(prefix.text);
    }
    sequencePreviewLimited.value = prefix.limited;
  } catch (error) {
    if (loadSequence !== sequenceLoadSequence || (error as { name?: string }).name === "AbortError") return;
    sequenceError.value = error instanceof Error ? error.message : "无法读取沿深度结果";
  } finally {
    if (loadSequence === sequenceLoadSequence) {
      sequenceLoading.value = false;
      sequenceAbortController = null;
    }
  }
}

function resetSequenceSelection() {
  sequenceLoadSequence += 1;
  sequenceAbortController?.abort();
  sequenceAbortController = null;
  sequencePreview.value = null;
  fractureIntervalPreview.value = null;
  sequenceError.value = "";
  sequencePreviewLimited.value = false;
  sequenceLoading.value = false;
}

function finiteCoordinate(value: unknown): number | null {
  if (typeof value !== "number" && (typeof value !== "string" || !value.trim())) return null;
  const coordinate = typeof value === "number" ? value : Number(value.trim());
  return Number.isFinite(coordinate) ? coordinate : null;
}

function parsePlanViewPreview(text: string): PlanViewPreview {
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error("水平配准平面图不是有效 JSON");
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("水平配准平面图结构无效");
  }
  const root = payload as Record<string, unknown>;
  if (root.contract_version !== "well-seismic.horizontal-registration-visualization.v1") {
    throw new Error("水平配准平面图合同版本不受支持");
  }
  const coordinateReference = root.coordinate_reference && typeof root.coordinate_reference === "object"
    ? root.coordinate_reference as Record<string, unknown>
    : {};
  const horizontalCrs = typeof coordinateReference.horizontal_crs_id === "string"
    ? coordinateReference.horizontal_crs_id.trim()
    : "";
  if (!horizontalCrs) throw new Error("水平配准平面图缺少水平 CRS 标识");

  const polygons = (Array.isArray(root.seismic_grids) ? root.seismic_grids : []).flatMap((rawGrid) => {
    if (!rawGrid || typeof rawGrid !== "object" || Array.isArray(rawGrid)) return [];
    const rawFootprint = (rawGrid as Record<string, unknown>).footprint_xy_m;
    if (!Array.isArray(rawFootprint)) return [];
    const polygon = rawFootprint.flatMap((rawPoint) => {
      if (!Array.isArray(rawPoint) || rawPoint.length < 2) return [];
      const x = finiteCoordinate(rawPoint[0]);
      const y = finiteCoordinate(rawPoint[1]);
      return x === null || y === null ? [] : [{ x, y }];
    });
    return polygon.length >= 3 ? [polygon] : [];
  });

  const wells = (Array.isArray(root.wells) ? root.wells : []).flatMap((rawWell) => {
    if (!rawWell || typeof rawWell !== "object" || Array.isArray(rawWell)) return [];
    const well = rawWell as Record<string, unknown>;
    const name = typeof well.well_name === "string" ? well.well_name.trim() : "";
    const rawStations = Array.isArray(well.path) ? well.path : [];
    const stations = rawStations.flatMap((rawStation) => {
      if (!rawStation || typeof rawStation !== "object" || Array.isArray(rawStation)) return [];
      const station = rawStation as Record<string, unknown>;
      const x = finiteCoordinate(station.x_m);
      const y = finiteCoordinate(station.y_m);
      if (x === null || y === null) return [];
      return [{
        x,
        y,
        traceX: finiteCoordinate(station.trace_x_m),
        traceY: finiteCoordinate(station.trace_y_m),
        covered: station.covered === true,
      }];
    });
    if (!name || !stations.length) return [];
    const stride = Math.max(1, Math.ceil(stations.length / 240));
    const sampled = stations.filter((_, index) => index % stride === 0 || index === stations.length - 1);
    return [{
      name,
      geometryMode: typeof well.geometry_mode === "string" ? well.geometry_mode : "unknown",
      horizontalStatus: typeof well.horizontal_status === "string" ? well.horizontal_status : "unknown",
      stations: sampled,
    }];
  });
  if (!polygons.length && !wells.length) throw new Error("水平配准平面图没有可绘制的地震范围或井位");

  const coordinates = [
    ...polygons.flat(),
    ...wells.flatMap((well) => well.stations.flatMap((station) => [
      { x: station.x, y: station.y },
      ...(station.traceX === null || station.traceY === null ? [] : [{ x: station.traceX, y: station.traceY }]),
    ])),
  ];
  const xValues = coordinates.map((point) => point.x);
  const yValues = coordinates.map((point) => point.y);
  const rawXMin = Math.min(...xValues);
  const rawXMax = Math.max(...xValues);
  const rawYMin = Math.min(...yValues);
  const rawYMax = Math.max(...yValues);
  const xPadding = Math.max((rawXMax - rawXMin) * 0.04, 1);
  const yPadding = Math.max((rawYMax - rawYMin) * 0.04, 1);
  return {
    horizontalCrs,
    polygons,
    wells,
    bounds: {
      xMin: rawXMin - xPadding,
      xMax: rawXMax + xPadding,
      yMin: rawYMin - yPadding,
      yMax: rawYMax + yPadding,
    },
  };
}

async function loadPlanViewPreview() {
  const group = selectedAcceptanceGroup.value;
  const artifact = selectedPlanViewArtifact.value;
  if (!group || !artifact) return;
  planViewAbortController?.abort();
  const controller = new AbortController();
  planViewAbortController = controller;
  const loadSequence = ++planViewLoadSequence;
  const snapshotId = loadedAcceptanceSnapshotId.value;
  const taskIdValue = group.taskId;
  planViewLoading.value = true;
  planViewError.value = "";
  planViewPreview.value = null;
  try {
    const response = await fetch(taskArtifactUrl(taskIdValue, artifact.name), {
      headers: { Range: "bytes=0-8388607" },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`平面图证据读取失败（HTTP ${response.status}）`);
    const prefix = await readResponsePrefix(response, 8 * 1024 * 1024);
    if (prefix.limited) throw new Error("水平配准平面图超过 8 MiB 安全读取上限");
    if (
      loadSequence !== planViewLoadSequence
      || snapshotId !== loadedAcceptanceSnapshotId.value
      || taskIdValue !== selectedAcceptanceGroup.value?.taskId
    ) return;
    planViewPreview.value = parsePlanViewPreview(prefix.text);
  } catch (error) {
    if (loadSequence !== planViewLoadSequence || (error as { name?: string }).name === "AbortError") return;
    planViewError.value = error instanceof Error ? error.message : "无法读取水平配准平面图";
  } finally {
    if (loadSequence === planViewLoadSequence) {
      planViewLoading.value = false;
      planViewAbortController = null;
    }
  }
}

function planViewX(x: number): number {
  const bounds = planViewPreview.value?.bounds;
  if (!bounds) return 0;
  return 30 + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin || 1)) * 940;
}

function planViewY(y: number): number {
  const bounds = planViewPreview.value?.bounds;
  if (!bounds) return 0;
  return 570 - ((y - bounds.yMin) / (bounds.yMax - bounds.yMin || 1)) * 540;
}

function planViewPolygonPath(polygon: Array<{ x: number; y: number }>): string {
  return polygon.map((point, index) =>
    `${index ? "L" : "M"}${planViewX(point.x).toFixed(2)} ${planViewY(point.y).toFixed(2)}`,
  ).join(" ") + " Z";
}

function planViewWellPath(well: PlanViewWell): string {
  return well.stations.map((station, index) =>
    `${index ? "L" : "M"}${planViewX(station.x).toFixed(2)} ${planViewY(station.y).toFixed(2)}`,
  ).join(" ");
}

function sequenceCurvePath(series: SequenceSeries): string {
  const preview = sequencePreview.value;
  if (!preview?.depths.length) return "";
  const depthMin = Math.min(...preview.depths);
  const depthMax = Math.max(...preview.depths);
  const valueSpan = series.max - series.min || 1;
  const depthSpan = depthMax - depthMin || 1;
  let path = "";
  let connected = false;
  series.values.forEach((value, index) => {
    if (value === null) {
      connected = false;
      return;
    }
    const x = 8 + ((value - series.min) / valueSpan) * 84;
    const y = 10 + ((preview.depths[index] - depthMin) / depthSpan) * 620;
    path += `${connected ? " L" : " M"}${x.toFixed(2)} ${y.toFixed(2)}`;
    connected = true;
  });
  return path.trim();
}

function flowStageArtifacts(stageId: FlowStagePresentation["id"]): FlowStageArtifact[] {
  if (stageId === "snapshot" && dataSnapshotTaskId.value) {
    return [{
      taskId: dataSnapshotTaskId.value,
      name: "snapshot_manifest",
      path: "snapshot_manifest.json",
      label: "快照清单",
      directory: false,
    }];
  }
  const source = stageId === "registration"
    ? { taskId: registrationTaskId.value, entries: registrationOutputEntries.value }
    : stageId === "prepared"
      ? { taskId: sampleBuildingTaskId.value, entries: preparedViewOutputEntries.value }
      : stageId === "artifacts"
        ? { taskId: predictionTaskId.value, entries: predictionOutputEntries.value }
        : null;
  if (!source?.taskId) return [];
  return source.entries.map(([name, path]) => ({
    taskId: source.taskId,
    name,
    path,
    label: artifactDisplayName(name, path),
    directory: isDirectoryOutput(name, path),
  }));
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

function isFullVolumeFaultPrediction(result: PredictionResult | null): boolean {
  if (!result || !isFaultVolumeModelId(result.model_id)) return false;
  const restoredFaultScope = faultPredictionScope(result);
  const restoredSpatialReceipt = result.standard_spatial_export;
  if (
    restoredSpatialReceipt?.is_full_survey === true
    || restoredSpatialReceipt?.scope === "full_survey"
  ) return true;
  if (
    restoredSpatialReceipt
    && (
      restoredSpatialReceipt.is_full_survey === false
      || Boolean(restoredSpatialReceipt.scope)
      || Boolean(restoredSpatialReceipt.coverage)
    )
  ) return false;
  return restoredFaultScope === "full_volume";
}

function faultPredictionScope(result: PredictionResult | null): string {
  return String(
    result?.inference?.faultseg_scope
    || result?.inference?.scope
    || "",
  ).trim().toLowerCase();
}

function isRepresentativeGrid128FaultPrediction(result: PredictionResult | null): boolean {
  return Boolean(
    result
    && isFaultVolumeModelId(result.model_id)
    && faultPredictionScope(result) === "representative_grid_128",
  );
}

function isCenterBlockFaultPrediction(result: PredictionResult | null): boolean {
  return Boolean(
    result
    && result.model_id === "faultseg_3d"
    && faultPredictionScope(result) === "center_block_1",
  );
}

function isCurrentFaultPrediction(result: PredictionResult | null): boolean {
  return isFullVolumeFaultPrediction(result)
    || isCenterBlockFaultPrediction(result);
}

function rememberPredictionHistory(
  executionTaskId: string,
  result: PredictionTaskResult,
  updatedAt = Date.now(),
): boolean {
  const taskKey = result.prediction.task_id || "";
  if (!predictionTaskDefinitions.value.some((definition) => definition.id === taskKey)) return false;
  const existing = predictionHistoryByTask.value[taskKey];
  if (existing && existing.updatedAt > updatedAt) return false;
  predictionHistoryByTask.value = {
    ...predictionHistoryByTask.value,
    [taskKey]: { executionTaskId, result, updatedAt },
  };
  if (!predictionHistorySnapshotId.value) {
    predictionHistorySnapshotId.value = dataSnapshotTaskId.value;
  }
  return true;
}

function indexPredictionHistory(tasks: readonly BackgroundTask[], snapshotId: string) {
  if (predictionHistorySnapshotId.value !== snapshotId) {
    predictionHistoryByTask.value = {};
    predictionHistorySnapshotId.value = snapshotId;
  }
  predictionHistoryByTask.value = collectLatestPredictionRuns<PredictionResult>(
    tasks,
    predictionTaskDefinitions.value.map((definition) => definition.id),
  );
}

function applyPredictionHistoryEntry(
  task: PredictionTaskKey,
  entry: PredictionRunHistoryEntry<PredictionResult>,
) {
  activePredictionTask.value = task;
  visualizationSourceTaskId.value = "";
  const result = entry.result as PredictionTaskResult;
  const sourceTaskId = result.source_task_id || dataSnapshotTaskId.value || taskId.value;
  visualizationBaseTaskId.value = sourceTaskId;
  selectedSeismicAssetIndex.value = 0;
  predictionSourceTaskId.value = sourceTaskId;
  predictionTaskId.value = entry.executionTaskId;
  predictionResult.value = result.prediction;
  selectedPredictionModelId.value = task === "horizon"
    ? SURFACE_SEG_MODEL_ID
    : selectablePredictionModelId(task, result.prediction.model_id);
  if (isFaultSegModel.value) {
    faultSegScope.value = normalizeFaultSegScope(result.prediction.inference?.faultseg_scope);
    predictionDevice.value = "auto";
    if (!isCurrentFaultPrediction(result.prediction)) {
      statusMessage.value = "历史代表块结果仅供查看；请重新运行中心单块或全区识别";
    }
  }
  if (isSurfaceSegModel.value) resetSurfaceSegDefaults();
  predictionCanvasMode.value = "result";
  try {
    window.sessionStorage.setItem(LAST_PREDICTION_TASK_STORAGE_KEY, entry.executionTaskId);
  } catch {
    appendRestorationWarning("浏览器未能保存当前预测成果引用；本页任务切换仍可使用");
  }
}

function selectDefaultPredictionModel() {
  selectedPredictionModelId.value = (
    (activePredictionTask.value === "facies_3d"
      ? runnablePredictionModels.value.find((model) => model.id === "wellfuse_facies_3d_f3_fast")
      : null)
    || runnablePredictionModels.value[0]
  )?.id || "";
  handlePredictionModelChange();
}

async function selectPredictionTask(task: PredictionTaskKey) {
  const selectedEntry = predictionHistoryByTask.value[task];
  if (
    activePredictionTask.value === task
    && predictionResult.value?.task_id === task
    && predictionTaskId.value === selectedEntry?.executionTaskId
  ) {
    centerPredictionTaskTab(task);
    return;
  }

  const selectionSequence = ++predictionTaskSelectionSequence;
  predictionTaskSwitching.value = false;
  activePredictionTask.value = task;
  if (selectedEntry) {
    applyPredictionHistoryEntry(task, selectedEntry);
    predictionTaskSwitching.value = true;
    try {
      const refreshedTask = await getTask(selectedEntry.executionTaskId);
      if (
        selectionSequence !== predictionTaskSelectionSequence
        || activePredictionTask.value !== task
        || refreshedTask.status !== "completed"
        || !refreshedTask.result
        || !("prediction" in refreshedTask.result)
      ) return;
      const refreshedResult = refreshedTask.result as PredictionTaskResult;
      rememberPredictionHistory(refreshedTask.task_id, refreshedResult, taskUpdatedAt(refreshedTask));
      const refreshedEntry = predictionHistoryByTask.value[task];
      if (refreshedEntry) applyPredictionHistoryEntry(task, refreshedEntry);
    } catch (error) {
      const reason = error instanceof Error ? error.message : "未知错误";
      appendRestorationWarning(`任务成果刷新失败，继续显示已缓存结果：${reason}`);
    } finally {
      if (selectionSequence === predictionTaskSelectionSequence) {
        predictionTaskSwitching.value = false;
      }
    }
    return;
  }

  visualizationSourceTaskId.value = "";
  visualizationBaseTaskId.value = dataSnapshotTaskId.value || taskId.value;
  selectedSeismicAssetIndex.value = 0;
  predictionSourceTaskId.value = dataSnapshotTaskId.value || taskId.value;
  predictionTaskId.value = "";
  predictionResult.value = null;
  predictionCanvasMode.value = "base";
  try {
    window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
  } catch {
    // Session storage is only a restoration hint.
  }
  selectDefaultPredictionModel();
}

function predictionTaskTabStatusLabel(task: PredictionTaskKey): string {
  if (runningPredictionTask.value === task && predictionBusy.value) return "运行中";
  if (activePredictionTask.value === task && predictionTaskSwitching.value) return "载入成果";
  const historyResult = predictionHistoryByTask.value[task]?.result.prediction;
  if (historyResult && isFaultVolumeModelId(historyResult.model_id) && !isCurrentFaultPrediction(historyResult)) return "旧抽样 · 需重跑";
  if (historyResult) return "有结果";
  const specification = predictionTaskDefinitions.value.find((item) => item.id === task);
  if (
    faultSegSnapshotSourceReady.value
    && specification?.runnable_model_ids.some((modelId) => isFaultVolumeModelId(modelId))
  ) return "可运行";
  if (!preparedViewReady.value) return "待融合";
  return specification?.runnable_model_ids.length ? "可运行" : "待接入";
}

function centerPredictionTaskTab(task: PredictionTaskKey, behavior: ScrollBehavior = "smooth") {
  void nextTick(() => {
    window.requestAnimationFrame(() => {
      const rail = predictionTaskRailElement.value;
      const button = Array.from(rail?.querySelectorAll<HTMLButtonElement>("[data-prediction-task-id]") || [])
        .find((candidate) => candidate.dataset.predictionTaskId === task);
      if (!rail || !button) return;
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      rail.scrollTo({
        top: centeredTaskScrollTop(
          rail.clientHeight,
          rail.scrollHeight,
          button.offsetTop,
          button.offsetHeight,
        ),
        behavior: reducedMotion ? "auto" : behavior,
      });
    });
  });
}

function handlePredictionTaskTabTransitionEnd(event: TransitionEvent, task: PredictionTaskKey) {
  if (activePredictionTask.value !== task || event.propertyName !== "min-height") return;
  centerPredictionTaskTab(task);
}

async function handlePredictionTaskTabKeydown(event: KeyboardEvent, task: PredictionTaskKey) {
  const definitions = predictionTaskDefinitions.value;
  const currentIndex = definitions.findIndex((definition) => definition.id === task);
  if (currentIndex < 0) return;
  let nextIndex: number | null = null;
  if (event.key === "ArrowUp") nextIndex = (currentIndex - 1 + definitions.length) % definitions.length;
  if (event.key === "ArrowDown") nextIndex = (currentIndex + 1) % definitions.length;
  if (event.key === "Home") nextIndex = 0;
  if (event.key === "End") nextIndex = definitions.length - 1;
  if (nextIndex === null) return;
  event.preventDefault();
  const nextTask = definitions[nextIndex]?.id;
  if (!nextTask) return;
  await selectPredictionTask(nextTask);
  await nextTick();
  const button = Array.from(
    predictionTaskRailElement.value?.querySelectorAll<HTMLButtonElement>("[data-prediction-task-id]") || [],
  ).find((candidate) => candidate.dataset.predictionTaskId === nextTask);
  button?.focus({ preventScroll: true });
  centerPredictionTaskTab(nextTask);
}

watch(
  [activeView, activePredictionTask, () => predictionTaskDefinitions.value.length],
  ([view, task]) => {
    if (view === "prediction" && task) centerPredictionTaskTab(String(task));
  },
  { flush: "post" },
);

watch(
  [activeView, layerPulseSelectedOutputKey],
  ([view, outputKey]) => {
    if (view === "layerpulse" && outputKey) centerLayerPulseOutputTab(String(outputKey));
  },
  { flush: "post" },
);

function openPredictionAfterFusion() {
  if (predictionEntryReady.value) {
    selectView("prediction");
    return;
  }
  selectView(isFaultSegModel.value ? "preparation" : "samples");
  statusMessage.value = isFaultSegModel.value
    ? "断层识别等待 SourceSnapshot 唯一三维 SEG-Y"
    : "预测入口将在井震精细标定与融合视图完成后开放";
  errorMessage.value = isFaultSegModel.value
    ? faultSegSnapshotSourceReason.value
    : "请先完成井震融合；当前模型不提供跳过融合的预测路径。";
}

function recommendedPreparationModelId(taskId: string): string {
  const task = predictionTaskDefinitions.value.find((item) => item.id === taskId);
  if (!task) return "";
  const availableModelIds = task.runnable_model_ids.filter((modelId) =>
    preparationDraftModels.value.some((model) => model.id === modelId),
  );
  return availableModelIds.find((modelId) => modelId.includes("_fast"))
    || (task.model_id && availableModelIds.includes(task.model_id) ? task.model_id : "")
    || availableModelIds[0]
    || "";
}

async function handlePreparationNextAction() {
  if (
    !preparedViewReady.value
    && !(isFaultVolumeModelId(currentScopeModelId.value) && faultSegSnapshotSourceReady.value)
  ) {
    showPreparationFusion();
    statusMessage.value = "预测入口将在井震精细标定与融合视图完成后开放";
    return;
  }
  if (!hasExplicitPreparationScope.value || !currentScopeModelId.value) {
    if (preparedViewReady.value) openPostFusionInferenceDestination();
    else selectView("prediction");
    return;
  }
  const taskKey = preparation.value?.task_readiness?.task_id || "";
  if (predictionTaskDefinitions.value.some((task) => task.id === taskKey)) {
    openReleaseRunner(taskKey, currentScopeModelId.value);
    return;
  }
  selectView("prediction");
}

function openReleaseRunner(task: string, modelId: string) {
  if (modelId === GEOPATH_TIE_MODEL_ID) {
    selectView("samples");
    statusMessage.value = "已打开轨迹感知实验候选与人工审核流程";
    return;
  }
  const taskSpec = predictionTaskDefinitions.value.find((item) => item.id === task);
  if (!taskSpec || !taskSpec.runnable_model_ids.includes(modelId)) {
    errorMessage.value = "该发布没有可用的在线运行合同。";
    return;
  }
  selectPredictionTask(task as PredictionTaskKey);
  selectedPredictionModelId.value = modelId;
  handlePredictionModelChange();
  openPredictionAfterFusion();
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
  const hash = window.location.hash.replace("#", "");
  if (hash === "visualization-logs") {
    visualizationMode.value = "logs";
    activeView.value = "visualization";
    sidebarDirectoryLevel.value = "primary";
    return;
  }
  const view = hash as ViewKey;
  if (view === "overview" || view === "samples" || view === "visualization" || navigation.some((item) => item.id === view)) {
    if (view === "samples") preparationScreen.value = "fusion";
    if (view === "visualization") predictionCanvasMode.value = "base";
    const nextView = canonicalView(view);
    if (nextView === "prediction" && !predictionEntryReady.value) {
      sidebarDirectoryLevel.value = "primary";
      preparationScreen.value = isFaultSegModel.value ? "input" : "fusion";
      activeView.value = "preparation";
      window.history.replaceState(null, "", "#preparation");
      statusMessage.value = isFaultSegModel.value
        ? "断层识别等待 SourceSnapshot 唯一三维 SEG-Y"
        : "预测入口将在井震精细标定与融合视图完成后开放";
      errorMessage.value = isFaultSegModel.value
        ? faultSegSnapshotSourceReason.value
        : "请先完成井震融合；当前模型不提供跳过融合的预测路径。";
      return;
    }
    activeView.value = nextView;
    sidebarDirectoryLevel.value = nextView === "prediction"
      ? "prediction"
      : nextView === "layerpulse"
        ? "layerpulse"
        : "primary";
    if (nextView !== view) window.history.replaceState(null, "", `#${nextView}`);
    if (nextView === "evaluation" && backendStatus.value === "online") void refreshAcceptanceSnapshot();
  }
}

function predictionParentTaskId(task: BackgroundTask, result: PredictionTaskResult): string {
  const taskWithParent = task as BackgroundTask & {
    source_task_id?: string;
    parent_task_id?: string;
  };
  return result.source_task_id || taskWithParent.source_task_id || taskWithParent.parent_task_id || "";
}

async function restorePreparationForSnapshot(
  snapshotId: string,
  options: {
    resetDownstream?: boolean;
    updateAcceptance?: boolean;
    persistActiveSnapshot?: boolean;
    offerRuntimeReview?: boolean;
  } = {},
): Promise<SnapshotDetailResponse> {
  const detail = await getSnapshotDetail(snapshotId);
  activeProjectId.value = detail.snapshot.project_id || activeProjectId.value;
  if (options.updateAcceptance) {
    acceptanceSnapshotDetail.value = detail;
    selectedAcceptanceSnapshotId.value = detail.snapshot.snapshot_id;
    loadedAcceptanceSnapshotId.value = detail.snapshot.snapshot_id;
  }
  const completed = detail.tasks
    .filter((task) => task.status === "completed")
    .sort((left, right) => taskUpdatedAt(right) - taskUpdatedAt(left));
  const preparationTaskId = detail.snapshot.created_by_task_id
    || completed.find((task) => task.task_type === "data_preparation")?.task_id;
  if (preparationTaskId) {
    const preparationTask = await getTask(preparationTaskId);
    if (preparationTask.status === "completed" && isWorkflowResult(preparationTask.result)) {
      restoreCompletedDataPreparationTask(preparationTask, {
        snapshotId: detail.snapshot.snapshot_id,
        resetDownstream: options.resetDownstream ?? false,
        persistActiveSnapshot: options.persistActiveSnapshot ?? false,
        offerRuntimeReview: options.offerRuntimeReview ?? true,
      });
    }
  } else {
    dataSnapshotTaskId.value = detail.snapshot.snapshot_id;
    rememberSourceSnapshot(detail.snapshot.snapshot_id, detail.snapshot.project_id, {
      persistActiveSnapshot: options.persistActiveSnapshot ?? false,
    });
  }
  indexPredictionHistory(detail.tasks, detail.snapshot.snapshot_id);
  return detail;
}

async function restorePredictionSource(sourceTaskId: string) {
  if (!sourceTaskId) return;
  predictionSourceTaskId.value = sourceTaskId;
  visualizationBaseTaskId.value = sourceTaskId;
  if (taskId.value === sourceTaskId && workflowResult.value) return;
  try {
    const sourceTask = await getTask(sourceTaskId);
    if (sourceTask.status !== "completed" || !sourceTask.result || "prediction" in sourceTask.result) return;
    if (sourceTask.task_type === "data_preparation" && isWorkflowResult(sourceTask.result)) {
      // Restore the whole sealed snapshot, not only its preparation payload, so
      // completed sibling predictions remain available in the task directory.
      await restorePreparationForSnapshot(sourceTaskId, { resetDownstream: false });
      return;
    } else if (sourceTask.task_type === "well_tie" && isWorkflowResult(sourceTask.result)) {
      taskId.value = sourceTask.task_id;
      registrationTaskId.value = sourceTask.task_id;
      dataSnapshotTaskId.value = sourceTask.result.registration?.source_snapshot_id || "";
      rememberSourceSnapshot(dataSnapshotTaskId.value);
      registrationResult.value = sourceTask.result;
      horizontalRegistrationTaskId.value = "";
      horizontalRegistrationResult.value = null;
    } else if (sourceTask.task_type === "sample_building" && isWorkflowResult(sourceTask.result)) {
      taskId.value = sourceTask.task_id;
      sampleBuildingTaskId.value = sourceTask.task_id;
      dataSnapshotTaskId.value = sourceTask.result.source_snapshot_id || "";
      rememberSourceSnapshot(dataSnapshotTaskId.value);
      registrationTaskId.value = sourceTask.result.registration_task_id || "";
      sampleResult.value = sourceTask.result;
    } else if (
      sourceTask.task_type === "horizontal_registration"
      && isHorizontalRegistrationTaskResult(sourceTask.result)
    ) {
      taskId.value = sourceTask.task_id;
      horizontalRegistrationTaskId.value = sourceTask.task_id;
      dataSnapshotTaskId.value = sourceTask.result.horizontal_registration.source_snapshot_id;
      rememberSourceSnapshot(dataSnapshotTaskId.value);
      registrationTaskId.value = "";
      registrationResult.value = null;
      sampleBuildingTaskId.value = "";
      sampleResult.value = null;
      horizontalRegistrationResult.value = sourceTask.result;
      if (!preparationResult.value && dataSnapshotTaskId.value) {
        await restorePreparationForSnapshot(dataSnapshotTaskId.value);
      }
    }
    initializeVisualization(isWorkflowResult(sourceTask.result) ? sourceTask.result : null);
  } catch (error) {
    // Keep the source id for the visualization endpoint even if the archived task summary is unavailable.
    const reason = error instanceof Error ? error.message : "未知错误";
    appendRestorationWarning(`预测来源 ${sourceTaskId.slice(0, 8)} 恢复不完整：${reason}`);
  }
}

async function restorePredictionTask(task: BackgroundTask) {
  if (isLayerPulsePredictionTask(task)) {
    await restoreLayerPulseTask(task);
    return;
  }
  if (task.status !== "completed" || !task.result || !("prediction" in task.result)) return;
  const result = task.result as PredictionTaskResult;
  const restoredTaskId = result.prediction.task_id || "";
  if (!predictionTaskDefinitions.value.some((definition) => definition.id === restoredTaskId)) {
    window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
    return;
  }
  rememberPredictionHistory(task.task_id, result, taskUpdatedAt(task));
  predictionTaskId.value = task.task_id;
  predictionResult.value = result.prediction;
  if (!isFaultVolumeModelId(result.prediction.model_id)) {
    registrationTaskId.value = result.registration_task_id || registrationTaskId.value;
    sampleBuildingTaskId.value = result.prepared_view_task_id || sampleBuildingTaskId.value;
  }
  activePredictionTask.value = restoredTaskId;
  selectedPredictionModelId.value = restoredTaskId === "horizon"
    ? SURFACE_SEG_MODEL_ID
    : selectablePredictionModelId(restoredTaskId, result.prediction.model_id);
  if (isFaultSegModel.value) {
    faultSegScope.value = normalizeFaultSegScope(result.prediction.inference?.faultseg_scope);
    predictionDevice.value = "auto";
    if (!isCurrentFaultPrediction(result.prediction)) {
      const notice = "历史代表块结果仅供查看；请重新运行中心单块或全区识别";
      statusMessage.value = notice;
      appendRestorationWarning(notice);
    }
  }
  if (isSurfaceSegModel.value) resetSurfaceSegDefaults();
  predictionCanvasMode.value = "result";
  await restorePredictionSource(predictionParentTaskId(task, result));
}

function taskUpdatedAt(task: BackgroundTask): number {
  const value = Date.parse(task.updated_at || task.created_at || "");
  return Number.isFinite(value) ? value : 0;
}

function snapshotUpdatedAt(snapshot: SourceSnapshotSummary): number {
  const value = Date.parse(snapshot.updated_at || snapshot.created_at || "");
  return Number.isFinite(value) ? value : 0;
}

async function newestSameSourceSnapshotId(sourceSnapshotId: string): Promise<string> {
  if (!sourceSnapshotId) return "";
  let sourceDetail: SnapshotDetailResponse;
  try {
    sourceDetail = await getSnapshotDetail(sourceSnapshotId);
  } catch {
    return sourceSnapshotId;
  }
  const projectId = sourceDetail.snapshot.project_id || activeProjectId.value;
  if (!projectId) return sourceSnapshotId;
  const catalog = await getProjectSnapshots(projectId);
  const source = catalog.snapshots.find((snapshot) => snapshot.snapshot_id === sourceSnapshotId)
    || sourceDetail.snapshot;
  const sourceContentSha256 = source.hashes?.source_content_sha256;
  if (!sourceContentSha256) return sourceSnapshotId;
  const sameSourceSnapshots = catalog.snapshots
    .filter((snapshot) =>
      snapshot.state === "sealed"
      && snapshot.contract_version === "well-seismic.source-snapshot.v3"
      && snapshot.hashes?.source_content_sha256 === sourceContentSha256,
    )
    .sort((left, right) => snapshotUpdatedAt(right) - snapshotUpdatedAt(left));
  return sameSourceSnapshots[0]?.snapshot_id || sourceSnapshotId;
}

async function adoptNewestSameSourceSnapshot(sourceSnapshotId: string): Promise<string> {
  let preferredSnapshotId = sourceSnapshotId;
  try {
    preferredSnapshotId = await newestSameSourceSnapshotId(sourceSnapshotId);
  } catch (error) {
    const reason = error instanceof Error ? error.message : "未知错误";
    appendRestorationWarning(`最新同源SourceSnapshot检查失败，将继续核验当前快照：${reason}`);
  }
  if (!preferredSnapshotId || preferredSnapshotId === sourceSnapshotId) return sourceSnapshotId;
  await restorePreparationForSnapshot(preferredSnapshotId, {
    resetDownstream: true,
    updateAcceptance: true,
    persistActiveSnapshot: true,
  });
  statusMessage.value = `已自动切换到同一批原始文件的最新封存快照 ${preferredSnapshotId.slice(0, 8)}，正在核验井震标定合同`;
  return preferredSnapshotId;
}

async function restoreLatestDurableWorkflow(options: { preferLatestSnapshot?: boolean } = {}) {
  let snapshotId = options.preferLatestSnapshot ? "" : dataSnapshotTaskId.value;
  if (!snapshotId) {
    const projectCatalog = await getProjects();
    const project = projectCatalog.projects.find((item) => item.project_id === activeProjectId.value)
      || projectCatalog.projects.find((item) => item.project_id === "local-default")
      || [...projectCatalog.projects].sort((left, right) =>
        Date.parse(right.updated_at || "") - Date.parse(left.updated_at || ""),
      )[0];
    if (!project) return;
    activeProjectId.value = project.project_id;
    const snapshotCatalog = await getProjectSnapshots(project.project_id);
    const sealedSnapshots = snapshotCatalog.snapshots.filter((snapshot) => snapshot.state === "sealed");
    const activeSnapshot = sealedSnapshots.find(
      (snapshot) => snapshot.snapshot_id === project.active_snapshot_id,
    );
    const latestSnapshot = [...sealedSnapshots]
      .sort((left, right) =>
        Date.parse(right.updated_at || right.created_at || "")
        - Date.parse(left.updated_at || left.created_at || ""),
      )[0];
    const selectedSnapshot = activeSnapshot || latestSnapshot;
    if (!selectedSnapshot) return;
    snapshotId = selectedSnapshot.snapshot_id;
  }

  const detail = await restorePreparationForSnapshot(snapshotId, {
    resetDownstream: true,
    updateAcceptance: true,
  });
  const completed = detail.tasks
    .filter((task) => task.status === "completed")
    .sort((left, right) => taskUpdatedAt(right) - taskUpdatedAt(left));

  const latestRegistration = completed.find((task) =>
    task.task_type === "well_tie" && isWorkflowResult(task.result) && task.result.registration,
  );
  const latestHorizontalRegistration = completed.find((task) =>
    task.task_type === "horizontal_registration"
    && isHorizontalRegistrationTaskResult(task.result),
  );
  const horizontalIsLatest = latestHorizontalRegistration
    && (!latestRegistration || taskUpdatedAt(latestHorizontalRegistration) > taskUpdatedAt(latestRegistration));
  if (
    horizontalIsLatest
    && latestHorizontalRegistration?.result
    && isHorizontalRegistrationTaskResult(latestHorizontalRegistration.result)
  ) {
    taskId.value = latestHorizontalRegistration.task_id;
    horizontalRegistrationTaskId.value = latestHorizontalRegistration.task_id;
    registrationTaskId.value = "";
    registrationResult.value = null;
    sampleBuildingTaskId.value = "";
    horizontalRegistrationResult.value = latestHorizontalRegistration.result;
    sampleResult.value = null;
  } else if (latestRegistration?.result && isWorkflowResult(latestRegistration.result)) {
    horizontalRegistrationTaskId.value = "";
    horizontalRegistrationResult.value = null;
    if (latestRegistration.result.registration?.candidate_status === "human_accepted_for_experimental_downstream_use") {
      installAcceptedGeoPathRegistration(latestRegistration.task_id, latestRegistration.result);
    } else {
      registrationTaskId.value = latestRegistration.task_id;
      registrationResult.value = latestRegistration.result;
    }
  }

  const matchingSamples = completed.filter((task) =>
    task.task_type === "sample_building"
    && isWorkflowResult(task.result)
    && task.result.registration_task_id === registrationTaskId.value,
  );
  const latestSample = matchingSamples[0];
  if (latestSample?.result && isWorkflowResult(latestSample.result)) {
    taskId.value = latestSample.task_id;
    sampleBuildingTaskId.value = latestSample.task_id;
    sampleResult.value = latestSample.result;
  } else if (registrationTaskId.value) {
    sampleBuildingTaskId.value = "";
    sampleResult.value = null;
  }

  // The visualization endpoint must stay on the newest derived workflow task
  // after a durable-state refresh.  Otherwise a page refresh silently falls
  // back to the preparation task and drops the registered well overlay.
  const latestVisualizationTask = latestSample
    || (!horizontalIsLatest ? latestRegistration : latestHorizontalRegistration);
  if (latestVisualizationTask) {
    taskId.value = latestVisualizationTask.task_id;
    initializeVisualization(
      isWorkflowResult(latestVisualizationTask.result) ? latestVisualizationTask.result : null,
    );
  }

  const latestGeoPath = completed.find((task) =>
    task.task_type === "model_prediction"
    && task.result
    && "prediction" in task.result
    && task.result.prediction.model_id === GEOPATH_TIE_MODEL_ID,
  );
  if (latestGeoPath?.result && "prediction" in latestGeoPath.result) {
    geoPathCandidateTaskId.value = latestGeoPath.task_id;
    geoPathCandidateResult.value = latestGeoPath.result.prediction;
  }

  const preferredHistoryEntry = predictionHistoryByTask.value[activePredictionTask.value]
    || Object.values(predictionHistoryByTask.value)
      .sort((left, right) => right.updatedAt - left.updatedAt)[0];
  if (preferredHistoryEntry) {
    const restorationSelectionSequence = predictionTaskSelectionSequence;
    try {
      const refreshedPredictionTask = await getTask(preferredHistoryEntry.executionTaskId);
      if (
        refreshedPredictionTask.status === "completed"
        && refreshedPredictionTask.result
        && "prediction" in refreshedPredictionTask.result
      ) {
        rememberPredictionHistory(
          refreshedPredictionTask.task_id,
          refreshedPredictionTask.result as PredictionTaskResult,
          taskUpdatedAt(refreshedPredictionTask),
        );
      }
      if (restorationSelectionSequence === predictionTaskSelectionSequence) {
        await restorePredictionTask(refreshedPredictionTask);
      }
    } catch (error) {
      if (restorationSelectionSequence === predictionTaskSelectionSequence) {
        const taskKey = preferredHistoryEntry.result.prediction.task_id;
        applyPredictionHistoryEntry(taskKey, preferredHistoryEntry);
      }
      const reason = error instanceof Error ? error.message : "未知错误";
      appendRestorationWarning(`历史预测成果刷新失败，继续使用快照内结果：${reason}`);
    }
  }

  const activeBackgroundTask = [...detail.tasks]
    .filter((task) =>
      (task.status === "queued" || task.status === "running")
      && (isReconnectableWorkflowTask(task) || task.task_type === "model_prediction"),
    )
    .sort((left, right) => taskUpdatedAt(right) - taskUpdatedAt(left))[0];
  if (activeBackgroundTask?.task_type === "model_prediction") {
    statusMessage.value = `已恢复快照 ${snapshotId.slice(0, 8)}，正在重新连接未完成的模型推理`;
    void reattachPredictionTask(activeBackgroundTask).catch((error) => {
      const reason = error instanceof Error ? error.message : "未知错误";
      appendRestorationWarning(`未完成预测任务重新连接失败：${reason}`);
    });
    return;
  }
  if (activeBackgroundTask && isReconnectableWorkflowTask(activeBackgroundTask)) {
    statusMessage.value = `已恢复快照 ${snapshotId.slice(0, 8)}，正在重新连接未完成的${acceptanceTaskLabel(activeBackgroundTask)}`;
    void reattachWorkflowTask(activeBackgroundTask).catch((error) => {
      const reason = error instanceof Error ? error.message : "未知错误";
      appendRestorationWarning(`未完成任务重新连接失败：${reason}`);
    });
    return;
  }

  if (completed.length) {
    statusMessage.value = `已从平台状态库恢复快照 ${snapshotId.slice(0, 8)} 的最近已完成阶段`;
  }
}

onMounted(async () => {
  const initialPreparationIntent: PreparationScreen | null = window.location.hash.replace("#", "") === "samples"
    ? "fusion"
    : null;
  const replayInitialViewIntent = () => {
    if (initialPreparationIntent !== "fusion" || activeView.value !== "preparation") return;
    preparationScreen.value = dataSnapshotTaskId.value ? "fusion" : "input";
  };
  componentUnmounted = false;
  syncViewFromHash();
  let freshSessionRequested = false;
  let pathConfigRestored = false;
  let sessionStorageUsable = true;
  let rememberedSourceSnapshotId = "";
  try {
    freshSessionRequested = window.sessionStorage.getItem(FRESH_SESSION_STORAGE_KEY) === "1";
    rememberedSourceSnapshotId = window.sessionStorage.getItem(LAST_SOURCE_SNAPSHOT_STORAGE_KEY) || "";
    // A cache-clear reload skips restoration exactly once. Leaving this marker
    // behind would make every later refresh look like a new empty session.
    if (freshSessionRequested) window.sessionStorage.removeItem(FRESH_SESSION_STORAGE_KEY);
    const cacheFlash = window.sessionStorage.getItem(CACHE_FLASH_STORAGE_KEY);
    if (cacheFlash) {
      cacheMessage.value = cacheFlash;
      window.sessionStorage.removeItem(CACHE_FLASH_STORAGE_KEY);
    }
    if (!freshSessionRequested) pathConfigRestored = restorePathConfig();
  } catch {
    sessionStorageUsable = false;
    appendRestorationWarning("浏览器会话存储不可用，刷新后可能无法恢复未提交的输入");
  }
  window.addEventListener("hashchange", syncViewFromHash);
  document.addEventListener("fullscreenchange", syncVisualizationFullscreen);

  try {
    const service = await health();
    backendStatus.value = "online";
    backendVersion.value = service.version;
  } catch {
    // Only the health request controls the global online/offline indicator.
    // Optional catalogs and restoration failures must not masquerade as an outage.
    backendStatus.value = "offline";
    releaseCatalogLoading.value = false;
    cacheStatusLoading.value = false;
    replayInitialViewIntent();
    return;
  }

  const [demoResult, capabilitiesResult, releaseResult] = await Promise.allSettled([
    demoPaths(),
    getCapabilities(),
    getReleaseCatalog(),
  ]);
  if (demoResult.status === "fulfilled") demo.value = demoResult.value;
  if (capabilitiesResult.status === "fulfilled") {
    capabilities.value = capabilitiesResult.value;
    normalizePreparationTarget();
    // A saved true/false value is an explicit user choice. Capabilities may
    // disable it, but must not turn a saved false back on.
    useLlmFallback.value = pathConfigRestored
      ? useLlmFallback.value && capabilitiesResult.value.llm.available
      : capabilitiesResult.value.llm.available;
  } else {
    const reason = capabilitiesResult.reason instanceof Error
      ? capabilitiesResult.reason.message
      : "未知错误";
    appendRestorationWarning(`平台能力信息加载失败：${reason}`);
  }
  if (releaseResult.status === "fulfilled") {
    releaseCatalog.value = releaseResult.value;
  } else if (capabilitiesResult.status === "fulfilled" && capabilitiesResult.value.artifact_releases) {
    releaseCatalog.value = capabilitiesResult.value.artifact_releases;
  } else {
    releaseCatalogError.value = releaseResult.reason instanceof Error
      ? releaseResult.reason.message
      : "无法读取模型发布目录";
  }
  releaseCatalogLoading.value = false;
  void refreshSystemCacheStatus();
  if (activeView.value === "evaluation") void refreshAcceptanceSnapshot();

  if (freshSessionRequested) {
    replayInitialViewIntent();
    return;
  }

  if (rememberedSourceSnapshotId && !dataSnapshotTaskId.value) {
    dataSnapshotTaskId.value = rememberedSourceSnapshotId;
  }

  let reattachedRunningTask = false;
  const rememberedTaskId = sessionStorageUsable
    ? window.sessionStorage.getItem(LAST_TASK_STORAGE_KEY)
    : null;
  if (rememberedTaskId) {
    let rememberedTask: BackgroundTask | null = null;
    try {
      rememberedTask = await getTask(rememberedTaskId);
    } catch (error) {
      if (error instanceof ApiRequestError && error.status === 404) {
        window.sessionStorage.removeItem(LAST_TASK_STORAGE_KEY);
      } else {
        reattachedRunningTask = true;
        void recoverRememberedTaskReference(rememberedTaskId).catch((recoveryError) => {
          const reason = recoveryError instanceof Error ? recoveryError.message : "未知错误";
          appendRestorationWarning(`后台任务重新连接失败：${reason}`);
        });
      }
    }
    if (rememberedTask) {
      try {
        if (
          rememberedTask.task_type === "data_preparation"
          && (rememberedTask.status === "queued" || rememberedTask.status === "running")
        ) {
          reattachedRunningTask = true;
          void reattachDataPreparationTask(rememberedTask).catch((reattachError) => {
            const reason = reattachError instanceof Error ? reattachError.message : "未知错误";
            appendRestorationWarning(`数据准备任务重新连接失败：${reason}`);
          });
        } else if (
          isReconnectableWorkflowTask(rememberedTask)
          && (rememberedTask.status === "queued" || rememberedTask.status === "running")
        ) {
          reattachedRunningTask = true;
          void reattachWorkflowTask(rememberedTask).catch((reattachError) => {
            const reason = reattachError instanceof Error ? reattachError.message : "未知错误";
            appendRestorationWarning(`井震工作流任务重新连接失败：${reason}`);
          });
        } else if (rememberedTask.status === "completed" && rememberedTask.result) {
          if ("prediction" in rememberedTask.result) {
            await restorePredictionTask(rememberedTask);
          } else if (rememberedTask.task_type === "data_preparation" && isWorkflowResult(rememberedTask.result)) {
            restoreCompletedDataPreparationTask(rememberedTask);
          } else if (
            isReconnectableWorkflowTask(rememberedTask)
            && (isWorkflowResult(rememberedTask.result) || isHorizontalRegistrationTaskResult(rememberedTask.result))
          ) {
            const sourceSnapshotId = workflowTaskSourceSnapshotId(rememberedTask);
            if (!preparationResult.value && sourceSnapshotId) {
              await restorePreparationForSnapshot(sourceSnapshotId);
            }
            installCompletedWorkflowTask(rememberedTask, rememberedTask.result, {
              persistActiveSnapshot: false,
            });
          }
        } else if (
          rememberedTask.status === "failed"
          || rememberedTask.status === "cancelled"
          || (rememberedTask.status === "completed" && !rememberedTask.result)
        ) {
          const fallback = rememberedTask.status === "cancelled"
            ? "之前的后台任务已取消，请重新运行"
            : "之前的后台任务未能恢复，请重新运行";
          errorMessage.value = rememberedTask.error?.message || rememberedTask.message || fallback;
          if (rememberedTask.task_type === "data_preparation") {
            preparationActivityPhase.value = "failed";
            preparationStatusMessage.value = errorMessage.value;
          }
          window.sessionStorage.removeItem(LAST_TASK_STORAGE_KEY);
        }
      } catch (error) {
        const reason = error instanceof Error ? error.message : "未知错误";
        appendRestorationWarning(`已找到历史任务，但页面状态恢复失败：${reason}`);
      }
    }
  }

  if (!reattachedRunningTask) {
    const rememberedPredictionTaskId = sessionStorageUsable
      ? window.sessionStorage.getItem(LAST_PREDICTION_TASK_STORAGE_KEY)
      : null;
    if (rememberedPredictionTaskId) {
      try {
        const rememberedPredictionTask = await getTask(rememberedPredictionTaskId);
        if (
          rememberedPredictionTask.task_type === "model_prediction"
          && (rememberedPredictionTask.status === "queued" || rememberedPredictionTask.status === "running")
        ) {
          reattachedRunningTask = true;
          void reattachPredictionTask(rememberedPredictionTask).catch((reattachError) => {
            const reason = reattachError instanceof Error ? reattachError.message : "未知错误";
            appendRestorationWarning(`预测任务重新连接失败：${reason}`);
          });
        } else if (rememberedPredictionTask.status === "completed") {
          await restorePredictionTask(rememberedPredictionTask);
        } else if (rememberedPredictionTask.status === "failed" || rememberedPredictionTask.status === "cancelled") {
          const failureDetail = rememberedPredictionTask.error?.message
            || rememberedPredictionTask.message
            || "之前的模型推理未完成，请重新运行";
          errorMessage.value = predictionFailureMessage(
            new Error(failureDetail),
            isFaultVolumeModelId(taskRequestString(rememberedPredictionTask, "model_id")),
          );
          window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
        }
      } catch (error) {
        if (error instanceof ApiRequestError && error.status === 404) {
          window.sessionStorage.removeItem(LAST_PREDICTION_TASK_STORAGE_KEY);
        } else {
          const reason = error instanceof Error ? error.message : "未知错误";
          appendRestorationWarning(`预测任务恢复失败：${reason}`);
        }
      }
    }
    const rememberedGeoPathTaskId = sessionStorageUsable
      ? window.sessionStorage.getItem(LAST_GEOPATH_CANDIDATE_TASK_STORAGE_KEY)
      : null;
    if (rememberedGeoPathTaskId) {
      try {
        const rememberedCandidateTask = await getTask(rememberedGeoPathTaskId);
        if (
          !reattachedRunningTask
          && rememberedCandidateTask.task_type === "model_prediction"
          && (rememberedCandidateTask.status === "queued" || rememberedCandidateTask.status === "running")
          && taskRequestString(rememberedCandidateTask, "model_id") === GEOPATH_TIE_MODEL_ID
        ) {
          reattachedRunningTask = true;
          void reattachPredictionTask(rememberedCandidateTask).catch((reattachError) => {
            const reason = reattachError instanceof Error ? reattachError.message : "未知错误";
            appendRestorationWarning(`轨迹感知实验候选重新连接失败：${reason}`);
          });
        } else if (
          rememberedCandidateTask.status === "completed"
          && rememberedCandidateTask.result
          && "prediction" in rememberedCandidateTask.result
          && rememberedCandidateTask.result.prediction.model_id === GEOPATH_TIE_MODEL_ID
        ) {
          geoPathCandidateTaskId.value = rememberedGeoPathTaskId;
          geoPathCandidateResult.value = rememberedCandidateTask.result.prediction;
        } else if (rememberedCandidateTask.status === "failed" || rememberedCandidateTask.status === "cancelled") {
          errorMessage.value = rememberedCandidateTask.error?.message
            || rememberedCandidateTask.message
            || "之前的轨迹感知实验候选任务未完成，请重新运行";
          window.sessionStorage.removeItem(LAST_GEOPATH_CANDIDATE_TASK_STORAGE_KEY);
        }
      } catch (error) {
        if (error instanceof ApiRequestError && error.status === 404) {
          window.sessionStorage.removeItem(LAST_GEOPATH_CANDIDATE_TASK_STORAGE_KEY);
        } else {
          const reason = error instanceof Error ? error.message : "未知错误";
          appendRestorationWarning(`轨迹感知实验候选恢复失败：${reason}`);
        }
      }
    }
    const rememberedAcceptedRegistrationId = sessionStorageUsable
      ? window.sessionStorage.getItem(LAST_GEOPATH_ACCEPTED_REGISTRATION_STORAGE_KEY)
      : null;
    if (rememberedAcceptedRegistrationId) {
      try {
        const rememberedAcceptanceTask = await getTask(rememberedAcceptedRegistrationId);
        if (
          rememberedAcceptanceTask.status === "completed"
          && rememberedAcceptanceTask.result
          && "registration" in rememberedAcceptanceTask.result
        ) {
          installAcceptedGeoPathRegistration(
            rememberedAcceptedRegistrationId,
            rememberedAcceptanceTask.result,
          );
        }
      } catch (error) {
        if (error instanceof ApiRequestError && error.status === 404) {
          window.sessionStorage.removeItem(LAST_GEOPATH_ACCEPTED_REGISTRATION_STORAGE_KEY);
        } else {
          const reason = error instanceof Error ? error.message : "未知错误";
          appendRestorationWarning(`轨迹感知实验候选审核结果恢复失败：${reason}`);
        }
      }
    }
    if (!reattachedRunningTask) {
      try {
        // The project active snapshot is durable authority. Browser session
        // references are restoration hints only and must never pin an older
        // semantic contract over a newer sealed snapshot.
        await restoreLatestDurableWorkflow({ preferLatestSnapshot: true });
      } catch (error) {
        if (rememberedSourceSnapshotId && dataSnapshotTaskId.value === rememberedSourceSnapshotId) {
          window.sessionStorage.removeItem(LAST_SOURCE_SNAPSHOT_STORAGE_KEY);
          dataSnapshotTaskId.value = "";
          appendRestorationWarning("已保存的SourceSnapshot已不可读取，改为选择项目最新封存快照");
          try {
            await restoreLatestDurableWorkflow({ preferLatestSnapshot: true });
          } catch (fallbackError) {
            const reason = fallbackError instanceof Error ? fallbackError.message : "未知错误";
            appendRestorationWarning(`平台历史状态恢复失败：${reason}。当前已恢复内容仍可使用，可刷新后重试`);
          }
        } else {
          const reason = error instanceof Error ? error.message : "未知错误";
          appendRestorationWarning(`平台历史状态恢复失败：${reason}。当前已恢复内容仍可使用，可刷新后重试`);
        }
      }
    }
  }
  if (sessionStorageUsable) {
    await restoreRememberedLayerPulseTask();
  }
  // Restoration may legitimately select the QC stage while rebuilding state.
  // Replay the user's legacy deep-link only after that asynchronous work ends.
  replayInitialViewIntent();
});

onBeforeUnmount(() => {
  componentUnmounted = true;
  sequenceLoadSequence += 1;
  sequenceAbortController?.abort();
  planViewLoadSequence += 1;
  planViewAbortController?.abort();
  stopPreparationClock();
  stopFusionClock();
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

      <nav v-if="sidebarDirectoryLevel === 'primary'" class="navigation directory-rail primary-directory-rail" aria-label="一级目录">
        <template v-for="section in ['workflow', 'system']" :key="section">
          <p class="nav-section">
            {{ section === "workflow" ? "工作流" : "系统" }}
          </p>
          <div
            v-for="item in navigation.filter((nav) => nav.section === section)"
            :key="item.id"
            class="nav-entry"
          >
            <button
              type="button"
              :class="['nav-item', { active: activeView === item.id }]"
              :data-view="item.id"
              :aria-current="activeView === item.id ? 'page' : undefined"
              @click="selectView(item.id)"
            >
              <span class="nav-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24">
                  <path v-for="path in navigationIconPaths[item.id]" :key="path" :d="path" />
                </svg>
              </span>
              <span class="nav-label">{{ item.label }}</span>
              <span v-if="navigationState(item.id)" class="nav-state">{{ navigationState(item.id) }}</span>
              <span class="nav-chevron" aria-hidden="true">{{ item.id === 'prediction' || item.id === 'layerpulse' ? '↗' : '›' }}</span>
            </button>
          </div>
        </template>
      </nav>
      <nav v-else-if="sidebarDirectoryLevel === 'prediction'" class="prediction-directory" aria-label="单任务推理二级目录">
        <header class="prediction-directory-heading">
          <button type="button" class="directory-back" @click="sidebarDirectoryLevel = 'primary'">
            <span aria-hidden="true">←</span>
            <strong>返回一级目录</strong>
          </button>
          <div><span>单任务推理目录</span><strong>共享井震融合基座</strong></div>
        </header>
        <div
          ref="predictionTaskRailElement"
          class="prediction-directory-list"
          role="tablist"
          aria-orientation="vertical"
        >
          <button
            v-for="(task, index) in predictionTaskDefinitions"
            :id="`prediction-task-tab-${task.id}`"
            :key="task.id"
            type="button"
            role="tab"
            class="prediction-task-tab"
            :class="{ active: activePredictionTask === task.id }"
            :data-prediction-task-id="task.id"
            :aria-selected="activePredictionTask === task.id"
            aria-controls="prediction-task-panel"
            :tabindex="task.id === (activePredictionTask || predictionTaskDefinitions[0]?.id) ? 0 : -1"
            @click="selectPredictionTask(task.id)"
            @keydown="handlePredictionTaskTabKeydown($event, task.id)"
            @transitionend="handlePredictionTaskTabTransitionEnd($event, task.id)"
          >
            <span>{{ index + 1 }}</span>
            <strong>{{ task.short_name || task.name }}</strong>
            <small :data-ready="Boolean(predictionHistoryByTask[task.id])">{{ predictionTaskTabStatusLabel(task.id) }}</small>
          </button>
        </div>
      </nav>
      <nav v-else class="prediction-directory layerpulse-directory" aria-label="LayerPulse 任务二级目录">
        <header class="prediction-directory-heading">
          <button type="button" class="directory-back" @click="sidebarDirectoryLevel = 'primary'">
            <span aria-hidden="true">←</span>
            <strong>返回一级目录</strong>
          </button>
          <div><span>LayerPulse 任务目录</span><strong>11 项统一解释</strong></div>
        </header>
        <div
          ref="layerPulseTaskRailElement"
          class="prediction-directory-list"
          role="tablist"
          aria-label="LayerPulse 输出任务"
          aria-orientation="vertical"
        >
          <button
            v-for="(output, index) in layerPulseOutputCatalog"
            :id="`layerpulse-output-tab-${output.key}`"
            :key="output.key"
            type="button"
            role="tab"
            class="prediction-task-tab"
            :class="{ active: layerPulseSelectedOutputKey === output.key }"
            :data-layerpulse-output-key="output.key"
            :aria-selected="layerPulseSelectedOutputKey === output.key"
            aria-controls="layerpulse-output-panel"
            :tabindex="layerPulseSelectedOutputKey === output.key ? 0 : -1"
            :title="output.name"
            @click="selectLayerPulseOutput(output.key)"
            @keydown="handleLayerPulseOutputTabKeydown($event, output.key)"
            @transitionend="handleLayerPulseOutputTabTransitionEnd($event, output.key)"
          >
            <span>{{ index + 1 }}</span>
            <strong>{{ output.shortName }}</strong>
            <small :data-ready="(layerPulseTaskStateForCurrentSnapshot.availableOutputKeys || []).includes(output.key)">
              {{ layerPulseOutputTabStatusLabel(output.key) }}
            </small>
          </button>
        </div>
      </nav>

      <div class="service-state">
        <span :class="['status-light', backendStatus]"></span>
        <div>
          <strong>{{ backendStatus === "online" ? "后端已连接" : backendStatus === "offline" ? "后端未连接" : "正在连接" }}</strong>
          <span>{{ backendVersion ? `平台 ${backendVersion}` : "127.0.0.1:725" }}</span>
        </div>
        <button v-if="backendStatus === 'offline'" type="button" class="service-retry" @click="retryBackendConnection">重连</button>
      </div>
    </aside>

    <main
      :class="[
        'main-content',
        {
          'overview-content': activeView === 'overview',
          'visualization-content': activeView === 'visualization',
          'prediction-workbench-content': activeView === 'prediction',
          'layerpulse-content': activeView === 'layerpulse',
          'assistant-content': activeView === 'assistant',
        },
      ]"
    >
      <div v-if="activeView !== 'overview'" :class="['topbar', { 'prediction-topbar': activeView === 'prediction' || activeView === 'layerpulse' }]">
        <div class="topbar-context"><span>地层慧眼</span><b>/</b><strong>{{ currentMeta.eyebrow }}</strong></div>
        <div class="topbar-account">
          <span :class="['topbar-status', backendStatus]"></span>
          <span>{{ backendStatus === "online" ? "本地工作区" : "服务未连接" }}</span>
          <b>⌄</b>
          <i aria-hidden="true">慧</i>
        </div>
      </div>
      <header
        v-if="activeView !== 'overview' && activeView !== 'models' && activeView !== 'assistant'"
        :class="[
          'page-header',
          {
            'immersive-page-header': activeView === 'visualization' || activeView === 'prediction' || activeView === 'layerpulse',
            'layerpulse-page-header': activeView === 'layerpulse',
          },
        ]"
      >
        <div>
          <p class="eyebrow">{{ currentMeta.eyebrow }}</p>
          <h1>{{ currentMeta.title }}</h1>
          <p>{{ currentMeta.description }}</p>
        </div>
        <div class="header-actions">
          <div v-if="activeView === 'layerpulse'" class="layerpulse-contract-badges" aria-label="LayerPulse 模型合同">
            <span>单 checkpoint</span>
            <span>唯一共享 Backbone</span>
            <span>一次 forward · {{ layerPulseModelContract.headCount }} 项</span>
            <span>F_final {{ layerPulseModelContract.fFinalChannels }}</span>
            <span>默认无时深表</span>
          </div>
          <button
            v-if="activeView === 'preparation' && runtimeContractReviewPending"
            type="button"
            class="secondary-button runtime-contract-reopen"
            @click="reopenRuntimeContractReview"
          >
            确认运行参数
          </button>
          <button
            v-if="activeView === 'preparation' && preparationScreen === 'input'"
            type="button"
            class="secondary-button preparation-cache-reset"
            :disabled="cacheClearing || backendStatus !== 'online'"
            title="停止全部排队和运行任务，清空可重建缓存及当前页面流程后重新开始；原始数据、封存快照、任务记录及已完成成果始终保留"
            @click="clearPlatformCaches"
          >
            <span :class="{ spinning: cacheClearing }" aria-hidden="true">↻</span>
            {{ cacheClearing ? "正在清理并重新开始…" : "清空缓存并重新开始" }}
          </button>
          <button
            v-if="activeView === 'preparation' && preparationScreen === 'input' && demo?.available"
            type="button"
            class="secondary-button"
            :disabled="preparationRunning"
            @click="applyDemo"
          >
            填入参考数据
          </button>
        </div>
      </header>
      <p
        v-if="activeView === 'preparation' && preparationScreen === 'input' && cacheMessage"
        :class="['preparation-cache-message', { error: cacheMessage.includes('失败') || cacheMessage.includes('未能') || cacheMessage.includes('暂不能') }]"
        role="status"
      >
        {{ cacheMessage }}
      </p>
      <p v-if="restorationWarning" class="error-message restoration-warning" role="status">
        {{ publicModelText(restorationWarning) }}
      </p>

      <template v-if="activeView === 'overview'">
        <section class="overview-hero" :style="{ backgroundImage: `url(${heroInterpretationCenterImage})` }">
          <header class="landing-header">
            <div class="landing-brand">
              <svg class="brand-mark" viewBox="0 0 160 160" aria-hidden="true">
                <path class="brand-layer brand-layer-primary" d="M18 52C42 34 62 37 80 51c18 14 37 18 62-2" />
                <path class="brand-layer brand-layer-secondary" d="M18 82c24-17 45-14 63 0 18 14 37 17 61-3" />
                <path class="brand-layer brand-layer-accent" d="M18 110c25-15 47-11 65 2 18 12 37 14 59-3" />
                <path class="brand-well" d="M78 24v81c0 14 8 24 27 31" />
                <circle cx="105" cy="136" r="8" />
              </svg>
              <div class="landing-brand-copy">
                <strong>地层慧眼</strong>
                <small>STRATA VISION</small>
              </div>
              <div
                class="landing-event-title"
                role="heading"
                aria-level="2"
                aria-label="2026年度中国青年科技创新“揭榜挂帅”擂台赛作品"
              >
                <span>2026年度中国青年</span>
                <strong class="landing-event-innovation">科技创新</strong>
                <i aria-hidden="true">“</i>
                <strong class="landing-event-command">揭榜挂帅</strong>
                <i aria-hidden="true">”</i>
                <span>擂台赛作品</span>
              </div>
            </div>
            <div class="landing-service"><span :class="['status-light', backendStatus]"></span>{{ backendStatus === 'online' ? '本地服务已连接' : backendStatus === 'offline' ? '服务未连接' : '正在连接服务' }}</div>
          </header>
          <div class="overview-copy">
            <p class="product-kicker"><span></span> 地震—测井多模态统一表征大模型</p>
            <h1>让油气甜点<br />有迹可循</h1>
            <p class="overview-summary">融合地震空间响应与测井精细表征，构建可解释的油气甜点识别能力。</p>
            <div class="product-hero-actions">
              <button type="button" class="hero-primary" @click="selectView('preparation')">
                <span>进入分析工作台</span><i aria-hidden="true">→</i>
              </button>
              <button type="button" class="hero-secondary" @click="selectView('models')">
                <span>了解方法</span><i aria-hidden="true">→</i>
              </button>
            </div>
          </div>
          <div class="overview-dock overview-primary-paths">
            <button type="button" @click="selectView('preparation')"><span>01</span><strong>数据与融合</strong><small>登记 · 质控 · 井震标定</small><i>→</i></button>
            <button type="button" @click="selectView('layerpulse')"><span>02</span><strong>LayerPulse 多模态融合基础模型</strong><small>共享底座 · 11 项统一解释</small><i>→</i></button>
            <button type="button" @click="selectView('prediction')"><span>03</span><strong>单任务推理模型（共享井震融合基座）</strong><small>配置 · 运行 · 联动查看</small><i>→</i></button>
          </div>
        </section>
      </template>

      <template v-else-if="activeView === 'preparation'">
        <section
          class="preparation-navigation three-stage"
          :class="{ 'compact-input-navigation': preparationScreen === 'input' }"
          aria-label="数据与融合步骤"
        >
          <button type="button" :class="{ active: preparationScreen === 'input' }" @click="showPreparationSourceStep">
            <span>1</span><div><strong>数据源</strong><small>登记文件或目录</small></div>
            <em v-if="preparationResult">已完成</em>
          </button>
          <b>→</b>
          <button type="button" :disabled="!preparation" :class="{ active: preparationScreen === 'pipeline' }" @click="showPreparationPipeline()">
            <span>2</span><div><strong>质控封存</strong><small>校验并处理阻断项</small></div>
            <em v-if="pendingConfirmationCount">{{ pendingConfirmationCount }} 项需集中补充</em>
          </button>
          <b>→</b>
          <button type="button" :disabled="!dataSnapshotTaskId" :class="{ active: preparationScreen === 'fusion' }" @click="showPreparationFusion">
            <span>3</span><div><strong>标定融合</strong><small>Registration · PreparedView</small></div>
            <em v-if="preparedViewReady">已完成</em>
            <em v-else-if="formalRegistrationReady">待构建</em>
          </button>
        </section>

        <section v-if="preparationScreen === 'input'" class="section-panel input-stage-panel">
          <div class="section-heading">
            <div><h2>输入数据源</h2><p>选择文件或目录；原始数据保持原位。</p></div>
            <span class="count-badge">已登记 {{ registeredCount }} 个路径</span>
          </div>

          <div class="path-groups">
            <section
              v-for="group in visiblePathGroups"
              :key="group.key"
              :class="[
                'path-group',
                {
                  'auxiliary-path-group': group.key === 'auxiliary',
                  'has-registered-paths': group.key === 'auxiliary' && group.paths.length > 0,
                },
              ]"
            >
              <div class="path-group-heading">
                <div>
                  <h3>{{ group.key === "wells" ? "井位与井轨迹" : group.title }} <span v-if="group.optional && group.key !== 'wells'">可选</span></h3>
                  <p>{{ group.key === "wells" ? "井口、海拔与 DEV 轨迹" : group.hint }}</p>
                </div>
                <button type="button" class="text-button" :disabled="preparationRunning" @click="addPath(group)">＋ 添加路径</button>
              </div>
              <div v-if="group.paths.length" class="path-list">
                <div v-for="(_, index) in group.paths" :key="index" class="path-row">
                  <span>{{ index + 1 }}</span>
                  <input v-model="group.paths[index]" type="text" :disabled="preparationRunning" placeholder="输入绝对路径，例如 D:\比赛数据\地震" @input="handleManualPathDraftChange" />
                  <button type="button" class="remove-button" :disabled="preparationRunning" @click="removePath(group, index)">移除</button>
                </div>
              </div>
              <p v-else class="empty-path">尚未添加该类数据。</p>
            </section>
          </div>

          <div class="run-bar">
            <div class="switches">
              <label><input v-model="recursive" type="checkbox" :disabled="preparationRunning" />递归读取子目录</label>
              <span class="run-assurance" title="读取必要道头并计算完整身份哈希。">完整快照校验</span>
              <label class="llm-task-switch" :title="capabilities?.llm.data_policy">
                <input v-model="useLlmFallback" type="checkbox" :disabled="preparationRunning || !capabilities?.llm.available" />
                智能补全
                <span>{{ capabilities?.llm.available ? "已就绪" : "未配置" }}</span>
              </label>
            </div>
            <div class="run-actions">
              <small>仅发送去路径化结构摘要；原文件不外发。</small>
              <button type="button" class="primary-button" :disabled="preparationRunning || backendStatus !== 'online'" @click="runDataPreparation">
                {{ preparationRunning ? "正在识别与校验…" : "识别并校验" }}
              </button>
            </div>
          </div>

          <div v-if="preparationRunning" class="task-progress preparation-progress" :data-phase="preparationActivityPhase">
            <div class="preparation-activity">
              <span class="reading-orbit" aria-hidden="true"><i></i></span>
              <div role="status" aria-live="polite" aria-atomic="true">
                <strong>{{ preparationActivityTitle }}</strong>
                <span>{{ preparationStatusMessage }}</span>
              </div>
              <b>{{ preparationProgress }}%</b>
            </div>
            <div v-if="preparationCurrentItem" class="preparation-current-file" :title="preparationCurrentItem">
              <span class="current-file-pulse" aria-hidden="true"></span>
              <div>
                <small>{{ preparationCurrentItemAction }}</small>
                <strong>{{ preparationCurrentItemName }}</strong>
              </div>
              <b>{{ preparationCurrentItemSizeBytes === null ? "正在获取大小" : formatBytes(preparationCurrentItemSizeBytes) }}</b>
            </div>
            <div class="preparation-timing">
              <span>已用时 {{ formatDuration(preparationElapsedSeconds) }}</span>
              <span v-if="preparationSubworkTotal && preparationSubworkUnit === 'bytes'">已校验 {{ formatBytes(preparationSubworkDone) }}/{{ formatBytes(preparationSubworkTotal) }}</span>
              <span v-else-if="preparationSubworkTotal && preparationSubworkUnit === 'traces'">已扫描 {{ preparationSubworkDone.toLocaleString() }}/{{ preparationSubworkTotal.toLocaleString() }} 道</span>
              <span v-else-if="preparationWorkTotal && preparationWorkUnit === 'assets'">已处理 {{ preparationWorkDone }}/{{ preparationWorkTotal }} 个可解析资产</span>
              <strong>{{ preparationEstimateLabel }}</strong>
            </div>
            <div
              class="progress-track"
              role="progressbar"
              aria-label="数据读取进度"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="preparationProgress"
            ><span :style="{ width: `${preparationProgress}%` }"></span></div>
            <small>耗时取决于数据规模和磁盘读取速度。</small>
          </div>
          <p v-if="errorMessage" class="error-message">{{ publicModelText(errorMessage) }}</p>

          <details id="source-contract-advanced" class="source-contract-advanced">
            <summary>
              <div><span>高级设置</span><strong>坐标、基准与解析合同</strong><small>证据不足时再补充。</small></div>
              <div class="source-contract-status">
                <b :class="{ warning: !horizontalSourceContractReady }">水平 {{ horizontalSourceContractReady ? '已核验' : '待核验' }}</b>
                <b :class="{ warning: !verticalSourceContractReady }">垂向 {{ verticalSourceContractReady ? '完整' : '待补充' }}</b>
                <b :class="{ warning: !timeSourceContractReady }">时间 {{ timeSourceContractReady ? 'TWT · SRD' : '待确认' }}</b>
                <b :class="{ warning: segyExplicitHeaderByteCount > 0 && segyExplicitHeaderByteCount < 5 }">{{ segySourceContractLabel }}</b>
              </div>
            </summary>
            <div class="datum-contract-panel">
              <div><strong>水平坐标、垂向空间与地震时间基准</strong><p>这些字段不会被删除：它们用于井震对齐和快照身份。文件头识别优先，未知值保持 unknown/null，不会按 0 猜测。</p></div>
              <label>目标/地震 CRS<input v-model="horizontalCrsId" type="text" :disabled="preparationRunning" placeholder="例如 EPSG:4547" /><span>平台统一到此投影坐标系</span></label>
              <label>井数据源 CRS<input v-model="wellSourceCrsId" type="text" :disabled="preparationRunning" placeholder="留空自动读取文件；否则例如 EPSG:21421" /><span>仅源坐标与地震不同时填写</span></label>
              <label>SEG-Y 源 CRS<input v-model="seismicSourceCrsId" type="text" :disabled="preparationRunning" placeholder="留空即与目标CRS相同" /></label>
              <label>水平单位<select v-model="horizontalUnit" :disabled="preparationRunning"><option value="unknown">unknown</option><option value="m">m</option><option value="ft">ft</option></select></label>
              <label>水平轴序<select v-model="horizontalAxisOrder" :disabled="preparationRunning"><option value="unknown">unknown</option><option value="XY">XY</option><option value="YX">YX</option></select></label>
              <label>坐标参考状态<input :value="coordinateReferenceVerified ? '已由转换回执与井震范围自动核验' : '运行后自动核验'" type="text" readonly /><span>不能用人工勾选绕过CRS冲突</span></label>
              <label>垂向 CRS<input v-model="verticalCrsId" type="text" :disabled="preparationRunning" placeholder="例如 LOCAL_MSL_CHENGDU" /></label>
              <label>SRD 高程<input v-model.number="seismicSrdElevation" type="number" step="0.01" :disabled="preparationRunning" placeholder="null" /><span>m MSL</span></label>
              <label>地震时间域<select v-model="seismicTimeDomain" :disabled="preparationRunning"><option value="unknown">unknown</option><option value="TWT">TWT</option><option value="OWT">OWT</option></select></label>
              <label>时间校正状态<select :value="seismicCorrectionState" :disabled="preparationRunning" @change="handleSeismicCorrectionStateChange"><option value="unknown">unknown（通用盘点不阻断）</option><option value="corrected_to_srd">corrected_to_srd（需一次物理声明）</option><option value="uncorrected">uncorrected</option></select></label>
              <label v-if="seismicCorrectionState === 'uncorrected' || seismicReplacementVelocity !== ''">替换速度<input v-model.number="seismicReplacementVelocity" type="number" step="1" min="1" :disabled="preparationRunning" placeholder="null" /><span>m/s</span></label>
            </div>
            <div class="source-contract-groups">
              <section>
                <header><div><span>SEG-Y GEOMETRY</span><strong>显式几何解析合同</strong></div><small>道头位置采用 1-based 字节号</small></header>
                <label>已知 Profile
                  <input v-model.trim="segyGeometryProfile" list="segy-profile-options" type="text" :disabled="preparationRunning" placeholder="留空自动识别；例如 standard_3d" />
                  <datalist id="segy-profile-options"><option value="standard_3d"></option></datalist>
                </label>
                <div class="segy-byte-grid">
                  <label>Inline<input v-model.number="segyInlineByte" type="number" min="1" max="237" :disabled="preparationRunning" placeholder="189" /></label>
                  <label>Crossline<input v-model.number="segyCrosslineByte" type="number" min="1" max="237" :disabled="preparationRunning" placeholder="193" /></label>
                  <label>X<input v-model.number="segyXByte" type="number" min="1" max="237" :disabled="preparationRunning" placeholder="181" /></label>
                  <label>Y<input v-model.number="segyYByte" type="number" min="1" max="237" :disabled="preparationRunning" placeholder="185" /></label>
                  <label>Scalar<input v-model.number="segyCoordinateScalarByte" type="number" min="1" max="239" :disabled="preparationRunning" placeholder="71" /></label>
                </div>
                <p>普通数据准备可选择已审计 Profile，或完整填写五个道头字节；只填部分字段不会被当成可信几何。高级轨迹感知校正可消费显式封存合同，或与当前资产 SHA、解析选项及几何指纹绑定的高置信自动检测收据；低置信、冲突或缺字段仍会失败关闭。</p>
              </section>
              <section>
                <header><div><span>WELL SOURCE UNITS</span><strong>原始井数据单位</strong></div><small>不同于平台规范单位</small></header>
                <label>井位 / 轨迹表源 X/Y 单位
                  <select v-model="wellCoordinateSourceUnit" :disabled="preparationRunning"><option value="unknown">unknown（需要证据）</option><option value="m">m</option><option value="ft">ft（读取时换算为m）</option></select>
                </label>
                <label>KB / GL / DF / RT 源高程单位
                  <select v-model="wellVerticalDatumSourceUnit" :disabled="preparationRunning"><option value="unknown">unknown（文件无证据时保持待确认）</option><option value="m">m</option><option value="ft">ft（读取时换算为m）</option></select>
                </label>
                <label>LAS TWT 曲线源单位
                  <select v-model="lasTwtSourceUnit" :disabled="preparationRunning"><option value="unknown">unknown（含TWT时阻断）</option><option value="ms">ms</option><option value="s">s</option><option value="us">μs</option></select>
                </label>
                <p>“水平单位”描述平台 CRS 规范；X/Y、井口高程和TWT是三个独立源合同，系统不会互相继承。只在对应单位明确时执行 ft→m 或 s/μs→ms 换算。</p>
              </section>
            </div>
          </details>
        </section>

        <template v-else-if="preparationScreen === 'pipeline' && preparation">
          <section v-if="workflowResult" class="section-panel source-statistics-panel" aria-label="地震与测井数据统计">
            <div class="section-heading source-statistics-heading">
              <div>
                <span class="section-kicker">SOURCE DATA PROFILE</span>
                <h2>地震与测井概览</h2>
                <p>仅展示本次实际读取与解析结果，不包含估算值。</p>
              </div>
              <a v-if="dataSnapshotTaskId" class="source-manifest-link" :href="taskArtifactUrl(dataSnapshotTaskId, 'snapshot_manifest')" target="_blank" rel="noopener" download>数据清单 ↗</a>
            </div>

            <div class="source-statistics-grid">
              <article class="source-statistic-card seismic-statistic-card">
                <header class="statistic-card-header">
                  <span class="statistic-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24"><path d="M3 8c3-4 5 4 8 0s5 4 10 0M3 12c3-4 5 4 8 0s5 4 10 0M3 16c3-4 5 4 8 0s5 4 10 0" /></svg>
                  </span>
                  <div>
                    <strong>地震数据</strong>
                    <small :title="primarySeismicStatistics?.name || ''">{{ primarySeismicStatistics?.name || "尚未解析主数据体" }}</small>
                  </div>
                  <b>{{ sourceDataStatistics.seismic.readableFileCount }}/{{ sourceDataStatistics.seismic.fileCount }} 可读</b>
                </header>

                <div class="statistic-hero">
                  <div><strong>{{ sourceDataStatistics.seismic.totalTraceCount ? sourceDataStatistics.seismic.totalTraceCount.toLocaleString() : "—" }}</strong><span>地震道</span></div>
                  <small>{{ sourceDataStatistics.seismic.fileCount }} 个文件 · {{ formatBytes(sourceDataStatistics.seismic.totalSizeBytes) }}</small>
                </div>

                <dl class="statistic-metrics">
                  <div><dt>Inline × Crossline</dt><dd>{{ primarySeismicStatistics?.inlineCount && primarySeismicStatistics?.crosslineCount ? `${primarySeismicStatistics.inlineCount.toLocaleString()} × ${primarySeismicStatistics.crosslineCount.toLocaleString()}` : "—" }}</dd></div>
                  <div><dt>每道样点</dt><dd>{{ primarySeismicStatistics?.samplesPerTrace ? primarySeismicStatistics.samplesPerTrace.toLocaleString() : "—" }}</dd></div>
                  <div><dt>采样间隔</dt><dd>{{ primarySeismicStatistics?.sampleIntervalMs === null || primarySeismicStatistics?.sampleIntervalMs === undefined ? "—" : `${primarySeismicStatistics.sampleIntervalMs} ms` }}</dd></div>
                </dl>

                <div class="statistic-footer">
                  <div class="statistic-progress-copy"><span>主体网格覆盖</span><strong>{{ seismicGridCoveragePercent === null ? "未解析" : `${seismicGridCoveragePercent.toFixed(1)}%` }}</strong></div>
                  <div
                    class="statistic-progress"
                    role="progressbar"
                    aria-label="主体网格覆盖率"
                    aria-valuemin="0"
                    aria-valuemax="100"
                    :aria-valuenow="seismicGridCoveragePercent === null ? undefined : Number(seismicGridCoveragePercent.toFixed(1))"
                  ><span :style="{ width: `${seismicGridCoveragePercent || 0}%` }"></span></div>
                </div>
              </article>

              <article class="source-statistic-card well-statistic-card">
                <header class="statistic-card-header">
                  <span class="statistic-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24"><path d="M5 3v18M5 5c5 0 2 5 7 5s2 5 7 5M5 19c4 0 3-4 7-4s3 4 7 4" /></svg>
                  </span>
                  <div><strong>井与测井</strong><small>{{ sourceDataStatistics.wells.logFileCount }} 个 LAS · {{ formatBytes(sourceDataStatistics.wells.totalSizeBytes) }}</small></div>
                  <b>{{ sourceDataStatistics.wells.trajectoryWellCount }} 口有轨迹</b>
                </header>

                <div class="statistic-hero">
                  <div><strong>{{ sourceDataStatistics.wells.wellCount || "—" }}</strong><span>口井</span></div>
                  <small>井实体、曲线与轨迹已统一归集</small>
                </div>

                <dl class="statistic-metrics">
                  <div><dt>测井采样点</dt><dd>{{ sourceDataStatistics.wells.totalLogSamples ? sourceDataStatistics.wells.totalLogSamples.toLocaleString() : "—" }}</dd></div>
                  <div><dt>曲线标识</dt><dd>{{ sourceDataStatistics.wells.curveIdentifiers.size || "—" }}</dd></div>
                  <div><dt>有轨迹井</dt><dd>{{ sourceDataStatistics.wells.trajectoryWellCount || "—" }}</dd></div>
                </dl>

                <div class="statistic-footer well-geometry-footer">
                  <div class="statistic-progress-copy"><span>井型分布</span><strong>{{ wellGeometryTotal ? `${wellGeometryTotal} 口已分类` : "未解析" }}</strong></div>
                  <div v-if="wellGeometryTotal" class="well-geometry-strip" aria-label="井型分布">
                    <span v-for="item in wellGeometrySegments" :key="item.id" :class="item.id" :style="{ width: `${item.value / wellGeometryTotal * 100}%` }" :title="`${item.label} ${item.value} 口`"></span>
                  </div>
                  <div v-if="wellGeometryTotal" class="well-geometry-legend">
                    <span v-for="item in wellGeometrySegments" :key="item.id"><i :class="item.id"></i>{{ item.label }} {{ item.value }}</span>
                  </div>
                </div>
              </article>
            </div>
            <p class="source-statistics-note">统计来自已读取的 SEG-Y 文件头、LAS 曲线与井实体；完整来源身份仍在后台封存并随下游任务传递。</p>
          </section>

          <section class="pipeline-next-step">
            <div>
              <span>当前状态</span>
              <strong v-if="!hasExplicitPreparationScope && sealedWellSeismicWorkflowReady && !registrationTaskId && !horizontalRegistrationTaskId">{{ acousticFineCalibrationCandidateReady ? `自动精细井震标定已就绪（${acousticFineCalibrationCandidateCount} 口井具备合格声波与轨迹证据）` : '标定证据待核验，最低证据不足将仅做几何覆盖' }}</strong>
              <strong v-else-if="!hasExplicitPreparationScope && horizontalRegistrationTaskId && !registrationTaskId">空间 QC 已完成；未形成融合资格，预测仍保持锁定</strong>
              <strong v-else-if="!hasExplicitPreparationScope && formalRegistrationFusionBlocked">精细标定已完成；未形成 fusion-ready 消费产品</strong>
              <strong v-else-if="!hasExplicitPreparationScope && preparedViewReady">PreparedView 已就绪，可以选择下游模型</strong>
              <strong v-else-if="!hasExplicitPreparationScope && formalRegistrationReady">精细标定已完成，下一步只为 fusion-ready 井构建 PreparedView</strong>
              <strong v-else-if="!hasExplicitPreparationScope">通用盘点已完成，请选择任务与模型</strong>
              <strong v-else-if="mainBlockingIssue">先处理 {{ effectiveBlockingCount }} 个当前模型阻断问题</strong>
              <strong v-else-if="!currentScopeModelId">请选择具体模型，避免“全部模型”引入最严格联合门禁</strong>
              <strong v-else-if="!currentPreparationGateReady">当前模型输入合同尚未满足</strong>
              <strong v-else>数据已满足后续工作流的基础放行条件</strong>
              <details class="next-step-details">
                <summary>查看运行说明</summary>
              <p v-if="!hasExplicitPreparationScope && sealedWellSeismicWorkflowReady && !registrationTaskId && !horizontalRegistrationTaskId">{{ acousticFineCalibrationCandidateReady ? '无需补造时深表：平台将从 AC/DT、真实轨迹和 KB/GL 建立时深先验，再用地震与冻结概率模型精细约束；只有通过物理门的井会进入融合视图。' : '平台将先核验可用标定证据；只有精细标定最低证据不足时才进入仅几何空间 QC，且不会伪造 TWT。' }}</p>
              <p v-else-if="!hasExplicitPreparationScope && horizontalRegistrationTaskId && !registrationTaskId">该结果不含 TWT、垂向标定或 PreparedView；必须完善数据合同并完成融合后，平台才会开放预测。</p>
              <p v-else-if="!hasExplicitPreparationScope && formalRegistrationFusionBlocked">标定候选已封存，但物理门未放行任何可消费井；平台不会生成 PreparedView，也不会把候选标成融合完成。</p>
              <p v-else-if="!hasExplicitPreparationScope && preparedViewReady">PreparedView 只是可供下游消费的受控输入；模型运行完成不代表已融合，结果卡中的 consumption receipt 才是最终依据。</p>
              <p v-else-if="!hasExplicitPreparationScope && formalRegistrationReady">标定已形成正式 Registration V3；系统将过滤非 fusion-ready 井，不会把空间 QC 结果伪装成融合视图。</p>
              <p v-else-if="!hasExplicitPreparationScope">当前显示的是模型无关数据盘点；无论后续选择哪个模型，平台都会先完成统一的井震标定与融合门禁。</p>
              <p v-else-if="mainBlockingIssue">这里只统计当前模型真正消费的输入；井位、地震几何等有效阻断仍须处理。</p>
              <p v-else-if="!currentScopeModelId">返回输入页选择推荐模型并重新准备；只有主动选择正式配准模型时，垂向、SRD 与时间合同才是硬门禁。</p>
              <p v-else-if="!currentPreparationGateReady">数据准备报告未通过当前任务门禁，或输入适配器尚未确认可用几何；请检查上方有效问题和模型输入。</p>
              <p v-else-if="currentScopeRequiresPreparedView">所选模型需要已准备数据视图；系统会先完成所需标定，再冻结可追溯的融合输入。</p>
              <p v-else-if="currentScopePrefersPreparedView">所选模型优先消费已准备数据视图；平台会先完成标定并冻结可追溯的融合输入。</p>
              <p v-else-if="currentScopeRequiresRegistration">所选模型需要井震注册；平台还会完成 PreparedView，随后才开放预测。</p>
              <p v-else>当前模型合同可以不消费融合特征，但平台工作流仍统一要求先完成融合，再开放预测。</p>
              </details>
            </div>
            <div class="next-step-actions">
              <button v-if="runtimeContractReviewPending" type="button" class="secondary-button" @click="reopenRuntimeContractReview">确认运行参数</button>
              <button v-if="hasExplicitPreparationScope && (mainBlockingIssue || pendingConfirmationCount)" type="button" class="secondary-button" @click="focusIssueReview">审核当前模型待处理建议</button>
              <button type="button" class="secondary-button" :disabled="!seismicInventory.length && !wellLogPreviews.length" @click="selectView('visualization')">
                {{ seismicInventory.length || wellLogPreviews.length ? "查看当前任务数据" : "当前任务无可视化资产" }}
              </button>
              <button
                type="button"
                class="primary-button"
                :disabled="registrationRunning || sampleRunning || (!preparedViewReady && (!dataSnapshotTaskId || effectiveBlockingCount > 0 || !currentPreparationGateReady))"
                @click="handlePreparationNextAction"
              >{{ nextAction.label }}</button>
            </div>
            <div v-if="registrationRunning || sampleRunning" class="task-progress preparation-progress fusion-progress" :data-phase="fusionActivityPhase" role="status" aria-live="polite">
              <div class="preparation-activity">
                <span class="reading-orbit" aria-hidden="true"><i></i></span>
                <div><strong>{{ fusionActivityTitle }}</strong><span>{{ publicModelText(statusMessage) }}</span></div>
                <b>{{ progress }}%</b>
              </div>
              <div class="preparation-timing"><span>已运行 {{ formatDuration(fusionElapsedSeconds) }}</span><strong>{{ fusionEtaSeconds === null ? '预计时间计算中' : `预计剩余 ${formatDuration(fusionEtaSeconds)}` }}</strong></div>
              <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
              <small>{{ fusionEstimateLabel }}</small>
            </div>
            <p v-if="errorMessage" class="error-message pipeline-action-error" role="alert">{{ publicModelText(errorMessage) }}</p>
            <div v-if="registrationPreparationRequired" class="registration-preparation-action">
              <span>不会自动离开当前结果页，也不会改写旧快照。</span>
              <button type="button" class="secondary-button" @click="openRegistrationPreparation">返回数据准备并重新封存</button>
            </div>
            <details v-if="registrationResult?.registration || sampleResult?.matching" class="pipeline-derived-results">
              <summary>查看标定与融合历史结果</summary>
            <div v-if="registrationResult?.registration" class="data-view-result">
              <div class="data-view-result-head">
                <span>井震时间标定成果</span>
                <strong>{{ registrationResult.registration.registered_well_count }} / {{ registrationResult.registration.well_count }} {{ registrationTimeAxisPresentation.resultLabel }}</strong>
              </div>
              <div class="data-view-result-metrics">
                <span><b>{{ p13RegistrationPresentation?.executionLabel || "未记录" }}</b>冻结概率模型</span>
                <span><b>{{ registrationResult.registration.fusion_ready_well_count }}</b>已通过高信度门禁</span>
                <span><b>DEV</b>XYZ/TVD几何权威源</span>
              </div>
              <div>
                <p v-if="registrationTimeAxisPresentation.limitation">{{ registrationTimeAxisPresentation.limitation }}</p>
                <p>{{ p13RegistrationPresentation?.detail }}</p>
                <div class="artifact-chip-list">
                  <template v-for="[name, path] in registrationOutputEntries" :key="name">
                    <span v-if="isDirectoryOutput(name, path)" class="directory-output-note">{{ publicModelText(name) }} · 目录需先打包</span>
                    <a v-else-if="registrationTaskId" :href="taskArtifactUrl(registrationTaskId, name)" target="_blank" rel="noopener" download>{{ artifactDisplayName(name, path) }}</a>
                  </template>
                  <span v-if="registrationResult.registration.output_directory" class="directory-output-note">成果目录已登记 · 不直接下载</span>
                </div>
              </div>
            </div>
            <div v-if="sampleResult?.matching" class="data-view-result">
              <div class="data-view-result-head">
                <span>井震多模态数据视图</span>
                <strong>{{ sampleResult.matching.sample_count.toLocaleString() }} 条候选样本</strong>
              </div>
              <div class="data-view-result-metrics">
                <span><b>{{ (sampleResult.matching.valid_window_count ?? 0).toLocaleString() }}</b>有效时间窗</span>
                <span><b>{{ (sampleResult.matching.training_eligible_count ?? 0).toLocaleString() }}</b>监督训练标签（不影响预训练模型推理）</span>
                <span><b>{{ sampleResult.matching.coordinate_reference_verified ? "已核验" : "待核验" }}</b>井震坐标参考</span>
                <span><b>{{ sampleResult.matching.vertical_datum_ready ? "已统一" : "已阻断" }}</b>MSL/SRD/KB/GL基准</span>
              </div>
              <div class="artifact-chip-list">
                <template v-for="[name, path] in preparedViewOutputEntries" :key="name">
                  <span v-if="isDirectoryOutput(name, path)" class="directory-output-note">{{ publicModelText(name) }} · 目录需先打包</span>
                  <a v-else-if="sampleBuildingTaskId" :href="taskArtifactUrl(sampleBuildingTaskId, name)" target="_blank" rel="noopener" download>{{ artifactDisplayName(name, path) }}</a>
                </template>
                <span v-if="sampleResult.matching.output_directory" class="directory-output-note">PreparedView 目录已登记 · 不直接下载</span>
              </div>
            </div>
            </details>
          </section>

          <section class="section-panel pipeline-panel">
            <div class="section-heading pipeline-heading">
              <div>
                <span class="section-kicker">{{ hasExplicitPreparationScope ? "当前任务 · 预处理审核" : "通用数据 · 非阻断盘点" }}</span>
                <h2>预处理与对齐流水线</h2>
                <p v-if="hasExplicitPreparationScope">点击步骤查看证据和指标；黄色需要复核，红色只表示所选模型的真实前置门禁。</p>
                <p v-else>当前尚未选择任务与模型；以下结果只做数据盘点，不会作为普通任务的红色阻断。</p>
              </div>
              <div class="pipeline-heading-actions">
                <div class="issue-summary">
                  <template v-if="hasExplicitPreparationScope">
                    <span v-if="effectiveBlockingCount" class="danger">{{ effectiveBlockingCount }} 个当前模型阻断</span>
                    <span v-else class="passed">当前模型 0 阻断</span>
                    <span v-if="currentAttentionIssues.length > effectiveBlockingCount" class="warn">{{ currentAttentionIssues.length - effectiveBlockingCount }} 项待处理</span>
                  </template>
                  <template v-else>
                    <span class="passed">通用盘点不设执行阻断</span>
                    <span>{{ preparationInventoryIssueCount }} 项审计记录</span>
                  </template>
                  <span>{{ preparation.summary.autofilled || 0 }} 已自动补全</span>
                  <span v-if="hasExplicitPreparationScope && effectiveSurveyInputRequiredCount">{{ effectiveSurveyInputRequiredCount }} 项需集中补充</span>
                  <span v-if="hasExplicitPreparationScope && unusedPreparationStageCount">{{ unusedPreparationStageCount }} 个步骤未参与当前任务</span>
                </div>
                <button
                  type="button"
                  class="primary-button compact"
                  :disabled="batchApplyingRecommendations || autofillEligibleCount === 0"
                  @click="applyAllSafeRecommendations"
                >
                  {{ batchApplyingRecommendations ? "正在隔离修复并复检…" : "应用已验证修复并规则复检" }}
                </button>
                <button type="button" class="secondary-button compact" @click="showPreparationInput">返回修改输入</button>
              </div>
            </div>

            <div v-if="preparationRunTiming" class="preparation-run-meta" aria-label="本次数据盘点运行时间">
              <span class="run-meta-mark" aria-hidden="true">✓</span>
              <div>
                <strong>本次数据盘点</strong>
                <p>
                  <time :datetime="preparationRunTiming.startedAt">{{ formatTaskTimestamp(preparationRunTiming.startedAt) }}</time>
                  <i aria-hidden="true">→</i>
                  <time :datetime="preparationRunTiming.finishedAt">{{ formatTaskTimestamp(preparationRunTiming.finishedAt) }}</time>
                </p>
              </div>
              <b>{{ preparationRunTiming.durationSeconds === null ? "耗时未记录" : `用时 ${formatDuration(preparationRunTiming.durationSeconds)}` }}</b>
              <small>同一份盘点报告 · 逐节点时间未记录</small>
            </div>

            <ol class="pipeline-stepper" aria-label="预处理与对齐执行流程">
              <li v-for="(stage, index) in preparation.stages" :key="stage.id" :class="[stageClass(stage), { selected: issueFilter === stage.id }]">
              <button
                type="button"
                @click="issueFilter = stage.id"
              >
                <span>{{ stageDisplayReady(stage) ? "✓" : index + 1 }}</span>
                <strong>{{ stageShortName(stage.id) }}</strong>
                <em>{{ !stageRequiredForCurrentRun(stage.id) ? stageStatusLabel(stage) : currentAttentionIssues.filter((issue) => issue.stage === stage.id).length ? `${currentAttentionIssues.filter((issue) => issue.stage === stage.id).length} 项需处理` : stageStatusLabel(stage) }}</em>
              </button>
              </li>
            </ol>

            <div v-if="selectedStage" class="pipeline-inspector">
              <div class="pipeline-inspector-copy"><span>当前步骤 · {{ stageStatusLabel(selectedStage) }}</span><strong>{{ selectedStage.name }}</strong><p>{{ stageDescription(selectedStage) }}</p><button type="button" class="text-button" @click="focusIssueReview">查看本步骤问题 ↓</button></div>
              <dl><template v-for="(value, key) in selectedStage.metrics" :key="key"><div><dt>{{ key }}</dt><dd>{{ value }}</dd></div></template></dl>
            </div>
            <button v-else type="button" class="all-issues-button" @click="showAllIssues">查看需处理的问题与建议</button>
          </section>

          <details id="issue-review" class="section-panel issue-review-panel evidence-review-drawer">
            <summary class="evidence-review-summary">
              <div>
                <span>AUTOMATED REVIEW</span>
                <strong>自动研判与证据缺口</strong>
                <small>通用盘点、地震垂向与时间合同及修复建议</small>
              </div>
              <div class="evidence-review-summary-status">
                <span>{{ currentAttentionIssues.length }} 项问题</span>
                <span>{{ seismicVerticalContractCandidates.length }} 项合同候选</span>
                <b :class="{ attention: evidenceReviewRequiresAction }">{{ evidenceReviewStatusLabel }}</b>
                <i aria-hidden="true">⌄</i>
              </div>
            </summary>
            <div class="evidence-review-body">
              <div :class="['pipeline-scope-note', { scoped: hasExplicitPreparationScope }]">
                <span>{{ currentScopeModelId ? "当前模型合同" : preparation.task_readiness?.task_id ? "任务范围" : "通用范围" }}</span>
                <div>
                  <strong v-if="currentScopeModelId">当前按「{{ currentScopeModelName }}」重新计算有效门禁</strong>
                  <strong v-else-if="preparation.task_readiness?.task_id">本次按「{{ preparationTargetTaskName }}」输入合同审核</strong>
                  <strong v-else>通用数据盘点已完成，尚未选择执行模型</strong>
                  <p v-if="currentScopeModelId">
                    当前门禁直接采用后端封存报告中的 required_for_task、blocking 与 attention_required；浏览器不再按 DataFlowSpec 二次改写。
                  </p>
                  <p v-else-if="preparation.task_readiness?.task_id">
                    该任务只要求 {{ preparationRequiredModalitiesLabel }}；标为“当前任务未使用”的步骤不会参与本次运行，但已登记数据仍完整保留。
                  </p>
                  <p v-else>读取失败、井位、垂向与时间问题均保留为盘点证据，不计入当前红色阻断；选择任务与推荐模型并重新准备后，平台才按实际输入合同放行。</p>
                  <p v-if="acousticFineCalibrationCandidateReady">
                    未提供独立时深表，但已识别 {{ acousticFineCalibrationCandidateCount }} 口井具备 AC/DT 声波输入；自动精细标定已就绪，将继续执行声波积分、地震约束与概率一致性门，而不是默认降为仅水平配准。
                  </p>
                </div>
              </div>

              <section v-if="seismicVerticalContractIssues.length" class="survey-contract-review" :data-deferred="hasDeferredRegistration">
                <header>
                  <div>
                    <span>SEISMIC VERTICAL CONTRACT</span>
                    <strong>地震垂向与时间合同 · 一次性合并确认</strong>
                    <p>Kimi 与本地规则只提取原文证据和候选值；采用后写入下一次数据准备表单，不会修改当前封存快照。</p>
                  </div>
                  <b v-if="hasDeferredRegistration && !hasExplicitPreparationScope">正式配准待补充 · 不影响普通任务</b>
                  <b v-else-if="hasDeferredRegistration">井震标定待补充 · 不影响当前{{ currentScopeModelShortName }}</b>
                  <b v-else>正式标定前需确认</b>
                </header>
                <div v-if="seismicVerticalContractCandidates.length" class="survey-contract-candidate-grid">
                  <article
                    v-for="candidate in seismicVerticalContractCandidates"
                    :key="candidate.field"
                    :class="{ unresolved: !contractCandidateCanPopulate(candidate) }"
                  >
                    <div>
                      <span>{{ contractCandidateFieldLabel(candidate.field) }}</span>
                      <em>{{ contractCandidateStatusLabel(candidate) }}</em>
                    </div>
                    <strong>{{ contractCandidateValueLabel(candidate) }}</strong>
                    <p>{{ contractCandidateEvidenceLabel(candidate) }}</p>
                    <small v-if="candidate.confidence !== null && candidate.confidence !== undefined">证据置信度 {{ Math.round(candidate.confidence * 100) }}%</small>
                    <small v-else>未以数值置信度替代原文证据</small>
                  </article>
                </div>
                <div v-else class="survey-contract-empty">
                  <strong>当前快照尚未生成结构化候选</strong>
                  <p>重新准备时可启用 Kimi 受控判读；没有明确处理证据时，系统仍会把 corrected_to_srd 保持为 unknown。</p>
                </div>
                <footer>
                  <div>
                    <strong v-if="pendingSeismicVerticalCandidateCount">{{ pendingSeismicVerticalCandidateCount }} 项Kimi/规则候选只需一次合并确认</strong>
                    <strong v-else-if="autoPopulatedSurveyContractFields.length">已自动判断并写入 {{ autoPopulatedSurveyContractFields.length }} 项安全候选，不会自动重跑</strong>
                    <strong v-else-if="adoptableSeismicVerticalCandidates.length">安全候选将在结果到达时自动写入下一次准备草稿</strong>
                    <strong v-else>没有可安全采用的候选值</strong>
                    <p v-if="unresolvedSurveyContractFields.length">仍缺证据：{{ unresolvedSurveyContractFields.map(contractCandidateFieldLabel).join('、') }}</p>
                    <p v-else>单击不会解除旧报告阻断；重新准备并封存新合同后才会重新计算正式标定门禁。</p>
                  </div>
                  <button
                    v-if="pendingSeismicVerticalCandidateCount"
                    type="button"
                    class="primary-button compact"
                    @click="applySeismicVerticalContractCandidates"
                  >一次合并确认并写入草稿</button>
                  <button v-else-if="unresolvedSurveyContractFields.length || !adoptableSeismicVerticalCandidates.length" type="button" class="secondary-button compact" @click="requestSeismicVerticalCandidateRefresh">重新提取证据候选</button>
                </footer>
              </section>
              <div class="issue-review-content">
            <div class="section-heading">
              <div><h2>问题与修复建议</h2><p>可从文件证据推断的字段由 Kimi 生成白名单结构化补丁；原文件保持只读，隔离重读与物理规则全部通过后才采纳。</p></div>
              <div class="issue-review-heading-actions">
                <button type="button" class="secondary-button compact" @click="openAdvancedDataContract">修复数据合同</button>
              </div>
            </div>
            <div v-if="filteredIssues.length" class="issue-list">
              <article v-for="issue in filteredIssues" :key="issue.id" :class="['issue-row', issue.display_severity || issue.severity]">
                <div class="issue-description">
                  <div class="issue-title-line"><span class="severity">{{ issue.display_severity || issue.severity }}</span><strong>{{ issue.title }}</strong></div>
                  <p>{{ issue.message }}</p>
                  <small v-if="(issue.affected_count || 0) > 1">影响 {{ issue.affected_count }} 个对象 · 首个来源：{{ issueSource(issue) }}</small>
                  <small v-else>来源：{{ issueSource(issue) }}</small>
                </div>
                <div class="recommendation-card">
                  <div class="recommendation-heading">
                    <span :class="['recommendation-source', { llm: issue.recommendation_source === 'LLM' }]">
                      {{ issue.recommendation_source === "LLM" ? "Kimi 建议" : "规则建议" }}
                    </span>
                    <em v-if="issue.recommendation_confidence !== null">置信度 {{ Math.round(issue.recommendation_confidence * 100) }}%</em>
                    <em v-else>平台知识库</em>
                  </div>
                  <strong>{{ issue.recommended_action || "保留来源记录" }}</strong>
                  <p>{{ issue.recommendation_reason }}</p>
                  <div :class="['confirmation-result', { accepted: ['已确认采用', '已启用转换插件', 'LLM已补全', 'LLM已补全并复检', '系统已自动处理'].includes(issue.confirmation_status) }]">
                    {{ issueStatusLabel(issue) }}<span v-if="issue.confirmed_action"> · {{ issue.confirmed_action }}</span>
                  </div>
                  <details v-if="issue.autofill_patch && Object.keys(issue.autofill_patch).length" class="autofill-details">
                    <summary>查看结构化补全与校验</summary>
                    <pre>{{ JSON.stringify(issue.autofill_patch, null, 2) }}</pre>
                    <small v-for="check in issue.autofill_validation || []" :key="check">✓ {{ check }}</small>
                  </details>
                  <div v-if="transformationDrafts[issue.id]" class="adapter-draft">
                    <div class="adapter-draft-head">
                      <div><span>{{ transformationDrafts[issue.id].provider === '平台规则编译器' ? '规则编译' : 'Kimi 生成' }}</span><strong>{{ transformationDrafts[issue.id].title }}</strong></div>
                      <em :class="{ passed: transformationDrafts[issue.id].valid }">{{ transformationDrafts[issue.id].status }}</em>
                    </div>
                    <p>{{ transformationDrafts[issue.id].explanation }}</p>
                    <div class="adapter-tests">
                      <span v-for="test in transformationDrafts[issue.id].tests" :key="test.name" :class="{ passed: test.passed }">{{ test.passed ? '✓' : '!' }} {{ test.name }} · {{ test.details }}</span>
                    </div>
                    <details><summary>查看声明预览（不会执行代码）</summary><pre>{{ transformationDrafts[issue.id].generated_code }}</pre></details>
                    <div class="adapter-draft-actions">
                      <span>{{ transformationDrafts[issue.id].model }} · 原始文件不会被改写</span>
                      <button type="button" class="primary-button compact" :disabled="!transformationDrafts[issue.id].valid || activatingDraftId === transformationDrafts[issue.id].id || transformationDrafts[issue.id].status === '已启用'" @click="enableTransformDraft(issue, transformationDrafts[issue.id])">
                        {{ transformationDrafts[issue.id].status === '已启用' ? '已启用' : activatingDraftId === transformationDrafts[issue.id].id ? '正在启用…' : '确认启用规则' }}
                      </button>
                    </div>
                  </div>
                </div>
              </article>
            </div>
            <div v-else class="empty-inline">当前阶段没有需要处理的问题，可以继续下一步。</div>
            <details v-if="auditIssues.length" class="full-audit-drawer">
              <summary>完整审计 · {{ auditIssues.length }} 条（不影响当前执行）</summary>
              <div class="audit-record-list">
                <article v-for="issue in auditIssues" :key="issue.id">
                  <div><span>审计记录</span><strong>{{ issue.title }}</strong></div>
                  <p>{{ issue.message }}</p>
                  <small>{{ issueStatusLabel(issue) }} · 来源：{{ issueSource(issue) }}</small>
                </article>
              </div>
            </details>
              </div>
            </div>
          </details>

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
              <span>{{ visualizationSourceTaskId ? `${publicModelText(predictionResult?.task_name || activePredictionTaskSpec?.name, '模型')}预测结果` : '当前任务可视化' }}</span>
              <strong>选择井震联合或测井独立工作台</strong>
              <small>{{ seismicInventory.length }} 个地震资产 / {{ previewVolumes.length }} 个三维体 / {{ seismicLinePreviews.length }} 条二维测线 · {{ wellLogPreviews.length }} 口井 / LAS 预览</small>
              <small v-if="wellGeometrySummary" class="well-geometry-summary">
                井型自动判别：直井 {{ wellGeometrySummary.counts.vertical || 0 }} · 斜井 {{ wellGeometrySummary.counts.deviated || 0 }} · 水平井 {{ wellGeometrySummary.counts.horizontal || 0 }}
              </small>
              <small v-if="wellTimeAlignmentSummary" class="well-geometry-summary">
                空间叠加：{{ wellTimeAlignmentSummary.embedded ?? wellTimeAlignmentSummary.aligned }} 口井轨迹可见 · {{ wellTimeAlignmentSummary.aligned }} 口具备{{ previewTimeAxisLabel }}{{ wellTimeAlignmentSummary.depthNormalizedPreviews ? ` · ${wellTimeAlignmentSummary.depthNormalizedPreviews} 口为相对TVD参考` : '' }}
              </small>
            </div>
            <nav class="visualization-mode-cards" aria-label="可视化类型">
              <button type="button" data-visualization-mode="joint" :aria-pressed="visualizationMode === 'seismic'" :disabled="!seismicPreviewCount" :class="['visualization-mode-card', { active: visualizationMode === 'seismic' }]" @click="selectVisualizationMode('seismic')">
                <span class="mode-card-icon seismic" aria-hidden="true">
                  <svg viewBox="0 0 28 28"><path d="m14 3 10 5.4v11.2L14 25 4 19.6V8.4L14 3Z"/><path d="m4.7 8.8 9.3 5.1 9.3-5.1M14 14v10.2"/><path d="M7.6 13.2c2.4-1.8 4.7 1.7 7.1 0s4.3-1.4 6.5.1"/></svg>
                </span>
                <span class="mode-card-copy"><small>WELL-SEISMIC JOINT VIEW</small><strong>井震联合展示</strong><em>地震切片与标定后的井位、井轨迹同场叠加</em></span>
                <b>{{ visualizationMode === 'seismic' ? '当前视图' : seismicPreviewCount ? '进入' : '无预览' }} <i>→</i></b>
              </button>
              <button type="button" data-visualization-mode="logs" :aria-pressed="visualizationMode === 'logs'" :disabled="!wellLogPreviews.length" :class="['visualization-mode-card', { active: visualizationMode === 'logs' }]" @click="selectVisualizationMode('logs')">
                <span class="mode-card-icon logs" aria-hidden="true">
                  <svg viewBox="0 0 28 28"><path d="M5 3v22M10 3v22M15 3v22M20 3v22M25 3v22"/><path d="M2 7h24M2 14h24M2 21h24"/><path class="mode-curve-a" d="M7 3c6 4-4 7 5 11s-5 7 3 11"/><path class="mode-curve-b" d="M19 3c-5 4 4 7-2 11s4 7-3 11"/></svg>
                </span>
                <span class="mode-card-copy"><small>LOG-ONLY VIEW</small><strong>仅测井展示</strong><em>不加载地震预览，独立查看常规九线综合图</em></span>
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
                  <h2>井震联合解释工作台</h2>
                  <span>当前数据：{{ selectedSeismicAsset?.name || '地震预览' }} · {{ selectedSeismicAsset?.kind }} · {{ wellTimeAlignmentSummary ? `${wellTimeAlignmentSummary.embedded ?? wellTimeAlignmentSummary.aligned} 口井轨迹已叠加` : '井位与井轨迹叠加' }}</span>
                </div>
                <div class="visualization-stage-actions">
                  <span>
                    项目可视化引擎 · 图层契约 {{ capabilities?.visualization.contract_version || "1.0" }}
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
                <iframe :key="visualizationUrl" title="当前任务井震联合解释工作台" :src="visualizationUrl" allow="fullscreen" allowfullscreen></iframe>
              </div>
            </section>
          </div>

          <section v-else-if="visualizationMode === 'seismic'" class="section-panel seismic-preview-empty">
            <strong>地震文件已登记，但当前没有可渲染的数据</strong>
            <p>系统会分别构建三维 Inline × Crossline 稀疏体和二维测线剖面；几何不完整或读取失败的文件仍保留在上方资产清单中，便于回到数据准备层复核。</p>
          </section>

          <section v-else-if="selectedWellLog" class="log-studio">
            <aside class="log-control-panel">
              <div class="log-panel-heading"><span>LOG-ONLY WORKBENCH</span><strong>仅测井展示</strong><p>该模式直接读取封存 LAS 预览，不加载或依赖地震场景。选择井，并按地质用途控制曲线图层。</p></div>
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

      <template v-if="activeView === 'preparation' && preparationScreen === 'fusion'">
        <section id="fusion-workbench" class="section-panel sample-builder data-fusion-workbench">
          <div class="section-heading">
            <div><span class="section-kicker">步骤 03</span><h2>井震标定与融合</h2><p>自动标定，并为合格井构建融合视图。</p></div>
            <span class="algorithm-badge">{{ preparedViewReady ? '融合视图已就绪' : sampleRunning ? '正在构建融合视图' : registrationRunning ? '正在精细标定' : formalRegistrationFusionBlocked ? '融合条件未通过' : horizontalRegistrationTaskId ? '仅空间 QC' : formalRegistrationReady ? '标定已完成' : '等待标定' }}</span>
          </div>
          <div class="fusion-status-rail" aria-label="井震融合状态">
            <article :data-ready="Boolean(dataSnapshotTaskId)"><span>01</span><div><strong>数据快照</strong><small>{{ dataSnapshotTaskId ? '已封存' : '未就绪' }}</small></div></article>
            <i aria-hidden="true">→</i>
            <article :data-ready="formalRegistrationReady"><span>02</span><div><strong>精细标定</strong><small>{{ registrationRunning ? `运行中 · ${progress}%` : formalRegistrationReady ? '已完成' : formalRegistrationFusionBlocked ? '融合条件未通过' : horizontalRegistrationTaskId ? '仅空间 QC' : '待运行' }}</small></div></article>
            <i aria-hidden="true">→</i>
            <article :data-ready="preparedViewReady"><span>03</span><div><strong>融合视图</strong><small>{{ preparedViewReady ? '可用于预测' : formalRegistrationFusionBlocked ? '未形成融合资格' : horizontalRegistrationTaskId && !registrationTaskId ? '融合条件待完善' : '待构建' }}</small></div></article>
          </div>
          <div class="sample-actions primary-registration-action">
            <button v-if="!registrationResult && !horizontalRegistrationResult" type="button" class="primary-button" :disabled="registrationRunning || sampleRunning || !dataSnapshotTaskId" @click="startDefaultWellSeismicWorkflow">
              {{ registrationRunning ? "正在标定…" : sampleRunning ? "正在构建融合视图…" : "开始标定与融合" }}
            </button>
            <button v-else-if="formalRegistrationReady && !preparedViewReady" type="button" class="primary-button" :disabled="sampleRunning" @click="runSampleBuilding">
              {{ sampleRunning ? "正在构建融合视图…" : "构建融合视图" }}
            </button>
            <button v-else-if="preparedViewReady" type="button" class="primary-button" @click="openPostFusionInferenceDestination">选择推理方式</button>
            <button v-else type="button" class="secondary-button" @click="openRegistrationPreparation">完善数据合同后重新融合</button>
            <p v-if="!dataSnapshotTaskId">请先完成数据封存。</p>
            <p v-else-if="!preparedViewReady">融合视图就绪前预测入口保持锁定；技术证据与实验候选保留在下方高级区域。</p>
            <p v-else>融合视图已通过完整门禁，可以进入预测。</p>
          </div>
          <section v-if="horizontalRegistrationResult" class="horizontal-registration-result">
            <div>
              <span>自动流程回退结果</span>
              <strong>仅完成几何空间覆盖检查</strong>
              <p>已匹配 {{ horizontalRegistrationResult.horizontal_registration.summary?.covered_station_count ?? 0 }}/{{ horizontalRegistrationResult.horizontal_registration.summary?.station_count ?? 0 }} 个轨迹站，{{ horizontalRegistrationResult.horizontal_registration.summary?.fully_covered_well_count ?? 0 }}/{{ horizontalRegistrationResult.horizontal_registration.summary?.well_count ?? 0 }} 口井完全覆盖。</p>
            </div>
            <p>当前证据不足以形成可审计 TWT，因此未授予融合资格。该结果不等同于精细标定成功；平台会继续锁定全部预测，直至完成正式融合。</p>
          </section>
          <details class="registration-advanced-panel">
            <summary>科学合同与高级候选</summary>
          <div class="contract-flow">
            <span>当前快照真实轨迹</span><b>→</b><span>实测时深（若有）/ AC·DT积分</span><b>→</b><span>SEG-Y约束 + 冻结概率模型</span><b>→</b>
            <span>时间均值 / 离散度候选</span><b>→</b><span>物理一致性门</span><b>→</b><span>{{ registrationTimeAxisPresentation.completionLabel }} / 仅几何 QC</span><b>→</b><span>仅合格井进入融合视图</span>
          </div>
          <div class="gate-grid">
            <article :class="{ ready: Boolean(dataSnapshotTaskId) }"><span>几何准备</span><strong>{{ dataSnapshotTaskId ? "快照已锁定" : "未就绪" }}</strong><p>DEV始终提供MD/TVD/XYZ，任何标定结果都不得替换它</p></article>
            <article :class="{ ready: Boolean(registrationResult?.registration?.registered_well_count) }"><span>冻结概率校正</span><strong>{{ p13RegistrationPresentation?.executionLabel || "等待执行" }}</strong><p>{{ p13RegistrationPresentation?.detail || "等待完整轨迹、声波和SEG-Y输入后运行概率标定模型" }}</p></article>
            <article :class="{ ready: Boolean(registrationResult?.registration?.registered_well_count) }"><span>{{ registrationTimeAxisPresentation.currentLabel }}</span><strong>{{ p13RegistrationPresentation?.primaryTrackLabel || "尚未生成" }}</strong><p v-if="registrationTimeAxisPresentation.limitation">{{ registrationTimeAxisPresentation.limitation }}</p><p v-if="p13RegistrationPresentation?.executed">模型候选：{{ p13RegistrationPresentation.acceptanceLabel }}；拒绝原因：{{ p13RegistrationPresentation.rejectionLabel }}</p><p v-else>模型未实际执行时不会显示为学习型标定成果</p></article>
            <article :class="{ ready: preparedViewReady }"><span>井震融合视图（仅合格井）</span><strong>{{ preparedViewReady ? `${(sampleResult?.prepared_view?.gates?.registration_fusion_ready_well_ids?.length || 0).toLocaleString()} 口融合井已封存` : "尚未构建" }}</strong><p>只包含通过时深质量、空间置信度与 CRS/单位核验的井；下游是否使用以消费收据为准</p></article>
          </div>
          <section class="alignment-candidate-card" aria-label="轨迹感知实验候选状态">
            <header>
              <div>
                <span>可选高级实验</span>
                <strong>轨迹感知井震校正</strong>
                <small>垂井、斜井、水平井共用；默认流程不会自动采用</small>
              </div>
              <b :data-available="Boolean(geoPathCandidateModel)">{{ geoPathCandidateModel?.status || '当前服务尚未加载' }}</b>
            </header>
            <div class="alignment-candidate-details">
              <article><span>科学状态</span><strong>{{ scientificStatusLabel(geoPathCandidateModel?.scientific_status || geoPathCandidateRelease?.scientific_status || 'candidate') }}</strong><small>不会自动覆盖默认概率约束或物理主轨</small></article>
              <article><span>运行状态</span><strong>{{ geoPathCandidateModel?.runtime_status || geoPathCandidateRelease?.runtime_status || '等待能力表刷新' }}</strong><small>{{ geoPathCandidateDataFlow?.runnable ? '适配器与 runner 已登记' : '需在后端重启后重新核验' }}</small></article>
              <article><span>必需输入</span><strong>{{ geoPathCandidateDataFlow?.required_modalities.map(formatModality).join(' / ') || 'SEG-Y / 测井 / 完整轨迹 / 已有标定先验' }}</strong><small>输出独立候选时深轨</small></article>
              <article><span>下游门禁</span><strong>默认不具备融合资格</strong><small>必须逐井检查接受率与修复状态，并显式接受后才生成新的实验性标定任务</small></article>
            </div>
            <footer>
              <div>
                <label>
                  <span>候选使用的封存 SEG-Y</span>
                  <select v-model="predictionSeismicPath" class="form-select">
                    <option value="">请选择三维地震</option>
                    <option v-for="source in predictionSources" :key="source.path" :value="source.path">{{ source.name }}</option>
                  </select>
                </label>
                <p v-if="geoPathSnapshotGeometryReady">{{ geoPathSnapshotGeometryEvidenceLabel }}</p>
                <p v-else class="candidate-contract-warning">当前数据快照既没有完整显式几何合同，也没有通过高级候选门禁的高置信自动几何收据。</p>
              </div>
              <div class="alignment-candidate-actions">
                <button type="button" class="secondary-button" @click="selectView('models')">查看证据与限制</button>
                <button v-if="!geoPathSnapshotGeometryReady" type="button" class="secondary-button" @click="openAdvancedDataContract">完善数据合同</button>
                <button
                  type="button"
                  class="primary-button"
                  :disabled="fusionWorkflowMutationRunning || !geoPathSnapshotGeometryReady || !geoPathCandidateDataFlow?.runnable || !dataSnapshotTaskId || !registrationTaskId || (!predictionSeismicPath && !predictionSources.length)"
                  @click="runGeoPathCandidate"
                >{{ geoPathCandidateRunning ? '正在运行轨迹感知校正…' : geoPathCandidateResult ? '重新生成实验候选' : '运行轨迹感知实验候选' }}</button>
              </div>
            </footer>
            <div v-if="geoPathCandidateRunning" class="task-progress candidate-progress">
              <div><span>{{ publicModelText(statusMessage) }}</span><strong>{{ progress }}%</strong></div>
              <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
            </div>
            <section v-if="geoPathCandidateReview" class="geopath-review-panel">
              <header>
                <div><span>不可变候选审核</span><strong>逐井选择并生成新的实验性标定</strong><small>候选哈希 {{ geoPathCandidateReview.candidate_manifest_sha256.slice(0, 16) }}… · 不确定度{{ geoPathCandidateReview.uncertainty_calibrated ? '已校准' : '未校准' }}</small></div>
                <b>{{ geoPathSelectedWellIds.length }} / {{ geoPathCandidateWells.length }} 已选择</b>
              </header>
              <div v-if="geoPathCandidateWells.length" class="geopath-well-review-list">
                <label v-for="well in geoPathCandidateWells" :key="well.id" :class="{ blocked: !well.acceptanceEligible, selected: geoPathSelectedWellIds.includes(well.id) }">
                  <input v-model="geoPathSelectedWellIds" type="checkbox" :value="well.id" :disabled="!well.acceptanceEligible || fusionWorkflowMutationRunning" />
                  <span><strong>{{ well.id }}</strong><small>{{ well.geometry }}</small></span>
                  <span><strong>{{ well.acceptedFraction === null ? '—' : `${(well.acceptedFraction * 100).toFixed(1)}%` }}</strong><small>模型接受点</small></span>
                  <span><strong>{{ well.apertureEligibleFraction === null ? '—' : `${(well.apertureEligibleFraction * 100).toFixed(1)}%` }}</strong><small>地震孔径覆盖</small></span>
                  <span><strong>{{ well.repairStatus }}</strong><small>{{ well.repairReason }}</small></span>
                  <b>{{ well.acceptanceEligible ? '可审核' : '禁止晋级' }}</b>
                </label>
              </div>
              <p v-else class="candidate-review-warning">后端尚未提供逐井 accepted_fraction / repair / acceptance_eligible 合同；前端已按失败关闭，不能接受任何井。</p>
              <div class="geopath-review-confirmation">
                <label>
                  <span>审核备注（可选）</span>
                  <textarea v-model="geoPathReviewNote" rows="2" maxlength="1000" class="form-select" placeholder="记录选择依据、异常井与适用边界"></textarea>
                </label>
                <label class="explicit-confirmation">
                  <input v-model="geoPathAcceptanceConfirmed" type="checkbox" :disabled="!geoPathSelectedEligible || fusionWorkflowMutationRunning" />
                  <span>我已逐井检查接受率、孔径覆盖和修复状态；同意所选井生成实验性标定。此操作不代表不确定度已校准。</span>
                </label>
                <button type="button" class="primary-button" :disabled="!geoPathAcceptanceReady || fusionWorkflowMutationRunning" @click="acceptSelectedGeoPathWells">
                  {{ geoPathAcceptanceRunning ? '正在生成新标定…' : `接受所选 ${geoPathSelectedWellIds.length} 口井` }}
                </button>
              </div>
              <div v-if="geoPathAcceptanceRunning" class="task-progress candidate-progress">
                <div><span>{{ publicModelText(statusMessage) }}</span><strong>{{ progress }}%</strong></div>
                <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
              </div>
              <div v-if="geoPathAcceptedRegistrationResult?.registration" class="accepted-registration-summary">
                <div><span>新实验性标定</span><strong>{{ geoPathAcceptedRegistrationTaskId }}</strong><small>{{ geoPathAcceptedRegistrationResult.registration.registered_well_count }} 口已显式接受 · 不确定度未校准</small></div>
                <button type="button" class="primary-button" :disabled="fusionWorkflowMutationRunning" @click="runSampleBuilding">以新标定重建融合视图</button>
              </div>
            </section>
          </section>
          </details>
          <div v-if="registrationRunning || sampleRunning" class="task-progress preparation-progress fusion-progress" :data-phase="fusionActivityPhase" role="status" aria-live="polite">
            <div class="preparation-activity">
              <span class="reading-orbit" aria-hidden="true"><i></i></span>
              <div><strong>{{ fusionActivityTitle }}</strong><span>{{ publicModelText(statusMessage) }}</span></div>
              <b>{{ progress }}%</b>
            </div>
            <div class="preparation-timing"><span>已运行 {{ formatDuration(fusionElapsedSeconds) }}</span><strong>{{ fusionEtaSeconds === null ? '预计时间计算中' : `预计剩余 ${formatDuration(fusionEtaSeconds)}` }}</strong></div>
            <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
            <small>{{ fusionEstimateLabel }}</small>
          </div>
          <p v-if="errorMessage" class="error-message">{{ publicModelText(errorMessage) }}</p>
          <div v-if="registrationPreparationRequired" class="registration-preparation-action">
            <span>不会自动跳转或放宽完整性门；原始文件路径仍在数据准备草稿中。</span>
            <button type="button" class="secondary-button" @click="openRegistrationPreparation">返回数据准备并重新封存</button>
          </div>
        </section>
        <details v-if="sampleResult?.matching" class="section-panel fusion-result-details">
          <summary>查看融合统计与成果</summary>
          <section class="summary-grid">
            <article><span>多模态样本</span><strong>{{ sampleResult.matching.sample_count.toLocaleString() }}</strong><p>融合视图记录</p></article>
            <article><span>可推理融合井</span><strong>{{ (sampleResult?.prepared_view?.gates?.registration_fusion_ready_well_ids?.length || 0).toLocaleString() }}</strong><p>可供兼容的预训练模型消费</p></article>
            <article><span>监督训练标签</span><strong>{{ (sampleResult.matching.training_eligible_count ?? 0).toLocaleString() }}</strong><p>为 0 不会阻断预训练模型推理</p></article>
            <article><span>可下载成果</span><strong>{{ preparedViewDownloadableCount }}</strong><p>原始数据未覆盖</p></article>
          </section>
        </details>
      </template>

      <template v-else-if="activeView === 'layerpulse'">
        <LayerPulseWorkbench
          :snapshot-id="dataSnapshotTaskId"
          :fusion-ready="preparedViewReady"
          :support-receipt="layerPulseSupportReceipt"
          :task-state="layerPulseTaskStateForCurrentSnapshot"
          v-model:selected-output-key="layerPulseSelectedOutputKey"
          v-model:canvas-mode="layerPulseCanvasMode"
          :base-visualization-url="layerPulseBaseVisualizationUrl"
          :result-visualization-url="layerPulseResultVisualizationUrl"
          :standalone-result-url="layerPulseStandaloneResultUrl"
          :output-downloads="layerPulseOutputDownloads"
          @run="runLayerPulse"
          @retry="runLayerPulse"
          @open-standalone="openLayerPulseStandalone"
        />
      </template>

      <template v-else-if="activeView === 'models'">
        <ModelCenter :capabilities="capabilities" />
      </template>

      <template v-else-if="activeView === 'prediction'">
        <section v-if="!activePredictionTaskSpec" id="prediction-task-panel" class="prediction-entry-layout" role="tabpanel">
          <section class="prediction-entry-viewer" aria-label="当前工区可视化">
            <header><div><span>LIVE VIEW</span><strong>当前工区</strong></div><button v-if="visualizationUrl" type="button" @click="predictionCanvasMode = 'base'">基础数据</button></header>
            <iframe v-if="visualizationUrl" :src="visualizationUrl" title="当前工区基础数据可视化" allow="fullscreen" allowfullscreen></iframe>
            <div v-else class="prediction-live-empty"><span aria-hidden="true">⌁</span><strong>尚无可视化数据</strong><p>先完成数据准备，再从上方选择预测任务。</p><button type="button" class="secondary-button" @click="selectView('preparation')">进入数据与融合</button></div>
          </section>
        </section>
        <section
          v-else-if="runnablePredictionModels.length || predictionResult"
          id="prediction-task-panel"
          class="section-panel faultseg-runner"
          role="tabpanel"
          :aria-labelledby="`prediction-task-tab-${activePredictionTask}`"
        >
          <div class="prediction-workbench-layout">
            <section class="prediction-live-stage" aria-label="预测联动可视化">
              <header>
                <div>
                  <span>LIVE VIEW</span>
                  <strong>{{ predictionWorkbenchShowingResult ? '预测结果' : '基础数据' }}</strong>
                  <small>{{ predictionWorkbenchPhase }}</small>
                </div>
                <nav aria-label="可视化内容切换">
                  <button
                    type="button"
                    :class="{ active: !predictionWorkbenchShowingResult }"
                    :disabled="!visualizationUrl"
                    @click="predictionCanvasMode = 'base'"
                  >基础数据</button>
                  <button
                    type="button"
                    :class="{ active: predictionWorkbenchShowingResult }"
                    :disabled="!predictionResultRenderable"
                    :title="wellSequenceLinkedViewerUnavailable ? '该历史结果缺少当前井侧展示合同，请重新运行一次' : '查看本次预测结果'"
                    @click="predictionCanvasMode = 'result'"
                  >预测结果</button>
                  <a v-if="predictionWorkbenchStandaloneUrl && !isWellSequenceResult" :href="predictionWorkbenchStandaloneUrl" target="_blank" rel="noopener">独立窗口 ↗</a>
                </nav>
              </header>
              <div class="prediction-live-frame">
                <iframe
                  v-if="predictionWorkbenchUrl"
                  :key="predictionWorkbenchUrl"
                  :title="predictionWorkbenchShowingResult ? '本次预测结果可视化' : '当前工区基础数据可视化'"
                  :src="predictionWorkbenchUrl"
                  allow="fullscreen"
                  allowfullscreen
                ></iframe>
                <div v-else class="prediction-live-empty">
                  <span aria-hidden="true">⌁</span>
                  <strong>尚无可视化数据</strong>
                  <p>先完成数据准备，运行或选择历史成果后会自动显示在这里。</p>
                  <button v-if="!dataSnapshotTaskId" type="button" class="secondary-button" @click="selectView('preparation')">进入数据与融合</button>
                </div>
                <div v-if="predictionBusyForActiveTask" class="prediction-live-overlay" role="status" aria-live="polite">
                  <span class="prediction-pulse" aria-hidden="true"></span>
                  <div>
                    <strong>{{ predictionWorkbenchPhase }}</strong>
                    <small>{{ publicModelText(statusMessage) }}</small>
                    <small class="prediction-connection-note">{{ publicModelText(predictionConnectionDetail) }}</small>
                  </div>
                  <b>{{ progress }}%</b>
                  <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
                </div>
              </div>
              <footer>
                <span>{{ selectedSeismicAsset?.name || predictionSeismicPath.replaceAll('\\', '/').split('/').at(-1) || '等待地震数据' }}</span>
                <span>{{ selectedPredictionModelName }}</span>
                <b :data-state="predictionWorkbenchPhase">{{ predictionWorkbenchPhase }}</b>
              </footer>
            </section>

            <aside class="prediction-control-panel" :aria-busy="predictionBusyForActiveTask">
              <div v-if="anotherPredictionTaskRunning" class="background-task-notice" role="status">
                <strong>{{ publicModelText(runningPredictionTaskSpec?.short_name, '另一项预测') }}正在后台运行</strong>
                <span>可继续查看当前任务成果与井联动视图；新任务提交需等待后台任务完成。</span>
                <small>{{ publicModelText(predictionConnectionDetail) }}</small>
              </div>
              <fieldset v-if="runnablePredictionModels.length" class="prediction-control-fieldset" :disabled="predictionBusy">
          <div class="section-heading">
            <div>
              <h2>{{ publicModelText(activePredictionTaskSpec?.name) }}</h2>
              <p>{{ publicModelText(activePredictionTaskSpec?.description) }}</p>
            </div>
            <span class="algorithm-badge">{{ isFaultSegModel ? (faultSegSnapshotSourceReady ? 'SourceSnapshot 可运行' : '待唯一三维体') : preparedViewReady ? '可运行' : '待融合' }}</span>
          </div>
          <p class="snapshot-policy">
            {{ isFaultSegModel ? `断层识别输入：${faultSegSnapshotSourceReason} ${predictionCompatibilityReason}` : `预测门禁：${preparedViewReady ? '井震融合视图已就绪，可以提交当前模型。' : '必须先完成井震精细标定与融合视图。'}` }}
          </p>
          <div class="gate-grid">
            <label>
              <span>推理模型</span>
              <select v-model="selectedPredictionModelId" class="form-select" @change="handlePredictionModelChange">
                <option v-for="model in runnablePredictionModels" :key="model.id" :value="model.id">{{ compactModelSpecPresentationName(model) }}</option>
              </select>
            </label>
            <label v-if="selectedModelRequiresSeismic">
              <span>三维地震文件</span>
              <select v-model="predictionSeismicPath" class="form-select" @change="handlePredictionSourceChange">
                <option value="">请选择 SEG-Y</option>
                <option v-for="source in predictionSources" :key="source.path" :value="source.path">
                  {{ source.name }} · {{ source.samples_per_trace }} × {{ source.trace_count }} 道
                </option>
              </select>
            </label>
            <label v-if="selectedModelRequiresSeismic && !isFaultSegModel && !isSurfaceSegModel && !isWellFuseGeobodyModel && !isHorizonModel && !isF3FaciesModel">
              <span>中心测试体块</span>
              <select v-model.number="predictionCropSize" class="form-select">
                <option :value="32">32³（快速验证）</option>
                <option :value="64">64³</option>
                <option :value="96">96³</option>
                <option :value="128">128³</option>
              </select>
            </label>
            <label v-if="!isFaultSegModel">
              <span>计算设备</span>
              <select v-model="predictionDevice" class="form-select">
                <option value="auto">自动选择</option>
                <option value="cpu">CPU</option>
                <option value="cuda">GPU / CUDA</option>
              </select>
            </label>
            <label v-if="selectedModelRequiresSeismic && !isFaultSegModel && !isSurfaceSegModel && !isHorizonModel && !isF3FaciesModel">
              <span>判别阈值 {{ predictionThreshold.toFixed(2) }}</span>
              <input v-model.number="predictionThreshold" type="range" min="0" max="1" step="0.01" class="form-range" />
            </label>
            <template v-if="isFaultSegModel">
              <div class="fault-scope-selector" role="radiogroup" aria-label="断层预测范围">
                <label
                  v-if="selectedPredictionModelId === 'faultseg_3d'"
                  :class="{ active: faultSegScope === 'center_block_1' }"
                >
                  <input v-model="faultSegScope" type="radio" value="center_block_1" :disabled="!faultSegFormalInputReady" />
                  <strong>工区中心单块（默认）</strong>
                  <small>{{ faultSegFormalInputReady ? '在 Z / Inline / Crossline 三轴中心确定性截取 1 个完整 128³ 子体积。' : '当前 Inline 数不足 128，正式 128³ 单块不可用。' }}</small>
                  <b>{{ faultSegFormalInputReady ? '推荐 · 最快' : '当前数据不可用' }}</b>
                </label>
                <label :class="{ active: faultSegScope === 'full_volume' }">
                  <input v-model="faultSegScope" type="radio" value="full_volume" :disabled="!faultSegFormalInputReady" />
                  <strong>全区连续重建</strong>
                  <small>{{ faultSegFormalInputReady ? '遍历全部重叠窗口并融合为连续概率体，数据量和磁盘占用都很大。' : '当前至少一个空间轴不足正式 128³ 窗口。' }}</small>
                  <b>{{ faultSegFormalInputReady ? '约 20–30 分钟' : '当前数据不可用' }}</b>
                </label>
              </div>
              <details class="source-contract-advanced fault-seg-development-options">
                <summary>
                  <div>
                    <span>OVERLAP-INFERENCE POLICY</span>
                    <strong>查看重叠推理策略</strong>
                    <small>{{ faultSegScope === 'center_block_1' ? '中心位置与 128³ 窗口固定；单块直接输出，不做重叠融合。' : '窗口参数固定，所有窗口统一使用同一 checkpoint 和概率融合合同。' }}</small>
                  </div>
                </summary>
                <div class="fault-seg-development-grid">
                  <p><strong>子体积</strong><span>128 × 128 × 128</span></p>
                  <p><strong>窗口重叠</strong><span>{{ faultSegScope === 'full_volume' ? '64 × 64 × 64' : '0 × 0 × 0' }}</span></p>
                  <p><strong>重建方式</strong><span>{{ faultSegScope === 'center_block_1' ? '中心单块直接输出' : '重叠区概率加权融合' }}</span></p>
                  <label>
                    <span>计算设备</span>
                    <select v-model="predictionDevice" class="form-select">
                      <option value="auto">自动选择（推荐）</option>
                      <option value="cpu">CPU</option>
                      <option value="cuda">GPU / CUDA</option>
                    </select>
                  </label>
                </div>
              </details>
            </template>
            <template v-if="isSurfaceSegModel">
              <details class="source-contract-advanced surface-seg-development-options">
                <summary>
                  <div>
                    <span>ADVANCED DEVELOPMENT</span>
                    <strong>高级开发与调试选项</strong>
                    <small>仅快速联调时切换少量 Inline；正式运行无需展开或修改。</small>
                  </div>
                </summary>
                <div class="surface-seg-development-grid">
                  <label>
                    <span>处理规模</span>
                    <select v-model="surfaceSegScope" class="form-select">
                      <option value="full">完整三维 SEG-Y（正式默认）</option>
                      <option value="smoke">开发验证（前若干 Inline）</option>
                    </select>
                  </label>
                  <label v-if="surfaceSegScope === 'smoke'">
                    <span>开发验证 Inline 数</span>
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
                <span>基础分割批量</span>
                <select v-model.number="surfaceSegformerBatchSize" class="form-select">
                  <option :value="1">1（最低显存）</option>
                  <option :value="2">2（推荐）</option>
                  <option :value="4">4</option>
                  <option :value="8">8</option>
                </select>
              </label>
                  <label>
                    <span>精细分割批量</span>
                    <select v-model.number="surfaceMask2formerBatchSize" class="form-select">
                      <option :value="1">1（8 GB 显存推荐）</option>
                      <option :value="2">2</option>
                      <option :value="4">4</option>
                    </select>
                  </label>
                </div>
              </details>
            </template>
            <template v-if="isF3FaciesModel">
              <label>
                <span>推理模式</span>
                <select v-model="f3FaciesMode" class="form-select">
                  <option value="auto_roi">自动选择有效三维 ROI（推荐）</option>
                  <option value="single_trace">自动选择代表性单道（快速检查）</option>
                  <option value="manual_roi">手动三维 ROI（高级）</option>
                </select>
              </label>
              <template v-if="f3FaciesMode === 'manual_roi'">
                <label><span>TWT 起点 / 样点数</span><div class="inline-fields"><input v-model.number="f3TStart" type="number" min="0" class="form-select" /><input v-model.number="f3TCount" type="number" min="1" class="form-select" /></div></label>
                <label><span>Inline 索引起点 / 数量</span><div class="inline-fields"><input v-model.number="f3InlineStart" type="number" min="0" class="form-select" /><input v-model.number="f3InlineCount" type="number" min="1" class="form-select" /></div></label>
                <label><span>Crossline 索引起点 / 数量</span><div class="inline-fields"><input v-model.number="f3CrosslineStart" type="number" min="0" class="form-select" /><input v-model.number="f3CrosslineCount" type="number" min="1" class="form-select" /></div></label>
              </template>
            </template>
          </div>
          <div v-if="!isFaultSegModel && selectedModelPrefersPreparedView" class="prepared-view-preference" :data-ready="preparedViewReady">
            <div>
              <strong>{{ preparedViewReady ? '将绑定现有 PreparedView' : '等待平台统一融合门禁' }}</strong>
              <p>最终以运行后的消费收据为准。</p>
            </div>
            <button v-if="!preparedViewReady" type="button" class="secondary-button compact" @click="selectView('samples')">进入标定融合</button>
          </div>
          <div class="sample-actions">
            <button type="button" class="primary-button" :disabled="predictionBusy || (!isFaultSegModel && fusionInputsMutating) || !selectedPredictionWorkflowGateReady || !predictionInputReady || !selectedPredictionModel" @click="runPrediction">
              {{ predictionRunButtonLabel }}
            </button>
            <p>{{ selectedModelHasCompletedResult ? '保留现有成果，完成后替换为新结果。' : isFaultSegModel ? (faultSegScope === 'full_volume' ? '完成全部重叠窗口并融合后，才会标记全区结果就绪。' : '默认运行工区中心单个 128³ 块，完成后直接显示中心块断层结果。') : '完成后自动切换当前页结果。' }}</p>
          </div>
          <details v-if="predictionResult && predictionTaskId" class="result-action-details result-download-action prediction-control-download">
            <summary>
              <strong>下载预测结果</strong>
              <small>{{ predictionPrimaryStandardDownloads.length ? `${predictionPrimaryStandardDownloads.length} 个标准文件` : '生成标准结果清单' }}</small>
            </summary>
            <div class="result-action-content">
              <div class="result-download-heading">
                <div>
                  <strong>标准成果文件</strong>
                  <p v-if="predictionStandardBundle && isFaultSegResult">{{ predictionPrimaryStandardDownloads.length }} 个全工区标准文件，包含概率体、可视化和完整性清单。</p>
                  <p v-else-if="predictionStandardBundle">{{ predictionStandardBundle.downloads.artifact_count }} 个可校验文件。</p>
                  <p v-else>打开清单时会核验历史产物，不会重新运行模型。</p>
                </div>
                <a :href="predictionStandardManifestUrl" target="_blank" rel="noopener" download class="secondary-button link-button result-manifest-link">结果清单</a>
              </div>
              <div v-if="predictionPrimaryStandardDownloads.length" class="output-list">
                <article v-for="artifact in predictionPrimaryStandardDownloads" :key="artifact.artifact_id">
                  <span>{{ publicModelText(artifact.output_key) }}</span>
                  <code>{{ publicModelText(artifact.filename) }}</code>
                  <small>{{ formatBytes(artifact.size_bytes) }} · SHA-256 {{ artifact.sha256.slice(0, 12) }}</small>
                  <a :href="backendPublicUrl(artifact.download_url)" target="_blank" rel="noopener" download>下载</a>
                </article>
              </div>
            </div>
          </details>
          <div v-if="predictionBusyForActiveTask" class="task-progress" role="status" aria-live="polite">
            <div><span>{{ publicModelText(statusMessage) }}</span><strong>{{ progress }}%</strong></div>
            <div class="progress-track"><span :style="{ width: `${progress}%` }"></span></div>
            <small class="prediction-connection-note">{{ publicModelText(predictionConnectionDetail) }}</small>
          </div>
          <p v-if="errorMessage" class="error-message" role="alert">{{ publicModelText(errorMessage) }}</p>
              </fieldset>
              <section v-else class="prediction-result-only-panel">
                <span>历史成果</span>
                <strong>{{ publicModelText(activePredictionTaskSpec?.name) }}</strong>
                <p>当前模型已退出在线运行目录，已有成果仍可在左侧查看和导出。</p>
              </section>
            </aside>
          </div>
          <section v-if="predictionResult && predictionTaskId" class="prediction-result-completion" aria-label="预测完成后的主要操作">
            <header class="result-completion-heading">
              <span>{{ wellSequenceLinkedViewerUnavailable ? '历史结果 · 需更新' : isLegacyHorizonResult ? '历史结果 · 只读' : isFaultSegResult ? (faultResultIsFullSurvey ? '全区断层识别完成' : faultResultIsCenterBlock ? '中心单块预测完成' : faultResultIsRepresentative128 ? '历史128块结果 · 只读' : '旧实验结果 · 只读') : '预测完成' }}</span>
              <strong>{{ wellSequenceLinkedViewerUnavailable ? `${publicModelText(activePredictionTaskSpec?.name || predictionResult.task_name)}需重新运行` : isFaultSegResult ? (faultResultIsFullSurvey ? '全区断层概率体已就绪' : faultResultIsCenterBlock ? '工区中心 128³ 结果已就绪' : faultResultIsRepresentative128 ? '历史代表块结果仅供查看' : '历史非正式结果仅供查看') : `${publicModelText(activePredictionTaskSpec?.name || predictionResult.task_name)}已就绪` }}</strong>
              <p v-if="wellSequenceLinkedViewerUnavailable">旧成果仍可下载；重新运行后会启用井位分类条联动。</p>
              <p v-else-if="isFaultSegResult && !isCurrentFaultPrediction(predictionResult)">旧结果仍可查看和下载，但不会作为当前流程完成成果。</p>
              <p v-else-if="faultResultIsCenterBlock">结果完整覆盖工区三轴中心的单个 128³ 块；如需连续全区成果，请选择全区连续重建。</p>
            </header>
            <div v-if="visualizationUrl || (predictionResultRenderable && !isWellSequenceResult)" class="result-view-toolbar">
              <button v-if="visualizationUrl" type="button" class="secondary-button compact" @click="predictionCanvasMode = 'base'">对比基础数据</button>
              <a v-if="predictionResultRenderable && !isWellSequenceResult" :href="predictionStandardVisualizationUrl" target="_blank" rel="noopener" class="secondary-button link-button compact">独立窗口 ↗</a>
            </div>
            <div v-if="isFaultSegResult" class="fault-result-primary-summary" :class="{ legacy: !isCurrentFaultPrediction(predictionResult) }" aria-label="断层结果摘要">
              <article>
                <span>解释结果</span>
                <strong>{{ faultResultIsFullSurvey ? '全区断层概率体' : faultResultIsCenterBlock ? '工区中心断层块' : faultResultIsRepresentative128 ? '历史 128 个断层代表块' : '历史非正式产物' }}</strong>
                <small>{{ faultResultIsFullSurvey ? '128³ 重叠滑窗 · 64³ 重叠 · 概率加权融合' : faultResultIsCenterBlock ? '128³ 单块 · 三轴中心确定性截取' : faultResultIsRepresentative128 ? '历史模式 · 128³ 独立块 · 只读' : '只读保留，不属于当前支持范围' }}</small>
              </article>
              <article>
                <span>空间范围</span>
                <strong>{{ faultResultScopeLabel }}</strong>
                <small>以标准空间交付收据为准</small>
              </article>
              <article>
                <span>查看方式</span>
                <strong>{{ faultResultIsFullSurvey ? '全区 3D / Inline / Crossline / 时间切片' : faultResultIsCenterBlock ? '中心块 3D / Inline / Crossline / 时间切片' : faultResultIsRepresentative128 ? '历史代表块网格 / 单块三维查看' : '旧结果只读查看' }}</strong>
                <small>{{ faultResultIsFullSurvey ? '所有视图来自同一重叠融合概率体' : faultResultIsCenterBlock ? '保留真实坐标与单次推理收据' : faultResultIsRepresentative128 ? '每块保留真实坐标与独立推理收据' : '不作为新流程完成状态' }}</small>
              </article>
            </div>
            <div class="result-action-grid result-secondary-action-grid">
              <details class="result-action-details result-technical-action">
                <summary>
                  <strong>更多技术详情</strong>
                  <small>{{ isFaultSegResult ? '概率审计、输入证明与完整产物' : '输入证明、状态与完整产物' }}</small>
                </summary>
                <div class="result-action-content result-technical-content">
                  <div class="result-technical-meta">
                    <article><span>任务 ID</span><code>{{ predictionTaskId }}</code></article>
                    <article><span>模型</span><strong>{{ modelPresentationName(predictionResult.model_id, predictionResult.model_name, predictionResult.scientific_status) }}</strong><code>{{ publicModelIdentifier(predictionResult.model_id) }}</code></article>
                    <article><span>科学状态</span><strong>{{ scientificStatusLabel(predictionResult.scientific_status) }}</strong><small>{{ predictionResult.scientific_status || 'unknown' }}</small></article>
                    <article><span>输出合同</span><strong>{{ predictionOutputContract || '未登记' }}</strong><small>{{ predictionResult.device }}</small></article>
                  </div>
                  <section class="prediction-input-attestation" aria-label="预测输入证明">
                    <article>
                      <span>实际数据来源</span>
                      <strong>{{ predictionSourceLabel }}</strong>
                      <small>{{ predictionSourceDetail }}</small>
                    </article>
                    <article :class="`registration-${predictionRegistrationStatus || 'unattested'}`">
                      <span>标定消费状态</span>
                      <strong>{{ predictionRegistrationLabel }}</strong>
                      <code>{{ predictionRegistrationStatus }}</code>
                      <small>{{ predictionRegistrationDetail }}</small>
                    </article>
                    <article>
                      <span>融合视图消费</span>
                      <strong>{{ predictionPreparedViewStatus === 'used' ? '已核验消费' : predictionPreparedViewStatus === 'available_not_used' ? '可用但未消费' : '未请求' }}</strong>
                      <code>{{ predictionPreparedViewStatus }}</code>
                      <small>{{ predictionResult.prepared_view_consumption?.prepared_view_id || sampleBuildingTaskId || '未绑定 PreparedView' }}</small>
                    </article>
                  </section>
                  <section v-if="predictionResult.standard_spatial_export?.status" class="result-technical-note">
                    <strong>空间交付</strong>
                    <p>{{ predictionSpatialScopeLabel }} · {{ predictionSpatialScopeDetail }}<template v-if="predictionResult.standard_spatial_export.slice_count"> · {{ predictionResult.standard_spatial_export.slice_count }} 个数值切片</template></p>
                  </section>
                  <section v-if="predictionVisualizationUrl && !isWellSequenceResult" class="result-technical-note">
                    <strong>可视化状态</strong>
                    <p v-if="predictionDisplayAccepted">已通过展示验收；主入口会打开标准可视化。</p>
                    <p v-else-if="isF3FaciesCandidateResult">六类地震相迁移候选可视化已就绪。</p>
                    <p v-else-if="predictionCandidateRenderable">实验候选可视化已就绪，不代表精度验收。</p>
                    <p v-else-if="predictionDisplayAcceptance?.diagnostics.length">{{ publicModelText(predictionDisplayAcceptance.diagnostics.join('；')) }}</p>
                    <p v-else>{{ publicModelText(predictionDisplayAcceptance?.reason_codes.join('；'), '历史任务未登记展示验收合同') }}</p>
                  </section>
                  <div v-if="isWellSequenceResult" class="summary-grid">
                    <article><span>数据集 / 井数</span><strong>{{ predictionResult.dataset }} · {{ predictionResult.well_count }}</strong><p>{{ predictionOutputContract || '井序列成果合同' }}</p></article>
                    <article v-if="predictionResult.target"><span>预测目标</span><strong>{{ predictionResult.target }}</strong><p>{{ predictionTargetSemantics }}</p></article>
                    <article v-if="predictionResult.validation"><span>{{ predictionResult.validation.metric_name }}</span><strong>{{ Number(predictionResult.validation.metric_value ?? 0).toFixed(4) }}</strong><p>整井三折验证；基线 {{ Number(predictionResult.validation.baseline_value ?? 0).toFixed(4) }}</p></article>
                    <article><span>输出轴 / 设备</span><strong>{{ predictionOutputAxes.join(' / ') || 'MD' }} · {{ predictionResult.device }}</strong><p>由结果合同识别为逐井序列，不依赖模型 ID 硬编码</p></article>
                  </div>
                  <div v-else-if="predictionResult.facies" class="summary-grid">
                    <article><span>{{ predictionResult.inference.mode === 'single_trace' ? '单道相序列' : '三维相 ROI' }}</span><strong>{{ predictionResult.facies.shape_t_inline_xline.join(' × ') }}</strong><p>TWT / Inline / Crossline</p></article>
                    <article><span>离散分类结果</span><strong>{{ predictionResult.facies.class_codes?.length || 0 }} 类</strong><p>有效道 {{ (Number(predictionResult.facies.valid_trace_fraction ?? 0) * 100).toFixed(2) }}%</p></article>
                    <article><span>公开稠密基准冻结测试</span><strong>mIoU {{ Number(predictionResult.facies.f3_frozen_test_miou ?? 0).toFixed(4) }}</strong><p>Macro-F1 {{ Number(predictionResult.facies.f3_frozen_test_macro_f1 ?? 0).toFixed(4) }}；不适用于当前工区验收</p></article>
                    <article><span>运行设备</span><strong>{{ predictionResult.device }}</strong><p>当前输出边界：跨工区迁移候选</p></article>
                  </div>
                  <div v-else class="summary-grid">
                    <article><span>{{ predictionResult.segmentation ? '输入剖面体' : '输入体块' }}</span><strong>{{ predictionInputShape.join(' × ') }}</strong><p>{{ predictionInputAxes.join(' / ') }}</p></article>
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
                  <section v-if="predictionOutputEntries.length" class="result-artifact-section">
                    <h3>完整原始产物</h3>
                    <div class="output-list">
                      <article v-for="[name, path] in predictionOutputEntries" :key="name">
                        <span>{{ publicModelText(name) }}</span><code>{{ artifactDisplayName(name, path) }}</code>
                        <span v-if="isDirectoryOutput(name, path)" class="directory-output-note">目录成果 · 请先在服务端打包</span>
                        <a v-else-if="path" :href="taskArtifactUrl(predictionTaskId, name)" target="_blank" rel="noopener" download>下载成果</a>
                      </article>
                    </div>
                  </section>
                </div>
              </details>
            </div>
          </section>
          <section class="model-flow-contract compact-model-flow" aria-label="所选模型运行链路">
            <header>
              <div>
                <span>WORKFLOW</span>
                <strong>本次运行链路</strong>
                <small>只保留关键节点；输入合同与成果文件已收进技术明细。</small>
              </div>
              <b>{{ selectedPredictionModelName }}</b>
            </header>
            <div class="five-stage-flow compact-stage-flow">
              <template v-for="(stage, index) in selectedModelFlowStages" :key="stage.id">
                <article :data-state="stage.state" :title="stage.detail">
                  <span>{{ stage.order }}</span>
                  <div>
                    <strong>{{ stage.name }}</strong>
                    <small>{{ stage.stateLabel }}</small>
                  </div>
                </article>
                <i v-if="index < selectedModelFlowStages.length - 1" aria-hidden="true"></i>
              </template>
            </div>
            <details class="model-flow-technical">
              <summary>
                <div>
                  <strong>输入与产物明细</strong>
                  <small>模型合同、降级策略、节点说明与文件入口</small>
                </div>
                <span>展开</span>
              </summary>
              <div class="model-flow-technical-body">
                <div class="model-flow-contract-identity">
                  <article>
                    <span>模型标识</span>
                    <code>{{ publicModelIdentifier(selectedPredictionModelId) }}</code>
                  </article>
                  <article>
                    <span>输出合同</span>
                    <strong>{{ selectedModelDataFlow?.output_contract || '等待后端声明' }}</strong>
                  </article>
                </div>
                <div class="model-requirement-grid">
                  <article>
                    <span>必需输入</span>
                    <strong v-if="selectedModelRequiredModalities.length">{{ selectedModelRequiredModalities.map(formatModality).join(' / ') }}</strong>
                    <strong v-else>无额外模态</strong>
                    <small>缺少任一必需输入时关闭运行入口</small>
                  </article>
                  <article>
                    <span>可选输入</span>
                    <strong v-if="selectedModelOptionalModalities.length">{{ selectedModelOptionalModalities.map(formatModality).join(' / ') }}</strong>
                    <strong v-else>无</strong>
                    <small>缺失时仅按声明的降级策略运行</small>
                  </article>
                  <article v-if="!isSnapshotOnlyDownstreamWellTask">
                    <span>数据来源</span>
                    <strong>{{ selectedModelSourceModes.map(formatSourceMode).join(' / ') }}</strong>
                    <small>本次：{{ formatSourceMode(selectedSourceMode) }}</small>
                  </article>
                  <article>
                    <span>标定 / 融合</span>
                    <strong>{{ selectedModelRegistrationPolicy === 'required' ? '标定必需' : selectedModelRegistrationPolicy === 'optional_control' ? '标定可选控制' : '不要求标定' }} · {{ selectedModelPreparedViewPolicyLabel }}</strong>
                    <small>{{ selectedModelDataFlow?.degradation_policy || '等待模型级降级策略' }}</small>
                  </article>
                </div>
                <div class="flow-stage-detail-list">
                  <article v-for="stage in selectedModelFlowStages" :key="`detail-${stage.id}`" :data-state="stage.state">
                    <header><span>{{ stage.order }}</span><strong>{{ stage.name }}</strong><b>{{ stage.stateLabel }}</b></header>
                    <p>{{ stage.detail }}</p>
                    <div v-if="flowStageArtifacts(stage.id).length" class="flow-stage-artifacts">
                      <template v-for="artifact in flowStageArtifacts(stage.id)" :key="`${artifact.taskId}:${artifact.name}`">
                        <span v-if="artifact.directory" :title="artifact.name">目录需打包</span>
                        <a v-else :href="taskArtifactUrl(artifact.taskId, artifact.name)" target="_blank" rel="noopener" download>{{ artifact.label }}</a>
                      </template>
                    </div>
                  </article>
                </div>
              </div>
            </details>
          </section>
        </section>
        <section
          v-else
          id="prediction-task-panel"
          class="section-panel prediction-task-empty"
          role="tabpanel"
          :aria-labelledby="`prediction-task-tab-${activePredictionTask}`"
        >
          <div class="section-heading">
            <div>
              <h2>{{ publicModelText(activePredictionTaskSpec?.name) }}</h2>
              <p>{{ publicModelText(activePredictionTaskSpec?.description) }} 当前尚未注册可运行模型；模型接入后会根据任务绑定自动出现在这里。</p>
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
        <section class="section-panel acceptance-workbench">
          <div class="section-heading">
            <div><span class="section-kicker">UNIFIED ACCEPTANCE</span><h2>统一结果验收工作台</h2><p>执行成功、可渲染、科学状态与定量验收分别呈现；candidate 永远不等于 accepted。</p></div>
            <div class="artifact-heading-actions">
              <button v-if="wellLogPreviews.length" type="button" class="secondary-button compact" @click="openWellLogWorkspace">查看 LAS 井曲线</button>
            </div>
          </div>
          <div class="acceptance-snapshot-picker">
            <label for="acceptance-snapshot-select">
              <span>验收数据组</span>
              <small>只切换结果视图，不修改当前工作流</small>
            </label>
            <select
              id="acceptance-snapshot-select"
              v-model="selectedAcceptanceSnapshotId"
              :disabled="acceptanceLoading || !acceptanceSnapshotCatalog.length"
              @change="loadSelectedAcceptanceSnapshot"
            >
              <option v-for="snapshot in acceptanceSnapshotCatalog" :key="snapshot.snapshot_id" :value="snapshot.snapshot_id">
                {{ acceptanceSnapshotOptionLabel(snapshot) }}
              </option>
            </select>
          </div>
          <div v-if="acceptanceSnapshotDetail" class="acceptance-snapshot-strip">
            <span>SourceSnapshot</span><strong>{{ acceptanceSnapshotSourceLabel }} · {{ acceptanceSnapshotDetail.snapshot.snapshot_id }}</strong><small>{{ acceptanceGroups.length }} 组任务证据</small>
          </div>
          <p v-if="acceptanceError" class="error-message">{{ publicModelText(acceptanceError) }}</p>
          <div v-if="acceptanceGroups.length" class="acceptance-layout">
            <aside class="acceptance-task-list" aria-label="验收任务分组">
              <button v-for="group in acceptanceGroups" :key="group.taskId" type="button" :class="{ selected: selectedAcceptanceGroup?.taskId === group.taskId }" @click="selectAcceptanceGroup(group.taskId)">
                <span>{{ group.family }}</span><strong>{{ group.label }}</strong><small>{{ group.executionStatus }} · {{ acceptanceStatusLabel(group.acceptanceStatus) }}</small>
              </button>
            </aside>
            <div v-if="selectedAcceptanceGroup" class="acceptance-detail">
              <div class="acceptance-status-grid">
                <article><span>执行状态</span><strong>{{ selectedAcceptanceGroup.executionStatus }}</strong><small>后台任务真实终态</small></article>
                <article><span>可视化状态</span><strong>{{ visualizationStatusLabel(selectedAcceptanceGroup.visualizationStatus) }}</strong><small>是否有明确渲染合同</small></article>
                <article><span>科学状态</span><strong>{{ selectedAcceptanceGroup.scientificStatus }}</strong><small>模型或产物声明</small></article>
                <article :class="selectedAcceptanceGroup.acceptanceStatus"><span>验收结论</span><strong>{{ acceptanceStatusLabel(selectedAcceptanceGroup.acceptanceStatus) }}</strong><small>仅 accepted 可声明验收通过</small></article>
              </div>
              <div v-if="selectedAcceptanceGroup.acceptanceStatus === 'candidate'" class="candidate-boundary">
                <strong>候选结果，不代表定量精度验收</strong><p>允许查看真实模型产物与不确定性，但没有把候选可视化提升为 accepted，也没有推断不存在的实测真值。</p>
              </div>
              <div v-if="selectedAcceptanceGroup.standardResult" class="prediction-visualization-entry standard-result-entry">
                <div><span>STANDARD RESULT</span><strong>当前下游任务的标准可视化与下载入口</strong><p>旧任务会在首次打开时核验并补建路径无关 Manifest；不会重跑模型，也不会把候选结果改称已验收。</p></div>
                <div class="artifact-heading-actions result-primary-actions">
                  <a :href="backendPublicUrl(`/api/v1/tasks/${encodeURIComponent(selectedAcceptanceGroup.taskId)}/standard-results/visualization`)" target="_blank" rel="noopener" class="primary-button link-button result-primary-link">查看本次预测结果</a>
                  <a :href="backendPublicUrl(`/api/v1/tasks/${encodeURIComponent(selectedAcceptanceGroup.taskId)}/standard-results/manifest`)" target="_blank" rel="noopener" download class="secondary-button link-button result-manifest-link">下载标准结果清单（JSON）</a>
                </div>
              </div>
              <iframe v-if="acceptanceVisualizationUrl" class="acceptance-cigvis" :src="acceptanceVisualizationUrl" title="统一数据与候选结果可视化"></iframe>
              <div v-else-if="!selectedPlanViewArtifact" class="empty-inline">此任务没有后端声明的可渲染结果；下方仍提供原始证据下载。</div>

              <section v-if="selectedPlanViewArtifact" class="plan-view-panel">
                <div class="plan-view-heading">
                  <div>
                    <span>HORIZONTAL REGISTRATION</span>
                    <strong>井—地震水平配准平面图</strong>
                    <p>只显示同一水平 CRS 下的井位/轨迹、最近地震道与覆盖范围；不推断时深关系，不声明垂向标定、融合样本或训练资格。</p>
                  </div>
                  <button type="button" class="secondary-button compact" :disabled="planViewLoading" @click="loadPlanViewPreview">
                    {{ planViewLoading ? '正在读取…' : planViewPreview ? '重新读取平面图' : '读取水平配准平面图' }}
                  </button>
                </div>
                <p v-if="planViewError" class="error-message">{{ planViewError }}</p>
                <div v-if="planViewPreview" class="plan-view-canvas">
                  <div class="plan-view-meta">
                    <span>{{ planViewPreview.horizontalCrs }}</span>
                    <span>{{ planViewPreview.wells.length }} 口可绘制井</span>
                    <span>{{ planViewPreview.polygons.length }} 个地震覆盖边界</span>
                  </div>
                  <svg viewBox="0 0 1000 600" role="img" aria-label="井轨迹与地震道覆盖水平配准平面图">
                    <rect x="0" y="0" width="1000" height="600" class="plan-view-background" />
                    <path
                      v-for="(polygon, polygonIndex) in planViewPreview.polygons"
                      :key="`footprint-${polygonIndex}`"
                      :d="planViewPolygonPath(polygon)"
                      class="plan-view-footprint"
                    />
                    <template v-for="well in planViewPreview.wells" :key="well.name">
                      <line
                        v-for="(station, stationIndex) in well.stations.filter((item) => item.traceX !== null && item.traceY !== null)"
                        :key="`${well.name}-trace-${stationIndex}`"
                        :x1="planViewX(station.x)"
                        :y1="planViewY(station.y)"
                        :x2="planViewX(station.traceX as number)"
                        :y2="planViewY(station.traceY as number)"
                        :class="['plan-view-link', { uncovered: !station.covered }]"
                      />
                      <path v-if="well.stations.length > 1" :d="planViewWellPath(well)" :class="['plan-view-well-path', { uncovered: well.horizontalStatus !== 'fully_covered' }]" />
                      <circle
                        v-for="(station, stationIndex) in well.stations"
                        :key="`${well.name}-station-${stationIndex}`"
                        :cx="planViewX(station.x)"
                        :cy="planViewY(station.y)"
                        :r="well.geometryMode === 'head_only' ? 6 : 3.5"
                        :class="['plan-view-station', { uncovered: !station.covered, 'head-only': well.geometryMode === 'head_only' }]"
                      />
                      <text :x="planViewX(well.stations[0].x) + 8" :y="planViewY(well.stations[0].y) - 7" class="plan-view-label">{{ well.name }}</text>
                    </template>
                  </svg>
                  <div class="plan-view-legend">
                    <span><i class="footprint"></i>地震有效覆盖边界</span>
                    <span><i class="covered"></i>覆盖内井位/轨迹</span>
                    <span><i class="head-only"></i>仅井口 XY</span>
                    <span><i class="link"></i>最近地震道连接</span>
                  </div>
                </div>
              </section>

              <section class="sequence-candidate-panel">
                <div class="sequence-panel-heading">
                  <div>
                    <span>WELL-SIDE CANDIDATE</span>
                    <strong>{{ selectedAcceptanceIsFracture ? '裂缝相对发育深度段' : '沿深度候选曲线' }}</strong>
                    <p v-if="selectedAcceptanceIsFracture">只展示后端已封存的低/中/高连续 MD 深度段，不显示原始分值或概率曲线。这是井侧相对排序，不是地震体素裂缝分割。</p>
                    <p v-else>只读取后端明确登记的 CSV；NPZ 不在浏览器中猜测结构。每条曲线独立归一化，仅用于趋势审阅。</p>
                  </div>
                  <div v-if="selectedAcceptanceGroup.sequenceArtifacts.length" class="sequence-actions">
                    <select v-model="selectedSequenceArtifact" @change="resetSequenceSelection">
                      <option v-for="artifact in selectedAcceptanceGroup.sequenceArtifacts" :key="artifact.name" :value="artifact.name">{{ artifactDisplayName(artifact.name, artifact.path) }}</option>
                    </select>
                    <button type="button" class="secondary-button sequence-load-button" :disabled="sequenceLoading || !selectedSequenceArtifact" @click="loadSequencePreview">{{ sequenceLoading ? '正在读取…' : selectedAcceptanceIsFracture ? '立即显示裂缝分段' : '立即显示候选曲线' }}</button>
                  </div>
                </div>
                <p v-if="sequenceError" class="error-message">{{ sequenceError }}</p>
                <p v-if="sequencePreviewLimited" class="sequence-preview-limit">大文件仅流式读取前 2 MiB 并抽样显示；完整文件仍可在证据区下载。</p>
                <div v-if="fractureIntervalPreview && selectedAcceptanceIsFracture" class="fracture-sequence-chart">
                  <div class="sequence-axis"><span>MD (m)</span><strong>{{ fractureIntervalPreview.depthMin.toFixed(1) }}</strong><strong>{{ fractureIntervalPreview.depthMax.toFixed(1) }}</strong></div>
                  <div class="fracture-interval-track">
                    <svg viewBox="0 0 200 640" preserveAspectRatio="none" role="img" aria-label="井侧裂缝相对发育连续深度段">
                      <rect x="20" y="10" width="160" height="620" class="fracture-track-background" />
                      <rect
                        v-for="(segment, index) in fractureIntervalPreview.segments"
                        :key="`${segment.level}-${segment.topMd}-${index}`"
                        x="21"
                        width="158"
                        :y="segment.y"
                        :height="segment.height"
                        :class="`fracture-band-${segment.level}`"
                      />
                    </svg>
                    <div class="fracture-sequence-legend">
                      <span><i class="high"></i>相对较强</span>
                      <span><i class="medium"></i>相对中等</span>
                      <span><i class="low"></i>相对较弱</span>
                    </div>
                  </div>
                  <div class="fracture-interval-list">
                    <header><strong>连续深度段</strong><small>{{ fractureIntervalPreview.segments.length }} 段</small></header>
                    <article v-for="(segment, index) in fractureIntervalPreview.segments" :key="`interval-${segment.topMd}-${index}`">
                      <i :class="segment.level"></i>
                      <div><strong>{{ segment.label }}</strong><span>{{ segment.topMd.toFixed(2) }}–{{ segment.bottomMd.toFixed(2) }} m</span></div>
                      <small>{{ segment.thicknessM.toFixed(2) }} m · {{ segment.sampleCount }} 点</small>
                    </article>
                    <em>科学边界：井侧测井候选，仅表达本井内相对发育级别；不生成或暗示三维空间裂缝体。</em>
                  </div>
                </div>
                <div v-else-if="sequencePreview" class="sequence-chart">
                  <div class="sequence-axis"><span>{{ sequencePreview.depthName }}</span><strong>{{ Math.min(...sequencePreview.depths).toFixed(1) }}</strong><strong>{{ Math.max(...sequencePreview.depths).toFixed(1) }}</strong></div>
                  <svg viewBox="0 0 100 640" preserveAspectRatio="none" role="img" aria-label="沿深度候选趋势">
                    <path v-for="(series, index) in sequencePreview.series" :key="series.id" :d="sequenceCurvePath(series)" :stroke="['#1769d2','#12a57a','#d98127','#8b5cf6','#d24f68','#2f8797'][index]" />
                  </svg>
                  <div class="sequence-legend"><span v-for="(series, index) in sequencePreview.series" :key="series.id"><i :style="{ background: ['#1769d2','#12a57a','#d98127','#8b5cf6','#d24f68','#2f8797'][index] }"></i>{{ series.id }} · {{ series.min.toPrecision(4) }}–{{ series.max.toPrecision(4) }}</span></div>
                </div>
                <div v-else-if="!selectedAcceptanceGroup.sequenceArtifacts.length" class="sequence-evidence-only">{{ selectedAcceptanceIsFracture ? '该历史任务未生成受封存的 fracture_intervals.csv；旧分值曲线不会作为标准裂缝结果展示，请重跑模型生成连续深度段。' : '没有登记可安全解析的 CSV 序列。可下载 NPZ/清单证据，但浏览器不会猜测轴序、dtype 或深度语义。' }}</div>
              </section>

              <section class="acceptance-evidence">
                <h3>可下载证据</h3>
                <div v-if="selectedAcceptanceGroup.artifacts.length" class="output-list">
                  <article v-for="artifact in selectedAcceptanceGroup.artifacts" :key="artifact.name">
                    <span>{{ publicModelText(artifact.name) }}</span><code>{{ artifactDisplayName(artifact.name, artifact.path) }}</code>
                    <span v-if="artifact.directory" class="directory-output-note">目录成果 · 不提供虚假下载链接</span>
                    <a v-else :href="taskArtifactUrl(selectedAcceptanceGroup.taskId, artifact.name)" target="_blank" rel="noopener" download>下载证据</a>
                  </article>
                </div>
                <div v-else class="empty-inline">该任务未登记可下载产物。</div>
                <ul v-if="selectedAcceptanceGroup.warnings.length" class="acceptance-limitations"><li v-for="warning in selectedAcceptanceGroup.warnings" :key="warning">{{ publicModelText(warning) }}</li></ul>
              </section>
            </div>
          </div>
          <div v-else-if="!acceptanceLoading" class="empty-inline">当前快照没有可验收的任务记录。</div>
        </section>
      </template>

      <template v-else-if="activeView === 'settings'">
        <section class="section-panel knowledge-config-panel" aria-labelledby="knowledge-config-title">
          <div class="section-heading">
            <div>
              <h2 id="knowledge-config-title">知识库与算法配置</h2>
              <p>配置与实现分离，比赛数据变化时优先增加映射或厂商配置，不修改核心代码。</p>
            </div>
          </div>
          <div
            class="knowledge-orbit"
            role="list"
            :aria-label="`${configurationLibraryNodes.length} 项可独立维护的知识库与算法配置`"
          >
            <svg
              class="knowledge-orbit-links"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              aria-hidden="true"
              focusable="false"
            >
              <ellipse class="knowledge-orbit-ring knowledge-orbit-ring-outer" cx="50" cy="50" rx="38" ry="38" />
              <ellipse class="knowledge-orbit-ring knowledge-orbit-ring-inner" cx="50" cy="50" rx="25" ry="25" />
              <line
                v-for="node in configurationLibraryNodes"
                :key="`link-${node.id}`"
                class="knowledge-orbit-link"
                x1="50"
                y1="50"
                :x2="node.x"
                :y2="node.y"
                vector-effect="non-scaling-stroke"
              />
            </svg>
            <div class="knowledge-orbit-halo" aria-hidden="true"></div>
            <div class="knowledge-orbit-core">
              <span>CONFIG KNOWLEDGE</span>
              <strong>知识与算法核</strong>
              <small>{{ configurationLibraryNodes.length }} 项配置独立维护</small>
            </div>
            <div
              v-for="library in configurationLibraryNodes"
              :key="library.id"
              class="knowledge-orbit-node"
              :style="library.style"
              :data-library-id="library.id"
              role="listitem"
            >
              <article class="knowledge-orbit-node-surface">
                <span class="knowledge-orbit-node-mark" aria-hidden="true"><i></i></span>
                <div class="knowledge-orbit-node-copy">
                  <span>可独立维护</span>
                  <h3>{{ publicModelText(library.name) }}</h3>
                  <code>{{ publicModelText(library.file) }}</code>
                </div>
              </article>
            </div>
          </div>
        </section>
        <section class="section-panel llm-settings-panel">
          <div class="section-heading">
            <div>
              <h2>Kimi 受控自动修复</h2>
              <p>启用后只向 Kimi 官方接口发送去路径化的表头、列结构、统计与错误摘要，不发送原文件或数据行；Kimi 只能生成白名单解析补丁，系统在隔离副本中重读并通过格式与物理规则后才采纳，不执行模型返回的脚本或任意文件操作。</p>
            </div>
            <span :class="['llm-status-pill', { ready: capabilities?.llm.available }]">
              {{ capabilities?.llm.available ? "已就绪" : capabilities?.llm.configured ? "未启用" : "待配置" }}
            </span>
          </div>
          <div class="llm-config-overview">
            <article><span>提供方 / 模式</span><strong>{{ capabilities?.llm.provider || "—" }} · {{ capabilities?.llm.api_mode || "—" }}</strong></article>
            <article><span>模型 / 推理强度</span><strong>{{ capabilities?.llm.model || "未配置" }}<template v-if="capabilities?.llm.reasoning_effort"> · {{ capabilities.llm.reasoning_effort }}</template></strong></article>
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
          <div v-if="capabilities?.llm.credential_file || capabilities?.llm.key_file || capabilities?.llm.config_file" class="llm-credential-location">
            <span>服务器端密钥文件（仅后端读取）</span>
            <code>{{ capabilities?.llm.credential_file || capabilities?.llm.key_file || capabilities?.llm.config_file }}</code>
            <small>填写 {{ capabilities?.llm.credential_variable || "KIMI_API_KEY" }}；请勿在浏览器中输入或保存密钥。</small>
          </div>
        </section>
        <section class="section-panel cache-settings-panel">
          <div class="section-heading">
            <div>
              <h2>运行缓存</h2>
              <p>清理可重建的切片与临时文件，并重置当前浏览器会话，让工作台以未选择任务的干净状态重新打开。</p>
            </div>
            <span class="cache-size-pill">{{ cacheStatusLoading ? "正在读取" : cacheStatusError ? "读取失败" : formatBytes(cacheStatus?.totals.bytes || 0) }}</span>
          </div>
          <div class="cache-safety-grid">
            <article><span>会被清理</span><strong>页面会话 · 临时切片 · 内存 LRU</strong></article>
            <article><span>始终保留</span><strong>原始数据 · 模型 · 任务记录 · 快照 · 预测成果</strong></article>
          </div>
          <div class="cache-settings-actions">
            <div>
              <strong>安全清理边界</strong>
              <p>后端不接收自定义路径，只能访问平台专用缓存目录；按下按钮会先停止全部排队和运行任务，再执行清理。</p>
              <small v-if="cacheMessage" :class="{ error: cacheMessage.includes('失败') }">{{ cacheMessage }}</small>
            </div>
            <button
              type="button"
              class="secondary-button cache-clear-button"
              :disabled="cacheClearing || backendStatus !== 'online'"
              @click="clearPlatformCaches"
            >
              {{ cacheClearing ? "正在清理并重新开始…" : "清空缓存并重新开始" }}
            </button>
          </div>
        </section>
      </template>

      <FloatingAssistant
        v-show="activeView === 'assistant'"
        :visible="activeView === 'assistant'"
        :task-id="taskId"
        context-label="地层慧眼本地工作区"
        :llm-available="Boolean(capabilities?.llm.available)"
        :llm-model="capabilities?.llm.model"
        @navigate="handleAssistantNavigation"
      />
    </main>

    <dialog
      ref="runtimeContractDialog"
      class="runtime-contract-dialog"
      aria-labelledby="runtime-contract-title"
      aria-describedby="runtime-contract-description"
      @cancel.prevent="returnFromRuntimeContractReview"
      @keydown.esc.prevent.stop="returnFromRuntimeContractReview"
    >
      <form
        v-if="runtimeContractReview"
        class="runtime-contract-shell"
        @submit.prevent="confirmRuntimeContractReview"
      >
        <header class="runtime-contract-header">
          <div class="runtime-contract-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" role="img">
              <path d="M7 9.5c4-3.4 7.2-3.4 10.5 0s6.3 3.4 8.5 1.1M6 16c4.4-3.6 8-3.6 11.3 0 3.1 3.4 5.9 3.4 8.7 1.1M7 22.5c4-3.4 7.2-3.4 10.5 0s6.3 3.4 8.5 1.1" />
              <path d="M9.2 6.5v19M22.8 6.5v19" />
            </svg>
          </div>
          <div>
            <span>读取完成</span>
            <h2 id="runtime-contract-title">确认本次运行参数</h2>
            <p id="runtime-contract-description">已根据当前文件填入推荐值，可直接确认，也可以在这里修改。</p>
          </div>
          <button
            type="button"
            class="runtime-contract-close"
            aria-label="返回数据设置"
            :disabled="runtimeContractSubmitting"
            @click="returnFromRuntimeContractReview"
          >
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5.5 5.5 9 9m0-9-9 9" /></svg>
          </button>
        </header>

        <div class="runtime-contract-context">
          <span><i></i>已按本次资产填入推荐参数</span>
          <span v-if="runtimeContractReview.time_depth_asset_count">
            已识别 {{ runtimeContractReview.time_depth_asset_count }} 份时深资产
          </span>
          <span v-else>未读取独立时深资产</span>
        </div>

        <section class="runtime-contract-fields" aria-label="可审核运行参数">
          <label
            v-for="field in runtimeContractReview.fields"
            :key="field.key"
            class="runtime-contract-field"
          >
            <span class="runtime-contract-field-label">
              <b>{{ field.label }}</b>
              <small v-if="field.group">{{ field.group }}</small>
            </span>
            <span class="runtime-contract-control">
              <select
                v-if="field.control === 'select'"
                :value="runtimeContractControlValue(field)"
                @change="updateRuntimeContractValue(field, $event)"
              >
                <option
                  v-for="choice in field.choices || []"
                  :key="String(choice.value)"
                  :value="String(choice.value)"
                >{{ choice.label }}</option>
              </select>
              <input
                v-else
                :type="field.control === 'number' ? 'number' : 'text'"
                :step="field.key === 'seismic_srd_elevation_m' ? 0.01 : field.control === 'number' ? 'any' : undefined"
                :min="field.key === 'seismic_replacement_velocity_mps' ? 1 : field.key === 'seismic_srd_elevation_m' ? -10000 : undefined"
                :max="field.key === 'seismic_srd_elevation_m' ? 10000 : undefined"
                :value="runtimeContractControlValue(field)"
                @input="updateRuntimeContractValue(field, $event)"
              />
              <em v-if="field.unit">{{ field.unit }}</em>
              <svg v-if="field.control === 'select'" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
            </span>
            <small v-if="field.helper" class="runtime-contract-helper">{{ field.helper }}</small>
          </label>
        </section>

        <p v-if="runtimeContractError" class="runtime-contract-error" role="alert">{{ runtimeContractError }}</p>
        <p class="runtime-contract-attestation">
          <b>本次确认声明</b>
          <span>{{ runtimeContractAttestationText }}</span>
        </p>

        <footer class="runtime-contract-actions">
          <p>确认后生成可继续运行的新快照，原始文件保持不变。</p>
          <div>
            <button
              type="button"
              class="runtime-contract-back"
              :disabled="runtimeContractSubmitting"
              @click="returnFromRuntimeContractReview"
            >返回数据设置</button>
            <button
              type="submit"
              class="runtime-contract-confirm"
              :disabled="runtimeContractSubmitting"
            >
              <span>{{ runtimeContractSubmitting ? "正在封存…" : "确认上述参数并继续" }}</span>
              <svg v-if="!runtimeContractSubmitting" viewBox="0 0 18 18" aria-hidden="true"><path d="M4 9h10m-4-4 4 4-4 4" /></svg>
            </button>
          </div>
        </footer>
      </form>
    </dialog>

    <PostFusionInferenceDialog
      :open="postFusionInferenceOpen"
      :snapshot-id="postFusionInferenceContext?.snapshotId || ''"
      :registration-task-id="postFusionInferenceContext?.registrationTaskId || ''"
      :prepared-view-id="postFusionInferenceContext?.preparedViewId || ''"
      :ready-well-count="postFusionInferenceContext?.readyWellCount || 0"
      @original="choosePostFusionOriginal"
      @layerpulse="choosePostFusionLayerPulse"
      @close="closePostFusionInferenceDestination"
    />

  </div>
</template>
