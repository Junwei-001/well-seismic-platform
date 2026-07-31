export interface DataPathsPayload {
  seismic_paths: string[];
  log_paths: string[];
  well_paths: string[];
  auxiliary_paths: string[];
  recursive: boolean;
  lightweight: boolean;
  use_llm_fallback: boolean;
}

export interface SampleBuildingPayload extends DataPathsPayload {
  output_directory?: string;
}

export interface PredictionPayload {
  task_id: string;
  model_id: string;
  seismic_path: string;
  source_task_id?: string;
  crop_start?: [number, number, number];
  crop_size?: [number, number, number];
  patch_size?: [number, number, number];
  overlap?: [number, number, number];
  threshold?: number;
  device?: "auto" | "cpu" | "cuda";
  output_directory?: string;
  options?: Record<string, unknown>;
}

export interface PredictionResult {
  task_id: string;
  task_name: string;
  model_id: string;
  model_name: string;
  checkpoint?: string;
  checkpoints?: Record<string, unknown>;
  checkpoint_epoch?: number;
  device: string;
  input: Record<string, unknown> & {
    axes: string[];
    shape_zyx?: number[];
    shape_ics?: number[];
  };
  inference: Record<string, unknown> & {
    patch_size?: number[];
    overlap?: number[];
    normalization?: string;
    threshold?: number;
    query_threshold?: number;
    mask_threshold?: number;
  };
  probability?: {
    shape_zyx: number[];
    min: number;
    max: number;
    mean: number;
    positive_fraction: number;
  };
  segmentation?: {
    shape_ics: number[];
    label_range: [number, number];
    instance_count: number;
    confidence_min?: number;
    confidence_max?: number;
    confidence_mean?: number;
    cross_inline_consistent?: boolean;
  };
  outputs: Record<string, string | null>;
}

export interface PredictionTaskResult {
  prediction: PredictionResult;
  source_task_id?: string;
}

export interface PreparationIssue {
  id: string;
  stage: string;
  severity: "错误" | "警告" | "提示";
  title: string;
  message: string;
  source: string;
  sources?: string[];
  affected_count?: number;
  action: string;
  blocking: boolean;
  candidate_actions: string[];
  recommended_action: string;
  recommendation_source: "LLM" | "规则";
  recommendation_confidence: number | null;
  recommendation_reason: string;
  confirmation_status: "待人工确认" | "已确认采用" | "暂不采用" | "无需确认" | "已启用转换插件";
  confirmed_action: string;
  confirmed_at: string;
}

export interface PreparationStage {
  id: string;
  name: string;
  description: string;
  status: "阻断" | "需确认" | "待执行" | "未就绪" | "就绪";
  metrics: Record<string, string | number>;
  issue_count: number;
}

export interface PreparationReport {
  stages: PreparationStage[];
  issues: PreparationIssue[];
  summary: {
    blocking: number;
    warnings: number;
    information: number;
  };
  gates: {
    can_visualize: boolean;
    can_build_samples: boolean;
    can_train_seismic_baseline: boolean;
    can_train_multimodal: boolean;
    can_run_high_confidence_fusion?: boolean;
    can_run_prediction: boolean;
  };
}

export interface SeismicAssetSummary {
  name: string;
  path: string;
  dimension: string;
  evidence: string;
  trace_count: number;
  samples_per_trace: number;
  sample_interval_ms: number;
  shape_zyx: number[];
  inline_count: number;
  crossline_count: number;
  grid_coverage: number;
  confidence: number;
  issues: string[];
  model_compatibility: Record<string, {
    ready: boolean;
    reason: string;
    adapter: string;
    expected_axes: string[];
    patch_size: number[];
    shape_ics?: number[];
    source_shape_zyx?: number[];
    geometry_mode?: string;
    duplicate_trace_count?: number;
    platform_ordered_grid?: boolean;
    native_inline_count?: number | null;
    recommended_options?: Record<string, string | number | boolean>;
  }>;
}

export interface SeismicLinePreview {
  name: string;
  path: string;
  lineAxis: string;
  traceValues: Array<number | null>;
  distanceValues: Array<number | null>;
  timeValues: Array<number | null>;
  image: {
    shape: number[];
    encoding: string;
    values: string;
  };
  preview: {
    sampled_traces: number;
    source_trace_count: number;
    amplitude_scale_p99: number;
  };
}

export interface WorkflowResult {
  summary: {
    assets: number;
    duplicates_skipped: number;
    wells: number;
    seismic_files: number;
    registered_seismic_files?: number;
    log_files: number;
    metadata_files: number;
    auxiliary_files: number;
    uncertain: number;
    errors: number;
  };
  rows: Array<{ type: string; count: number; evidence: string; status: string }>;
  seismic: SeismicAssetSummary[];
  wells: WellEntity[];
  well_entities?: WellEntity[];
  assets: Array<{
    id: string;
    role: string;
    dataset: string;
    stage: string;
    path: string;
    size: number;
  }>;
  metadata_detection: Array<Record<string, unknown>>;
  errors: Array<Record<string, string>>;
  preparation: PreparationReport;
  data_snapshot?: {
    contract_version: string;
    snapshot_id: string | null;
    semantics: "model_neutral";
    source_assets: Record<string, number>;
    canonical_data: {
      seismic_geometry: {
        axes: string[];
        registered: number;
        readable: number;
        renderable_3d: number;
        renderable_2d?: number;
      };
      well_entities: { count: number };
      well_logs: { count: number };
    };
    derived_views: {
      visualization_preview: { status: string; model_specific: boolean };
      well_seismic_samples: {
        status: string;
        optional: boolean;
        model_specific: boolean;
        sample_count?: number;
        valid_window_count?: number;
        training_eligible_count?: number;
        output_directory?: string;
      };
    };
    downstream_policy: string;
  };
  visualization_preview?: {
    volumes: Array<{ name: string; path: string }>;
    lines2d: SeismicLinePreview[];
    trajectories: Array<{ name: string }>;
    wellLogs: WellLogPreview[];
    seismicInventory: SeismicAssetSummary[];
    issues: string[];
    source: string;
  };
  matching?: {
    sample_count: number;
    valid_window_count?: number;
    training_eligible_count?: number;
    coordinate_reference_verified?: boolean;
    vertical_alignment_counts?: Record<string, number>;
    output_directory: string;
    output_files: Record<string, string>;
  };
}

export interface WellLogPreviewCurve {
  id: string;
  label: string;
  unit: string;
  color: string;
  scale: "linear" | "log";
  values: Array<number | null>;
  validCount: number;
}

export interface WellLogPreview {
  id: string;
  name: string;
  wellName: string;
  source: string;
  version: string;
  depthUnit: string;
  depth: Array<number | null>;
  curves: WellLogPreviewCurve[];
  coverage: string;
}

export interface TransformationDraft {
  id: string;
  task_id: string;
  issue_id: string;
  title: string;
  explanation: string;
  confidence: number;
  operations: Array<Record<string, string | number>>;
  generated_code: string;
  tests: Array<{ name: string; passed: boolean; details: string }>;
  valid: boolean;
  status: "待人工启用" | "未通过验证" | "已启用";
  provider: string;
  model: string;
  generation_error: string;
}

export interface AssistantResponse {
  answer: string;
  actions: Array<{ label: string; target: string }>;
  source: string;
  provider: string;
  model: string;
  request_id: string;
  warning?: string;
}

export interface WellEntity {
  well_uid: string;
  name: string;
  aliases: string[];
  head_count: number;
  log_count: number;
  trajectory_count: number;
  conflicts: string[];
  logs: Array<{
    source: string;
    samples: number;
    curves: string[];
    issues: string[];
  }>;
}

export interface BackgroundTask {
  task_id: string;
  task_type?: "data_preparation" | "sample_building" | "model_prediction";
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  result: WorkflowResult | PredictionTaskResult | null;
  error: { type: string; message: string } | null;
}

export interface DemoPaths {
  available: boolean;
  seismic_paths: string[];
  log_paths: string[];
  well_paths: string[];
  auxiliary_paths: string[];
}

export interface ModelSpec {
  id: string;
  name: string;
  category: string;
  status: string;
  description: string;
  inputs: string[];
  outputs: string[];
  version: string;
  configurable: boolean;
  implementation?: string;
  metadata?: Record<string, unknown>;
}

export interface ModelInputAdapterCapability {
  model_id: string;
  source_formats: string[];
  array_axes: string[];
  tensor_axes: string[];
  dtype: string;
  patch_size?: number[];
  overlap?: number[];
  normalization: string;
  requires_logs: boolean;
  input_mode?: string;
  inference_plane?: string[];
  slice_axis?: string;
}

export interface PredictionTaskCapability {
  id: string;
  name: string;
  short_name: string;
  description: string;
  outputs: string[];
  output: string;
  required_modalities: string[];
  evaluation_metrics: string[];
  order: number;
  contract_version: string;
  model_id?: string;
  model_ids: string[];
  runnable_model_ids: string[];
  available: boolean;
  status: "可运行" | "等待模型插件";
}

export interface FusionStrategyCapability {
  id: string;
  name: string;
  stage: string;
  status: string;
  description: string;
  inputs: string[];
  output: string;
  training_required: boolean;
  recommended_for: string;
  contract_version: string;
}

export interface Capabilities {
  workflow: Array<{ id: string; name: string; purpose: string }>;
  plugin_contract: {
    entry_point_group: string;
    required_methods: string[];
    stable_inputs: string[];
  };
  plugin_load_errors: Array<{ plugin: string; error: string }>;
  visualization: {
    contract_version: string;
    scene_endpoint: string;
    engine?: {
      available: boolean;
      version: string;
      backend: string;
      preferred_backend: string;
      viser_available: boolean;
      fallback_backend: string;
      web_engine: string;
      error?: string;
    };
    layers: Array<{ id: string; name: string; status: string }>;
    extension_points: string[];
  };
  models: ModelSpec[];
  model_input_adapters: ModelInputAdapterCapability[];
  prediction_runner_model_ids: string[];
  prediction_tasks: PredictionTaskCapability[];
  interpretation_task_contract: {
    entry_point_group: string;
    plugin_load_errors: Array<{ plugin: string; error: string }>;
    model_binding: string;
    runner_binding: string;
  };
  runtime_plugin_contract: {
    input_adapter_entry_point_group: string;
    prediction_runner_entry_point_group: string;
    fusion_strategy_entry_point_group: string;
    plugin_load_errors: Array<{ plugin: string; error: string }>;
  };
  fusion_strategies: FusionStrategyCapability[];
  configuration_libraries: Array<{ id: string; name: string; file: string }>;
  llm: {
    enabled: boolean;
    configured: boolean;
    available: boolean;
    provider: string;
    api_mode: string;
    base_url: string;
    model: string;
    api_key_configured: boolean;
    min_confidence: number;
    max_calls_per_task: number;
    max_context_chars: number;
    send_file_names: boolean;
    allowed_decisions: string[];
    missing: string[];
    data_policy: string;
    trigger_policy: string;
    guardrails: string[];
    required_environment: string[];
  };
}

async function jsonRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    const detail =
      typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(detail || `请求失败：${response.status}`);
  }
  return response.json() as Promise<T>;
}

function postTask(url: string, payload: DataPathsPayload | SampleBuildingPayload): Promise<{ task_id: string }> {
  return jsonRequest(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function createPrediction(payload: PredictionPayload): Promise<{ task_id: string }> {
  return jsonRequest("/api/v1/prediction/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function health(): Promise<{ status: string; version: string }> {
  return jsonRequest("/api/v1/health");
}

export function demoPaths(): Promise<DemoPaths> {
  return jsonRequest("/api/v1/demo-paths");
}

export function getCapabilities(): Promise<Capabilities> {
  return jsonRequest("/api/v1/capabilities");
}

export function createDataPreparation(payload: DataPathsPayload): Promise<{ task_id: string }> {
  return postTask("/api/v1/data-preparation/tasks", payload);
}

export function createSampleBuilding(payload: SampleBuildingPayload): Promise<{ task_id: string }> {
  return postTask("/api/v1/data-preparation/multimodal-view-tasks", payload);
}

export function getTask(taskId: string): Promise<BackgroundTask> {
  return jsonRequest(`/api/v1/tasks/${encodeURIComponent(taskId)}`);
}

export function confirmPreparationIssue(
  taskId: string,
  issueId: string,
  decision: "确认采用" | "暂不采用",
  action = "",
): Promise<PreparationIssue> {
  return jsonRequest(
    `/api/v1/tasks/${encodeURIComponent(taskId)}/issues/${encodeURIComponent(issueId)}/confirmation`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, action }),
    },
  );
}

export function generateTransformationDraft(taskId: string, issueId: string): Promise<TransformationDraft> {
  return jsonRequest(
    `/api/v1/tasks/${encodeURIComponent(taskId)}/issues/${encodeURIComponent(issueId)}/transformation-drafts`,
    { method: "POST" },
  );
}

export function activateTransformationDraft(draftId: string): Promise<TransformationDraft> {
  return jsonRequest(`/api/v1/transformation-drafts/${encodeURIComponent(draftId)}/activation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmation: "确认启用" }),
  });
}

export function chatWithAssistant(message: string, taskId?: string): Promise<AssistantResponse> {
  return jsonRequest("/api/v1/assistant/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, task_id: taskId || null }),
  });
}
