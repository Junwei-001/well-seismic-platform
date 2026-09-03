export interface SurveyAttestationPayload {
  contract_version: "well-seismic.survey-attestation.v1";
  declared_srd_elevation_m: number;
  vertical_reference?: "MSL";
  time_domain?: "TWT";
  correction_state?: "corrected_to_srd";
  declaration_text?: string;
  source: "human_user";
  confirmation_channel: "user_ui";
  confirmed_at?: string;
}

export interface DataPathsPayload {
  seismic_paths: string[];
  survey_paths: string[];
  log_paths: string[];
  well_paths: string[];
  time_depth_paths: string[];
  interpretation_paths: string[];
  auxiliary_paths: string[];
  recursive: boolean;
  lightweight: boolean;
  use_llm_fallback: boolean;
  horizontal_crs_id?: string;
  well_source_crs_id?: string;
  seismic_source_crs_id?: string;
  horizontal_unit?: "m" | "ft" | "unknown";
  horizontal_axis_order?: "XY" | "YX" | "unknown";
  coordinate_reference_verified?: boolean;
  seismic_srd_elevation_m?: number;
  vertical_crs_id?: string;
  seismic_replacement_velocity_mps?: number;
  seismic_time_domain?: "TWT" | "OWT" | "unknown";
  seismic_correction_state?: "corrected_to_srd" | "uncorrected" | "unknown";
  segy_geometry_profile?: string;
  segy_inline_byte?: number;
  segy_crossline_byte?: number;
  segy_x_byte?: number;
  segy_y_byte?: number;
  segy_coordinate_scalar_byte?: number;
  well_coordinate_source_unit?: "m" | "ft" | "unknown";
  well_vertical_datum_source_unit?: "m" | "ft" | "unknown";
  las_twt_source_unit?: "ms" | "s" | "us" | "unknown";
  time_depth_default_depth_domain?: "md" | "tvd" | "tvdss" | "unknown";
  time_depth_default_depth_unit?: "m" | "ft" | "unknown";
  time_depth_default_time_unit?: "ms" | "s" | "us" | "unknown";
  time_depth_default_depth_datum?: "KB" | "GL" | "DF" | "RT" | "MSL" | "unknown";
  time_depth_default_depth_convention?: "depth_below_msl_positive_down" | "elevation_positive_up" | "unknown";
  time_depth_default_time_reference?: "SRD" | "KB" | "GL" | "DF" | "RT" | "unknown";
  time_depth_default_time_domain?: "TWT" | "OWT" | "unknown";
  time_depth_default_correction_state?: "corrected_to_srd" | "uncorrected" | "unknown";
  target_task_id?: string;
  target_model_id?: string;
  target_scope_explicit?: boolean;
  survey_attestation?: SurveyAttestationPayload;
}

export interface SampleBuildingPayload extends DataPathsPayload {
  output_directory?: string;
  source_snapshot_id?: string;
  registration_task_id?: string;
}

export interface RegistrationPreflightResponse {
  contract_version: "well-seismic.registration-preflight.v1";
  source_snapshot_id: string;
  derived_snapshot_id: string;
  resolution:
    | "reused_complete_snapshot"
    | "derived_bounded_machine_decision"
    | "reused_native_relative_snapshot";
  effective_request: SampleBuildingPayload;
  system_evidence_receipt: Record<string, unknown> | null;
  execution_contract?: Record<string, unknown> | null;
}

export type RuntimeContractValue = string | number | boolean;

export interface RuntimeContractReviewField {
  key: string;
  label: string;
  value: RuntimeContractValue;
  control: "text" | "number" | "select";
  choices?: Array<{ value: RuntimeContractValue; label: string }>;
  unit?: string;
  group?: string;
  helper?: string;
}

export interface RuntimeContractReview {
  contract_version: "well-seismic.runtime-contract-review.v1";
  required: boolean;
  profile_id: "CN_GENERAL_RUN_FIRST_V1" | string;
  fields: RuntimeContractReviewField[];
  values: Record<string, RuntimeContractValue>;
  time_depth_asset_count?: number;
}

export interface RuntimeContractConfirmationResponse {
  contract_version: "well-seismic.runtime-contract-confirmation.v1";
  source_snapshot_id: string;
  derived_snapshot_id: string;
  effective_request: SampleBuildingPayload;
  runtime_contract_receipt: Record<string, unknown>;
}

/**
 * The preflight endpoint intentionally fails closed.  A permitted geometry-only
 * fallback must be carried by the server's structured
 * `horizontal_fallback_allowed` flag; the absence of an independent time-depth
 * table is not sufficient by itself, and a generic 409 can never authorize
 * reuse of a snapshot for horizontal registration.
 */
export type RegistrationPreflightFailureKind =
  | "horizontal_only"
  | "blocked_integrity"
  | "needs_preparation"
  | "failed";

export interface HorizontalRegistrationPayload {
  source_snapshot_id: string;
  output_directory?: string;
}

export interface PredictionPayload {
  task_id: string;
  model_id: string;
  seismic_path?: string;
  raw_well_paths?: string[];
  raw_well_root?: string;
  trajectory_paths?: string[];
  source_task_id?: string;
  registration_task_id?: string;
  prepared_view_task_id?: string;
  crop_start?: [number, number, number];
  crop_size?: [number, number, number];
  patch_size?: [number, number, number];
  overlap?: [number, number, number];
  threshold?: number;
  device?: "auto" | "cpu" | "cuda";
  output_directory?: string;
  options?: Record<string, unknown>;
}

export type ResultDisplayStatus = "accepted" | "failed_diagnostic" | "unavailable";

export interface ResultDisplayAcceptanceDecision {
  contract_version: string;
  display_status: ResultDisplayStatus;
  quantitative_status: "passed" | "failed" | "not_run" | "unavailable";
  visual_status: "passed" | "failed" | "not_run" | "unavailable";
  panels: Record<string, Record<string, unknown>>;
  diagnostics: string[];
  reason_codes: string[];
  missing_panels: string[];
  panels_visible: boolean;
  diagnostics_visible: boolean;
}

export type RegistrationConsumptionStatus =
  | "not_requested"
  | "available_not_used"
  | "lineage_only"
  | "used"
  | "unattested";

export interface RegistrationConsumptionDecision {
  contract_version: string;
  status: RegistrationConsumptionStatus;
  registration_requested: boolean;
  registration_available: boolean;
  registration_consumed: boolean;
  claim_sources: string[];
  evidence: Record<string, unknown> | null;
  issues: string[];
}

export interface PreparedViewConsumptionDecision {
  status: "not_requested" | "available_not_used" | "used";
  prepared_view_id?: string;
  prepared_view_sha256?: string;
  prepared_view_consumed: boolean;
}

export interface PredictionSourceIdentity {
  kind: "seismic_file" | "raw_well_files" | "sealed_snapshot_wells" | "registered_well_dataset" | string;
  path?: string;
  sha256?: string;
  source_snapshot_id?: string;
  source_snapshot_fingerprint?: string;
  raw_well_paths?: string[];
  raw_well_root?: string;
  dataset?: string;
  well_ids?: string[];
  integrity_status?: string;
}

export interface RegistrationCandidateWellReview {
  well_id?: string;
  well_uid?: string;
  well_name?: string;
  geometry?: string;
  accepted_fraction?: number;
  aperture_eligible_fraction?: number;
  repair_status?: string;
  repair_reason?: string;
  acceptance_eligible?: boolean;
  coverage?: number;
}

export interface RegistrationCandidateReview {
  status: string;
  candidate_manifest_sha256: string;
  candidate_product_sha256?: string;
  well_ids: string[];
  wells?: RegistrationCandidateWellReview[];
  requires_explicit_well_selection: boolean;
  uncertainty_calibrated: boolean;
  accept_endpoint: string;
  confirmation: "ACCEPT_GEOPATH_CANDIDATE";
}

export interface StandardResultArtifact {
  artifact_id: string;
  output_key: string;
  source_kind: "file" | "directory_child" | string;
  relative_path?: string | null;
  filename: string;
  format: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
  download_url: string;
}

export interface StandardResultVisualizationAsset extends StandardResultArtifact {
  asset_id: string;
  name: string;
  kind: "volume" | "surface" | "points" | "trajectory" | "well_curve" | "table" | "image" | "diagnostic_table" | string;
  role: string;
  renderer: string;
  visualization_url: string;
  axis_order: string[];
  units: Record<string, string>;
  geometry: Record<string, unknown>;
  visible_by_default: boolean;
}

export interface StandardResultBundle {
  contract_version: "well-seismic.standard-result-bundle.v1" | string;
  manifest_contract_version: string;
  bundle_id: string;
  execution_task_id: string;
  interpretation_task_id: string;
  model_id: string;
  bundle_sha256: string;
  source_snapshot: { id: string; sha256?: string | null };
  output_integrity: { contract_version: string; sha256: string };
  visualization: {
    available: boolean;
    entry_url?: string | null;
    platform_viewer_url?: string | null;
    preferred_asset_id?: string | null;
    renderers: string[];
    assets: StandardResultVisualizationAsset[];
  };
  downloads: {
    manifest_url: string;
    artifact_count: number;
    artifacts: StandardResultArtifact[];
    groups: Array<{
      output_key: string;
      kind: "directory" | string;
      size_bytes: number;
      file_count: number;
      sha256: string;
      download_strategy: string;
      child_artifact_ids: string[];
      archive_artifact_id?: string | null;
    }>;
  };
}

export interface PredictionResult {
  schema_version?: string;
  task_id: string;
  task_name: string;
  model_id: string;
  model_name: string;
  checkpoint?: string;
  checkpoints?: Record<string, unknown>;
  checkpoint_epoch?: number;
  device: string;
  dataset?: string;
  well_count?: number;
  target?: string;
  units?: string;
  validation?: {
    metric_name: string;
    metric_value: number;
    baseline_value?: number;
    hydrocarbon_f1?: number;
    r2?: number;
    macro_f1?: number;
    applies_to_current_survey?: boolean;
  };
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
    mode?: "roi" | "single_trace" | string;
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
  facies?: {
    shape_t_inline_xline: number[];
    class_codes: number[];
    confidence_mean: number;
    uncertainty_mean: number;
    valid_trace_fraction: number;
    f3_frozen_test_macro_f1: number;
    f3_frozen_test_miou: number;
  };
  task_catalog?: Array<{
    name?: string;
    display_name?: string;
    output_key: string;
    kind: "classification" | "regression" | string;
    channels?: number;
    background_index?: number | null;
    class_names?: string[];
    selection?: string | null;
    logits_shape_ctix?: number[];
    output_shape_tix?: number[];
    finite?: boolean;
    artifact_key?: string | null;
    preview_artifact_key?: string | null;
    primary_download_artifact_key?: string | null;
    download_artifact_keys?: Array<{
      artifact_key: string;
      format?: string;
      role?: string;
    }>;
    class_legend_artifact_key?: string | null;
    legend_download_artifact_key?: string | null;
    [key: string]: unknown;
  }>;
  outputs: Record<string, string | null>;
  model_executed?: boolean;
  execution_status?: string;
  scientific_status?: string;
  validated_scope?: string;
  warnings?: string[];
  well_outputs?: Array<{
    well_id: string;
    sample_count?: number;
    prediction_csv?: string;
    prediction_npz?: string;
    source_path?: string;
    [key: string]: unknown;
  }>;
  artifact_bundle?: {
    id?: string;
    scientific_status?: string;
    warnings?: string[];
    layers?: Array<Record<string, unknown>>;
    source_snapshot_id?: string | null;
    [key: string]: unknown;
  };
  standard_result_bundle?: StandardResultBundle;
  standard_spatial_export?: {
    contract_version?: string;
    status?: string;
    authoritative_output_key?: string;
    axis_order?: string[];
    shape?: number[];
    slice_count?: number;
    slice_bundle_output_key?: string;
    slice_bundle_sha256?: string;
    coverage?: string;
    scope?: "full_survey" | "declared_roi" | string;
    is_complete_for_declared_roi?: boolean;
    is_full_survey?: boolean;
    roi?: {
      axis_order?: string[];
      roi_start?: number[];
      roi_shape?: number[];
      source_shape?: number[];
      full_survey?: boolean;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
  /** Stable result semantics.  New runners should prefer these fields over
   * model-id-specific presentation rules. */
  output_contract?: string | {
    contract_version?: string;
    primary_output?: string;
    primary_semantics?: string;
    primary_decision_rule?: string;
    target?: string;
    [key: string]: unknown;
  };
  output_axes?: string[];
  result_contract?: string | {
    id?: string;
    schema_version?: string;
    axes?: string[];
    output_axes?: string[];
  };
  candidate_review?: RegistrationCandidateReview;
  diagnostics?: Array<Record<string, unknown>>;
  registration_consumption?: RegistrationConsumptionDecision;
  registration_consumed?: boolean;
  registration_usage?: RegistrationConsumptionStatus;
  prepared_view_consumption?: PreparedViewConsumptionDecision;
  provenance?: Record<string, unknown> & {
    prediction_source_identity?: PredictionSourceIdentity;
    source_snapshot_id?: string;
    source_snapshot_fingerprint?: string;
    registration_usage?: RegistrationConsumptionStatus;
    prepared_view_usage?: "not_requested" | "available_not_used" | "used";
  };
  display_acceptance_decision?: ResultDisplayAcceptanceDecision;
  candidate_visualization_decision?: {
    contract_version: string;
    display_status: "experimental_candidate" | "engineering_candidate" | "unavailable";
    renderable: boolean;
    scientific_status: string;
    validated_scope?: string;
    visualization_kind?: "named_horizon_surfaces" | "fault_probability_volume" | string;
    horizon_names?: string[];
    quantitative_acceptance_claimed: false;
    truth_metrics_used: false;
    error_metrics_used: false;
    reason_codes: string[];
  };
}

export interface PredictionTaskResult {
  prediction: PredictionResult;
  source_task_id?: string;
  registration_task_id?: string;
  prepared_view_task_id?: string;
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
  original_blocking?: boolean;
  required_for_task?: boolean;
  group_key?: string;
  scope_disposition?: "inventory_only" | string;
  attention_required?: boolean;
  display_bucket?: "must_attention" | "audit" | string;
  display_severity?: "错误" | "警告" | "提示" | string;
  affected_entities?: string[];
  candidate_actions: string[];
  recommended_action: string;
  recommendation_source: "LLM" | "规则";
  recommendation_confidence: number | null;
  recommendation_reason: string;
  confirmation_status: "待人工确认" | "已确认采用" | "暂不采用" | "无需确认" | "已启用转换插件" | "本任务不需要" | "LLM已补全" | "LLM已补全并复检" | "系统已自动处理" | "需一次集中补充";
  confirmed_action: string;
  confirmed_at: string;
  resolution_mode?: "none" | "llm_autofill" | "rule_autofill" | "survey_input";
  autofill_patch?: Record<string, unknown>;
  autofill_validation?: string[];
  /** Evidence-bounded candidates emitted by the local rules/Kimi review.
   * They are suggestions for the next immutable snapshot, never mutations of
   * the snapshot that produced this report. */
  contract_candidates?: SourceContractCandidate[];
  confirmation_group?: string;
}

export type SourceContractCandidateField =
  | "vertical_crs_id"
  | "seismic_srd_elevation_m"
  | "seismic_time_domain"
  | "seismic_correction_state";

export interface SourceContractCandidate {
  field: SourceContractCandidateField | string;
  value: string | number | boolean | null;
  confidence?: number | null;
  status?: "verified" | "candidate" | "review_required" | "insufficient" | "conflict" | string;
  source?: string;
  evidence?: string | string[];
  inference_source?: string;
  requires_human_confirmation?: boolean;
  auto_applied?: boolean;
}

export interface SurveyContractCandidateGroup {
  schema_version: string;
  confirmation_group: "seismic_vertical_contract" | string;
  candidates: SourceContractCandidate[];
  confirmation_required: boolean;
  unresolved_fields: string[];
  policy: string;
}

export type SurveyContractRequestPatch = Partial<Pick<
  DataPathsPayload,
  "vertical_crs_id" | "seismic_srd_elevation_m" | "seismic_time_domain" | "seismic_correction_state"
>>;

export interface PreparationStage {
  id: string;
  name: string;
  description: string;
  status: "阻断" | "需确认" | "待执行" | "未就绪" | "就绪" | "本任务不需要";
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
    not_required?: number;
    autofilled?: number;
    survey_input_required?: number;
    attention_required?: number;
    audit_findings?: number;
  };
  gates: {
    can_visualize: boolean;
    can_build_samples: boolean;
    can_train_seismic_baseline: boolean;
    can_train_multimodal: boolean;
    can_run_high_confidence_fusion?: boolean;
    can_run_prediction: boolean;
    can_run_selected_task?: boolean;
  };
  task_readiness?: {
    task_id: string | null;
    model_id?: string | null;
    required_modalities: string[];
    required_stages: string[];
    not_required_stages: string[];
    ready: boolean;
    scope_mode?: string;
    model_contract?: Record<string, unknown>;
    time_depth_policy?: PreparationTimeDepthPolicy;
    registration_entry_policy?: {
      mode?: string;
      native_relative_data_capability?: boolean;
      native_relative_candidate_well_count?: number;
      native_relative_registration_ready?: boolean;
      native_relative_well_receipts?: Array<{
        well_id?: string;
        eligible?: boolean;
        [key: string]: unknown;
      }>;
    };
  };
  survey_contract_candidate?: SurveyContractCandidateGroup;
  request_patch?: SurveyContractRequestPatch;
  runtime_contract_review?: RuntimeContractReview;
}

export interface PreparationTimeDepthPolicy {
  provided_control_required: boolean;
  provided_control_well_count: number;
  acoustic_candidate_well_count: number;
  missing_provided_control_blocks_current_task: boolean;
  model_forbids_time_depth_supervision: boolean;
  horizontal_only_never_implies_twt: boolean;
  training_requires_accepted_vertical_alignment: boolean;
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
    shape_zyx?: number[];
    shape_ics?: number[];
    source_shape_zyx?: number[];
    geometry_mode?: string;
    duplicate_trace_count?: number;
    platform_ordered_grid?: boolean;
    native_inline_count?: number | null;
    default_scope_ready?: boolean;
    formal_input_ready?: boolean;
    available_scopes?: string[];
    supported_scopes?: string[];
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
    sha256?: string;
    geometry_fingerprint?: string | null;
    asset_options_sha256?: string;
    geometry_identity?: {
      contract_version?: string;
      profile?: string;
      geometry_fingerprint?: string;
    } | null;
  }>;
  metadata_detection: Array<Record<string, unknown>>;
  errors: Array<Record<string, string>>;
  preparation: PreparationReport;
  data_snapshot?: {
    contract_version: string;
    snapshot_id: string | null;
    project_id?: string;
    state?: string;
    snapshot_sha256?: string;
    snapshot_manifest_path?: string;
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
    trajectories: Array<{
      name: string;
      geometryType?: "vertical" | "deviated" | "horizontal";
      geometryLabel?: string;
      maxInclinationDeg?: number;
      lateralDisplacementM?: number;
    }>;
    wellGeometrySummary?: {
      counts: Record<"vertical" | "deviated" | "horizontal", number>;
      labels: Record<string, string>;
      classification: string;
      verticalScale: string;
    };
    wellTimeAlignmentSummary?: {
      aligned: number;
      horizontalOnly: number;
      embedded?: number;
      depthNormalizedPreviews?: number;
      preferredSource: string;
      timeAxisDomain?: string;
      timeAxisDomains?: string[];
      provenTwt?: boolean;
      alignmentMeaning?: string;
      registrationCandidates?: number;
      policy: string;
    };
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
    vertical_datum_ready?: boolean;
    vertical_datum_verified_count?: number;
    vertical_alignment_counts?: Record<string, number>;
    output_directory: string;
    output_files: Record<string, string>;
  };
  registration?: {
    contract_version: string;
    registration_id: string;
    source_snapshot_id?: string;
    well_count: number;
    alignment_attempt_count?: number;
    registered_well_count: number;
    fusion_ready_well_count: number;
    primary_fusion_ready_well_count?: number;
    fusion_feature_well_count?: number;
    fusion_consumption_well_count?: number;
    candidate_well_count: number;
    blocked_well_count?: number;
    business_status?: "usable" | "partially_usable" | "blocked" | "experimental_human_accepted_candidate";
    candidate_status?: "candidate_not_promoted" | "human_accepted_for_experimental_downstream_use" | string;
    accepted_well_ids?: string[];
    parent_registration_task_id?: string;
    candidate_prediction_task_id?: string;
    candidate_manifest_sha256?: string;
    uncertainty_calibration_status?: string;
    alignment_runtime?: string;
    registration_source_policy?: string;
    p13_runtime_attempted?: boolean;
    p13_checkpoint_executed?: boolean;
    p13_runtime_status?: string;
    p13_eligible_well_count?: number;
    p13_executed_well_count?: number;
    p13_raw_candidate_accepted_count?: number;
    p13_rejection_reasons?: string[];
    p13_records?: Array<Record<string, unknown>>;
    p13_skipped_wells?: Array<Record<string, unknown>>;
    p13_errors?: Array<Record<string, unknown>>;
    execution_contract?: {
      mode?: string;
      time_domain?: string;
      time_axis_ready?: boolean;
      timeDomain?: string;
      timeAxisReady?: boolean;
      [key: string]: unknown;
    };
    status_counts: Record<string, number>;
    method_counts: Record<string, number>;
    can_build_multimodal_view: boolean;
    fusion_product_status?: "experimental_fusion_ready" | "fine_registration_complete_nonfusion" | string;
    active_consumption_product_role?: "fusion_consumption_product" | "physical_primary" | null | string;
    downstream_fusion_ready_well_count?: number;
    output_directory: string;
    output_files: Record<string, string>;
  };
  prepared_view?: {
    contract_version: string;
    state: "ready" | "unavailable_legacy_unpinned" | string;
    view_id: string;
    source_snapshot_id: string;
    manifest_path?: string;
    manifest_sha256?: string;
    view_sha256?: string;
    gates?: {
      registration_fusion_ready_well_ids?: string[];
      registration_product_role?: string;
      sample_count?: number;
      valid_window_count?: number;
      training_eligible_count?: number;
      coordinate_reference_verified?: boolean;
      vertical_datum_ready?: boolean;
    };
  };
  source_snapshot_id?: string;
  registration_task_id?: string;
}

export interface HorizontalRegistrationSummary {
  seismic_grid_count?: number;
  well_count?: number;
  excluded_well_count?: number;
  station_count?: number;
  covered_station_count?: number;
  fully_covered_well_count?: number;
  plan_view_usable_well_count?: number;
  business_status?: string;
}

/**
 * Horizontal registration is a read-only spatial derivative. It deliberately
 * does not carry the source snapshot's assets, canonical data, LAS previews,
 * or preparation report and therefore must never be treated as WorkflowResult.
 */
export interface HorizontalRegistrationTaskResult {
  summary: HorizontalRegistrationSummary;
  data_snapshot: {
    snapshot_id: string;
    project_id?: string;
    snapshot_contract_version?: string;
    source_snapshot_fingerprint?: string;
    integrity_status?: string;
    relationship: "read_only_derived_view" | string;
  };
  horizontal_registration: {
    contract_version: string;
    horizontal_registration_id: string;
    source_snapshot_id: string;
    product_kind: "horizontal_registration" | string;
    scientific_scope?: {
      time_depth_used?: boolean;
      vertical_registration?: string;
      registration_v3_generated?: boolean;
      fusion_ready?: boolean;
      training_eligible?: boolean;
    };
    summary?: HorizontalRegistrationSummary;
    can_build_multimodal_view: false;
    fusion_ready: false;
    training_eligible: false;
    output_directory: string;
    output_files: Record<string, string>;
    output_integrity?: Record<string, string>;
    horizontal_registration_lineage_sha256?: string;
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
  task_type?: "data_preparation" | "well_tie" | "horizontal_registration" | "sample_building" | "model_prediction" | string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "superseded";
  project_id?: string;
  snapshot_id?: string;
  parent_task_id?: string;
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  preparation_duration_seconds?: number;
  progress: number;
  message: string;
  progress_detail?: {
    phase?: "submitting" | "validating" | "cataloging" | "reading" | "hashing" | "summarizing" | "caching" | "completed" | "failed";
    work_done?: number;
    work_total?: number;
    unit?: "assets" | "bytes";
    current_item?: string | null;
    current_item_size_bytes?: number | null;
    started_at?: string;
    current_started_at?: string | null;
    subwork_done?: number;
    subwork_total?: number;
    subunit?: "traces" | "bytes" | null;
    can_estimate?: boolean;
  };
  preparation_estimate?: {
    duration_seconds: number;
    samples: number;
    confidence: "low" | "medium";
    basis: "matching_configuration_history";
  } | null;
  request?: Record<string, unknown>;
  result: WorkflowResult | HorizontalRegistrationTaskResult | PredictionTaskResult | null;
  error: { type: string; message: string } | null;
}

export interface ProjectSummary {
  project_id: string;
  name?: string;
  kind?: string;
  active_snapshot_id?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface ActiveProjectSnapshotResponse {
  contract_version: "well-seismic.active-snapshot.v1" | string;
  active_snapshot_id: string;
  project: ProjectSummary;
}

export interface SourceSnapshotSummary {
  snapshot_id: string;
  project_id: string;
  state: string;
  contract_version: string;
  hashes?: {
    source_content_sha256?: string;
    semantics_sha256?: string;
    registration_evidence_sha256?: string;
    snapshot_sha256?: string;
    [key: string]: string | undefined;
  };
  created_by_task_id?: string;
  created_at?: string;
  updated_at?: string;
  display_name?: string;
  primary_seismic_path?: string;
}

export interface SnapshotDetailResponse {
  contract_version: string;
  snapshot: SourceSnapshotSummary;
  tasks: BackgroundTask[];
  artifact_bundles: Array<Record<string, unknown>>;
}

export interface DemoPaths {
  available: boolean;
  seismic_paths: string[];
  survey_paths: string[];
  log_paths: string[];
  well_paths: string[];
  time_depth_paths?: string[];
  interpretation_paths: string[];
  auxiliary_paths: string[];
  /** Optional source-contract fields supplied by a demo catalog. */
  contract?: Partial<DataPathsPayload>;
  source_contract?: Partial<DataPathsPayload>;
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
  scientific_status?: ReleaseScientificStatus | string;
  runtime_status?: ReleaseRuntimeStatus | string;
  evidence_class?: string;
  warnings?: string[];
  metadata?: Record<string, unknown>;
}

export type ReleaseScientificStatus =
  | "validated"
  | "conditional"
  | "candidate"
  | "failed"
  | "legacy"
  | "unknown";

export type ReleaseRuntimeStatus =
  | "runnable"
  | "precomputed_only"
  | "adapter_required"
  | "blocked"
  | "unknown";

export interface ReleaseArtifact {
  id?: string;
  name?: string;
  role?: string;
  kind?: string;
  layer?: "surface" | "volume" | "well_curve" | "table" | "report" | null;
  format?: string;
  media_type?: string;
  path?: string;
  relative_path?: string;
  exists?: boolean;
  sha256?: string;
  size_bytes?: number;
  integrity_status?: string;
  unit?: string;
  axis_order?: string;
  uncertainty_definition?: string;
  download_url?: string;
  downloadable_url?: string;
  description?: string;
}

export type ReleaseArtifactValue = ReleaseArtifact | string | null;
export type ReleaseArtifactCollection =
  | ReleaseArtifactValue[]
  | Record<string, ReleaseArtifactValue>;

/**
 * A frozen model/result release.  Scientific evidence and runtime availability
 * are deliberately independent: a release can be scientifically useful while
 * only exposing immutable, precomputed artifacts.
 */
export interface ArtifactRelease {
  id: string;
  name: string;
  display_name?: string;
  family?: string;
  task?: string;
  task_id?: string;
  version?: string;
  description?: string;
  summary?: string;
  scientific_status: ReleaseScientificStatus;
  runtime_status: ReleaseRuntimeStatus;
  evidence_class: string;
  scope?: string[];
  warnings?: string[];
  limitations?: string[];
  inputs?: string[];
  outputs?: string[];
  metrics?: Record<string, string | number | boolean | null>;
  manifest_path?: string;
  checkpoint_path?: string;
  source_root?: string;
  legacy?: boolean;
  source?: string;
  release_type?: "artifact" | "model" | string;
  available?: boolean;
  artifact_count?: number;
  available_artifact_count?: number;
  model_id?: string;
  runner_id?: string | null;
  artifacts?: ReleaseArtifactCollection;
  precomputed_artifacts?: ReleaseArtifactCollection;
  metadata?: Record<string, unknown>;
}

export interface ReleaseCatalogResponse {
  schema_version?: string;
  read_only?: boolean;
  release_count?: number;
  releases: ArtifactRelease[];
  artifact_root?: string;
  summary?: {
    scientific_status?: Record<string, number>;
    runtime_status?: Record<string, number>;
    task?: Record<string, number>;
    artifact_count?: number;
    available_artifact_count?: number;
  };
  statuses?: {
    scientific?: string[];
    runtime?: string[];
  };
}

export interface SystemCacheStatus {
  contract_version: string;
  clear_endpoint: string;
  scopes: Array<Record<string, string | number | boolean>>;
  totals: {
    files: number;
    directories: number;
    bytes: number;
    memory_entries: number;
  };
  errors: Array<{ scope: string; error: string }>;
  protected: string[];
}

export interface SystemCacheClearResult {
  contract_version: string;
  scopes: Array<Record<string, string | number>>;
  files_removed: number;
  directories_removed: number;
  bytes_reclaimed: number;
  memory_entries_removed: number;
  tasks_cancelled: number;
  cancelled_tasks: Array<{
    task_id: string;
    task_type: string;
    previous_status: string;
    status: "cancelled";
  }>;
  errors: Array<{ scope: string; error: string; entry?: string }>;
  protected: string[];
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
  requires_seismic?: boolean;
  requires_registration?: boolean;
  supported_datasets?: string[];
  allowed_datasets?: string[];
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
  requires_registration?: boolean;
  prerequisite_task_ids?: string[];
  active?: boolean;
  show_in_prediction_menu?: boolean;
  status: string;
  model_runtime_statuses?: string[];
  model_scientific_statuses?: string[];
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

export type RegistrationPolicy = "none" | "optional_control" | "required";
export type PreparedViewPolicy = "none" | "optional" | "preferred" | "required";

export interface ModelDataFlowSpec {
  model_id: string;
  task_id: string;
  source_modes: string[];
  target_source_modes: string[];
  required_modalities: string[];
  optional_modalities: string[];
  registration_policy: RegistrationPolicy;
  prepared_view_policy: PreparedViewPolicy;
  prepared_view_consumed: boolean;
  accepted_domains: string[];
  output_contract: string;
  degradation_policy: string;
  adapter_registered: boolean;
  runner_registered: boolean;
  runnable: boolean;
}

export interface DataFlowContract {
  version: string;
  scope: string;
  source_snapshot: string;
  prepared_view: string;
  registration: string;
  input_attestation: string;
  policy: string;
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
  model_data_flows?: ModelDataFlowSpec[];
  data_flow_contract?: DataFlowContract;
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
  artifact_releases?: ReleaseCatalogResponse;
  persistence?: {
    contract_version: string;
    backend: string;
    history_persistent?: boolean;
    restart_safe: boolean;
    task_endpoint: string;
    artifact_release_endpoint: string;
    interrupted_task_policy?: string;
  };
  configuration_libraries: Array<{ id: string; name: string; file: string }>;
  llm: {
    enabled: boolean;
    configured: boolean;
    available: boolean;
    provider: string;
    api_mode: string;
    base_url: string;
    model: string;
    reasoning_effort?: string;
    api_key_configured: boolean;
    credential_variable?: string;
    credential_file?: string;
    credential_template_file?: string;
    credential_policy?: string;
    config_file?: string;
    key_file?: string;
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

export type ApiRequestErrorKind = "network" | "http" | "protocol" | "timeout" | "aborted";

export interface ApiRequestContext {
  method: string;
  url: string;
  timeoutMs?: number;
}

export interface ApiRequestErrorOptions {
  kind?: ApiRequestErrorKind;
  request?: ApiRequestContext;
  requestId?: string;
  responseBody?: unknown;
  cause?: unknown;
}

export class ApiRequestError extends Error {
  public readonly kind: ApiRequestErrorKind;
  public readonly request?: ApiRequestContext;
  public readonly requestId?: string;
  public readonly responseBody?: unknown;

  constructor(
    public readonly status: number,
    message: string,
    options: ApiRequestErrorOptions = {},
  ) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = "ApiRequestError";
    this.kind = options.kind ?? (status > 0 ? "http" : "network");
    this.request = options.request;
    this.requestId = options.requestId;
    this.responseBody = options.responseBody;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

function registrationPreflightFailureDetail(error: ApiRequestError): Record<string, unknown> {
  const body = error.responseBody;
  if (!isRecord(body)) return {};
  const detail = isRecord(body.detail) ? body.detail : body;
  return detail;
}

function registrationPreflightFailureCode(detail: Record<string, unknown>): string {
  const code = detail.code ?? detail.error_code ?? detail.reason_code;
  return typeof code === "string" ? code.trim().toLowerCase() : "";
}

/**
 * Classify only an explicitly declared preflight downgrade.  This is a pure
 * boundary function so callers cannot accidentally turn every HTTP 409 into a
 * horizontal-registration request.
 */
export function classifyRegistrationPreflightFailure(
  error: unknown,
): RegistrationPreflightFailureKind {
  if (!(error instanceof ApiRequestError) || error.status !== 409) return "failed";
  const detail = registrationPreflightFailureDetail(error);
  const code = registrationPreflightFailureCode(detail);
  const category = typeof detail.category === "string"
    ? detail.category.trim().toLowerCase()
    : "";
  // Integrity and semantic drift always win over a conflicting downgrade
  // flag.  A malformed response must fail closed instead of creating a
  // derivative from an untrusted snapshot.
  if (
    code === "source_snapshot_integrity_verification_failed"
    || code === "source_snapshot_semantic_drift"
    || category === "source_snapshot_integrity"
    || category === "source_snapshot_semantic_drift"
  ) {
    return "blocked_integrity";
  }
  // This boolean is the server's explicit authorization to create an
  // horizontal-only derivative.  Do not infer it from HTTP status or prose.
  if (detail.horizontal_fallback_allowed === true) {
    return "horizontal_only";
  }
  // `requires_new_snapshot` is an action hint, not an integrity diagnosis.
  // Source-quality failures (for example, an unreadable trajectory) also set
  // it, and must retain their real reason instead of being mislabeled as
  // semantic drift.
  if (category === "source_quality_unavailable" || detail.requires_new_snapshot === true) {
    return "needs_preparation";
  }
  return "failed";
}

export const DEFAULT_API_REQUEST_TIMEOUT_MS = 30_000;

interface JsonRequestInit extends RequestInit {
  /** Set to 0 to opt out for a deliberately long-running transfer. */
  timeoutMs?: number;
}

interface CombinedAbortSignal {
  signal: AbortSignal;
  abortKind: () => "timeout" | "aborted" | undefined;
  cleanup: () => void;
}

function combineAbortSignal(
  externalSignal: AbortSignal | null | undefined,
  timeoutMs: number,
): CombinedAbortSignal {
  const controller = new AbortController();
  let origin: "timeout" | "aborted" | undefined;
  let timeoutHandle: ReturnType<typeof setTimeout> | undefined;

  const abortFromCaller = () => {
    if (controller.signal.aborted) return;
    origin = "aborted";
    controller.abort(externalSignal?.reason);
  };

  if (externalSignal?.aborted) {
    abortFromCaller();
  } else if (externalSignal) {
    externalSignal.addEventListener("abort", abortFromCaller, { once: true });
  }

  if (!controller.signal.aborted && timeoutMs > 0) {
    timeoutHandle = globalThis.setTimeout(() => {
      if (controller.signal.aborted) return;
      origin = "timeout";
      controller.abort(new DOMException(`请求超过 ${timeoutMs}ms`, "TimeoutError"));
    }, timeoutMs);
  }

  return {
    signal: controller.signal,
    abortKind: () => origin,
    cleanup: () => {
      if (timeoutHandle !== undefined) globalThis.clearTimeout(timeoutHandle);
      externalSignal?.removeEventListener("abort", abortFromCaller);
    },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isHorizontalRegistrationTaskResult(
  value: unknown,
): value is HorizontalRegistrationTaskResult {
  if (!isRecord(value) || !isRecord(value.horizontal_registration)) return false;
  const horizontal = value.horizontal_registration;
  return typeof horizontal.horizontal_registration_id === "string"
    && typeof horizontal.source_snapshot_id === "string"
    && isRecord(value.data_snapshot)
    && typeof value.data_snapshot.snapshot_id === "string";
}

export function isWorkflowResult(value: unknown): value is WorkflowResult {
  return isRecord(value)
    && !isHorizontalRegistrationTaskResult(value)
    && (
      isRecord(value.preparation)
      || isRecord(value.registration)
      || isRecord(value.matching)
    );
}

function stringifySafely(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function normalizeErrorDetail(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    return value.map(normalizeErrorDetail).filter(Boolean).join("；");
  }
  if (!isRecord(value)) return String(value);

  if (value.detail !== undefined) return normalizeErrorDetail(value.detail);

  const location = Array.isArray(value.loc)
    ? value.loc.map((part) => String(part)).filter(Boolean).join(".")
    : "";
  const messageCandidate = value.msg ?? value.message ?? value.error ?? value.title ?? value.reason;
  const message = normalizeErrorDetail(messageCandidate);
  if (message) return location ? `${location}：${message}` : message;

  return stringifySafely(value);
}

function limitErrorMessage(message: string): string {
  const normalized = message.trim();
  const limit = 4_000;
  return normalized.length > limit ? `${normalized.slice(0, limit)}…` : normalized;
}

function parseErrorBody(rawText: string): unknown {
  const trimmed = rawText.trim();
  if (!trimmed) return undefined;
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return trimmed;
  }
}

function responseRequestId(response: Response): string | undefined {
  return response.headers.get("x-request-id")
    ?? response.headers.get("x-correlation-id")
    ?? undefined;
}

async function jsonRequest<T>(url: string, init: JsonRequestInit = {}): Promise<T> {
  const {
    timeoutMs = DEFAULT_API_REQUEST_TIMEOUT_MS,
    signal: externalSignal,
    ...fetchInit
  } = init;
  const request: ApiRequestContext = {
    method: (fetchInit.method || "GET").toUpperCase(),
    url,
    timeoutMs: timeoutMs > 0 ? timeoutMs : undefined,
  };
  const combinedSignal = combineAbortSignal(externalSignal, timeoutMs);

  try {
    const response = await fetch(url, { ...fetchInit, signal: combinedSignal.signal });
    const requestId = responseRequestId(response);
    const rawText = await response.text();

    if (!response.ok) {
      const body = parseErrorBody(rawText);
      const detail = limitErrorMessage(normalizeErrorDetail(body));
      throw new ApiRequestError(
        response.status,
        detail || response.statusText || `请求失败：${response.status}`,
        {
          kind: "http",
          request,
          requestId,
          responseBody: body,
        },
      );
    }

    const trimmedBody = rawText.trim();
    if (!trimmedBody) {
      throw new ApiRequestError(response.status, "服务返回了空响应，无法读取 JSON", {
        kind: "protocol",
        request,
        requestId,
      });
    }

    try {
      return JSON.parse(trimmedBody) as T;
    } catch (error) {
      throw new ApiRequestError(response.status, "服务返回了无法解析的 JSON 响应", {
        kind: "protocol",
        request,
        requestId,
        responseBody: limitErrorMessage(trimmedBody),
        cause: error,
      });
    }
  } catch (error) {
    if (error instanceof ApiRequestError) throw error;

    const abortKind = combinedSignal.abortKind();
    if (abortKind === "timeout") {
      throw new ApiRequestError(0, `请求超时（${timeoutMs}ms）`, {
        kind: "timeout",
        request,
        cause: error,
      });
    }
    if (abortKind === "aborted" || (error instanceof DOMException && error.name === "AbortError")) {
      throw new ApiRequestError(0, "请求已取消", {
        kind: "aborted",
        request,
        cause: error,
      });
    }
    throw new ApiRequestError(0, "无法连接服务，请检查网络或后端状态", {
      kind: "network",
      request,
      cause: error,
    });
  } finally {
    combinedSignal.cleanup();
  }
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

export interface RegistrationCandidateAcceptancePayload {
  accepted_well_ids: string[];
  expected_candidate_manifest_sha256: string;
  confirmation: "ACCEPT_GEOPATH_CANDIDATE";
  review_note?: string;
}

export function acceptRegistrationCandidate(
  predictionTaskId: string,
  payload: RegistrationCandidateAcceptancePayload,
): Promise<{ task_id: string; status: string; message: string }> {
  return jsonRequest(
    `/api/v1/registration/candidates/${encodeURIComponent(predictionTaskId)}/accept`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export interface RawWellUpload {
  name: string;
  path: string;
  size: number;
  sha256: string;
}

export function uploadRawWellFile(file: File): Promise<RawWellUpload> {
  return jsonRequest(
    `/api/v1/raw-well-files?filename=${encodeURIComponent(file.name)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    },
  );
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

export function getProjects(): Promise<{ contract_version: string; projects: ProjectSummary[] }> {
  return jsonRequest("/api/v1/projects");
}

export function setActiveProjectSnapshot(
  projectId: string,
  snapshotId: string,
): Promise<ActiveProjectSnapshotResponse> {
  return jsonRequest(`/api/v1/projects/${encodeURIComponent(projectId)}/active-snapshot`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ snapshot_id: snapshotId }),
  });
}

export function getProjectSnapshots(projectId: string): Promise<{
  contract_version: string;
  project: ProjectSummary;
  snapshots: SourceSnapshotSummary[];
}> {
  return jsonRequest(`/api/v1/projects/${encodeURIComponent(projectId)}/snapshots`);
}

export function getSnapshotDetail(snapshotId: string): Promise<SnapshotDetailResponse> {
  return jsonRequest(`/api/v1/snapshots/${encodeURIComponent(snapshotId)}`);
}

export function getReleaseCatalog(): Promise<ReleaseCatalogResponse> {
  return jsonRequest("/api/v1/releases");
}

export function getSystemCache(): Promise<SystemCacheStatus> {
  return jsonRequest("/api/v1/system/cache");
}

export function clearSystemCache(): Promise<SystemCacheClearResult> {
  return jsonRequest("/api/v1/system/cache/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmation: "CLEAR_REGENERABLE_CACHE" }),
  });
}

export function createDataPreparation(payload: DataPathsPayload): Promise<{ task_id: string }> {
  return postTask("/api/v1/data-preparation/tasks", payload);
}

export function createRegistration(payload: SampleBuildingPayload): Promise<{ task_id: string }> {
  return postTask("/api/v1/registration/tasks", payload);
}

export function preflightRegistration(sourceSnapshotId: string): Promise<RegistrationPreflightResponse> {
  return jsonRequest("/api/v1/registration/preflight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_snapshot_id: sourceSnapshotId }),
  });
}

export function confirmRuntimeContract(
  sourceSnapshotId: string,
  values: Record<string, RuntimeContractValue>,
  attestation: SurveyAttestationPayload,
): Promise<RuntimeContractConfirmationResponse> {
  return jsonRequest("/api/v1/registration/runtime-contract", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_snapshot_id: sourceSnapshotId,
      values,
      confirmation: "CONFIRM_RUNTIME_CONTRACT",
      attestation,
    }),
  });
}

export function createHorizontalRegistration(
  payload: HorizontalRegistrationPayload,
): Promise<{ task_id: string }> {
  return jsonRequest("/api/v1/horizontal-registration/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
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

export function applyRecommendedPreparationIssues(taskId: string): Promise<{
  preparation: PreparationReport;
  batch: { batch_id: string; applied_issue_ids: string[]; skipped_blocking_issue_ids: string[] };
  applied_count: number;
  skipped_blocking_count: number;
}> {
  return jsonRequest(`/api/v1/tasks/${encodeURIComponent(taskId)}/issues/batch-actions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "apply_recommended" }),
  });
}

export function autofillPreparationIssues(taskId: string): Promise<{
  preparation: PreparationReport;
  autofilled_count: number;
  survey_input_required_count: number;
}> {
  return jsonRequest(`/api/v1/tasks/${encodeURIComponent(taskId)}/issues/llm-autofill`, {
    method: "POST",
  });
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
