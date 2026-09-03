export const LAYER_PULSE_VIEW_ID = "layerpulse" as const;
export const LAYER_PULSE_TASK_ID = "layerpulse" as const;
export const LAYER_PULSE_MODEL_ID = "layerpulse_geochronograph_f3x200cf" as const;
export const layerPulseModelContract = {
  parameterCount: 174_697_519,
  fFinalChannels: 96,
  headCount: 11,
  classificationHeadCount: 6,
  headInput: "shared_F_final_only",
  oneForwardReturnsAllTasks: true,
  timeDepthRequiredAtForward: false,
} as const;

export type LayerPulseSupportStatus = "ready" | "degraded" | "blocked";
export type LayerPulseTaskStatus =
  | "idle"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";
export type LayerPulseOutputKind = "classification" | "regression";
export type LayerPulseOutputGroup =
  | "structure"
  | "stratigraphy"
  | "deposition"
  | "property"
  | "well_and_reasoning";

export interface LayerPulseSupportCheck {
  id: string;
  label: string;
  status: LayerPulseSupportStatus;
  detail: string;
  required: boolean;
  evidence?: string;
}

export interface LayerPulseSupportReceipt {
  contract_version: "layerpulse.platform-support.v1" | string;
  snapshot_id: string;
  model_id: string;
  status: LayerPulseSupportStatus;
  checks: LayerPulseSupportCheck[];
  warnings: string[];
  dataset_label?: string;
  evaluated_at?: string;
}

export interface LayerPulseSupportSummary {
  status: LayerPulseSupportStatus;
  label: string;
  detail: string;
  readyCount: number;
  degradedCount: number;
  blockedCount: number;
}

export interface LayerPulseTaskState {
  status: LayerPulseTaskStatus;
  taskId: string;
  progress: number;
  message: string;
  error?: string;
  availableOutputKeys?: readonly string[];
}

export interface LayerPulseLegendItem {
  index: number;
  id: string;
  label: string;
  color: string;
  background: boolean;
}

export interface LayerPulseContinuousLegendStop {
  position: number;
  label: string;
  color: string;
}

export interface LayerPulseOutputDefinition {
  key: string;
  name: string;
  shortName: string;
  group: LayerPulseOutputGroup;
  kind: LayerPulseOutputKind;
  channels: number;
  description: string;
  classes: readonly LayerPulseLegendItem[];
  continuousLegend: readonly LayerPulseContinuousLegendStop[];
  backgroundIndex: number | null;
  decode: "direct_argmax_dim1" | null;
  usesThreshold: false;
}

const background = (color = "#d8e0e8"): LayerPulseLegendItem => ({
  index: 0,
  id: "background",
  label: "背景",
  color,
  background: true,
});

function classificationOutput(
  definition: Omit<LayerPulseOutputDefinition, "kind" | "channels" | "backgroundIndex" | "decode" | "usesThreshold" | "continuousLegend"> & {
    classes: readonly LayerPulseLegendItem[];
  },
): LayerPulseOutputDefinition {
  return {
    ...definition,
    kind: "classification",
    channels: definition.classes.length,
    continuousLegend: [],
    backgroundIndex: 0,
    decode: "direct_argmax_dim1",
    usesThreshold: false,
  };
}

function regressionOutput(
  definition: Omit<LayerPulseOutputDefinition, "kind" | "channels" | "classes" | "backgroundIndex" | "decode" | "usesThreshold">,
): LayerPulseOutputDefinition {
  return {
    ...definition,
    kind: "regression",
    channels: 1,
    classes: [],
    backgroundIndex: null,
    decode: null,
    usesThreshold: false,
  };
}

/**
 * Canonical eleven-output contract copied from the selected single-checkpoint
 * delivery. Classification entries always expose background-inclusive logits
 * and use direct argmax along the class dimension.
 */
export const layerPulseOutputCatalog: readonly LayerPulseOutputDefinition[] = [
  classificationOutput({
    key: "fault_logits",
    name: "断层识别",
    shortName: "断层",
    group: "structure",
    description: "背景与断层的完整二类 logits。",
    classes: [
      background(),
      { index: 1, id: "fault", label: "断层", color: "#ed5a4f", background: false },
    ],
  }),
  classificationOutput({
    key: "unconformity_logits",
    name: "不整合识别",
    shortName: "不整合",
    group: "stratigraphy",
    description: "不整合与侵蚀势垒的完整二类 logits。",
    classes: [
      background(),
      { index: 1, id: "unconformity", label: "不整合", color: "#f2a93b", background: false },
    ],
  }),
  classificationOutput({
    key: "facies_logits",
    name: "F3 六类地震相",
    shortName: "地震相",
    group: "deposition",
    description: "背景加六类 F3 地层相的完整七类 logits。",
    classes: [
      background("#d9e0e6"),
      { index: 1, id: "upper_ns", label: "Upper NS", color: "#78b7e5", background: false },
      { index: 2, id: "middle_ns", label: "Middle NS", color: "#53a891", background: false },
      { index: 3, id: "lower_ns", label: "Lower NS", color: "#91bd54", background: false },
      { index: 4, id: "rijnland_chalk", label: "Rijnland / Chalk", color: "#e0c34f", background: false },
      { index: 5, id: "scruff", label: "Scruff", color: "#dd8845", background: false },
      { index: 6, id: "zechstein", label: "Zechstein", color: "#9c72c7", background: false },
    ],
  }),
  classificationOutput({
    key: "channel_logits",
    name: "河道内部单元",
    shortName: "河道",
    group: "deposition",
    description: "背景与四类河道单元的完整五类 logits。",
    classes: [
      background(),
      { index: 1, id: "channel_1", label: "Channel 1", color: "#54b7c8", background: false },
      { index: 2, id: "channel_2", label: "Channel 2", color: "#347fc4", background: false },
      { index: 3, id: "channel_3", label: "Channel 3", color: "#745eb8", background: false },
      { index: 4, id: "channel_4", label: "Channel 4", color: "#c45a8d", background: false },
    ],
  }),
  classificationOutput({
    key: "karst_logits",
    name: "岩溶识别",
    shortName: "岩溶",
    group: "deposition",
    description: "背景与岩溶的完整二类 logits。",
    classes: [
      background(),
      { index: 1, id: "karst", label: "岩溶", color: "#a96c42", background: false },
    ],
  }),
  regressionOutput({
    key: "rgt",
    name: "相对地质时间",
    shortName: "RGT",
    group: "stratigraphy",
    description: "多频相位约束的连续相对地质时间场。",
    continuousLegend: [
      { position: 0, label: "早", color: "#234b88" },
      { position: 0.5, label: "中", color: "#51b3a2" },
      { position: 1, label: "晚", color: "#f0d35e" },
    ],
  }),
  regressionOutput({
    key: "impedance",
    name: "阻抗预测",
    shortName: "阻抗",
    group: "property",
    description: "与输入尺度合同一致的连续阻抗预测体。",
    continuousLegend: [
      { position: 0, label: "低", color: "#2c69b0" },
      { position: 0.5, label: "中", color: "#f4f4e8" },
      { position: 1, label: "高", color: "#b74343" },
    ],
  }),
  regressionOutput({
    key: "porosity",
    name: "孔隙度预测",
    shortName: "孔隙度",
    group: "property",
    description: "共享 Backbone 输出的连续孔隙度属性体。",
    continuousLegend: [
      { position: 0, label: "低", color: "#403a78" },
      { position: 0.5, label: "中", color: "#4aa8a1" },
      { position: 1, label: "高", color: "#e5d95b" },
    ],
  }),
  regressionOutput({
    key: "well_match",
    name: "无时深井震匹配",
    shortName: "井震匹配",
    group: "well_and_reasoning",
    description: "以 MD 与井轨迹为输入的连续井震匹配场，时深表不是 forward 必需输入。",
    continuousLegend: [
      { position: 0, label: "低匹配", color: "#53667a" },
      { position: 1, label: "高匹配", color: "#16a17b" },
    ],
  }),
  classificationOutput({
    key: "connectivity_logits",
    name: "构造连通性",
    shortName: "连通性",
    group: "structure",
    description: "背景与构造连通域的完整二类 logits。",
    classes: [
      background(),
      { index: 1, id: "connected", label: "连通", color: "#2d9f74", background: false },
    ],
  }),
  regressionOutput({
    key: "uncertainty",
    name: "局部不确定性",
    shortName: "不确定性",
    group: "well_and_reasoning",
    description: "传播与跨区块解释使用的连续局部不确定性场。",
    continuousLegend: [
      { position: 0, label: "低", color: "#2d9f74" },
      { position: 0.5, label: "中", color: "#e5bb4f" },
      { position: 1, label: "高", color: "#d7524b" },
    ],
  }),
] as const;

export const layerPulseOutputGroupLabels: Readonly<Record<LayerPulseOutputGroup, string>> = {
  structure: "构造",
  stratigraphy: "地层",
  deposition: "沉积地质体",
  property: "属性储层",
  well_and_reasoning: "井震与地质推理",
};

export const layerPulseSupportStatusLabels: Readonly<Record<LayerPulseSupportStatus, string>> = {
  ready: "完整支持",
  degraded: "降级支持",
  blocked: "当前阻断",
};

export const layerPulseTaskStatusLabels: Readonly<Record<LayerPulseTaskStatus, string>> = {
  idle: "等待运行",
  queued: "等待 GPU",
  running: "统一推理中",
  completed: "全部输出已返回",
  failed: "推理失败",
  cancelled: "任务已取消",
};

export function createIdleLayerPulseTaskState(): LayerPulseTaskState {
  return {
    status: "idle",
    taskId: "",
    progress: 0,
    message: "等待运行 LayerPulse 智能解释",
    availableOutputKeys: [],
  };
}

export function isLayerPulseTaskActive(status: LayerPulseTaskStatus): boolean {
  return status === "queued" || status === "running";
}

export function layerPulseOutputByKey(key: string): LayerPulseOutputDefinition {
  return layerPulseOutputCatalog.find((output) => output.key === key) || layerPulseOutputCatalog[0];
}

export function summarizeLayerPulseSupport(
  receipt: LayerPulseSupportReceipt | null | undefined,
): LayerPulseSupportSummary {
  if (!receipt) {
    return {
      status: "blocked",
      label: "等待支持核验",
      detail: "尚未收到与当前 SourceSnapshot 绑定的支持能力收据。",
      readyCount: 0,
      degradedCount: 0,
      blockedCount: 0,
    };
  }

  const readyCount = receipt.checks.filter((check) => check.status === "ready").length;
  const degradedCount = receipt.checks.filter((check) => check.status === "degraded").length;
  const blockedCount = receipt.checks.filter((check) => check.status === "blocked").length;
  const requiredBlocked = receipt.checks.some((check) => check.required && check.status === "blocked");
  const status: LayerPulseSupportStatus = requiredBlocked || receipt.status === "blocked"
    ? "blocked"
    : degradedCount > 0 || blockedCount > 0 || receipt.status === "degraded"
      ? "degraded"
      : "ready";

  return {
    status,
    label: layerPulseSupportStatusLabels[status],
    detail: `${readyCount} 项就绪 · ${degradedCount} 项降级 · ${blockedCount} 项阻断`,
    readyCount,
    degradedCount,
    blockedCount,
  };
}
