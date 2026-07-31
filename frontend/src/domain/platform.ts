import type {
  PredictionTaskCapability,
  WellLogPreview,
  WellLogPreviewCurve,
} from "../api";

export type PathGroupKey = "seismic" | "logs" | "wells" | "auxiliary";
export type PreparationScreen = "input" | "pipeline";
export type PredictionTaskKey = string;
export type ViewKey =
  | "overview"
  | "preparation"
  | "visualization"
  | "samples"
  | "models"
  | "prediction"
  | "evaluation"
  | "settings";

export interface PathGroup {
  key: PathGroupKey;
  title: string;
  hint: string;
  optional?: boolean;
  paths: string[];
}

interface NavigationItem {
  id: ViewKey;
  label: string;
  section: "project" | "workflow" | "system";
}

interface ConventionalCurveSlot {
  curveId: string;
  displayId: string;
  name: string;
  color: string;
}

interface ConventionalCurveGroup {
  id: "lithology" | "porosity" | "resistivity";
  order: string;
  title: string;
  subtitle: string;
  slots: ConventionalCurveSlot[];
}

export const navigation: NavigationItem[] = [
  { id: "preparation", label: "数据准备", section: "workflow" },
  { id: "visualization", label: "可视化", section: "workflow" },
  { id: "prediction", label: "预测解释", section: "workflow" },
  { id: "evaluation", label: "评估导出", section: "workflow" },
  { id: "models", label: "模型中心", section: "system" },
  { id: "settings", label: "配置中心", section: "system" },
];

export const navigationIconPaths: Record<ViewKey, string[]> = {
  overview: ["M4 4h6v6H4z", "M14 4h6v6h-6z", "M4 14h6v6H4z", "M14 14h6v6h-6z"],
  preparation: ["M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Z", "M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6", "M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"],
  visualization: ["m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z", "m4.5 7.8 7.5 4.3 7.5-4.3", "M12 12v9"],
  samples: ["M6 6h.01", "M18 6h.01", "M12 18h.01", "M6.5 6.5 11 0", "m7 7 4 9", "m17 7-4 9"],
  models: ["M9 3v3", "M15 3v3", "M9 18v3", "M15 18v3", "M3 9h3", "M3 15h3", "M18 9h3", "M18 15h3", "M7 6h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z"],
  prediction: ["M4 19V5", "M4 19h16", "m7 15 4-5 3 3 5-7"],
  evaluation: ["M7 3h8l4 4v14H7z", "M15 3v5h5", "m10 14 2 2 4-5"],
  settings: ["M4 7h10", "M18 7h2", "M4 17h2", "M10 17h10", "M14 4v6", "M7 14v6"],
};

export const conventionalCurveGroups: ConventionalCurveGroup[] = [
  {
    id: "lithology",
    order: "一",
    title: "岩性分层曲线",
    subtitle: "识别泥质、层界与井眼条件",
    slots: [
      { curveId: "SP", displayId: "SP", name: "自然电位", color: "#174ea6" },
      { curveId: "GR", displayId: "GR", name: "自然伽马", color: "#198754" },
      { curveId: "CAL", displayId: "CAL", name: "井径", color: "#c23b33" },
    ],
  },
  {
    id: "porosity",
    order: "二",
    title: "三孔隙度曲线",
    subtitle: "联合表征声波、中子与密度响应",
    slots: [
      { curveId: "DT", displayId: "AC", name: "声波时差", color: "#22272e" },
      { curveId: "NPHI", displayId: "CNL", name: "补偿中子", color: "#1d5fbf" },
      { curveId: "RHOB", displayId: "DEN", name: "补偿密度", color: "#d33f35" },
    ],
  },
  {
    id: "resistivity",
    order: "三",
    title: "电阻率曲线",
    subtitle: "对数尺度对比冲洗带与深浅电阻率",
    slots: [
      { curveId: "MSFL", displayId: "MSFL", name: "微球聚焦", color: "#7b542c" },
      { curveId: "RS", displayId: "LLS", name: "浅侧向", color: "#2458a6" },
      { curveId: "RT", displayId: "LLD", name: "深侧向", color: "#be3028" },
    ],
  },
];

const curveSlotAliases: Record<string, string[]> = {
  SP: ["SP", "SSP", "SPONT"],
  GR: ["GR", "GRC", "GAM", "GAMMA", "CGR", "SGR", "HGR"],
  CAL: ["CAL", "CALI", "HCAL", "CALD", "CALX", "CALY", "C1", "C2"],
  DT: ["DT", "DTC", "AC", "P_AC", "SONIC", "DTCO", "DTP", "DT4P", "DT24"],
  NPHI: ["NPHI", "CNL", "TNPH", "CNC", "NPH", "NPOR", "NEUT"],
  RHOB: ["RHOB", "DEN", "DENS", "ZDEN", "RHOZ", "BD", "FDC"],
  MSFL: ["MSFL", "RXO", "RXOZ", "SFL", "SFLU", "MCFL"],
  RS: ["RS", "LLS", "ILM", "RMED", "AT30", "AHT30", "HRLA3", "RSHAL", "RESS"],
  RT: ["RT", "LLD", "ILD", "RDEP", "HDRS", "AT90", "AHT90", "HRLA5", "RD", "RESD"],
};

function normalizeCurveMnemonic(value: string): string {
  return value.trim().toUpperCase().replace(/[\s.\-]+/g, "_").replace(/_\d+$/, "");
}

function curveMatchesSlot(curveId: string, slotId: string): boolean {
  const normalized = normalizeCurveMnemonic(curveId);
  return (curveSlotAliases[slotId] || [slotId]).some(
    (alias) => normalizeCurveMnemonic(alias) === normalized,
  );
}

export function availableCurveForSlot(
  log: WellLogPreview | null,
  slotId: string,
): WellLogPreviewCurve | null {
  if (!log) return null;
  return log.curves.find((curve) => curve.validCount > 0 && curveMatchesSlot(curve.id, slotId)) || null;
}

export function conventionalCoverage(log: WellLogPreview): string {
  const count = conventionalCurveGroups.reduce(
    (total, group) => total + group.slots.filter((slot) => availableCurveForSlot(log, slot.curveId)).length,
    0,
  );
  return `${count}/9`;
}

export const viewMeta: Record<ViewKey, { eyebrow: string; title: string; description: string }> = {
  overview: {
    eyebrow: "地层慧眼 · 项目工作台",
    title: "油气甜点智能识别的地震—测井多模态统一表征大模型",
    description: "统一组织地震、测井和井轨迹表征，为油气甜点、有利储层与有利目标识别提供可解释的数据和模型基座。",
  },
  preparation: {
    eyebrow: "数据阶段 01",
    title: "数据准备与预处理",
    description: "在一个模块内完成路径登记、曲线清洗、井实体合并、轨迹对齐和地震几何检查。",
  },
  visualization: {
    eyebrow: "数据阶段 02",
    title: "CIGVis 地震解释工作台",
    description: "以 CIGVis 统一呈现三维地震、二维测线与井轨迹，测井曲线保留独立精细工作台。",
  },
  samples: {
    eyebrow: "数据阶段 03",
    title: "井震空间对齐与样本构建",
    description: "独立运行可替换的空间对齐器，输出带掩码、置信度和来源记录的多模态样本。",
  },
  models: {
    eyebrow: "模型阶段 04",
    title: "模型与井震融合中心",
    description: "统一管理地震模型、测井编码、对齐模块、融合策略和下游解释运行器。",
  },
  prediction: {
    eyebrow: "解释阶段 05",
    title: "下游预测与地质解释",
    description: "按任务契约接入断层、地层分割、层位、沉积相、裂缝、储层和有利目标模型。",
  },
  evaluation: {
    eyebrow: "交付阶段 06",
    title: "评估、对比与结果导出",
    description: "集中管理样本、模型、预测结果、评价指标、版本与可追溯输出。",
  },
  settings: {
    eyebrow: "系统能力",
    title: "知识映射、算法与智能判断配置",
    description: "曲线、单位、井字段、SEG-Y版本、清洗、对齐、融合和LLM兜底策略均可独立维护。",
  },
};

export function createDefaultPathGroups(): PathGroup[] {
  return [
    { key: "seismic", title: "地震数据", hint: "SEG-Y文件或包含二维/三维地震数据的一个或多个目录", paths: [""] },
    { key: "logs", title: "测井数据", hint: "LAS文件或目录；支持不同版本、曲线命名和单位", paths: [""] },
    { key: "wells", title: "井位、海拔与井轨迹", hint: "可分文件、合并表、每井单文件或多井总表", optional: true, paths: [""] },
    { key: "auxiliary", title: "其他辅助数据", hint: "标签、解释成果或说明文件；不参与基础匹配", optional: true, paths: [] },
  ];
}

export const fallbackInterpretationTasks: PredictionTaskCapability[] = [
  {
    id: "fault",
    name: "断层识别",
    short_name: "断层分割",
    description: "识别断层概率体、断层面与二值分割结果。",
    outputs: ["断层概率体", "断层分割体"],
    output: "断层概率体 / 断层分割体",
    required_modalities: ["三维地震"],
    evaluation_metrics: ["Dice", "IoU", "连通性"],
    order: 10,
    contract_version: "1.0",
    model_id: "faultseg_3d",
    model_ids: ["faultseg_3d"],
    runnable_model_ids: ["faultseg_3d"],
    available: true,
    status: "可运行",
  },
  {
    id: "strata",
    name: "地层分割",
    short_name: "地层分割",
    description: "逐 Inline 识别有序地层实例，输出标签体、置信度体和彩色剖面。",
    outputs: ["地层实例标签体", "分割置信度体", "彩色剖面"],
    output: "地层实例标签体 / 分割置信度体 / 彩色剖面",
    required_modalities: ["规则三维后叠加地震"],
    evaluation_metrics: ["mIoU", "边界误差", "跨线连续性"],
    order: 20,
    contract_version: "1.0",
    model_id: "seismic_surface_seg",
    model_ids: ["seismic_surface_seg"],
    runnable_model_ids: ["seismic_surface_seg"],
    available: true,
    status: "可运行",
  },
  ...[
    ["horizon", "层位追踪", "层位面与拾取置信度"],
    ["facies", "沉积相预测", "沉积相分类与概率体"],
    ["fracture", "裂缝识别", "裂缝概率体与优势方位"],
    ["reservoir", "有利储层", "储层概率与品质表征"],
    ["target", "有利目标", "目标区概率与候选连通体"],
  ].map(([id, name, description], index) => ({
    id,
    name,
    short_name: name,
    description,
    outputs: [description],
    output: description,
    required_modalities: [],
    evaluation_metrics: [],
    order: 30 + index * 10,
    contract_version: "1.0",
    model_ids: [],
    runnable_model_ids: [],
    available: false,
    status: "等待模型插件" as const,
  })),
];
