import type {
  PredictionTaskCapability,
  WellLogPreview,
  WellLogPreviewCurve,
} from "../api";

export type PathGroupKey = "seismic" | "survey" | "logs" | "wells" | "timeDepth" | "interpretations" | "auxiliary";
export type PreparationScreen = "input" | "pipeline" | "fusion";
export type PredictionTaskKey = string;
export type ViewKey =
  | "overview"
  | "preparation"
  | "visualization"
  | "samples"
  | "models"
  | "prediction"
  | "layerpulse"
  | "evaluation"
  | "settings"
  | "assistant";

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
  { id: "preparation", label: "数据与融合", section: "workflow" },
  { id: "layerpulse", label: "LayerPulse 多模态融合基础模型", section: "workflow" },
  { id: "prediction", label: "单任务推理模型（共享井震融合基座）", section: "workflow" },
  { id: "models", label: "模型中心", section: "system" },
  { id: "settings", label: "平台设置", section: "system" },
  { id: "assistant", label: "慧眼AI", section: "system" },
];

export const navigationIconPaths: Record<ViewKey, string[]> = {
  overview: ["M4 4h6v6H4z", "M14 4h6v6h-6z", "M4 14h6v6H4z", "M14 14h6v6h-6z"],
  preparation: ["M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Z", "M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6", "M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"],
  visualization: ["m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z", "m4.5 7.8 7.5 4.3 7.5-4.3", "M12 12v9"],
  samples: ["M6 6h.01", "M18 6h.01", "M12 18h.01", "M6.5 6.5 11 0", "m7 7 4 9", "m17 7-4 9"],
  models: ["M9 3v3", "M15 3v3", "M9 18v3", "M15 18v3", "M3 9h3", "M3 15h3", "M18 9h3", "M18 15h3", "M7 6h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z"],
  prediction: ["M4 19V5", "M4 19h16", "m7 15 4-5 3 3 5-7"],
  layerpulse: ["M4 15c3-5 5-7 8-7s5 2 8 7", "M4 19c3-5 5-7 8-7s5 2 8 7", "M12 4v16", "M8 5.5 12 0"],
  evaluation: ["M7 3h8l4 4v14H7z", "M15 3v5h5", "m10 14 2 2 4-5"],
  settings: ["M4 7h10", "M18 7h2", "M4 17h2", "M10 17h10", "M14 4v6", "M7 14v6"],
  assistant: ["M3 12s3.4-6 9-6 9 6 9 6-3.4 6-9 6-9-6-9-6Z", "M9 12a3 3 0 1 0 6 0 3 3 0 0 0-6 0Z", "M5 12h3l1.2-2.5 1.6 5 1.7-7 1.5 5.5 1.2-2H19"],
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
    eyebrow: "地层慧眼",
    title: "井震一体化智能解释平台",
    description: "从数据整理、井震融合到预测解释，在一个本地工作区完成。",
  },
  preparation: {
    eyebrow: "数据与融合",
    title: "数据准备与井震精细标定",
    description: "登记工区数据，完成校验、标定与融合视图。",
  },
  visualization: {
    eyebrow: "单任务推理模型（共享井震融合基座）",
    title: "单任务推理模型（共享井震融合基座）",
    description: "旧可视化入口已并入单任务推理工作台。",
  },
  samples: {
    eyebrow: "数据与融合",
    title: "数据准备与井震精细标定",
    description: "旧井震标定入口已并入数据工作台。",
  },
  models: {
    eyebrow: "LayerPulse 模型中心",
    title: "参数与架构图谱",
    description: "查看单 checkpoint 多任务基础模型的参数规模、共享表征与 11 个输出头。",
  },
  prediction: {
    eyebrow: "单任务推理模型（共享井震融合基座）",
    title: "单任务推理模型（共享井震融合基座）",
    description: "从二级目录选择任务，左侧配置推理，右侧联动查看数据与结果。",
  },
  layerpulse: {
    eyebrow: "LayerPulse 智能解释",
    title: "LayerPulse 多模态融合基础模型",
    description: "以相对地质时间为坐标、以构造连通域为传播拓扑、以井曲线为高分辨率语义锚点。",
  },
  evaluation: {
    eyebrow: "成果交付",
    title: "验收与导出",
    description: "核对结果状态并下载可追溯成果。",
  },
  settings: {
    eyebrow: "平台设置",
    title: "能力与配置",
    description: "管理本地算法、知识映射与缓存。",
  },
  assistant: {
    eyebrow: "慧眼AI",
    title: "井震智能研判助手",
    description: "围绕当前工作区、任务状态和质量证据进行连续对话。",
  },
};

export function createDefaultPathGroups(): PathGroup[] {
  return [
    { key: "seismic", title: "地震数据", hint: "SEG-Y 文件或目录；大文件保持原位", paths: [""] },
    { key: "survey", title: "测区坐标", hint: "坐标和 Inline/Crossline 映射", optional: true, paths: [] },
    { key: "logs", title: "测井数据", hint: "LAS 文件或目录", paths: [""] },
    { key: "wells", title: "井位与轨迹", hint: "井口、海拔与 DEV 轨迹", optional: true, paths: [""] },
    { key: "timeDepth", title: "时深 / Checkshot / VSP", hint: "可选；提供后优先作为井震标定控制", optional: true, paths: [] },
    { key: "interpretations", title: "解释成果", hint: "层位、断层、岩性或沉积相", optional: true, paths: [] },
    { key: "auxiliary", title: "辅助文件", hint: "测区坐标等；合同、配置或索引", optional: true, paths: [] },
  ];
}

export const primaryPredictionTaskIds = [
  "fault",
  "horizon",
  "well_property",
  "fluid_interpretation",
  "facies_1d",
  "facies_3d",
  "fracture_development",
] as const;

export function isPrimaryPredictionTaskId(
  id: string,
): id is (typeof primaryPredictionTaskIds)[number] {
  return (primaryPredictionTaskIds as readonly string[]).includes(id);
}

const primaryPredictionTaskLabels: Record<(typeof primaryPredictionTaskIds)[number], string> = {
  fault: "断层识别",
  horizon: "层位识别",
  well_property: "储层物性预测",
  fluid_interpretation: "流体解释",
  facies_1d: "一维地震相分类",
  facies_3d: "三维地震相分割",
  fracture_development: "井侧裂缝发育排序",
};

const wellPropertyCompletionModelIds = new Set([
  "wellfuse_den_p18",
  "wellfuse_por_p18",
  "wellfuse_log_perm_p18",
  "wellfuse_sw_p18",
  "wellfuse_vsh_p18",
]);

/**
 * The public reservoir-property runner exposes whole-well prediction models.
 * Older curve-completion releases remain available for archived-result
 * reproducibility, but must not reappear in the task's new-run selector.
 */
export function isWellPropertyCompletionModelId(modelId: string | undefined): boolean {
  return Boolean(modelId && wellPropertyCompletionModelIds.has(modelId));
}

function predictionOnlyWellPropertyTask(
  task: PredictionTaskCapability,
): PredictionTaskCapability {
  const modelIds = task.model_ids.filter((modelId) => !isWellPropertyCompletionModelId(modelId));
  const runnableModelIds = task.runnable_model_ids.filter(
    (modelId) => !isWellPropertyCompletionModelId(modelId),
  );
  return {
    ...task,
    model_ids: modelIds,
    runnable_model_ids: runnableModelIds,
    model_id: isWellPropertyCompletionModelId(task.model_id)
      ? runnableModelIds[0]
      : task.model_id,
    available: runnableModelIds.length > 0,
  };
}

const horizonSegmentationPresentation = {
  description: "对 SEG-Y 地震剖面进行逐像素地层实例分割，输出地层标签体、置信度和标准 SEG-Y 成果。",
  outputs: ["地层实例标签体", "分割置信度体", "标签 SEG-Y"],
  output: "地层实例标签体 / 分割置信度体 / 标签 SEG-Y",
  evaluation_metrics: ["mIoU", "Macro-F1", "边界连续性"],
};

const horizonSegmentationRuntime = {
  model_id: "seismic_surface_seg",
  model_ids: ["seismic_surface_seg"],
  runnable_model_ids: ["seismic_surface_seg"],
  available: true,
  status: "可运行",
};

export function primaryPredictionTasks(
  tasks: PredictionTaskCapability[],
): PredictionTaskCapability[] {
  const source = new Map(tasks.map((task) => [task.id, task]));
  return primaryPredictionTaskIds.flatMap((id) => {
    const task = source.get(id);
    if (!task || task.active === false) return [];
    const label = primaryPredictionTaskLabels[id];
    const publicTask = id === "well_property"
      ? predictionOnlyWellPropertyTask(task)
      : task;
    return [{
      ...publicTask,
      ...(id === "horizon" ? {
        ...horizonSegmentationPresentation,
        ...horizonSegmentationRuntime,
      } : {}),
      name: label,
      short_name: label,
    }];
  });
}

export const fallbackInterpretationTasks: PredictionTaskCapability[] = [
  {
    id: "fault",
    name: "断层识别",
    short_name: "断层识别",
    description: "默认预测工区三轴中心的单个 128³ 完整块；也可选择耗时较长的全区重叠滑窗连续重建。",
    outputs: ["中心单块预测", "可选全区断层概率体", "确定性断层掩码与空间切片"],
    output: "中心单块结果 / 可选全区概率体 / 确定性掩码",
    required_modalities: ["三维地震"],
    evaluation_metrics: ["Dice", "IoU", "连通性"],
    order: 10,
    contract_version: "1.0",
    model_id: "faultseg_3d",
    model_ids: ["faultseg_3d", "faultnet_china_field"],
    runnable_model_ids: ["faultseg_3d", "faultnet_china_field"],
    available: true,
    status: "可运行",
  },
  {
    id: "horizon",
    name: "层位识别",
    short_name: "层位识别",
    ...horizonSegmentationPresentation,
    ...horizonSegmentationRuntime,
    required_modalities: ["二维或三维地震"],
    order: 20,
    contract_version: "1.0",
  },
  ...[
    ["well_property", "储层物性预测", "孔隙度、渗透率、水饱和度、泥质含量和密度预测"],
    ["fluid_interpretation", "流体解释", "干层、水层、油层、气层与混合层解释结论"],
    ["facies_1d", "一维地震相分类", "沿井深输出确定相序列与连续相层段"],
    ["facies_3d", "三维地震相分割", "输出三维离散相体与分类切片"],
    ["fracture_development", "井侧裂缝发育排序", "沿井深输出低/中/高相对发育连续深度段；不是三维地震裂缝体分割"],
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
