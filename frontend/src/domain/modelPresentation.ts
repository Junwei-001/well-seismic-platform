import type { ArtifactRelease, ModelSpec } from "../api";

interface ModelPresentationDescriptor {
  task: string;
  input: string;
  output: string;
  scientificBoundary?: string;
}

const descriptors: Record<string, ModelPresentationDescriptor> = {
  faultseg_3d: {
    task: "慧眼三维断层识别",
    input: "SEG-Y 地震体",
    output: "默认工区中心单个128³完整块；可选重叠滑窗加权融合的全区概率体",
    scientificBoundary: "中心单块结果不声明连续全区覆盖；全区模式单独封存完整空间成果",
  },
  faultnet_china_field: {
    task: "慧眼区域增强断层识别",
    input: "SEG-Y 地震体",
    output: "逐窗 min-max 并重叠融合的全区断层概率体",
    scientificBoundary: "外部预训练迁移候选；目标工区微调后按空间分区独立验收",
  },
  seismic_surface_seg: {
    task: "地层实例分割",
    input: "SEG-Y 地震体",
    output: "地层标签体与置信度",
  },
  wellfuse_align_p13: {
    task: "自动精细井震标定",
    input: "SEG-Y + LAS + 完整轨迹",
    output: "TWT 与不确定性",
    scientificBoundary: "参考工区已验证，新工区须通过物理门",
  },
  "WellFuse-GeoAlign-12B-Direct-v1": {
    task: "多模态井震对齐",
    input: "SEG-Y + LAS + 完整轨迹",
    output: "TWT 与联合表征",
  },
  wellfuse_align_geopath_tie_v1: {
    task: "轨迹感知井震校正",
    input: "井震数据 + 完整轨迹",
    output: "候选时深轨与不确定性",
  },
  wellfuse_horizon_p17: {
    task: "历史四层位成果",
    input: "既有时间域 SEG-Y 任务",
    output: "四个候选层位面与不确定性",
    scientificBoundary: "只读，不再用于新推理",
  },
  wellfuse_facies_1d_p17: {
    task: "井侧沉积相分类",
    input: "九线 LAS + 完整轨迹",
    output: "确定相序列与连续相层段",
    scientificBoundary: "参考工区已验证，新工区为实验候选",
  },
  wellfuse_facies_3d_p17: {
    task: "三维地震相分割",
    input: "SEG-Y 地震体",
    output: "离散候选相体与分类切片",
  },
  wellfuse_facies_1d_chengdu_fast: {
    task: "井侧沉积相分类",
    input: "登记九线 LAS + 完整轨迹",
    output: "确定相序列与连续相层段",
    scientificBoundary: "登记数据集内已验证",
  },
  wellfuse_facies_3d_f3_fast: {
    task: "六类三维地震相分割",
    input: "时间域 SEG-Y",
    output: "离散相体与分类切片",
    scientificBoundary: "公开稠密基准条件通过，新工区为迁移候选",
  },
  wellfuse_channel_p17: {
    task: "河道地质体识别",
    input: "SEG-Y 地震体",
    output: "河道概率与几何属性",
  },
  wellfuse_karst_p17: {
    task: "岩溶地质体识别",
    input: "SEG-Y 地震体",
    output: "岩溶概率与几何属性",
  },
  wellfuse_fracture_development_utah_fast: {
    task: "井侧裂缝发育排序",
    input: "LAS/CSV 测井",
    output: "低/中/高连续MD发育段",
  },
  wellfuse_fault_p17_failed: {
    task: "稀疏断层研究档案",
    input: "稀疏断层棒 + 地震体",
    output: "失败审计证据",
  },
  wellfuse_den_p18: {
    task: "历史井侧密度成果",
    input: "历史封存井曲线",
    output: "DEN 曲线与不确定性",
    scientificBoundary: "只读历史成果",
  },
  wellfuse_por_p18: {
    task: "历史井侧孔隙度成果",
    input: "历史封存井曲线",
    output: "POR 曲线与不确定性",
    scientificBoundary: "只读历史成果",
  },
  wellfuse_log_perm_p18: {
    task: "历史井侧渗透率成果",
    input: "历史封存井曲线",
    output: "LOG_PERM 曲线与不确定性",
    scientificBoundary: "只读历史成果",
  },
  wellfuse_sw_p18: {
    task: "历史井侧含水饱和度成果",
    input: "历史封存井曲线",
    output: "SW 曲线与不确定性",
    scientificBoundary: "只读历史成果",
  },
  wellfuse_vsh_p18: {
    task: "历史井侧泥质含量成果",
    input: "历史封存井曲线",
    output: "VSH 曲线与不确定性",
    scientificBoundary: "只读历史成果",
  },
  wellfuse_fluid_interpretation_fast: {
    task: "五类流体解释",
    input: "登记九线 LAS",
    output: "连续确定流体层段 CSV",
    scientificBoundary: "登记数据集内已验证",
  },
  wellfuse_den_northwest_fast: {
    task: "井侧密度预测",
    input: "封存快照井资产",
    output: "DEN 整井曲线",
    scientificBoundary: "参考数据验证通过；新快照保留结果审计",
  },
  wellfuse_por_northwest_fast: {
    task: "井侧孔隙度预测",
    input: "封存快照井资产",
    output: "POR 整井曲线",
    scientificBoundary: "参考数据验证通过；新快照保留结果审计",
  },
  wellfuse_log_perm_northwest_fast: {
    task: "井侧渗透率预测",
    input: "封存快照井资产",
    output: "LOG_PERM 整井曲线",
    scientificBoundary: "参考数据验证通过；新快照保留结果审计",
  },
  wellfuse_sw_northwest_fast: {
    task: "井侧含水饱和度预测",
    input: "封存快照井资产",
    output: "SW 整井曲线",
    scientificBoundary: "参考数据验证通过；新快照保留结果审计",
  },
  wellfuse_vsh_northwest_fast: {
    task: "井侧泥质含量预测",
    input: "封存快照井资产",
    output: "VSH 整井曲线",
    scientificBoundary: "参考数据验证通过；新快照保留结果审计",
  },
  wellfuse_hydrocarbon_p18: {
    task: "含烃指示证据",
    input: "井曲线 + 完整轨迹",
    output: "含烃概率与不确定性",
    scientificBoundary: "条件通过，仅提供冻结成果",
  },
  wellfuse_fluid_8class_p18_failed: {
    task: "八类流体研究档案",
    input: "井曲线 + 完整轨迹",
    output: "失败审计证据",
  },
  wellfuse_water_gas_p18_failed: {
    task: "水气判别研究档案",
    input: "井曲线 + 完整轨迹",
    output: "失败审计证据",
  },
  seismic_baseline: {
    task: "地震单模态分割基线",
    input: "地震切片或子体",
    output: "目标概率与分割结果",
  },
  well_log_encoder: {
    task: "测井特征编码",
    input: "标准 LAS + 深度轴",
    output: "测井特征序列",
  },
  trajectory_encoder: {
    task: "井轨迹位置编码",
    input: "MD/TVD/XYZ 轨迹",
    output: "空间位置特征",
  },
  well_seismic_alignment: {
    task: "可学习井震对齐接口",
    input: "地震邻域 + 测井 + 轨迹",
    output: "对齐特征与不确定性",
  },
  confidence_gated_fusion: {
    task: "置信度门控井震融合",
    input: "井震特征 + 质量掩码",
    output: "统一融合特征",
  },
  learnable_fusion: {
    task: "可学习井震融合接口",
    input: "井震编码 + 位置编码",
    output: "统一融合特征",
  },
};

const publicTechnicalIdentifiers: Record<string, string> = {
  faultseg_3d: "project-fault-3d-primary",
  faultnet_china_field: "project-fault-3d-regional",
  seismic_surface_seg: "project-strata-instance-segmentation",
};

const publicTextReplacements: Array<[RegExp, string]> = [
  [/CIG[\s_-]*Bench/gi, "公开地质体基准"],
  [/CIGVis/gi, "项目可视化"],
  [/(?<![A-Za-z0-9_])CIG(?![A-Za-z0-9_])/gi, "项目可视化"],
  [/(?<![A-Za-z0-9])(?:Seismic[\s_-]*Foundation[\s_-]*Model|SFM(?:[\s_-]*Base(?:[\s_-]*224)?|[\s_-]*(?:tokens?|view[\s_-]*mask))?)(?![A-Za-z0-9])/gi, "空间特征融合模块"],
  [/(?<![A-Za-z0-9])MOMENT(?:[\s_-]*1[\s_-]*small|[\s_-]*(?:tokens?|patch[\s_-]*mask))?(?![A-Za-z0-9])/gi, "测井序列特征模块"],
  [/(?<![A-Za-z0-9_])NCS(?![A-Za-z0-9_])/gi, "全局编码组件"],
  [/ViT(?:25D|3D)/gi, "地震特征编码分支"],
  [/Fault[\s_-]*Seg(?:3D)?/gi, "慧眼三维断层模型"],
  [/Fault[\s_-]*Net/gi, "慧眼区域断层模型"],
  [/Mask2Former/gi, "精细地层分割分支"],
  [/SegFormer/gi, "基础地层分割分支"],
  [/(?:seismic|legacy)?[\s_-]*surface[\s_-]*seg/gi, "有序地层实例分割"],
  [/(?:3D[\s_-]*)?U[\s_-]?Net/gi, "三维卷积分割网络"],
  [/TorchScript/gi, "封装推理模型"],
  [/Transformer/gi, "多模态序列网络"],
  [/\bViser\b/gi, "三维交互引擎"],
  [/(?:Plotly|Matplotlib)/gi, "二维成果引擎"],
  [/开源(?:模型)?/g, "项目构建模型"],
];

export function publicModelText(value: unknown, fallback = ""): string {
  let text = String(value === undefined || value === null || value === "" ? fallback : value);
  for (const [pattern, replacement] of publicTextReplacements) {
    text = text.replace(pattern, replacement);
  }
  return text;
}

export function publicModelIdentifier(modelId: unknown): string {
  const id = String(modelId || "").trim();
  if (!id) return "project-model";
  return publicTechnicalIdentifiers[id] || publicModelText(id, "project-model");
}

const scientificLabels: Record<string, string> = {
  validated: "科学验证通过",
  passed: "科学验证通过",
  conditional: "条件通过",
  candidate: "实验候选",
  experimental: "实验候选",
  failed: "未通过，仅保留证据",
  blocked: "未通过，仅保留证据",
  legacy: "既有基线",
  descriptive: "描述性证据",
  unassessed: "尚未评估",
  unknown: "证据待登记",
};

export function scientificStatusLabel(status?: string): string {
  const normalized = String(status || "unknown").toLowerCase();
  const exact = scientificLabels[normalized];
  if (exact) return exact;
  if (normalized.includes("failed") || normalized.includes("blocked")) {
    return scientificLabels.failed;
  }
  if (normalized.includes("conditional")) return scientificLabels.conditional;
  if (normalized.includes("candidate") || normalized.includes("experimental")) {
    return scientificLabels.candidate;
  }
  if (normalized.includes("validated") || normalized.includes("passed")) {
    return scientificLabels.validated;
  }
  if (normalized.includes("not_evaluated") || normalized.includes("unassessed")) {
    return scientificLabels.unassessed;
  }
  return scientificLabels.unknown;
}

export function modelPresentationName(
  modelId: string | undefined,
  fallbackName: string | undefined,
  scientificStatus?: string,
): string {
  const descriptor = modelId ? descriptors[modelId] : undefined;
  if (!descriptor) {
    const fallback = publicModelText(fallbackName?.trim(), "未命名模型");
    return `${fallback}｜${scientificStatusLabel(scientificStatus)}`;
  }
  const boundary = descriptor.scientificBoundary || scientificStatusLabel(scientificStatus);
  return `${descriptor.task}（${descriptor.input} → ${descriptor.output}｜${boundary}）`;
}

export function modelSpecPresentationName(model: Pick<ModelSpec, "id" | "name" | "scientific_status">): string {
  return modelPresentationName(model.id, model.name, model.scientific_status);
}

export function releasePresentationName(release: Pick<ArtifactRelease, "id" | "model_id" | "name" | "display_name" | "scientific_status">): string {
  return modelPresentationName(
    release.model_id || release.id,
    release.display_name || release.name || release.id,
    release.scientific_status,
  );
}
