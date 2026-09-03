from __future__ import annotations

from typing import Any


SAMPLE_FIELDS_ZH = {
    "well_uid": "井唯一标识", "well_name": "井名", "log_source": "测井来源",
    "seismic_source": "地震来源", "md": "测量深度MD", "tvd": "垂直深度TVD",
    "z_msl_m": "MSL绝对高程z_msl_m（向上为正）",
    "depth_below_msl_m": "海平面以下深度（向下为正）",
    "depth_below_srd_m": "地震处理基准面以下深度",
    "x": "X坐标", "y": "Y坐标",
    "trace_index": "地震道索引", "inline": "主测线Inline", "crossline": "联络测线Crossline",
    "distance": "井震水平距离", "seismic_coordinate": "地震时间或深度坐标",
    "horizontal_confidence": "水平匹配置信度", "vertical_method": "垂向匹配方法",
    "vertical_confidence": "垂向匹配置信度", "vertical_status": "垂向标定状态",
    "vertical_uncertainty_ms": "垂向时间不确定度毫秒", "well_features": "测井特征",
    "well_mask": "测井有效掩码", "seismic_window": "地震窗口",
    "seismic_window_valid": "地震窗口有效", "coordinate_reference_verified": "坐标参考已核验",
    "vertical_datum_verified": "垂向基准已统一核验",
    "training_eligible": "可用于多模态训练",
    "provenance": "来源记录",
    "fused_features": "井震融合特征", "fusion_weight": "融合门控权重",
    "fusion_metadata": "融合算法记录",
}
SAMPLE_FIELDS_EN = {value: key for key, value in SAMPLE_FIELDS_ZH.items()}


PROVENANCE_FIELDS_ZH = {
    "well_head": "井位来源", "trajectory": "井轨迹来源", "log": "测井文件",
    "seismic": "地震文件", "curve_mapping_version": "曲线知识库版本",
    "segy_profile": "SEG-Y读取配置", "segy_revision": "SEG-Y版本",
    "log_asset_id": "测井资产标识", "log_stage": "测井处理阶段",
    "log_version": "测井资产版本", "seismic_asset_id": "地震资产标识",
    "seismic_stage": "地震处理阶段", "seismic_version": "地震资产版本",
    "neighbor_trace_indices": "邻域地震道索引", "neighbor_weights": "邻域插值权重",
    "vertical_alignment": "垂向标定记录", "vertical_datum": "垂向基准归一化记录",
}

FUSION_FIELDS_ZH = {
    "algorithm": "算法名称", "version": "算法版本", "curve_order": "曲线顺序",
    "horizontal_confidence": "水平匹配置信度", "vertical_confidence": "垂向匹配置信度",
}


def sample_to_chinese(sample: dict[str, Any]) -> dict[str, Any]:
    output = {SAMPLE_FIELDS_ZH.get(key, key): value for key, value in sample.items()}
    provenance_key = SAMPLE_FIELDS_ZH["provenance"]
    if isinstance(output.get(provenance_key), dict):
        output[provenance_key] = {PROVENANCE_FIELDS_ZH.get(key, key): value for key, value in output[provenance_key].items()}
    fusion_key = SAMPLE_FIELDS_ZH["fusion_metadata"]
    if isinstance(output.get(fusion_key), dict):
        output[fusion_key] = {FUSION_FIELDS_ZH.get(key, key): value for key, value in output[fusion_key].items()}
    return output


def sample_to_internal(sample: dict[str, Any]) -> dict[str, Any]:
    return {SAMPLE_FIELDS_EN.get(key, key): value for key, value in sample.items()}
