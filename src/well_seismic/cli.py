from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import WellSeismicPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifest-driven well/seismic preprocessing")
    parser.add_argument("--input-root", "--输入目录", help="比赛输入根目录；无需预先提供清单")
    parser.add_argument("--manifest", help="高级模式：显式输入清单")
    parser.add_argument("--seismic-dir", "--地震路径", action="append", help="地震数据绝对路径；可重复指定")
    parser.add_argument("--log-dir", "--测井路径", action="append", help="LAS测井数据绝对路径；可重复指定")
    parser.add_argument(
        "--metadata-dir",
        "--井基础与轨迹路径",
        "--井相关目录",
        action="append",
        help="可选：井位、海拔和井轨迹绝对路径；可重复指定",
    )
    parser.add_argument(
        "--auxiliary-dir",
        "--其他辅助路径",
        "--辅助目录",
        action="append",
        help="可选：其他辅助数据绝对路径；可重复指定",
    )
    default_config = Path(__file__).resolve().parents[2] / "configs"
    parser.add_argument("--config-dir", "--配置目录", default=str(default_config), help="知识库和预处理配置目录")
    parser.add_argument("--output", "--输出目录", required=True, help="中文输出目录")
    parser.add_argument("--inspect-only", action="store_true", help="Read and validate data without building samples")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    explicit_paths = bool(args.seismic_dir or args.log_dir or args.metadata_dir or args.auxiliary_dir)
    modes = int(bool(args.input_root)) + int(bool(args.manifest)) + int(explicit_paths)
    if modes != 1:
        parser.error("必须且只能选择一种输入方式：--input-root、--manifest，或分别指定--seismic-dir与--log-dir")
    if explicit_paths and (not args.seismic_dir or not args.log_dir):
        parser.error("分别指定目录时，--seismic-dir和--log-dir均为必填")
    if explicit_paths:
        pipeline = WellSeismicPipeline.from_input_paths(
            seismic_directory=args.seismic_dir,
            log_directory=args.log_dir,
            metadata_directory=args.metadata_dir,
            auxiliary_directory=args.auxiliary_dir,
            config_dir=args.config_dir,
        ).ingest()
    elif args.input_root:
        pipeline = WellSeismicPipeline.from_input_root(args.input_root, args.config_dir).ingest()
    else:
        pipeline = WellSeismicPipeline(args.manifest, args.config_dir).ingest()
    if not args.inspect_only:
        pipeline.build_samples()
    paths = pipeline.write_outputs(args.output)
    summary = pipeline.quality_report()["summary"]
    chinese_summary = {
        "数据资产数": summary["assets"], "跳过重复文件数": summary["duplicates_skipped"],
        "井实体数": summary["wells"], "地震文件数": summary["seismic_files"],
        "多模态样本数": summary["samples"], "读取错误数": summary["errors"],
    }
    print(json.dumps({"汇总": chinese_summary, "输出文件": {k: str(v) for k, v in paths.items()}}, ensure_ascii=False, indent=2))
    return 0 if not pipeline.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
