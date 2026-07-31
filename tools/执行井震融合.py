from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from well_seismic.config import load_yaml
from well_seismic.datasets import JsonlMultimodalDataset
from well_seismic.fusion import build_fusion
from well_seismic.output_schema import sample_to_chinese


def main() -> None:
    parser = argparse.ArgumentParser(description="对已匹配的多模态样本执行可配置井震融合")
    parser.add_argument(
        "--输入",
        type=Path,
        default=PROJECT / "输出结果" / "当前数据运行结果" / "多模态样本.jsonl",
    )
    parser.add_argument(
        "--输出目录",
        type=Path,
        default=PROJECT / "输出结果" / "当前数据运行结果",
    )
    parser.add_argument(
        "--配置",
        type=Path,
        default=PROJECT / "configs" / "fusion.yaml",
    )
    args = parser.parse_args()

    dataset = JsonlMultimodalDataset(args.输入)
    config = load_yaml(args.配置).get("fusion", {})
    fusion = build_fusion(config)
    samples = fusion.fit_transform(dataset)

    args.输出目录.mkdir(parents=True, exist_ok=True)
    sample_path = args.输出目录 / "井震融合样本.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample_to_chinese(sample), ensure_ascii=False) + "\n")
    state_path = args.输出目录 / "井震融合算法状态.json"
    state_path.write_text(
        json.dumps(fusion.state_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print({
        "融合样本数": len(samples),
        "融合算法": fusion.state_dict().get("algorithm", type(fusion).__name__),
        "融合样本": str(sample_path),
        "算法状态": str(state_path),
    })


if __name__ == "__main__":
    main()
