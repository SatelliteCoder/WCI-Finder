#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主动声呐目标识别与定位 —— 单文件启动器。

等价于原 scripts/run_active_sonar_classification.ps1，依次执行：
  1) src/active_sonar_pipeline.py        候选检测 / 定位 / 渲染数据导出
  2) src/run_sonar_yolo_inference.py      YOLO 目标分类
  3) src/merge_target_classification.py  分类与定位结果合并

用法示例：
  python test.py
  python test.py --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all
  python test.py --threshold-k-mad 3.8 --model-conf 0.25 --image-size 640
  python test.py --data-root D:/sonar_data
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# 包根目录 = test.py 所在目录
PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_ROOT / "src"
DEFAULT_WEIGHTS = (
    PACKAGE_ROOT
    / "models"
    / "active_sonar_target_classifier_uatd_yolov8"
    / "weights"
    / "active_sonar_target_classifier_best.pt"
)


def product_id_from_all_name(survey_id: str, all_name: str) -> str:
    """与 src/active_sonar_pipeline.py 中同名函数保持一致。"""
    base = Path(all_name).stem
    match = re.match(r"^(\d{4})_(\d{8})_\d+_(.+)$", base)
    if match:
        line, date, platform = match.groups()
        return f"{date}_{line}_{platform}"
    clean_survey = re.sub(r"[^\w]+", "_", survey_id)
    clean_base = re.sub(r"[^\w]+", "_", base)
    return f"{clean_survey}_{clean_base}"


def run_step(name: str, cmd: list[str], env: dict[str, str]) -> None:
    print("\n[" + name + "]")
    print("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(PACKAGE_ROOT), env=env)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {name}. Exit code: {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="主动声呐目标识别与定位单文件启动器"
    )
    parser.add_argument("--survey-id", default="20181112_survey")
    parser.add_argument("--all-name", default="0000_20181112_080147_TecnopescaII.all")
    parser.add_argument("--product-id", default="")
    parser.add_argument("--threshold-k-mad", type=float, default=3.8)
    parser.add_argument("--bottom-guard-samples", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=80)
    parser.add_argument("--model-conf", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--data-root", default="")
    args = parser.parse_args()

    # 统一 UTF-8 输出（本进程 + 子进程）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    data_root = Path(args.data_root) if args.data_root else PACKAGE_ROOT / "data"
    output_root = PACKAGE_ROOT / "outputs"
    weights = DEFAULT_WEIGHTS
    product_id = args.product_id or product_id_from_all_name(args.survey_id, args.all_name)
    product_dir = output_root / product_id
    survey_root = data_root / args.survey_id
    all_file = survey_root / args.all_name

    print("Active sonar target classification and localization")
    print(f"Package root: {PACKAGE_ROOT}")
    print(f"Data root:    {data_root}")
    print(f"Output dir:   {product_dir}")
    print(f"Survey id:    {args.survey_id}")
    print(f"ALL file:     {args.all_name}")
    print(f"Weights:      {weights}")

    if not all_file.exists():
        raise SystemExit(
            f"ALL file not found: {all_file}. "
            f"Use --data-root to point to the folder that contains {args.survey_id}."
        )
    if not weights.exists():
        raise SystemExit(f"Model weights not found: {weights}")

    py = sys.executable

    # [1/3] 候选检测、定位与渲染数据导出
    run_step(
        "1/3] Candidate detection, localization, and render-data export",
        [
            py, str(SRC_DIR / "active_sonar_pipeline.py"),
            "--survey-id", args.survey_id,
            "--all-name", args.all_name,
            "--product-id", product_id,
            "--data-root", str(data_root),
            "--output-root", str(output_root),
            "--threshold-k-mad", str(args.threshold_k_mad),
            "--bottom-guard-samples", str(args.bottom_guard_samples),
            "--max-regions", str(args.max_regions),
        ],
        env,
    )

    # [2/3] YOLO 目标分类
    run_step(
        "2/3] YOLO target classification",
        [
            py, str(SRC_DIR / "run_sonar_yolo_inference.py"),
            "--product-dir", str(product_dir),
            "--weights", str(weights),
            "--conf", str(args.model_conf),
            "--imgsz", str(args.image_size),
        ],
        env,
    )

    # [3/3] 合并分类与定位结果
    run_step(
        "3/3] Merge classification and localization results",
        [
            py, str(SRC_DIR / "merge_target_classification.py"),
            "--product-dir", str(product_dir),
            "--model-conf", str(args.model_conf),
        ],
        env,
    )

    print("")
    print("Done. Key outputs:")
    print(f"  {product_dir / 'final_targets.json'}")
    print(f"  {product_dir / 'final_targets.csv'}")
    print(f"  {product_dir / 'echogram_render_data.json'}")
    print(f"  {product_dir / 'candidate_wci_render_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
