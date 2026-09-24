#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主动声呐目标候选检测与定位 —— 单文件启动器。

等价于原 scripts/run_active_sonar_classification.ps1，依次执行：
  1) src/active_sonar_pipeline.py        候选检测 / 定位 / 渲染数据导出
  2) src/run_yolo_wal_inference.py       YOLO-WAL WCI 流体/伪影检测（默认）
  3) src/merge_target_classification.py  模型结果、规则候选类型与定位结果合并

默认运行 YOLO-WAL WCI 模型。UATD YOLO 来自前视声呐数据集，和当前 EM2040CD
多波束水柱 WCI 数据不匹配；如需备用对比实验，可手动传 --classifier uatd_yolo。

用法示例：
  python test.py
  python test.py --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all
  python test.py --threshold-k-mad 3.8
  python test.py --classifier yolo_wal --model-conf 0.25 --image-size 960
  python test.py --classifier uatd_yolo --model-conf 0.25 --image-size 640
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
DEFAULT_UATD_WEIGHTS = (
    PACKAGE_ROOT
    / "models"
    / "active_sonar_target_classifier_uatd_yolov8"
    / "weights"
    / "active_sonar_target_classifier_best.pt"
)
DEFAULT_YOLO_WAL_WEIGHTS = (
    PACKAGE_ROOT
    / "models"
    / "yolo_wal_wci_fluid_detector"
    / "weights"
    / "best_combined_gazcogne1.pt"
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
    print("\n[" + name + "]", flush=True)
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
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
    parser.add_argument(
        "--classifier",
        choices=["rule", "yolo_wal", "uatd_yolo", "yolo"],
        default="yolo_wal",
        help="yolo_wal=WCI专用流体/伪影模型；rule=仅使用规则候选类型；uatd_yolo=备用UATD前视声呐模型；yolo=uatd_yolo兼容别名",
    )
    parser.add_argument("--model-conf", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--data-root", default="")
    args = parser.parse_args()
    classifier = "uatd_yolo" if args.classifier == "yolo" else args.classifier

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
    weights = DEFAULT_YOLO_WAL_WEIGHTS if classifier == "yolo_wal" else DEFAULT_UATD_WEIGHTS
    product_id = args.product_id or product_id_from_all_name(args.survey_id, args.all_name)
    product_dir = output_root / product_id
    survey_root = data_root / args.survey_id
    all_file = survey_root / args.all_name

    print("Active sonar target candidate detection and localization", flush=True)
    print(f"Package root: {PACKAGE_ROOT}", flush=True)
    print(f"Data root:    {data_root}", flush=True)
    print(f"Output dir:   {product_dir}", flush=True)
    print(f"Survey id:    {args.survey_id}", flush=True)
    print(f"ALL file:     {args.all_name}", flush=True)
    print(f"Classifier:   {classifier}", flush=True)
    if classifier in {"yolo_wal", "uatd_yolo"}:
        print(f"Weights:      {weights}", flush=True)

    if not all_file.exists():
        raise SystemExit(
            f"ALL file not found: {all_file}. "
            f"Use --data-root to point to the folder that contains {args.survey_id}."
        )
    if classifier in {"yolo_wal", "uatd_yolo"} and not weights.exists():
        raise SystemExit(f"Model weights not found: {weights}")

    py = sys.executable

    # [1/3] 候选检测、定位与渲染数据导出
    run_step(
        "1/2] Candidate detection, localization, and render-data export" if classifier == "rule" else "1/3] Candidate detection, localization, and render-data export",
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

    if classifier == "yolo_wal":
        # [2/3] YOLO-WAL 是 WCI 领域模型，用于流体/气体羽流检测和伪影过滤。
        run_step(
            "2/3] YOLO-WAL WCI fluid/FP detection",
            [
                py, str(SRC_DIR / "run_yolo_wal_inference.py"),
                "--product-dir", str(product_dir),
                "--weights", str(weights),
                "--conf", str(args.model_conf),
                "--imgsz", str(args.image_size if args.image_size != 640 else 960),
            ],
            env,
        )
    elif classifier == "uatd_yolo":
        # [2/3] UATD YOLO 仅保留为备用对比模型。
        run_step(
            "2/3] UATD YOLO backup target classification experiment",
            [
                py, str(SRC_DIR / "run_sonar_yolo_inference.py"),
                "--product-dir", str(product_dir),
                "--weights", str(weights),
                "--conf", str(args.model_conf),
                "--imgsz", str(args.image_size),
            ],
            env,
        )

    # 合并候选类型与定位结果
    run_step(
        "2/2] Merge candidate type and localization results" if classifier == "rule" else "3/3] Merge classification and localization results",
        [
            py, str(SRC_DIR / "merge_target_classification.py"),
            "--product-dir", str(product_dir),
            "--classifier", classifier,
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
