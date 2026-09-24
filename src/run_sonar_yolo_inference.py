from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = (
    ROOT
    / "models"
    / "active_sonar_target_classifier_uatd_yolov8"
    / "weights"
    / "active_sonar_target_classifier_best.pt"
)


LABEL_MAP = {
    "ball": ("geometric_target_candidate", "规则几何目标候选"),
    "circle cage": ("cage_candidate", "圆形笼体候选"),
    "cube": ("geometric_target_candidate", "立方体目标候选"),
    "cylinder": ("geometric_target_candidate", "圆柱体目标候选"),
    "human body": ("human_like_target_candidate", "人形目标候选"),
    "metal bucket": ("debris_or_artificial_object_candidate", "金属桶/人工物候选"),
    "plane": ("artificial_object_candidate", "平面人工物候选"),
    "rov": ("rov_candidate", "ROV目标候选"),
    "square cage": ("cage_candidate", "方形笼体候选"),
    "tyre": ("debris_or_artificial_object_candidate", "轮胎/人工物候选"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def as_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def normalize_label(label: str) -> tuple[str, str, str]:
    normalized = " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())
    final_label, display_label = LABEL_MAP.get(
        normalized,
        ("unknown_model_target_candidate", "模型未知目标候选"),
    )
    return final_label, display_label, normalized


def row_from_detection(det: dict[str, Any]) -> dict[str, Any]:
    bbox = det.get("bbox_xyxy_px", {})
    return {
        "product_id": det.get("product_id", ""),
        "region_id": det.get("region_id", ""),
        "crop_png": det.get("crop_png", ""),
        "overlay_png": det.get("overlay_png", ""),
        "model_class_id": det.get("class_id", ""),
        "model_class_name": det.get("class_name", ""),
        "final_label": det.get("final_label", ""),
        "display_label": det.get("display_label", ""),
        "confidence": det.get("confidence", ""),
        "x1_px": bbox.get("x1", ""),
        "y1_px": bbox.get("y1", ""),
        "x2_px": bbox.get("x2", ""),
        "y2_px": bbox.get("y2", ""),
    }


def write_csv(path: Path, detections: list[dict[str, Any]]) -> None:
    fields = [
        "product_id",
        "region_id",
        "crop_png",
        "overlay_png",
        "model_class_id",
        "model_class_name",
        "final_label",
        "display_label",
        "confidence",
        "x1_px",
        "y1_px",
        "x2_px",
        "y2_px",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for det in detections:
            writer.writerow(row_from_detection(det))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a pretrained sonar target YOLO model on exported candidate echo crops."
    )
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--model-name",
        default="Sonar-Threat-Detection-YOLOv8-UATD",
        help="Model name written into output metadata.",
    )
    args = parser.parse_args()

    product_dir = args.product_dir.resolve()
    weights = args.weights.resolve()
    manifest_path = product_dir / "candidate_crop_manifest.json"
    overlay_dir = product_dir / "sonar_yolo_overlays"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run src/active_sonar_pipeline.py first so candidate crops exist."
        )
    if not weights.exists():
        raise FileNotFoundError(f"Missing sonar YOLO weights: {weights}")

    manifest = load_json(manifest_path)
    crops = manifest.get("crops", [])
    model = YOLO(str(weights))
    names = getattr(model, "names", {})

    detections: list[dict[str, Any]] = []
    per_region: dict[str, dict[str, Any]] = {}
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for crop in crops:
        region_id = str(crop.get("region_id", ""))
        crop_path = as_path(str(crop.get("ml_input_png") or crop.get("wci_crop_png") or crop.get("crop_png", ""))).resolve()
        if not crop_path.exists():
            per_region[region_id] = {
                "region_id": region_id,
                "status": "crop_missing",
                "detections": [],
            }
            continue

        results = model.predict(
            source=str(crop_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )
        result = results[0]
        overlay_path = overlay_dir / f"{region_id}_sonar_yolo_overlay.png"
        cv2.imwrite(str(overlay_path), result.plot())

        region_detections: list[dict[str, Any]] = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = str(names.get(cls_id, cls_id))
            final_label, display_label, normalized_class_name = normalize_label(class_name)
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            xywhn = [float(v) for v in box.xywhn[0].tolist()]
            det = {
                "product_id": crop.get("product_id", ""),
                "region_id": region_id,
                "crop_png": str(crop_path),
                "overlay_png": str(overlay_path),
                "class_id": cls_id,
                "class_name": class_name,
                "normalized_class_name": normalized_class_name,
                "final_label": final_label,
                "display_label": display_label,
                "confidence": conf,
                "bbox_xyxy_px": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                },
                "bbox_xywhn": {
                    "x_center": xywhn[0],
                    "y_center": xywhn[1],
                    "width": xywhn[2],
                    "height": xywhn[3],
                },
            }
            detections.append(det)
            region_detections.append(det)

        best_detection = max(region_detections, key=lambda item: item["confidence"], default=None)
        per_region[region_id] = {
            "region_id": region_id,
            "status": "sonar_yolo_positive" if region_detections else "sonar_yolo_negative",
            "detection_count": len(region_detections),
            "best_confidence": best_detection["confidence"] if best_detection else 0.0,
            "best_detection": best_detection,
            "overlay_png": str(overlay_path),
            "detections": region_detections,
        }

    payload = {
        "model": {
            "name": args.model_name,
            "source": "https://github.com/lMelkorl/Sonar-Threat-Detection-YOLO",
            "dataset": "UATD underwater acoustic target detection dataset",
            "task": "active/front-looking sonar target detection on candidate sonar crops",
            "weights": str(weights),
            "classes": names,
            "confidence_threshold": args.conf,
            "imgsz": args.imgsz,
        },
        "scope_note": (
            "This model classifies UATD sonar target classes: ball, cage, cube, cylinder, "
            "human body, metal bucket, plane, rov, and tyre. It is a practical pretrained "
            "sonar target classifier, not a dedicated submarine classifier."
        ),
        "source_manifest": str(manifest_path),
        "region_results": list(per_region.values()),
        "detections": detections,
    }
    write_json(product_dir / "sonar_yolo_detections.json", payload)
    write_csv(product_dir / "sonar_yolo_detections.csv", detections)

    print(f"Sonar YOLO regions processed: {len(crops)}")
    print(f"Sonar YOLO target detections: {len(detections)}")
    print(product_dir / "sonar_yolo_detections.json")
    print(product_dir / "sonar_yolo_detections.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
