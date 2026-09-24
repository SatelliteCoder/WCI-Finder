from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR_YOLOV5 = ROOT / "src" / "vendor" / "yolov5"
DEFAULT_WEIGHTS = (
    ROOT
    / "models"
    / "yolo_wal_wci_fluid_detector"
    / "weights"
    / "best_combined_gazcogne1.pt"
)

# YOLO-WAL weights are YOLOv5 checkpoints. New PyTorch versions default to
# weights_only=True, which cannot load old YOLOv5 checkpoint objects.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
if str(VENDOR_YOLOV5) not in sys.path:
    sys.path.insert(0, str(VENDOR_YOLOV5))

from models.common import DetectMultiBackend  # type: ignore  # noqa: E402
from utils.augmentations import letterbox  # type: ignore  # noqa: E402
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh  # type: ignore  # noqa: E402
from utils.torch_utils import select_device  # type: ignore  # noqa: E402


LABEL_MAP = {
    "fluide": ("gas_plume_candidate", "流体/气体羽流候选"),
    "fluid": ("gas_plume_candidate", "流体/气体羽流候选"),
    "gas plume": ("gas_plume_candidate", "流体/气体羽流候选"),
    "fp": ("wci_false_positive_candidate", "WCI伪影候选"),
    "false positive": ("wci_false_positive_candidate", "WCI伪影候选"),
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
    return path if path.is_absolute() else ROOT / path


def normalize_label(label: str) -> tuple[str, str, str]:
    normalized = " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())
    final_label, display_label = LABEL_MAP.get(
        normalized,
        ("unknown_wci_model_candidate", "WCI模型未知候选"),
    )
    return final_label, display_label, normalized


def preprocess_image(img0: np.ndarray, imgsz: int, stride: int) -> tuple[torch.Tensor, np.ndarray]:
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).float() / 255.0
    if tensor.ndim == 3:
        tensor = tensor[None]
    return tensor, img


def draw_overlay(img: np.ndarray, detections: list[dict[str, Any]], path: Path) -> None:
    overlay = img.copy()
    for det in detections:
        bbox = det["bbox_xyxy_px"]
        x1, y1, x2, y2 = (int(round(bbox[k])) for k in ("x1", "y1", "x2", "y2"))
        class_name = str(det.get("class_name", ""))
        conf = float(det.get("confidence", 0.0))
        color = (0, 180, 255) if det.get("final_label") == "gas_plume_candidate" else (160, 160, 160)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{class_name} {conf:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


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
        description="Run YOLO-WAL WCI fluid/false-positive detector on exported WCI candidate crops."
    )
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--model-name",
        default="YOLO-WAL-WCI-fluid-detector",
        help="Model name written into output metadata.",
    )
    args = parser.parse_args()

    product_dir = args.product_dir.resolve()
    weights = args.weights.resolve()
    manifest_path = product_dir / "candidate_crop_manifest.json"
    overlay_dir = product_dir / "yolo_wal_overlays"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run src/active_sonar_pipeline.py first so candidate crops exist."
        )
    if not weights.exists():
        raise FileNotFoundError(f"Missing YOLO-WAL weights: {weights}")
    if not VENDOR_YOLOV5.exists():
        raise FileNotFoundError(f"Missing vendored YOLOv5 runtime: {VENDOR_YOLOV5}")

    manifest = load_json(manifest_path)
    crops = manifest.get("crops", [])
    device = select_device(args.device)
    model = DetectMultiBackend(str(weights), device=device)
    stride = int(model.stride)
    names = getattr(model, "names", {0: "fluide", 1: "FP"})
    model.warmup(imgsz=(1, 3, args.imgsz, args.imgsz))

    detections: list[dict[str, Any]] = []
    per_region: dict[str, dict[str, Any]] = {}
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for crop in crops:
        region_id = str(crop.get("region_id", ""))
        crop_path = as_path(
            str(crop.get("ml_input_png") or crop.get("wci_crop_png") or crop.get("crop_png", ""))
        ).resolve()
        if not crop_path.exists():
            per_region[region_id] = {
                "region_id": region_id,
                "status": "crop_missing",
                "detections": [],
            }
            continue

        img0 = cv2.imread(str(crop_path))
        if img0 is None:
            per_region[region_id] = {
                "region_id": region_id,
                "status": "crop_unreadable",
                "detections": [],
            }
            continue

        tensor, _ = preprocess_image(img0, args.imgsz, stride)
        tensor = tensor.to(device)
        pred = model(tensor)
        pred = non_max_suppression(pred, args.conf, args.iou)
        det = pred[0]
        overlay_path = overlay_dir / f"{region_id}_yolo_wal_overlay.png"

        region_detections: list[dict[str, Any]] = []
        if len(det):
            det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det.tolist():
                cls_id = int(cls)
                class_name = str(names.get(cls_id, cls_id))
                final_label, display_label, normalized_class_name = normalize_label(class_name)
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                xywh = xyxy2xywh(torch.tensor([[x1, y1, x2, y2]])).view(-1).tolist()
                h, w = img0.shape[:2]
                det_payload = {
                    "product_id": crop.get("product_id", ""),
                    "region_id": region_id,
                    "crop_png": str(crop_path),
                    "overlay_png": str(overlay_path),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "normalized_class_name": normalized_class_name,
                    "final_label": final_label,
                    "display_label": display_label,
                    "confidence": float(conf),
                    "bbox_xyxy_px": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "bbox_xywhn": {
                        "x_center": float(xywh[0] / w),
                        "y_center": float(xywh[1] / h),
                        "width": float(xywh[2] / w),
                        "height": float(xywh[3] / h),
                    },
                }
                detections.append(det_payload)
                region_detections.append(det_payload)

        draw_overlay(img0, region_detections, overlay_path)
        best_detection = max(region_detections, key=lambda item: item["confidence"], default=None)
        best_label = str(best_detection.get("final_label", "")) if best_detection else ""
        status = "yolo_wal_negative"
        if best_detection and best_label == "gas_plume_candidate":
            status = "yolo_wal_fluid_positive"
        elif best_detection and best_label == "wci_false_positive_candidate":
            status = "yolo_wal_false_positive"
        elif best_detection:
            status = "yolo_wal_positive"

        per_region[region_id] = {
            "region_id": region_id,
            "status": status,
            "detection_count": len(region_detections),
            "best_confidence": best_detection["confidence"] if best_detection else 0.0,
            "best_detection": best_detection,
            "overlay_png": str(overlay_path),
            "detections": region_detections,
        }

    payload = {
        "model": {
            "name": args.model_name,
            "source": "https://github.com/perrettymea/YOLO-WAL-fluid-detection-WCI-data",
            "dataset": "YOLO-WAL WCI fluid/gas plume detector",
            "task": "multibeam WCI fluid/gas-plume detection and FP filtering",
            "weights": str(weights),
            "classes": names,
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "imgsz": args.imgsz,
        },
        "scope_note": (
            "YOLO-WAL is a WCI-domain model for fluid/gas-plume detection and false-positive filtering. "
            "It is not a fish/submarine/general artificial-object classifier."
        ),
        "source_manifest": str(manifest_path),
        "region_results": list(per_region.values()),
        "detections": detections,
    }
    write_json(product_dir / "yolo_wal_detections.json", payload)
    write_csv(product_dir / "yolo_wal_detections.csv", detections)

    print(f"YOLO-WAL regions processed: {len(crops)}")
    print(f"YOLO-WAL detections: {len(detections)}")
    print(product_dir / "yolo_wal_detections.json")
    print(product_dir / "yolo_wal_detections.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
