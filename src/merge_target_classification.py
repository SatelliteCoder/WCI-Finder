from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def find_by_region(items: list[dict[str, Any]], region_id: str) -> dict[str, Any] | None:
    for item in items:
        if str(item.get("region_id", "")) == region_id:
            return item
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "product_id",
        "region_id",
        "final_label",
        "display_label",
        "final_confidence",
        "classification_source",
        "model_class_name",
        "model_display_label",
        "yolo_wal_status",
        "yolo_wal_best_confidence",
        "sonar_yolo_status",
        "sonar_yolo_best_confidence",
        "rule_classification",
        "rule_confidence",
        "ping",
        "beam",
        "sample_start",
        "sample_end",
        "latitude_deg",
        "longitude_deg",
        "depth_m",
        "slant_range_m",
        "distance_to_track_m",
        "bearing_relative_to_heading_deg",
        "side_of_track",
        "crop_png",
        "sonar_yolo_overlay_png",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge active-sonar rule candidates, Python-derived coordinates, and optional YOLO experiment results."
    )
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument(
        "--classifier",
        choices=["rule", "yolo_wal", "uatd_yolo", "yolo"],
        default="rule",
        help="rule=do not load model outputs; yolo_wal=use WCI YOLO-WAL outputs; uatd_yolo=use backup UATD YOLO outputs; yolo=uatd_yolo compatibility alias",
    )
    parser.add_argument("--model-conf", type=float, default=0.25)
    parser.add_argument("--fish-conf", type=float, default=None)
    args = parser.parse_args()
    classifier = "uatd_yolo" if args.classifier == "yolo" else args.classifier
    model_conf = args.model_conf if args.fish_conf is None else args.fish_conf

    product_dir = args.product_dir.resolve()
    regions_payload = load_json(product_dir / "candidate_regions.json")
    positions_payload = load_json(product_dir / "candidate_positions.json")
    crop_payload = load_json(product_dir / "candidate_crop_manifest.json")
    yolo_wal_path = product_dir / "yolo_wal_detections.json"
    sonar_yolo_path = product_dir / "sonar_yolo_detections.json"
    fish_yolo_path = product_dir / "fish_yolo_detections.json"
    yolo_payload: dict[str, Any] = {"region_results": []}
    if classifier in {"yolo_wal", "uatd_yolo"}:
        if classifier == "yolo_wal":
            yolo_input_path = yolo_wal_path
        else:
            yolo_input_path = sonar_yolo_path if sonar_yolo_path.exists() else fish_yolo_path
        if not yolo_input_path.exists():
            raise FileNotFoundError(
                f"Missing YOLO output under {product_dir}. "
                "Run the matching YOLO inference script first or use --classifier rule."
            )
        yolo_payload = load_json(yolo_input_path)

    regions = regions_payload.get("regions", [])
    positions = positions_payload.get("positions", [])
    crops = crop_payload.get("crops", [])
    yolo_regions = yolo_payload.get("region_results", [])

    final_targets: list[dict[str, Any]] = []
    for region in regions:
        region_id = str(region.get("region_id", ""))
        position = find_by_region(positions, region_id) or {}
        crop = find_by_region(crops, region_id) or {}
        yolo = find_by_region(yolo_regions, region_id) or {
            "status": "not_run",
            "best_confidence": 0.0,
            "detections": [],
        }

        yolo_best = float(yolo.get("best_confidence", 0.0) or 0.0)
        yolo_positive = (
            classifier in {"yolo_wal", "uatd_yolo"}
            and str(yolo.get("status", "")) in {
                "yolo_wal_fluid_positive",
                "yolo_wal_false_positive",
                "yolo_wal_positive",
                "sonar_yolo_positive",
            }
            and yolo_best >= model_conf
        )
        best_detection = yolo.get("best_detection") or {}
        rule_label = str(region.get("classification", "unknown_target_candidate"))
        if rule_label == "noise_candidate":
            continue
        rule_conf = float(region.get("classification_confidence", region.get("confidence", 0.0)) or 0.0)

        if yolo_positive:
            model_class_name = str(best_detection.get("class_name", ""))
            if classifier == "yolo_wal":
                final_label = str(best_detection.get("final_label") or "wci_model_candidate")
                display_label = str(best_detection.get("display_label") or final_label)
                source = "YOLO_WAL_WCI_fluid_detector"
            elif model_class_name.lower() == "fish":
                final_label = "fish_verified_candidate"
                display_label = "鱼群候选-模型确认"
                source = "UATD_YOLO_backup" if sonar_yolo_path.exists() else "FishDetectionAI_YOLO_pretrained_on_sonar_fish_images"
            else:
                final_label = str(best_detection.get("final_label") or "model_target_candidate")
                display_label = str(best_detection.get("display_label") or final_label)
                source = "UATD_YOLO_backup" if sonar_yolo_path.exists() else "FishDetectionAI_YOLO_pretrained_on_sonar_fish_images"
            final_conf = yolo_best
        elif rule_label == "fish_school_candidate":
            final_label = "fish_rule_candidate_unverified"
            display_label = "鱼群候选-规则未确认"
            final_conf = rule_conf
            source = "rule_based_geometry_intensity_only"
        else:
            final_label = rule_label
            display_label = ""
            final_conf = rule_conf
            source = "rule_based_geometry_intensity_only"

        if classifier == "rule":
            yolo = {
                "status": "disabled",
                "best_confidence": 0.0,
                "detections": [],
                "overlay_png": "",
            }
            yolo_best = 0.0
            best_detection = {}

        mp = position.get("map_position", {})
        geom = position.get("observation_geometry", {})
        source_region = position.get("source_region", region)
        final_targets.append(
            {
                "product_id": region.get("product_id", ""),
                "region_id": region_id,
                "final_label": final_label,
                "display_label": display_label,
                "final_confidence": final_conf,
                "classification_source": source,
                "scope_warning": (
                    "Final label is a rule-based water-column candidate type, not a validated trained WCI classifier."
                    if classifier == "rule"
                    else (
                        "Final label uses YOLO-WAL, a WCI-domain fluid/gas-plume detector, when it fires; it is not a fish/submarine/general artificial-object classifier."
                        if classifier == "yolo_wal"
                        else "Final label uses backup UATD forward-looking-sonar detector when it fires, then falls back to rule-based water-column candidate typing."
                    )
                ),
                "model_class_name": best_detection.get("class_name", ""),
                "model_display_label": best_detection.get("display_label", ""),
                "yolo_wal_status": yolo.get("status", "not_run") if classifier == "yolo_wal" else "",
                "yolo_wal_best_confidence": yolo_best if classifier == "yolo_wal" else "",
                "yolo_wal_detections": yolo.get("detections", []) if classifier == "yolo_wal" else [],
                "sonar_yolo_status": yolo.get("status", "not_run"),
                "sonar_yolo_best_confidence": yolo_best,
                "sonar_yolo_detections": yolo.get("detections", []),
                "rule_classification": rule_label,
                "rule_confidence": rule_conf,
                "fish_yolo_status": yolo.get("status", "not_run"),
                "fish_yolo_best_confidence": yolo_best,
                "fish_yolo_detections": yolo.get("detections", []),
                "ping": source_region.get("ping", ""),
                "beam": source_region.get("beam", ""),
                "sample_start": source_region.get("sample_start", ""),
                "sample_end": source_region.get("sample_end", ""),
                "latitude_deg": mp.get("latitude_deg", ""),
                "longitude_deg": mp.get("longitude_deg", ""),
                "depth_m": mp.get("depth_m", ""),
                "easting_m": mp.get("easting_m", ""),
                "northing_m": mp.get("northing_m", ""),
                "height_m": mp.get("height_m", ""),
                "slant_range_m": geom.get("slant_range_m", ""),
                "distance_to_track_m": geom.get("distance_to_track_m", ""),
                "bearing_relative_to_heading_deg": geom.get("bearing_relative_to_heading_deg", ""),
                "side_of_track": geom.get("side_of_track", ""),
                "crop_png": crop.get("crop_png", ""),
                "crop_preview_png": crop.get("preview_png", ""),
                "sonar_yolo_overlay_png": yolo.get("overlay_png", ""),
                "fish_yolo_overlay_png": yolo.get("overlay_png", ""),
            }
        )

    payload = {
        "product_id": regions_payload.get("product_id", ""),
        "pipeline": (
            "active_sonar_rule_candidate_detection_coordinate_projection_v1"
            if classifier == "rule"
            else (
                "active_sonar_candidate_detection_coordinate_projection_yolo_wal_wci_classification_v1"
                if classifier == "yolo_wal"
                else "active_sonar_candidate_detection_coordinate_projection_uatd_yolo_backup_classification_v1"
            )
        ),
        "inputs": {
            "candidate_regions_json": str(product_dir / "candidate_regions.json"),
            "candidate_positions_json": str(product_dir / "candidate_positions.json"),
            "candidate_crop_manifest_json": str(product_dir / "candidate_crop_manifest.json"),
            "yolo_wal_detections_json": str(yolo_wal_path) if classifier == "yolo_wal" else "",
            "sonar_yolo_detections_json": str(sonar_yolo_path) if classifier == "uatd_yolo" else "",
        },
        "method": {
            "candidate_detection": "Python raw Kongsberg .all/.wcd water-column reader -> adaptive MAD threshold -> connected components",
            "coordinate_projection": "Python WCI beam geometry + navigation -> latitude/longitude/depth",
            "target_classification": (
                "Rule-based candidate typing from WCI geometry and intensity features; YOLO disabled by default"
                if classifier == "rule"
                else (
                    "YOLO-WAL WCI fluid/gas-plume detector on exported WCI candidate crops; falls back to rule-based typing when no model hit is present"
                    if classifier == "yolo_wal"
                    else "Backup pretrained UATD sonar YOLO model on exported candidate crops; experimental only for this WCI domain"
                )
            ),
        },
        "limitations": (
            [
                "Current labels are candidate types assigned by WCI rules, not validated trained-model classes.",
                "Formal accuracy requires manually labelled WCI ground truth.",
                "A dedicated WCI classifier requires Zenodo/Echoview-style feature reproduction or WCI-labelled training data.",
            ]
            if classifier == "rule"
            else (
                [
                    "YOLO-WAL covers WCI fluid/gas-plume and FP classes only.",
                    "It does not classify fish, submarine, ROV, mine, or general artificial objects.",
                    "Formal project-specific accuracy still requires labelled WCI validation data.",
                ]
                if classifier == "yolo_wal"
                else [
                "The integrated pretrained model covers UATD forward-looking-sonar classes, not every WCI target.",
                "If the model does not detect a class, the pipeline falls back to rule-based candidate typing.",
                "A dedicated submarine classifier still requires a submarine/ship WCI target dataset and training or fine-tuning.",
                ]
            )
        ),
        "targets": final_targets,
    }
    write_json(product_dir / "final_targets.json", payload)
    write_csv(product_dir / "final_targets.csv", final_targets)

    print(f"Final targets: {len(final_targets)}")
    print(product_dir / "final_targets.json")
    print(product_dir / "final_targets.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
