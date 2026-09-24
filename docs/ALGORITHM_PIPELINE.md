# Active Sonar Candidate Detection Pipeline

This package uses a Python runtime pipeline for Kongsberg `.all/.wcd` multibeam water-column data.

The default workflow runs YOLO-WAL after WCI candidate extraction. YOLO-WAL is a WCI-domain detector for `fluide` and `FP` classes, so it is used for fluid/gas-plume detection and false-positive filtering. UATD YOLO is retained only as a backup experiment because it was trained on forward-looking sonar data, not EM2040CD WCI data.

## Default Pipeline

1. `src/active_sonar_pipeline.py`

   Reads raw Kongsberg `.all/.wcd` water-column data, builds the echogram, estimates the bottom, detects anomalous water-column candidates, projects candidate centers to coordinates, and exports render data plus candidate crops.

2. `src/run_yolo_wal_inference.py`

   Runs the vendored YOLOv5 runtime with the YOLO-WAL WCI checkpoint on exported candidate WCI crops. It writes `yolo_wal_detections.json/csv` plus per-candidate overlay images.

3. `src/merge_target_classification.py`

   Merges YOLO-WAL outputs, fallback rule-based candidate types, source indices, and Python-derived coordinates into `final_targets.json` and `final_targets.csv`.

Data requirement: the source line must provide a matching `.all/.wcd` pair. `Coffee_files`, `.mat`, `.dat`, and `.evi` files are not needed.

## Run

```powershell
cd 主动声呐目标识别模型算法包
python test.py --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all --product-id demo_product --threshold-k-mad 3.8 --bottom-guard-samples 4 --max-regions 80
```

For external data:

```powershell
python test.py --data-root D:\sonar_data --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all
```

Rule-only fallback:

```powershell
python test.py --classifier rule --product-id demo_rule_only
```

Backup UATD YOLO experiment:

```powershell
python test.py --classifier uatd_yolo --product-id demo_uatd_yolo_backup --model-conf 0.25 --image-size 640
```

## Key Outputs

Each run writes to:

```text
outputs/<product_id>
```

- `echogram_render_data.json`: render data for the echogram.
- `candidate_overlay.json`: target boxes in echogram coordinates.
- `single_ping_render_data.json`: render data for the WCI fan view.
- `candidate_wci_render_data/<region_id>.json`: target-specific WCI fan render data.
- `candidate_regions.json/csv`: candidate region source indices and rule labels.
- `candidate_positions.json/csv`: latitude, longitude, depth, slant range, cross-track distance, and relative bearing.
- `candidate_crop_manifest.json/csv`: crop paths and source indices.
- `yolo_wal_detections.json/csv`: YOLO-WAL WCI fluid/FP detections on candidate crops.
- `final_targets.json/csv`: final model/rule candidate type and positioning results consumed by the UI.

## Label Meaning

- `fish_school_candidate`: rule-based fish-school-like candidate.
- `gas_plume_candidate`: rule-based gas-plume-like candidate.
- `platform_candidate`: rule-based strong compact side return / structure candidate.
- `unknown_target_candidate`: retained anomalous echo candidate that does not match a stronger rule.
- `gas_plume_candidate`: can also come from YOLO-WAL `fluide` detections.
- `wci_false_positive_candidate`: YOLO-WAL `FP` detections.

YOLO-WAL classes are limited to WCI fluid/gas-plume and FP detection. The package still does not provide a validated trained classifier for fish/submarine/UUV/general artificial-object classes. Formal project-specific accuracy requires labelled WCI ground truth.

## Next Model Route

For a validated trained model, use one of these routes:

1. Keep YOLO-WAL as the WCI fluid/FP detector for plume workflows.
2. If the requirement expands to fish/submarine/UUV/artificial-object classification, collect or obtain WCI-labelled training data and train a dedicated WCI classifier. Do not reuse forward-looking-sonar UATD accuracy as WCI accuracy.
