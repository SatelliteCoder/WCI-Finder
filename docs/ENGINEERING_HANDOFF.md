# ActiveSonar Engineering Handoff

## Runtime Entry

Use this command for the no-MATLAB WCI candidate pipeline with YOLO-WAL enabled:

```powershell
cd 主动声呐目标识别模型算法包
python test.py --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all --product-id demo_product --threshold-k-mad 3.8 --bottom-guard-samples 4 --max-regions 80
```

Main stages:

1. `src/active_sonar_pipeline.py`
   Reads raw Kongsberg `.all/.wcd` water-column data, detects candidates, estimates coordinates, and exports render/crop data.
2. `src/run_yolo_wal_inference.py`
   Runs YOLO-WAL on WCI candidate crops and writes `yolo_wal_detections.json/csv`.
3. `src/merge_target_classification.py`
   Merges YOLO-WAL detections, candidate geometry, positioning, and fallback rule candidate types into `final_targets.json/csv`.

## Model Registry

Default runtime entry:

```text
models/MODEL_REGISTRY.json -> yolo_wal_wci_fluid_detector
```

The YOLO-WAL weights are under:

```text
models/yolo_wal_wci_fluid_detector/weights/best_combined_gazcogne1.pt
```

This model is a WCI-domain detector for `fluide` and `FP`, not a general fish/submarine/artificial-object classifier.

The UATD YOLO weights remain under:

```text
models/active_sonar_target_classifier_uatd_yolov8/weights/
```

They are not used by default because that model is trained on forward-looking sonar imagery, not EM2040CD WCI data. It can be run manually with `--classifier uatd_yolo`.

## Data Requirement

The Python runtime does not call MATLAB and does not require CoFFee cache files. Put raw Kongsberg data pairs under:

```text
data/<survey_id>/<line_name>.all
data/<survey_id>/<line_name>.wcd
```

The delivery zip does not include large raw data by default. Use `--data-root` if the host system keeps data outside the package.

## Product Contract

The current package provides:

- raw `.all/.wcd` parsing
- echogram generation
- anomalous WCI candidate detection
- YOLO-WAL WCI fluid/FP detection on candidate crops
- fallback candidate type assignment by WCI geometry/intensity rules
- coordinate projection
- JSON/CSV outputs for UI display

It does not provide a validated trained classifier for fish/submarine/UUV/artificial-object classes. That requires WCI-labelled training data and a model validation set.

## Key Outputs

Each run writes to:

```text
outputs/<product_id>
```

Important files:

- `product.json`
- `echogram_render_data.json`
- `candidate_overlay.json`
- `single_ping_render_data.json`
- `candidate_wci_render_data/<region_id>.json`
- `candidate_regions.json/csv`
- `candidate_positions.json/csv`
- `candidate_crop_manifest.json/csv`
- `yolo_wal_detections.json/csv`
- `final_targets.json/csv`
