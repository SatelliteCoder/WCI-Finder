# ActiveSonar Engineering Handoff

## Runtime Entry

Use this command for the full no-MATLAB classification pipeline:

```powershell
cd E:\Underwater-project\Act_Sonar
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_classification_pipeline.ps1 `
  -SurveyId "20190724_survey" `
  -AllName "0001_20190724_072738_TecnopescaII.all" `
  -ThresholdKMad 3.8 `
  -BottomGuardSamples 4 `
  -MaxRegions 80 `
  -ModelConf 0.25
```

Main stages:

1. `src/active_sonar_pipeline.py`
   Reads raw Kongsberg `.all/.wcd` water-column data, detects candidates, estimates coordinates, and exports render/model-input data.
2. `src/run_sonar_yolo_inference.py`
   Runs the default YOLO sonar target classifier.
3. `src/merge_target_classification.py`
   Merges candidate geometry, positioning, and model classification into `final_targets.json/csv`.

## Model Layout

Canonical runtime model directory:

```text
model/active_sonar_classifiers
```

Default model:

```text
model/active_sonar_classifiers/uatd_yolov8n/weights/best.pt
```

Model registry:

```text
model/active_sonar_classifiers/model_registry.json
```

Third-party source/reference copy:

```text
model/Sonar-Threat-Detection-YOLO
```

Legacy fish-only model set, not used by the default full pipeline:

```text
model/FishDetectionAI
```

## Data Requirement

The current Python runtime does not call MATLAB and does not require CoFFee cache files. Put raw Kongsberg data pairs under:

```text
data/<survey_id>/<line_name>.all
data/<survey_id>/<line_name>.wcd
```

The package data directory should contain only `.all/.wcd` files. `.mat`, `.dat`, `.evi`, and `Coffee_files` are not required.

## Equivalence To Previous MATLAB Pipeline

The Python version is functionally equivalent at the product-contract level:

- same high-level pipeline: raw WCI parsing -> echogram candidate detection -> coordinate projection -> WCI crops -> YOLO classification -> final target JSON/CSV
- same output contract for the YuanTing UI
- same default YOLO classification model

It is not a line-by-line or bit-exact MATLAB clone:

- MATLAB previously used CoFFee helper functions directly.
- Python now reads Kongsberg `.all/.wcd` directly and reimplements the needed detection/projection/export logic.
- Candidate ordering/counts can differ because bottom estimation, connected-component processing, and fallback navigation projection are Python implementations.

For delivery, validate representative survey lines before acceptance and keep output comparison records.

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
- `sonar_yolo_detections.json/csv`
- `final_targets.json/csv`
