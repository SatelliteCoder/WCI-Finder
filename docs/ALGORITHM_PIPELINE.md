# Active Sonar Target Classification Pipeline

This project now uses a Python runtime pipeline. The echogram is used only for fast candidate screening. The downstream classifier uses candidate-local single-ping WCI crops whenever they are available.

## Model

Default pretrained model:

```text
model/active_sonar_classifiers/uatd_yolov8n/weights/best.pt
```

Source code and weights:

```text
https://github.com/lMelkorl/Sonar-Threat-Detection-YOLO
```

The model is a YOLOv8 UATD sonar target detector. Its pretrained classes are:

```text
ball, circle cage, cube, cylinder, human body, metal bucket, plane, rov, square cage, tyre
```

It is not a dedicated submarine model. Submarine or vessel-specific classification still requires a matching labelled dataset and fine-tuning.

## Pipeline

1. `src/active_sonar_pipeline.py`

   Python reads raw Kongsberg `.all/.wcd` water-column data, builds the echogram, estimates the bottom, detects anomalous water-column candidates, projects candidate centers to coordinates, and exports both render data and ML input crops. It does not start MATLAB and does not require `.mat` files.

2. `src/run_sonar_yolo_inference.py`

   Python + Ultralytics YOLO runs the UATD sonar target model. It uses `ml_input_png` from `candidate_crop_manifest.json`; this points to a candidate-local WCI crop when available.

3. `src/merge_target_classification.py`

   Python merges rule candidates, Python-derived coordinates, WCI crop paths, and YOLO model results into `final_targets.json` and `final_targets.csv`.

Data requirement: the source line must provide a matching `.all/.wcd` pair. `Coffee_files`, `.mat`, `.dat`, and `.evi` files are not needed.

## Run

```powershell
cd E:\Underwater-project\Act_Sonar
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_classification_pipeline.ps1 `
  -SurveyId "20181112_survey" `
  -AllName "0000_20181112_080147_TecnopescaII.all" `
  -ThresholdKMad 3.8 `
  -BottomGuardSamples 4 `
  -MaxRegions 80 `
  -ModelConf 0.25
```

For another survey line, change `-SurveyId` and `-AllName`.

## Key Outputs

Each run writes to:

```text
E:\Underwater-project\Act_Sonar\outputs\<product_id>
```

- `echogram_render_data.json`: ECharts render data for the echogram.
- `candidate_overlay.json`: target boxes in echogram coordinates.
- `single_ping_render_data.json`: ECharts render data for the WCI fan view.
- `candidate_wci_render_data/<region_id>.json`: target-specific WCI fan render data. The YuanTing UI should load this file when the selected target changes.
- `candidate_regions.json/csv`: candidate region source indices.
- `candidate_positions.json/csv`: latitude, longitude, depth, slant range, cross-track distance, and relative bearing computed in Python from raw navigation, attitude, and WCI beam geometry.
- `candidate_crop_manifest.json/csv`: crop paths and source indices. `ml_input_png` is the default model input.
- `candidate_crops/*_wci_crop.png`: candidate-local WCI crops used by the sonar YOLO classifier.
- `sonar_yolo_detections.json/csv`: raw model detections and model class names.
- `final_targets.json/csv`: final target classification and positioning results consumed by the YuanTing UI.

## Label Meaning

- `artificial_object_candidate`: model-detected artificial object candidate, currently mapped from UATD `plane`.
- `cage_candidate`: model-detected cage target, mapped from `circle cage` or `square cage`.
- `debris_or_artificial_object_candidate`: model-detected debris/artificial object, mapped from `metal bucket` or `tyre`.
- `geometric_target_candidate`: model-detected geometric target, mapped from `ball`, `cube`, or `cylinder`.
- `human_like_target_candidate`: model-detected human-body-like target.
- `rov_candidate`: model-detected ROV target.
- `fish_rule_candidate_unverified`: rule-based fish-school-like candidate, not confirmed by the UATD model.
- `gas_plume_candidate`: rule-based natural-gas plume candidate.
- `platform_candidate`: rule-based strong compact side return.
- `unknown_target_candidate`: retained anomalous echo candidate that does not match a stronger rule or model class.

## Positioning Chain

The classification crop is only the ML input. Defensible positioning remains tied to the raw water-column indices:

```text
product_id -> region_id -> ping -> beam -> sample/range -> Python WCI projection -> latitude/longitude/depth
```
