# 主动声呐目标识别模型算法包

## 目录结构

```text
src/
  active_sonar_pipeline.py              原始 .all/.wcd 读取、候选目标检测、坐标定位、渲染数据导出
  run_yolo_wal_inference.py             YOLO-WAL WCI 流体/伪影检测脚本，默认运行
  run_sonar_yolo_inference.py           UATD YOLO 备用实验脚本，默认不运行
  merge_target_classification.py        合并模型结果、候选类型和定位结果

models/
  MODEL_REGISTRY.json
  yolo_wal_wci_fluid_detector/
    weights/
      best_combined_gazcogne1.pt
  active_sonar_target_classifier_uatd_yolov8/
    model_config.yaml
    weights/
      active_sonar_target_classifier_best.pt

test.py                                 单文件启动器（候选检测/定位 → YOLO-WAL/规则/UATD → 合并结果）

data/
  README.md
  20181112_survey/                     标准样例输入数据，仅保留 0000 测线 .all/.wcd

requirements.txt
```

## 输入要求

当前交付数据只保留原始 `.all/.wcd` 文件，不包含 `Coffee_files`、`.mat`、`.dat` 或 `.evi` 中间文件。

开发目录中可放置原始示例测线：

```text
data/20181112_survey/
  0000_20181112_080147_TecnopescaII.all
  0000_20181112_080147_TecnopescaII.wcd

data/20190724_survey/
  0000_20190724_071809_TecnopescaII.all
  0000_20190724_071809_TecnopescaII.wcd
  ...
```

仓库内只保留一组可复现实验的标准样例数据：

```text
data/20181112_survey/
  0000_20181112_080147_TecnopescaII.all
  0000_20181112_080147_TecnopescaII.wcd
```

这组数据用于验证 YOLO-WAL 集成后的完整流程。`.all/.wcd` 通过 Git LFS 管理，clone 后如未自动拉取大文件，请执行：

```bash
git lfs pull
```

甲方使用自己的数据时，按以下结构把 `.all/.wcd` 数据放到算法包内部 `data/` 目录，或通过 `--data-root` 指向外部数据目录：

```text
data/<survey_id>/
  <line_name>.all
  <line_name>.wcd
```

说明：当前 Python 处理链路已经内置 Kongsberg 原始数据解析，不需要 `Coffee_files/fData.mat`。放入同名 `.all/.wcd` 数据对后即可直接运行。

## 安装依赖

```bash
cd 主动声呐目标识别模型算法包
python -m pip install -r requirements.txt
```

## 运行

默认运行 YOLO-WAL WCI 模型。该模型是多波束 WCI 领域模型，用于检测 `fluide`（流体/气体羽流候选）和 `FP`（伪影/假阳性候选）。它不是鱼、潜艇、人工物体多类别分类器。

UATD YOLO 仍随包保留，但来自前视声呐数据集，和当前 EM2040CD 多波束水柱 WCI 数据不匹配，只作为备用实验模型。

PowerShell（推荐写成一行，避免续接符问题）：

```powershell
cd 主动声呐目标识别模型算法包
python test.py --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all --product-id demo_product --threshold-k-mad 3.8 --bottom-guard-samples 4 --max-regions 80
```

也可以直接用默认参数运行（等价于上面这条）：

```powershell
python test.py --product-id demo_product
```

仓库随附的标准输出位于：

```text
outputs/yolo_wal_branch_test/
```

它由以下命令生成：

```powershell
python test.py --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all --product-id yolo_wal_branch_test --threshold-k-mad 3.8 --bottom-guard-samples 4 --max-regions 80
```

预期高层结果：12 个候选区域，YOLO-WAL 检出 2 个 `fluide`，最终输出 12 条 `final_targets`。

PowerShell 多行写法（行尾用反引号 `` ` `` 续接，注意反引号后不能有空格）：

```powershell
python test.py `
  --survey-id 20181112_survey `
  --all-name 0000_20181112_080147_TecnopescaII.all `
  --product-id demo_product `
  --threshold-k-mad 3.8 --bottom-guard-samples 4 `
  --max-regions 80
```

上面这条命令使用的是算法包内部相对路径：

```text
.\data\20181112_survey\
.\outputs\demo_product\
```

如果总系统已经有自己的原始数据目录，也可以额外传 `--data-root` 指向外部数据目录。

只跑规则候选检测与定位，不运行深度学习模型：

```powershell
python test.py --product-id demo_rule_only --classifier rule
```

如果需要复现旧的 UATD YOLO 备用实验，需要显式传入：

```powershell
python test.py --product-id demo_uatd_yolo_backup --classifier uatd_yolo --model-conf 0.25 --image-size 640
```

该结果不能作为当前 WCI 数据的正式分类精度依据。

## 输出

输出目录：

```text
outputs/<ProductId>/
```

关键结果：

```text
final_targets.json                      候选目标类型与定位结果
final_targets.csv
yolo_wal_detections.json                YOLO-WAL WCI 流体/伪影检测结果
yolo_wal_detections.csv
echogram_render_data.json               回波图渲染数据
candidate_overlay.json                  目标检测框数据
candidate_wci_render_data/<region>.json 每个目标对应的 WCI 扇形图数据
track_points.csv                        测线轨迹
```
