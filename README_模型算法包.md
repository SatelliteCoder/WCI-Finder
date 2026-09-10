# 主动声呐目标识别模型算法包

## 目录结构

```text
src/
  active_sonar_pipeline.py              原始 .all/.wcd 读取、候选目标检测、坐标定位、渲染数据导出
  run_sonar_yolo_inference.py           YOLO 主动声呐目标分类
  merge_target_classification.py        合并候选、定位和模型分类结果

models/
  MODEL_REGISTRY.json
  active_sonar_target_classifier_uatd_yolov8/
    model_config.yaml
    weights/
      active_sonar_target_classifier_best.pt

test.py                                 单文件启动器（候选检测 → YOLO 分类 → 结果合并）

data/
  20181112_survey/                     随包原始示例数据，仅保留 .all/.wcd
  20190724_survey/                     随包原始示例数据，仅保留 .all/.wcd

requirements.txt
```

## 输入要求

当前交付数据只保留原始 `.all/.wcd` 文件，不包含 `Coffee_files`、`.mat`、`.dat` 或 `.evi` 中间文件。

算法包已内置原始示例测线：

```text
data/20181112_survey/
  0000_20181112_080147_TecnopescaII.all
  0000_20181112_080147_TecnopescaII.wcd

data/20190724_survey/
  0000_20190724_071809_TecnopescaII.all
  0000_20190724_071809_TecnopescaII.wcd
  ...
```

如果替换自己的数据，也按以下结构放到算法包内部 `data/` 目录：

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

PowerShell（推荐写成一行，避免续接符问题）：

```powershell
cd 主动声呐目标识别模型算法包
python test.py --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all --product-id demo_product --threshold-k-mad 3.8 --bottom-guard-samples 4 --max-regions 80 --model-conf 0.25
```

也可以直接用默认参数运行（等价于上面这条）：

```powershell
python test.py --product-id demo_product
```

PowerShell 多行写法（行尾用反引号 `` ` `` 续接，注意反引号后不能有空格）：

```powershell
python test.py `
  --survey-id 20181112_survey `
  --all-name 0000_20181112_080147_TecnopescaII.all `
  --product-id demo_product `
  --threshold-k-mad 3.8 --bottom-guard-samples 4 `
  --max-regions 80 --model-conf 0.25
```

上面这条命令使用的是算法包内部相对路径：

```text
.\data\20181112_survey\
.\outputs\demo_product\
.\models\active_sonar_target_classifier_uatd_yolov8\weights\active_sonar_target_classifier_best.pt
```

如果总系统已经有自己的原始数据目录，也可以额外传 `--data-root` 指向外部数据目录。

## 输出

输出目录：

```text
outputs/<ProductId>/
```

关键结果：

```text
final_targets.json                      最终目标分类与定位结果
final_targets.csv
echogram_render_data.json               回波图渲染数据
candidate_overlay.json                  目标检测框数据
candidate_wci_render_data/<region>.json 每个目标对应的 WCI 扇形图数据
track_points.csv                        测线轨迹
```

