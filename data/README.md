# Demo Data

This repository includes one reproducible demo line for the YOLO-WAL WCI pipeline:

```text
data/20181112_survey/
  0000_20181112_080147_TecnopescaII.all
  0000_20181112_080147_TecnopescaII.wcd
```

Use this data when verifying the repository after clone:

```powershell
python test.py --survey-id 20181112_survey --all-name 0000_20181112_080147_TecnopescaII.all --product-id yolo_wal_branch_test --threshold-k-mad 3.8 --bottom-guard-samples 4 --max-regions 80
```

The `.all/.wcd` files are tracked with Git LFS. Other survey lines are intentionally excluded because the full raw-data folder is several gigabytes.
