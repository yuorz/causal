
# Simple FS ML

## Layout

```
simple_fs_ml/
├── cli.py
├── README.md
├── logs/        # metrics.ndjson only
└── simple_fs_ml/
  ├── __init__.py
  ├── data.py      # load/split/save CSV
  ├── features.py    # exclude + selectors + one-hot prep
  ├── models.py      # PyTorch models + training loop
  └── utils.py     # seed + ndjson writer
```

## Install

```bash
pip install -U pip
pip install pandas numpy scikit-learn torch tqdm
```

## Split (AmesHousing.csv example, dataset should be ',' separated .csv file)

```bash
python cli.py split --dataset ./data/AmesHousing.csv --train-name AmesHousing_train.csv --valid-name AmesHousing_valid.csv
```

## Train (Regression: `SalePrice`)

```bash
python cli.py train --dataset ./data/AmesHousing.csv --task regression --target SalePrice --exclude Order PID --method corr --ratio 0.3 --model mlp --log-dir logs
```

## Train (Classification: a categorical target, e.g., `ExterQual`)

```bash
python cli.py train --dataset ./data/AmesHousing.csv --task classification --target ExterQual --exclude Order PID --method random --ratio 0.2 --model linear --log-dir logs
```

### Extending

- **More selectors**: add a function to `features.py` and register in `FEATURE_SELECTORS`.
- **More models**: implement a class and add to `MODEL_REGISTRY` in `models.py`.
  - For non-PyTorch models (e.g., SVM/XGBoost), you can write a small adapter that exposes the same `forward`/training steps or create a parallel training function.

## Output

`logs/metrics.ndjson` will contain one JSON line per run, e.g.:
```json
{"params": {...}, "metrics": {"rmse": 22144.23}}
```
or
```json
{"params": {...}, "metrics": {"auroc": 0.8931}}
```
