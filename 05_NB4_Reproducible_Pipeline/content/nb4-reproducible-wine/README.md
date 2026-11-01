# NB4 · Reproducible Wine Classification Pipeline

Project developed for Session 5 of **Research Methods and Scientific Integrity in AI and Advanced Technologies**.

## Objective

Demonstrate a reproducible Machine Learning workflow using:

- a versioned dataset;
- a dataset fingerprint;
- random seeds;
- a leakage-free Pipeline;
- cross-validation;
- MLflow;
- Git;
- pinned dependencies;
- clear reproduction instructions.

## Dataset

Wine Dataset included in `scikit-learn`.

- Rows: 178
- Predictive features: 13
- Classes: 3
- MD5: `2925c4a8cd201c79d97d3d283429ef23`

## Pipeline

`StandardScaler -> LogisticRegression`

The preprocessing step is placed inside the Pipeline so the scaler is fitted only on the training data of each fold.

## Evaluation

5-fold `StratifiedKFold` with `shuffle=True`.

Seeds evaluated: `0, 1, 2, 3, 42`.

### Overall Results

- Accuracy: **0.9810 ± 0.0031**
- Macro F1: **0.9807 ± 0.0027**

Detailed results are stored in `results_summary.csv`.

## How to Reproduce

```bash
git clone <YOUR-REPOSITORY-URL>
cd nb4-reproducible-wine
pip install -r requirements.txt
PYTHONHASHSEED=0 python src/train.py --seed 42
```

## Expected Output with Seed 42

```json
{"seed": 42, "data_md5": "2925c4a8cd201c79d97d3d283429ef23", "accuracy_mean": 0.98333, "accuracy_sd": 0.01521, "f1_macro_mean": 0.9829, "f1_macro_sd": 0.01561}
```

## Environment Used

- Python: 3.13.16
- Dependencies: see `requirements.txt`
- Dataset MD5: `2925c4a8cd201c79d97d3d283429ef23`

## Reproducibility

Two runs with the same seed should produce exactly the same output.
In addition, the five experiments make it possible to report performance as mean ± standard deviation instead of relying on a single lucky run.
