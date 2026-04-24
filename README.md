# Classical-ML-project

A comprehensive, reusable ML pipeline for **binary classification** built with scikit-learn, XGBoost, LightGBM, and Optuna.  
The notebook walks through exploratory data analysis, data cleaning, feature engineering, and trains **13 different algorithms** (logistic regression, trees, SVM, ensemble methods, neural nets) with hyperparameter sweeps. It includes rich visualisations (ROC curves, feature importance, confusion matrices) and statistical model-comparison analysis.

---

## Features

| Section | What it covers |
|---------|----------------|
| **EDA** | Class distribution, feature distributions by class, correlation heatmap, box-plots |
| **Preprocessing** | Median/mode imputation, label encoding, stratified train/test split, StandardScaler |
| **Feature Engineering** | ANOVA F-score selection (SelectKBest), PCA 2-D visualisation, cumulative variance plot |
| **13 Classifiers** | Logistic Regression, Decision Tree, Random Forest, Extra Trees, Bagging, Gradient Boosting, AdaBoost, XGBoost, LightGBM, SVM, KNN, Naive Bayes, MLP Neural Network |
| **Hyperparameter Tuning** | `RandomizedSearchCV` for 5 key models + **Optuna** Bayesian optimisation for Random Forest |
| **Evaluation** | 5-fold stratified CV · Accuracy · Precision · Recall · F1 · ROC-AUC |
| **Visualisations** | Comparative ROC curves, performance bar chart, top-6 confusion matrices, feature importance (6 tree models), LR coefficients |
| **Statistical Analysis** | Wilcoxon signed-rank pairwise p-value matrix, mean CV-rank chart, Youden-J optimal threshold |
| **Save artefacts** | Best model + scaler + results CSV saved to `outputs/` via `joblib` |

---

## Algorithms Covered

| # | Algorithm | Family |
|---|-----------|--------|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Tree |
| 3 | Random Forest | Ensemble – Bagging |
| 4 | Extra Trees | Ensemble – Bagging |
| 5 | Bagging Classifier | Ensemble – Bagging |
| 6 | Gradient Boosting | Ensemble – Boosting |
| 7 | AdaBoost | Ensemble – Boosting |
| 8 | XGBoost | Ensemble – Boosting |
| 9 | LightGBM | Ensemble – Boosting |
| 10 | Support Vector Machine (RBF) | Kernel |
| 11 | K-Nearest Neighbours | Instance-based |
| 12 | Gaussian Naive Bayes | Probabilistic |
| 13 | MLP Neural Network | Neural Network |

---

## Requirements

**Python 3.10+**

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries:

- `scikit-learn >= 1.3`
- `xgboost >= 2.0`
- `lightgbm >= 4.0`
- `optuna >= 3.3`
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

> ⚠ Training 13 models with cross-validation and hyperparameter tuning requires **medium-to-high RAM** (8 GB recommended). Reduce `N_ITER` or `CV_FOLDS` in the notebook to lower resource usage.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/anergaratea/Classical-ML-project.git
cd Classical-ML-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook scikit_ML_Pipeline_Binary_Notebook.ipynb
```

Run cells top-to-bottom. The notebook uses scikit-learn's built-in **breast cancer** dataset by default.

### Adapt to Your Own Dataset

1. Replace the data-loading block in **Section 1** with:
   ```python
   df = pd.read_csv("your_data.csv")
   TARGET_COL = "your_target_column"
   ```
2. Ensure the target column contains **binary values** (0/1 or two distinct labels).
3. Re-run all cells.

---

## Output Files

After running the full notebook, an `outputs/` directory is created containing:

| File | Description |
|------|-------------|
| `best_model.pkl` | Best-performing model serialised with `joblib` |
| `scaler.pkl` | Fitted `StandardScaler` for inference |
| `test_results.csv` | Test-set metrics for all 13 models |

---

## Extension Ideas

1. **Multiclass / regression** – swap classifiers for `MultiOutputClassifier` or regressors.
2. **More algorithms** – add CatBoost, ensemble stacking (`StackingClassifier`).
3. **MLflow tracking** – log metrics and artefacts with `mlflow.sklearn.log_model`.
4. **Domain features** – plug in text (TF-IDF) or image (CNN embeddings) feature extractors.
5. **Parallelism / early stopping** – use Dask or early-stopping callbacks in XGBoost/LightGBM.

---

## Difficulty

**Advanced.** The pipeline involves many algorithms, rigorous cross-validation, Bayesian hyperparameter optimisation, and statistical model comparison. Some knowledge of ML evaluation metrics is assumed.

