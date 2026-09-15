# Gig Economy Rideshare Analytics

**Course:** 23CSE301 Machine Learning — Capstone Project (Review 1)

## Team

| Member | Responsible for |
|---|---|
| Member 1 | Dataset & EDA (Section A) · Regression algorithms 1–5 |
| Member 2 | Preprocessing & Feature Engineering (Section B) · Regression algorithms 6–10 · Hyperparameter tuning |
| Member 3 | Classification Track Part A · Regression comparison table · Best-model visualisation · Presentation |

*(Fill in actual names/roll numbers above.)*

## Project Overview

This project focuses on analyzing and predicting outcomes using a gig-economy rideshare dataset. The dataset contains information related to rideshare trips and is used to perform data preprocessing, exploratory analysis, regression, and classification using multiple machine learning algorithms.

The main objective of this project is to preprocess the dataset and compare different machine learning algorithms based on their performance, applied to the **gig economy / rider–driver earnings and burnout** problem track:

- **Regression (numeric target):** `total_earnings` — a driver's total earnings for a given day, predicted from workload and behaviour features aggregated per driver-day.
- **Classification, Part A (categorical target):** `burnout_risk` (Low / Medium / High) — a tier derived from hours driven per day, predicted from the same workload/behaviour features (excluding hours-driven itself, to avoid label leakage).

## Dataset

**Dataset used:** Chicago Taxi Trips Dataset (`chicago_taxi_sample.csv`)

The dataset contains information about rideshare/taxi trips, including trip-related and fare-related attributes.

- **Samples:** 30,000
- **Columns:** 23

Since the raw data is trip-level, the regression and classification targets above are built by **aggregating trips into driver-days** (one row per driver + calendar date) — see the notebook's Section B for the full aggregation logic. Detailed EDA findings (distributions, correlations, skew, class balance) are documented as written observations directly in the notebook's Section A markdown cells.

*(The raw CSV is not committed to this repo due to size — see `data/README.md` for how to obtain it.)*

## Repository Structure

```
/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md          # dataset source + how to obtain chicago_taxi_sample.csv
├── notebooks/
│   └── gig_economy_earnings_burnout_review1.ipynb
├── models/                # saved model files (.pkl via joblib), optional
└── app/                   # GUI / deployment code, only if attempting the bonus
```

## Environment Setup

```bash
git clone https://github.com/Shristi0620/gig-economy-rideshare-analytics.git
cd gig-economy-rideshare-analytics

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## How to Run

1. Obtain `chicago_taxi_sample.csv` (see `data/README.md`) and place it in the `data/` folder.
2. Either copy the CSV next to the notebook, or update `DATA_PATH` in the notebook's first code cell to point to `data/chicago_taxi_sample.csv`.
3. Launch Jupyter:
   ```bash
   jupyter notebook notebooks/gig_economy_earnings_burnout_review1.ipynb
   ```
4. Run **Kernel → Restart & Run All** to execute every cell top-to-bottom.
5. Review the `**Observation:**` markdown cells throughout the notebook — each is filled in with the team's own written interpretation of that section's plot.

## Project Workflow

The project includes the following major steps:

1. Data loading
2. Data preprocessing
3. Handling missing values
4. Handling categorical and numerical features
5. Feature scaling
6. Exploratory data analysis
7. Regression
8. Classification
9. Model evaluation
10. Comparison of machine learning algorithms

### Regression

Multiple regression algorithms are implemented and evaluated to predict the selected numerical target variable (`total_earnings`): Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, and KNN Regressor.

### Classification

Multiple classification algorithms are implemented and evaluated to predict the selected categorical target variable (`burnout_risk`), Part A: Logistic Regression, KNN, Naive Bayes, Decision Tree, and SVM.

## Results

The performance of the implemented machine learning models is evaluated using appropriate evaluation metrics.

### Regression Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-fold cross-validated R² (top 2 models)

**All 10 regression models, ranked by R² — copy from the `reg_comparison` DataFrame in the notebook's Section C2:**

| Algorithm | R² | RMSE | MAE |
|---|---|---|---|
| _(fill in from reg_comparison)_ | | | |

**Best regression model:** _(name)_ — _(one line: why it won, e.g. highest test R² and confirmed by 5-fold CV)_.

### Classification Metrics (Part A)

- Accuracy
- Weighted F1 Score
- Confusion Matrix

**All 5 classification models — copy from the `clf_comparison` DataFrame in the notebook's Section D2:**

| Algorithm | Accuracy | Weighted F1 |
|---|---|---|
| _(fill in from clf_comparison)_ | | |

**Best classifier:** _(name)_ — _(one line justification; note here if `burnout_risk` classes are imbalanced, since that's why weighted F1 is the primary metric rather than accuracy)_.

The detailed results, all visualisations, and model comparisons are available in the Jupyter Notebook (`notebooks/gig_economy_earnings_burnout_review1.ipynb`).

## Academic Integrity

Code scaffolding for the notebook was assisted by Claude (Anthropic). Analysis, interpretation, feature-engineering rationale, and written observations are the team's own, per the course's academic integrity guidelines.
