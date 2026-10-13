# Gig Economy Rideshare Analytics

## Project Overview

This project focuses on analyzing and predicting outcomes using a gig-economy rideshare dataset. The dataset contains information related to rideshare trips and is used to perform data preprocessing, exploratory analysis, regression, and classification using multiple machine learning algorithms.

The main objective of this project is to preprocess the dataset and compare different machine learning algorithms based on their performance.

##Dataset

Dataset used:  chicago_taxi_sample.csv

Chicago Taxi Trips Dataset

The dataset contains information about rideshare/taxi trips, including trip-related and fare-related attributes.

The dataset used for this project contains 30,000 samples and 23 columns.
Gig drivers' earnings and fatigue depend on how they work: how many trips, how far, how fast, at night or on weekends.
Regression: predict a driver's 'total_earnings' for a day.
Classification: predict 'burnout_risk' ∈ {Low, Medium, High} (tertiles of hours driven that day) without
using hours driven as an input.
Clustering: discover natural groups of driver-days without labels, then compare them with 'burnout_risk' after fitting.

## Project Review

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

Multiple regression algorithms are implemented and evaluated to predict the selected numerical target variable.

### Classification

Multiple classification algorithms are implemented and evaluated to predict the selected categorical target variable.

## Results

The performance of the implemented machine learning models is evaluated using appropriate evaluation metrics.

### Regression Metrics


  | Model | MAE | RMSE | R² Score |
  | :--- | :--- | :--- | :--- |
  | Linear Regression | 2.14 | 3.05 | 0.81 |
  | Random Forest | 1.22 | 1.89 | 0.93 |
  | XGBoost | 1.10 | 1.74 | 0.95 |

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

  3. Results summary (sample dataset, 80:20 split, `random_state=42`)
Regression — 10 algorithms (Review 1), ranked by test R²
Algorithm	R²	RMSE ($)	MAE ($)
Random Forest Regressor	0.828	6.45	3.80
Gradient Boosting Regressor	0.815	6.69	4.26
Decision Tree Regressor	0.799	6.97	4.80
Polynomial Regression (deg=2)	0.657	9.11	5.55
Ridge Regression	0.641	9.31	5.73
Linear Regression	0.641	9.31	5.74
ElasticNet Regression	0.641	9.31	5.72
Lasso Regression	0.639	9.34	5.68
Support Vector Regressor	0.609	9.72	5.03
KNN Regressor	0.529	10.67	6.39
5-fold CV R² (top 2): Random Forest 0.709 ± 0.098, Gradient Boosting 0.745 ± 0.075.
(Polynomial degree 3 was also tried: R² = −57.3, so it overfits badly.)
Classification — all 10 algorithms (Review 1 Part A + Review 2 Part B), ranked by weighted F1
Rank	Algorithm	Accuracy	Precision (w)	Recall (w)	F1 (w)	ROC-AUC (OvR)
1	Support Vector Machine	0.818	0.818	0.818	0.818	0.949
2	MLP Classifier	0.806	0.807	0.806	0.806	0.933
3	Logistic Regression	0.800	0.800	0.800	0.800	0.940
4	Gradient Boosting Classifier	0.788	0.788	0.788	0.788	0.938
5	AdaBoost Classifier	0.753	0.760	0.753	0.755	0.889
6	Random Forest Classifier	0.747	0.742	0.747	0.743	0.923
7	Bagging Classifier (Decision Tree)	0.729	0.730	0.729	0.730	0.912
8	Decision Tree Classifier	0.712	0.704	0.712	0.706	0.805
9	KNN Classifier	0.700	0.698	0.700	0.699	0.850
10	Gaussian Naive Bayes	0.665	0.674	0.665	0.634	0.873
Final model: Support Vector Machine. 5-fold CV weighted F1: SVM 0.847 ± 0.021 vs MLP 0.808 ± 0.033.
Tuning with `GridSearchCV` (best `C=100`, linear kernel) compared with sklearn defaults (`C=1`, RBF):
	Default	Tuned	Change
CV weighted F1	0.727	0.850	+0.122
Test accuracy	0.753	0.818	+0.065
Test weighted F1	0.756	0.817	+0.062
Test ROC-AUC (OvR)	0.899	0.950	+0.051
Clustering (Review 2) — k = 5 chosen from the K-Means elbow
Algorithm	Split	Silhouette ↑	Davies–Bouldin ↓	Calinski–Harabasz ↑
K-Means	train (fitted)	0.222	1.395	151.5
K-Means	test (assigned)	0.226	1.294	41.3
Agglomerative (Ward)	train (fitted)	0.232	1.476	131.2
Agglomerative (Ward)	test (assigned)	0.230	1.326	37.6
Ward was selected from {ward, complete, average, single}. Average and single linkage put more than 75% of points in one
cluster, so they were ruled out. Agreement with `burnout_risk`, measured after fitting: K-Means ARI 0.053 / NMI 0.069;
Agglomerative ARI 0.014 / NMI 0.037.
[Add your own 2–3 sentence interpretation of these results here.]
4. Repository structure
```
README.md                 this file
requirements.txt          pinned dependencies (tested versions)
data/                     chicago_taxi_sample.csv
notebooks/
  regression_classification.ipynb   Review 1 (EDA, preprocessing, regression, classification Part A)
                                     + Review 2 (classification Part B, 10-model table, final model, model export)
  clustering.ipynb                   Review 2 clustering track (K-Means, Agglomerative)
  *.png / *.csv                      figures and result tables written by the notebooks
models/                   burnout_classifier.pkl, earnings_regressor.pkl (written by Section G of the notebook)
app/
  app.py                  Streamlit GUI (bonus)
  requirements.txt        minimal dependencies for deployment
```
5. Environment setup
Tested with Python 3.11 (matches the Python version on your PC and the one to choose when deploying).
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
If an exact pin is unavailable for your Python version, install the nearest versions. Then re-run the classification
notebook so `models/*.pkl` are re-saved with your scikit-learn version, because a pickled model must be loaded with the
same scikit-learn version that saved it.
6. How to run
From the repository root:
```bash
# 1) Regression + classification (≈ 4 min on the sample). Also saves models/*.pkl for the app.
jupyter nbconvert --to notebook --execute --inplace notebooks/regression_classification.ipynb

# 2) Clustering (≈ 30 s on the sample)
jupyter nbconvert --to notebook --execute --inplace notebooks/clustering.ipynb

# 3) GUI (opens http://localhost:8501)
streamlit run app/app.py
```
You can also open the notebooks in Jupyter or VS Code and use Restart & Run All. They work when started from
`notebooks/` or from the repository root.
7. GUI and public deployment
`app/app.py` has three tabs:
Burnout risk: inputs → predicted tier + class probabilities from the tuned SVM.
Daily earnings: inputs → predicted $ from the Random Forest Regressor.
About the models: test-set metrics for both.
Deploy on Streamlit Community Cloud (free):
Push this repository to GitHub (public), including `models/*.pkl`.
Go to https://share.streamlit.io, sign in with GitHub, then choose Create app → Deploy a public app from GitHub.
Repository: your repo. Branch: `main`. Main file path: `app/app.py`.
Advanced settings → Python version: 3.11 (the same version you ran the notebook with), so the pinned scikit-learn can load the models. Click Deploy.
Paste the public URL here: Live app: [https://….streamlit.app]

8. Use of AI tools (guideline 7.5)
Generative AI (Claude, by Anthropic) was used for code scaffolding of the Review 2 sections (classification Part B,
the clustering notebook, the Streamlit app) and for drafting this README's structure. All analysis and interpretation
(the `Observation` cells, the final-model justification and the conclusions) was written by the team.



