"""Gig-driver Burnout & Earnings Predictor — Streamlit GUI (Review 2 bonus).

Loads the models saved by Section G of notebooks/regression_classification.ipynb:
  models/burnout_classifier.pkl   -> predicts burnout_risk (Low / Medium / High)
  models/earnings_regressor.pkl   -> predicts total_earnings for the driver-day ($)

Run locally from the repository root:
    streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Find the models folder. Works with the folder named "models" or "models." (as in the GitHub repo),
# whether it sits next to the app folder or inside it.
_HERE = Path(__file__).resolve().parent
MODELS_DIR = next((d for d in [_HERE.parent / "models", _HERE.parent / "models.", _HERE / "models"]
                   if (d / "burnout_classifier.pkl").exists()), _HERE.parent / "models")

FEATURE_LABELS = {
    "trip_count": "Number of trips in the day",
    "total_miles": "Total miles driven",
    "avg_speed_mph": "Average trip speed (mph)",
    "pct_night_trips": "Share of trips at night (10pm–5am)",
    "pct_weekend_trips": "Share of trips on a weekend",
    "num_unique_pickup_areas": "Distinct pickup community areas",
    "num_unique_dropoff_areas": "Distinct drop-off community areas",
    "total_earnings": "Total earnings for the day ($)",
}
SHARE_FEATURES = {"pct_night_trips", "pct_weekend_trips"}
INTEGER_FEATURES = {"trip_count", "num_unique_pickup_areas", "num_unique_dropoff_areas"}


@st.cache_resource
def load_bundle(name):
    return joblib.load(MODELS_DIR / name)


def to_model_input(bundle, values):
    """Scale numeric + one-hot encode categorical exactly as in the notebook (train-fitted transformers)."""
    row = pd.DataFrame([values])
    num = bundle["scaler"].transform(row[bundle["numeric_features"]])
    cat = bundle["encoder"].transform(row[bundle["categorical_features"]])
    return pd.DataFrame(np.hstack([num, cat]), columns=bundle["feature_order"])


def feature_input(feature, ranges, key):
    """One input widget per feature, bounded by the range seen in the training data."""
    r = ranges[feature]
    label = FEATURE_LABELS.get(feature, feature)
    if feature in SHARE_FEATURES:
        return st.slider(label, 0.0, 1.0, float(round(r["median"], 2)), 0.05, key=key)
    if feature in INTEGER_FEATURES:
        hi = max(int(np.ceil(r["max"])) * 2, 2)
        return st.number_input(label, min_value=1, max_value=hi, value=int(round(r["median"])) or 1, step=1, key=key)
    hi = float(np.ceil(r["max"] * 2))
    return st.number_input(label, min_value=0.0, max_value=hi, value=float(round(r["median"], 2)), step=0.5, key=key)


st.set_page_config(page_title="Gig Driver Burnout & Earnings", layout="wide")
st.title("Gig-driver Burnout & Earnings Predictor")
st.caption("23CSE301 ML Capstone — trained on Chicago taxi driver-days. Enter a driver's day and get the models' predictions.")

try:
    burnout = load_bundle("burnout_classifier.pkl")
    earnings = load_bundle("earnings_regressor.pkl")
except FileNotFoundError:
    st.error("Model files not found in `models/`. Run `notebooks/regression_classification.ipynb` top-to-bottom first "
             "(Section G saves them).")
    st.stop()

tab_burnout, tab_earnings, tab_about = st.tabs(["Burnout risk (classification)", "Daily earnings (regression)", "About the models"])

with tab_burnout:
    st.subheader("Predict the burnout-risk tier for a driver-day")
    st.write("Hours driven are **not** an input — the tier is predicted from workload, timing and earnings patterns.")
    left, right = st.columns(2)
    values = {}
    feats = burnout["numeric_features"]
    for i, f in enumerate(feats):
        with (left if i < (len(feats) + 1) // 2 else right):
            values[f] = feature_input(f, burnout["input_ranges"], key=f"b_{f}")
    values["dominant_payment_type"] = st.selectbox("Most common payment type that day", burnout["payment_types"], key="b_pay")

    if st.button("Predict burnout risk", type="primary"):
        X = to_model_input(burnout, values)
        pred = burnout["model"].predict(X)[0]
        proba = pd.Series(burnout["model"].predict_proba(X)[0], index=burnout["model"].classes_)[burnout["class_order"]]
        st.metric("Predicted burnout risk", pred)
        chart_df = pd.DataFrame({"Burnout tier": proba.index, "Probability": proba.values})
        st.bar_chart(chart_df, x="Burnout tier", y="Probability", horizontal=True, sort=False,
                     x_label="Burnout tier", y_label="Predicted probability")  # labels follow x=/y=, not screen axes
        st.caption(f"Model: {burnout['model_name']} (tuned). Probabilities are the model's own estimates.")

with tab_earnings:
    st.subheader("Predict a driver's total earnings for the day")
    left, right = st.columns(2)
    values_r = {}
    feats = earnings["numeric_features"]
    for i, f in enumerate(feats):
        with (left if i < (len(feats) + 1) // 2 else right):
            values_r[f] = feature_input(f, earnings["input_ranges"], key=f"r_{f}")
    values_r["dominant_payment_type"] = st.selectbox("Most common payment type that day", earnings["payment_types"], key="r_pay")

    if st.button("Predict earnings", type="primary"):
        pred = float(earnings["model"].predict(to_model_input(earnings, values_r))[0])
        st.metric("Predicted total earnings", f"${max(pred, 0):,.2f}")
        st.caption(f"Model: {earnings['model_name']}. Typical error on the test set (MAE): "
                   f"${earnings['test_metrics']['MAE']:.2f}.")

with tab_about:
    st.subheader("Test-set performance of the deployed models")
    st.write(f"**Burnout classifier — {burnout['model_name']}**")
    st.table(pd.Series(burnout["test_metrics"], name="Score").round(4))
    st.write(f"**Earnings regressor — {earnings['model_name']}**")
    st.table(pd.Series(earnings["test_metrics"], name="Score").round(4))
    st.info("Inputs outside the range seen in training (a small sample of 2019 Chicago taxi trips) "
            "may give unreliable predictions.")
