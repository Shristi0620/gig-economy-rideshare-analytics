"""Shift Check — gig-driver burnout & earnings predictor (Streamlit GUI, Review 2 bonus).

Loads the models saved by Section G of notebooks/regression_classification.ipynb:
  models/burnout_classifier.pkl   -> predicts burnout_risk (Low / Medium / High)
  models/earnings_regressor.pkl   -> predicts total_earnings for the driver-day ($)

Run locally from the repository root:
    python -m streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ----------------------------------------------------------------------------- design tokens
ASPHALT = "#1F2933"      # header band, primary text
OVERCAST = "#EEF1F4"     # page background
CURB = "#FFFFFF"         # input panels
LANE = "#F2B705"         # lane-line amber, used only as an accent stripe
INK_MUTED = "#52606D"    # secondary text
TIER_COLORS = {"Low": "#2E8B7A", "Medium": "#C98A04", "High": "#C2412D"}
TIER_MEANING = {
    "Low": "A light day. Workload and timing look like a short, low-strain shift.",
    "Medium": "A moderate day. Some signs of a longer shift; worth keeping an eye on breaks.",
    "High": "A heavy day. The pattern matches the longest shifts in the data, where fatigue risk is highest.",
}

FEATURE_LABELS = {
    "trip_count": "Trips completed",
    "total_miles": "Miles driven",
    "avg_speed_mph": "Avg speed (mph)",
    "pct_night_trips": "Share of trips at night (10pm–5am)",
    "pct_weekend_trips": "Share of trips on a weekend",
    "num_unique_pickup_areas": "Pickup areas visited",
    "num_unique_dropoff_areas": "Drop-off areas visited",
    "total_earnings": "Earnings for the day ($)",
}
FEATURE_HELP = {
    "avg_speed_mph": "Low speed usually means heavy traffic or waiting time.",
    "pct_night_trips": "0 = no night trips, 1 = every trip at night.",
    "pct_weekend_trips": "0 = weekday only, 1 = every trip on Saturday/Sunday.",
    "num_unique_pickup_areas": "Number of different Chicago community areas the driver picked up in.",
    "num_unique_dropoff_areas": "Number of different Chicago community areas the driver dropped off in.",
}
GROUPS = [
    ("Workload", ["trip_count", "total_miles", "avg_speed_mph"]),
    ("Timing", ["pct_night_trips", "pct_weekend_trips"]),
    ("Area covered", ["num_unique_pickup_areas", "num_unique_dropoff_areas"]),
    ("Money", ["total_earnings"]),
]
SHARE_FEATURES = {"pct_night_trips", "pct_weekend_trips"}
INTEGER_FEATURES = {"trip_count", "num_unique_pickup_areas", "num_unique_dropoff_areas"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, label, input, button, textarea {{
    font-family: 'Barlow', 'Segoe UI', sans-serif;
}}
.stApp {{ background: {OVERCAST}; color: {ASPHALT}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.5rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', 'Segoe UI', sans-serif; color: {ASPHALT}; letter-spacing: 0.01em; }}

/* Header band: asphalt with a dashed lane line underneath */
.road {{
    background: {ASPHALT}; color: #F5F7FA; border-radius: 14px;
    padding: 1.6rem 2rem 1.9rem; position: relative; overflow: hidden; margin-bottom: 1.4rem;
}}
.road::after {{
    content: ""; position: absolute; left: 2rem; right: 2rem; bottom: 0.85rem; height: 4px;
    background: repeating-linear-gradient(90deg, {LANE} 0 34px, transparent 34px 58px);
}}
.road h1 {{ color: #FFFFFF; font-size: 2.9rem; line-height: 1.0; margin: 0 0 0.4rem; font-weight: 700; }}
.road p {{ color: #CBD2D9; font-size: 1.05rem; margin: 0; max-width: 62ch; }}

/* Input groups */
.group-title {{
    font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.25rem;
    color: {ASPHALT}; border-left: 4px solid {LANE}; padding-left: 0.55rem; margin: 0.3rem 0 0.6rem;
}}
div[data-testid="stForm"] {{ background: {CURB}; border: 1px solid #D9DEE4; border-radius: 14px; padding: 1.3rem 1.4rem; }}
.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
    background: {ASPHALT}; color: #FFFFFF; border: none; border-radius: 10px;
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.15rem; font-weight: 600; padding: 0.55rem 1.6rem;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{ background: #323F4B; color: {LANE}; }}
div[data-testid="stFormSubmitButton"] > button:focus-visible {{ outline: 3px solid {LANE}; outline-offset: 2px; }}

/* Verdict panel — the one bold element */
.verdict {{ background: {CURB}; border-radius: 14px; padding: 1.5rem 1.6rem; border: 1px solid #D9DEE4; }}
.verdict .kicker {{ color: {INK_MUTED}; font-size: 1rem; margin-bottom: 0.1rem; }}
.verdict .tier {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 4.2rem; line-height: 1; margin: 0.1rem 0 0.5rem; }}
.verdict .meaning {{ font-size: 1.05rem; color: {ASPHALT}; max-width: 52ch; margin-bottom: 1.2rem; }}
.strip {{ display: flex; height: 22px; border-radius: 6px; overflow: hidden; gap: 2px; background: {CURB}; }}
.strip span {{ display: block; height: 100%; }}
.legend {{ display: flex; justify-content: space-between; margin-top: 0.55rem; gap: 0.8rem; flex-wrap: wrap; }}
.legend div {{ font-size: 0.98rem; color: {ASPHALT}; }}
.legend i {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 0.35rem; vertical-align: -1px; }}
.legend b {{ font-weight: 600; }}
.legend .picked {{ font-weight: 600; text-decoration: underline; text-underline-offset: 3px; }}
.money {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 4.4rem; line-height: 1; color: {ASPHALT}; margin: 0.15rem 0 0.6rem; }}
.facts {{ display: flex; gap: 2.2rem; flex-wrap: wrap; color: {INK_MUTED}; font-size: 1rem; }}
.facts b {{ color: {ASPHALT}; font-size: 1.35rem; font-family: 'Barlow Condensed', sans-serif; display: block; }}
.placeholder {{ border: 2px dashed #C3CBD3; border-radius: 14px; padding: 2.2rem 1.6rem; color: {INK_MUTED}; font-size: 1.05rem; }}
.note {{ color: {INK_MUTED}; font-size: 0.92rem; margin-top: 0.9rem; }}

/* Tabs */
button[data-baseweb="tab"] p {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.2rem; font-weight: 600; }}
div[data-baseweb="tab-highlight"] {{ background-color: {LANE}; height: 4px; }}

@media (max-width: 700px) {{
    .road h1 {{ font-size: 2.2rem; }}
    .verdict .tier, .money {{ font-size: 3.2rem; }}
}}
</style>
"""


# ----------------------------------------------------------------------------- helpers
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
    help_text = FEATURE_HELP.get(feature)
    if feature in SHARE_FEATURES:
        return st.slider(label, 0.0, 1.0, float(round(r["median"], 2)), 0.05, key=key, help=help_text)
    if feature in INTEGER_FEATURES:
        hi = max(int(np.ceil(r["max"])) * 2, 2)
        return st.number_input(label, min_value=1, max_value=hi, value=int(round(r["median"])) or 1,
                               step=1, key=key, help=help_text)
    hi = float(np.ceil(r["max"] * 2))
    return st.number_input(label, min_value=0.0, max_value=hi, value=float(round(r["median"], 2)),
                           step=0.5, key=key, help=help_text)


def shift_form(bundle, prefix, submit_label):
    """Grouped inputs inside a form, so the model runs only when the button is pressed."""
    values = {}
    with st.form(f"{prefix}_form", border=False):
        for title, feats in GROUPS:
            feats = [f for f in feats if f in bundle["numeric_features"]]
            if not feats:
                if title == "Money":
                    st.markdown(f'<div class="group-title">{title}</div>', unsafe_allow_html=True)
                    values["dominant_payment_type"] = st.selectbox(
                        "Most common payment type", bundle["payment_types"], key=f"{prefix}_pay")
                continue
            st.markdown(f'<div class="group-title">{title}</div>', unsafe_allow_html=True)
            cols = st.columns(len(feats) if len(feats) > 1 else 2)
            for col, f in zip(cols, feats):
                with col:
                    values[f] = feature_input(f, bundle["input_ranges"], key=f"{prefix}_{f}")
            if title == "Money":
                values["dominant_payment_type"] = cols[1].selectbox(
                    "Most common payment type", bundle["payment_types"], key=f"{prefix}_pay")
        submitted = st.form_submit_button(submit_label)
    return submitted, values


def verdict_html(pred, proba):
    segments = "".join(
        f'<span style="width:{max(p * 100, 0.8):.2f}%;background:{TIER_COLORS[t]}" title="{t}: {p:.0%}"></span>'
        for t, p in proba.items())
    legend = "".join(
        f'<div class="{"picked" if t == pred else ""}"><i style="background:{TIER_COLORS[t]}"></i>'
        f'{t} <b>{p:.0%}</b></div>' for t, p in proba.items())
    return (f'<div class="verdict"><div class="kicker">Predicted burnout risk</div>'
            f'<div class="tier" style="color:{TIER_COLORS[pred]}">{pred}</div>'
            f'<div class="meaning">{TIER_MEANING[pred]}</div>'
            f'<div class="kicker">How confident the model is in each tier</div>'
            f'<div class="strip">{segments}</div><div class="legend">{legend}</div></div>')


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Shift Check — driver burnout & earnings", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="road"><h1>Shift Check</h1>'
    "<p>Describe one taxi driver's day in Chicago. Our trained models estimate how likely that day is to "
    "lead to burnout, and how much it should earn.</p></div>",
    unsafe_allow_html=True)

try:
    burnout = load_bundle("burnout_classifier.pkl")
    earnings = load_bundle("earnings_regressor.pkl")
except FileNotFoundError:
    st.error("Model files are missing from the `models` folder. Run `notebooks/regression_classification.ipynb` "
             "from top to bottom (Section G saves them), then reload this page.")
    st.stop()

tab_burnout, tab_earnings, tab_about = st.tabs(["Burnout risk", "Daily earnings", "How it works"])

with tab_burnout:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("### The driver's day")
        st.caption("Hours driven are not asked for: the model infers the strain of the shift from the pattern of work.")
        submitted, values = shift_form(burnout, "b", "Check burnout risk")
    with right:
        st.markdown("### Result")
        if submitted:
            X = to_model_input(burnout, values)
            pred = burnout["model"].predict(X)[0]
            proba = pd.Series(burnout["model"].predict_proba(X)[0],
                              index=burnout["model"].classes_)[burnout["class_order"]]
            st.session_state["b_result"] = (pred, proba)
        if "b_result" in st.session_state:
            pred, proba = st.session_state["b_result"]
            st.markdown(verdict_html(pred, proba), unsafe_allow_html=True)
            st.markdown(f'<div class="note">Model: {burnout["model_name"]} (tuned). '
                        f'Test accuracy {burnout["test_metrics"]["Accuracy"]:.0%}, '
                        f'ROC-AUC {burnout["test_metrics"]["ROC-AUC (OvR)"]:.2f}.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="placeholder">Fill in the day on the left and press '
                        '<b>Check burnout risk</b> to see the predicted tier.</div>', unsafe_allow_html=True)

with tab_earnings:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("### The driver's day")
        st.caption("Earnings are predicted from workload, timing, area covered and payment type.")
        submitted_r, values_r = shift_form(earnings, "r", "Predict earnings")
    with right:
        st.markdown("### Result")
        if submitted_r:
            pred_r = max(float(earnings["model"].predict(to_model_input(earnings, values_r))[0]), 0.0)
            st.session_state["r_result"] = (pred_r, values_r)
        if "r_result" in st.session_state:
            pred_r, v = st.session_state["r_result"]
            per_trip = pred_r / v["trip_count"] if v["trip_count"] else 0
            per_mile = f"${pred_r / v['total_miles']:,.2f}" if v["total_miles"] > 0 else "n/a"
            st.markdown(
                f'<div class="verdict"><div class="kicker">Predicted earnings for the day</div>'
                f'<div class="money">${pred_r:,.2f}</div>'
                f'<div class="facts"><div><b>${per_trip:,.2f}</b>per trip</div>'
                f'<div><b>{per_mile}</b>per mile</div>'
                f'<div><b>±${earnings["test_metrics"]["MAE"]:.2f}</b>typical error</div></div></div>',
                unsafe_allow_html=True)
            st.markdown(f'<div class="note">Model: {earnings["model_name"]}. Typical error is the mean absolute '
                        f'error on the held-out test days.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="placeholder">Fill in the day on the left and press '
                        '<b>Predict earnings</b> to see the estimate.</div>', unsafe_allow_html=True)

with tab_about:
    st.markdown("### How the prediction is made")
    st.markdown(
        "1. **Your inputs** describe one driver-day, the same features the models were trained on.\n"
        "2. **Scaling and encoding:** numbers are standardised and the payment type is one-hot encoded, using the "
        "scaler and encoder fitted on the training data only.\n"
        f"3. **Burnout model:** a tuned {burnout['model_name']} picks Low, Medium or High and gives a probability "
        "for each.\n"
        f"4. **Earnings model:** a {earnings['model_name']} estimates the day's total in dollars.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"#### Burnout classifier — {burnout['model_name']}")
        st.dataframe(pd.Series(burnout["test_metrics"], name="Test score").round(3), width="stretch")
    with c2:
        st.markdown(f"#### Earnings regressor — {earnings['model_name']}")
        st.dataframe(pd.Series(earnings["test_metrics"], name="Test score").round(3), width="stretch")
    st.info("The models were trained on a small 2019 sample of Chicago taxi trips (848 driver-days). Values far "
            "outside that range can give unreliable predictions.")
