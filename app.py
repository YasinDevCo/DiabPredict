"""
DiabPredict - Intelligent Diabetes Risk Predictor
==================================================
Senior Machine Learning Engineer & Streamlit Implementation
Dataset: Pima Indians Diabetes Database
Model: Random Forest Classifier (Optimized & Cached)
Features: Full Bilingual Support (Persian / English), RTL Typography,
          Interactive Prediction Engine, Clinical Analytics, and Comprehensive User Guide.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from translations import t
from guide_content import GUIDE_FA, GUIDE_EN

# -----------------------------------------------------------------------------
# 0. STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DiabPredict | سامانه هوشمند ارزیابی ریسک دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "language" not in st.session_state:
    st.session_state["language"] = "fa"  # Persian default per user specification

if "has_predicted" not in st.session_state:
    st.session_state["has_predicted"] = False

# Parameter slider states
if "sl_preg" not in st.session_state:
    st.session_state["sl_preg"] = 1
if "sl_gluc" not in st.session_state:
    st.session_state["sl_gluc"] = 120
if "sl_bp" not in st.session_state:
    st.session_state["sl_bp"] = 70
if "sl_skin" not in st.session_state:
    st.session_state["sl_skin"] = 20
if "sl_ins" not in st.session_state:
    st.session_state["sl_ins"] = 79
if "sl_bmi" not in st.session_state:
    st.session_state["sl_bmi"] = 25.0
if "sl_dpf" not in st.session_state:
    st.session_state["sl_dpf"] = 0.50
if "sl_age" not in st.session_state:
    st.session_state["sl_age"] = 33

# -----------------------------------------------------------------------------
# 1. LANGUAGE SELECTOR (TOP OF SIDEBAR)
# -----------------------------------------------------------------------------
lang_options = ["🇮🇷 فارسی (Persian)", "🇬🇧 English"]
current_index = 0 if st.session_state["language"] == "fa" else 1

selected_lang_str = st.sidebar.selectbox(
    "🌐 تغییر زبان / Change Language",
    options=lang_options,
    index=current_index,
    key="language_selector",
)

active_lang = "fa" if "فارسی" in selected_lang_str else "en"
st.session_state["language"] = active_lang
is_rtl = active_lang == "fa"

# -----------------------------------------------------------------------------
# 2. DYNAMIC STYLING (RTL FOR PERSIAN / LTR FOR ENGLISH)
# -----------------------------------------------------------------------------
if is_rtl:
    custom_css = """
    <style>
        /* Import Vazirmatn Persian Font */
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800;900&display=swap');

        html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"], 
        .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, label, input, button, select, textarea {
            font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }

        /* RTL Base Container */
        [data-testid="stAppViewContainer"] {
            direction: rtl !important;
            text-align: right !important;
        }

        [data-testid="stSidebar"] {
            direction: rtl !important;
            text-align: right !important;
            background-color: #0b1120;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        [data-testid="stSidebarContent"] {
            direction: rtl !important;
            text-align: right !important;
        }

        /* RTL Lists, Quotes, Tables */
        ul, ol {
            padding-right: 1.5rem !important;
            padding-left: 0 !important;
            margin-right: 0 !important;
        }
        li {
            margin-bottom: 0.4rem;
        }
        blockquote {
            border-right: 4px solid #38bdf8 !important;
            border-left: none !important;
            padding-right: 1rem !important;
            padding-left: 0.5rem !important;
            background: rgba(56, 189, 248, 0.05);
            border-radius: 6px;
        }
        table {
            direction: rtl !important;
            text-align: right !important;
        }
        th, td {
            text-align: right !important;
        }

        /* Hero Banner */
        .hero-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
            padding: 2.2rem 2.4rem;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            direction: rtl;
            text-align: right;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
            direction: rtl;
            text-align: right;
        }

        .hero-badge {
            background: rgba(14, 165, 233, 0.25);
            border: 1px solid rgba(56, 189, 248, 0.6);
            color: #38bdf8;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 4px 14px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 6px;
        }

        .hero-desc {
            font-size: 1.05rem;
            color: #cbd5e1;
            margin-top: 0.8rem;
            line-height: 1.8;
            max-width: 950px;
        }

        /* RTL Metrics */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(226, 232, 240, 0.12);
            border-radius: 14px;
            padding: 16px 20px;
            direction: rtl !important;
            text-align: right !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.1);
        }

        [data-testid="stMetricLabel"] {
            direction: rtl !important;
            text-align: right !important;
            justify-content: flex-start !important;
        }

        [data-testid="stMetricValue"] {
            direction: rtl !important;
            text-align: right !important;
        }

        [data-testid="stMetricDelta"] {
            direction: rtl !important;
            text-align: right !important;
            justify-content: flex-start !important;
        }

        [data-testid="stMetricDelta"] svg {
            margin-left: 6px !important;
            margin-right: 0px !important;
        }

        /* RTL Tabs */
        [data-testid="stTabs"] {
            direction: rtl !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            direction: rtl !important;
            flex-direction: row !important;
            gap: 8px;
        }

        [data-testid="stTabs"] button {
            direction: rtl !important;
            font-family: 'Vazirmatn', sans-serif !important;
            font-size: 0.95rem !important;
        }

        /* Alerts in RTL */
        [data-testid="stAlert"] {
            direction: rtl !important;
            text-align: right !important;
            border-radius: 12px;
        }

        /* Expanders in RTL */
        [data-testid="stExpander"] {
            direction: rtl !important;
            text-align: right !important;
            border-radius: 12px;
        }

        /* Slider Label Alignment */
        [data-testid="stSlider"] label {
            text-align: right !important;
            direction: rtl !important;
            font-size: 0.9rem !important;
        }

        /* Primary Buttons */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            border: 1px solid #38bdf8 !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.2rem !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.39) !important;
        }

        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.55) !important;
            transform: scale(1.01);
        }

        .risk-tier-card {
            border-right: 4px solid var(--tier-accent) !important;
            border-left: none !important;
            border-radius: 10px;
            padding: 10px 14px;
            background: rgba(255,255,255,0.05);
            margin-top: 12px;
            text-align: right;
            direction: rtl;
        }

        /* Table & DataFrame RTL */
        [data-testid="stDataFrame"] {
            direction: rtl !important;
        }

        .sidebar-header {
            padding: 8px 0 16px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 14px;
            text-align: right;
            direction: rtl;
        }
    </style>
    """
else:
    custom_css = """
    <style>
        /* Import Google Font for English */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"], 
        .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, label, input, button {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        [data-testid="stAppViewContainer"] {
            direction: ltr !important;
            text-align: left !important;
        }

        [data-testid="stSidebar"] {
            direction: ltr !important;
            text-align: left !important;
            background-color: #0b1120;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-left: none !important;
        }

        .hero-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
            padding: 2.2rem 2.4rem;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            direction: ltr;
            text-align: left;
        }

        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
            direction: ltr;
            text-align: left;
        }

        .hero-badge {
            background: rgba(14, 165, 233, 0.25);
            border: 1px solid rgba(56, 189, 248, 0.6);
            color: #38bdf8;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 6px;
        }

        .hero-desc {
            font-size: 1.05rem;
            color: #cbd5e1;
            margin-top: 0.8rem;
            line-height: 1.6;
            max-width: 950px;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(226, 232, 240, 0.12);
            border-radius: 14px;
            padding: 16px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.1);
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            border: 1px solid #38bdf8 !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            letter-spacing: 0.02em !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.2rem !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.39) !important;
        }

        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.55) !important;
            transform: scale(1.01);
        }

        .risk-tier-card {
            border-left: 4px solid var(--tier-accent) !important;
            border-right: none !important;
            border-radius: 10px;
            padding: 10px 14px;
            background: rgba(255,255,255,0.05);
            margin-top: 12px;
            text-align: left;
            direction: ltr;
        }

        .sidebar-header {
            padding: 8px 0 16px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 14px;
        }
    </style>
    """

st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATA HANDLING & CACHING
# -----------------------------------------------------------------------------
import os

LOCAL_DATA_PATH = "diabetes.csv"
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome',
]

FEATURE_COLS = COLUMN_NAMES[:-1]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Downloads and caches the Pima Indians Diabetes Dataset directly from GitHub or local file."""
    if os.path.exists(LOCAL_DATA_PATH):
        try:
            return pd.read_csv(LOCAL_DATA_PATH, names=COLUMN_NAMES)
        except Exception:
            pass

    try:
        df = pd.read_csv(DATA_URL, names=COLUMN_NAMES)
        try:
            df.to_csv(LOCAL_DATA_PATH, index=False, header=False)
        except Exception:
            pass
        return df
    except Exception as err:
        if os.path.exists(LOCAL_DATA_PATH):
            return pd.read_csv(LOCAL_DATA_PATH, names=COLUMN_NAMES)
        raise err


@st.cache_resource(show_spinner=False)
def train_model(data_df: pd.DataFrame):
    """
    Trains a RandomForestClassifier(random_state=42) on an 80/20 train/test split.
    Calculates and returns the model and metadata.
    """
    X = data_df[FEATURE_COLS]
    y = data_df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    importance_df = pd.DataFrame(
        {
            'Feature': FEATURE_COLS,
            'Importance': rf_model.feature_importances_,
        }
    ).sort_values(by='Importance', ascending=False)

    metadata = {
        'accuracy': accuracy,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_samples': len(data_df),
        'feature_importance': importance_df,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }

    return rf_model, metadata


# Load cached data and trained model
data = load_data()
model, meta = train_model(data)

# -----------------------------------------------------------------------------
# 4. SIDEBAR LAYOUT & USER INPUT SLIDERS
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div class="sidebar-header">
        <h2 style="margin: 0; color: #f8fafc; font-size: 1.35rem; display: flex; align-items: center; gap: 8px;">
            <span>📋</span> {t("sidebar_header_title", active_lang)}
        </h2>
        <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
            {t("sidebar_header_desc", active_lang)}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def apply_preset(p, g, bp, st_val, ins, b, d, a):
    st.session_state["sl_preg"] = p
    st.session_state["sl_gluc"] = g
    st.session_state["sl_bp"] = bp
    st.session_state["sl_skin"] = st_val
    st.session_state["sl_ins"] = ins
    st.session_state["sl_bmi"] = float(b)
    st.session_state["sl_dpf"] = float(d)
    st.session_state["sl_age"] = a
    st.session_state["has_predicted"] = True


# Quick-preset helper buttons
with st.sidebar.expander(t("presets_expander", active_lang), expanded=False):
    preset_col1, preset_col2 = st.columns(2)
    preset_col1.button(
        t("preset_low", active_lang),
        use_container_width=True,
        on_click=apply_preset,
        args=(0, 90, 68, 18, 65, 21.5, 0.22, 24),
    )
    preset_col2.button(
        t("preset_high", active_lang),
        use_container_width=True,
        on_click=apply_preset,
        args=(6, 175, 88, 38, 230, 38.4, 1.25, 52),
    )
    st.button(
        t("preset_reset", active_lang),
        use_container_width=True,
        on_click=apply_preset,
        args=(1, 120, 70, 20, 79, 25.0, 0.50, 33),
    )

# Sliders with bilingual labels and descriptions
pregnancies = st.sidebar.slider(
    t("slider_preg", active_lang),
    min_value=0,
    max_value=17,
    key="sl_preg",
    help=t("slider_preg_help", active_lang),
)

glucose = st.sidebar.slider(
    t("slider_gluc", active_lang),
    min_value=0,
    max_value=200,
    key="sl_gluc",
    help=t("slider_gluc_help", active_lang),
)

blood_pressure = st.sidebar.slider(
    t("slider_bp", active_lang),
    min_value=0,
    max_value=122,
    key="sl_bp",
    help=t("slider_bp_help", active_lang),
)

skin_thickness = st.sidebar.slider(
    t("slider_skin", active_lang),
    min_value=0,
    max_value=99,
    key="sl_skin",
    help=t("slider_skin_help", active_lang),
)

insulin = st.sidebar.slider(
    t("slider_ins", active_lang),
    min_value=0,
    max_value=846,
    key="sl_ins",
    help=t("slider_ins_help", active_lang),
)

bmi = st.sidebar.slider(
    t("slider_bmi", active_lang),
    min_value=0.0,
    max_value=67.1,
    step=0.1,
    key="sl_bmi",
    help=t("slider_bmi_help", active_lang),
)

diabetes_pedigree = st.sidebar.slider(
    t("slider_dpf", active_lang),
    min_value=0.07,
    max_value=2.42,
    step=0.01,
    key="sl_dpf",
    help=t("slider_dpf_help", active_lang),
)

age = st.sidebar.slider(
    t("slider_age", active_lang),
    min_value=21,
    max_value=81,
    key="sl_age",
    help=t("slider_age_help", active_lang),
)

# Patient Input Vector DataFrame
input_dict = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': diabetes_pedigree,
    'Age': age,
}
user_input_df = pd.DataFrame([input_dict])

st.sidebar.markdown("---")
sidebar_predict = st.sidebar.button(
    f"**{t('predict_button', active_lang)}**",
    key="sidebar_btn",
    type="primary",
    use_container_width=True,
)

st.sidebar.markdown(
    f"""
    <div style="font-size: 0.78rem; color: #64748b; text-align: center; margin-top: 10px; line-height: 1.5;">
        {t("sidebar_footer_text", active_lang)}
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 5. MAIN PAGE: HERO BANNER & BASELINE METRICS
# -----------------------------------------------------------------------------
baseline_acc_pct = meta['accuracy'] * 100

st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-badge">{t("hero_badge", active_lang)}</div>
        <h1 class="hero-title">
            <span>🩺</span> {t("hero_title", active_lang)}
        </h1>
        <p class="hero-desc">
            {t("hero_desc", active_lang)}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Summary Key Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric(
        label=t("metric_accuracy_label", active_lang),
        value=f"{baseline_acc_pct:.2f}%",
        delta=t("metric_accuracy_delta", active_lang),
        help=t("metric_accuracy_help", active_lang),
    )

with col_m2:
    st.metric(
        label=t("metric_algo_label", active_lang),
        value=t("metric_algo_val", active_lang),
        delta=t("metric_algo_delta", active_lang),
        help=t("metric_algo_help", active_lang),
    )

with col_m3:
    st.metric(
        label=t("metric_cohort_label", active_lang),
        value=t("metric_cohort_val", active_lang, total=meta['total_samples']),
        delta=t("metric_cohort_delta", active_lang, train=meta['train_samples'], test=meta['test_samples']),
        help=t("metric_cohort_help", active_lang),
    )

with col_m4:
    st.metric(
        label=t("metric_cache_label", active_lang),
        value=t("metric_cache_val", active_lang),
        delta=t("metric_cache_delta", active_lang),
        help=t("metric_cache_help", active_lang),
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. PREDICTION ENGINE TRIGGER & MAIN ACTION
# -----------------------------------------------------------------------------
action_col1, action_col2 = st.columns([1.5, 3.5])

with action_col1:
    main_predict = st.button(
        f"**{t('predict_button', active_lang)}**",
        key="main_btn",
        type="primary",
        use_container_width=True,
    )

with action_col2:
    st.caption(t("action_hint", active_lang))

if main_predict or sidebar_predict:
    st.session_state.has_predicted = True

# -----------------------------------------------------------------------------
# 7. MAIN TABS (INCLUDING COMPREHENSIVE USER GUIDE & TUTORIAL)
# -----------------------------------------------------------------------------
tab_labels = [
    t("tab_prediction", active_lang),
    t("tab_patient", active_lang),
    t("tab_analytics", active_lang),
    t("tab_data", active_lang),
    t("tab_guide", active_lang),
]

tab_prediction, tab_patient, tab_analytics, tab_data, tab_guide = st.tabs(tab_labels)

# -----------------------------------------------------------------------------
# TAB 1: PREDICTION & RISK ASSESSMENT
# -----------------------------------------------------------------------------
with tab_prediction:
    if st.session_state.has_predicted:
        prediction = int(model.predict(user_input_df)[0])
        probabilities = model.predict_proba(user_input_df)[0]
        prob_negative = probabilities[0]
        prob_positive = probabilities[1]
        risk_percentage = prob_positive * 100
        healthy_percentage = prob_negative * 100

        st.subheader(t("risk_eval_title", active_lang))

        if prediction == 1:
            st.error(
                t("high_risk_alert", active_lang, prob=risk_percentage),
                icon="⚠️",
            )
        else:
            st.success(
                t("low_risk_alert", active_lang, healthy_prob=healthy_percentage, prob=risk_percentage),
                icon="🟢",
            )

        r_col1, r_col2 = st.columns([1.2, 2.8])

        with r_col1:
            diff_50 = abs(risk_percentage - 50)
            if risk_percentage >= 50:
                delta_str = t("delta_vs_threshold_pos", active_lang, val=diff_50)
            else:
                delta_str = t("delta_vs_threshold_neg", active_lang, val=diff_50)

            st.metric(
                label=t("metric_risk_prob", active_lang),
                value=f"{risk_percentage:.1f}%",
                delta=delta_str,
                delta_color="inverse" if risk_percentage >= 50 else "normal",
            )

            # Categorical Risk Tier
            if risk_percentage < 30:
                tier_color = "#10b981"
                tier_label = t("tier_low", active_lang)
            elif risk_percentage < 60:
                tier_color = "#f59e0b"
                tier_label = t("tier_moderate", active_lang)
            else:
                tier_color = "#ef4444"
                tier_label = t("tier_high", active_lang)

            st.markdown(
                f"""
                <div class="risk-tier-card" style="--tier-accent: {tier_color};">
                    <span style="color: {tier_color}; font-weight: 700; font-size: 0.95rem;">
                        {tier_label}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r_col2:
            st.markdown(f"**{t('scale_title', active_lang)}**")
            progress_ratio = float(np.clip(prob_positive, 0.0, 1.0))
            st.progress(progress_ratio)

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #94a3b8; margin-top: 6px; direction: ltr;">
                    <span>{t("scale_low", active_lang)}</span>
                    <span>{t("scale_mod", active_lang)}</span>
                    <span>{t("scale_high", active_lang)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Clinical Context & Top Factor Analysis
        st.subheader(t("drivers_header", active_lang))
        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            st.markdown(f"##### {t('obs_header', active_lang)}")
            warnings = []

            if glucose >= 140:
                warnings.append(t("obs_glucose_high", active_lang, val=glucose))
            elif glucose < 100:
                warnings.append(t("obs_glucose_opt", active_lang, val=glucose))

            if bmi >= 30.0:
                warnings.append(t("obs_bmi_high", active_lang, val=bmi))
            elif bmi < 25.0:
                warnings.append(t("obs_bmi_opt", active_lang, val=bmi))

            if blood_pressure >= 80:
                warnings.append(t("obs_bp_high", active_lang, val=blood_pressure))

            if age >= 45:
                warnings.append(t("obs_age_high", active_lang, val=age))

            if warnings:
                for w in warnings:
                    st.markdown(w)
            else:
                st.info(t("obs_all_normal", active_lang))

        with insight_col2:
            st.markdown(f"##### {t('recs_header', active_lang)}")
            if prediction == 1 or risk_percentage >= 50:
                st.warning(t("recs_high", active_lang))
            else:
                st.success(t("recs_low", active_lang))

    else:
        st.info(t("welcome_info", active_lang), icon="ℹ️")

# -----------------------------------------------------------------------------
# TAB 2: PATIENT CLINICAL OVERVIEW
# -----------------------------------------------------------------------------
with tab_patient:
    st.subheader(t("patient_tab_header", active_lang))

    if active_lang == "fa":
        reference_data = [
            {
                t("col_param", active_lang): "تعداد بارداری‌ها (Pregnancies)",
                t("col_val", active_lang): pregnancies,
                t("col_ref", active_lang): "۰ تا ۴",
                t("col_unit", active_lang): "تعداد",
                t("col_status", active_lang): "نرمال" if pregnancies <= 4 else "بیشتر از میانگین",
            },
            {
                t("col_param", active_lang): "گلوکز پلاسما (Glucose)",
                t("col_val", active_lang): glucose,
                t("col_ref", active_lang): "۷۰ تا ۱۴۰",
                t("col_unit", active_lang): "mg/dL",
                t("col_status", active_lang): "نرمال" if glucose < 140 else "بالا (اختلال تحمل گلوکز)",
            },
            {
                t("col_param", active_lang): "فشار خون دیاستولیک (Blood Pressure)",
                t("col_val", active_lang): blood_pressure,
                t("col_ref", active_lang): "۶۰ تا ۸۰",
                t("col_unit", active_lang): "mm Hg",
                t("col_status", active_lang): "نرمال" if blood_pressure <= 80 else "پیش‌فشارخون / بالا",
            },
            {
                t("col_param", active_lang): "ضخامت چین پوستی (Skin Thickness)",
                t("col_val", active_lang): skin_thickness,
                t("col_ref", active_lang): "۱۰ تا ۳۰",
                t("col_unit", active_lang): "mm",
                t("col_status", active_lang): "نرمال" if skin_thickness <= 30 else "بالاتر از حد متوسط",
            },
            {
                t("col_param", active_lang): "انسولین سرم ۲ ساعته (Insulin)",
                t("col_val", active_lang): insulin,
                t("col_ref", active_lang): "۱۶ تا ۱۶۶",
                t("col_unit", active_lang): "μU/mL",
                t("col_status", active_lang): "نرمال" if insulin <= 166 else "بالا (مقاومت به انسولین)",
            },
            {
                t("col_param", active_lang): "شاخص توده بدنی (BMI)",
                t("col_val", active_lang): bmi,
                t("col_ref", active_lang): "۱۸.۵ تا ۲۴.۹",
                t("col_unit", active_lang): "kg/m²",
                t("col_status", active_lang): "نرمال" if bmi < 25 else ("اضافه‌وزن" if bmi < 30 else "چاقی (ریسک بالا)"),
            },
            {
                t("col_param", active_lang): "تابع شجره‌نامه دیابت (DPF)",
                t("col_val", active_lang): diabetes_pedigree,
                t("col_ref", active_lang): "کمتر از ۰.۵۰",
                t("col_unit", active_lang): "شاخص",
                t("col_status", active_lang): "ریسک وراثتی پایین" if diabetes_pedigree < 0.5 else "سابقه خانوادگی بالا",
            },
            {
                t("col_param", active_lang): "سن بیمار (Age)",
                t("col_val", active_lang): age,
                t("col_ref", active_lang): "۲۱ تا ۴۵",
                t("col_unit", active_lang): "سال",
                t("col_status", active_lang): "بازه استاندارد" if age < 45 else "رده سنی نیازمند پایش (≥۴۵)",
            },
        ]
    else:
        reference_data = [
            {
                t("col_param", active_lang): "Pregnancies",
                t("col_val", active_lang): pregnancies,
                t("col_ref", active_lang): "0 - 4",
                t("col_unit", active_lang): "count",
                t("col_status", active_lang): "Normal" if pregnancies <= 4 else "Elevated",
            },
            {
                t("col_param", active_lang): "Glucose",
                t("col_val", active_lang): glucose,
                t("col_ref", active_lang): "70 - 140",
                t("col_unit", active_lang): "mg/dL",
                t("col_status", active_lang): "Normal" if glucose < 140 else "High (Impaired)",
            },
            {
                t("col_param", active_lang): "Blood Pressure",
                t("col_val", active_lang): blood_pressure,
                t("col_ref", active_lang): "60 - 80",
                t("col_unit", active_lang): "mm Hg",
                t("col_status", active_lang): "Normal" if blood_pressure <= 80 else "Pre-hypertension",
            },
            {
                t("col_param", active_lang): "Skin Thickness",
                t("col_val", active_lang): skin_thickness,
                t("col_ref", active_lang): "10 - 30",
                t("col_unit", active_lang): "mm",
                t("col_status", active_lang): "Normal" if skin_thickness <= 30 else "Above Average",
            },
            {
                t("col_param", active_lang): "Insulin",
                t("col_val", active_lang): insulin,
                t("col_ref", active_lang): "16 - 166",
                t("col_unit", active_lang): "μU/mL",
                t("col_status", active_lang): "Normal" if insulin <= 166 else "High",
            },
            {
                t("col_param", active_lang): "BMI",
                t("col_val", active_lang): bmi,
                t("col_ref", active_lang): "18.5 - 24.9",
                t("col_unit", active_lang): "kg/m²",
                t("col_status", active_lang): "Normal" if bmi < 25 else ("Overweight" if bmi < 30 else "Obese"),
            },
            {
                t("col_param", active_lang): "Diabetes Pedigree",
                t("col_val", active_lang): diabetes_pedigree,
                t("col_ref", active_lang): "< 0.50",
                t("col_unit", active_lang): "index",
                t("col_status", active_lang): "Low Hereditary Risk" if diabetes_pedigree < 0.5 else "Elevated Family Risk",
            },
            {
                t("col_param", active_lang): "Age",
                t("col_val", active_lang): age,
                t("col_ref", active_lang): "21 - 45",
                t("col_unit", active_lang): "years",
                t("col_status", active_lang): "Standard Range" if age < 45 else "Elevated Age Tier",
            },
        ]

    ref_df = pd.DataFrame(reference_data)
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 3: MODEL INTELLIGENCE & ANALYTICS
# -----------------------------------------------------------------------------
with tab_analytics:
    st.subheader(t("analytics_tab_header", active_lang))
    st.caption(t("analytics_tab_caption", active_lang))

    fi_df = meta['feature_importance'].copy()
    fi_df[t('weight_col', active_lang)] = (fi_df['Importance'] * 100).round(2)

    # Localize feature names if in Persian
    if active_lang == "fa":
        feature_map_fa = {
            'Glucose': 'گلوکز (Glucose)',
            'BMI': 'شاخص توده بدنی (BMI)',
            'Age': 'سن (Age)',
            'DiabetesPedigreeFunction': 'شجره‌نامه ژنتیکی (DPF)',
            'BloodPressure': 'فشار خون (BP)',
            'Insulin': 'انسولین (Insulin)',
            'SkinThickness': 'ضخامت پوست (Skin)',
            'Pregnancies': 'تعداد بارداری‌ها (Preg)',
        }
        fi_df[t('feature_col', active_lang)] = fi_df['Feature'].map(lambda x: feature_map_fa.get(x, x))
    else:
        fi_df[t('feature_col', active_lang)] = fi_df['Feature']

    col_chart, col_matrix = st.columns([3, 2])

    with col_chart:
        chart_data = fi_df.set_index(t('feature_col', active_lang))[t('weight_col', active_lang)]
        st.bar_chart(
            data=chart_data,
            horizontal=True,
            use_container_width=True,
        )

    with col_matrix:
        st.markdown(f"##### {t('cm_header', active_lang)}")
        cm = meta['confusion_matrix']
        cm_df = pd.DataFrame(
            cm,
            columns=[t('cm_pred_neg', active_lang), t('cm_pred_pos', active_lang)],
            index=[t('cm_act_neg', active_lang), t('cm_act_pos', active_lang)],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            t(
                "cm_caption",
                active_lang,
                test_count=meta['test_samples'],
                acc=baseline_acc_pct,
            )
        )

# -----------------------------------------------------------------------------
# TAB 4: DATASET EXPLORER
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader(t("data_tab_header", active_lang))
    st.caption(t("data_tab_caption", active_lang))

    st.dataframe(data.head(15), use_container_width=True)

    with st.expander(t("data_summary_expander", active_lang)):
        st.dataframe(data.describe().round(2), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: COMPREHENSIVE USER GUIDE & TUTORIAL (صفحه راهنما و آموزش جامع)
# -----------------------------------------------------------------------------
with tab_guide:
    if active_lang == "fa":
        st.markdown(GUIDE_FA, unsafe_allow_html=True)
    else:
        st.markdown(GUIDE_EN, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER & MEDICAL DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
    <div style="font-size: 0.82rem; color: #64748b; line-height: 1.7; text-align: center; max-width: 1000px; margin: 0 auto;">
        {t("footer_disclaimer", active_lang)}
    </div>
    """,
    unsafe_allow_html=True,
)
