"""
DiabPredict - Intelligent Diabetes Risk Predictor
==================================================
Senior Machine Learning Engineer & Streamlit Implementation
Dataset: Pima Indians Diabetes Database
Model: Random Forest Classifier (Optimized & Cached)
Features: Full Bilingual Support (Persian / English), Precision RTL Typography,
          Interactive Prediction Engine, Clinical Analytics, and Comprehensive User Guide.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from translations import t
from guide_content import GUIDE_FA, GUIDE_EN
from architecture_content import ARCHITECTURE_FA, ARCHITECTURE_EN

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
    st.session_state["language"] = "fa"

if "has_predicted" not in st.session_state:
    st.session_state["has_predicted"] = False

# Parameter slider defaults
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
# Clean labels without emoji flag combos that render as 'GB' / 'IR' on Windows
lang_options = ["فارسی (Persian)", "English"]
current_index = 0 if st.session_state["language"] == "fa" else 1

selected_lang_str = st.sidebar.selectbox(
    "🌐 تغییر زبان / Select Language",
    options=lang_options,
    index=current_index,
    key="language_selector",
)

active_lang = "fa" if "فارسی" in selected_lang_str else "en"
st.session_state["language"] = active_lang
is_rtl = active_lang == "fa"

# -----------------------------------------------------------------------------
# 2. BULLETPROOF CSS: PRESERVING MATERIAL ICONS + PRECISION RTL TYPOGRAPHY
# -----------------------------------------------------------------------------
if is_rtl:
    custom_css = """
    <style>
        /* Import Vazirmatn Font */
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800;900&display=swap');

        /* 1. CRITICAL: Preserve Streamlit and Material Symbols icons from font overrides */
        [data-testid="stIconMaterial"], 
        [data-testid="stIcon"], 
        [data-testid="stExpanderToggleIcon"],
        [data-testid="stToolbar"] *,
        [data-testid="stMainMenu"] *,
        [data-testid="baseButton-header"] *,
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="collapsedControl"] *,
        .material-symbols-rounded, 
        .material-icons,
        [class*="material-symbols"],
        [class*="stIcon"] {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
            font-feature-settings: "liga" 1 !important;
            font-variant-ligatures: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
        }

        /* 2. Apply Vazirmatn to text elements only */
        html, body, 
        [data-testid="stAppViewContainer"], 
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] span:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded):not([data-testid="stExpanderToggleIcon"]),
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] span,
        h1, h2, h3, h4, h5, h6, 
        p, li, td, th {
            font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }

        /* RTL Base Container */
        [data-testid="stAppViewContainer"] {
            direction: rtl !important;
            text-align: right !important;
        }

        /* Sidebar RTL */
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
            line-height: 1.85;
            max-width: 950px;
        }

        /* 3. Metric Cards: Fix truncation and ellipis */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(226, 232, 240, 0.12);
            border-radius: 14px;
            padding: 16px 18px;
            direction: rtl !important;
            text-align: right !important;
            min-height: 125px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        [data-testid="stMetricLabel"] {
            direction: rtl !important;
            text-align: right !important;
            justify-content: flex-start !important;
            white-space: normal !important;
            overflow: visible !important;
        }

        [data-testid="stMetricLabel"] p {
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: #94a3b8 !important;
            white-space: normal !important;
        }

        [data-testid="stMetricValue"] {
            direction: rtl !important;
            text-align: right !important;
            white-space: normal !important;
            overflow: visible !important;
        }

        [data-testid="stMetricValue"] div {
            font-size: 1.45rem !important;
            font-weight: 800 !important;
            color: #f8fafc !important;
            white-space: normal !important;
            line-height: 1.3 !important;
        }

        [data-testid="stMetricDelta"] {
            direction: rtl !important;
            text-align: right !important;
            justify-content: flex-start !important;
            white-space: normal !important;
        }

        [data-testid="stMetricDelta"] div {
            font-size: 0.78rem !important;
            white-space: normal !important;
        }

        [data-testid="stMetricDelta"] svg {
            margin-left: 6px !important;
            margin-right: 0px !important;
        }

        /* 4. Expander RTL & Chevron Fix */
        [data-testid="stExpander"] {
            direction: rtl !important;
            text-align: right !important;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background: rgba(255, 255, 255, 0.02) !important;
        }

        [data-testid="stExpander"] details summary {
            direction: rtl !important;
            text-align: right !important;
            padding: 10px 14px !important;
        }

        [data-testid="stExpander"] details summary p {
            direction: rtl !important;
            text-align: right !important;
            font-weight: 600 !important;
            color: #f1f5f9 !important;
            margin: 0 !important;
        }

        /* 5. RTL Tabs */
        [data-testid="stTabs"] {
            direction: rtl !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            direction: rtl !important;
            flex-direction: row !important;
            gap: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        [data-testid="stTabs"] button {
            direction: rtl !important;
            font-family: 'Vazirmatn', sans-serif !important;
            font-size: 0.95rem !important;
            padding: 10px 16px !important;
            border-radius: 8px 8px 0 0 !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #38bdf8 !important;
        }

        /* 6. Alerts in RTL */
        [data-testid="stAlert"] {
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

        .sidebar-header {
            padding: 8px 0 16px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 14px;
            text-align: right;
            direction: rtl;
        }

        /* 7. Comprehensive Guide RTL Wrapper */
        .guide-rtl-wrapper {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Vazirmatn', sans-serif !important;
            line-height: 2 !important;
        }
        .guide-rtl-wrapper * {
            direction: rtl !important;
            text-align: right !important;
        }
        .guide-rtl-wrapper h1, .guide-rtl-wrapper h2, .guide-rtl-wrapper h3, .guide-rtl-wrapper h4 {
            direction: rtl !important;
            text-align: right !important;
            color: #f8fafc !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
            margin-top: 26px;
        }
        .guide-rtl-wrapper p, .guide-rtl-wrapper span {
            direction: rtl !important;
            text-align: right !important;
            color: #cbd5e1 !important;
            font-size: 0.98rem !important;
            line-height: 1.95 !important;
        }
        .guide-rtl-wrapper ul, .guide-rtl-wrapper ol {
            direction: rtl !important;
            text-align: right !important;
            padding-right: 2rem !important;
            padding-left: 0 !important;
            margin: 12px 0 !important;
        }
        .guide-rtl-wrapper li {
            direction: rtl !important;
            text-align: right !important;
            margin-bottom: 8px !important;
            color: #cbd5e1 !important;
        }
        .guide-rtl-wrapper table {
            width: 100% !important;
            direction: rtl !important;
            text-align: right !important;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            margin: 20px 0 !important;
            background: rgba(15, 23, 42, 0.6) !important;
        }
        .guide-rtl-wrapper th {
            background: rgba(14, 165, 233, 0.2) !important;
            color: #38bdf8 !important;
            padding: 12px 16px !important;
            font-weight: 700 !important;
            direction: rtl !important;
            text-align: right !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12) !important;
        }
        .guide-rtl-wrapper td {
            padding: 12px 16px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
            direction: rtl !important;
            text-align: right !important;
            color: #e2e8f0 !important;
        }
        .guide-rtl-wrapper blockquote {
            border-right: 4px solid #38bdf8 !important;
            border-left: none !important;
            padding: 14px 18px !important;
            background: rgba(14, 165, 233, 0.08) !important;
            border-radius: 10px !important;
            direction: rtl !important;
            text-align: right !important;
            margin: 20px 0 !important;
        }
    </style>
    """
else:
    custom_css = """
    <style>
        /* Import Google Font for English */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        /* 1. CRITICAL: Preserve Streamlit and Material Symbols icons from font overrides */
        [data-testid="stIconMaterial"], 
        [data-testid="stIcon"], 
        [data-testid="stExpanderToggleIcon"],
        [data-testid="stToolbar"] *,
        [data-testid="stMainMenu"] *,
        [data-testid="baseButton-header"] *,
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="collapsedControl"] *,
        .material-symbols-rounded, 
        .material-icons,
        [class*="material-symbols"],
        [class*="stIcon"] {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
            font-feature-settings: "liga" 1 !important;
            font-variant-ligatures: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
        }

        /* 2. Apply font to text elements only */
        html, body, 
        [data-testid="stAppViewContainer"], 
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] span:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded):not([data-testid="stExpanderToggleIcon"]),
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] span,
        h1, h2, h3, h4, h5, h6, 
        p, li, td, th {
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
            padding: 16px 18px;
            min-height: 125px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        [data-testid="stMetricLabel"] p {
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: #94a3b8 !important;
            white-space: normal !important;
        }

        [data-testid="stMetricValue"] div {
            font-size: 1.45rem !important;
            font-weight: 800 !important;
            color: #f8fafc !important;
            white-space: normal !important;
        }

        [data-testid="stMetricDelta"] div {
            font-size: 0.78rem !important;
            white-space: normal !important;
        }

        [data-testid="stExpander"] {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background: rgba(255, 255, 255, 0.02) !important;
        }

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

# Summary Key Metrics Row (With wrapped labels and no ellipses truncation)
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
# 7. MAIN TABS
# -----------------------------------------------------------------------------
tab_labels = [
    t("tab_prediction", active_lang),
    t("tab_patient", active_lang),
    t("tab_analytics", active_lang),
    t("tab_data", active_lang),
    t("tab_guide", active_lang),
    t("tab_architecture", active_lang),
]

tab_prediction, tab_patient, tab_analytics, tab_data, tab_guide, tab_arch = st.tabs(tab_labels)

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
# TAB 2: PATIENT CLINICAL OVERVIEW (100% RTL NATIVE TABLE)
# -----------------------------------------------------------------------------
with tab_patient:
    st.subheader(t("patient_tab_header", active_lang))

    if active_lang == "fa":
        reference_data = [
            ("تعداد بارداری‌ها (Pregnancies)", str(pregnancies), "۰ تا ۴", "تعداد", "نرمال" if pregnancies <= 4 else "بیشتر از میانگین", "normal" if pregnancies <= 4 else "warning"),
            ("گلوکز پلاسما (Glucose)", f"{glucose}", "۷۰ تا ۱۴۰", "mg/dL", "نرمال" if glucose < 140 else "بالا (اختلال تحمل)", "normal" if glucose < 140 else "critical"),
            ("فشار خون دیاستولیک (Blood Pressure)", f"{blood_pressure}", "۶۰ تا ۸۰", "mm Hg", "نرمال" if blood_pressure <= 80 else "پیش‌فشارخون / بالا", "normal" if blood_pressure <= 80 else "warning"),
            ("ضخامت چین پوستی (Skin Thickness)", f"{skin_thickness}", "۱۰ تا ۳۰", "mm", "نرمال" if skin_thickness <= 30 else "بالاتر از حد متوسط", "normal" if skin_thickness <= 30 else "warning"),
            ("انسولین سرم ۲ ساعته (Insulin)", f"{insulin}", "۱۶ تا ۱۶۶", "μU/mL", "نرمال" if insulin <= 166 else "بالا (مقاومت انسولین)", "normal" if insulin <= 166 else "critical"),
            ("شاخص توده بدنی (BMI)", f"{bmi:.1f}", "۱۸.۵ تا ۲۴.۹", "kg/m²", "نرمال" if bmi < 25 else ("اضافه‌وزن" if bmi < 30 else "چاقی (ریسک بالا)"), "normal" if bmi < 25 else ("warning" if bmi < 30 else "critical")),
            ("تابع شجره‌نامه دیابت (DPF)", f"{diabetes_pedigree:.2f}", "کمتر از ۰.۵۰", "شاخص", "ریسک وراثتی پایین" if diabetes_pedigree < 0.5 else "سابقه خانوادگی بالا", "normal" if diabetes_pedigree < 0.5 else "warning"),
            ("سن بیمار (Age)", f"{age}", "۲۱ تا ۴۵", "سال", "بازه استاندارد" if age < 45 else "رده سنی نیازمند پایش (≥۴۵)", "normal" if age < 45 else "warning"),
        ]
        headers = ["شاخص بالینی", "مقدار کنونی بیمار", "محدوده نرمال مرجع", "واحد سنجش", "وضعیت بالینی"]
    else:
        reference_data = [
            ("Pregnancies", str(pregnancies), "0 - 4", "count", "Normal" if pregnancies <= 4 else "Elevated", "normal" if pregnancies <= 4 else "warning"),
            ("Glucose", f"{glucose}", "70 - 140", "mg/dL", "Normal" if glucose < 140 else "High (Impaired)", "normal" if glucose < 140 else "critical"),
            ("Blood Pressure", f"{blood_pressure}", "60 - 80", "mm Hg", "Normal" if blood_pressure <= 80 else "Pre-hypertension", "normal" if blood_pressure <= 80 else "warning"),
            ("Skin Thickness", f"{skin_thickness}", "10 - 30", "mm", "Normal" if skin_thickness <= 30 else "Above Average", "normal" if skin_thickness <= 30 else "warning"),
            ("Insulin", f"{insulin}", "16 - 166", "μU/mL", "Normal" if insulin <= 166 else "High", "normal" if insulin <= 166 else "critical"),
            ("BMI", f"{bmi:.1f}", "18.5 - 24.9", "kg/m²", "Normal" if bmi < 25 else ("Overweight" if bmi < 30 else "Obese"), "normal" if bmi < 25 else ("warning" if bmi < 30 else "critical")),
            ("Diabetes Pedigree", f"{diabetes_pedigree:.2f}", "< 0.50", "index", "Low Hereditary Risk" if diabetes_pedigree < 0.5 else "Elevated Family Risk", "normal" if diabetes_pedigree < 0.5 else "warning"),
            ("Age", f"{age}", "21 - 45", "years", "Standard Range" if age < 45 else "Elevated Age Tier", "normal" if age < 45 else "warning"),
        ]
        headers = ["Clinical Parameter", "Current Value", "Normal Reference", "Unit", "Clinical Status"]

    # Generate precision HTML table with 100% RTL support and badges (no leading indentation to avoid code blocks)
    align_dir = "rtl" if is_rtl else "ltr"
    align_text = "right" if is_rtl else "left"

    table_rows = []
    for param, val, ref, unit, status, badge_tier in reference_data:
        if badge_tier == "normal":
            badge_style = "background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.35);"
        elif badge_tier == "warning":
            badge_style = "background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.35);"
        else:
            badge_style = "background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.35);"

        table_rows.append(
            f'<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.07);">'
            f'<td style="padding: 14px 16px; font-weight: 600; color: #f8fafc; text-align: {align_text};">{param}</td>'
            f'<td style="padding: 14px 16px; font-weight: 700; color: #38bdf8; text-align: {align_text};">{val}</td>'
            f'<td style="padding: 14px 16px; color: #94a3b8; text-align: {align_text};">{ref}</td>'
            f'<td style="padding: 14px 16px; color: #64748b; text-align: {align_text};">{unit}</td>'
            f'<td style="padding: 14px 16px; text-align: {align_text};">'
            f'<span style="display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; {badge_style}">{status}</span>'
            f'</td>'
            f'</tr>'
        )

    rows_joined = "".join(table_rows)

    custom_table_html = (
        f'<div style="direction: {align_dir}; text-align: {align_text}; overflow-x: auto; margin-top: 1.2rem; border-radius: 14px; border: 1px solid rgba(255, 255, 255, 0.1); background: rgba(15, 23, 42, 0.45); box-shadow: 0 4px 12px rgba(0,0,0,0.15);">'
        f'<table style="width: 100%; border-collapse: collapse; font-size: 0.92rem; direction: {align_dir}; text-align: {align_text};">'
        f'<thead>'
        f'<tr style="background: rgba(14, 165, 233, 0.12); border-bottom: 1px solid rgba(255, 255, 255, 0.1);">'
        f'<th style="padding: 14px 16px; font-weight: 700; color: #38bdf8; text-align: {align_text};">{headers[0]}</th>'
        f'<th style="padding: 14px 16px; font-weight: 700; color: #38bdf8; text-align: {align_text};">{headers[1]}</th>'
        f'<th style="padding: 14px 16px; font-weight: 700; color: #38bdf8; text-align: {align_text};">{headers[2]}</th>'
        f'<th style="padding: 14px 16px; font-weight: 700; color: #38bdf8; text-align: {align_text};">{headers[3]}</th>'
        f'<th style="padding: 14px 16px; font-weight: 700; color: #38bdf8; text-align: {align_text};">{headers[4]}</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>{rows_joined}</tbody>'
        f'</table>'
        f'</div>'
    )
    st.markdown(custom_table_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: MODEL INTELLIGENCE & ANALYTICS (100% RTL BARS & CONFUSION MATRIX)
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
            'Age': 'سن بیمار (Age)',
            'DiabetesPedigreeFunction': 'شجره‌نامه ژنتیکی (DPF)',
            'BloodPressure': 'فشار خون دیاستولیک (BP)',
            'Insulin': 'انسولین سرم (Insulin)',
            'SkinThickness': 'ضخامت پوست (Skin)',
            'Pregnancies': 'تعداد بارداری‌ها (Preg)',
        }
        fi_df[t('feature_col', active_lang)] = fi_df['Feature'].map(lambda x: feature_map_fa.get(x, x))
    else:
        fi_df[t('feature_col', active_lang)] = fi_df['Feature']

    col_chart, col_matrix = st.columns([3, 2])

    with col_chart:
        # High fidelity RTL progress bars
        max_w = float(fi_df[t('weight_col', active_lang)].max()) if len(fi_df) > 0 else 1.0
        bar_rows = []

        for _, r in fi_df.iterrows():
            fname = r[t('feature_col', active_lang)]
            wval = float(r[t('weight_col', active_lang)])
            pct_bar = (wval / max_w) * 100

            if is_rtl:
                bar_rows.append(
                    f'<div style="margin-bottom: 14px;">'
                    f'<div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 5px; direction: rtl;">'
                    f'<span style="font-weight: 600; color: #f1f5f9;">{fname}</span>'
                    f'<span style="color: #38bdf8; font-weight: 700; direction: ltr;">{wval:.2f}%</span>'
                    f'</div>'
                    f'<div style="background: rgba(255,255,255,0.06); border-radius: 8px; height: 10px; width: 100%; overflow: hidden; direction: rtl;">'
                    f'<div style="background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%); height: 100%; width: {pct_bar}%; border-radius: 8px;"></div>'
                    f'</div>'
                    f'</div>'
                )
            else:
                bar_rows.append(
                    f'<div style="margin-bottom: 14px;">'
                    f'<div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 5px; direction: ltr;">'
                    f'<span style="font-weight: 600; color: #f1f5f9;">{fname}</span>'
                    f'<span style="color: #38bdf8; font-weight: 700;">{wval:.2f}%</span>'
                    f'</div>'
                    f'<div style="background: rgba(255,255,255,0.06); border-radius: 8px; height: 10px; width: 100%; overflow: hidden; direction: ltr;">'
                    f'<div style="background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%); height: 100%; width: {pct_bar}%; border-radius: 8px;"></div>'
                    f'</div>'
                    f'</div>'
                )

        card_title = "اهمیت نسبی بیومارکرها در مدل" if is_rtl else "Relative Biomarker Importance"
        chart_dir = "rtl" if is_rtl else "ltr"
        bars_content = "".join(bar_rows)
        custom_chart_html = (
            f'<div style="direction: {chart_dir}; padding: 18px 20px; border-radius: 14px; background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.1);">'
            f'<div style="font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 14px;">📊 {card_title}</div>'
            f'{bars_content}'
            f'</div>'
        )
        st.markdown(custom_chart_html, unsafe_allow_html=True)

    with col_matrix:
        cm = meta['confusion_matrix']
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        total_cm = tn + fp + fn + tp

        if is_rtl:
            cm_title = "🧪 ماتریس درهم‌ریختگی (Confusion Matrix)"
            cm_sub = "مقایسه پیش‌بینی‌های مدل با نتایج واقعی در داده‌های آزمون"
            lbl_tn = "پیش‌بینی صحیح سالم (TN)"
            lbl_tp = "پیش‌بینی صحیح دیابت (TP)"
            lbl_fp = "خطای مثبت کاذب (FP)"
            lbl_fn = "خطای منفی کاذب (FN)"
            total_lbl = "از کل نمونه‌های تست"
        else:
            cm_title = "🧪 Test Set Confusion Matrix"
            cm_sub = "Model predictions vs. actual ground truth on test cohort"
            lbl_tn = "True Negative (Healthy)"
            lbl_tp = "True Positive (Diabetic)"
            lbl_fp = "False Positive (Type I Error)"
            lbl_fn = "False Negative (Type II Error)"
            total_lbl = "of test split"

        cm_caption_text = t("cm_caption", active_lang, test_count=meta['test_samples'], acc=baseline_acc_pct)
        custom_cm_html = (
            f'<div style="direction: {chart_dir}; padding: 18px 20px; border-radius: 14px; background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.1);">'
            f'<div style="font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 4px;">{cm_title}</div>'
            f'<div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 16px;">{cm_sub}</div>'
            f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">'
            f'<div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 14px 10px; text-align: center;">'
            f'<div style="font-size: 0.76rem; color: #10b981; font-weight: 600;">{lbl_tn}</div>'
            f'<div style="font-size: 1.65rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{tn}</div>'
            f'<div style="font-size: 0.74rem; color: #94a3b8;">{(tn/total_cm*100):.1f}% {total_lbl}</div>'
            f'</div>'
            f'<div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 10px; padding: 14px 10px; text-align: center;">'
            f'<div style="font-size: 0.76rem; color: #f87171; font-weight: 600;">{lbl_fp}</div>'
            f'<div style="font-size: 1.65rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{fp}</div>'
            f'<div style="font-size: 0.74rem; color: #94a3b8;">{(fp/total_cm*100):.1f}% {total_lbl}</div>'
            f'</div>'
            f'<div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 10px; padding: 14px 10px; text-align: center;">'
            f'<div style="font-size: 0.76rem; color: #fbbf24; font-weight: 600;">{lbl_fn}</div>'
            f'<div style="font-size: 1.65rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{fn}</div>'
            f'<div style="font-size: 0.74rem; color: #94a3b8;">{(fn/total_cm*100):.1f}% {total_lbl}</div>'
            f'</div>'
            f'<div style="background: rgba(14, 165, 233, 0.12); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 10px; padding: 14px 10px; text-align: center;">'
            f'<div style="font-size: 0.76rem; color: #38bdf8; font-weight: 600;">{lbl_tp}</div>'
            f'<div style="font-size: 1.65rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{tp}</div>'
            f'<div style="font-size: 0.74rem; color: #94a3b8;">{(tp/total_cm*100):.1f}% {total_lbl}</div>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top: 14px; font-size: 0.8rem; color: #94a3b8; text-align: center; direction: {chart_dir};">'
            f'{cm_caption_text}'
            f'</div>'
            f'</div>'
        )
        st.markdown(custom_cm_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 4: DATASET EXPLORER (100% RTL STATISTICS & METRICS)
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader(t("data_tab_header", active_lang))
    st.caption(t("data_tab_caption", active_lang))

    # Cohort Breakdown Cards in Persian/English
    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    total_recs = len(data)
    pos_recs = int((data['Outcome'] == 1).sum())
    neg_recs = int((data['Outcome'] == 0).sum())

    with d_c1:
        st.metric(
            label="تعداد کل رکوردهای بیماران" if is_rtl else "Total Patient Records",
            value=f"{total_recs:,}",
            delta="۱۰۰٪ جامعه نمونه" if is_rtl else "100% Dataset Cohort",
        )
    with d_c2:
        st.metric(
            label="شاخص‌های بالینی (ویژگی‌ها)" if is_rtl else "Biomarker Features",
            value="۸ متغیر ورودی" if is_rtl else "8 Input Features",
            delta="+ ۱ ستون برچسب" if is_rtl else "+ 1 Outcome Label",
        )
    with d_c3:
        st.metric(
            label="پرونده‌های دارای دیابت (مثبت)" if is_rtl else "Diabetic Cases (Positive)",
            value=f"{pos_recs}",
            delta=f"{(pos_recs/total_recs*100):.1f}٪ شیوع در جامعه" if is_rtl else f"{(pos_recs/total_recs*100):.1f}% Prevalence",
            delta_color="inverse",
        )
    with d_c4:
        st.metric(
            label="پرونده‌های بدون دیابت (سالم)" if is_rtl else "Healthy Controls (Negative)",
            value=f"{neg_recs}",
            delta=f"{(neg_recs/total_recs*100):.1f}٪ از کل داده‌ها" if is_rtl else f"{(neg_recs/total_recs*100):.1f}% Cohort Ratio",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**{'نمونه داده‌های ثبت‌شده در پایگاه پیما (۱۵ سطر اول):' if is_rtl else 'Raw Sample Records (First 15 Rows):'}**")
    
    # Styled dataframe with right-aligned text
    if is_rtl:
        styled_sample = data.head(15).style.set_properties(**{
            'text-align': 'right',
            'direction': 'rtl',
            'font-family': 'Vazirmatn, Tahoma, sans-serif'
        })
        st.dataframe(styled_sample, use_container_width=True)
    else:
        st.dataframe(data.head(15), use_container_width=True)

    # Column explanations in Persian/English expander
    expander_label = "📖 راهنمای متغیرها و نام ستون‌های داده" if is_rtl else "📖 Dataset Columns Glossary"
    with st.expander(expander_label, expanded=False):
        if is_rtl:
            glossary_html = (
                '<div style="direction: rtl; text-align: right; line-height: 1.9; font-size: 0.92rem; color: #cbd5e1;">'
                '• <b>Pregnancies:</b> تعداد دفعات بارداری بیمار.<br>'
                '• <b>Glucose:</b> غلظت گلوکز پلاسما در دقیقه ۱۲۰ آزمون تحمل گلوکز خوراکی (mg/dL).<br>'
                '• <b>BloodPressure:</b> فشار خون دیاستولیک بر حسب میلی‌متر جیوه (mm Hg).<br>'
                '• <b>SkinThickness:</b> ضخامت چین پوستی عضله سه‌سر بازو بر حسب میلی‌متر (mm).<br>'
                '• <b>Insulin:</b> سطح سرمی انسولین ۲ ساعت پس از مصرف گلوکز (μU/mL).<br>'
                '• <b>BMI:</b> شاخص توده بدنی (کیلوگرم بر توان دوم قد بر حسب متر).<br>'
                '• <b>DiabetesPedigreeFunction:</b> تابع ریاضی محاسبه سابقه ژنتیکی و خویشاوندی دیابت.<br>'
                '• <b>Age:</b> سن شناسنامه‌ای بیمار به سال.<br>'
                '• <b>Outcome:</b> تشخیص بالینی نهایی (۰ = سالم / منفی، ۱ = مبتلا به دیابت / مثبت).'
                '</div>'
            )
        else:
            glossary_html = (
                '<div style="direction: ltr; text-align: left; line-height: 1.9; font-size: 0.92rem; color: #cbd5e1;">'
                '• <b>Pregnancies:</b> Number of times pregnant.<br>'
                '• <b>Glucose:</b> Plasma glucose concentration at 2 hours in an oral glucose tolerance test (mg/dL).<br>'
                '• <b>BloodPressure:</b> Diastolic blood pressure (mm Hg).<br>'
                '• <b>SkinThickness:</b> Triceps skin fold thickness (mm).<br>'
                '• <b>Insulin:</b> 2-Hour serum insulin level (μU/mL).<br>'
                '• <b>BMI:</b> Body mass index (weight in kg / (height in m)^2).<br>'
                '• <b>DiabetesPedigreeFunction:</b> Genetic scoring synthesizing family history of diabetes.<br>'
                '• <b>Age:</b> Patient chronological age in years.<br>'
                '• <b>Outcome:</b> Class variable (0 = Negative / Healthy, 1 = Positive / Diabetic).'
                '</div>'
            )
        st.markdown(glossary_html, unsafe_allow_html=True)

    with st.expander(t("data_summary_expander", active_lang), expanded=False):
        if is_rtl:
            styled_desc = data.describe().round(2).style.set_properties(**{
                'text-align': 'right',
                'direction': 'rtl',
                'font-family': 'Vazirmatn, Tahoma, sans-serif'
            })
            st.dataframe(styled_desc, use_container_width=True)
        else:
            st.dataframe(data.describe().round(2), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 5: COMPREHENSIVE USER GUIDE & TUTORIAL (100% RTL CONTAINER)
# -----------------------------------------------------------------------------
with tab_guide:
    if active_lang == "fa":
        st.markdown(
            f"""
            <div class="guide-rtl-wrapper">
                {GUIDE_FA}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(GUIDE_EN, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 6: SYSTEM ARCHITECTURE & IMPLEMENTATION (معماری سیستم و چرایی طراحی)
# -----------------------------------------------------------------------------
with tab_arch:
    if active_lang == "fa":
        st.markdown(
            f"""
            <div class="guide-rtl-wrapper">
                {ARCHITECTURE_FA}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(ARCHITECTURE_EN, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# FOOTER & MEDICAL DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
    <div style="font-size: 0.82rem; color: #64748b; line-height: 1.7; text-align: center; max-width: 1000px; margin: 0 auto; direction: {'rtl' if is_rtl else 'ltr'};">
        {t("footer_disclaimer", active_lang)}
    </div>
    """,
    unsafe_allow_html=True,
)
