"""
DiabPredict - Localization & Translations Module
Supports Persian (Farsi) and English with medical-grade terminology.
"""

TEXTS = {
    # ---------------- PAGE CONFIG & HERO ----------------
    "page_title": {
        "fa": "DiabPredict | سامانه هوشمند ارزیابی و پیش‌بینی ریسک دیابت",
        "en": "DiabPredict | Intelligent Diabetes Risk Predictor",
    },
    "hero_badge": {
        "fa": "سامانه هوشمند پشتیبان تصمیم‌گیری بالینی (Clinical ML Support)",
        "en": "Clinical ML Decision Support System",
    },
    "hero_title": {
        "fa": "سامانه هوشمند پیش‌بینی خطر دیابت",
        "en": "DiabPredict: Intelligent Diabetes Risk Predictor",
    },
    "hero_desc": {
        "fa": "موتور تشخیصی و غربالگری پیشرفته با بهره‌گیری از مدل گروهی جنگل تصادفی (Random Forest Classifier) آموزش‌دیده بر روی پایگاه داده استاندارد پیما (Pima Indians Dataset). کالیبره‌شده برای طبقه‌بندی بلادرنگ خطر دیابت نوع ۲ بر اساس شاخص‌های زیستی و متابولیک.",
        "en": "An enterprise-grade diagnostic estimation engine leveraging an ensemble Random Forest Classifier trained on the benchmark Pima Indians Diabetes Dataset. Calibrated for real-time risk stratification based on metabolic and phenotypic biomarkers.",
    },

    # ---------------- METRICS BAR ----------------
    "metric_accuracy_label": {
        "fa": "🎯 دقت پایه مدل",
        "en": "🎯 Model Baseline Accuracy",
    },
    "metric_accuracy_delta": {
        "fa": "مجموعه آزمون طبقه‌بندی‌شده ۲۰٪",
        "en": "Stratified 80/20 Test Set",
    },
    "metric_accuracy_help": {
        "fa": "ارزیابی‌شده روی ۱۵۴ نمونه آزمون مستقل با random_state=42",
        "en": "Evaluated on held-out test split (n = 154) with random_state=42",
    },
    "metric_algo_label": {
        "fa": "🌲 معماری الگوریتم",
        "en": "🌲 Algorithm Architecture",
    },
    "metric_algo_val": {
        "fa": "جنگل تصادفی (Random Forest)",
        "en": "Random Forest",
    },
    "metric_algo_delta": {
        "fa": "۱۰۰ درخت تصمیم گروهی",
        "en": "100 Ensembled Trees",
    },
    "metric_algo_help": {
        "fa": "یادگیری جمعی بگینگ با معیار تفکیک ناخالصی جینی (Gini)",
        "en": "Bagging ensemble with Gini impurity splitting criterion",
    },
    "metric_cohort_label": {
        "fa": "📊 جامعه آماری نمونه‌ها",
        "en": "📊 Cohort Size",
    },
    "metric_cohort_val": {
        "fa": "{total:,} بیمار",
        "en": "{total:,} Patients",
    },
    "metric_cohort_delta": {
        "fa": "{train} آموزش / {test} آزمون",
        "en": "{train} Train / {test} Test",
    },
    "metric_cohort_help": {
        "fa": "تعداد کل پرونده‌ها در پایگاه داده دیابت سرخ‌پوستان پیما",
        "en": "Full sample count in the Pima Indians Diabetes dataset",
    },
    "metric_cache_label": {
        "fa": "⚡ وضعیت کَش و اجرا",
        "en": "⚡ Caching Status",
    },
    "metric_cache_val": {
        "fa": "فعال و بهینه‌سازی‌شده",
        "en": "Active & Optimized",
    },
    "metric_cache_delta": {
        "fa": "پاسخ‌دهی آنی اسلایدرها",
        "en": "Zero Latency Slider State",
    },
    "metric_cache_help": {
        "fa": "بهینه‌سازی شده با st.cache_data و st.cache_resource برای استنتاج زیر میلی‌ثانیه‌ای",
        "en": "@st.cache_data & @st.cache_resource keep inference sub-millisecond",
    },

    # ---------------- SIDEBAR ----------------
    "sidebar_language": {
        "fa": "🌐 زبان / Language",
        "en": "🌐 Language / زبان",
    },
    "sidebar_header_title": {
        "fa": "📋 پارامترهای بالینی بیمار",
        "en": "📋 Patient Parameters",
    },
    "sidebar_header_desc": {
        "fa": "شاخص‌های پزشکی و بیومارکرها را برای بررسی ریسک دیابت تنظیم کنید",
        "en": "Adjust clinical indicators to assess diabetes risk",
    },
    "presets_expander": {
        "fa": "⚡ الگوهای سریع بالینی و بازنشانی",
        "en": "⚡ Clinical Presets & Reset",
    },
    "preset_low": {
        "fa": "🟢 پرونده کم‌خطر",
        "en": "🟢 Low Risk",
    },
    "preset_high": {
        "fa": "🔴 پرونده پرخطر",
        "en": "🔴 High Risk",
    },
    "preset_reset": {
        "fa": "🔄 بازنشانی به مقادیر پیش‌فرض",
        "en": "🔄 Reset to Defaults",
    },
    "predict_button": {
        "fa": "🩺 ارزیابی خطر دیابت",
        "en": "Predict Diabetes Risk",
    },
    "sidebar_footer_text": {
        "fa": "طراحی‌شده با scikit-learn و Streamlit<br>اجرای بهینه پایپ‌لاین با کش هوشمند",
        "en": "Powered by scikit-learn & Streamlit<br>Cached Pipeline Execution",
    },

    # ---------------- SLIDERS ----------------
    "slider_preg": {
        "fa": "تعداد بارداری‌ها (Pregnancies)",
        "en": "Pregnancies",
    },
    "slider_preg_help": {
        "fa": "تعداد دفعات بارداری بیمار",
        "en": "Number of times pregnant",
    },
    "slider_gluc": {
        "fa": "گلوکز پلاسما (mg/dL)",
        "en": "Glucose (mg/dL)",
    },
    "slider_gluc_help": {
        "fa": "غلظت گلوکز پلاسما ۲ ساعت پس از تست تحمل گلوکز خوراکی (OGTT)",
        "en": "Plasma glucose concentration at 2 hours in an oral glucose tolerance test",
    },
    "slider_bp": {
        "fa": "فشار خون دیاستولیک (mm Hg)",
        "en": "Blood Pressure (mm Hg)",
    },
    "slider_bp_help": {
        "fa": "فشار خون دیاستولیک بر حسب میلی‌متر جیوه (فشار مینیمم)",
        "en": "Diastolic blood pressure (mm Hg)",
    },
    "slider_skin": {
        "fa": "ضخامت چین پوستی (mm)",
        "en": "Skin Thickness (mm)",
    },
    "slider_skin_help": {
        "fa": "ضخامت چین پوستی عضله سه‌سر بازو بر حسب میلی‌متر",
        "en": "Triceps skin fold thickness (mm)",
    },
    "slider_ins": {
        "fa": "انسولین سرم ۲ ساعته (μU/mL)",
        "en": "Insulin (μU/mL)",
    },
    "slider_ins_help": {
        "fa": "سطح انسولین سرم خون ۲ ساعت پس از مصرف گلوکز",
        "en": "2-Hour serum insulin level",
    },
    "slider_bmi": {
        "fa": "شاخص توده بدنی (BMI)",
        "en": "BMI (Body Mass Index)",
    },
    "slider_bmi_help": {
        "fa": "شاخص توده بدنی = وزن (کیلوگرم) تقسیم بر توان دوم قد (متر)",
        "en": "Body Mass Index = Weight in kg / (Height in m)^2",
    },
    "slider_dpf": {
        "fa": "تابع شجره‌نامه دیابت (DPF)",
        "en": "Diabetes Pedigree Function",
    },
    "slider_dpf_help": {
        "fa": "نمره ژنتیکی سنتزکننده سابقه خانوادگی ابتلا به دیابت و نسبت خویشاوندی",
        "en": "Genetic score representing diabetes family history synthesis",
    },
    "slider_age": {
        "fa": "سن بیمار (سال)",
        "en": "Age (years)",
    },
    "slider_age_help": {
        "fa": "سن شناسنامه‌ای بیمار به سال",
        "en": "Patient chronological age",
    },

    # ---------------- ACTION BAR ----------------
    "action_hint": {
        "fa": "👈 برای اجرای تحلیل هوش مصنوعی روی پارامترهای تنظیم‌شده در نوار کناری، دکمه بالا را لمس کنید.",
        "en": "👈 Click the button to run real-time inference on the current sidebar parameter configuration.",
    },

    # ---------------- TABS ----------------
    "tab_prediction": {
        "fa": "🩺 ارزیابی و پیش‌بینی ریسک",
        "en": "🩺 Prediction & Risk Assessment",
    },
    "tab_patient": {
        "fa": "📋 خلاصه شاخص‌های بیمار",
        "en": "📋 Patient Biomarker Summary",
    },
    "tab_analytics": {
        "fa": "📊 هوش مدل و تحلیل داده‌ها",
        "en": "📊 Model Intelligence & Analytics",
    },
    "tab_data": {
        "fa": "🔍 کاوشگر مجموعه داده",
        "en": "🔍 Dataset Explorer",
    },
    "tab_guide": {
        "fa": "📖 راهنما و آموزش جامع",
        "en": "📖 User Guide & Tutorial",
    },

    # ---------------- TAB 1: PREDICTION ----------------
    "welcome_info": {
        "fa": "👋 **خوش آمدید!** شاخص‌های بالینی بیمار را از منوی کناری سمت راست تنظیم کرده و سپس بر روی دکمه **«ارزیابی خطر دیابت»** کلیک کنید تا پیش‌بینی و برآورد احتمالی در لحظه تولید شود.",
        "en": "👋 **Welcome!** Adjust the patient parameters in the sidebar on the left and click **'Predict Diabetes Risk'** to generate the real-time machine learning prediction and probability distribution.",
    },
    "risk_eval_title": {
        "fa": "نتیجه ارزیابی تشخیصی خطر دیابت",
        "en": "Diagnostic Risk Evaluation",
    },
    "high_risk_alert": {
        "fa": "🚨 **خطر بالای ابتلا به دیابت شناسایی شد (نتیجه مثبت)**\n\nمدل هوشمند این پروفایل بالینی را به عنوان **مستعد دیابت (Positive)** با احتمال تخمینی **{prob:.1f}٪** طبقه‌بندی کرده است.",
        "en": "🚨 **High Risk of Diabetes Detected (Positive Result)**\n\nThe ensemble model classifies this patient profile as **Positive for Diabetes Risk** with an estimated disease probability of **{prob:.1f}%**.",
    },
    "low_risk_alert": {
        "fa": "✅ **خطر پایین ابتلا به دیابت ارزیابی شد (نتیجه منفی)**\n\nمدل هوشمند این پروفایل بالینی را به عنوان **سالم / کم‌خطر (Negative)** با احتمال سلامت **{healthy_prob:.1f}٪** (نمره خطر: **{prob:.1f}٪**) ارزیابی کرده است.",
        "en": "✅ **Low Risk of Diabetes Detected (Negative Result)**\n\nThe ensemble model classifies this patient profile as **Negative for Diabetes Risk** with an estimated healthy probability of **{healthy_prob:.1f}%** (Risk Score: **{prob:.1f}%**).",
    },
    "metric_risk_prob": {
        "fa": "احتمال پیش‌بینی‌شده ابتلا",
        "en": "Predicted Risk Probability",
    },
    "delta_vs_threshold_pos": {
        "fa": "+{val:.1f}٪ نسبت به مرز تصمیم ۵۰٪",
        "en": "+{val:.1f}% vs 50% Threshold",
    },
    "delta_vs_threshold_neg": {
        "fa": "-{val:.1f}٪ نسبت به مرز تصمیم ۵۰٪",
        "en": "-{val:.1f}% vs 50% Threshold",
    },
    "tier_low": {
        "fa": "پروفایل کم‌خطر و نرمال",
        "en": "Low Risk Profile",
    },
    "tier_moderate": {
        "fa": "ریسک متوسط / نیازمند غربالگری دوره‌ای",
        "en": "Moderate / Elevated Risk",
    },
    "tier_high": {
        "fa": "خطر بالا / هشدار اقدام بالینی فوری",
        "en": "High Risk / Critical Alert",
    },
    "scale_title": {
        "fa": "نوار مقیاس خطر دیابت (احتمال ۰٪ تا ۱۰۰٪):",
        "en": "Risk Scale Gauge (0% – 100% Probability):",
    },
    "scale_low": {
        "fa": "🟢 **۰٪ - ۳۰٪**: ریسک پایین",
        "en": "🟢 **0% - 30%**: Low Risk",
    },
    "scale_mod": {
        "fa": "🟡 **۳۰٪ - ۶۰٪**: ریسک متوسط / پایش",
        "en": "🟡 **30% - 60%**: Moderate / Monitoring",
    },
    "scale_high": {
        "fa": "🔴 **۶۰٪ - ۱۰۰٪**: ریسک بالا / اقدام درمانی",
        "en": "🔴 **60% - 100%**: High / Actionable",
    },
    "drivers_header": {
        "fa": "عوامل فیزیولوژیک کلیدی و توصیه‌های بالینی",
        "en": "Key Physiological Risk Drivers",
    },
    "obs_header": {
        "fa": "🔍 مشاهدات فاکتورهای بالینی بیمار:",
        "en": "🔍 Patient Parameter Observations:",
    },
    "obs_all_normal": {
        "fa": "تمامی شاخص‌های اصلی بیمار در محدوده استاندارد فیزیولوژیک قرار دارند.",
        "en": "All primary patient indicators are within typical physiological ranges.",
    },
    "obs_glucose_high": {
        "fa": "• **قند خون بالا ({val} mg/dL)**: افزایش سطح قند پلاسما پس از ۲ ساعت نشان‌دهنده اختلال در تحمل گلوکز و مقاومت به انسولین است.",
        "en": "• **High Glucose ({val} mg/dL)**: Elevated 2-hour glucose levels indicate impaired glucose tolerance.",
    },
    "obs_glucose_opt": {
        "fa": "• **قند خون ایده‌آل ({val} mg/dL)**: کاملاً در بازه نرمال قند خون پس از غذا قرار دارد.",
        "en": "• **Optimal Glucose ({val} mg/dL)**: Well within normal fasting/postprandial range.",
    },
    "obs_bmi_high": {
        "fa": "• **شاخص توده بدنی بالا ({val:.1f})**: قرارگیری BMI در رده چاقی (≥ ۳۰.۰) مقاومت به انسولین را به شدت افزایش می‌دهد.",
        "en": "• **High BMI ({val:.1f})**: BMI in the obese category (≥30.0) significantly elevates insulin resistance.",
    },
    "obs_bmi_opt": {
        "fa": "• **شاخص توده بدنی نرمال ({val:.1f})**: قرارگیری وزن در محدوده سالم ریسک سندرم متابولیک را کاهش می‌دهد.",
        "en": "• **Normal BMI ({val:.1f})**: Healthy weight category lowers metabolic risk.",
    },
    "obs_bp_high": {
        "fa": "• **فشار خون دیاستولیک افزایش‌یافته ({val} mm Hg)**: فشار دیاستولیک بالای ۸۰ نیازمند پایش سیستم قلبی‌عروقی است.",
        "en": "• **Elevated Blood Pressure ({val} mm Hg)**: Diastolic pressure above 80 mm Hg warrants cardiovascular monitoring.",
    },
    "obs_age_high": {
        "fa": "• **سن ({val} سال)**: سن بالای ۴۵ سال از فاکتورهای شناخته‌شده افزایش احتمال دیابت نوع ۲ است.",
        "en": "• **Age ({val} yrs)**: Age ≥45 is a clinically recognized baseline risk factor.",
    },
    "recs_header": {
        "fa": "💡 توصیه‌ها و اقدامات پیشنهادی بالینی:",
        "en": "💡 Clinical Recommendations:",
    },
    "recs_high": {
        "fa": "۱. انجام آزمایش‌های تکمیلی تشخیصی: **هموگلوبین گلیکوزیله (HbA1c)** و قند خون ناشتا (FBS).\n۲. مراجعه به پزشک متخصص غدد یا پزشک عمومی جهت بررسی تکمیلی.\n۳. اصلاح فوری رژیم غذایی (کاهش کربوهیدرات‌های ساده) و حداقل ۱۵۰ دقیقه فعالیت هوازی منظم در هفته.",
        "en": "1. Schedule comprehensive laboratory tests: **HbA1c** and fasting blood glucose.\n2. Consult with an endocrinologist or primary care physician.\n3. Initiate structured dietary intervention (low glycemic index foods) and 150 min/wk aerobic exercise.",
    },
    "recs_low": {
        "fa": "۱. تداوم فعالیت بدنی منظم و رعایت الگوهای تغذیه سالم و سرشار از فیبر.\n۲. غربالگری دوره‌ای سالیانه قند و چربی خون.\n۳. کنترل دوره‌ای وزن، شاخص توده بدنی و فشار خون.",
        "en": "1. Maintain current physical activity levels and balanced nutritional habits.\n2. Routine annual metabolic screening recommended.\n3. Continue monitoring BMI and blood pressure periodically.",
    },

    # ---------------- TAB 2: BIOMARKERS ----------------
    "patient_tab_header": {
        "fa": "مقایسه شاخص‌های بیمار با استانداردهای رفرنس بالینی",
        "en": "Current Patient Input Values vs. Clinical Reference Norms",
    },
    "col_param": {"fa": "شاخص بالینی", "en": "Parameter"},
    "col_val": {"fa": "مقدار بیمار", "en": "Current Value"},
    "col_ref": {"fa": "محدوده نرمال مرجع", "en": "Normal Reference"},
    "col_unit": {"fa": "واحد سنجش", "en": "Unit"},
    "col_status": {"fa": "وضعیت بالینی", "en": "Status"},

    # ---------------- TAB 3: ANALYTICS ----------------
    "analytics_tab_header": {
        "fa": "تحلیل اهمیت ویژگی‌ها در مدل جنگل تصادفی (Random Forest)",
        "en": "Random Forest Feature Importance Analysis",
    },
    "analytics_tab_caption": {
        "fa": "میزان وزن و اهمیت نسبی هریک از نشانگرهای زیستی بر مبنای شاخص ناخالصی جینی در ۱۰۰ درخت تصمیم.",
        "en": "Relative Gini importance of each biomarker in the trained 100-tree Random Forest model.",
    },
    "cm_header": {
        "fa": "🧪 ماتریس درهم‌ریختگی روی داده‌های آزمون (Confusion Matrix)",
        "en": "🧪 Test Set Confusion Matrix",
    },
    "cm_pred_neg": {"fa": "پیش‌بینی سالم (منفی)", "en": "Predicted Negative"},
    "cm_pred_pos": {"fa": "پیش‌بینی دیابت (مثبت)", "en": "Predicted Positive"},
    "cm_act_neg": {"fa": "واقعی سالم (منفی)", "en": "Actual Negative"},
    "cm_act_pos": {"fa": "واقعی دیابت (مثبت)", "en": "Actual Positive"},
    "cm_caption": {
        "fa": "ارزیابی روی {test_count} مورد آزمون مستقل (تقسیم‌بندی طبقه‌بندی‌شده ۸۰/۲۰). دقت مدل: **{acc:.2f}٪**.",
        "en": "Evaluated on {test_count} test cases (80/20 Stratified Split). Baseline accuracy: **{acc:.2f}%**.",
    },
    "weight_col": {"fa": "وزن نسبی (%)", "en": "Relative Weight (%)"},
    "feature_col": {"fa": "ویژگی / بیومارکر", "en": "Feature"},

    # ---------------- TAB 4: DATASET ----------------
    "data_tab_header": {
        "fa": "کاوش در پایگاه داده دیابت پیما (Pima Indians)",
        "en": "Pima Indians Diabetes Dataset",
    },
    "data_tab_caption": {
        "fa": "دریافت‌شده مستقیماً از منبع معتبر با بارگذاری بهینه در کَش حافظه.",
        "en": "Directly loaded from verified repository (Cached with @st.cache_data).",
    },
    "data_summary_expander": {
        "fa": "📈 خلاصه آماری توصیفی داده‌ها (Describe)",
        "en": "📈 Dataset Summary Statistics",
    },

    # ---------------- FOOTER ----------------
    "footer_disclaimer": {
        "fa": "<b>سلب مسئولیت پزشکی و حقوقی:</b> سامانه DiabPredict یک مدل اثبات مفهوم هوش مصنوعی بوده و صرفاً برای اهداف آموزشی و غربالگری کمکی توسعه یافته است. این سامانه تحت هیچ شرایطی جایگزین تشخیص قطعی پزشک، قضاوت بالینی تخصصی یا آزمایش‌های پاراکلینیکی رسمی نمی‌باشد. لطفاً برای هرگونه تصمیم درمانی با پزشک معتمد مشورت نمایید.",
        "en": "<b>Clinical Disclaimer:</b> DiabPredict is an artificial intelligence demonstration model intended strictly for educational and screening assistance purposes. It does not replace professional medical diagnosis, clinical judgment, or laboratory consultation. Always consult a licensed healthcare provider for medical decisions.",
    },
}

def t(key: str, lang: str = "fa", **kwargs) -> str:
    """Retrieve translated string by key and language, formatting if kwargs provided."""
    entry = TEXTS.get(key, {})
    text = entry.get(lang, entry.get("en", key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text

