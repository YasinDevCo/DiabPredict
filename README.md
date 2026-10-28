# 🩺 DiabPredict - Intelligent Diabetes Risk Predictor
### سامانه هوشمند ارزیابی و غربالگری ریسک دیابت نوع ۲

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade clinical decision-support application built with **Streamlit** and **scikit-learn** that predicts Type 2 Diabetes risk in real-time. Features full bilingual support (**Persian 🇮🇷 & English 🇬🇧**), native RTL typography, and an ensemble **Random Forest Classifier** trained on the benchmark **Pima Indians Diabetes Dataset**.

---

## 🌟 Key Features / ویژگی‌های کلیدی

- **🌐 Bilingual & RTL Support (دو زبانه و راست‌چین):**
  - Seamless toggle between **Persian (فارسی)** and **English**.
  - Custom RTL layout with Google Font `Vazirmatn` for Persian speakers.
- **🌲 High-Performance ML Engine:**
  - 100-tree **Random Forest Classifier** with Gini impurity splitting criterion.
  - Stratified 80/20 train/test split achieving >75% baseline diagnostic accuracy.
- **⚡ Sub-Millisecond Inference:**
  - Optimized caching with `@st.cache_data` and `@st.cache_resource` for zero slider latency.
- **🩺 Clinical Intelligence & Recommendations:**
  - Real-time probability estimation with 3 risk stratification tiers (Low, Moderate, High).
  - Automated biomarker observation (highlighting high glucose, elevated BMI, hypertension).
  - Personalized evidence-based clinical recommendations and HbA1c screening advice.
- **📊 Deep Analytics & Model Transparency:**
  - Interactive feature importance visualization.
  - Test set confusion matrix.
  - Raw dataset explorer and descriptive statistics.
- **📖 Comprehensive User Guide & Tutorial (راهنما و آموزش جامع):**
  - Built-in multi-section documentation explaining the model architecture, biomarkers, FAQ, and medical disclaimer.

---

## 🚀 Quick Start / نحوه راه‌اندازی و اجرا

### 1. Clone the repository:
```bash
git clone https://github.com/YasinDevCo/DiabPredict.git
cd DiabPredict
```

### 2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the application:
```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 📋 Clinical Biomarkers / شاخص‌های بالینی مورد بررسی

| Parameter | Unit | Normal Reference |
| :--- | :---: | :---: |
| **Pregnancies** (تعداد بارداری‌ها) | Count | 0 - 4 |
| **Glucose** (گلوکز پلاسما در تست تحمل ۲ ساعته) | mg/dL | 70 - 140 |
| **Blood Pressure** (فشار خون دیاستولیک) | mm Hg | 60 - 80 |
| **Skin Thickness** (ضخامت چین پوستی عضله سه‌سر بازو) | mm | 10 - 30 |
| **Insulin** (انسولین سرم ۲ ساعته) | μU/mL | 16 - 166 |
| **BMI** (شاخص توده بدنی) | kg/m² | 18.5 - 24.9 |
| **Diabetes Pedigree Function** (شاخص شجره‌نامه ژنتیکی) | Index | < 0.50 |
| **Age** (سن بیمار) | Years | 21 - 45 |

---

## ⚠️ Medical Disclaimer / سلب مسئولیت پزشکی
> **EN:** DiabPredict is an artificial intelligence demonstration model intended strictly for educational and screening assistance purposes. It does not replace professional medical diagnosis, clinical judgment, or laboratory consultation.
>
> **FA:** سامانه DiabPredict یک مدل اثبات مفهوم هوش مصنوعی بوده و صرفاً برای اهداف آموزشی و غربالگری کمکی توسعه یافته است. این سامانه جایگزین تشخیص پزشک یا آزمایش‌های پاراکلینیکی رسمی نمی‌باشد.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
