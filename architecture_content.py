"""
DiabPredict - System Architecture & Implementation Details Module
Deep dive into technical decisions, ML pipeline design, and UI/UX engineering rationale.
"""

ARCHITECTURE_FA = """
# 🏗️ معماری جامع سیستم و جزئیات پیاده‌سازی گام‌به‌گام

این بخش به صورت تخصصی و موشکافانه، معماری مهندسی نرم‌افزار، پشت‌صحنه مدل هوش مصنوعی، تصمیمات طراحی و چرایی (Rationale) انتخاب هر راهکار در توسعه سامانه **DiabPredict** را تبیین می‌کند.

---

### ۱. معماری کلی نرم‌افزار و تفکیک دغدغه‌ها (Separation of Concerns)

سامانه DiabPredict بر اساس یک معماری چندلایه و ماژولار طراحی شده است:

```
┌────────────────────────────────────────────────────────┐
│                   رابط کاربری کاربر                    │
│        (Streamlit UI + Vazirmatn Dynamic RTL)          │
└───────────────────────────┬────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│     هسته      │   │  ماژول ترجمه  │   │ مستندات علمی  │
│    برنامه     │   │ و بومی‌سازی    │   │  و معماری فنی │
│   (app.py)    │   │(translations) │   │ (guide/arch)  │
└───────┬───────┘   └───────────────┘   └───────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│               پایپ‌لاین یادگیری ماشین                 │
│    Random Forest (100 Trees) + Caching Optimization   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                    لایه داده‌ها                        │
│   Pima Indians Dataset + Offline Fallback Mechanism    │
└────────────────────────────────────────────────────────┘
```

#### چرایی این معماری (Why?):
* **تفکیک دغدغه‌ها (SoC):** اگر تمام متون دوزبانه، قالب‌های مستندات و کدهای یادگیری ماشین در یک فایل واحد نوشته می‌شدند، حجم کد به بیش از ۳۰۰۰ خط نامرتب می‌رسید. تفکیک متون در `translations.py` و مستندات در فایل‌های اختصاصی، نگهداری و خطایابی کد را بسیار ساده کرده است.
* **تغییر زبان بدون دستکاری منطق مدل:** افزودن زبان‌های جدید (مثلاً عربی یا فرانسوی) صرفاً با افزودن یک کلید در دیکشنری ترجمه امکان‌پذیر است بدون اینکه به منطق محاسباتی مدل آسیبی برسد.

---

### ۲. گام اول: مهندسی لایه داده و پایداری آفلاین (Data Resilience)

#### چگونگی پیاده‌سازی:
1. استفاده از مجموعه داده استاندارد **Pima Indians Diabetes Database** شامل ۷۶۸ نمونه بالینی با ۸ متغیر مستقل و ۱ برچسب هدف.
2. طراحی تابع `load_data()` مجهز به دکوراتور `@st.cache_data`.
3. ایجاد مکانیسم دوگانه ذخیره‌سازی محلی و دانلود اینترنتی:
   - برنامه ابتدا بررسی می‌کند آیا فایل محلی `diabetes.csv` روی دیسک وجود دارد یا خیر.
   - در صورت وجود، بدون نیاز به اینترنت داده را می‌خواند.
   - در غیر این صورت، داده را از مخزن معتبر دانلود کرده و یک نسخه محلی برای دفعات بعد ذخیره می‌نماید.

#### چرایی این تصمیمات (Why?):
* **چرا `@st.cache_data`؟** ماهیت فریم‌ورک Streamlit این است که با هر تغییر جزئی در اسلایدرها، کل کد از بالا تا پایین مجدداً اجرا می‌شود (Re-run). بدون کش، با هر حرکت اسلایدر کل دیتاست دوباره از گیت‌هاب دانلود می‌شد که این امر باعث ایجاد تاخیر چند ثانیه‌ای و احتمال مسدود شدن توسط گیت‌هاب (Rate Limiting) می‌شد. با کش، داده تنها یک بار در رم قرار می‌گیرد و دسترسی به آن صفر میلی‌ثانیه است.
* **چرا مکانیزم ذخیره محلی (Offline Fallback)؟** در محیط‌های بالینی، شبکه‌های بیمارستانی یا شرایط تحریم و اختلال اینترنت، وابستگی صرف به URL خارجی موجب شکست برنامه می‌شد. این مکانیزم تضمین می‌کند که نرم‌افزار در هر شرایطی بدون اینترنت ۱۰۰٪ قابل اجرا باشد.

---

### ۳. گام دوم: پایپ‌لاین یادگیری ماشین و دلایل انتخاب الگوریتم

#### چگونگی پیاده‌سازی:
1. جداسازی داده‌ها به دو بخش **۸۰٪ آموزش (Train) و ۲۰٪ آزمون (Test)** با استفاده از `stratify=y` و تنظیم دانه تصادفی ثابت (`random_state=42`).
2. ساخت و آموزش مدل یادگیری جمعی **Random Forest Classifier** با ۱۰۰ درخت تصمیم (Decision Trees).
3. استخراج ضریب اهمیت ویژگی‌ها بر مبنای کاهش ناخالصی جینی (Mean Decrease in Impurity).
4. ارزیابی مدل روی داده‌های آزمون و استخراج ماتریس درهم‌ریختگی (Confusion Matrix).
5. ذخیره‌سازی آبجکت مدل در رم با دکوراتور `@st.cache_resource`.

#### چرایی انتخاب Random Forest به جای سایر الگوریتم‌ها (Why Random Forest?):
* **در برابر رگرسیون لجستیک (Logistic Regression):** متغیرهای بالینی بدن انسان برهم‌کنش‌های خطی ساده ندارند. رابطه گلوکز، سن، ضخامت پوست و BMI کاملاً غیرخطی است. رگرسیون خطی از درک این تعاملات پیچیده ناتوان است.
* **در برابر شبکه‌های عصبی و یادگیری عمیق (Deep Learning):** یادگیری عمیق برای داده‌های جدولی کوچک (۷۶۸ رکورد و ۸ ویژگی) شدیداً مستعد بیش‌برازش (Overfitting) است و نیازمند منابع سنگین است. علاوه بر این، شبکه‌های عصبی اصطلاحاً «جعبه سیاه» هستند و پزشک نمی‌تواند وزن تک‌تک متغیرها را مستقیماً تحلیل کند.
* **مزیت درخت‌های گروهی (Ensemble Bagging):** جنگل تصادفی از تکنیک بگینگ (Bootstrap Aggregating) بهره می‌برد. میانگین‌گیری از ۱۰۰ درخت مانع از وابستگی مدل به نویزهای تصادفی داده‌ها شده و واریانس مدل را به شدت کاهش می‌دهد.
* **تفسیرپذیری بالینی (Explainability):** استخراج ضرایب اهمیت ویژگی‌ها (Feature Importance) به کادر درمان نشان می‌دهد که چه شاخص‌هایی بیشترین نقش را در خروجی داشته‌اند (مانند گلوکز و BMI).

---

### ۴. گام سوم: موتور استنتاج احتمالی و طبقه‌بندی ریسک (Risk Stratification)

#### چگونگی پیاده‌سازی:
1. استفاده از متد `model.predict_proba()` به جای متد ساده `model.predict()`.
2. استخراج درصد احتمال ابتلای بیمار به دیابت (بین ۰٪ تا ۱۰۰٪).
3. دسته‌بندی ریسک به سه سطح بالینی استاندارد:
   - 🟢 **کم‌خطر (۰٪ تا ۳۰٪):** وضعیت متابولیک طبیعی و پایدار.
   - 🟡 **خطر متوسط (۳۰٪ تا ۶۰٪):** نیازمند اصلاح تغذیه و پایش دوره‌ای.
   - 🔴 **خطر بالا (۶۰٪ تا ۱۰۰٪):** نیازمند مداخله و انجام آزمایش‌های تشخیصی نظیر HbA1c.
4. تحلیل خودکار شاخص‌های بحرانی بیمار و صدور توصیه‌های بالینی متناظر.

#### چرایی برآورد احتمالی به جای خروجی باینری (Why Probabilistic?):
* در علم پزشکی، تصمیم‌گیری بالینی صفر و یکی نیست. دو بیمار که هر دو توسط مدل به عنوان "سالم" ارزیابی شده‌اند، ممکن است احتمال یکی ۵٪ و دیگری ۴۹٪ باشد. بیماری که روی ۴۹٪ قرار دارد در مرز پرخطر است.
* نمایش احتمال پیوسته و گیج بصری به کاربر و پزشک این امکان را می‌دهد که تغییرات بیمار با تغییر دوز یا تغییر وزن را به صورت عددی ملموس پیگیری نمایند.

---

### ۵. گام چهارم: مهندسی چالش‌های رابط کاربری و راست‌چین‌سازی (UI/UX & RTL Engineering)

این بخش یکی از پیچیده‌ترین بخش‌های توسعه بوده است، زیرا Streamlit اساساً برای زبان‌های چپ‌به‌راست (LTR) ساخته شده است.

#### چالش ۱: به‌هم‌ریختگی آیکون‌های Material Symbols (مانند `_arrow_right` و `light_mode`)
* **علت ایجاد مشکل:** وقتی در CSS قانون `font-family: 'Vazirmatn' !important;` به صورت عمومی روی تگ‌های `span` اعمال شد، فونت اختصاصی آیکون‌های استریم‌لیت جایگزین گردید. مرورگر چون دیگر فونت آیکون را نداشت، نام لیگچر انگلیسی آیکون را به صورت متن خام روی دکمه‌ها و منوها انداخت!
* **راهکار حل مهندسی:** در فایل `app.py`، یک استثنای صریح برای تمامی المان‌های آیکونی (`[data-testid="stIconMaterial"]`, `[data-testid="stExpanderToggleIcon"]`, `.material-symbols-rounded`) تعریف شد تا فونت آن‌ها همواره روی `Material Symbols Rounded` قفل بماند.

#### چالش ۲: رندر شدن جدول HTML به صورت متن کد خام (The 4-Space Code Block Trap)
* **علت ایجاد مشکل:** در مشخصات CommonMark، هر متنی که با ۴ فاصله یا یک تب از ابتدای خط فاصله داشته باشد، به عنوان **بلوک کد (Code Block)** تفسیر می‌شود. در پایتون، وقتی یک f-string چندخطی درون یک حلقه با تورفتگی نوشته می‌شود، خطوط داخلی دارای ۸ تا ۱۲ فاصله ابتدایی می‌شوند. در نتیجه، Markdown آن را کُد تلقی کرده و به جای جدول، متن خام تگ‌ها را در یک کادر خاکستری چاپ می‌کرد! همچنین تداخل کاراکترهای دوزبانه باعث جابجایی اسلش و تولید `<td/>` می‌شد.
* **راهکار حل مهندسی:** تولید ردیف‌های جدول به صورت تک‌رشته‌های پیوسته بدون فاصله ابتدایی (Zero Leading Indentation) تغییر یافت. به این ترتیب Markdown دیگر آن را کد در نظر نگرفته و به عنوان یک المان HTML واقعی با `unsafe_allow_html=True` کامپایل می‌کند.

#### چالش ۳: راست‌چین‌سازی جدول‌ها و داده‌ها (Dataframe Alignment)
* **علت ایجاد مشکل:** کامپوننت `st.dataframe` استریم‌لیت بر روی بوم Canvas گرافیکی رندر می‌شود و استایل‌های معمولی CSS روی سلول‌های آن اعمال نمی‌شوند.
* **راهکار حل مهندسی:** استفاده از **Pandas Styler** (`df.style.set_properties(**{'text-align': 'right', 'direction': 'rtl'}))` که خصوصیات ترازبندی را مستقیماً به داده‌ها تزریق می‌کند، همراه با استایل‌های اختصاصی برای `st.table`.

---

### ۶. گام پنجم: مدیریت وضعیت برنامه (Session State Architecture)

#### چگونگی پیاده‌سازی:
تمام ۸ متغیر ورودی، متغیر زبان فعال (`fa`/`en`) و پرچم وضعیت صدور نتیجه (`has_predicted`) در `st.session_state` مدیریت می‌شوند. دکمه‌های الگوهای سریع بالینی (Presets) مستقیماً با تابع `apply_preset()` این وضعیت‌ها را مقداردهی می‌کنند.

#### چرایی (Why?):
* بدون Session State، هر بار که کاربر زبان برنامه را تغییر می‌داد یا تب جدیدی را انتخاب می‌کرد، فرم ریست می‌شد و مقادیر واردشده بیمار از بین می‌رفت. با این معماری، وضعیت کاربر در تمام طول جلسه کاری پایدار باقی می‌ماند.

---

### 📊 جدول خلاصه تصمیمات معماری (Architecture Summary Table)

| جزء سیستم | فناوری / راهکار | علت و چرایی تصمیم (Rationale) |
| :--- | :---: | :--- |
| **هسته الگوریتم** | Random Forest (100 Trees) | روابط غیرخطی قوی بین بیومارکرها، مقاومت در برابر بیش‌برازش در داده‌های کوچک، تفسیرپذیری بالا |
| **ارزیابی ریسک** | Continuous Probability (`predict_proba`) | اجتناب از خطای تصمیم‌گیری سیاه و سفید در پزشکی؛ امکان پایش روند بهبود یا تشدید بیمار |
| **کش کردن داده و مدل** | `@st.cache_data` & `@st.cache_resource` | کاهش زمان پاسخ‌دهی به زیر ۵ میلی‌ثانیه و جلوگیری از دانلود مجدد دیتاست با تغییر اسلایدرها |
| **پایداری داده‌ها** | Offline File Fallback (`diabetes.csv`) | اجرای کامل بدون نیاز به اینترنت؛ رفع وابستگی به سرورهای خارجی در محیط‌های درمانی |
| **تایپوگرافی و فونت** | Google Font Vazirmatn | بهترین خوانایی و استانداردهای طراحی برای زبان فارسی بر روی مانیتورها و تبلت‌ها |
| **حل تداخل آیکون‌ها** | Material Symbols Preservation CSS | جلوگیری از نمایش متن لیگچرهای آیکونی نظیر `_arrow_right` روی آکاردئون‌ها و منوها |
| **رندرینگ جداول** | Zero-Indent HTML Table + Pandas Styler | دور زدن تله بلوک کد در مارک‌داون و ایجاد جداول زیبای بالینی کاملاً راست‌چین |
"""

ARCHITECTURE_EN = """
# 🏗️ Comprehensive System Architecture & Engineering Rationale

This document provides a deep architectural walkthrough of **DiabPredict**, detailing the engineering decisions, machine learning pipeline, and the "Why" behind every design choice.

---

### 1. Overall System Architecture & Separation of Concerns

DiabPredict is organized into distinct decoupled modules:
* `app.py`: Streamlit core presentation layer, session state orchestration, and ML pipeline invocation.
* `translations.py`: Localization dictionary containing over 95 medical-grade translation keys.
* `guide_content.py`: Comprehensive user manual, clinical biomarker reference, and FAQ.
* `architecture_content.py`: Deep technical documentation, ML explainability, and UI/UX design rationale.

#### Rationale (Why?):
* **Separation of Concerns:** Consolidating bilingual strings, documentation, and ML logic into a single file would lead to an unmaintainable 3,000+ line monolith. Decoupling ensures that localization and content updates never risk breaking model logic.
* **Extensibility:** Adding future languages requires zero modifications to the core application logic.

---

### 2. Data Layer Engineering & Offline Resilience

#### Implementation:
1. Benchmark **Pima Indians Diabetes Dataset** (768 patient samples, 8 clinical features, 1 outcome target).
2. Cached via Streamlit's `@st.cache_data`.
3. Dual-storage strategy: The loader checks for local `diabetes.csv` first. If missing, it fetches once from GitHub and caches a local snapshot.

#### Rationale (Why?):
* **Why `@st.cache_data`?** Streamlit reruns the entire script upon every user interaction (e.g., slider change). Without caching, the CSV would be downloaded repeatedly, introducing severe latency and triggering GitHub rate limits. Caching keeps data retrieval sub-millisecond.
* **Why Offline Fallback?** Healthcare environments frequently operate under strict firewalls or intermittent connectivity. The local snapshot ensures the application runs 100% offline.

---

### 3. Machine Learning Architecture: Why Random Forest?

#### Implementation:
* Stratified 80/20 train/test split with fixed seed (`random_state=42`).
* 100-tree bagging ensemble using Gini impurity reduction criterion.
* Cached model persistence via `@st.cache_resource`.

#### Rationale (Why Random Forest over alternatives?):
* **Versus Logistic Regression:** Biomarkers (Glucose, Insulin, BMI, Age) exhibit complex, non-linear pathophysiological interactions that linear models fail to capture.
* **Versus Deep Learning:** Neural networks on small tabular datasets (768 records) suffer from severe overfitting, high latency, and lack interpretability.
* **Ensemble Stability:** Averaging over 100 decorrelated decision trees significantly reduces variance without increasing bias.
* **Clinical Interpretability:** Enables direct extraction of Gini feature importances, fulfilling the explainable AI (XAI) requirement for medical screening.

---

### 4. Probabilistic Risk Stratification Engine

#### Implementation:
* Utilizing `predict_proba()` to estimate a continuous disease probability (0% to 100%).
* 3-tier clinical classification:
  * 🟢 **Low Risk (0% - 30%):** Stable metabolic profile.
  * 🟡 **Moderate / Monitoring (30% - 60%):** Borderline indicators.
  * 🔴 **High Risk / Critical Alert (60% - 100%):** Actionable clinical indication.

#### Rationale (Why Probabilistic?):
* Clinical pathology is non-binary. A patient with 49% risk and a patient with 10% risk both fall into the "Negative" category in a hard binary classification, yet the 49% patient is on the threshold of disease. Continuous probability and visual progress scales empower nuanced clinical decision-making.

---

### 5. UI/UX & RTL Engineering Solutions

#### Problem 1: Broken Material Icons (`_arrow_right`, `light_mode`)
* **Root Cause:** Applying global `font-family: 'Vazirmatn' !important;` to general elements (`span`, `button`) overrides Streamlit's Material Symbols font, causing browsers to render the ligature text literally instead of drawing the icon glyph.
* **Solution:** Explicitly protecting all icon elements (`[data-testid="stIconMaterial"]`, `.material-symbols-rounded`) with scoped font rules.

#### Problem 2: HTML Tables Rendering as Raw Text
* **Root Cause (The 4-Space Rule):** In CommonMark/Markdown, any line with 4 or more leading spaces is parsed as an indented code block (`<pre><code>`). Indented multi-line f-strings triggered this rule, printing escaped HTML in a gray code box.
* **Solution:** Eliminating leading whitespace in string concatenation, allowing Markdown to compile genuine HTML tables with `unsafe_allow_html=True`.

#### Problem 3: DataFrame Alignment
* **Root Cause:** Streamlit's `st.dataframe` uses HTML5 Canvas, which ignores CSS text direction.
* **Solution:** Applying Pandas Styler properties (`style.set_properties(**{'text-align': 'right', 'direction': 'rtl'})`) for right-aligned cells and headers.
"""

