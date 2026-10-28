"""
DiabPredict - User Guide & Tutorial Content Module
Comprehensive educational guide, model architecture documentation, and step-by-step tutorial.
"""

GUIDE_FA = """
# 📖 راهنمای جامع و آموزش کاربری سامانه DiabPredict

---

### ۱. سامانه DiabPredict چیست؟
**سامانه هوشمند DiabPredict** یک بستر پیشرفته مبتنی بر هوش مصنوعی و یادگیری ماشین (Machine Learning) است که جهت **ارزیابی زودهنگام و غربالگری ریسک ابتلا به دیابت نوع ۲** طراحی شده است. 

این نرم‌افزار با تلفیق داده‌های بالینی، آزمایشگاهی و بیومارکرهای فردی، احتمال آماری ابتلای فرد به دیابت را در کسری از ثانیه محاسبه کرده و سطح خطر را همراه با توصیه‌های بالینی شخصی‌سازی‌شده ارائه می‌دهد.

---

### ۲. این برنامه برای چه کاری است و چه هدفی دارد؟
دیابت نوع ۲ یکی از شایع‌ترین بیماری‌های مزمن متابولیک در جهان است که اغلب در مراحل اولیه بدون علامت یا با علائم خاموش پیشرفت می‌کند. طبق گزارش فدراسیون بین‌المللی دیابت (IDF)، تقریباً **از هر دو فرد مبتلا به دیابت، یک نفر از بیماری خود بی‌خبر است**.

#### اهداف کلیدی سامانه:
* **غربالگری اولیه سریع و کم‌هزینه:** شناسایی افراد مستعد دیابت پیش از بروز عوارض غیرقابل جبران قلبی، عروقی و کلیوی.
* **پشتیبان تصمیم‌گیری بالینی (CDSS):** کمک به پزشکان و کادر درمان برای ارزیابی چندبعدی بیمار با در نظر گرفتن تعاملات پیچیده بین بیومارکرها.
* **افزایش آگاهی و خودمراقبتی:** ارتقای سواد سلامت کاربران و ترغیب به اصلاح بهنگام سبک زندگی، رژیم غذایی و فعالیت بدنی.
* **ابزار آموزشی و پژوهشی:** بستری تعاملی برای دانشجویان علوم پزشکی، هوش مصنوعی و داده‌کاوان جهت مشاهده کاربرد یادگیری ماشین در سلامت دیجیتال.

---

### ۳. پشتوانه علمی و معماری فنی هوش مصنوعی
سامانه DiabPredict از مدل قدرتمند **جنگل تصادفی (Random Forest Classifier)** استفاده می‌کند که یکی از استانداردترین و معتبرترین الگوریتم‌های یادگیری تحت نظارت (Supervised Learning) برای داده‌های بالینی است:

* **مجموعه داده استاندارد پیما (Pima Indians Diabetes Database):** این داده‌ها توسط مؤسسه ملی دیابت و بیماری‌های گوارشی و کلیوی ایالات متحده (NIDDK) جمع‌آوری شده و از معروف‌ترین بنچ‌مارک‌های جهانی در تحقیقات پزشکی دیابت به شمار می‌رود.
* **ساختار الگوریتم یادگیری جمعی (Ensemble):** مدل از **۱۰۰ درخت تصمیم مستقل** تشکیل شده است. در زمان ارزیابی، هر درخت بر اساس زیرمجموعه‌ای از داده‌ها و ویژگی‌ها رأی می‌دهد و نتیجه نهایی بر پایه تجمیع آرا (Bagging) مشخص می‌شود. این معماری مانع از بروز بیش‌برازش (Overfitting) شده و تعمیم‌پذیری بالایی دارد.
* **دقت و اعتبارسنجی:** مدل با تقسیم‌بندی دقیق **۸۰٪ آموزش و ۲۰٪ آزمون (Stratified)** ارزیابی شده و دقتی بالای ۷۵٪ را به ثبت رسانده است.

---

### ۴. آموزش گام‌به‌گام کار با برنامه (Step-by-Step Tutorial)

#### 🔹 گام اول: انتخاب زبان و تم
* در بالای منوی کناری (سایدبار سمت راست)، می‌توانید زبان رابط کاربری را بین **«فارسی»** و **«English»** تغییر دهید. در حالت فارسی، کلیه متون، نمودارها، جداول و چیدمان‌ها به صورت کاملاً راست‌چین (RTL) بهینه می‌شوند.

#### 🔹 گام دوم: تنظیم شاخص‌های بیمار
* در منوی کناری، اسلایدرهای اختصاصی برای **۸ فاکتور بالینی** بیمار تعبیه شده‌اند:
  * تعداد بارداری‌ها، قند خون (گلوکز ۲ ساعته)، فشار خون، ضخامت پوست، انسولین، شاخص توده بدنی (BMI)، شجره‌نامه دیابت (DPF) و سن.
* **نکته میان‌بر:** برای آزمایش سریع رفتار مدل، می‌توانید روی منوی کرکره‌ای **«⚡ الگوهای سریع بالینی و بازنشانی»** کلیک کرده و یکی از گزینه‌های **🟢 پرونده کم‌خطر**، **🔴 پرونده پرخطر** یا **🔄 بازنشانی** را بزنید تا مقادیر به صورت خودکار پر شوند.

#### 🔹 گام سوم: اجرای ارزیابی هوشمند
* بر روی دکمه بزرگ **«🩺 ارزیابی خطر دیابت»** در بالای صفحه یا منوی کناری کلیک کنید. مدل در لحظه شروع به تحلیل بردار ویژگی‌های بیمار می‌نماید.

#### 🔹 گام چهارم: تحلیل و تفسیر نتایج
پس از اجرا، در زبانه اول (**🩺 ارزیابی و پیش‌بینی ریسک**)، نتایج زیر قابل مشاهده است:
1. **پیام نتیجه قطعی:** هشدار قرمز در صورت تشخیص خطر بالا (Positive) یا پیام سبز در صورت ارزیابی کم‌خطر (Negative).
2. **درصد دقیق احتمال ریسک:** کارت متریک نشان‌دهنده درصد دقیق احتمال دیابت و فاصله آن از مرز تصمیم‌گیری ۵۰٪.
3. **نشانگر سطح ریسک (Tiers):**
   * 🟢 **پروفایل کم‌خطر (۰٪ تا ۳۰٪):** وضعیت نرمال متابولیک.
   * 🟡 **ریسک متوسط (۳۰٪ تا ۶۰٪):** نیازمند اصلاح تغذیه و پایش فصلی.
   * 🔴 **خطر بالا (۶۰٪ تا ۱۰۰٪):** وضعیت نیازمند مداخله و بررسی تخصصی پزشک.
4. **گیج بصری پیشرفت:** نوار وضعیت گرافیکی متناسب با درصد محاسبه‌شده.

#### 🔹 گام پنجم: بررسی مشاهدات و توصیه‌های بالینی
* در بخش پایین زبانه پیش‌بینی، سیستم هوشمند به‌صورت خودکار شاخص‌های بحرانی بیمار (مانند قند خون بالای ۱۴۰، چاقی یا فشار خون بالا) را برجسته کرده و توصیه‌های مراقبتی و آزمایش‌های لازم (مانند آزمایش HbA1c) را پیشنهاد می‌دهد.

#### 🔹 گام ششم: بهره‌گیری از سایر زبانه‌ها
* **📋 خلاصه شاخص‌های بیمار:** جدولی مقایسه‌ای از داده‌های واردشده در کنار محدوده‌های مرجع بین‌المللی و وضعیت سلامت هر شاخص.
* **📊 هوش مدل و تحلیل داده‌ها:** مشاهده نمودار میله‌ای اهمیت نسبی بیومارکرها (Feature Importance) و ماتریس درهم‌ریختگی (Confusion Matrix).
* **🔍 کاوشگر مجموعه داده:** دسترسی مستقیم به داده‌های مرجع Pima Indians، جدول نمونه‌ها و خلاصه آمار توصیفی (میانگین، انحراف معیار، چارک‌ها).

---

### ۵. راهنمای تخصصی شاخص‌های بالینی (Biomarkers Guide)

| شاخص بالینی | واحد | بازه نرمال مرجع | مفهوم فیزیولوژیک و اهمیت در دیابت |
| :--- | :---: | :---: | :--- |
| **گلوکز ۲ ساعته (Glucose)** | mg/dL | ۷۰ تا ۱۴۰ | مهم‌ترین بیومارکر؛ قند بالای ۱۴۰ نشانه اختلال تحمل گلوکز و بالای ۲۰۰ نشانه دیابت آشکار است. |
| **شاخص توده بدنی (BMI)** | kg/m² | ۱۸.۵ تا ۲۴.۹ | نسبت وزن به توان دو قد؛ مقادیر بالای ۳۰ به عنوان چاقی عامل اصلی مقاومت به انسولین است. |
| **سن (Age)** | سال | ۲۱ تا ۸۱ | با افزایش سن به‌ویژه پس از ۴۵ سالگی، کارایی سلول‌های بتای پانکراس کاهش می‌یابد. |
| **شجره‌نامه دیابت (DPF)** | ضریب | کمتر از ۰.۵۰ | نمره محاسباتی سابقه خانوادگی؛ سابقه دیابت در والدین و خویشاوندان ریسک ژنتیکی را بالا می‌برد. |
| **فشار خون دیاستولیک (BP)** | mm Hg | ۶۰ تا ۸۰ | فشار خون استراحتی؛ پرفشاری خون پیوند تنگاتنگی با سندرم متابولیک دارد. |
| **انسولین ۲ ساعته (Insulin)** | μU/mL | ۱۶ تا ۱۶۶ | انسولین ترشح‌شده پس از قند؛ سطوح بسیار بالا نشان‌دهنده مقاومت شدید سلول‌ها به انسولین است. |
| **ضخامت چین پوستی (Skin)** | mm | ۱۰ تا ۳۰ | معیار اندازه‌گیری چربی زیرپوستی بازو که بازتاب‌دهنده درصد کلی چربی بدن است. |
| **تعداد بارداری‌ها (Pregnancies)** | عدد | ۰ تا ۴ | بارداری‌های متعدد ممکن است ریسک سندروم مقاومت به انسولین پس از زایمان را افزایش دهد. |

---

### ۶. پرسش‌های متداول (FAQ)

**سؤال ۱: آیا این نرم‌افزار می‌تواند جایگزین آزمایشگاه یا ویزیت پزشک شود؟**  
*خیر.* این برنامه صرفاً ابزار غربالگری و تخمین آماری است. تشخیص قطعی دیابت تنها با بررسی آزمایش‌های تخصصی نظیر HbA1c، قند خون ناشتا (FBS) و معاینه بالینی توسط پزشک واجد شرایط انجام می‌پذیرد.

**سؤال ۲: اگر نتیجه ارزیابی «خطر بالا» شد چه باید کرد؟**  
خونسردی خود را حفظ کنید. نتیجه مثبت به معنی ابتلای قطعی نیست، بلکه هشداری است برای مراجعه به پزشک متخصص داخلی یا غدد جهت انجام آزمایش خون تکمیلی و دریافت برنامه اصلاح تغذیه.

**سؤال ۳: شاخص شجره‌نامه دیابت (DPF) چگونه تعیین می‌شود؟**  
این شاخص فرمولی ریاضی است که شدت سابقه خانوادگی دیابت و سن ابتلای بستگان درجه ۱ و ۲ را مدلسازی می‌کند. به طور معمول برای افرادی که هیچ سابقه خانوادگی ندارند کمتر از ۰.۲۵ و برای افرادی با والدین یا خواهر/برادر مبتلا بالای ۰.۶۰ در نظر گرفته می‌شود.

---

### ⚠️ سلب مسئولیت پزشکی
> اطلاعات ارائه شده در این برنامه صرفاً جنبه آموزشی، تحقیقاتی و نمایشی دارد. به هیچ عنوان نباید از خروجی‌های این سامانه برای خوددرمانی، تجویز دارو یا تغییر دوز درمانی استفاده شود. همواره به راهنمایی پزشکان و کادر رسمی بهداشت و درمان اعتماد کنید.
"""

GUIDE_EN = """
# 📖 DiabPredict Comprehensive User Guide & Tutorial

---

### 1. What is DiabPredict?
**DiabPredict** is an intelligent, machine learning-powered clinical decision-support system designed for **early screening and predictive risk stratification of Type 2 Diabetes**.

By synthesizing phenotypic markers, laboratory values, and metabolic indicators, DiabPredict computes statistical disease probability in sub-milliseconds, visualizing risk tiers and personalized lifestyle and diagnostic recommendations.

---

### 2. Purpose and Objectives
Type 2 Diabetes is a pervasive chronic disease that frequently progresses without overt early symptoms. According to the International Diabetes Federation (IDF), **nearly 1 in 2 adults living with diabetes are undiagnosed**.

#### Primary Goals:
* **Accessible Early Screening:** Rapid identification of at-risk individuals prior to irreversible microvascular and macrovascular complications.
* **Clinical Decision Support (CDSS):** Assisting clinicians and nurses in synthesizing multi-parameter metabolic profiles.
* **Patient Empowerment:** Educating users on the physiological drivers of metabolic syndrome and fostering proactive lifestyle modifications.
* **Educational Benchmark:** An interactive testbed for health informatics, biomedical engineering, and data science students.

---

### 3. Machine Learning Architecture & Background
DiabPredict utilizes a high-performance **Random Forest Classifier**, an industry-standard bagging ensemble algorithm:

* **Pima Indians Diabetes Database:** Collected by the US National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), this dataset serves as a benchmark for diabetes prediction.
* **100-Tree Bagging Ensemble:** The model aggregates predictions from 100 decorrelated decision trees using Gini impurity splitting criterion, mitigating overfitting while capturing non-linear interactions between biomarkers.
* **Stratified Validation:** Evaluated on a stratified 80/20 train/test split, ensuring balanced representation and achieving over 75% baseline diagnostic accuracy.

---

### 4. Step-by-Step Tutorial

#### 🔹 Step 1: Language & Orientation
* Toggle between **"🇮🇷 فارسی"** and **"🇬🇧 English"** at the top of the sidebar. When Persian is selected, full Right-to-Left (RTL) styling and localized typography are dynamically activated.

#### 🔹 Step 2: Adjust Patient Parameters
* The sidebar accommodates **8 key physiological sliders**:
  * Pregnancies, Glucose, Diastolic Blood Pressure, Skin Thickness, Serum Insulin, BMI, Diabetes Pedigree Function, and Age.
* **Quick Presets:** Expand **"⚡ Clinical Presets & Reset"** to test predefined clinical profiles: **🟢 Low Risk**, **🔴 High Risk**, or **🔄 Reset to Defaults**.

#### 🔹 Step 3: Run Inference
* Click **"Predict Diabetes Risk"** in the sidebar or main header. The inference pipeline processes the patient feature vector instantaneously.

#### 🔹 Step 4: Interpret the Diagnostic Evaluation
Under Tab 1 (**🩺 Prediction & Risk Assessment**), review:
1. **Diagnostic Outcome:** High Risk alert (Positive) or Low Risk confirmation (Negative).
2. **Probability Metric:** Calculated risk percentage and margin versus the 50% clinical decision boundary.
3. **Risk Stratification Tiers:**
   * 🟢 **Low Risk Profile (0% - 30%):** Healthy metabolic profile.
   * 🟡 **Moderate / Monitoring (30% - 60%):** Borderline indicators requiring lifestyle vigilance.
   * 🔴 **High Risk / Critical Alert (60% - 100%):** Clinically significant predisposition warranting lab confirmation.
4. **Visual Gauge:** Gradient progress bar indicating the exact risk position.

#### 🔹 Step 5: Review Physiological Drivers & Recommendations
* Review auto-detected abnormalities (such as impaired glucose >140 mg/dL, elevated BMI, or hypertension) and evidence-based guidance (e.g., HbA1c testing, endocrinology consultation, aerobic exercise regimens).

#### 🔹 Step 6: Explore Deep Analytics Tabs
* **📋 Patient Biomarker Summary:** Compare input values against reference ranges and clinical status indicators.
* **📊 Model Intelligence & Analytics:** Inspect the feature importance chart and test set confusion matrix.
* **🔍 Dataset Explorer:** View sample rows and statistical distributions of the underlying Pima Indians dataset.

---

### 5. Clinical Biomarkers Reference Table

| Biomarker | Unit | Clinical Reference | Physiological Role & Diabetes Relevance |
| :--- | :---: | :---: | :--- |
| **Glucose (2-Hour OGTT)** | mg/dL | 70 - 140 | Plasma glucose concentration; levels ≥140 indicate impaired glucose tolerance, ≥200 indicate diabetes. |
| **BMI** | kg/m² | 18.5 - 24.9 | Body Mass Index; values ≥30 denote obesity, a primary driver of peripheral insulin resistance. |
| **Age** | years | 21 - 81 | Progressive decline in pancreatic beta-cell sensitivity occurs after age 45. |
| **Diabetes Pedigree (DPF)** | score | < 0.50 | Quantitative score synthesizing genetic predisposition and familial disease incidence. |
| **Blood Pressure** | mm Hg | 60 - 80 | Diastolic blood pressure; hypertension frequently clusters within metabolic syndrome. |
| **Insulin (2-Hour)** | μU/mL | 16 - 166 | Elevated post-challenge serum insulin highlights compensatory hyperinsulinemia. |
| **Skin Thickness** | mm | 10 - 30 | Triceps skin fold measurement reflecting subcutaneous adipose distribution. |
| **Pregnancies** | count | 0 - 4 | Multiparity may correlate with history of gestational diabetes and postpartum risk. |

---

### 6. Frequently Asked Questions (FAQ)

**Q1: Can this application replace official laboratory testing?**  
*No.* DiabPredict is a preliminary screening tool. Definitive diagnosis requires standardized laboratory evaluations (Fasting Blood Sugar, Oral Glucose Tolerance, HbA1c) interpreted by licensed medical doctors.

**Q2: What action should be taken if a High Risk prediction occurs?**  
Do not panic. A high score signifies statistical probability, not a definitive diagnosis. Schedule an appointment with your primary care provider or endocrinologist for comprehensive blood tests.

---

### ⚠️ Medical Disclaimer
> All information and computational estimates produced by DiabPredict are strictly intended for educational, screening, and research demonstration purposes. They must never be used for autonomous medical decision-making, clinical diagnosis, or therapeutic modification.
"""

