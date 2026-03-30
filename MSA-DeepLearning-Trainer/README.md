# 🧠 Deep Learning Trainer V2

Advanced Deep Learning model for MSA Trading Bot.

## 📁 الهيكل الجديد (Organized Structure)

```
MSA-DeepLearning-Trainer/
├── 📂 core/                    # الملفات الأساسية
│   ├── __init__.py
│   ├── trainer.py              # المنسق الرئيسي
│   ├── deep_trainer_v2.py      # نقطة البداية
│   ├── database.py             # الاتصال بقاعدة البيانات
│   ├── db_manager.py           # إدارة البيانات
│   ├── features.py             # حساب الميزات
│   ├── alerts.py               # نظام التنبيهات
│   └── config.py               # الإعدادات
│
├── 📂 models/                  # النماذج الجديدة
│   ├── __init__.py
│   ├── sentiment_model.py      # 🎭 تحليل المشاعر
│   ├── crypto_news_model.py    # 📰 تحليل أخبار Crypto
│   └── volume_prediction_model.py  # 📊 التنبؤ بالحجم
│
├── 📂 consultants/             # المستشارون (النماذج القديمة)
│   ├── __init__.py
│   └── models.py               # 8 نماذج المستشارين
│
├── 📂 trained_models/          # النماذج المدربة (.pkl)
│   ├── smart_money_model.pkl
│   ├── risk_model.pkl
│   ├── anomaly_model.pkl
│   ├── exit_model.pkl
│   ├── pattern_model.pkl
│   ├── liquidity_model.pkl
│   ├── chart_cnn_model.pkl
│   ├── meta_learner_model.pkl
│   ├── sentiment_model.pkl
│   ├── crypto_news_model.pkl
│   └── volume_prediction_model.pkl
│
├── 📂 data/                    # البيانات
│   └── ...
│
├── __init__.py                 # حزمة رئيسية
├── README.md                   # هذا الملف
├── requirements.txt            # المتطلبات
└── .env                        # متغيرات البيئة
```

## 🎯 الميزات الجديدة

### 1️⃣ نموذج المشاعر (Sentiment Analysis)
- **الهدف:** تحليل مشاعر السوق (خوف/طمع/حياد)
- **المدخلات:** Twitter, Reddit, News APIs
- **المخرجات:** score من -100 (خوف) إلى +100 (طمع)
- **الفائدة:** يتجنب الشراء في حالة الذعر الجماعي

### 2️⃣ نموذج أخبار Crypto
- **الهدف:** فهم تأثير الأخبار على السعر
- **المدخلات:** عناوين أخبار العملات المحددة
- **المخرجات:** تصنيف (إيجابي/سلبي/محايد)
- **الفائدة:** يتنبأ بالتحركات قبل حدوثها

### 3️⃣ نموذج التنبؤ بالحجم (Volume Prediction)
- **الهدف:** التنبؤ بحجم التداول المستقبلي
- **المدخلات:** تاريخ الحجم + مؤشرات فنية
- **المخرجات:** الحجم المتوقع خلال 1-4 ساعات
- **الفائدة:** يكتشف الانفجارات قبل وقوعها

### 4️⃣ تحسين نموذج الملك
- **الميزات الجديدة:**
  - دقة كل مستشار (consultant_accuracy)
  - تقلب أداء المستشارين (consultant_volatility)
  - تأثير الأخبار (news_impact)
  - حالة السوق الاستثنائية (is_black_swan)
  - احتمالية زيادة الحجم (volume_spike_prob)

## 🚀 التشغيل

### التشغيل المحلي:
```bash
cd scripts/MSA-DeepLearning-Trainer
python core/deep_trainer_v2.py
```

### التشغيل التلقائي (كل 6 ساعات):
```bash
# على Linux/Mac
chmod +x start_v2.sh
./start_v2.sh

# على Windows
start_training_local.bat
```

## 📊 النماذج المدربة

### المستشارون (8 نماذج):
1. 🐋 Smart Money Tracker
2. 🛡️ Risk Manager
3. 🚨 Anomaly Detector
4. 🎯 Exit Strategy
5. 🧠 Pattern Recognition
6. 💧 Liquidity Analyzer
7. 📊 Chart CNN
8. 👑 Meta-Learner (الملك)

### النماذج الجديدة (3 نماذج):
9. 🎭 Sentiment Analysis
10. 📰 Crypto News
11. 📊 Volume Prediction

**المجموع: 11 نموذج + الملك = 12 نموذج إجمالي**

## 🔧 التكوين

### ملف .env:
```
DATABASE_URL=postgresql://user:password@host:port/database
ENCRYPTION_KEY=your_encryption_key_here
```

### المتطلبات:
- Python 3.10+
- PostgreSQL Database
- 512MB RAM minimum

## 📈 التحسينات المتوقعة

1. **دقة أفضل:** +15-20% في القرارات
2. **اكتشاف مبكر:** للانفجارات والأخبار المؤثرة
3. **تكيف أفضل:** مع حالات السوق الاستثنائية
4. **قرارات أكثر ذكاءً:** من الملك المحسن

## 🎓 كيف يعمل النظام

1. **تحميل البيانات:** من قاعدة البيانات
2. **تدريب المستشارين:** 8 نماذج LightGBM
3. **تدريب النماذج الجديدة:** 3 نماذج إضافية
4. **تدريب الملك:** يتعلم من آراء المستشارين
5. **حفظ النماذج:** في قاعدة البيانات + ملفات .pkl
6. **البوت يقرأ:** النماذج الجديدة ويستخدمها

## 🔄 دورة الحياة

```
كل 6 ساعات:
1. 📥 تحميل الصفقات من الداتابيز
2. 🧮 حساب الميزات (Features)
3. 🎓 تدريب 11 نموذج (LightGBM)
4. 👑 تدريب الملك من آراء المستشارين
5. 💾 حفظ النماذج (.pkl + Database)
6. 🤖 البوت يقرأ النماذج الجديدة
7. 📈 تحسين القرارات المستقبلية
```

## 📞 الدعم

لأي استفسارات أو مشاكل، تواصل مع فريق MSA Trading Bot.

---

**تم التطوير بواسطة MSA Trading Bot Team 🚀**
