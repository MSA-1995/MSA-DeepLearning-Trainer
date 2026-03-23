"""
🧠 Training Models - 8 LightGBM Models
Each function trains one model and returns (model, accuracy).
"""

import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from features import calculate_enhanced_features


# ========== Helper ==========

def _build_dataset(trades, feature_fn, label_fn):
    """Build features/labels arrays from trades list."""
    features_list, labels_list = [], []
    for trade in trades:
        try:
            data = trade.get('data', {})
            if isinstance(data, str):
                data = json.loads(data)
            features_list.append(feature_fn(data, trade))
            labels_list.append(label_fn(trade))
        except:
            continue
    return features_list, labels_list


def _train_lgb(X, y, n_estimators=100, max_depth=5, learning_rate=0.1):
    """Train LightGBM classifier and return (model, accuracy)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


def train_meta_learner_model(trades, trained_models):
    """👑🧠 Train the Meta-Learner (The New King)
     يتعلم من قرارات المستشارين الآخرين لاتخاذ قرار نهائي أكثر ذكاءً
    """
    print("\n👑🧠 Training Meta-Learner Model (The New King)...")

    if not trained_models or len(trained_models) < 7:
        print("⚠️ Not enough trained consultant models to train the Meta-Learner.")
        return None

    # 1. بناء مجموعة بيانات جديدة من آراء المستشارين
    meta_features = []
    final_labels = []

    # استبعاد الملك الجديد نفسه من قائمة المستشارين للمدخلات
    consultant_models = {k: v for k, v in trained_models.items() if k != 'meta_learner' and v is not None}

    for trade in trades:
        try:
            data = trade.get('data', {})
            if isinstance(data, str):
                data = json.loads(data)

            # الحصول على آراء (توقعات) كل مستشار لهذه الصفقة
            consultant_opinions = []
            for model_name, model in consultant_models.items():
                # نحتاج إلى بناء نفس الميزات التي تدرب عليها كل مستشار
                # هذه عملية معقدة، سنقوم بتبسيطها الآن بالاعتماد على النقاط المسجلة مباشرة
                # إذا كانت النقاط غير موجودة، سنستخدم 0 كقيمة افتراضية
                opinion = data.get(f'{model_name}_score', 0) 
                consultant_opinions.append(opinion)
            
            # أضف رأي المستشارين كـ "ميزات" للملك الجديد
            meta_features.append(consultant_opinions)
            
            # الهدف: هل الصفقة كانت ناجحة؟
            final_labels.append(1 if float(trade.get('profit_percent', 0)) > 0.8 else 0)

        except Exception as e:
            # print(f"Skipping trade for Meta-Learner due to error: {e}")
            continue

    if len(meta_features) < 100:
        print(f"⚠️ Not enough data for Meta-Learner ({len(meta_features)} trades found)")
        return None

    # 2. تدريب الملك الجديد
    # نستخدم مصنف أقوى قليلاً لأنه يتعلم من بيانات معقدة
    model, accuracy = _train_lgb(
        np.array(meta_features),
        np.array(final_labels),
        n_estimators=300, 
        max_depth=4, # عمق أقل لتجنب الحفظ الزائد (Overfitting)
        learning_rate=0.03
    )
    
    print(f"👑🧠 Meta-Learner Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


# ========== Models ==========

def train_ai_brain_model(trades, voting_scores=None):
    """👑 Train AI Brain (الملك) - القرار النهائي"""
    print("\n👑 Training AI Brain Model (LightGBM)...")
    scores = (voting_scores or {}).get('ai_brain', {})

    def features(data, trade):
        return [
            data.get('rsi', 50),          data.get('macd', 0),
            data.get('volume_ratio', 1),   data.get('price_momentum', 0),
            data.get('confidence', 60),    data.get('mtf_score', 0),
            data.get('risk_score', 0),     data.get('anomaly_score', 0),
            data.get('exit_score', 0),     data.get('pattern_score', 0),
            data.get('ranking_score', 0),  data.get('atr', 1),
            data.get('ema_crossover', 0),  data.get('bid_ask_spread', 0),
            data.get('volume_trend', 0),   data.get('price_change_1h', 0),
            scores.get('tp_accuracy', 0.5),    scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5),    scores.get('sell_accuracy', 0.5),
        ]

    fl, ll = _build_dataset(trades, features, lambda t: 1 if float(t.get('profit_percent', 0)) > 0.8 else 0)
    if len(fl) < 50:
        print("⚠️ Not enough data for AI Brain")
        return None

    model, accuracy = _train_lgb(np.array(fl), np.array(ll), n_estimators=250, max_depth=8, learning_rate=0.04)
    print(f"👑 AI Brain Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


def train_smart_money_model(trades, voting_scores=None):
    """🐋 Train Smart Money Tracker"""
    print("\n🐋 Training Smart Money Model (LightGBM)...")
    scores = (voting_scores or {}).get('smart_money', {})

    def features(data, trade):
        return [
            data.get('rsi', 50),           data.get('macd', 0),
            data.get('volume_ratio', 1),    data.get('price_momentum', 0),
            data.get('atr', 1),             data.get('ema_crossover', 0),
            data.get('bid_ask_spread', 0),  data.get('volume_trend', 0),
            data.get('price_change_1h', 0),
            scores.get('tp_accuracy', 0.5), scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5), scores.get('sell_accuracy', 0.5),
        ]

    fl, ll = _build_dataset(trades, features, lambda t: 1 if float(t.get('profit_percent', 0)) > 0.8 else 0)
    if len(fl) < 50:
        print("⚠️ Not enough data for Smart Money")
        return None

    model, accuracy = _train_lgb(np.array(fl), np.array(ll), n_estimators=150, max_depth=6, learning_rate=0.08)
    print(f"🐋 Smart Money Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


def train_risk_model(trades, voting_scores=None):
    """🛡️ Train Risk Manager"""
    print("\n🛡️ Training Risk Model (LightGBM)...")
    scores = (voting_scores or {}).get('risk', {})

    def features(data, trade):
        return [
            data.get('rsi', 50),           data.get('volume_ratio', 1),
            data.get('confidence', 60),     data.get('price_momentum', 0),
            data.get('atr', 1),             data.get('ema_crossover', 0),
            data.get('bid_ask_spread', 0),  data.get('volume_trend', 0),
            data.get('price_change_1h', 0),
            scores.get('tp_accuracy', 0.5), scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5), scores.get('sell_accuracy', 0.5),
        ]

    fl, ll = _build_dataset(trades, features, lambda t: 1 if float(t.get('profit_percent', 0)) < -1.0 else 0)
    if len(fl) < 50:
        print("⚠️ Not enough data for Risk")
        return None

    model, accuracy = _train_lgb(np.array(fl), np.array(ll), n_estimators=100, max_depth=5, learning_rate=0.1)
    print(f"✅ Risk Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


def train_anomaly_model(trades, voting_scores=None):
    """🚨 Train Anomaly Detector"""
    print("\n🚨 Training Anomaly Model (LightGBM)...")
    scores = (voting_scores or {}).get('anomaly', {})

    def features(data, trade):
        return [
            data.get('rsi', 50),           data.get('macd', 0),
            data.get('volume_ratio', 1),    data.get('price_momentum', 0),
            data.get('atr', 1),             data.get('ema_crossover', 0),
            data.get('bid_ask_spread', 0),  data.get('volume_trend', 0),
            data.get('price_change_1h', 0),
            scores.get('tp_accuracy', 0.5), scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5), scores.get('sell_accuracy', 0.5),
        ]

    fl, ll = _build_dataset(trades, features, lambda t: 1 if float(t.get('profit_percent', 0)) < -1.5 else 0)
    if len(fl) < 50:
        print("⚠️ Not enough data for Anomaly")
        return None

    model, accuracy = _train_lgb(np.array(fl), np.array(ll), n_estimators=100, max_depth=5, learning_rate=0.1)
    print(f"✅ Anomaly Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


def train_exit_model(trades, voting_scores=None):
    """🎯 Train Exit Strategy"""
    print("\n🎯 Training Exit Model (LightGBM)...")
    scores = (voting_scores or {}).get('exit', {})

    def features(data, trade):
        return [
            data.get('rsi', 50),           data.get('macd', 0),
            data.get('confidence', 60),     data.get('price_momentum', 0),
            data.get('atr', 1),             data.get('ema_crossover', 0),
            data.get('bid_ask_spread', 0),  data.get('volume_trend', 0),
            data.get('price_change_1h', 0),
            scores.get('tp_accuracy', 0.5), scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5), scores.get('sell_accuracy', 0.5),
        ]

    def label(trade):
        p = float(trade.get('profit_percent', 0))
        return 1 if (p > 1.0 or p < -1.0) else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Exit")
        return None

    model, accuracy = _train_lgb(np.array(fl), np.array(ll), n_estimators=100, max_depth=5, learning_rate=0.1)
    print(f"✅ Exit Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


def train_pattern_model(trades, voting_scores=None):
    """🧠 Train Pattern Recognition"""
    print("\n🧠 Training Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('pattern', {})

    def features(data, trade):
        return [
            data.get('rsi', 50),           data.get('macd', 0),
            data.get('volume_ratio', 1),    data.get('price_momentum', 0),
            data.get('confidence', 60),     data.get('atr', 1),
            data.get('ema_crossover', 0),   data.get('bid_ask_spread', 0),
            data.get('volume_trend', 0),    data.get('price_change_1h', 0),
            scores.get('tp_accuracy', 0.5), scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5), scores.get('sell_accuracy', 0.5),
        ]

    fl, ll = _build_dataset(trades, features, lambda t: 1 if float(t.get('profit_percent', 0)) > 0.8 else 0)
    if len(fl) < 50:
        print("⚠️ Not enough data for Pattern")
        return None

    model, accuracy = _train_lgb(np.array(fl), np.array(ll), n_estimators=150, max_depth=6, learning_rate=0.08)
    print(f"✅ Pattern Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


def train_liquidity_model(trades, voting_scores=None):
    """💧 Train Liquidity Analyzer"""
    print("\n💧 Training Liquidity Model (LightGBM)...")
    scores = (voting_scores or {}).get('liquidity', {})

    # Aggregate per-symbol stats
    coin_data = {}
    for trade in trades:
        try:
            symbol = trade.get('symbol')
            profit = float(trade.get('profit_percent', 0))
            data   = trade.get('data', {})
            if isinstance(data, str):
                data = json.loads(data)

            if symbol not in coin_data:
                coin_data[symbol] = {
                    'profits': [], 'count': 0,
                    'volume_ratios': [], 'bid_ask_spreads': [], 'volume_trends': [],
                    'depth_ratios': [], 'liquidity_scores': [],
                    'price_impacts': [], 'volume_consistencies': []
                }

            cd = coin_data[symbol]
            cd['profits'].append(profit)
            cd['count'] += 1
            cd['volume_ratios'].append(data.get('volume_ratio', 1.0))
            cd['bid_ask_spreads'].append(data.get('bid_ask_spread', 0))
            cd['volume_trends'].append(data.get('volume_trend', 0))

            liq = data.get('liquidity', {})
            cd['depth_ratios'].append(liq.get('depth_ratio', 1.0))
            cd['liquidity_scores'].append(liq.get('liquidity_score', 50))
            cd['price_impacts'].append(liq.get('price_impact', 0.5))
            cd['volume_consistencies'].append(liq.get('volume_consistency', 50))
        except:
            continue

    def _avg(lst): return sum(lst) / len(lst) if lst else 0

    features_list, labels_list = [], []
    for symbol, cd in coin_data.items():
        if cd['count'] < 3:
            continue
        avg_profit   = _avg(cd['profits'])
        win_rate     = sum(1 for p in cd['profits'] if p > 0) / len(cd['profits'])
        spread_vol   = float(np.std(cd['bid_ask_spreads'])) if len(cd['bid_ask_spreads']) > 1 else 0

        features_list.append([
            avg_profit,             win_rate,              cd['count'],
            max(cd['profits']),     min(cd['profits']),
            _avg(cd['volume_ratios']),      _avg(cd['bid_ask_spreads']),
            _avg(cd['volume_trends']),      _avg(cd['depth_ratios']),
            _avg(cd['liquidity_scores']),   _avg(cd['price_impacts']),
            _avg(cd['volume_consistencies']), spread_vol,
            scores.get('tp_accuracy', 0.5), scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5), scores.get('sell_accuracy', 0.5),
        ])
        labels_list.append(1 if (avg_profit > 0.8 and win_rate > 0.55) else 0)

    if len(features_list) < 20:
        print("⚠️ Not enough coins for Liquidity")
        return None

    model, accuracy = _train_lgb(np.array(features_list), np.array(labels_list),
                                  n_estimators=200, max_depth=6, learning_rate=0.05)
    print(f"💧 Liquidity Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


def train_chart_cnn_model(trades, voting_scores=None):
    """📊 Train Chart Pattern Analyzer"""
    print("\n📊 Training Chart Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('cnn', {})

    def features(data, trade):
        base = calculate_enhanced_features(data)
        base.extend([
            scores.get('tp_accuracy', 0.5),    scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5),     scores.get('sell_accuracy', 0.5),
        ])
        return base

    fl, ll = _build_dataset(trades, features, lambda t: 1 if float(t.get('profit_percent', 0)) > 0.8 else 0)
    if len(fl) < 50:
        print("⚠️ Not enough data for Chart Pattern")
        return None

    model, accuracy = _train_lgb(np.array(fl), np.array(ll), n_estimators=150, max_depth=6, learning_rate=0.08)
    print(f"📊 Chart Pattern Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


def train_rescue_model(trades, voting_scores=None):
    """🤪 Train Rescue Scalper (The Crazy Jester)"""
    print("\n🤪 Training Rescue Model (LightGBM)...")
    # لا نحتاج تصويت المستشارين هنا، الخبل يعمل منفرداً

    def features(data, trade):
        return [
            data.get('rsi', 50),          data.get('macd', 0),
            data.get('volume_ratio', 1),  data.get('price_momentum', 0),
            data.get('atr', 1),           data.get('ema_crossover', 0),
            data.get('bid_ask_spread', 0), data.get('volume_trend', 0),
        ]

    # تصفية الصفقات: نتدرب فقط على صفقات الإنقاذ أو الزومبي
    rescue_trades = []
    for t in trades:
        data = t.get('data', {})
        if isinstance(data, str): data = json.loads(data)
        
        # محاولة اكتشاف صفقة إنقاذ من البيانات أو السبب (إذا توفر)
        # بما أن السبب قد لا يكون متاحاً دائماً في الـ data blob، نعتمد على المنطق التقريبي
        # أو إذا كانت hours_held > 70 (إذا توفرت المعلومة)
        # هنا سنفترض أننا نمرر كل الصفقات وسيتعلم هو الأنماط الناجحة للخروج من مواقف صعبة
        # لكن لزيادة الدقة، يفضل وجود flag. حالياً سندربه على الخروج الذكي بشكل عام.
        rescue_trades.append(t)

    fl, ll = _build_dataset(rescue_trades, features, lambda t: 1 if float(t.get('profit_percent', 0)) > 0 else 0)
    
    if len(fl) < 20: # نقبل ببيانات أقل لهذا الموديل في البداية
        print("⚠️ Not enough data for Rescue Model yet")
        return None

    model, accuracy = _train_lgb(np.array(fl), np.array(ll), n_estimators=100, max_depth=4, learning_rate=0.05)
    print(f"🤪 Rescue Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy
