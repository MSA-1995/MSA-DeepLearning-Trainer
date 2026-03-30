"""
🧠 Training Models -  LightGBM Models
Each function trains one model and returns (model, accuracy).
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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


def _train_lgb(X, y, feature_names, n_estimators=100, max_depth=5, learning_rate=0.1):
    """Train LightGBM classifier and return (model, accuracy)."""
    # تحويل البيانات إلى DataFrame مع أسماء الميزات
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y, name='target')

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42, stratify=y_s)
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1
    )
    # التدريب باستخدام DataFrame يضمن حفظ أسماء الميزات
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


def train_meta_learner_model(db_manager, trained_models, voting_scores=None):
    """👑🧠 Train the Meta-Learner (The New King) all at once.
     يتعلم من قرارات المستشارين الآخرين لاتخاذ قرار نهائي أكثر ذكاءً
    """
    print("\n👑🧠 Training Meta-Learner Model (The New King)...")

    if not trained_models or len(trained_models) < 7:
        print("⚠️ Not enough trained consultant models to train the Meta-Learner.")
        return None
    
    # استيراد النماذج الجديدة
    try:
        import sys
        import os
        # Add parent directory to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from models.sentiment_model import SentimentAnalyzer
        from models.crypto_news_model import CryptoNewsAnalyzer
        from models.volume_prediction_model import VolumePredictor
        sentiment_analyzer = SentimentAnalyzer()
        news_analyzer = CryptoNewsAnalyzer()
        volume_predictor = VolumePredictor()
    except Exception as e:
        print(f"⚠️ Could not load new models: {e}")
        sentiment_analyzer = None
        news_analyzer = None
        volume_predictor = None

    # 1. تحميل البيانات المساعدة (قرارات براين، الأفخاخ) مرة واحدة
    print("Loading auxiliary data (AI decisions, traps)...")
    try:
        ai_decisions = db_manager.load_ai_decisions(limit=50000)
        trap_memory = db_manager.load_traps(limit=10000)
        ai_decisions_df = pd.DataFrame(ai_decisions)
        trap_memory_df = pd.DataFrame(trap_memory)
        
        # الحصول على العدد الإجمالي للصفقات
        total_trades = db_manager.get_total_trades_count()
        if total_trades == 0:
            print("⚠️ No trades found to train the Meta-Learner.")
            return None
        print(f"📈 Found {total_trades} total trades. Processing all for the King.")

    except Exception as e:
        print(f"❌ Error loading initial data for Meta-Learner: {e}")
        print("💡 Hint: Make sure the 'get_total_trades_count' method is implemented in db_manager.py")
        return None

    # 2. بناء مجموعة البيانات دفعة واحدة
    meta_features = []
    final_labels = []
    consultant_models = {k: v for k, v in trained_models.items() if k != 'meta_learner' and v is not None}

    print(f"  -> Processing all {total_trades} trades for King at once...")
    try:
        trades_batch = db_manager.load_training_data(limit=total_trades)
        if trades_batch:
            trades_df = pd.DataFrame([dict(t) for t in trades_batch])

            for index, trade in trades_df.iterrows():
                try:
                    data = trade.get('data', {})
                    if isinstance(data, str):
                        data = json.loads(data)

                    consultant_opinions = [data.get(f'{model_name}_score', 0.5) for model_name in consultant_models.keys()]
                    
                    # حساب دقة كل مستشار (جديد)
                    consultant_accuracies = []
                    for model_name in consultant_models.keys():
                        accuracy_key = f'{model_name}_accuracy'
                        accuracy = voting_scores.get(model_name, {}).get('accuracy', 0.5) if voting_scores else 0.5
                        consultant_accuracies.append(accuracy)
                    
                    # حساب تقلب أداء المستشارين (جديد)
                    consultant_volatility = np.std(consultant_opinions) if len(consultant_opinions) > 1 else 0
                    
                    symbol = trade['symbol']
                    buy_time = None
                    if 'buy_time' in trade and pd.notna(trade['buy_time']):
                        buy_time = pd.to_datetime(trade['buy_time'])
                    else:
                        trade_time = pd.to_datetime(trade['timestamp'])
                        decisions_before_trade = ai_decisions_df[
                            (ai_decisions_df['symbol'] == symbol) &
                            (ai_decisions_df['decision'] == 'BUY') &
                            (pd.to_datetime(ai_decisions_df['timestamp']) < trade_time)
                        ]
                        if not decisions_before_trade.empty:
                            last_buy_decision = decisions_before_trade.sort_values('timestamp', ascending=False).iloc[0]
                            buy_time = pd.to_datetime(last_buy_decision['timestamp'])

                    # إذا لم يتم العثور على buy_time، استخدم وقت الصفقة
                    if buy_time is None:
                        buy_time = pd.to_datetime(trade['timestamp'])

                    relevant_decision = None
                    if not ai_decisions_df.empty:
                        decisions_before_buy = ai_decisions_df[
                            (ai_decisions_df['symbol'] == symbol) &
                            (pd.to_datetime(ai_decisions_df['timestamp']) < buy_time)
                        ]
                        if not decisions_before_buy.empty:
                            relevant_decision = decisions_before_buy.sort_values('timestamp', ascending=False).iloc[0]

                    brain_confidence = relevant_decision['confidence'] if relevant_decision is not None else 50
                    
                    was_trapped = False
                    if not trap_memory_df.empty:
                        was_trapped = not trap_memory_df[
                            (trap_memory_df['symbol'] == symbol) &
                            (pd.to_datetime(trap_memory_df['timestamp']) > buy_time - pd.Timedelta(hours=24)) &
                            (pd.to_datetime(trap_memory_df['timestamp']) < buy_time)
                        ].empty
                    
                    # تأثير الأخبار (جديد)
                    news_impact = 0
                    if news_analyzer:
                        news_data = data.get('news', {})
                        news_impact = news_analyzer.get_news_impact_score(news_data)
                    
                    # حالة السوق الاستثنائية (جديد)
                    is_black_swan = 0
                    if sentiment_analyzer:
                        sentiment_data = data.get('sentiment', {})
                        sentiment_score = sentiment_analyzer.get_sentiment_score(sentiment_data)
                        is_black_swan = 1 if abs(sentiment_score) > 80 else 0
                    
                    # احتمالية زيادة الحجم (جديد)
                    volume_spike_prob = 0
                    if volume_predictor:
                        volume_spike_prob = volume_predictor.get_volume_spike_probability(data)
                    
                    # بناء الميزات المحسنة
                    features = (
                        consultant_opinions +  # آراء المستشارين
                        consultant_accuracies +  # دقة المستشارين (جديد)
                        [brain_confidence, 1 if was_trapped else 0] +  # الميزات القديمة
                        [consultant_volatility, news_impact, is_black_swan, volume_spike_prob]  # الميزات الجديدة
                    )
                    meta_features.append(features)
                    final_labels.append(1 if float(trade.get('profit_percent', 0)) > 0.8 else 0)

                except Exception as e_inner:
                    # This will silently skip a single problematic trade
                    continue
        
    except Exception as e_outer:
        print(f"❌ Failed to process trades: {e_outer}")

    if len(meta_features) < 100:
        print(f"⚠️ Not enough data for Meta-Learner ({len(meta_features)} trades found)")
        return None

    # 3. تدريب الملك الجديد
    print(f"\nCollected {len(meta_features)} samples. Now training the King...")
    
    # تحديد أسماء الميزات للملك (محسنة)
    meta_feature_names = (
        [f'consultant_{name}' for name in consultant_models.keys()] +  # آراء المستشارين
        [f'consultant_{name}_accuracy' for name in consultant_models.keys()] +  # دقة المستشارين
        ['brain_confidence', 'was_trapped'] +  # الميزات القديمة
        ['consultant_volatility', 'news_impact', 'is_black_swan', 'volume_spike_prob']  # الميزات الجديدة
    )

    model, accuracy = _train_lgb(
        meta_features,
        np.array(final_labels),
        feature_names=meta_feature_names,
        n_estimators=300, 
        max_depth=5,
        learning_rate=0.03
    )
    
    print(f"👑🧠 Meta-Learner Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


# ========== Models ==========




def train_smart_money_model(trades, voting_scores=None):
    """🐋 Smart Money Tracker - يتعلم متى الأكابر يشترون
    الوظيفة: اكتشاف دخول الأموال الذكية
    الـ Label: صفقة ربحية كبيرة (profit > 1%)
    """
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

    feature_names = [
        'rsi', 'macd', 'volume_ratio', 'price_momentum', 'atr', 'ema_crossover', 
        'bid_ask_spread', 'volume_trend', 'price_change_1h', 'tp_accuracy', 
        'amount_accuracy', 'sl_accuracy', 'sell_accuracy'
    ]
    
    # Smart Money يتعلم: متى الأكابر يدخلون (ربح كبير)
    def label(t):
        trade_quality = t.get('trade_quality', '')
        profit = float(t.get('profit_percent', 0))
        return 1 if trade_quality == 'GREAT' or profit > 1.5 else 0
    
    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Smart Money")
        return None

    model, accuracy = _train_lgb(fl, np.array(ll), feature_names, n_estimators=150, max_depth=6, learning_rate=0.08)
    print(f"🐋 Smart Money Model: Accuracy {accuracy*100:.2f}% | Learns: GREAT trades")
    return model, accuracy


def train_risk_model(trades, voting_scores=None):
    """🛡️ Risk Manager - يتعلم يتجنب الأخطار
    الوظيفة: اكتشاف الفخاخ والصفقات الخاسرة
    الـ Label: صفقة فخ أو محفوفة بالمخاطر
    """
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

    feature_names = [
        'rsi', 'volume_ratio', 'confidence', 'price_momentum', 'atr', 'ema_crossover',
        'bid_ask_spread', 'volume_trend', 'price_change_1h', 'tp_accuracy', 
        'amount_accuracy', 'sl_accuracy', 'sell_accuracy'
    ]
    
    # Risk يتعلم: كيف يكتشف الفخاخ (TRAP, RISKY)
    def label(t):
        trade_quality = t.get('trade_quality', '')
        return 1 if trade_quality in ['TRAP', 'RISKY'] else 0
    
    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Risk")
        return None

    model, accuracy = _train_lgb(fl, np.array(ll), feature_names, n_estimators=100, max_depth=5, learning_rate=0.1)
    print(f"🛡️ Risk Model: Accuracy {accuracy*100:.2f}% | Learns: TRAP/RISKY trades")
    return model, accuracy


def train_anomaly_model(trades, voting_scores=None):
    """🚨 Anomaly Detector - يكتشف الحالات الشاذة
    الوظيفة: اكتشاف تذبذب غير طبيعي
    الـ Label: صفقة محفوفة بالمخاطر
    """
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

    feature_names = [
        'rsi', 'macd', 'volume_ratio', 'price_momentum', 'atr', 'ema_crossover',
        'bid_ask_spread', 'volume_trend', 'price_change_1h', 'tp_accuracy', 
        'amount_accuracy', 'sl_accuracy', 'sell_accuracy'
    ]
    
    # Anomaly يتعلم: كيف يكتشف الشذوذ (RISKY)
    def label(t):
        trade_quality = t.get('trade_quality', '')
        profit = float(t.get('profit_percent', 0))
        return 1 if trade_quality == 'RISKY' or profit < -0.5 else 0
    
    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Anomaly")
        return None

    model, accuracy = _train_lgb(fl, np.array(ll), feature_names, n_estimators=100, max_depth=5, learning_rate=0.1)
    print(f"🚨 Anomaly Model: Accuracy {accuracy*100:.2f}% | Learns: RISKY trades")
    return model, accuracy


def train_exit_model(trades, voting_scores=None):
    """🎯 Exit Strategy - يتعلم متى البيع
    الوظيفة: اكتشاف الوقت المثالي للبيع
    الـ Label: صفقة ناجحة (GOOD, GREAT) أو خسارة صغيرة
    """
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

    # Exit يتعلم: متى البيع صح (ربح جيد أو خسارة مقبولة)
    def label(t):
        trade_quality = t.get('trade_quality', '')
        return 1 if trade_quality in ['GREAT', 'GOOD'] else 0

    feature_names = [
        'rsi', 'macd', 'confidence', 'price_momentum', 'atr', 'ema_crossover',
        'bid_ask_spread', 'volume_trend', 'price_change_1h', 'tp_accuracy', 
        'amount_accuracy', 'sl_accuracy', 'sell_accuracy'
    ]
    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Exit")
        return None

    model, accuracy = _train_lgb(fl, np.array(ll), feature_names, n_estimators=100, max_depth=5, learning_rate=0.1)
    print(f"🎯 Exit Model: Accuracy {accuracy*100:.2f}% | Learns: GOOD/GREAT exits")
    return model, accuracy


def train_pattern_model(trades, voting_scores=None):
    """🧠 Pattern Recognition - يتعلم الأنماط الناجحة
    الوظيفة: اكتشاف أنماط الشموع الصحيحة
    الـ Label: صفقة ممتازة (GREAT only)
    """
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

    feature_names = [
        'rsi', 'macd', 'volume_ratio', 'price_momentum', 'confidence', 'atr',
        'ema_crossover', 'bid_ask_spread', 'volume_trend', 'price_change_1h',
        'tp_accuracy', 'amount_accuracy', 'sl_accuracy', 'sell_accuracy'
    ]
    
    # Pattern يتعلم: الأنماط الناجحة فقط (GREAT)
    def label(t):
        trade_quality = t.get('trade_quality', '')
        return 1 if trade_quality == 'GREAT' else 0
    
    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Pattern")
        return None

    model, accuracy = _train_lgb(fl, np.array(ll), feature_names, n_estimators=150, max_depth=6, learning_rate=0.08)
    print(f"🧠 Pattern Model: Accuracy {accuracy*100:.2f}% | Learns: GREAT patterns")
    return model, accuracy


def train_liquidity_model(trades, voting_scores=None):
    """💧 Liquidity Analyzer - يتعلم السوائلة الجيدة
    الوظيفة: اكتشاف العملات ذات السيولة الجيدة
    الـ Label: عملة ذات سيولة جيدة (ربح > 0.8% و win rate > 55%)
    """
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
            avg_profit,             win_rate,              
            cd['count'],
            max(cd['profits']),     min(cd['profits']),
            _avg(cd['volume_ratios']),      _avg(cd['bid_ask_spreads']),
            _avg(cd['volume_trends']),      _avg(cd['depth_ratios']),
            _avg(cd['liquidity_scores']),   _avg(cd['price_impacts']),
            _avg(cd['volume_consistencies']), spread_vol,
            scores.get('tp_accuracy', 0.5), scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5), scores.get('sell_accuracy', 0.5),
        ])
        # Liquidity يتعلم: عملة ذات سيولة جيدة (ربح جيد و win rate عالي)
        labels_list.append(1 if (avg_profit > 0.5 and win_rate > 0.50) else 0)

    feature_names = [
        'avg_profit', 
        'win_rate',   
        'trade_count', 'max_profit', 'min_profit',
        'avg_volume_ratio', 'avg_bid_ask_spread', 'avg_volume_trend', 'avg_depth_ratio',
        'avg_liquidity_score', 'avg_price_impact', 'avg_volume_consistency', 'spread_volatility',
        'tp_accuracy', 'amount_accuracy', 'sl_accuracy', 'sell_accuracy'
    ]
    if len(features_list) < 20:
        print("⚠️ Not enough coins for Liquidity")
        return None

    model, accuracy = _train_lgb(features_list, np.array(labels_list), feature_names,
                                  n_estimators=200, max_depth=6, learning_rate=0.05)
    print(f"💧 Liquidity Model: Accuracy {accuracy*100:.2f}% | Learns: GOOD liquidity coins")
    return model, accuracy


def train_chart_cnn_model(trades, voting_scores=None):
    """📊 Chart Pattern Analyzer - يتعلم أنماط الشارت
    الوظيفة: تحليل أنماط الشارت البيانية
    الـ Label: صفقة GOOD أو GREAT
    """
    print("\n📊 Training Chart Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('cnn', {})

    def features(data, trade):
        base = calculate_enhanced_features(data, trade)
        base.extend([
            scores.get('tp_accuracy', 0.5),    scores.get('amount_accuracy', 0.5),
            scores.get('sl_accuracy', 0.5),     scores.get('sell_accuracy', 0.5),
        ])
        return base

    # الميزات الأساسية (34 من features.py)
    base_feature_names = [
        'rsi', 'macd', 'volume_ratio', 'price_momentum',
        'bb_position', 'atr_estimate', 'stochastic', 'ema_signal',
        'volume_strength', 'momentum_strength',
        'atr', 'ema_crossover', 'bid_ask_spread', 'volume_trend', 'price_change_1h',
        'trade_quality_score', 'advisor_vote_consensus', 'is_trap_trade',
        'profit_magnitude', 'hours_held_normalized', 'is_profitable',
        # Market Context & Time Features
        'btc_trend_normalized', 'is_bullish_market', 'hour_normalized',
        'is_asian_session', 'is_european_session', 'is_us_session', 'optimal_hold_score',
        # Fibonacci Features
        'fib_score', 'fib_level_encoded',
        # Market Regime Features (جديد)
        'regime_score', 'regime_adx', 'volatility_ratio', 'position_multiplier',
        # Flash Crash Protection Features (جديد)
        'flash_risk_score', 'flash_crash_detected', 'whale_dump_detected', 'cascade_risk_score'
    ]
    feature_names = base_feature_names + ['tp_accuracy', 'amount_accuracy', 'sl_accuracy', 'sell_accuracy']

    # Chart CNN يتعلم: أنماط الشارت الناجحة (GOOD, GREAT)
    def label(t):
        trade_quality = t.get('trade_quality', '')
        return 1 if trade_quality in ['GREAT', 'GOOD'] else 0
    
    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Chart Pattern")
        return None

    model, accuracy = _train_lgb(fl, np.array(ll), feature_names, n_estimators=150, max_depth=6, learning_rate=0.08)
    
    # Feature Importance Tracking
    try:
        importances = model.feature_importances_
        print(f"📊 Chart Pattern Model: Accuracy {accuracy*100:.2f}% | Learns: GOOD/GREAT charts")
        print(f"   Top 5 features:")
        sorted_idx = np.argsort(importances)[::-1][:5]
        for i in sorted_idx:
            print(f"   - {feature_names[i]}: {importances[i]:.1f}")
    except:
        print(f"📊 Chart Pattern Model: Accuracy {accuracy*100:.2f}% | Learns: GOOD/GREAT charts")
    
    return model, accuracy
