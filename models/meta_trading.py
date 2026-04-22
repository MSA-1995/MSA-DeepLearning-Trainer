"""
👑 Meta Model - The Intelligent King
Learns decision patterns from all trades independently
Not dependent on other models - standalone decision maker
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from features import extract_features


# ── Feature names ─────────────────────────────────────────────
FEATURE_NAMES = [
    # Technical
    'rsi', 'macd_diff', 'volume_ratio', 'price_momentum', 'atr',
    # News
    'news_score', 'news_pos', 'news_neg', 'news_total', 'news_ratio', 'has_news',
    # Sentiment
    'sent_score', 'fear_greed', 'fear_greed_norm', 'is_fearful', 'is_greedy',
    # Liquidity
    'liq_score', 'depth_ratio', 'price_impact', 'good_liq',
    # Smart Money
    'whale_activity', 'exchange_inflow',
    # Social
    'social_volume', 'market_sentiment',
    # Consultants
    'consensus', 'buy_count', 'sell_count',
    # Derived
    'risk_score', 'opportunity', 'market_quality',
    'momentum_strength', 'volatility_level',
    # Symbol Memory - Basic
    'sym_win_rate', 'sym_avg_profit', 'sym_trap_count', 'sym_total', 'sym_is_reliable',
    # Symbol Memory - Extended
    'sym_sentiment_avg', 'sym_whale_avg', 'sym_profit_loss_ratio', 'sym_volume_trend',
    'sym_panic_avg', 'sym_optimism_avg', 'sym_smart_stop_loss',
    # Symbol Memory - Boost columns
    'sym_courage_boost', 'sym_time_memory', 'sym_pattern_score', 'sym_win_rate_boost',
    # Context
    'hours_held',
]


def train_meta_trading(
    trades,
    voting_scores=None,
    since_timestamp=None,
    db_manager=None,
    trained_consultants=None,
):
    """
    Meta Model - learns to make decisions based on market features + symbol memory.
    Trains independently on features extracted from trades.
    """
    print("\n👑 Training Meta Model (Standalone + Symbol Memory)...")

    # ── Load symbol memory ────────────────────────────────────
    symbol_memory = {}
    if db_manager:
        try:
            symbol_memory = db_manager.load_symbol_memory()
            print(f"  🧠 Loaded memory for {len(symbol_memory)} symbols")
        except Exception as e:
            print(f"  ⚠️ Error loading symbol memory: {e}")

    # ── Extract features ──────────────────────────────────────
    features_list, labels_list = [], []
    skipped = 0

    for trade in trades:
        try:
            features = extract_features(trade, symbol_memory)

            profit = float(
                trade.get('profit_percent',
                trade.get('profit',
                trade.get('pnl', 0))) or 0
            )
            meta_label = trade.get('meta_label', 1 if profit > 0 else 0)

            features_list.append(features)
            labels_list.append(meta_label)

        except Exception as e:
            skipped += 1
            if skipped == 1:
                print(f"  ⚠️ First sample error: {e}")
            continue

    print(f"  📊 Extracted {len(features_list)} samples (skipped {skipped})")

    if not features_list:
        print("  ❌ No features extracted, skipping meta model training.")
        return None

    X = pd.DataFrame(features_list, columns=FEATURE_NAMES)
    y = pd.Series(labels_list, name='target')

    # ── Fix Survivorship Bias ─────────────────────────────────
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count

    if pos_count > 0 and neg_count / max(pos_count, 1) < 0.4:
        print(f"  ⚠️ Imbalance: {pos_count} pos vs {neg_count} neg — generating synthetic negatives...")
        rng         = np.random.default_rng(42)
        n_synthetic = pos_count - neg_count
        pos_indices = np.where(y == 1)[0]
        chosen      = rng.choice(pos_indices, size=n_synthetic, replace=True)

        syn = X.iloc[chosen].copy().reset_index(drop=True)
        syn['rsi']            = rng.uniform(65,    95,    n_synthetic)
        syn['macd_diff']      = rng.uniform(-2.0,  -0.1,  n_synthetic)
        syn['volume_ratio']   = rng.uniform(0.1,   0.7,   n_synthetic)
        syn['consensus']      = rng.uniform(0.0,   0.3,   n_synthetic)
        syn['buy_count']      = rng.integers(0,    2,     n_synthetic).astype(float)
        syn['sell_count']     = rng.integers(3,    7,     n_synthetic).astype(float)
        syn['price_momentum'] = rng.uniform(-0.05, -0.001, n_synthetic)
        syn['opportunity']    = rng.uniform(0,     10,    n_synthetic)
        syn['risk_score']     = rng.uniform(15,    40,    n_synthetic)
        syn['market_quality'] = rng.uniform(0.1,   0.35,  n_synthetic)

        X = pd.concat([X, syn],                                  ignore_index=True)
        y = pd.concat([y, pd.Series([0] * n_synthetic)],        ignore_index=True)
        print(f"  ✅ Added {n_synthetic} synthetic negatives")

    pos   = int(y.sum())
    neg   = len(y) - pos
    ratio = neg / max(pos, 1)
    print(f"  Label balance: {pos} positive ({pos / len(y) * 100:.1f}%) | {neg} negative | ratio={ratio:.1f}x")

    # ── Stratified K-Fold Cross-Validation ────────────────────
    skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls    = []

    model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.4,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        p1     = report.get('1', {}).get('precision', 0)
        r1     = report.get('1', {}).get('recall',    0)

        accuracies.append(acc)
        precisions.append(p1)
        recalls.append(r1)
        print(f"  Fold {fold + 1}: Accuracy {acc * 100:.2f}% | Precision: {p1:.2f} | Recall: {r1:.2f}")

    avg_accuracy  = sum(accuracies) / len(accuracies)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall    = sum(recalls)    / len(recalls)

    print(f"  📊 CV Average → Accuracy: {avg_accuracy * 100:.2f}% | Precision: {avg_precision:.2f} | Recall: {avg_recall:.2f}")

    # ── Retrain on full data for saving ───────────────────────
    model.fit(X, y)

    print(f"  👑 Meta Model: CV Avg Accuracy {avg_accuracy * 100:.2f}% (with Symbol Memory)")
    return model, avg_accuracy