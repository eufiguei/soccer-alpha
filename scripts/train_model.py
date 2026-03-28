#!/usr/bin/env python3
"""
Train AH prediction models with walk-forward validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
import lightgbm as lgb
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'
MODEL_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/models'
BACKTEST_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/backtests'

print("Loading features...")
df = pd.read_parquet(f'{DATA_DIR}/features_bettable.parquet')
df = df.sort_values('Date').reset_index(drop=True)

print(f"Dataset: {len(df)} matches")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# ============================================================
# FEATURE SELECTION
# ============================================================
feature_cols = [
    'home_form_pts', 'home_form_gf', 'home_form_ga', 'home_form_gd',
    'away_form_pts', 'away_form_gf', 'away_form_ga', 'away_form_gd',
    'home_prob', 'away_prob', 'draw_prob',
    'odds_ratio', 'home_favorite', 'prob_edge',
    'ah_line',
    'h2h_matches',
    'match_num', 'early_season',
]

# Check which features exist
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"Features: {feature_cols}")

X = df[feature_cols].copy()
y = df['ah_target'].values

# Fill missing with median
X = X.fillna(X.median())

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
print("\n" + "="*60)
print("WALK-FORWARD VALIDATION")
print("="*60)

# Split by time periods (seasons)
# Train: 2019-2022, Test: 2023
# Train: 2019-2023, Test: 2024

def get_season_year(date):
    """Football season year (Aug-Jul)"""
    if date.month >= 8:
        return date.year
    return date.year - 1

df['season_year'] = df['Date'].apply(get_season_year)
print(f"\nSeason distribution:")
print(df['season_year'].value_counts().sort_index())

# Define train/test splits
splits = [
    {'train_end': '2022-07-31', 'test_start': '2022-08-01', 'test_end': '2023-07-31', 'name': 'test_2223'},
    {'train_end': '2023-07-31', 'test_start': '2023-08-01', 'test_end': '2024-07-31', 'name': 'test_2324'},
    {'train_end': '2024-07-31', 'test_start': '2024-08-01', 'test_end': '2025-07-31', 'name': 'test_2425'},
]

all_results = []

for split in splits:
    print(f"\n{'-'*50}")
    print(f"Split: {split['name']}")
    
    train_mask = df['Date'] <= split['train_end']
    test_mask = (df['Date'] >= split['test_start']) & (df['Date'] <= split['test_end'])
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    df_test = df[test_mask].copy()
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    if len(X_test) < 100:
        print("  Skipping - not enough test data")
        continue
    
    # Train models
    models = {
        'lgbm': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, 
                                    random_state=42, verbose=-1),
        'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                                          random_state=42),
        'lr': LogisticRegression(max_iter=1000, random_state=42),
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        print(f"  {name}: Acc={acc:.3f}, AUC={auc:.3f}, Brier={brier:.3f}")
        
        # Store predictions for backtesting
        df_test[f'{name}_prob'] = y_prob
        df_test[f'{name}_pred'] = y_pred
        
        all_results.append({
            'split': split['name'],
            'model': name,
            'accuracy': acc,
            'auc': auc,
            'brier': brier,
            'n_test': len(X_test),
            'base_rate': y_test.mean()
        })
    
    # Ensemble
    df_test['ensemble_prob'] = (df_test['lgbm_prob'] + df_test['gb_prob'] + df_test['lr_prob']) / 3
    df_test['ensemble_pred'] = (df_test['ensemble_prob'] > 0.5).astype(int)
    
    acc = accuracy_score(y_test, df_test['ensemble_pred'])
    auc = roc_auc_score(y_test, df_test['ensemble_prob'])
    print(f"  ensemble: Acc={acc:.3f}, AUC={auc:.3f}")
    
    all_results.append({
        'split': split['name'],
        'model': 'ensemble',
        'accuracy': acc,
        'auc': auc,
        'brier': brier_score_loss(y_test, df_test['ensemble_prob']),
        'n_test': len(X_test),
        'base_rate': y_test.mean()
    })

# ============================================================
# AGGREGATE RESULTS
# ============================================================
print("\n" + "="*60)
print("AGGREGATE RESULTS")
print("="*60)

results_df = pd.DataFrame(all_results)
print("\nBy model (averaged across splits):")
print(results_df.groupby('model')[['accuracy', 'auc', 'brier']].mean())

print("\nBy split (best model):")
for split in results_df['split'].unique():
    split_data = results_df[results_df['split'] == split]
    best = split_data.loc[split_data['accuracy'].idxmax()]
    print(f"  {split}: {best['model']} with Acc={best['accuracy']:.3f}")

# Save results
results_df.to_csv(f'{BACKTEST_DIR}/model_results.csv', index=False)

# ============================================================
# TRAIN FINAL MODEL ON ALL DATA UP TO 2024
# ============================================================
print("\n" + "="*60)
print("TRAINING FINAL MODEL")
print("="*60)

train_mask = df['Date'] <= '2024-07-31'
X_final = X[train_mask]
y_final = y[train_mask]

print(f"Training on {len(X_final)} matches")

final_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, 
                                  random_state=42, verbose=-1)
final_model.fit(X_final, y_final)

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importance:")
print(importance)

# Save model
with open(f'{MODEL_DIR}/lgbm_ah.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print(f"\nModel saved to {MODEL_DIR}/lgbm_ah.pkl")
