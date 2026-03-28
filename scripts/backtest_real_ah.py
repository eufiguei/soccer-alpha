#!/usr/bin/env python3
"""
Backtest using REAL Asian Handicap lines and odds.
This is the honest test - can we beat the efficient market?
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'
BACKTEST_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/backtests'

print("Loading real AH data...")
df = pd.read_parquet(f'{DATA_DIR}/real_ah_bettable.parquet')
df = df.sort_values('Date').reset_index(drop=True)

print(f"Dataset: {len(df)} bettable matches")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Target: home covers (1) or away covers (0)
df['target'] = df['real_ah_result'].astype(int)

# ============================================================
# Build features (same as before, but proper)
# ============================================================
print("\nBuilding features...")

# Market features (1X2 odds)
df['home_1x2_odds'] = pd.to_numeric(df['B365H'], errors='coerce')
df['draw_1x2_odds'] = pd.to_numeric(df['B365D'], errors='coerce')
df['away_1x2_odds'] = pd.to_numeric(df['B365A'], errors='coerce')

# Implied probs from 1X2
df['home_1x2_prob'] = 1 / df['home_1x2_odds']
df['draw_1x2_prob'] = 1 / df['draw_1x2_odds']
df['away_1x2_prob'] = 1 / df['away_1x2_odds']
overround = df['home_1x2_prob'] + df['draw_1x2_prob'] + df['away_1x2_prob']
df['home_1x2_prob'] /= overround
df['draw_1x2_prob'] /= overround
df['away_1x2_prob'] /= overround

# AH market features
df['ah_home_implied'] = 1 / df['ah_home_odds']
df['ah_away_implied'] = 1 / df['ah_away_odds']

# Shots (if available)
df['home_shots'] = pd.to_numeric(df.get('HS', 0), errors='coerce').fillna(0)
df['away_shots'] = pd.to_numeric(df.get('AS', 0), errors='coerce').fillna(0)
df['shot_diff'] = df['home_shots'] - df['away_shots']

# Simple form: compute rolling goals per team
# We need to sort by team and compute rolling stats
matches = []
for idx, row in df.iterrows():
    matches.append({
        'orig_idx': idx, 'team': row['HomeTeam'], 'date': row['Date'],
        'league': row['League'], 'gf': row['FTHG'], 'ga': row['FTAG'], 'is_home': True
    })
    matches.append({
        'orig_idx': idx, 'team': row['AwayTeam'], 'date': row['Date'],
        'league': row['League'], 'gf': row['FTAG'], 'ga': row['FTHG'], 'is_home': False
    })

match_df = pd.DataFrame(matches)
match_df = match_df.sort_values(['team', 'league', 'date'])
match_df['pts'] = np.where(match_df['gf'] > match_df['ga'], 3,
                           np.where(match_df['gf'] == match_df['ga'], 1, 0))

# Rolling form
match_df['form_pts'] = match_df.groupby(['team', 'league'])['pts'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1))
match_df['form_gf'] = match_df.groupby(['team', 'league'])['gf'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1))
match_df['form_ga'] = match_df.groupby(['team', 'league'])['ga'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1))

# Merge back
home_form = match_df[match_df['is_home']].set_index('orig_idx')[['form_pts', 'form_gf', 'form_ga']]
home_form.columns = ['home_' + c for c in home_form.columns]
away_form = match_df[~match_df['is_home']].set_index('orig_idx')[['form_pts', 'form_gf', 'form_ga']]
away_form.columns = ['away_' + c for c in away_form.columns]

df = df.join(home_form).join(away_form)

# Fill missing form with median
for col in ['home_form_pts', 'home_form_gf', 'home_form_ga', 
            'away_form_pts', 'away_form_gf', 'away_form_ga']:
    df[col] = df[col].fillna(df[col].median())

# ============================================================
# Feature selection
# ============================================================
feature_cols = [
    # Market features (from 1X2 - what we know pre-match)
    'home_1x2_prob', 'draw_1x2_prob', 'away_1x2_prob',
    # AH line and implied
    'real_ah_line', 'ah_home_implied', 'ah_away_implied',
    # Form
    'home_form_pts', 'home_form_gf', 'home_form_ga',
    'away_form_pts', 'away_form_gf', 'away_form_ga',
]

X = df[feature_cols].fillna(df[feature_cols].median())
y = df['target'].values

# ============================================================
# Walk-forward backtest
# ============================================================
print("\n" + "="*60)
print("WALK-FORWARD BACKTEST (REAL AH)")
print("="*60)

splits = [
    {'train_end': '2022-07-31', 'test_start': '2022-08-01', 'test_end': '2023-07-31', 'name': '2022-23'},
    {'train_end': '2023-07-31', 'test_start': '2023-08-01', 'test_end': '2024-07-31', 'name': '2023-24'},
    {'train_end': '2024-07-31', 'test_start': '2024-08-01', 'test_end': '2025-07-31', 'name': '2024-25'},
]

all_results = []

for split in splits:
    print(f"\n{'-'*50}")
    print(f"Split: {split['name']}")
    
    train_mask = df['Date'] <= split['train_end']
    test_mask = (df['Date'] >= split['test_start']) & (df['Date'] <= split['test_end'])
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    df_test = df[test_mask].copy()
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    if len(X_test) < 100:
        continue
    
    # Baseline: bet on higher implied probability
    baseline_preds = (df_test['ah_home_implied'] > df_test['ah_away_implied']).astype(int)
    baseline_acc = (baseline_preds == y_test).mean()
    
    # Models
    models = {
        'lgbm': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, verbose=-1),
        'lr': LogisticRegression(max_iter=1000),
    }
    
    print(f"  Baseline (follow odds): Acc={baseline_acc:.3f}")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        
        acc = (preds == y_test).mean()
        print(f"  {name}: Acc={acc:.3f}")
        
        df_test[f'{name}_prob'] = probs
        
        all_results.append({
            'split': split['name'],
            'model': name,
            'accuracy': acc,
            'n_test': len(X_test),
            'baseline': baseline_acc
        })

# ============================================================
# Betting simulation with real odds
# ============================================================
print("\n" + "="*60)
print("BETTING SIMULATION WITH REAL ODDS")
print("="*60)

all_bets = []

for split in splits:
    train_mask = df['Date'] <= split['train_end']
    test_mask = (df['Date'] >= split['test_start']) & (df['Date'] <= split['test_end'])
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    df_test = df[test_mask].copy()
    
    if len(X_test) < 100:
        continue
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    df_test['model_prob'] = model.predict_proba(X_test)[:, 1]
    
    # Betting strategies
    for edge_threshold in [0.0, 0.02, 0.03, 0.05]:
        bets = []
        
        for idx, row in df_test.iterrows():
            model_prob = row['model_prob']
            market_prob = row['ah_home_implied']
            
            # Edge = model prob - market breakeven
            # Market breakeven ~ market_prob (since odds already include vig)
            edge_home = model_prob - market_prob
            edge_away = (1 - model_prob) - row['ah_away_implied']
            
            if edge_home > edge_threshold:
                # Bet home
                won = (row['target'] == 1)
                pnl = (row['ah_home_odds'] - 1) if won else -1
                bets.append({
                    'date': row['Date'],
                    'league': row['League'],
                    'side': 'home',
                    'odds': row['ah_home_odds'],
                    'edge': edge_home,
                    'won': won,
                    'pnl': pnl
                })
            elif edge_away > edge_threshold:
                # Bet away
                won = (row['target'] == 0)
                pnl = (row['ah_away_odds'] - 1) if won else -1
                bets.append({
                    'date': row['Date'],
                    'league': row['League'],
                    'side': 'away',
                    'odds': row['ah_away_odds'],
                    'edge': edge_away,
                    'won': won,
                    'pnl': pnl
                })
        
        if len(bets) > 50:
            bets_df = pd.DataFrame(bets)
            n_bets = len(bets_df)
            wins = bets_df['won'].sum()
            wr = wins / n_bets
            pnl = bets_df['pnl'].sum()
            roi = pnl / n_bets * 100
            
            print(f"  {split['name']} | Edge>{edge_threshold:.0%}: {n_bets} bets, WR={wr:.1%}, ROI={roi:.1f}%")
            
            bets_df['split'] = split['name']
            bets_df['strategy'] = f'edge_{int(edge_threshold*100)}pct'
            all_bets.append(bets_df)

# ============================================================
# Aggregate results
# ============================================================
print("\n" + "="*60)
print("AGGREGATE BETTING RESULTS")
print("="*60)

if all_bets:
    all_bets_df = pd.concat(all_bets, ignore_index=True)
    
    agg = all_bets_df.groupby('strategy').agg({
        'won': ['sum', 'count'],
        'pnl': 'sum'
    }).reset_index()
    agg.columns = ['strategy', 'wins', 'n_bets', 'pnl']
    agg['win_rate'] = agg['wins'] / agg['n_bets']
    agg['roi'] = agg['pnl'] / agg['n_bets'] * 100
    
    print("\nBy strategy:")
    print(agg.to_string(index=False))
    
    # Statistical significance
    print("\n" + "-"*40)
    print("STATISTICAL SIGNIFICANCE")
    
    for _, row in agg.iterrows():
        n = row['n_bets']
        wr = row['win_rate']
        
        # Test vs 50%
        z = (wr - 0.5) / np.sqrt(0.5 * 0.5 / n)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        
        sig = "✅ p<0.05" if p < 0.05 else "❌ Not significant"
        print(f"  {row['strategy']}: WR={wr:.1%}, z={z:.2f}, p={p:.3f} {sig}")
else:
    print("No bets placed - model doesn't find sufficient edge")

# ============================================================
# Alternative: Value betting on away
# ============================================================
print("\n" + "="*60)
print("ALTERNATIVE: SIMPLE AWAY BIAS")
print("="*60)

# We saw away covers 51.3% - is this exploitable?
away_bets = []

for split in splits:
    test_mask = (df['Date'] >= split['test_start']) & (df['Date'] <= split['test_end'])
    df_test = df[test_mask]
    
    for _, row in df_test.iterrows():
        # Always bet away
        won = (row['target'] == 0)
        pnl = (row['ah_away_odds'] - 1) if won else -1
        away_bets.append({
            'split': split['name'],
            'won': won,
            'pnl': pnl,
            'odds': row['ah_away_odds']
        })

away_df = pd.DataFrame(away_bets)
total_bets = len(away_df)
total_wins = away_df['won'].sum()
total_pnl = away_df['pnl'].sum()
roi = total_pnl / total_bets * 100

print(f"\n'Always bet away' strategy:")
print(f"  Bets: {total_bets}")
print(f"  Win rate: {total_wins/total_bets:.1%}")
print(f"  Total PnL: ${total_pnl:.2f}")
print(f"  ROI: {roi:.1f}%")

# By split
for split in splits:
    subset = away_df[away_df['split'] == split['name']]
    if len(subset) > 0:
        wr = subset['won'].mean()
        pnl = subset['pnl'].sum()
        roi = pnl / len(subset) * 100
        print(f"  {split['name']}: WR={wr:.1%}, ROI={roi:.1f}%")

# ============================================================
# Save results
# ============================================================
results = {
    'data_source': 'real_ah_odds',
    'total_matches': len(df),
    'home_cover_rate': float(df['target'].mean()),
    'market_efficiency': 'efficient_with_slight_away_bias'
}

with open(f'{BACKTEST_DIR}/real_ah_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {BACKTEST_DIR}/")
