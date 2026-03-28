#!/usr/bin/env python3
"""
Rigorous betting backtest with edge detection and multiple strategy tests.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'
BACKTEST_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/backtests'

print("Loading features...")
df = pd.read_parquet(f'{DATA_DIR}/features_bettable.parquet')
df = df.sort_values('Date').reset_index(drop=True)

# Feature columns
feature_cols = [
    'home_form_pts', 'home_form_gf', 'home_form_ga', 'home_form_gd',
    'away_form_pts', 'away_form_gf', 'away_form_ga', 'away_form_gd',
    'home_prob', 'away_prob', 'draw_prob',
    'odds_ratio', 'home_favorite', 'prob_edge',
    'ah_line',
    'h2h_matches',
    'match_num', 'early_season',
]
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].fillna(df[feature_cols].median())
y = df['ah_target'].values

# ============================================================
# WALK-FORWARD BETTING SIMULATION
# ============================================================
print("\n" + "="*60)
print("WALK-FORWARD BETTING SIMULATION")
print("="*60)

# AH odds are typically around 1.90-1.95 (5% vig each side)
AH_ODDS = 1.91  # Typical Asian book odds
BREAKEVEN = 1 / AH_ODDS  # ~0.524

def simulate_betting(df_test, prob_col, y_test, min_edge=0.0, bet_side='model'):
    """
    Simulate flat betting on predicted side.
    bet_side: 'model' = bet whatever model predicts
              'home' = always bet home covers
              'away' = always bet away covers
    """
    results = []
    
    for idx, (_, row) in enumerate(df_test.iterrows()):
        prob = row[prob_col]
        actual = y_test[idx]
        
        if bet_side == 'model':
            # Bet home if prob > 0.5, bet away otherwise
            if prob > 0.5:
                edge = prob - BREAKEVEN
                bet_home = True
            else:
                edge = (1 - prob) - BREAKEVEN
                bet_home = False
        elif bet_side == 'home':
            edge = prob - BREAKEVEN
            bet_home = True
        else:  # away
            edge = (1 - prob) - BREAKEVEN
            bet_home = False
        
        # Only bet if edge > threshold
        if edge < min_edge:
            continue
        
        # Outcome
        if bet_home:
            won = (actual == 1)
        else:
            won = (actual == 0)
        
        pnl = (AH_ODDS - 1) if won else -1  # Flat $1 bets
        
        results.append({
            'date': row['Date'],
            'league': row['League'],
            'prob': prob,
            'edge': edge,
            'bet_home': bet_home,
            'actual': actual,
            'won': won,
            'pnl': pnl
        })
    
    return pd.DataFrame(results)

# Define test periods
splits = [
    {'train_end': '2022-07-31', 'test_start': '2022-08-01', 'test_end': '2023-07-31', 'name': '2022-23'},
    {'train_end': '2023-07-31', 'test_start': '2023-08-01', 'test_end': '2024-07-31', 'name': '2023-24'},
    {'train_end': '2024-07-31', 'test_start': '2024-08-01', 'test_end': '2025-07-31', 'name': '2024-25'},
]

all_bets = []
strategy_results = []

for split in splits:
    print(f"\n{'='*50}")
    print(f"Testing: {split['name']}")
    
    train_mask = df['Date'] <= split['train_end']
    test_mask = (df['Date'] >= split['test_start']) & (df['Date'] <= split['test_end'])
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    df_test = df[test_mask].copy()
    
    if len(X_test) < 100:
        continue
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    df_test['model_prob'] = model.predict_proba(X_test)[:, 1]
    
    # Test multiple strategies
    strategies = [
        ('model_all', 'model', 0.0),
        ('model_edge_3pct', 'model', 0.03),
        ('model_edge_5pct', 'model', 0.05),
        ('model_edge_8pct', 'model', 0.08),
        ('model_edge_10pct', 'model', 0.10),
        ('always_away', 'away', 0.0),  # Baseline: always bet away covers
    ]
    
    for strat_name, bet_side, min_edge in strategies:
        bets = simulate_betting(df_test, 'model_prob', y_test, min_edge, bet_side)
        
        if len(bets) < 50:
            continue
        
        n_bets = len(bets)
        wins = bets['won'].sum()
        wr = wins / n_bets
        total_pnl = bets['pnl'].sum()
        roi = total_pnl / n_bets * 100
        
        print(f"  {strat_name}: {n_bets} bets, WR={wr:.1%}, ROI={roi:.1f}%, PnL=${total_pnl:.2f}")
        
        bets['split'] = split['name']
        bets['strategy'] = strat_name
        all_bets.append(bets)
        
        strategy_results.append({
            'split': split['name'],
            'strategy': strat_name,
            'n_bets': n_bets,
            'wins': wins,
            'win_rate': wr,
            'roi': roi,
            'pnl': total_pnl
        })

# ============================================================
# AGGREGATE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("AGGREGATE RESULTS (All Splits Combined)")
print("="*60)

all_bets_df = pd.concat(all_bets, ignore_index=True)
strat_df = pd.DataFrame(strategy_results)

# Aggregate by strategy
agg = strat_df.groupby('strategy').agg({
    'n_bets': 'sum',
    'wins': 'sum',
    'pnl': 'sum'
}).reset_index()
agg['win_rate'] = agg['wins'] / agg['n_bets']
agg['roi'] = agg['pnl'] / agg['n_bets'] * 100

print("\nStrategy performance (all test data):")
print(agg.sort_values('roi', ascending=False).to_string(index=False))

# ============================================================
# STATISTICAL SIGNIFICANCE
# ============================================================
print("\n" + "="*60)
print("STATISTICAL SIGNIFICANCE")
print("="*60)

from scipy import stats

for strat in agg['strategy'].unique():
    data = agg[agg['strategy'] == strat].iloc[0]
    n = int(data['n_bets'])
    wins = int(data['wins'])
    wr = data['win_rate']
    
    # One-sample proportion test vs breakeven (0.524)
    z = (wr - BREAKEVEN) / np.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n)
    p_value = 1 - stats.norm.cdf(z)
    
    # Confidence interval
    se = np.sqrt(wr * (1 - wr) / n)
    ci_low = wr - 1.96 * se
    ci_high = wr + 1.96 * se
    
    significant = p_value < 0.05
    sig_str = "✅ SIGNIFICANT" if significant else "❌ Not significant"
    
    print(f"\n{strat}:")
    print(f"  N={n}, WR={wr:.1%}, 95% CI=[{ci_low:.1%}, {ci_high:.1%}]")
    print(f"  vs breakeven {BREAKEVEN:.1%}: z={z:.2f}, p={p_value:.4f}")
    print(f"  {sig_str}")

# ============================================================
# BONFERRONI CORRECTION
# ============================================================
print("\n" + "="*60)
print("BONFERRONI CORRECTION (6 strategies)")
print("="*60)

alpha = 0.05
bonferroni_alpha = alpha / 6

print(f"Original alpha: {alpha}")
print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")

for strat in agg['strategy'].unique():
    data = agg[agg['strategy'] == strat].iloc[0]
    n = int(data['n_bets'])
    wr = data['win_rate']
    
    z = (wr - BREAKEVEN) / np.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n)
    p_value = 1 - stats.norm.cdf(z)
    
    passes = p_value < bonferroni_alpha
    result = "✅ PASSES" if passes else "❌ FAILS"
    print(f"  {strat}: p={p_value:.6f} {result}")

# ============================================================
# BY LEAGUE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("BEST STRATEGY BY LEAGUE")
print("="*60)

best_strat = agg.loc[agg['roi'].idxmax(), 'strategy']
best_bets = all_bets_df[all_bets_df['strategy'] == best_strat]

league_stats = best_bets.groupby('league').agg({
    'won': ['sum', 'count'],
    'pnl': 'sum'
}).reset_index()
league_stats.columns = ['league', 'wins', 'n_bets', 'pnl']
league_stats['win_rate'] = league_stats['wins'] / league_stats['n_bets']
league_stats['roi'] = league_stats['pnl'] / league_stats['n_bets'] * 100

print(f"\nStrategy: {best_strat}")
print(league_stats.sort_values('roi', ascending=False).to_string(index=False))

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'summary': {
        'total_matches': len(df),
        'test_matches': len(all_bets_df),
        'best_strategy': best_strat,
        'best_roi': float(agg.loc[agg['roi'].idxmax(), 'roi']),
        'best_win_rate': float(agg.loc[agg['roi'].idxmax(), 'win_rate']),
        'breakeven': BREAKEVEN,
        'ah_odds': AH_ODDS
    },
    'strategies': agg.to_dict('records'),
    'by_league': league_stats.to_dict('records'),
    'by_split': strat_df.to_dict('records')
}

with open(f'{BACKTEST_DIR}/betting_results.json', 'w') as f:
    json.dump(results, f, indent=2)

all_bets_df.to_parquet(f'{BACKTEST_DIR}/all_bets.parquet', index=False)

print(f"\nResults saved to {BACKTEST_DIR}/")
