#!/usr/bin/env python3
"""
Final validation of discovered edges with proper train/test split.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'
BACKTEST_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/backtests'

print("Loading real AH data...")
df = pd.read_parquet(f'{DATA_DIR}/real_ah_bettable.parquet')
df = df.sort_values('Date').reset_index(drop=True)
df['target'] = df['real_ah_result'].astype(int)

print(f"Total: {len(df)} matches")

# ============================================================
# TRAIN/TEST SPLIT (Clean separation)
# ============================================================
# Train: 2019-2023 (discovery)
# Test: 2024-2025 (validation)

train_mask = df['Date'] < '2023-08-01'
test_mask = df['Date'] >= '2023-08-01'

df_train = df[train_mask]
df_test = df[test_mask]

print(f"Train: {len(df_train)} matches (2019-2023)")
print(f"Test: {len(df_test)} matches (2023-2025)")

# ============================================================
# DISCOVER EDGES IN TRAIN SET
# ============================================================
print("\n" + "="*60)
print("DISCOVERY (Train Set 2019-2023)")
print("="*60)

def analyze_line(df, line, side='away'):
    subset = df[df['real_ah_line'] == line]
    if len(subset) < 30:
        return None
    
    if side == 'away':
        wins = (subset['target'] == 0).sum()
        odds_col = 'ah_away_odds'
    else:
        wins = (subset['target'] == 1).sum()
        odds_col = 'ah_home_odds'
    
    n = len(subset)
    wr = wins / n
    pnl = sum((row[odds_col] - 1) if (row['target'] == (0 if side == 'away' else 1)) else -1 
              for _, row in subset.iterrows())
    roi = pnl / n * 100
    
    z = (wr - 0.5) / np.sqrt(0.5 * 0.5 / n)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {'line': line, 'side': side, 'n': n, 'wr': wr, 'roi': roi, 'p': p}

discovered_edges = []

print("\nBy AH line (bet AWAY unless noted):")
for line in sorted(df_train['real_ah_line'].unique()):
    if abs(line) > 2.5:
        continue
    
    # Test away first
    r = analyze_line(df_train, line, 'away')
    if r and r['p'] < 0.05:
        action = "bet AWAY" if r['wr'] > 0.5 else "bet HOME"
        final_side = 'away' if r['wr'] > 0.5 else 'home'
        actual_wr = r['wr'] if r['wr'] > 0.5 else (1 - r['wr'])
        print(f"   Line {line:+.2f}: {r['n']} bets, WR={r['wr']:.1%} → {action}, p={r['p']:.4f}")
        
        discovered_edges.append({
            'line': line,
            'side': final_side,
            'train_n': r['n'],
            'train_wr': actual_wr if final_side == 'away' else (1-r['wr']),
            'train_roi': r['roi'] if final_side == 'away' else -r['roi'],
            'train_p': r['p']
        })

# ============================================================
# VALIDATE IN TEST SET
# ============================================================
print("\n" + "="*60)
print("VALIDATION (Test Set 2023-2025)")
print("="*60)

validated_strategies = []

for edge in discovered_edges:
    line = edge['line']
    side = edge['side']
    
    subset = df_test[df_test['real_ah_line'] == line]
    if len(subset) < 20:
        print(f"   Line {line:+.2f}: Insufficient test data ({len(subset)} matches)")
        continue
    
    if side == 'away':
        wins = (subset['target'] == 0).sum()
        odds_col = 'ah_away_odds'
    else:
        wins = (subset['target'] == 1).sum()
        odds_col = 'ah_home_odds'
    
    n = len(subset)
    wr = wins / n
    pnl = sum((row[odds_col] - 1) if (row['target'] == (0 if side == 'away' else 1)) else -1 
              for _, row in subset.iterrows())
    roi = pnl / n * 100
    
    # Combined significance
    combined_n = edge['train_n'] + n
    combined_wins = int(edge['train_wr'] * edge['train_n']) + wins
    combined_wr = combined_wins / combined_n
    
    z = (combined_wr - 0.5) / np.sqrt(0.5 * 0.5 / combined_n)
    p_combined = 2 * (1 - stats.norm.cdf(abs(z)))
    
    validated = (wr > 0.5) and (roi > 0)
    status = "✅ VALIDATED" if validated else "❌ Failed"
    
    print(f"\n   Line {line:+.2f} ({side.upper()}):")
    print(f"      Train: {edge['train_n']} bets, WR={edge['train_wr']:.1%}, ROI={edge['train_roi']:.1f}%")
    print(f"      Test:  {n} bets, WR={wr:.1%}, ROI={roi:.1f}%")
    print(f"      Combined: {combined_n} bets, WR={combined_wr:.1%}, p={p_combined:.4f}")
    print(f"      {status}")
    
    if validated:
        validated_strategies.append({
            'line': line,
            'side': side,
            'train_n': edge['train_n'],
            'train_wr': edge['train_wr'],
            'train_roi': edge['train_roi'],
            'test_n': n,
            'test_wr': wr,
            'test_roi': roi,
            'combined_n': combined_n,
            'combined_wr': combined_wr,
            'combined_p': p_combined
        })

# ============================================================
# FINAL STRATEGY
# ============================================================
print("\n" + "="*60)
print("FINAL VALIDATED STRATEGY")
print("="*60)

if validated_strategies:
    print("\n🎯 DEPLOY THESE EDGES:")
    
    total_train_bets = 0
    total_test_bets = 0
    total_train_pnl = 0
    total_test_pnl = 0
    
    for strat in validated_strategies:
        print(f"\n   • AH Line {strat['line']:+.2f}: Bet {strat['side'].upper()}")
        print(f"     Train: {strat['train_n']} bets, {strat['train_wr']:.1%} WR, {strat['train_roi']:.1f}% ROI")
        print(f"     Test:  {strat['test_n']} bets, {strat['test_wr']:.1%} WR, {strat['test_roi']:.1f}% ROI")
        
        total_train_bets += strat['train_n']
        total_test_bets += strat['test_n']
        total_train_pnl += strat['train_n'] * strat['train_roi'] / 100
        total_test_pnl += strat['test_n'] * strat['test_roi'] / 100
    
    print(f"\n📊 COMBINED PERFORMANCE:")
    print(f"   Train: {total_train_bets} bets, ${total_train_pnl:.2f} profit, {total_train_pnl/total_train_bets*100:.1f}% ROI")
    print(f"   Test:  {total_test_bets} bets, ${total_test_pnl:.2f} profit, {total_test_pnl/total_test_bets*100:.1f}% ROI")
    
    # Bonferroni check
    n_strategies = len(discovered_edges)
    bonf_alpha = 0.05 / n_strategies
    passes_bonferroni = [s for s in validated_strategies if s['combined_p'] < bonf_alpha]
    
    print(f"\n🔬 STATISTICAL RIGOR:")
    print(f"   Strategies tested: {n_strategies}")
    print(f"   Bonferroni α: {bonf_alpha:.4f}")
    print(f"   Pass Bonferroni: {len(passes_bonferroni)}/{len(validated_strategies)}")
    
    if passes_bonferroni:
        print("\n   ✅ BONFERRONI-VALIDATED EDGES:")
        for s in passes_bonferroni:
            print(f"      • Line {s['line']:+.2f}: p={s['combined_p']:.6f}")
else:
    print("\n❌ No edges survived out-of-sample validation")

# ============================================================
# SAVE FINAL RESULTS
# ============================================================
final_results = {
    'methodology': {
        'train_period': '2019-08-01 to 2023-07-31',
        'test_period': '2023-08-01 to 2025-05-25',
        'train_matches': len(df_train),
        'test_matches': len(df_test),
        'total_strategies_tested': len(discovered_edges),
        'bonferroni_alpha': float(0.05 / len(discovered_edges)) if discovered_edges else None
    },
    'validated_strategies': validated_strategies,
    'recommendation': 'DEPLOY' if len(passes_bonferroni) > 0 else 'MONITOR' if validated_strategies else 'REJECT'
}

with open(f'{BACKTEST_DIR}/final_results.json', 'w') as f:
    json.dump(final_results, f, indent=2, default=str)

print(f"\nResults saved to {BACKTEST_DIR}/final_results.json")

# ============================================================
# EXPECTED VALUE CALCULATION
# ============================================================
if validated_strategies:
    print("\n" + "="*60)
    print("EXPECTED VALUE PROJECTION")
    print("="*60)
    
    # Estimate annual opportunity
    test_months = (pd.to_datetime(df_test['Date'].max()) - pd.to_datetime(df_test['Date'].min())).days / 30
    monthly_bets = total_test_bets / test_months
    monthly_profit = total_test_pnl / test_months
    
    print(f"\n   Test period: {test_months:.1f} months")
    print(f"   Monthly bets: ~{monthly_bets:.0f}")
    print(f"   Monthly profit (flat $1 bets): ~${monthly_profit:.2f}")
    
    # Project annual with Kelly sizing
    avg_wr = sum(s['test_wr'] * s['test_n'] for s in validated_strategies) / total_test_bets
    avg_odds = 1.91  # Typical AH odds
    kelly_fraction = (avg_wr * (avg_odds - 1) - (1 - avg_wr)) / (avg_odds - 1)
    
    print(f"\n   Average WR: {avg_wr:.1%}")
    print(f"   Full Kelly fraction: {kelly_fraction:.1%}")
    print(f"   Recommended (1/4 Kelly): {kelly_fraction/4:.1%}")
    
    # With $100 bankroll
    bankroll = 100
    bet_size = bankroll * kelly_fraction / 4
    monthly_profit_sized = monthly_profit * bet_size
    annual_profit = monthly_profit_sized * 12
    
    print(f"\n   With ${bankroll} bankroll:")
    print(f"   Bet size: ${bet_size:.2f}")
    print(f"   Monthly expected profit: ${monthly_profit_sized:.2f}")
    print(f"   Annual expected profit: ${annual_profit:.2f}")
    print(f"   Annual ROI: {annual_profit/bankroll*100:.1f}%")
