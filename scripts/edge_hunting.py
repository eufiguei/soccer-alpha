#!/usr/bin/env python3
"""
Deep edge hunting - test multiple hypotheses systematically.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'

print("Loading real AH data...")
df = pd.read_parquet(f'{DATA_DIR}/real_ah_bettable.parquet')
df = df.sort_values('Date').reset_index(drop=True)

df['target'] = df['real_ah_result'].astype(int)

# Add features
df['home_1x2_odds'] = pd.to_numeric(df['B365H'], errors='coerce')
df['away_1x2_odds'] = pd.to_numeric(df['B365A'], errors='coerce')
df['home_1x2_prob'] = 1 / df['home_1x2_odds']
df['away_1x2_prob'] = 1 / df['away_1x2_odds']

df['home_shots'] = pd.to_numeric(df.get('HS', 0), errors='coerce').fillna(0)
df['away_shots'] = pd.to_numeric(df.get('AS', 0), errors='coerce').fillna(0)

# Test set (2023-24 and 2024-25)
df_test = df[df['Date'] >= '2022-08-01'].copy()
print(f"Test matches: {len(df_test)}")

def test_hypothesis(df_subset, description, side='away'):
    """Test a betting strategy"""
    if len(df_subset) < 50:
        return None
    
    if side == 'away':
        wins = (df_subset['target'] == 0).sum()
        odds_col = 'ah_away_odds'
    else:
        wins = (df_subset['target'] == 1).sum()
        odds_col = 'ah_home_odds'
    
    n = len(df_subset)
    wr = wins / n
    
    # PnL with actual odds
    pnl = sum(
        (row[odds_col] - 1) if (row['target'] == (0 if side == 'away' else 1)) else -1
        for _, row in df_subset.iterrows()
    )
    roi = pnl / n * 100
    
    # Statistical test vs 50%
    z = (wr - 0.5) / np.sqrt(0.5 * 0.5 / n)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'hypothesis': description,
        'side': side,
        'n_bets': n,
        'wins': wins,
        'win_rate': wr,
        'pnl': pnl,
        'roi': roi,
        'z_score': z,
        'p_value': p,
        'significant': p < 0.05
    }

results = []

print("\n" + "="*70)
print("HYPOTHESIS TESTING")
print("="*70)

# ============================================================
# HYPOTHESIS 1: Heavy favorites (big negative AH lines)
# ============================================================
print("\n1. HEAVY FAVORITES")
print("-"*50)

# Fade heavy favorites (bet away when home is -1.5 or worse)
mask = df_test['real_ah_line'] <= -1.5
r = test_hypothesis(df_test[mask], "Fade big favorites (AH <= -1.5)", 'away')
if r:
    results.append(r)
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}%, p={r['p_value']:.3f}")

# More extreme
mask = df_test['real_ah_line'] <= -2.0
r = test_hypothesis(df_test[mask], "Fade huge favorites (AH <= -2)", 'away')
if r:
    results.append(r)
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}%, p={r['p_value']:.3f}")

# ============================================================
# HYPOTHESIS 2: Underdogs
# ============================================================
print("\n2. UNDERDOGS")
print("-"*50)

# Back underdogs (bet away when home is AH +0.5 or more)
mask = df_test['real_ah_line'] >= 0.5
r = test_hypothesis(df_test[mask], "Back underdogs (AH >= +0.5)", 'away')
if r:
    results.append(r)
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}%, p={r['p_value']:.3f}")

# Home underdogs (home gets goals, bet home)
mask = df_test['real_ah_line'] >= 1.0
r = test_hypothesis(df_test[mask], "Home underdog (AH >= +1)", 'home')
if r:
    results.append(r)
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}%, p={r['p_value']:.3f}")

# ============================================================
# HYPOTHESIS 3: By league
# ============================================================
print("\n3. BY LEAGUE")
print("-"*50)

for league in df_test['League'].unique():
    mask = df_test['League'] == league
    r = test_hypothesis(df_test[mask], f"{league} - bet away", 'away')
    if r:
        results.append(r)
        sig = "✅" if r['significant'] else ""
        print(f"   {league}: {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}% {sig}")

# ============================================================
# HYPOTHESIS 4: Early season
# ============================================================
print("\n4. EARLY SEASON (First 5 matchdays)")
print("-"*50)

# First ~50 matches per season
df_test['match_num'] = df_test.groupby(['League', 'Season']).cumcount() + 1
mask = df_test['match_num'] <= 50
r = test_hypothesis(df_test[mask], "Early season - bet away", 'away')
if r:
    results.append(r)
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}%, p={r['p_value']:.3f}")

# ============================================================
# HYPOTHESIS 5: Odds discrepancy
# ============================================================
print("\n5. ODDS DISCREPANCY (1X2 vs AH implied)")
print("-"*50)

# When 1X2 odds suggest home more strongly than AH, maybe bet away?
df_test['prob_diff'] = df_test['home_1x2_prob'] - (1/df_test['ah_home_odds'])
mask = df_test['prob_diff'] > 0.05  # 1X2 thinks home is stronger than AH market
r = test_hypothesis(df_test[mask], "1X2 overprices home (diff>5%)", 'away')
if r:
    results.append(r)
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}%, p={r['p_value']:.3f}")

# ============================================================
# HYPOTHESIS 6: Line shopping (high away odds)
# ============================================================
print("\n6. LINE VALUE (High AH away odds)")
print("-"*50)

mask = df_test['ah_away_odds'] >= 2.0
r = test_hypothesis(df_test[mask], "High away odds (>=2.0)", 'away')
if r:
    results.append(r)
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}%, p={r['p_value']:.3f}")

mask = df_test['ah_away_odds'] >= 2.05
r = test_hypothesis(df_test[mask], "Very high away odds (>=2.05)", 'away')
if r:
    results.append(r)
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}%, p={r['p_value']:.3f}")

# ============================================================
# HYPOTHESIS 7: AH line value
# ============================================================
print("\n7. SPECIFIC AH LINES")
print("-"*50)

for line in sorted(df_test['real_ah_line'].unique()):
    if abs(line) > 2.5:
        continue
    mask = df_test['real_ah_line'] == line
    if mask.sum() >= 50:
        r = test_hypothesis(df_test[mask], f"AH line {line:+.2f}", 'away')
        if r:
            sig = "✅" if r['significant'] else ""
            roi_str = f"{r['roi']:+.1f}%"
            print(f"   Line {line:+.2f}: {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={roi_str} {sig}")
            results.append(r)

# ============================================================
# HYPOTHESIS 8: Combo strategies
# ============================================================
print("\n8. COMBO STRATEGIES")
print("-"*50)

# Serie A + big favorite
mask = (df_test['League'] == 'SerieA') & (df_test['real_ah_line'] <= -1.0)
r = test_hypothesis(df_test[mask], "Serie A big favorites (fade)", 'away')
if r:
    results.append(r)
    sig = "✅" if r['significant'] else ""
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}% {sig}")

# La Liga + underdogs
mask = (df_test['League'] == 'LaLiga') & (df_test['real_ah_line'] >= 0.5)
r = test_hypothesis(df_test[mask], "La Liga underdogs", 'away')
if r:
    results.append(r)
    sig = "✅" if r['significant'] else ""
    print(f"   {r['n_bets']} bets, WR={r['win_rate']:.1%}, ROI={r['roi']:.1f}% {sig}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY: SIGNIFICANT FINDINGS")
print("="*70)

results_df = pd.DataFrame(results)
significant = results_df[results_df['significant']]

if len(significant) > 0:
    print("\n🚨 STRATEGIES WITH p < 0.05:")
    for _, row in significant.iterrows():
        print(f"\n   {row['hypothesis']}")
        print(f"   Side: {row['side']}, Bets: {row['n_bets']}, WR: {row['win_rate']:.1%}")
        print(f"   PnL: ${row['pnl']:.2f}, ROI: {row['roi']:.1f}%, p={row['p_value']:.4f}")
else:
    print("\n❌ No strategies reached p < 0.05 significance")

# Bonferroni correction
print("\n" + "-"*50)
n_tests = len(results)
bonferroni_alpha = 0.05 / n_tests
print(f"Bonferroni correction: {n_tests} tests, α = {bonferroni_alpha:.4f}")

bonf_sig = results_df[results_df['p_value'] < bonferroni_alpha]
if len(bonf_sig) > 0:
    print("✅ PASSES BONFERRONI:")
    print(bonf_sig[['hypothesis', 'n_bets', 'win_rate', 'roi', 'p_value']].to_string())
else:
    print("❌ No strategy passes Bonferroni correction")

# Save results
results_df.to_csv(f'{DATA_DIR}/../backtests/hypothesis_tests.csv', index=False)
print(f"\nSaved {len(results)} hypothesis tests")
