"""
Alpha Combination Analysis - Build unified decision tree from consistent rules
Test out-of-sample on 2024-25 data and generate picker update
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Loading data and building features...")
df = pd.read_parquet('data/real_ah_bettable.parquet')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Rebuild features (fast version using groupby)
teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())

for col in ['h_gf', 'h_ga', 'h_form', 'a_gf', 'a_ga', 'a_form']:
    df[col] = np.nan

for team in teams:
    mask_h = df['HomeTeam'] == team
    idx_h = df.index[mask_h]
    if len(idx_h) >= 3:
        goals_h = df.loc[mask_h, 'FTHG'].values
        conc_h = df.loc[mask_h, 'FTAG'].values
        pts_h = np.where(goals_h > conc_h, 3, np.where(goals_h == conc_h, 1, 0)).astype(float)
        for j in range(len(idx_h)):
            if j >= 3:
                s = max(0, j - 6)
                df.loc[idx_h[j], 'h_gf'] = np.mean(goals_h[s:j])
                df.loc[idx_h[j], 'h_ga'] = np.mean(conc_h[s:j])
                df.loc[idx_h[j], 'h_form'] = np.mean(pts_h[s:j])

    mask_a = df['AwayTeam'] == team
    idx_a = df.index[mask_a]
    if len(idx_a) >= 3:
        goals_a = df.loc[mask_a, 'FTAG'].values
        conc_a = df.loc[mask_a, 'FTHG'].values
        pts_a = np.where(goals_a > conc_a, 3, np.where(goals_a == conc_a, 1, 0)).astype(float)
        for j in range(len(idx_a)):
            if j >= 3:
                s = max(0, j - 6)
                df.loc[idx_a[j], 'a_gf'] = np.mean(goals_a[s:j])
                df.loc[idx_a[j], 'a_ga'] = np.mean(conc_a[s:j])
                df.loc[idx_a[j], 'a_form'] = np.mean(pts_a[s:j])

# Derived features
df['form_gap'] = df['h_form'] - df['a_form']
df['gf_gap'] = df['h_gf'] - df['a_gf']
df['attack_vs_defense'] = df['h_gf'] - df['a_ga']
df['implied_gap'] = df['home_implied'] - df['away_implied']
df['odds_ratio'] = df['ah_home_odds'] / df['ah_away_odds']
df['month'] = df['Date'].dt.month

df_feat = df.dropna(subset=['h_form', 'a_form', 'h_gf', 'a_gf'])
print(f"Rows with features: {len(df_feat)}")

SEASONS = sorted(df_feat['Season'].unique())

# ============================================================
# DEFINE VALIDATED RULES
# ============================================================
# Each rule: (name, mask_fn, bet_side)

def rule_AH_minus025(d):
    """AH -0.25 → HOME (6/6 seasons, WR=59.2%)"""
    return d['real_ah_line'] == -0.25

def rule_AH_plus025(d):
    """AH +0.25 → AWAY (6/6 seasons, WR=64.3%)"""
    return d['real_ah_line'] == 0.25

def rule_positive_lines(d):
    """Positive AH lines (home underdog) → AWAY (5/6 seasons, WR=54.3%)"""
    return d['real_ah_line'] > 0

def rule_cold_home(d):
    """Cold home team (form < 0.8) → AWAY (5/6 seasons, WR=53.8%)"""
    return d['h_form'] < 0.8

def rule_AH075_weak_attack(d):
    """AH -0.75 + home attack weak vs away defense → AWAY (4/6 seasons, WR=57.2%)"""
    return (d['real_ah_line'] == -0.75) & (d['attack_vs_defense'] < 0.3)

def rule_cold_home_form(d):
    """Cold home + weaker form gap → AWAY (4/6 seasons, WR=54.2%)"""
    return (d['h_form'] < 1.0) & (d['form_gap'] < 0)

def rule_mediocre_both(d):
    """Both mediocre form (1.2-1.8) → AWAY (4/6 seasons, WR=53.8%)"""
    return (1.2 <= d['h_form'] <= 1.8) and (1.2 <= d['a_form'] <= 1.8)

# Marginal but strong candidates
def rule_AH175(d):
    """AH -1.75 → AWAY (known signal, WR=67.3%)"""
    return d['real_ah_line'] == -1.75

def rule_AH_150(d):
    """AH -1.5 → AWAY (marginal, WR=54.4%)"""
    return d['real_ah_line'] == -1.5

def rule_heavy_fav_form(d):
    """Lines -1.0 to -1.5 + form_gap > 1.5 → HOME (marginal, WR=56.2%)"""
    return (d['real_ah_line'].between(-1.5, -1.0)) & (d['form_gap'] > 1.5)

# ============================================================
# UNIFIED PICKER LOGIC
# ============================================================
def unified_picker(d):
    """
    Returns (bet_side, rule_name, confidence) or None if no bet
    Priority order: highest confidence signals first
    """
    
    # TIER 1: 6/6 seasons (highest confidence)
    if rule_AH_minus025(d):
        return ('home', 'AH_-0.25_HOME', 'HIGH')
    
    if rule_AH_plus025(d):
        return ('away', 'AH_+0.25_AWAY', 'HIGH')
    
    # TIER 2: 5/6 seasons + strong form signal
    if rule_AH175(d):
        return ('away', 'AH_-1.75_AWAY', 'HIGH')
    
    if rule_cold_home(d) and d['real_ah_line'] < -0.25:
        return ('away', 'COLD_HOME_AWAY', 'HIGH')
    
    # TIER 3: 4/6 seasons with specific conditions
    if rule_AH075_weak_attack(d):
        return ('away', 'AH_-0.75_WEAK_ATK_AWAY', 'MEDIUM')
    
    if rule_positive_lines(d) and d['h_form'] < 1.5:
        return ('away', 'POS_LINE_COLD_HOME_AWAY', 'MEDIUM')
    
    if rule_cold_home_form(d):
        return ('away', 'COLD_HOME_FORM_AWAY', 'MEDIUM')
    
    if rule_mediocre_both(d):
        return ('away', 'MEDIOCRE_BOTH_AWAY', 'MEDIUM')
    
    # Positive lines general
    if rule_positive_lines(d):
        return ('away', 'POS_LINE_AWAY', 'MEDIUM')
    
    return None

# ============================================================
# TEST UNIFIED PICKER ON FULL DATA AND OOS
# ============================================================
print("\n" + "="*60)
print("TESTING UNIFIED PICKER")
print("="*60)

# Apply picker to all rows with features
bets = []
for idx, row in df_feat.iterrows():
    result = unified_picker(row)
    if result:
        bet_side, rule_name, confidence = result
        if bet_side == 'home':
            won = (row['real_ah_result'] == 1.0)
            odds = row['ah_home_odds']
        else:
            won = (row['real_ah_result'] == 0.0)
            odds = row['ah_away_odds']
        
        bets.append({
            'Date': row['Date'],
            'Season': row['Season'],
            'HomeTeam': row['HomeTeam'],
            'AwayTeam': row['AwayTeam'],
            'real_ah_line': row['real_ah_line'],
            'bet_side': bet_side,
            'rule': rule_name,
            'confidence': confidence,
            'odds': odds,
            'won': won,
            'roi': (odds - 1) if won else -1,
        })

bets_df = pd.DataFrame(bets)
print(f"\nTotal bets generated: {len(bets_df)}")
print(f"Coverage: {len(bets_df)/len(df_feat)*100:.1f}% of all games")

print("\n--- OVERALL STATS ---")
print(f"WR: {bets_df['won'].mean():.1%}")
print(f"ROI: {bets_df['roi'].mean():+.1%}")
print(f"Total seasons: {bets_df['Season'].nunique()}")

print("\n--- BY SEASON ---")
for s in SEASONS:
    s_bets = bets_df[bets_df['Season'] == s]
    if len(s_bets) > 0:
        print(f"  {s}: n={len(s_bets)} WR={s_bets['won'].mean():.1%} ROI={s_bets['roi'].mean():+.1%}")

print("\n--- BY RULE ---")
for rule in bets_df['rule'].unique():
    r_bets = bets_df[bets_df['rule'] == rule]
    print(f"  {rule}: n={len(r_bets)} WR={r_bets['won'].mean():.1%} ROI={r_bets['roi'].mean():+.1%}")

print("\n--- BY CONFIDENCE TIER ---")
for conf in ['HIGH', 'MEDIUM']:
    c_bets = bets_df[bets_df['confidence'] == conf]
    if len(c_bets) > 0:
        print(f"  {conf}: n={len(c_bets)} WR={c_bets['won'].mean():.1%} ROI={c_bets['roi'].mean():+.1%}")

# ============================================================
# OUT-OF-SAMPLE: 2024-25 ONLY
# ============================================================
print("\n--- 2024-25 OUT-OF-SAMPLE ---")
oos = bets_df[bets_df['Season'] == '2425']
print(f"Total bets: {len(oos)}")
print(f"WR: {oos['won'].mean():.1%}")
print(f"ROI: {oos['roi'].mean():+.1%}")

print("\nBreakdown by rule (2024-25):")
for rule in oos['rule'].unique():
    r_bets = oos[oos['rule'] == rule]
    print(f"  {rule}: n={len(r_bets)} WR={r_bets['won'].mean():.1%} ROI={r_bets['roi'].mean():+.1%}")

print("\nBreakdown by confidence (2024-25):")
for conf in ['HIGH', 'MEDIUM']:
    c_bets = oos[oos['confidence'] == conf]
    if len(c_bets) > 0:
        print(f"  {conf}: n={len(c_bets)} WR={c_bets['won'].mean():.1%} ROI={c_bets['roi'].mean():+.1%}")

# ============================================================
# COMBINATION ANALYSIS: Do rules overlap or add alpha?
# ============================================================
print("\n--- COMBINATION ANALYSIS ---")
print("Testing: HIGH confidence only vs HIGH+MEDIUM")

high_bets = bets_df[bets_df['confidence'] == 'HIGH']
print(f"\nHIGH ONLY: n={len(high_bets)} WR={high_bets['won'].mean():.1%} ROI={high_bets['roi'].mean():+.1%}")

med_bets = bets_df[bets_df['confidence'] == 'MEDIUM']
print(f"MEDIUM ONLY: n={len(med_bets)} WR={med_bets['won'].mean():.1%} ROI={med_bets['roi'].mean():+.1%}")

# Can we stack rules for even higher conviction?
# AH -0.25 AND cold home form gap
stacked = df_feat[
    (df_feat['real_ah_line'] == -0.25) & 
    (df_feat['h_form'] < 1.2) & 
    (df_feat['a_form'] > 1.5)
]
if len(stacked) > 30:
    wr = (stacked['real_ah_result'] == 1.0).mean()
    print(f"\nSTACKED AH -0.25 + cold home + hot away: n={len(stacked)} WR={wr:.1%}")

stacked2 = df_feat[
    (df_feat['real_ah_line'] == 0.25) & 
    (df_feat['a_form'] > 1.8)
]
if len(stacked2) > 30:
    wr = (stacked2['real_ah_result'] == 0.0).mean()
    print(f"STACKED AH +0.25 + strong away form: n={len(stacked2)} WR={wr:.1%}")

# ============================================================
# WHAT % OF UPCOMING GAMES CAN WE BET?
# ============================================================
print("\n--- COVERAGE ANALYSIS ---")
all_games = len(df_feat)
betted = len(bets_df)
print(f"Total games: {all_games}")
print(f"Games with a bet: {betted} ({betted/all_games*100:.1f}%)")
print(f"HIGH confidence bets: {len(high_bets)} ({len(high_bets)/all_games*100:.1f}%)")
print(f"MEDIUM confidence bets: {len(med_bets)} ({len(med_bets)/all_games*100:.1f}%)")

print("\nBet breakdown by AH line:")
for line in sorted(bets_df['real_ah_line'].unique()):
    n = len(bets_df[bets_df['real_ah_line'] == line])
    side = bets_df[bets_df['real_ah_line'] == line]['bet_side'].mode()[0]
    wr = bets_df[bets_df['real_ah_line'] == line]['won'].mean()
    print(f"  AH {line:+.2f}: n={n} ({side}) WR={wr:.1%}")

print("\nDone!")

# Save bet results
bets_df.to_csv('research/unified_picker_results.csv', index=False)
print("Saved to research/unified_picker_results.csv")
