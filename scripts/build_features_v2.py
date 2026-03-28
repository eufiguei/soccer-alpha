#!/usr/bin/env python3
"""
Feature engineering for Asian Handicap edge detection - Optimized version
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'

print("Loading raw odds data...")
df = pd.read_parquet(f'{DATA_DIR}/all_odds_raw.parquet')

# Parse dates
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'FTHG', 'FTAG', 'HomeTeam', 'AwayTeam'])
df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
df = df.dropna(subset=['FTHG', 'FTAG'])
df = df.sort_values(['League', 'Date']).reset_index(drop=True)

print(f"Total matches after cleaning: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Matches by league:\n{df['League'].value_counts()}")

# ============================================================
# FEATURE 1: Form (last 5 matches for each team) - VECTORIZED
# ============================================================
print("\nComputing form features (vectorized)...")

# Create a universal match-level dataframe for all teams
matches = []
for idx, row in df.iterrows():
    # Home perspective
    matches.append({
        'orig_idx': idx,
        'team': row['HomeTeam'],
        'date': row['Date'],
        'league': row['League'],
        'gf': row['FTHG'],
        'ga': row['FTAG'],
        'is_home': True
    })
    # Away perspective
    matches.append({
        'orig_idx': idx,
        'team': row['AwayTeam'],
        'date': row['Date'],
        'league': row['League'],
        'gf': row['FTAG'],
        'ga': row['FTHG'],
        'is_home': False
    })

match_df = pd.DataFrame(matches)
match_df['pts'] = np.where(match_df['gf'] > match_df['ga'], 3,
                           np.where(match_df['gf'] == match_df['ga'], 1, 0))

# Sort and compute rolling
match_df = match_df.sort_values(['team', 'league', 'date'])

# Rolling form within team/league
match_df['form_pts'] = match_df.groupby(['team', 'league'])['pts'].transform(
    lambda x: x.rolling(5, min_periods=1).sum().shift(1))
match_df['form_gf'] = match_df.groupby(['team', 'league'])['gf'].transform(
    lambda x: x.rolling(5, min_periods=1).sum().shift(1))
match_df['form_ga'] = match_df.groupby(['team', 'league'])['ga'].transform(
    lambda x: x.rolling(5, min_periods=1).sum().shift(1))
match_df['form_gd'] = match_df['form_gf'] - match_df['form_ga']

# Split back to home/away
home_form = match_df[match_df['is_home']].set_index('orig_idx')[['form_pts', 'form_gf', 'form_ga', 'form_gd']]
home_form.columns = ['home_' + c for c in home_form.columns]
away_form = match_df[~match_df['is_home']].set_index('orig_idx')[['form_pts', 'form_gf', 'form_ga', 'form_gd']]
away_form.columns = ['away_' + c for c in away_form.columns]

df = df.join(home_form).join(away_form)
print(f"  Form computed for {len(df)} matches")

# ============================================================
# FEATURE 2: Market-derived features
# ============================================================
print("Computing market features...")

df['home_odds'] = pd.to_numeric(df['B365H'], errors='coerce')
df['draw_odds'] = pd.to_numeric(df['B365D'], errors='coerce')
df['away_odds'] = pd.to_numeric(df['B365A'], errors='coerce')

# Implied probabilities
df['home_implied'] = 1 / df['home_odds']
df['draw_implied'] = 1 / df['draw_odds']
df['away_implied'] = 1 / df['away_odds']

# Overround
df['overround'] = df['home_implied'] + df['draw_implied'] + df['away_implied']

# Normalized probabilities
df['home_prob'] = df['home_implied'] / df['overround']
df['draw_prob'] = df['draw_implied'] / df['overround']
df['away_prob'] = df['away_implied'] / df['overround']

# Derived
df['odds_ratio'] = df['home_odds'] / df['away_odds']
df['home_favorite'] = (df['home_prob'] > df['away_prob']).astype(int)
df['prob_edge'] = df['home_prob'] - df['away_prob']

# ============================================================
# FEATURE 3: Asian Handicap line derivation
# ============================================================
print("Computing AH lines and outcomes...")

def derive_ah_line(edge):
    """Derive AH line from probability edge"""
    if edge > 0.35: return -2.0
    elif edge > 0.25: return -1.5
    elif edge > 0.18: return -1.0
    elif edge > 0.12: return -0.75
    elif edge > 0.06: return -0.5
    elif edge > 0.02: return -0.25
    elif edge > -0.02: return 0.0
    elif edge > -0.06: return 0.25
    elif edge > -0.12: return 0.5
    elif edge > -0.18: return 0.75
    elif edge > -0.25: return 1.0
    elif edge > -0.35: return 1.5
    else: return 2.0

df['ah_line'] = df['prob_edge'].apply(derive_ah_line)

# AH result: 1=home covers, 0=away covers, 0.5=push
def ah_result(home_goals, away_goals, ah_line):
    adj = home_goals + ah_line - away_goals
    if adj > 0: return 1
    elif adj < 0: return 0
    else: return 0.5

df['ah_result'] = df.apply(lambda r: ah_result(r['FTHG'], r['FTAG'], r['ah_line']), axis=1)

print(f"  AH line distribution:\n{df['ah_line'].value_counts().sort_index()}")

# ============================================================
# FEATURE 4: Match stats
# ============================================================
print("Adding match stats...")

df['home_shots'] = pd.to_numeric(df.get('HS', np.nan), errors='coerce')
df['away_shots'] = pd.to_numeric(df.get('AS', np.nan), errors='coerce')
df['home_sot'] = pd.to_numeric(df.get('HST', np.nan), errors='coerce')
df['away_sot'] = pd.to_numeric(df.get('AST', np.nan), errors='coerce')
df['home_corners'] = pd.to_numeric(df.get('HC', np.nan), errors='coerce')
df['away_corners'] = pd.to_numeric(df.get('AC', np.nan), errors='coerce')

df['shots_ratio'] = df['home_shots'] / (df['home_shots'] + df['away_shots'] + 0.1)
df['sot_ratio'] = df['home_sot'] / (df['home_sot'] + df['away_sot'] + 0.1)

# ============================================================
# FEATURE 5: H2H (simplified - use existing season data)
# ============================================================
print("Computing H2H (simplified)...")

# Create matchup key
df['matchup'] = df.apply(lambda r: tuple(sorted([r['HomeTeam'], r['AwayTeam']])), axis=1)

# Count previous meetings between these teams (within dataset)
h2h_counts = df.groupby('matchup').cumcount()
df['h2h_matches'] = h2h_counts

# ============================================================
# FEATURE 6: Season timing
# ============================================================
print("Computing season timing...")

df['match_num'] = df.groupby(['League', 'Season']).cumcount() + 1
df['early_season'] = (df['match_num'] <= 50).astype(int)

# ============================================================
# FEATURE 7: Goal-based features
# ============================================================
print("Computing goal features...")

df['total_goals'] = df['FTHG'] + df['FTAG']
df['goal_diff'] = df['FTHG'] - df['FTAG']

# ============================================================
# CLEAN AND SAVE
# ============================================================
print("\nCleaning and saving features...")

feature_cols = [
    'Date', 'League', 'Season', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',
    # Form
    'home_form_pts', 'home_form_gf', 'home_form_ga', 'home_form_gd',
    'away_form_pts', 'away_form_gf', 'away_form_ga', 'away_form_gd',
    # Market
    'home_odds', 'draw_odds', 'away_odds',
    'home_prob', 'draw_prob', 'away_prob',
    'odds_ratio', 'home_favorite', 'overround', 'prob_edge',
    # AH
    'ah_line', 'ah_result',
    # Stats
    'shots_ratio', 'sot_ratio', 'home_shots', 'away_shots',
    # H2H
    'h2h_matches',
    # Timing
    'match_num', 'early_season',
    # Goals
    'total_goals', 'goal_diff'
]

df_final = df[[c for c in feature_cols if c in df.columns]].copy()

# Drop rows with missing critical features
df_final = df_final.dropna(subset=['home_prob', 'away_prob', 'ah_result', 'home_form_pts', 'away_form_pts'])

# Remove pushes for classification
df_bets = df_final[df_final['ah_result'] != 0.5].copy()
df_bets['ah_target'] = df_bets['ah_result'].astype(int)

print(f"\nFinal dataset: {len(df_final)} matches")
print(f"Bettable (no push): {len(df_bets)} matches")
print(f"\nAH result distribution:")
print(df_bets['ah_target'].value_counts())
print(f"\nHome covers rate: {df_bets['ah_target'].mean():.3f}")

# Save
df_final.to_parquet(f'{DATA_DIR}/features_all.parquet', index=False)
df_bets.to_parquet(f'{DATA_DIR}/features_bettable.parquet', index=False)

print(f"\nSaved to {DATA_DIR}/")
print("\nSample of features:")
print(df_bets[['Date', 'League', 'HomeTeam', 'AwayTeam', 'home_prob', 'ah_line', 'ah_target']].head(10))
