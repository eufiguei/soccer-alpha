#!/usr/bin/env python3
"""
SOCCER ALPHA V3 — Market Signal Features Pipeline
Adds:
  1. AH line movement (B365 vs market average)
  2. AH odds gap (Max - Min spread)
  3. Season position proxies (wins - losses)
  4. Match week within season
  5. Closing vs opening line shift
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

BASE = Path('/root/.openclaw/workspace/projects/soccer-alpha')

print("=" * 70)
print("SOCCER ALPHA V3 — Market Signal + Season Context Features")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────────
df_raw = pd.read_parquet(BASE / 'data/real_ah_bettable.parquet')
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw = df_raw.sort_values('Date').reset_index(drop=True)
print(f"\nLoaded {len(df_raw)} matches: {df_raw['Date'].min().date()} → {df_raw['Date'].max().date()}")

# ── TASK 1: AH LINE MOVEMENT ANALYSIS ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TASK 1: AH LINE MOVEMENT ANALYSIS")
print("=" * 50)

# Check available AH odds columns
ah_cols = [c for c in df_raw.columns if 'AH' in c.upper()]
print(f"AH columns: {ah_cols}")

# Build line movement features
df = df_raw.copy()

# Opening line movement: B365 vs market average (opening)
if 'B365AHH' in df.columns and 'AvgAHH' in df.columns:
    df['ah_line_movement_open'] = df['B365AHH'] - df['AvgAHH']
    print(f"\nOpening AH movement stats:")
    print(df['ah_line_movement_open'].describe())

# Closing line movement: B365C vs AvgC
if 'B365CAHH' in df.columns and 'AvgCAHH' in df.columns:
    df['ah_line_movement_close'] = df['B365CAHH'] - df['AvgCAHH']
    print(f"\nClosing AH movement stats:")
    print(df['ah_line_movement_close'].describe())

# Opening to closing line shift
if 'AHh' in df.columns and 'AHCh' in df.columns:
    df['ah_line_shift'] = df['AHCh'] - df['AHh']  # positive = line moved toward home
    print(f"\nLine shift (open→close) stats:")
    print(df['ah_line_shift'].describe())

# Odds spread (sharp money indicator — tight spread = sharp market)
if 'MaxAHH' in df.columns and 'AvgAHH' in df.columns:
    df['ah_odds_spread_home'] = df['MaxAHH'] - df['AvgAHH']
    df['ah_odds_spread_away'] = df['MaxAHA'] - df['AvgAHA'] if 'MaxAHA' in df.columns else np.nan

# Closing odds spread
if 'MaxCAHH' in df.columns and 'AvgCAHH' in df.columns:
    df['ah_close_spread_home'] = df['MaxCAHH'] - df['AvgCAHH']

# ── ANALYZE PREDICTIVE POWER ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("PREDICTIVE POWER OF MARKET SIGNALS")
print("=" * 50)

ah_analysis = {}

# Bin line movement and check AH outcome
if 'ah_line_movement_open' in df.columns:
    df['ah_home_won'] = (df['real_ah_result'] == 1.0).astype(int)
    df['ah_movement_bucket'] = pd.cut(df['ah_line_movement_open'],
                                       bins=[-0.2, -0.05, -0.02, 0.02, 0.05, 0.2],
                                       labels=['strong_fade_home', 'mild_fade_home', 'neutral', 'mild_steam_home', 'strong_steam_home'])
    
    bucket_stats = df.groupby('ah_movement_bucket').agg(
        n_matches=('ah_home_won', 'count'),
        home_wr=('ah_home_won', 'mean'),
        avg_movement=('ah_line_movement_open', 'mean')
    ).round(4)
    
    print("\nAH Outcome by Market Movement Bucket:")
    print(bucket_stats.to_string())
    
    ah_analysis['by_movement_bucket'] = bucket_stats.reset_index().to_dict(orient='records')

# Closing line vs opening line shift
if 'ah_line_shift' in df.columns:
    df['line_shift_bucket'] = pd.cut(df['ah_line_shift'],
                                      bins=[-3, -0.25, -0.01, 0.01, 0.25, 3],
                                      labels=['big_move_away', 'small_move_away', 'no_change', 'small_move_home', 'big_move_home'])
    
    shift_stats = df.groupby('line_shift_bucket').agg(
        n_matches=('ah_home_won', 'count'),
        home_wr=('ah_home_won', 'mean')
    ).round(4)
    
    print("\nAH Outcome by Line Shift Bucket:")
    print(shift_stats.to_string())
    
    ah_analysis['by_line_shift'] = shift_stats.reset_index().to_dict(orient='records')

# ── TASK 2: SEASON POSITION PROXY ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TASK 2: SEASON POSITION PROXY (CUMULATIVE ROLLING)")
print("=" * 50)

def build_season_position_features(df):
    """
    For each team at each match, compute cumulative season stats UP TO (not including) that match.
    Uses all appearances (home or away) for rolling cumulative.
    """
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Build per-match performance records for each team
    all_records = []
    for _, row in df.iterrows():
        # Home team record
        all_records.append({
            'date': row['Date'],
            'season': row['Season'],
            'team': row['HomeTeam'],
            'match_idx': _,
            'gf': row['FTHG'],
            'ga': row['FTAG'],
            'result': 1 if row['FTHG'] > row['FTAG'] else (0 if row['FTHG'] == row['FTAG'] else -1)
        })
        # Away team record
        all_records.append({
            'date': row['Date'],
            'season': row['Season'],
            'team': row['AwayTeam'],
            'match_idx': _,
            'gf': row['FTAG'],
            'ga': row['FTHG'],
            'result': 1 if row['FTAG'] > row['FTHG'] else (0 if row['FTHG'] == row['FTAG'] else -1)
        })
    
    records_df = pd.DataFrame(all_records).sort_values(['team', 'season', 'date'])
    
    # Cumulative stats per team per season (shifted to avoid leakage)
    records_df['cum_wins'] = records_df.groupby(['team', 'season'])['result'].transform(
        lambda x: (x == 1).cumsum().shift(1).fillna(0)
    )
    records_df['cum_losses'] = records_df.groupby(['team', 'season'])['result'].transform(
        lambda x: (x == -1).cumsum().shift(1).fillna(0)
    )
    records_df['cum_draws'] = records_df.groupby(['team', 'season'])['result'].transform(
        lambda x: (x == 0).cumsum().shift(1).fillna(0)
    )
    records_df['cum_gf'] = records_df.groupby(['team', 'season'])['gf'].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    records_df['cum_ga'] = records_df.groupby(['team', 'season'])['ga'].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    records_df['season_position_proxy'] = records_df['cum_wins'] - records_df['cum_losses']
    records_df['season_points'] = records_df['cum_wins'] * 3 + records_df['cum_draws']
    records_df['match_num_in_season'] = records_df.groupby(['team', 'season']).cumcount()
    
    # Create lookup: (match_idx, team) -> stats
    lookup = records_df.set_index(['match_idx', 'team'])
    
    home_pos, away_pos = [], []
    home_pts, away_pts = [], []
    home_gf_season, away_gf_season = [], []
    home_ga_season, away_ga_season = [], []
    match_week = []
    
    for idx, row in df.iterrows():
        h_key = (idx, row['HomeTeam'])
        a_key = (idx, row['AwayTeam'])
        
        if h_key in lookup.index:
            h = lookup.loc[h_key]
            home_pos.append(h['season_position_proxy'])
            home_pts.append(h['season_points'])
            home_gf_season.append(h['cum_gf'])
            home_ga_season.append(h['cum_ga'])
            match_week.append(h['match_num_in_season'])
        else:
            home_pos.append(0)
            home_pts.append(0)
            home_gf_season.append(0)
            home_ga_season.append(0)
            match_week.append(0)
        
        if a_key in lookup.index:
            a = lookup.loc[a_key]
            away_pos.append(a['season_position_proxy'])
            away_pts.append(a['season_points'])
            away_gf_season.append(a['cum_gf'])
            away_ga_season.append(a['cum_ga'])
        else:
            away_pos.append(0)
            away_pts.append(0)
            away_gf_season.append(0)
            away_ga_season.append(0)
    
    df['home_season_position_proxy'] = home_pos
    df['away_season_position_proxy'] = away_pos
    df['home_season_points'] = home_pts
    df['away_season_points'] = away_pts
    df['home_gf_season'] = home_gf_season
    df['away_gf_season'] = away_gf_season
    df['home_ga_season'] = home_ga_season
    df['away_ga_season'] = away_ga_season
    df['match_week'] = match_week
    df['season_position_diff'] = df['home_season_position_proxy'] - df['away_season_position_proxy']
    df['season_points_diff'] = df['home_season_points'] - df['away_season_points']
    
    return df

print("Building season position features (this may take 30s)...")
df = build_season_position_features(df)
print(f"Season features built. Match week range: {df['match_week'].min()} - {df['match_week'].max()}")
print(f"Position proxy range: {df['home_season_position_proxy'].min()} - {df['home_season_position_proxy'].max()}")

# ── ELO FEATURES (from V2) ─────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("BUILDING ELO FEATURES")
print("=" * 50)

def build_elo_features(df):
    df = df.sort_values('Date').reset_index(drop=True)
    elo = {}
    home_elo_before, away_elo_before = [], []
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        h_elo = elo.get(home, 1500)
        a_elo = elo.get(away, 1500)
        home_elo_before.append(h_elo)
        away_elo_before.append(a_elo)
        
        hg, ag = row['FTHG'], row['FTAG']
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        actual_h = 1 if hg > ag else (0.5 if hg == ag else 0)
        k = 32
        elo[home] = h_elo + k * (actual_h - exp_h)
        elo[away] = a_elo + k * ((1 - actual_h) - (1 - exp_h))
    
    df['home_elo'] = home_elo_before
    df['away_elo'] = away_elo_before
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    return df

def build_rolling_features(df):
    df = df.sort_values('Date').reset_index(drop=True)
    
    for col in ['home_gf_6', 'home_ga_6', 'home_form_pts', 'home_sot_6',
                'away_gf_6', 'away_ga_6', 'away_form_pts', 'away_sot_6']:
        df[col] = np.nan
    
    for team in df['HomeTeam'].unique():
        home_idx = df.index[df['HomeTeam'] == team].tolist()
        away_idx = df.index[df['AwayTeam'] == team].tolist()
        all_idx = sorted(home_idx + away_idx)
        
        if len(all_idx) < 2:
            continue
        
        gf_series = pd.Series(index=all_idx, dtype=float)
        ga_series = pd.Series(index=all_idx, dtype=float)
        pts_series = pd.Series(index=all_idx, dtype=float)
        sot_series = pd.Series(index=all_idx, dtype=float)
        
        for i in all_idx:
            row = df.loc[i]
            if row['HomeTeam'] == team:
                gf_series[i] = row['FTHG']
                ga_series[i] = row['FTAG']
                pts_series[i] = 3 if row['FTHG'] > row['FTAG'] else (1 if row['FTHG'] == row['FTAG'] else 0)
                sot_series[i] = row.get('HST', np.nan)
            else:
                gf_series[i] = row['FTAG']
                ga_series[i] = row['FTHG']
                pts_series[i] = 3 if row['FTAG'] > row['FTHG'] else (1 if row['FTHG'] == row['FTAG'] else 0)
                sot_series[i] = row.get('AST', np.nan)
        
        gf_roll = gf_series.shift(1).rolling(6, min_periods=3).mean()
        ga_roll = ga_series.shift(1).rolling(6, min_periods=3).mean()
        pts_roll = pts_series.shift(1).rolling(6, min_periods=3).mean()
        sot_roll = sot_series.shift(1).rolling(6, min_periods=3).mean()
        
        for i in home_idx:
            if i in gf_roll.index:
                df.at[i, 'home_gf_6'] = gf_roll[i]
                df.at[i, 'home_ga_6'] = ga_roll[i]
                df.at[i, 'home_form_pts'] = pts_roll[i]
                df.at[i, 'home_sot_6'] = sot_roll[i]
        
        for i in away_idx:
            if i in gf_roll.index:
                df.at[i, 'away_gf_6'] = gf_roll[i]
                df.at[i, 'away_ga_6'] = ga_roll[i]
                df.at[i, 'away_form_pts'] = pts_roll[i]
                df.at[i, 'away_sot_6'] = sot_roll[i]
    
    return df

print("Building Elo features...")
df = build_elo_features(df)
print("Building rolling features...")
df = build_rolling_features(df)
print("All features built!")

# ── TASK 3: V3 MODEL WITH WALK-FORWARD VALIDATION ─────────────────────────────
print("\n" + "=" * 50)
print("TASK 3: V3 WALK-FORWARD VALIDATION")
print("=" * 50)

V3_FEATURES = [
    # Market signals (new)
    'ah_line_movement_open', 'ah_line_movement_close', 'ah_line_shift',
    'ah_odds_spread_home', 'ah_close_spread_home',
    # Season context (new)
    'home_season_position_proxy', 'away_season_position_proxy',
    'season_position_diff', 'season_points_diff', 'match_week',
    # Elo (from V2)
    'elo_diff',
    # Rolling form (from V2)
    'home_gf_6', 'home_ga_6', 'home_form_pts',
    'away_gf_6', 'away_ga_6', 'away_form_pts',
    # Market odds
    'home_implied', 'away_implied',
    # AH context
    'real_ah_line',
]

# Only include features that exist
V3_FEATURES = [f for f in V3_FEATURES if f in df.columns]
print(f"Features used: {V3_FEATURES}")

TARGET = 'actual_margin'
df['actual_margin'] = df['FTHG'] - df['FTAG']

FOLDS = [
    ('2019-08-01', '2022-05-31', '2022-06-01', '2023-05-31', '2022-2023'),
    ('2019-08-01', '2023-05-31', '2023-06-01', '2024-05-31', '2023-2024'),
    ('2019-08-01', '2024-05-31', '2024-06-01', '2025-05-31', '2024-2025'),
]

THRESHOLD = 0.5  # Confidence threshold
all_fold_results = []
detailed_results = []

def compute_ah_bet_result(predicted_margin, ah_line, actual_margin, threshold):
    """
    Returns (should_bet, side, won) where won=1.0 home won, 0.0 away won, 0.5 push
    """
    calibrated = predicted_margin
    adj_margin = calibrated - ah_line
    
    if abs(adj_margin) < threshold:
        return False, None, None
    
    side = 'home' if adj_margin > 0 else 'away'
    
    # Actual AH result
    actual_adj = actual_margin - ah_line
    if actual_adj > 0:
        won = 1.0
    elif actual_adj < 0:
        won = 0.0
    else:
        won = 0.5
    
    if side == 'away':
        won = 1.0 - won  # flip for away bet
    
    return True, side, won

for train_start, train_end, test_start, test_end, fold_name in FOLDS:
    train_mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
    test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Drop rows with too many NaN features
    feat_subset = [f for f in V3_FEATURES if f in train_df.columns]
    train_clean = train_df.dropna(subset=feat_subset, how='any')
    test_clean = test_df.dropna(subset=feat_subset, how='any')
    
    if len(train_clean) < 100 or len(test_clean) < 10:
        print(f"\n{fold_name}: Insufficient data — train={len(train_clean)}, test={len(test_clean)}")
        continue
    
    X_train = train_clean[feat_subset].values
    y_train = train_clean[TARGET].values
    X_test = test_clean[feat_subset].values
    y_test = test_clean[TARGET].values
    
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    # AH betting simulation
    bets = []
    for i, (_, row) in enumerate(test_clean.iterrows()):
        should_bet, side, won = compute_ah_bet_result(
            preds[i], row['real_ah_line'], row['actual_margin'], THRESHOLD
        )
        if should_bet:
            bets.append({
                'Date': row['Date'],
                'HomeTeam': row['HomeTeam'],
                'AwayTeam': row['AwayTeam'],
                'fold': fold_name,
                'predicted_margin': preds[i],
                'actual_margin': row['actual_margin'],
                'ah_line': row['real_ah_line'],
                'real_ah_result': row['real_ah_result'],
                'side': side,
                'won': won,
                'ah_line_movement': row.get('ah_line_movement_open', np.nan),
                'ah_line_shift': row.get('ah_line_shift', np.nan),
                'home_season_pos': row.get('home_season_position_proxy', np.nan),
                'away_season_pos': row.get('away_season_position_proxy', np.nan),
            })
            detailed_results.append(bets[-1])
    
    bets_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    n_bets = len(bets_df)
    
    if n_bets > 0:
        wr = bets_df['won'].mean()
        wins = (bets_df['won'] == 1.0).sum()
        pushes = (bets_df['won'] == 0.5).sum()
    else:
        wr, wins, pushes = 0, 0, 0
    
    result = {
        'fold': fold_name,
        'train_n': len(train_clean),
        'test_n': len(test_clean),
        'n_bets': n_bets,
        'mae': round(mae, 4),
        'win_rate': round(wr, 4),
        'wins': wins,
        'pushes': pushes,
    }
    all_fold_results.append(result)
    
    print(f"\n{fold_name}: train={len(train_clean)}, test={len(test_clean)}, bets={n_bets}")
    print(f"  MAE={mae:.3f}, WR={wr:.3%}")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("SAVING V3 ARTIFACTS")
print("=" * 50)

# Save detailed bet results
detailed_df = pd.DataFrame(detailed_results)
if len(detailed_df) > 0:
    detailed_df.to_csv(BASE / 'backtests/v3_validation.csv', index=False)
    print(f"Saved {len(detailed_df)} bets to backtests/v3_validation.csv")

# Save AH movement analysis
with open(BASE / 'data/ah_movement_analysis.json', 'w') as f:
    json.dump(ah_analysis, f, indent=2, default=str)
print("Saved data/ah_movement_analysis.json")

# Train final model on all data up to 2025-05
print("\nTraining final V3 model on all data...")
feat_subset = [f for f in V3_FEATURES if f in df.columns]
df_final = df.dropna(subset=feat_subset, how='any')
X_all = df_final[feat_subset].values
y_all = df_final['actual_margin'].values

final_model = LGBMRegressor(
    n_estimators=200, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
final_model.fit(X_all, y_all)

with open(BASE / 'models/v3_model.pkl', 'wb') as f:
    pickle.dump({'model': final_model, 'features': feat_subset}, f)
print("Saved models/v3_model.pkl")

# Feature importance
importances = pd.DataFrame({
    'feature': feat_subset,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
importances.to_csv(BASE / 'models/importance_v3.csv', index=False)
print("\nTop 10 feature importances:")
print(importances.head(10).to_string(index=False))

# Print fold summary
print("\n" + "=" * 50)
print("V3 FOLD SUMMARY")
print("=" * 50)
print(f"{'Fold':<15} {'Train N':>8} {'Test N':>8} {'Bets':>6} {'MAE':>6} {'WR':>8}")
print("-" * 55)
for r in all_fold_results:
    print(f"{r['fold']:<15} {r['train_n']:>8} {r['test_n']:>8} {r['n_bets']:>6} {r['mae']:>6.3f} {r['win_rate']:>8.1%}")

# Return fold results for use by other scripts
print("\nV3 DONE — fold_results:", json.dumps(all_fold_results, indent=2))
