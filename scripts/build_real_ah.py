#!/usr/bin/env python3
"""
Use REAL Asian Handicap odds from the data instead of derived lines.
The data contains:
  - AHh: Asian Handicap line for home team
  - B365AHH: Bet365 AH home odds
  - B365AHA: Bet365 AH away odds
  - AvgAHH/AvgAHA: Average AH odds
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'

print("Loading raw data with AH odds...")
raw = pd.read_parquet(f'{DATA_DIR}/all_odds_raw.parquet')

print(f"Total rows: {len(raw)}")

# Key AH columns
ah_cols = ['AHh', 'B365AHH', 'B365AHA', 'AvgAHH', 'AvgAHA', 'FTHG', 'FTAG']
for col in ah_cols:
    if col in raw.columns:
        print(f"  {col}: {raw[col].notna().sum()} non-null ({raw[col].notna().mean()*100:.1f}%)")

# Clean and prepare
df = raw.copy()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'FTHG', 'FTAG', 'HomeTeam', 'AwayTeam'])
df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')

# Get actual AH line
df['real_ah_line'] = pd.to_numeric(df['AHh'], errors='coerce')
df['ah_home_odds'] = pd.to_numeric(df['B365AHH'], errors='coerce')
df['ah_away_odds'] = pd.to_numeric(df['B365AHA'], errors='coerce')

# If B365 AH not available, use average
df['ah_home_odds'] = df['ah_home_odds'].fillna(pd.to_numeric(df['AvgAHH'], errors='coerce'))
df['ah_away_odds'] = df['ah_away_odds'].fillna(pd.to_numeric(df['AvgAHA'], errors='coerce'))

# Filter to rows with real AH lines
df_ah = df.dropna(subset=['real_ah_line', 'ah_home_odds', 'ah_away_odds'])
print(f"\nMatches with real AH data: {len(df_ah)}")

if len(df_ah) < 1000:
    print("\nInsufficient AH data. Checking alternative approach...")
    
    # Many rows might have AH handicap only in recent seasons
    print("\nAH data by season:")
    df['Season'] = df['Season'].astype(str)
    ah_by_season = df.groupby('Season')['real_ah_line'].apply(lambda x: x.notna().sum())
    print(ah_by_season)
    
    # Check if we have enough without AH handicap odds but with the line
    df_line_only = df.dropna(subset=['real_ah_line'])
    print(f"\nMatches with AH line (may not have odds): {len(df_line_only)}")
else:
    df = df_ah

# ============================================================
# Compute AH result with REAL line
# ============================================================
print("\nComputing AH results with real market lines...")

def real_ah_result(home_goals, away_goals, ah_line):
    """
    Real AH result calculation.
    AH line is typically expressed as home handicap (e.g., -0.5 means home gives 0.5 goals)
    """
    adj = home_goals + ah_line - away_goals
    
    # Handle quarter lines (e.g., -0.75 = half on -0.5, half on -1.0)
    if ah_line % 0.5 != 0:  # Quarter line
        line1 = np.floor(ah_line * 2) / 2
        line2 = np.ceil(ah_line * 2) / 2
        
        adj1 = home_goals + line1 - away_goals
        adj2 = home_goals + line2 - away_goals
        
        result1 = 1 if adj1 > 0 else (0 if adj1 < 0 else 0.5)
        result2 = 1 if adj2 > 0 else (0 if adj2 < 0 else 0.5)
        
        return (result1 + result2) / 2
    else:
        if adj > 0: return 1
        elif adj < 0: return 0
        else: return 0.5

df_ah['real_ah_result'] = df_ah.apply(
    lambda r: real_ah_result(r['FTHG'], r['FTAG'], r['real_ah_line']), axis=1
)

# ============================================================
# Analyze REAL AH market
# ============================================================
print("\n" + "="*60)
print("REAL ASIAN HANDICAP MARKET ANALYSIS")
print("="*60)

# Overall home vs away cover rate
home_cover = df_ah['real_ah_result'].mean()
print(f"\nOverall home cover rate: {home_cover:.1%}")
print(f"Overall away cover rate: {1-home_cover:.1%}")

# By AH line (rounded to nearest 0.5 for grouping)
df_ah['ah_line_group'] = (df_ah['real_ah_line'] * 2).round() / 2

print("\nBy AH line:")
line_stats = df_ah.groupby('ah_line_group').agg({
    'real_ah_result': ['mean', 'count']
}).reset_index()
line_stats.columns = ['ah_line', 'home_cover_rate', 'count']
for _, row in line_stats.iterrows():
    if row['count'] >= 50:
        print(f"  AH {row['ah_line']:+.2f}: Home covers {row['home_cover_rate']:.1%} (n={int(row['count'])})")

# ============================================================
# Check if real AH market is efficient (50/50)
# ============================================================
print("\n" + "="*60)
print("MARKET EFFICIENCY CHECK")
print("="*60)

# Remove pushes
df_bets = df_ah[df_ah['real_ah_result'].isin([0, 1])].copy()
print(f"\nBettable matches (no push): {len(df_bets)}")

home_wr = df_bets['real_ah_result'].mean()
print(f"Home covers: {home_wr:.1%}")
print(f"Away covers: {1-home_wr:.1%}")

# Statistical test
from scipy import stats
n = len(df_bets)
z = (home_wr - 0.5) / np.sqrt(0.5 * 0.5 / n)
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"\nTest vs 50%: z={z:.2f}, p={p:.4f}")
if p < 0.05:
    print("⚠️ Market is NOT perfectly efficient at 50/50")
else:
    print("✅ Market appears efficiently priced at ~50/50")

# ============================================================
# Check implied probabilities from odds
# ============================================================
print("\n" + "="*60)
print("IMPLIED PROBABILITIES FROM AH ODDS")
print("="*60)

df_bets['home_implied'] = 1 / df_bets['ah_home_odds']
df_bets['away_implied'] = 1 / df_bets['ah_away_odds']
df_bets['overround'] = df_bets['home_implied'] + df_bets['away_implied']

print(f"\nAverage overround: {df_bets['overround'].mean():.3f}")
print(f"Average home implied prob: {df_bets['home_implied'].mean():.3f}")
print(f"Average away implied prob: {df_bets['away_implied'].mean():.3f}")

# Breakeven
breakeven = 1 / (df_bets['overround'] / 2)  # Approx fair odds after vig
print(f"Implied breakeven: ~{1/breakeven.mean():.1%}")

# ============================================================
# SAVE
# ============================================================
df_bets.to_parquet(f'{DATA_DIR}/real_ah_bettable.parquet', index=False)
print(f"\nSaved {len(df_bets)} matches to real_ah_bettable.parquet")

# Now check: can we find edge in REAL AH market?
print("\n" + "="*60)
print("EDGE HUNTING IN REAL AH MARKET")
print("="*60)

# Group by implied probability bins
df_bets['home_imp_bin'] = pd.cut(df_bets['home_implied'], 
                                  bins=[0, 0.45, 0.48, 0.50, 0.52, 0.55, 1.0],
                                  labels=['<45%', '45-48%', '48-50%', '50-52%', '52-55%', '>55%'])

print("\nActual home cover rate by implied probability:")
for bin_label in df_bets['home_imp_bin'].dropna().unique():
    subset = df_bets[df_bets['home_imp_bin'] == bin_label]
    actual = subset['real_ah_result'].mean()
    implied = subset['home_implied'].mean()
    edge = actual - implied
    n = len(subset)
    print(f"  {bin_label}: Implied {implied:.1%}, Actual {actual:.1%}, Edge {edge:+.1%} (n={n})")
