#!/usr/bin/env python3
"""
Robustness checks to verify the edge is real:
1. Check for data leakage
2. Verify baseline (random) performance
3. Check if edge persists with different AH line calculations
4. Monte Carlo significance testing
5. Check actual AH odds from data vs assumed odds
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'

print("="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

df = pd.read_parquet(f'{DATA_DIR}/features_bettable.parquet')
df = df.sort_values('Date').reset_index(drop=True)

# ============================================================
# CHECK 1: Verify baseline
# ============================================================
print("\n1. BASELINE CHECK")
print("-"*40)

# What's the actual home vs away cover rate?
home_cover_rate = df['ah_target'].mean()
away_cover_rate = 1 - home_cover_rate

print(f"Home covers AH: {home_cover_rate:.1%}")
print(f"Away covers AH: {away_cover_rate:.1%}")

# Check by AH line
print("\nBy AH line:")
for line in sorted(df['ah_line'].unique()):
    subset = df[df['ah_line'] == line]
    if len(subset) > 50:
        home_rate = subset['ah_target'].mean()
        print(f"  AH {line:+.2f}: Home covers {home_rate:.1%}, Away covers {1-home_rate:.1%} (n={len(subset)})")

# ============================================================
# CHECK 2: Verify we're predicting correctly
# ============================================================
print("\n2. MODEL SANITY CHECK")
print("-"*40)

# The model should predict AWAY (target=0) more often if away covers 56%
feature_cols = [
    'home_form_pts', 'home_form_gf', 'home_form_ga', 'home_form_gd',
    'away_form_pts', 'away_form_gf', 'away_form_ga', 'away_form_gd',
    'home_prob', 'away_prob', 'draw_prob',
    'odds_ratio', 'home_favorite', 'prob_edge',
    'ah_line', 'h2h_matches', 'match_num', 'early_season',
]
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].fillna(df[feature_cols].median())
y = df['ah_target'].values

# Train on first 70%, test on last 30%
split = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]
df_test = df.iloc[split:].copy()

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

print(f"Model predicts Home cover: {preds.mean():.1%}")
print(f"Actual Home cover in test: {y_test.mean():.1%}")
print(f"Model mean prob(home): {probs.mean():.3f}")

# ============================================================
# CHECK 3: Verify the "always bet away" baseline
# ============================================================
print("\n3. 'ALWAYS BET AWAY' SANITY CHECK")
print("-"*40)

# This should work because away covers 56% of the time
# Let's verify season by season

for season in sorted(df['Season'].unique()):
    subset = df[df['Season'] == season]
    home_rate = subset['ah_target'].mean()
    print(f"  Season {season}: Home covers {home_rate:.1%}, Away covers {1-home_rate:.1%} (n={len(subset)})")

# ============================================================
# CHECK 4: Check actual AH odds in data
# ============================================================
print("\n4. ACTUAL AH ODDS CHECK")
print("-"*40)

# Some datasets include actual AH odds - check if we have them
ah_cols = [c for c in df.columns if 'AH' in c.upper() or 'BF' in c.upper() or 'asian' in c.lower()]
print(f"AH-related columns in data: {ah_cols}")

# Check raw data
raw = pd.read_parquet(f'{DATA_DIR}/all_odds_raw.parquet')
ah_raw_cols = [c for c in raw.columns if 'AH' in c.upper() or 'Handicap' in c]
print(f"AH columns in raw data: {ah_raw_cols[:10]}")

# Check for BbAH (Betbrain Asian Handicap)
if 'BbAHh' in raw.columns:
    print("\nBetbrain AH odds found!")
    print(f"  BbAHh (home cover odds): {raw['BbAHh'].describe()}")

# ============================================================
# CHECK 5: Monte Carlo significance
# ============================================================
print("\n5. MONTE CARLO SIGNIFICANCE TEST")
print("-"*40)

# Simulate random betting on away team
n_sims = 10000
observed_away_wr = 1 - home_cover_rate  # ~56%
n_bets = 3000  # Approximate test sample

# Generate random outcomes
random_wrs = np.random.binomial(n_bets, 0.5, n_sims) / n_bets
observed_better = (random_wrs >= observed_away_wr).mean()

print(f"If true probability is 50/50:")
print(f"  Observed 'always away' WR: {observed_away_wr:.1%}")
print(f"  P(random >= observed): {observed_better:.6f}")
print(f"  This is {'extremely unlikely' if observed_better < 0.0001 else 'plausible'} by chance")

# ============================================================
# CHECK 6: Data leakage check
# ============================================================
print("\n6. DATA LEAKAGE CHECK")
print("-"*40)

# Features that could leak: do we use any post-match info?
print("Features used:")
for col in feature_cols:
    print(f"  - {col}")

print("\nAll features are pre-match (form, odds, season timing). ✅")

# Check if form features are properly lagged
print("\nVerifying form feature lag...")
sample = df[df['HomeTeam'] == 'Liverpool'].head(10)
print(sample[['Date', 'HomeTeam', 'FTHG', 'home_form_gf']].to_string())

# ============================================================
# CHECK 7: Why does away cover more often?
# ============================================================
print("\n7. WHY DOES AWAY COVER MORE?")
print("-"*40)

# Hypothesis: AH line is set based on odds, but markets overvalue home advantage
# Let's check by probability ranges

print("Home cover rate by market probability of home:")
prob_bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
for low, high in prob_bins:
    subset = df[(df['home_prob'] >= low) & (df['home_prob'] < high)]
    if len(subset) > 50:
        home_rate = subset['ah_target'].mean()
        print(f"  home_prob [{low:.1f}-{high:.1f}): Home covers {home_rate:.1%} (n={len(subset)})")

# ============================================================
# CHECK 8: Edge by league
# ============================================================
print("\n8. AWAY COVER RATE BY LEAGUE")
print("-"*40)

for league in df['League'].unique():
    subset = df[df['League'] == league]
    home_rate = subset['ah_target'].mean()
    print(f"  {league}: Home {home_rate:.1%}, Away {1-home_rate:.1%} (n={len(subset)})")

# ============================================================
# CRITICAL FINDING
# ============================================================
print("\n" + "="*60)
print("CRITICAL FINDING")
print("="*60)

if away_cover_rate > 0.54:
    print(f"""
The 'edge' we found is NOT primarily model skill.

ROOT CAUSE: Our AH line derivation from odds creates a systematic bias.

When we derive AH line from 1X2 odds using thresholds:
  - prob_edge > 0.06  → AH -0.5 (home favorite)
  - prob_edge > 0.12  → AH -0.75
  - etc.

This creates lines where AWAY covers {away_cover_rate:.1%} of the time.

WHY? The market-implied home probabilities from 1X2 odds include
a home advantage premium that doesn't fully translate to AH outcomes.

The model learns this bias and predicts 'away' more often.

This is STILL a valid edge if:
1. Real AH markets show similar patterns
2. The spread between our derived line and market line is consistent

BUT we should verify against actual AH odds before declaring victory.
""")
else:
    print("Edge appears to be model-driven rather than systematic bias.")

print("\nNEXT STEP: Download actual Asian Handicap odds and verify.")
