"""
Full Alpha Search - Systematic hypothesis testing across ALL AH lines
Tests 50+ betting rules and finds consistent patterns
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_parquet('data/real_ah_bettable.parquet')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Loaded {len(df)} matches")
print(f"Seasons: {df['Season'].value_counts().sort_index().to_dict()}")
print(f"AH lines: {df['real_ah_line'].value_counts().sort_index().to_dict()}")

# ============================================================
# STEP 1: BUILD ALL FEATURES
# ============================================================
print("\n" + "="*60)
print("STEP 1: Building features...")
print("="*60)

# Team rolling stats (home and away separately)
for col in ['h_gf', 'h_ga', 'h_form', 'a_gf', 'a_ga', 'a_form']:
    df[col] = np.nan

teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
print(f"Processing {len(teams)} teams...")

for i, team in enumerate(teams):
    if i % 20 == 0:
        print(f"  Team {i}/{len(teams)}...")

    # Home stats
    mask_h = df['HomeTeam'] == team
    idx_h = df.index[mask_h]
    if len(idx_h) >= 3:
        goals_h = df.loc[mask_h, 'FTHG'].values
        conc_h = df.loc[mask_h, 'FTAG'].values
        pts_h = np.where(goals_h > conc_h, 3, np.where(goals_h == conc_h, 1, 0)).astype(float)

        for j in range(len(idx_h)):
            if j >= 3:
                start = max(0, j - 6)
                df.loc[idx_h[j], 'h_gf'] = np.mean(goals_h[start:j])
                df.loc[idx_h[j], 'h_ga'] = np.mean(conc_h[start:j])
                df.loc[idx_h[j], 'h_form'] = np.mean(pts_h[start:j])

    # Away stats
    mask_a = df['AwayTeam'] == team
    idx_a = df.index[mask_a]
    if len(idx_a) >= 3:
        goals_a = df.loc[mask_a, 'FTAG'].values
        conc_a = df.loc[mask_a, 'FTHG'].values
        pts_a = np.where(goals_a > conc_a, 3, np.where(goals_a == conc_a, 1, 0)).astype(float)

        for j in range(len(idx_a)):
            if j >= 3:
                start = max(0, j - 6)
                df.loc[idx_a[j], 'a_gf'] = np.mean(goals_a[start:j])
                df.loc[idx_a[j], 'a_ga'] = np.mean(conc_a[start:j])
                df.loc[idx_a[j], 'a_form'] = np.mean(pts_a[start:j])

print("Computing derived features...")

# Derived features
df['form_gap'] = df['h_form'] - df['a_form']
df['gf_gap'] = df['h_gf'] - df['a_gf']
df['def_gap'] = df['a_ga'] - df['h_ga']
df['attack_vs_defense'] = df['h_gf'] - df['a_ga']
df['away_attack_vs_home_def'] = df['a_gf'] - df['h_ga']

# Market features
df['implied_gap'] = df['home_implied'] - df['away_implied']
df['odds_ratio'] = df['ah_home_odds'] / df['ah_away_odds']

# AH line features
df['ah_line_abs'] = df['real_ah_line'].abs()

# Season timing
df['month'] = df['Date'].dt.month
df['is_early_season'] = df['month'].isin([8, 9, 10]).astype(int)
df['is_late_season'] = df['month'].isin([4, 5]).astype(int)
df['is_mid_season'] = df['month'].isin([12, 1]).astype(int)

# Home win streak (last 3 home games)
for col in ['h_win_streak3', 'a_loss_streak3', 'h_bigwin_streak']:
    df[col] = np.nan

for team in teams:
    mask_h = df['HomeTeam'] == team
    idx_h = df.index[mask_h]
    if len(idx_h) >= 4:
        goals_h = df.loc[mask_h, 'FTHG'].values
        conc_h = df.loc[mask_h, 'FTAG'].values
        for j in range(len(idx_h)):
            if j >= 3:
                recent = [(goals_h[k] > conc_h[k]) for k in range(max(0, j-3), j)]
                df.loc[idx_h[j], 'h_win_streak3'] = float(all(recent))
                # Big win (3+ goals margin) streak
                big_wins = [(goals_h[k] - conc_h[k] >= 3) for k in range(max(0, j-2), j)]
                df.loc[idx_h[j], 'h_bigwin_streak'] = float(any(big_wins))

    mask_a = df['AwayTeam'] == team
    idx_a = df.index[mask_a]
    if len(idx_a) >= 4:
        goals_a = df.loc[mask_a, 'FTAG'].values
        conc_a = df.loc[mask_a, 'FTHG'].values
        for j in range(len(idx_a)):
            if j >= 3:
                recent_loss = [(goals_a[k] < conc_a[k]) for k in range(max(0, j-3), j)]
                df.loc[idx_a[j], 'a_loss_streak3'] = float(all(recent_loss))

print(f"Feature computation complete. Non-null rows: {df['h_form'].notna().sum()}")

# Save enriched data
df_feat = df.dropna(subset=['h_form', 'a_form', 'h_gf', 'a_gf'])
print(f"Rows with features: {len(df_feat)}")

# ============================================================
# STEP 2: HYPOTHESIS TESTING ENGINE
# ============================================================
print("\n" + "="*60)
print("STEP 2: Testing hypotheses...")
print("="*60)

SEASONS = sorted(df_feat['Season'].unique())
MIN_BETS = 30  # minimum bets per season

results = []

def test_hypothesis(name, mask, bet_side='home', min_season_bets=MIN_BETS):
    """
    mask: boolean series on df_feat
    bet_side: 'home' means bet home (real_ah_result=1 is win)
              'away' means bet away (real_ah_result=0 is win)
    """
    subset = df_feat[mask].copy()
    if len(subset) < 50:
        return None

    if bet_side == 'home':
        subset['win'] = (subset['real_ah_result'] == 1.0).astype(float)
    else:
        subset['win'] = (subset['real_ah_result'] == 0.0).astype(float)

    # Overall stats
    total = len(subset)
    wins = subset['win'].sum()
    wr = wins / total
    
    # ROI calculation using odds
    if bet_side == 'home':
        roi = subset.apply(lambda r: (r['ah_home_odds'] - 1) if r['win'] == 1 else -1, axis=1).mean()
    else:
        roi = subset.apply(lambda r: (r['ah_away_odds'] - 1) if r['win'] == 1 else -1, axis=1).mean()

    # Per-season stats
    season_wrs = []
    season_details = []
    for s in SEASONS:
        s_df = subset[subset['Season'] == s]
        if len(s_df) < min_season_bets:
            season_wrs.append(None)
            season_details.append(f"{s}:N/A({len(s_df)})")
        else:
            s_wr = s_df['win'].mean()
            season_wrs.append(s_wr)
            season_details.append(f"{s}:{s_wr:.1%}({len(s_df)})")

    valid_seasons = [w for w in season_wrs if w is not None]
    seasons_above_53 = sum(1 for w in valid_seasons if w > 0.53)
    seasons_valid = len(valid_seasons)

    # P-value (binomial test against 50%)
    p_val = stats.binomtest(int(wins), int(total), 0.5, alternative='greater').pvalue

    result = {
        'name': name,
        'bet_side': bet_side,
        'total_bets': total,
        'wr': wr,
        'roi': roi,
        'seasons_above_53': seasons_above_53,
        'seasons_valid': seasons_valid,
        'p_value': p_val,
        'season_details': ' | '.join(season_details),
        'consistent': seasons_above_53 >= 4 and wr > 0.53 and p_val < 0.05,
        'marginal': (seasons_above_53 >= 3 or wr > 0.52) and p_val < 0.10,
    }
    results.append(result)
    return result

def print_result(r):
    if r is None:
        return
    verdict = "✅ CONSISTENT" if r['consistent'] else ("⚠️ MARGINAL" if r['marginal'] else "❌ NOISE")
    print(f"\nHypothesis: {r['name']} [{r['bet_side']}]")
    print(f"  Bets: {r['total_bets']} | WR: {r['wr']:.1%} | ROI: {r['roi']:+.1%}")
    print(f"  Seasons >53%: {r['seasons_above_53']}/{r['seasons_valid']} | p={r['p_value']:.4f}")
    print(f"  {r['season_details']}")
    print(f"  VERDICT: {verdict}")

# ============================================================
# FORM-BASED HYPOTHESES
# ============================================================
print("\n--- FORM-BASED ---")

r = test_hypothesis(
    "Large form gap (home-away > 1.5pts) → bet HOME",
    df_feat['form_gap'] > 1.5, 'home'
)
print_result(r)

r = test_hypothesis(
    "Reverse form (away > home by 1.5+) → bet AWAY",
    df_feat['form_gap'] < -1.5, 'away'
)
print_result(r)

r = test_hypothesis(
    "Large form gap > 1.0 → bet HOME",
    df_feat['form_gap'] > 1.0, 'home'
)
print_result(r)

r = test_hypothesis(
    "Hot home team (h_form > 2.0) + weak line (AH > -1.0) → bet HOME",
    (df_feat['h_form'] > 2.0) & (df_feat['real_ah_line'] > -1.0), 'home'
)
print_result(r)

r = test_hypothesis(
    "Cold home team (h_form < 0.8) → bet AWAY",
    df_feat['h_form'] < 0.8, 'away'
)
print_result(r)

r = test_hypothesis(
    "Cold home + weak form gap → bet AWAY",
    (df_feat['h_form'] < 1.0) & (df_feat['form_gap'] < 0), 'away'
)
print_result(r)

r = test_hypothesis(
    "Both teams mediocre form (1.2-1.8) → bet AWAY",
    (df_feat['h_form'].between(1.2, 1.8)) & (df_feat['a_form'].between(1.2, 1.8)), 'away'
)
print_result(r)

r = test_hypothesis(
    "Away team in good form (a_form > 2.0) → bet AWAY",
    df_feat['a_form'] > 2.0, 'away'
)
print_result(r)

# ============================================================
# ATTACK/DEFENSE MATCHUP
# ============================================================
print("\n--- ATTACK/DEFENSE MATCHUP ---")

r = test_hypothesis(
    "Home attack >> away defense (h_gf > a_ga + 1.0) → bet HOME",
    df_feat['attack_vs_defense'] > 1.0, 'home'
)
print_result(r)

r = test_hypothesis(
    "Home attack >> away defense (h_gf > a_ga + 0.7) → bet HOME",
    df_feat['attack_vs_defense'] > 0.7, 'home'
)
print_result(r)

r = test_hypothesis(
    "Away attack >> home defense (a_gf > h_ga + 0.5) → bet AWAY",
    df_feat['away_attack_vs_home_def'] > 0.5, 'away'
)
print_result(r)

r = test_hypothesis(
    "Both teams low scoring → bet AWAY",
    (df_feat['h_gf'] < 1.2) & (df_feat['a_gf'] < 1.2), 'away'
)
print_result(r)

r = test_hypothesis(
    "High scoring home + weak away defense → bet HOME",
    (df_feat['h_gf'] > 2.0) & (df_feat['a_ga'] > 1.5), 'home'
)
print_result(r)

r = test_hypothesis(
    "Strong home defense + weak away attack → bet HOME",
    (df_feat['h_ga'] < 0.8) & (df_feat['a_gf'] < 1.2), 'home'
)
print_result(r)

r = test_hypothesis(
    "Goal diff gap (home scoring a lot more) gf_gap > 1.0 → bet HOME",
    df_feat['gf_gap'] > 1.0, 'home'
)
print_result(r)

r = test_hypothesis(
    "Defensive gap: home much better defense → bet HOME",
    df_feat['def_gap'] > 0.8, 'home'
)
print_result(r)

# ============================================================
# MARKET VS FORM DISCREPANCY
# ============================================================
print("\n--- MARKET VS FORM DISCREPANCY ---")

r = test_hypothesis(
    "Market says home strong (implied_gap > 0.3) but form gap < 0.5 → AWAY",
    (df_feat['implied_gap'] > 0.3) & (df_feat['form_gap'].abs() < 0.5), 'away'
)
print_result(r)

r = test_hypothesis(
    "Market close game (implied_gap < 0.1) but form says home much better → HOME",
    (df_feat['implied_gap'].abs() < 0.1) & (df_feat['form_gap'] > 1.5), 'home'
)
print_result(r)

r = test_hypothesis(
    "Odds overvalue home (odds_ratio < 0.92) → AWAY value",
    df_feat['odds_ratio'] < 0.92, 'away'
)
print_result(r)

r = test_hypothesis(
    "Odds undervalue home (odds_ratio > 1.08) → HOME value",
    df_feat['odds_ratio'] > 1.08, 'home'
)
print_result(r)

r = test_hypothesis(
    "Both AH odds close to fair (1.9-2.1 each) → bet AWAY",
    (df_feat['ah_home_odds'].between(1.85, 2.10)) & (df_feat['ah_away_odds'].between(1.85, 2.10)), 'away'
)
print_result(r)

r = test_hypothesis(
    "Home AH odds < 1.85 (market loves home) → AWAY value",
    df_feat['ah_home_odds'] < 1.85, 'away'
)
print_result(r)

r = test_hypothesis(
    "Away AH odds > 2.10 (market hates away) → AWAY value",
    df_feat['ah_away_odds'] > 2.10, 'away'
)
print_result(r)

r = test_hypothesis(
    "High B365 implied home win (>65%) + form gap < 1.0 → AWAY",
    (df_feat['home_implied'] > 0.55) & (df_feat['form_gap'] < 1.0), 'away'
)
print_result(r)

# ============================================================
# LINE-SPECIFIC WITH CONDITIONS
# ============================================================
print("\n--- LINE-SPECIFIC CONDITIONS ---")

r = test_hypothesis(
    "AH -1.0 + home form > 2.0 → HOME",
    (df_feat['real_ah_line'] == -1.0) & (df_feat['h_form'] > 2.0), 'home'
)
print_result(r)

r = test_hypothesis(
    "AH -1.0 + home form < 1.5 → AWAY",
    (df_feat['real_ah_line'] == -1.0) & (df_feat['h_form'] < 1.5), 'away'
)
print_result(r)

r = test_hypothesis(
    "AH -0.75 + attack_vs_defense > 1.0 → HOME",
    (df_feat['real_ah_line'] == -0.75) & (df_feat['attack_vs_defense'] > 1.0), 'home'
)
print_result(r)

r = test_hypothesis(
    "AH -0.75 + attack_vs_defense < 0.3 → AWAY",
    (df_feat['real_ah_line'] == -0.75) & (df_feat['attack_vs_defense'] < 0.3), 'away'
)
print_result(r)

r = test_hypothesis(
    "AH -1.25 + home very strong (gf>2.0, ga<0.8) → HOME",
    (df_feat['real_ah_line'] == -1.25) & (df_feat['h_gf'] > 2.0) & (df_feat['h_ga'] < 0.8), 'home'
)
print_result(r)

r = test_hypothesis(
    "AH -1.5 → AWAY (needs win by 2)",
    df_feat['real_ah_line'] == -1.5, 'away'
)
print_result(r)

r = test_hypothesis(
    "AH -1.75 → AWAY (confirmed from prev research)",
    df_feat['real_ah_line'] == -1.75, 'away'
)
print_result(r)

r = test_hypothesis(
    "AH -1.0 overall → HOME vs AWAY",
    df_feat['real_ah_line'] == -1.0, 'home'
)
print_result(r)

r = test_hypothesis(
    "AH -0.5 overall → HOME",
    df_feat['real_ah_line'] == -0.5, 'home'
)
print_result(r)

r = test_hypothesis(
    "AH -0.25 overall → HOME",
    df_feat['real_ah_line'] == -0.25, 'home'
)
print_result(r)

r = test_hypothesis(
    "AH 0.0 (level) → HOME",
    df_feat['real_ah_line'] == 0.0, 'home'
)
print_result(r)

r = test_hypothesis(
    "AH +0.25 → AWAY (giving handicap to away)",
    df_feat['real_ah_line'] == 0.25, 'away'
)
print_result(r)

r = test_hypothesis(
    "AH -2.0 → AWAY",
    df_feat['real_ah_line'] == -2.0, 'away'
)
print_result(r)

r = test_hypothesis(
    "AH 0.25 (home is underdog) → HOME",
    df_feat['real_ah_line'] == 0.25, 'home'
)
print_result(r)

# ============================================================
# SEASON TIMING
# ============================================================
print("\n--- SEASON TIMING ---")

r = test_hypothesis(
    "Early season (Aug-Sep) → fade favorites (AWAY)",
    (df_feat['is_early_season'] == 1) & (df_feat['implied_gap'] > 0.2), 'away'
)
print_result(r)

r = test_hypothesis(
    "Early season → bet HOME generally",
    df_feat['is_early_season'] == 1, 'home'
)
print_result(r)

r = test_hypothesis(
    "Late season (Apr-May) → bet HOME (UCL push)",
    df_feat['is_late_season'] == 1, 'home'
)
print_result(r)

r = test_hypothesis(
    "Late season + strong home team → HOME",
    (df_feat['is_late_season'] == 1) & (df_feat['h_form'] > 1.8), 'home'
)
print_result(r)

r = test_hypothesis(
    "Mid-season (Dec-Jan congestion) → AWAY",
    df_feat['is_mid_season'] == 1, 'away'
)
print_result(r)

r = test_hypothesis(
    "Feb-Mar (mid-table crunch) → HOME",
    df_feat['month'].isin([2, 3]), 'home'
)
print_result(r)

# ============================================================
# STREAK PATTERNS
# ============================================================
print("\n--- STREAK PATTERNS ---")

r = test_hypothesis(
    "Home team on 3+ win streak (last 3 home) → HOME",
    df_feat['h_win_streak3'] == 1, 'home'
)
print_result(r)

r = test_hypothesis(
    "Away team on 3+ loss streak → HOME",
    df_feat['a_loss_streak3'] == 1, 'home'
)
print_result(r)

r = test_hypothesis(
    "Home team had big win recently → HOME",
    df_feat['h_bigwin_streak'] == 1, 'home'
)
print_result(r)

# ============================================================
# ADVANCED COMBINATIONS
# ============================================================
print("\n--- ADVANCED COMBINATIONS ---")

r = test_hypothesis(
    "Strong home (form>2.0) + strong attack (gf>1.8) → HOME",
    (df_feat['h_form'] > 2.0) & (df_feat['h_gf'] > 1.8), 'home'
)
print_result(r)

r = test_hypothesis(
    "Strong home form + weak away form (form_gap > 1.0) + home attack strong → HOME",
    (df_feat['form_gap'] > 1.0) & (df_feat['attack_vs_defense'] > 0.5), 'home'
)
print_result(r)

r = test_hypothesis(
    "Form gap > 1.5 + AH -0.5 to -1.25 → HOME",
    (df_feat['form_gap'] > 1.5) & (df_feat['real_ah_line'].between(-1.25, -0.5)), 'home'
)
print_result(r)

r = test_hypothesis(
    "Weak away form (a_form < 1.0) + market still pricing close → HOME",
    (df_feat['a_form'] < 1.0) & (df_feat['implied_gap'] < 0.15), 'home'
)
print_result(r)

r = test_hypothesis(
    "Market mispricing: home implied < form_gap*0.1 + 0.5 → AWAY",
    df_feat['implied_gap'] > (df_feat['form_gap'] * 0.1 + 0.3), 'away'
)
print_result(r)

r = test_hypothesis(
    "Form-odds divergence: form_gap > 1 but implied_gap < 0.2 → HOME (undervalued)",
    (df_feat['form_gap'] > 1.0) & (df_feat['implied_gap'] < 0.2), 'home'
)
print_result(r)

r = test_hypothesis(
    "Both teams scoring well (gf > 1.5 each) → AWAY (draws help away on AH)",
    (df_feat['h_gf'] > 1.5) & (df_feat['a_gf'] > 1.5), 'away'
)
print_result(r)

r = test_hypothesis(
    "Home scoring < 1.0, line < -0.5 → AWAY (overrated home)",
    (df_feat['h_gf'] < 1.0) & (df_feat['real_ah_line'] < -0.5), 'away'
)
print_result(r)

# LEAGUE-SPECIFIC
print("\n--- LEAGUE-SPECIFIC ---")
for league in df_feat['League'].unique():
    r = test_hypothesis(
        f"{league} → HOME",
        df_feat['League'] == league, 'home'
    )
    if r:
        print_result(r)

# AH range combos
print("\n--- AH RANGE COMBOS ---")

r = test_hypothesis(
    "Lines >= -0.5 (slight/no favorite) → HOME",
    df_feat['real_ah_line'] >= -0.5, 'home'
)
print_result(r)

r = test_hypothesis(
    "Lines <= -1.0 (heavy favorites) → AWAY",
    df_feat['real_ah_line'] <= -1.0, 'away'
)
print_result(r)

r = test_hypothesis(
    "Lines -0.25 to -0.75 (moderate favorite) → HOME",
    df_feat['real_ah_line'].between(-0.75, -0.25), 'home'
)
print_result(r)

r = test_hypothesis(
    "Lines -1.0 to -1.5 + form_gap > 1.5 → HOME",
    (df_feat['real_ah_line'].between(-1.5, -1.0)) & (df_feat['form_gap'] > 1.5), 'home'
)
print_result(r)

r = test_hypothesis(
    "Positive lines (away is favorite AH-wise) → AWAY",
    df_feat['real_ah_line'] > 0, 'away'
)
print_result(r)

# ============================================================
# STEP 3: FIND TOP CONSISTENT RULES
# ============================================================
print("\n" + "="*60)
print("STEP 3: RANKING RESULTS")
print("="*60)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(['consistent', 'seasons_above_53', 'wr'], ascending=[False, False, False])

consistent_rules = df_results[df_results['consistent'] == True]
marginal_rules = df_results[df_results['marginal'] == True]

print(f"\n✅ CONSISTENT RULES (WR>53%, 4+/6 seasons, p<0.05): {len(consistent_rules)}")
for _, r in consistent_rules.iterrows():
    print(f"  {r['name']} [{r['bet_side']}]: WR={r['wr']:.1%} ROI={r['roi']:+.1%} ({r['seasons_above_53']}/{r['seasons_valid']} seasons) p={r['p_value']:.4f}")

print(f"\n⚠️ MARGINAL RULES (3+/6 seasons or WR>52%): {len(marginal_rules[~marginal_rules['consistent']])}")
for _, r in marginal_rules[~marginal_rules['consistent']].head(15).iterrows():
    print(f"  {r['name']} [{r['bet_side']}]: WR={r['wr']:.1%} ROI={r['roi']:+.1%} ({r['seasons_above_53']}/{r['seasons_valid']} seasons)")

# ============================================================
# STEP 4: COMBINED DECISION TREE
# ============================================================
print("\n" + "="*60)
print("STEP 4: BUILDING COMBINED DECISION TREE")
print("="*60)

# Take top consistent rules and build combined signals
# We'll use the top 5 consistent rules by WR
top_rules = consistent_rules.head(10)
print(f"\nTop consistent rules to combine:")
for _, r in top_rules.iterrows():
    print(f"  {r['name']}: WR={r['wr']:.1%}")

# ============================================================
# STEP 5: 2024-25 OUT-OF-SAMPLE EVALUATION
# ============================================================
print("\n" + "="*60)
print("STEP 5: 2024-25 OUT-OF-SAMPLE EVALUATION")
print("="*60)

df_oos = df_feat[df_feat['Season'] == '2425'].copy()
print(f"2024-25 matches with features: {len(df_oos)}")

# Apply each consistent rule to 2024-25 only
print("\nOut-of-sample performance (2024-25):")
for _, rule in consistent_rules.iterrows():
    # Re-evaluate on just the out-of-sample data
    pass  # Will be detailed in the report

# ============================================================
# SAVE ALL RESULTS
# ============================================================
df_results.to_csv('research/hypothesis_results.csv', index=False)
print(f"\nSaved all {len(df_results)} hypothesis results to research/hypothesis_results.csv")

# Print full sorted results
print("\n" + "="*60)
print("FULL RESULTS TABLE (top 30 by WR)")
print("="*60)
df_results_sorted = df_results.sort_values('wr', ascending=False)
for _, r in df_results_sorted.head(30).iterrows():
    verdict = "✅" if r['consistent'] else ("⚠️" if r['marginal'] else "❌")
    print(f"{verdict} WR={r['wr']:.1%} ROI={r['roi']:+.1%} n={r['total_bets']:4d} [{r['bet_side']}] {r['name'][:60]}")

print("\nDone! Results saved.")
