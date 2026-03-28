"""
Per-Game AH Prediction System
Finds the AH line closest to 2.0 odds and predicts which side to bet based on:
- Team-specific H2H history
- Recent form (home/away venue-specific)
- Goal margin prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / 'data' / 'real_ah_bettable.parquet'


def load_data():
    """Load the main dataset."""
    df = pd.read_parquet(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def get_team_profile(df, team, as_home=True, n_matches=8, before_date=None):
    """Get recent performance stats for a team at a specific venue."""
    if before_date is not None:
        df = df[df['Date'] < pd.Timestamp(before_date)]

    if as_home:
        recent = df[df['HomeTeam'] == team].tail(n_matches)
        if len(recent) < 3:
            return None
        goals_scored = recent['FTHG'].mean()
        goals_conceded = recent['FTAG'].mean()
        wins = (recent['FTHG'] > recent['FTAG']).sum()
        draws = (recent['FTHG'] == recent['FTAG']).sum()
    else:
        recent = df[df['AwayTeam'] == team].tail(n_matches)
        if len(recent) < 3:
            return None
        goals_scored = recent['FTAG'].mean()
        goals_conceded = recent['FTHG'].mean()
        wins = (recent['FTAG'] > recent['FTHG']).sum()
        draws = (recent['FTAG'] == recent['FTHG']).sum()

    losses = len(recent) - wins - draws
    form_pts = (wins * 3 + draws) / len(recent)

    return {
        'team': team,
        'venue': 'home' if as_home else 'away',
        'n': len(recent),
        'goals_scored_avg': round(float(goals_scored), 2),
        'goals_conceded_avg': round(float(goals_conceded), 2),
        'goal_diff': round(float(goals_scored - goals_conceded), 2),
        'form_pts_per_game': round(float(form_pts), 2),
        'wins': int(wins),
        'draws': int(draws),
        'losses': int(losses),
        'record': f'{int(wins)}W {int(draws)}D {int(losses)}L',
    }


def get_h2h(df, home_team, away_team, n=6, before_date=None):
    """Get head-to-head history between two specific teams."""
    if before_date is not None:
        df = df[df['Date'] < pd.Timestamp(before_date)]

    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].tail(n)

    if len(h2h) < 2:
        return None

    # From home_team perspective
    home_wins = (
        ((h2h['HomeTeam'] == home_team) & (h2h['FTHG'] > h2h['FTAG'])).sum() +
        ((h2h['AwayTeam'] == home_team) & (h2h['FTAG'] > h2h['FTHG'])).sum()
    )
    away_wins = (
        ((h2h['HomeTeam'] == away_team) & (h2h['FTHG'] > h2h['FTAG'])).sum() +
        ((h2h['AwayTeam'] == away_team) & (h2h['FTAG'] > h2h['FTHG'])).sum()
    )
    draws = (h2h['FTHG'] == h2h['FTAG']).sum()

    avg_home_goals = h2h.apply(
        lambda r: r['FTHG'] if r['HomeTeam'] == home_team else r['FTAG'], axis=1
    ).mean()
    avg_away_goals = h2h.apply(
        lambda r: r['FTAG'] if r['HomeTeam'] == home_team else r['FTHG'], axis=1
    ).mean()

    # Last 3 meetings results
    last_results = []
    for _, row in h2h.tail(3).iterrows():
        if row['HomeTeam'] == home_team:
            if row['FTHG'] > row['FTAG']:
                last_results.append(f"{home_team} W ({int(row['FTHG'])}-{int(row['FTAG'])})")
            elif row['FTHG'] < row['FTAG']:
                last_results.append(f"{away_team} W ({int(row['FTHG'])}-{int(row['FTAG'])})")
            else:
                last_results.append(f"Draw ({int(row['FTHG'])}-{int(row['FTAG'])})")
        else:
            if row['FTHG'] > row['FTAG']:
                last_results.append(f"{away_team} W ({int(row['FTAG'])}-{int(row['FTHG'])})")
            elif row['FTHG'] < row['FTAG']:
                last_results.append(f"{home_team} W ({int(row['FTAG'])}-{int(row['FTHG'])})")
            else:
                last_results.append(f"Draw ({int(row['FTHG'])}-{int(row['FTAG'])})")

    return {
        'n_meetings': len(h2h),
        'home_team_wins': int(home_wins),
        'away_team_wins': int(away_wins),
        'draws': int(draws),
        'avg_home_goals': round(float(avg_home_goals), 2),
        'avg_away_goals': round(float(avg_away_goals), 2),
        'predicted_margin': round(float(avg_home_goals - avg_away_goals), 2),
        'last_3_results': last_results,
    }


def predict_match(df, home_team, away_team, ah_line, date=None):
    """
    Given a specific match and AH line, predict which side to bet.
    Returns detailed reasoning with stats.
    """
    home_profile = get_team_profile(df, home_team, as_home=True, before_date=date)
    away_profile = get_team_profile(df, away_team, as_home=False, before_date=date)
    h2h = get_h2h(df, home_team, away_team, before_date=date)

    if not home_profile or not away_profile:
        return {
            'match': f'{home_team} vs {away_team}',
            'ah_line': ah_line,
            'recommendation': 'SKIP',
            'confidence': 'N/A',
            'reason': 'Insufficient data',
            'home_profile': home_profile,
            'away_profile': away_profile,
            'h2h': h2h,
        }

    # === Signal computation ===
    signals = []
    weights = []

    # Signal 1: Attack vs defence matchup (home attack vs away defence, minus reverse)
    home_strength = home_profile['goals_scored_avg'] - away_profile['goals_conceded_avg']
    away_strength = away_profile['goals_scored_avg'] - home_profile['goals_conceded_avg']
    form_signal = home_strength - away_strength
    signals.append(form_signal)
    weights.append(0.35)

    # Signal 2: Goal difference proxy
    gd_signal = (home_profile['goal_diff'] - away_profile['goal_diff']) * 0.5
    signals.append(gd_signal)
    weights.append(0.25)

    # Signal 3: Form points per game
    form_pts_signal = (home_profile['form_pts_per_game'] - away_profile['form_pts_per_game']) * 0.8
    signals.append(form_pts_signal)
    weights.append(0.20)

    # Signal 4: H2H if enough meetings
    if h2h and h2h['n_meetings'] >= 3:
        signals.append(h2h['predicted_margin'])
        weights.append(0.20)
    else:
        # Renormalize to sum to 1.0
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

    # Weighted prediction of goal margin (positive = home team favored)
    predicted_margin = sum(s * w for s, w in zip(signals, weights))

    # === AH decision ===
    # ah_line < 0 means home team gives handicap (e.g. -0.5 means home must win by 1+)
    # ah_line > 0 means home team receives handicap (e.g. +0.5 means home just needs to not lose by 1+)
    if ah_line < 0:
        required_for_home = abs(ah_line)
        edge = predicted_margin - required_for_home
    else:
        required_for_away_cover = -ah_line  # negative = away needs to not lose by this much
        edge = predicted_margin - required_for_away_cover

    # Decision thresholds
    if edge > 0.5:
        recommendation = 'BET HOME'
        confidence = 'HIGH' if edge > 0.8 else 'MEDIUM'
    elif edge < -0.5:
        recommendation = 'BET AWAY'
        confidence = 'HIGH' if edge < -0.8 else 'MEDIUM'
    else:
        recommendation = 'SKIP'
        confidence = 'LOW'

    # Build reasoning text
    reasoning = (
        f"{home_team} home: {home_profile['record']} last {home_profile['n']}, "
        f"{home_profile['goals_scored_avg']:.1f} scored/{home_profile['goals_conceded_avg']:.1f} conceded. "
        f"{away_team} away: {away_profile['record']} last {away_profile['n']}, "
        f"{away_profile['goals_scored_avg']:.1f} scored/{away_profile['goals_conceded_avg']:.1f} conceded. "
    )
    if h2h:
        reasoning += (
            f"H2H ({h2h['n_meetings']} meetings): avg "
            f"{h2h['avg_home_goals']:.1f}-{h2h['avg_away_goals']:.1f} "
            f"({h2h['home_team_wins']}W-{h2h['draws']}D-{h2h['away_team_wins']}L for {home_team}). "
        )
    reasoning += (
        f"Predicted margin: {predicted_margin:+.2f} goals. "
        f"AH line: {ah_line:+.2f} (edge: {edge:+.2f})."
    )

    return {
        'match': f'{home_team} vs {away_team}',
        'ah_line': ah_line,
        'predicted_margin': round(predicted_margin, 2),
        'edge': round(edge, 2),
        'recommendation': recommendation,
        'confidence': confidence,
        'reasoning': reasoning,
        'home_profile': home_profile,
        'away_profile': away_profile,
        'h2h': h2h,
    }


def estimate_ah_from_odds(home_odds, draw_odds, away_odds):
    """Estimate AH line from 1X2 odds."""
    h = 1 / home_odds
    d = 1 / draw_odds
    a = 1 / away_odds
    total = h + d + a
    home_p = h / total
    away_p = a / total
    edge = home_p - away_p

    if edge > 0.25:   return -1.0
    elif edge > 0.15: return -0.75
    elif edge > 0.08: return -0.5
    elif edge > 0.03: return -0.25
    elif edge > -0.03: return 0.0
    elif edge > -0.08: return 0.25
    elif edge > -0.15: return 0.5
    else:             return 1.0


def find_ah_line_closest_to_evens(df, home_team, away_team, before_date=None):
    """
    Find the AH line for a specific matchup from historical data
    where AvgAHH and AvgAHA are closest to 2.0 (true 50/50).
    """
    if before_date is not None:
        subset = df[df['Date'] < pd.Timestamp(before_date)]
    else:
        subset = df

    matches = subset[
        (subset['HomeTeam'] == home_team) & (subset['AwayTeam'] == away_team)
    ]

    if len(matches) == 0:
        return None

    # Find line where avg odds are closest to 2.0
    if 'AvgAHH' in matches.columns:
        matches = matches.copy()
        matches['odds_dist_from_evens'] = (matches['AvgAHH'] - 2.0).abs()
        best = matches.loc[matches['odds_dist_from_evens'].idxmin()]
        return {
            'ah_line': best['real_ah_line'],
            'avg_ahh': best['AvgAHH'],
            'avg_aha': best['AvgAHA'],
        }

    return None


def check_ah_result(row, recommendation):
    """Check if a recommendation was correct for a historical match."""
    if recommendation == 'SKIP':
        return None

    home_goals = row['FTHG']
    away_goals = row['FTAG']
    ah_line = row['real_ah_line']
    margin = home_goals - away_goals

    # Adjusted margin with handicap
    adjusted = margin + ah_line  # e.g. ah_line=-0.5 means home needs +0.5 buffer

    if adjusted > 0:
        actual = 'HOME'
    elif adjusted < 0:
        actual = 'AWAY'
    else:
        actual = 'PUSH'  # exact tie on handicap

    if actual == 'PUSH':
        return None  # push — stake returned, skip

    predicted = 'HOME' if recommendation == 'BET HOME' else 'AWAY'
    return predicted == actual


if __name__ == '__main__':
    # Quick demo
    df = load_data()
    print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Teams: {df['HomeTeam'].nunique()} unique teams")

    # Man United vs Everton demo
    result = predict_match(df, 'Man United', 'Everton', ah_line=-0.25)
    print(f"\n{result['match']}")
    print(f"Recommendation: {result['recommendation']} ({result['confidence']})")
    print(f"Reasoning: {result['reasoning']}")
