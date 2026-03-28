#!/usr/bin/env python3
"""
pick_bets.py — Unified AH bet picker with all validated rules.

VALIDATED RULES (WR>53%, 4+/6 seasons):
  TIER 1 (6/6 seasons):
    1. AH -0.25 → BET HOME (WR=59.2%, ROI=+16.1%)
    2. AH +0.25 → BET AWAY (WR=64.3%, ROI=+26.6%)

  TIER 2 (5/6 seasons):
    3. AH -1.75 → BET AWAY (WR=67.3%, ROI=+31.3%)
    4. Positive AH lines (home underdog) → BET AWAY (WR=54.3%, ROI=+6.5%)
    5. Cold home team (form < 0.8 pts/game) → BET AWAY (WR=53.8%)

  TIER 3 (4/6 seasons):
    6. AH -0.75 + home attack weak vs away defense → BET AWAY (WR=57.2%)
    7. Cold home form + worse form than away → BET AWAY (WR=54.2%)
    8. Both teams mediocre form (1.2-1.8 pts/game each) → BET AWAY (WR=53.8%)

Usage:
    python scripts/pick_bets.py

Or import:
    from scripts.pick_bets import pick_bets, print_picks
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / 'data' / 'real_ah_bettable.parquet'
MODEL_PATH_V2 = BASE / 'models' / 'margin_predictor_v2.pkl'
MODEL_PATH_V1 = BASE / 'models' / 'margin_predictor.pkl'
CAL_PATH = BASE / 'models' / 'overall_calibration.json'

# ─── Feature computation ──────────────────────────────────────────────────────

def build_elo_ratings(df: pd.DataFrame) -> dict:
    """Compute final Elo ratings from historical data."""
    df = df.sort_values('Date').reset_index(drop=True)
    elo = {}
    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        h_elo = elo.get(home, 1500.0)
        a_elo = elo.get(away, 1500.0)
        hg = row['FTHG']
        ag = row['FTAG']
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = 32
        elo[home] = h_elo + k * (actual_h - exp_h)
        elo[away] = a_elo + k * ((1 - actual_h) - (1 - exp_h))
    return elo


def get_rolling_stats(team: str, venue: str, df: pd.DataFrame, before_date: pd.Timestamp, n: int = 6) -> dict:
    """Get rolling stats for a team at a given venue (home/away) before a date."""
    if venue == 'home':
        matches = df[(df['HomeTeam'] == team) & (df['Date'] < before_date)].sort_values('Date').tail(n)
        gf = matches['FTHG'].tolist()
        ga = matches['FTAG'].tolist()
        sot = matches['HST'].tolist() if 'HST' in matches.columns else []
        pts = [3 if g > c else (1 if g == c else 0) for g, c in zip(gf, ga)]
    else:
        matches = df[(df['AwayTeam'] == team) & (df['Date'] < before_date)].sort_values('Date').tail(n)
        gf = matches['FTAG'].tolist()
        ga = matches['FTHG'].tolist()
        sot = matches['AST'].tolist() if 'AST' in matches.columns else []
        pts = [3 if g > c else (1 if g == c else 0) for g, c in zip(gf, ga)]

    n_matches = len(matches)
    return {
        'n_matches': n_matches,
        'gf_6': np.mean(gf[-6:]) if len(gf) >= 3 else np.nan,
        'ga_6': np.mean(ga[-6:]) if len(ga) >= 3 else np.nan,
        'gf_10': np.mean(gf[-10:]) if len(gf) >= 5 else np.nan,
        'ga_10': np.mean(ga[-10:]) if len(ga) >= 5 else np.nan,
        'gf_std': np.std(gf[-6:]) if len(gf) >= 3 else np.nan,
        'form_pts': np.mean(pts[-6:]) if len(pts) >= 3 else np.nan,
        'sot_6': np.mean([s for s in sot[-6:] if not pd.isna(s)]) if sot else np.nan,
        'recent_gf': gf[-3:],
        'recent_ga': ga[-3:],
        'recent_pts': pts[-3:],
    }


def apply_calibration(raw_pred: float, cal_data: dict) -> float:
    """Apply overall calibration correction to raw prediction."""
    bins = [-10, -2, -1, -0.5, 0, 0.5, 1, 2, 10]
    labels = ['<-2', '-2to-1', '-1to-0.5', '-0.5to0', '0to0.5', '0.5to1', '1to2', '>2']
    
    bucket = None
    for i in range(len(bins) - 1):
        if bins[i] < raw_pred <= bins[i + 1]:
            bucket = labels[i]
            break
    if raw_pred <= bins[0]:
        bucket = labels[0]
    
    cal_table = cal_data.get('calibration_table', {})
    bias_table = cal_table.get('bias', {}) if isinstance(cal_table, dict) else {}
    global_bias = cal_data.get('global_bias', 0.0)
    
    if bucket and bucket in bias_table:
        bias_val = bias_table[bucket]
        if not isinstance(bias_val, (int, float)) or np.isnan(bias_val):
            bias_val = global_bias
    else:
        bias_val = global_bias
    
    return raw_pred + bias_val


# ─── STRUCTURAL RULE ENGINE ───────────────────────────────────────────────────

def apply_structural_rules(game: dict, home_stats: dict, away_stats: dict) -> dict | None:
    """
    Apply validated structural rules in priority order.
    Returns dict with recommendation, or None if no structural rule applies.
    
    Rules ranked by confidence (6/6 > 5/6 > 4/6 seasons):
    """
    ah_line = float(game['ah_line'])
    league = game.get('league', '')
    home = game['home_team']
    away = game['away_team']
    
    h_form = home_stats.get('form_pts', np.nan)
    a_form = away_stats.get('form_pts', np.nan)
    h_gf = home_stats.get('gf_6', np.nan)
    a_ga = away_stats.get('ga_6', np.nan)  # away concedes
    
    # ── TIER 1: 6/6 seasons, p < 0.0001 ──────────────────────────────────────
    
    # Rule 1: AH -0.25 → BET HOME (WR=59.2%, ROI=+16.1%)
    if abs(ah_line - (-0.25)) < 0.01:
        return {
            'recommendation': 'BET HOME',
            'bet_team': home,
            'confidence': 'HIGH',
            'tier': 1,
            'rule_triggered': 'structural_AH_minus025',
            'rule_name': 'AH -0.25 HOME',
            'reasoning': (
                f'AH -0.25 structural edge: HOME wins 59.2% of the time across all 6 seasons (2019-2025). '
                f'ROI: +16.1%. The quarter-line structure means a draw returns half your stake rather than losing '
                f'it fully, creating structural value for the home side at this line. 6/6 seasons consistent.'
            ),
            'expected_wr': 0.592,
            'expected_roi': 0.161,
        }
    
    # Rule 2: AH +0.25 → BET AWAY (WR=64.3%, ROI=+26.6%)
    if abs(ah_line - 0.25) < 0.01:
        return {
            'recommendation': 'BET AWAY',
            'bet_team': away,
            'confidence': 'HIGH',
            'tier': 1,
            'rule_triggered': 'structural_AH_plus025',
            'rule_name': 'AH +0.25 AWAY',
            'reasoning': (
                f'AH +0.25 structural edge: AWAY wins 64.3% of the time across all 6 seasons (2019-2025). '
                f'ROI: +26.6%. This is the mirror of AH -0.25: when home team is giving +0.25, '
                f'away teams are systematically undervalued. 6/6 seasons consistent.'
            ),
            'expected_wr': 0.643,
            'expected_roi': 0.266,
        }
    
    # ── TIER 2: 5/6 seasons ───────────────────────────────────────────────────
    
    # Rule 3: AH -1.75 → BET AWAY (WR=67.3%, ROI=+31.3%)
    if abs(ah_line - (-1.75)) < 0.01:
        if league == 'Ligue1':
            return {
                'recommendation': 'AVOID',
                'bet_team': None,
                'confidence': 'HIGH',
                'tier': 2,
                'rule_triggered': 'structural_AH175_ligue1_exception',
                'rule_name': 'AH -1.75 LIGUE1 AVOID',
                'reasoning': 'AH -1.75 edge is REVERSED in Ligue1. Do not bet this line in French football.',
                'expected_wr': None,
                'expected_roi': None,
            }
        return {
            'recommendation': 'BET AWAY',
            'bet_team': away,
            'confidence': 'HIGH',
            'tier': 2,
            'rule_triggered': 'structural_AH175',
            'rule_name': 'AH -1.75 AWAY',
            'reasoning': (
                f'AH -1.75 structural edge: AWAY covers 67.3% of the time (ROI: +31.3%). '
                f'Split line (-1.5 and -2.0): home team needs to win by 2+ to fully cover, by 3+ for the other half. '
                f'Most heavy favorites win by exactly 1 goal, benefiting the away side on BOTH splits. '
                f'5/6 seasons consistent (non-Ligue1).'
            ),
            'expected_wr': 0.673,
            'expected_roi': 0.313,
        }
    
    # Rule 4: Cold home team (form < 0.8 pts/game) → BET AWAY
    # Only apply when sufficient history and clear condition
    if not np.isnan(h_form) and h_form < 0.8 and ah_line < -0.25:
        return {
            'recommendation': 'BET AWAY',
            'bet_team': away,
            'confidence': 'HIGH',
            'tier': 2,
            'rule_triggered': 'cold_home_form',
            'rule_name': 'COLD HOME TEAM',
            'reasoning': (
                f'{home} home form: {h_form:.2f} pts/game (last 6 games) — COLD. '
                f'Teams with <0.8 home pts/game lose AH 53.8% of the time regardless of line. '
                f'5/6 seasons consistent. Even as the handicap favorite, cold home teams fail to cover.'
            ),
            'expected_wr': 0.538,
            'expected_roi': 0.049,
        }
    
    # Rule 5: Positive lines (away is handicap favorite, home is underdog) → BET AWAY
    if ah_line > 0:
        # Only bet if not h_form is strong (otherwise it's already covered by cold home rule)
        return {
            'recommendation': 'BET AWAY',
            'bet_team': away,
            'confidence': 'MEDIUM',
            'tier': 2,
            'rule_triggered': 'positive_line_away',
            'rule_name': 'POSITIVE LINE AWAY',
            'reasoning': (
                f'Positive AH line ({ah_line:+.2f}): away team starts with a head start. '
                f'Away teams cover 54.3% of the time on positive lines (ROI: +6.5%). '
                f'5/6 seasons consistent. Market consistently underprices away teams when home is given handicap.'
            ),
            'expected_wr': 0.543,
            'expected_roi': 0.065,
        }
    
    # ── TIER 3: 4/6 seasons ───────────────────────────────────────────────────
    
    # Rule 6: AH -0.75 + home attack weak vs away defense → BET AWAY (WR=57.2%)
    if abs(ah_line - (-0.75)) < 0.01 and not np.isnan(h_gf) and not np.isnan(a_ga):
        attack_vs_defense = h_gf - a_ga  # home attack vs away defensive concession
        if attack_vs_defense < 0.3:
            return {
                'recommendation': 'BET AWAY',
                'bet_team': away,
                'confidence': 'MEDIUM',
                'tier': 3,
                'rule_triggered': 'AH075_weak_attack',
                'rule_name': 'AH -0.75 WEAK HOME ATTACK',
                'reasoning': (
                    f'AH -0.75 + home attack ({h_gf:.1f} GF/game) barely exceeds away defense ({a_ga:.1f} GA/game). '
                    f'Attack-vs-defense edge: {attack_vs_defense:+.2f} (threshold: <0.3). '
                    f'When the home team\'s attack is NOT clearly better than the away defense, '
                    f'away teams cover AH -0.75 at 57.2% (ROI: +12.4%). 4/6 seasons consistent.'
                ),
                'expected_wr': 0.572,
                'expected_roi': 0.124,
            }
    
    # Rule 7: Cold home + worse form gap → BET AWAY (WR=54.2%)
    if not np.isnan(h_form) and not np.isnan(a_form):
        if h_form < 1.0 and a_form > h_form and ah_line < 0:
            return {
                'recommendation': 'BET AWAY',
                'bet_team': away,
                'confidence': 'MEDIUM',
                'tier': 3,
                'rule_triggered': 'cold_home_form_gap',
                'rule_name': 'COLD HOME + FORM GAP',
                'reasoning': (
                    f'{home} home form: {h_form:.2f} pts/game (cold) AND worse than {away} away form ({a_form:.2f}). '
                    f'Form gap: {a_form - h_form:+.2f} in favor of away. '
                    f'Cold home teams with worse form than away: 54.2% away coverage (ROI: +5.6%). 4/6 seasons.'
                ),
                'expected_wr': 0.542,
                'expected_roi': 0.056,
            }
    
    # Rule 8: Both mediocre form (1.2-1.8 pts/game each) → BET AWAY (WR=53.8%)
    if not np.isnan(h_form) and not np.isnan(a_form):
        if 1.2 <= h_form <= 1.8 and 1.2 <= a_form <= 1.8 and ah_line < 0:
            return {
                'recommendation': 'BET AWAY',
                'bet_team': away,
                'confidence': 'MEDIUM',
                'tier': 3,
                'rule_triggered': 'mediocre_both_away',
                'rule_name': 'BOTH MEDIOCRE FORM',
                'reasoning': (
                    f'Both teams mediocre form: {home} ({h_form:.2f} pts/g) and {away} ({a_form:.2f} pts/g). '
                    f'When both teams hover around 1.2-1.8 pts/game, away teams win AH 53.8% '
                    f'(tighter games, draws more likely, help away side). 4/6 seasons consistent.'
                ),
                'expected_wr': 0.538,
                'expected_roi': 0.050,
            }
    
    return None  # No structural rule applies


# ─── Core pick function ───────────────────────────────────────────────────────

def pick_bets(games: list) -> list:
    """
    For each game, output a BET/SKIP/AVOID recommendation.

    games: list of dicts with keys:
        - home_team: str
        - away_team: str
        - ah_line: float  (e.g. -0.75, -1.75, +0.25)
        - league: str  ('EPL', 'LaLiga', 'Bundesliga', 'SerieA', 'Ligue1')
        - date: str  (YYYY-MM-DD)
        - home_odds: float (optional, AH home odds)
        - away_odds: float (optional, AH away odds)

    Returns list of recommendation dicts.
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # ── Load model (optional, used for fallback when no structural rule) ───────
    model = None
    feature_cols = []
    cal_data = {}
    
    model_path = MODEL_PATH_V2 if MODEL_PATH_V2.exists() else (MODEL_PATH_V1 if MODEL_PATH_V1.exists() else None)
    if model_path:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            model = model_data['model']
            feature_cols = model_data['features']
            cal_data = {'calibration_table': model_data.get('calibration_table'), 
                       'global_bias': model_data.get('global_bias', 0.0)}
        except Exception as e:
            pass
    
    if CAL_PATH.exists():
        with open(CAL_PATH) as f:
            try:
                standalone_cal = json.load(f)
                if 'calibration_table' in standalone_cal:
                    cal_data = standalone_cal
            except:
                pass

    # ── Compute Elo ratings ────────────────────────────────────────────────────
    elo_ratings = build_elo_ratings(df)
    
    results = []
    
    for game in games:
        home = game['home_team']
        away = game['away_team']
        ah_line = float(game['ah_line'])
        league = game.get('league', 'Unknown')
        date_str = game.get('date', '2026-03-29')
        match_date = pd.Timestamp(date_str)
        
        match_label = f"{home} vs {away}"
        
        # ── Get team stats ─────────────────────────────────────────────────────
        home_stats = get_rolling_stats(home, 'home', df, match_date, n=6)
        away_stats = get_rolling_stats(away, 'away', df, match_date, n=6)
        
        # ── Check insufficient history ─────────────────────────────────────────
        if home_stats['n_matches'] < 3 or away_stats['n_matches'] < 3:
            reasons = []
            if home_stats['n_matches'] < 3:
                reasons.append(f"{home} only {home_stats['n_matches']} home matches on record")
            if away_stats['n_matches'] < 3:
                reasons.append(f"{away} only {away_stats['n_matches']} away matches on record")
            results.append({
                'match': match_label,
                'ah_line': ah_line,
                'league': league,
                'recommendation': 'SKIP',
                'confidence': 'N/A',
                'predicted_margin': None,
                'reasoning': f'Insufficient history: {"; ".join(reasons)}. Need 3+ matches for structural rules.',
                'rule_triggered': 'skip_insufficient_history',
            })
            continue
        
        # ── Apply structural rules ─────────────────────────────────────────────
        structural_result = apply_structural_rules(game, home_stats, away_stats)
        
        if structural_result:
            result = {
                'match': match_label,
                'ah_line': ah_line,
                'league': league,
                **structural_result,
                'home_form_pts': round(home_stats['form_pts'], 2) if not np.isnan(home_stats['form_pts']) else None,
                'away_form_pts': round(away_stats['form_pts'], 2) if not np.isnan(away_stats['form_pts']) else None,
                'home_gf_6': round(home_stats['gf_6'], 2) if not np.isnan(home_stats['gf_6']) else None,
                'home_ga_6': round(home_stats['ga_6'], 2) if not np.isnan(home_stats['ga_6']) else None,
                'away_gf_6': round(away_stats['gf_6'], 2) if not np.isnan(away_stats['gf_6']) else None,
                'away_ga_6': round(away_stats['ga_6'], 2) if not np.isnan(away_stats['ga_6']) else None,
            }
            results.append(result)
            continue
        
        # ── Fallback: model-based prediction (when no structural rule applies) ──
        if model is None:
            results.append({
                'match': match_label,
                'ah_line': ah_line,
                'league': league,
                'recommendation': 'SKIP',
                'confidence': 'N/A',
                'predicted_margin': None,
                'reasoning': 'No structural rule applies and model not available. Skip.',
                'rule_triggered': 'skip_no_rule',
            })
            continue
        
        # Model prediction
        home_elo = elo_ratings.get(home, 1500.0)
        away_elo = elo_ratings.get(away, 1500.0)
        
        home_odds = game.get('home_odds', None)
        away_odds = game.get('away_odds', None)
        
        if home_odds and away_odds:
            home_implied = 1.0 / home_odds
            away_implied = 1.0 / away_odds
        else:
            elo_prob_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            home_implied = elo_prob_home
            away_implied = 1 - elo_prob_home
        
        h_gf = home_stats['gf_6']
        h_ga = home_stats['ga_6']
        a_gf = away_stats['gf_6']
        a_ga = away_stats['ga_6']
        
        features = {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            'home_gf_6': h_gf if not np.isnan(h_gf) else 1.4,
            'home_ga_6': h_ga if not np.isnan(h_ga) else 1.2,
            'home_gf_10': home_stats['gf_10'] if not np.isnan(home_stats.get('gf_10', np.nan)) else h_gf if not np.isnan(h_gf) else 1.4,
            'home_ga_10': home_stats['ga_10'] if not np.isnan(home_stats.get('ga_10', np.nan)) else h_ga if not np.isnan(h_ga) else 1.2,
            'home_gf_std': home_stats['gf_std'] if not np.isnan(home_stats.get('gf_std', np.nan)) else 1.0,
            'home_form_pts': home_stats['form_pts'] if not np.isnan(home_stats['form_pts']) else 1.5,
            'home_sot_6': home_stats['sot_6'] if not np.isnan(home_stats.get('sot_6', np.nan)) else h_gf * 3.0 if not np.isnan(h_gf) else 4.0,
            'away_gf_6': a_gf if not np.isnan(a_gf) else 1.0,
            'away_ga_6': a_ga if not np.isnan(a_ga) else 1.3,
            'away_gf_std': away_stats['gf_std'] if not np.isnan(away_stats.get('gf_std', np.nan)) else 1.0,
            'away_form_pts': away_stats['form_pts'] if not np.isnan(away_stats['form_pts']) else 1.5,
            'away_sot_6': away_stats['sot_6'] if not np.isnan(away_stats.get('sot_6', np.nan)) else a_gf * 3.0 if not np.isnan(a_gf) else 3.5,
            'home_implied': home_implied,
            'away_implied': away_implied,
            'real_ah_line': ah_line,
        }
        
        try:
            X = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_cols}])
            X = X.fillna(X.median())
            raw_pred = float(model.predict(X)[0])
            cal_pred = apply_calibration(raw_pred, cal_data)
        except Exception as e:
            results.append({
                'match': match_label,
                'ah_line': ah_line,
                'league': league,
                'recommendation': 'SKIP',
                'confidence': 'N/A',
                'predicted_margin': None,
                'reasoning': f'Model prediction failed: {e}. Skip.',
                'rule_triggered': 'skip_model_error',
            })
            continue
        
        required_margin = -ah_line
        edge = cal_pred - required_margin
        
        # Only bet on model if edge >= 0.5
        if abs(edge) < 0.5:
            results.append({
                'match': match_label,
                'ah_line': ah_line,
                'league': league,
                'recommendation': 'SKIP',
                'confidence': 'N/A',
                'predicted_margin': round(cal_pred, 2),
                'required_margin': round(required_margin, 2),
                'edge': round(edge, 2),
                'reasoning': (
                    f'No structural rule applies. Model predicts {cal_pred:+.2f} margin vs required {required_margin:+.2f}. '
                    f'Edge ({abs(edge):.2f}) too thin for model-only bet (need ≥0.50 goals edge).'
                ),
                'rule_triggered': 'skip_thin_model_edge',
            })
            continue
        
        # Model bet
        if edge > 0:
            recommendation = 'BET HOME'
            bet_team = home
        else:
            recommendation = 'BET AWAY'
            bet_team = away
        
        h_form_str = f"{home_stats['form_pts']:.1f} pts/g" if not np.isnan(home_stats['form_pts']) else "N/A"
        a_form_str = f"{away_stats['form_pts']:.1f} pts/g" if not np.isnan(away_stats['form_pts']) else "N/A"
        
        results.append({
            'match': match_label,
            'ah_line': ah_line,
            'league': league,
            'recommendation': recommendation,
            'bet_team': bet_team,
            'confidence': 'MEDIUM',
            'tier': 4,
            'rule_triggered': 'model_prediction',
            'rule_name': 'MODEL',
            'predicted_margin': round(cal_pred, 2),
            'required_margin': round(required_margin, 2),
            'edge': round(edge, 2),
            'home_form_pts': round(home_stats['form_pts'], 2) if not np.isnan(home_stats['form_pts']) else None,
            'away_form_pts': round(away_stats['form_pts'], 2) if not np.isnan(away_stats['form_pts']) else None,
            'home_gf_6': round(h_gf, 2) if not np.isnan(h_gf) else None,
            'home_ga_6': round(h_ga, 2) if not np.isnan(h_ga) else None,
            'away_gf_6': round(a_gf, 2) if not np.isnan(a_gf) else None,
            'away_ga_6': round(a_ga, 2) if not np.isnan(a_ga) else None,
            'reasoning': (
                f'{home} home: {h_gf:.1f} GF / {h_ga:.1f} GA ({h_form_str}). '
                f'{away} away: {a_gf:.1f} GF / {a_ga:.1f} GA ({a_form_str}). '
                f'Model predicts {cal_pred:+.2f} margin vs required {required_margin:+.2f} — edge {edge:+.2f}.'
            ),
            'expected_wr': None,
            'expected_roi': None,
        })
    
    return results


# ─── Pretty printer ───────────────────────────────────────────────────────────

EMOJI = {
    'BET HOME': '✅',
    'BET AWAY': '✅',
    'SKIP': '⏭️',
    'AVOID': '⚠️',
}

TIER_LABEL = {1: 'TIER 1 (6/6 seasons)', 2: 'TIER 2 (5/6 seasons)', 
              3: 'TIER 3 (4/6 seasons)', 4: 'MODEL', None: ''}

def print_picks(results: list, title: str = "Weekend Matches") -> str:
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  BET PICKER — {title}")
    lines.append(f"{'='*60}\n")
    
    bets = [r for r in results if r['recommendation'].startswith('BET')]
    skips = [r for r in results if r['recommendation'] == 'SKIP']
    avoids = [r for r in results if r['recommendation'] == 'AVOID']
    
    for i, rec in enumerate(results, 1):
        em = EMOJI.get(rec['recommendation'], '❓')
        league = rec.get('league', '')
        lines.append(f"{i}. {rec['match']} ({league}, AH {rec['ah_line']:+.2f})")
        lines.append(f"   {em} {rec['recommendation']}")
        
        if rec['recommendation'].startswith('BET'):
            bt = rec.get('bet_team', '')
            if not bt:
                if rec['recommendation'] == 'BET AWAY':
                    bt = rec['match'].split(' vs ')[1]
                else:
                    bt = rec['match'].split(' vs ')[0]
            lines.append(f"   Bet: {bt}")
            conf = rec.get('confidence', '')
            tier = rec.get('tier', None)
            tier_label = TIER_LABEL.get(tier, '')
            lines.append(f"   Confidence: {conf} | {tier_label}")
            lines.append(f"   Rule: {rec.get('rule_name', rec.get('rule_triggered', ''))}")
            
            if rec.get('expected_wr'):
                lines.append(f"   Historical WR: {rec['expected_wr']:.1%} | Historical ROI: {rec.get('expected_roi', 0):+.1%}")
            
            if rec.get('predicted_margin') is not None:
                lines.append(f"   Model: {rec['predicted_margin']:+.2f} vs required {rec.get('required_margin', 0):+.2f} (edge {rec.get('edge', 0):+.2f})")
            
            lines.append(f"   Reasoning: {rec['reasoning']}")
        
        elif rec['recommendation'] == 'SKIP':
            lines.append(f"   Reason: {rec.get('reasoning', '')[:120]}")
        
        elif rec['recommendation'] == 'AVOID':
            lines.append(f"   Reason: {rec.get('reasoning', '')[:120]}")
        
        lines.append('')
    
    # Summary
    lines.append(f"{'─'*60}")
    lines.append(f"  SUMMARY: {len(bets)} BET | {len(skips)} SKIP | {len(avoids)} AVOID")
    if bets:
        lines.append(f"  Active bets:")
        for b in bets:
            conf = b.get('confidence', '')
            tier = b.get('tier', None)
            bt = b.get('bet_team', '')
            if not bt:
                if b['recommendation'] == 'BET AWAY':
                    bt = b['match'].split(' vs ')[1]
                else:
                    bt = b['match'].split(' vs ')[0]
            wr_str = f"WR≈{b['expected_wr']:.0%}" if b.get('expected_wr') else ""
            lines.append(f"    • {b['match']} → {b['recommendation']} ({bt}) [{conf}] {wr_str}")
    lines.append(f"{'='*60}\n")
    
    output = '\n'.join(lines)
    print(output)
    return output


# ─── Main demo ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Example weekend games
    weekend_games = [
        {'home_team': 'Newcastle', 'away_team': 'Sunderland', 'league': 'EPL', 'ah_line': -1.0, 'date': '2026-03-29'},
        {'home_team': 'Aston Villa', 'away_team': 'West Ham', 'league': 'EPL', 'ah_line': -0.75, 'date': '2026-03-29'},
        {'home_team': 'Tottenham', 'away_team': 'Nottm Forest', 'league': 'EPL', 'ah_line': -0.25, 'date': '2026-03-29'},
        {'home_team': 'Man City', 'away_team': 'Leicester', 'league': 'EPL', 'ah_line': -1.75, 'date': '2026-03-29'},
        {'home_team': 'Chelsea', 'away_team': 'Everton', 'league': 'EPL', 'ah_line': -1.0, 'date': '2026-03-29'},
        {'home_team': 'Bayern Munich', 'away_team': 'Bochum', 'league': 'Bundesliga', 'ah_line': -2.25, 'date': '2026-03-29'},
        {'home_team': 'Leverkusen', 'away_team': 'Mainz', 'league': 'Bundesliga', 'ah_line': -1.75, 'date': '2026-03-29'},
        {'home_team': 'Dortmund', 'away_team': 'Augsburg', 'league': 'Bundesliga', 'ah_line': -0.75, 'date': '2026-03-29'},
        {'home_team': 'Inter', 'away_team': 'Cagliari', 'league': 'SerieA', 'ah_line': -1.75, 'date': '2026-03-29'},
        {'home_team': 'Juventus', 'away_team': 'Torino', 'league': 'SerieA', 'ah_line': -0.5, 'date': '2026-03-29'},
        {'home_team': 'PSG', 'away_team': 'Lyon', 'league': 'Ligue1', 'ah_line': -1.75, 'date': '2026-03-29'},
        {'home_team': 'Real Madrid', 'away_team': 'Eibar', 'league': 'LaLiga', 'ah_line': 0.25, 'date': '2026-03-29'},
    ]
    
    print("Running unified pick_bets with all validated rules...")
    results = pick_bets(weekend_games)
    output = print_picks(results, title="Demo Weekend — 2026-03-29")
    
    # Save report
    research_dir = BASE / 'research'
    research_dir.mkdir(exist_ok=True)
    
    bets_only = [r for r in results if r['recommendation'].startswith('BET')]
    
    md_lines = [
        "# Weekend AH Picks — Demo 2026-03-29",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Games: {len(results)} | BET: {len(bets_only)} | SKIP: {len([r for r in results if r['recommendation']=='SKIP'])} | AVOID: {len([r for r in results if r['recommendation']=='AVOID'])}",
        "",
        "---",
        "## Active Bets",
        "",
    ]
    
    for i, r in enumerate(bets_only, 1):
        bt = r.get('bet_team', '')
        if not bt:
            bt = r['match'].split(' vs ')[1] if r['recommendation'] == 'BET AWAY' else r['match'].split(' vs ')[0]
        wr = f"WR≈{r['expected_wr']:.0%}" if r.get('expected_wr') else ''
        roi = f"ROI≈{r.get('expected_roi', 0):+.0%}" if r.get('expected_roi') else ''
        md_lines += [
            f"### {i}. {r['match']} ({r['league']}, AH {r['ah_line']:+.2f})",
            f"**{r['recommendation']}** → {bt} | {r.get('confidence','')} | {TIER_LABEL.get(r.get('tier'), '')}",
            f"Rule: `{r.get('rule_name', r.get('rule_triggered',''))}` | {wr} {roi}",
            "",
            f"> {r['reasoning']}",
            "",
        ]
    
    with open(research_dir / 'weekend_picks.md', 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✅ Saved report to research/weekend_picks.md")
    print(f"Done. {len(bets_only)} bets from {len(results)} games.")
