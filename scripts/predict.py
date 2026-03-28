"""
Step 8: Prediction pipeline for upcoming matches.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime


def load_models():
    """Load trained models."""
    models_dir = Path(__file__).parent.parent / 'models'
    
    with open(models_dir / 'margin_predictor.pkl', 'rb') as f:
        margin_data = pickle.load(f)
    
    with open(models_dir / 'ah_cover_classifier.pkl', 'rb') as f:
        ah_data = pickle.load(f)
    
    return {
        'margin_model': margin_data['model'],
        'margin_features': margin_data['features'],
        'ah_model': ah_data['model'],
        'ah_features': ah_data['features']
    }


def get_team_stats(team, venue, df, match_date, n_matches=6):
    """Get rolling stats for a team before a specific date."""
    
    if venue == 'home':
        team_matches = df[(df['HomeTeam'] == team) & (df['Date'] < match_date)].tail(n_matches)
        goals_for = team_matches['FTHG'].mean() if len(team_matches) > 0 else np.nan
        goals_against = team_matches['FTAG'].mean() if len(team_matches) > 0 else np.nan
        shots_on_target = team_matches['home_shots_on_target_h6'].mean() if len(team_matches) > 0 else np.nan
        shots = team_matches['home_shots_h6'].mean() if len(team_matches) > 0 else np.nan
    else:
        team_matches = df[(df['AwayTeam'] == team) & (df['Date'] < match_date)].tail(n_matches)
        goals_for = team_matches['FTAG'].mean() if len(team_matches) > 0 else np.nan
        goals_against = team_matches['FTHG'].mean() if len(team_matches) > 0 else np.nan
        shots_on_target = team_matches['away_shots_on_target_a6'].mean() if len(team_matches) > 0 else np.nan
        shots = team_matches['away_shots_a6'].mean() if len(team_matches) > 0 else np.nan
    
    return {
        'goals_for': goals_for,
        'goals_against': goals_against,
        'shots_on_target': shots_on_target,
        'shots': shots,
        'xg': shots_on_target * 0.33 if not pd.isna(shots_on_target) else np.nan,
        'n_matches': len(team_matches)
    }


def predict_match(home_team, away_team, ah_line, home_odds=1.90, away_odds=1.90, 
                  match_date=None, league='EPL', models=None, df=None):
    """
    Predict outcome of an upcoming match.
    
    Returns betting recommendation based on predicted margin vs AH line.
    """
    if models is None:
        models = load_models()
    
    if df is None:
        data_path = Path(__file__).parent.parent / 'data' / 'features_engineered.parquet'
        df = pd.read_parquet(data_path)
    
    if match_date is None:
        match_date = pd.Timestamp.now()
    else:
        match_date = pd.Timestamp(match_date)
    
    # Get recent stats for both teams
    home_stats = get_team_stats(home_team, 'home', df, match_date)
    away_stats = get_team_stats(away_team, 'away', df, match_date)
    
    # Build features
    features = {
        'home_goals_scored_h6': home_stats['goals_for'],
        'home_shots_on_target_h6': home_stats['shots_on_target'],
        'home_shots_h6': home_stats['shots'],
        'home_xg_h6': home_stats['xg'],
        'home_goals_conceded_h6': home_stats['goals_against'],
        
        'away_goals_scored_a6': away_stats['goals_for'],
        'away_shots_on_target_a6': away_stats['shots_on_target'],
        'away_shots_a6': away_stats['shots'],
        'away_xg_a6': away_stats['xg'],
        'away_goals_conceded_a6': away_stats['goals_against'],
        
        'home_form_pts': 1.5,  # Placeholder - would need full computation
        'away_form_pts': 1.5,  # Placeholder
        
        'h2h_avg_home_goals': np.nan,  # Would need H2H lookup
        'h2h_avg_away_goals': np.nan,
        'h2h_home_wins': np.nan,
        
        'market_home_implied': 1 / home_odds if home_odds else np.nan,
        'market_away_implied': 1 / away_odds if away_odds else np.nan,
        'market_draw_implied': 0.25,  # Placeholder
        
        'match_week': 20,  # Placeholder
        'is_early_season': 0,
        'is_late_season': 0,
    }
    
    # Fill NaN with dataset medians
    for col in models['margin_features']:
        if pd.isna(features.get(col)):
            features[col] = df[col].median() if col in df.columns else 0
    
    # Predict margin
    X = pd.DataFrame([{k: features[k] for k in models['margin_features']}])
    predicted_margin = models['margin_model'].predict(X)[0]
    
    # Add AH line for classifier
    features['market_ah_line'] = ah_line
    X_ah = pd.DataFrame([{k: features.get(k, 0) for k in models['ah_features']}])
    ah_home_prob = models['ah_model'].predict_proba(X_ah)[0][1]
    
    # Decide bet
    home_edge = predicted_margin + ah_line
    
    min_edge = 0.5  # Based on backtest results
    
    if home_edge > min_edge:
        bet = 'BET_HOME'
        bet_team = home_team
        bet_odds = home_odds
        edge = home_edge
    elif -home_edge > min_edge:
        bet = 'BET_AWAY'
        bet_team = away_team
        bet_odds = away_odds
        edge = -home_edge
    else:
        bet = 'SKIP'
        bet_team = None
        bet_odds = None
        edge = abs(home_edge)
    
    # Interpret predicted result
    if predicted_margin > 0.75:
        pred_result = f"Home wins by {int(round(predicted_margin))} goals"
    elif predicted_margin > 0.25:
        pred_result = "Home wins narrowly"
    elif predicted_margin > -0.25:
        pred_result = "Draw likely"
    elif predicted_margin > -0.75:
        pred_result = "Away wins narrowly"
    else:
        pred_result = f"Away wins by {int(round(abs(predicted_margin)))} goals"
    
    return {
        'match': f"{home_team} vs {away_team}",
        'predicted_margin': round(predicted_margin, 2),
        'predicted_result': pred_result,
        'ah_line': ah_line,
        'ah_home_prob': round(ah_home_prob, 3),
        'bet_recommendation': bet,
        'bet_team': bet_team,
        'edge': round(edge, 2),
        'odds': bet_odds,
        'reasoning': f"Home xG: {features['home_xg_h6']:.1f}, Away xG: {features['away_xg_a6']:.1f}, "
                    f"Home goals scored: {features['home_goals_scored_h6']:.1f}, "
                    f"Away goals conceded: {features['away_goals_conceded_a6']:.1f}",
        'data_quality': f"Home matches: {home_stats['n_matches']}, Away matches: {away_stats['n_matches']}"
    }


if __name__ == "__main__":
    # Load models and data
    models = load_models()
    data_path = Path(__file__).parent.parent / 'data' / 'features_engineered.parquet'
    df = pd.read_parquet(data_path)
    
    print("Soccer Alpha - Match Prediction System")
    print("=" * 60)
    
    # Test predictions on sample matches from the dataset (recent ones)
    recent = df[df['Date'] >= '2025-03-01'].head(5)
    
    print("\nRecent matches in dataset (for testing predictions):")
    for _, row in recent.iterrows():
        pred = predict_match(
            home_team=row['HomeTeam'],
            away_team=row['AwayTeam'],
            ah_line=row['market_ah_line'],
            match_date=row['Date'],
            league=row['League'],
            models=models,
            df=df
        )
        
        print(f"\n{pred['match']}")
        print(f"  Predicted margin: {pred['predicted_margin']:+.2f} ({pred['predicted_result']})")
        print(f"  AH Line: {pred['ah_line']}")
        print(f"  Recommendation: {pred['bet_recommendation']}")
        if pred['bet_team']:
            print(f"  Bet: {pred['bet_team']} @ {pred['odds']:.2f} (edge: {pred['edge']:.2f})")
        print(f"  AH home cover prob: {pred['ah_home_prob']:.1%}")
        
        # Show actual result
        actual = row['goal_margin']
        actual_result = 'H' if actual > 0 else ('A' if actual < 0 else 'D')
        print(f"  ACTUAL: {row['FTHG']}-{row['FTAG']} ({actual_result})")
