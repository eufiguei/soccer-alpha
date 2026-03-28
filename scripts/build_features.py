"""
Step 1 & 2: Build rolling team features for goal margin prediction.
Creates per-match features based on each team's recent performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def compute_points(result, is_home):
    """Compute points from match result."""
    if result == 'H':
        return 3 if is_home else 0
    elif result == 'A':
        return 0 if is_home else 3
    else:
        return 1

def build_rolling_features(df, n_matches=6):
    """
    Build rolling features for each team before each match.
    Computes home-specific stats for home team, away-specific for away team.
    """
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize feature columns
    feature_cols = []
    
    # We'll build a dict of team stats over time
    # For each team, track separate home and away performance
    
    teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    
    # Pre-compute match history for each team
    # Track: goals_for, goals_against, shots_on_target, points
    
    # Output dataframe
    features_list = []
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = row['Date']
        league = row['League']
        
        # Get previous matches for home team (at home only)
        home_prev_home = df[(df['HomeTeam'] == home_team) & 
                            (df['Date'] < match_date) &
                            (df['League'] == league)].tail(n_matches)
        
        # Get previous matches for away team (away only)
        away_prev_away = df[(df['AwayTeam'] == away_team) & 
                            (df['Date'] < match_date) &
                            (df['League'] == league)].tail(n_matches)
        
        # Get all previous matches for form calculation
        home_all_prev = pd.concat([
            df[(df['HomeTeam'] == home_team) & (df['Date'] < match_date) & (df['League'] == league)].assign(is_home=True),
            df[(df['AwayTeam'] == home_team) & (df['Date'] < match_date) & (df['League'] == league)].assign(is_home=False)
        ]).sort_values('Date').tail(n_matches)
        
        away_all_prev = pd.concat([
            df[(df['HomeTeam'] == away_team) & (df['Date'] < match_date) & (df['League'] == league)].assign(is_home=True),
            df[(df['AwayTeam'] == away_team) & (df['Date'] < match_date) & (df['League'] == league)].assign(is_home=False)
        ]).sort_values('Date').tail(n_matches)
        
        # Skip if not enough history
        if len(home_prev_home) < 3 or len(away_prev_away) < 3:
            continue
        
        # HOME TEAM FEATURES (from home matches only)
        home_goals_scored_h6 = home_prev_home['FTHG'].mean() if len(home_prev_home) > 0 else np.nan
        home_goals_conceded_h6 = home_prev_home['FTAG'].mean() if len(home_prev_home) > 0 else np.nan
        home_shots_on_target_h6 = home_prev_home['HST'].mean() if len(home_prev_home) > 0 and 'HST' in home_prev_home.columns else np.nan
        home_shots_h6 = home_prev_home['HS'].mean() if len(home_prev_home) > 0 and 'HS' in home_prev_home.columns else np.nan
        home_xg_h6 = home_shots_on_target_h6 * 0.33 if not pd.isna(home_shots_on_target_h6) else np.nan
        
        # AWAY TEAM FEATURES (from away matches only)
        away_goals_scored_a6 = away_prev_away['FTAG'].mean() if len(away_prev_away) > 0 else np.nan
        away_goals_conceded_a6 = away_prev_away['FTHG'].mean() if len(away_prev_away) > 0 else np.nan
        away_shots_on_target_a6 = away_prev_away['AST'].mean() if len(away_prev_away) > 0 and 'AST' in away_prev_away.columns else np.nan
        away_shots_a6 = away_prev_away['AS'].mean() if len(away_prev_away) > 0 and 'AS' in away_prev_away.columns else np.nan
        away_xg_a6 = away_shots_on_target_a6 * 0.33 if not pd.isna(away_shots_on_target_a6) else np.nan
        
        # FORM (all venues, last 6)
        home_form_pts = 0
        for _, m in home_all_prev.iterrows():
            if m.get('is_home', False):
                home_form_pts += compute_points(m['FTR'], True)
            else:
                home_form_pts += compute_points(m['FTR'], False)
        home_form_pts = home_form_pts / len(home_all_prev) if len(home_all_prev) > 0 else np.nan
        
        away_form_pts = 0
        for _, m in away_all_prev.iterrows():
            if m.get('is_home', False):
                away_form_pts += compute_points(m['FTR'], True)
            else:
                away_form_pts += compute_points(m['FTR'], False)
        away_form_pts = away_form_pts / len(away_all_prev) if len(away_all_prev) > 0 else np.nan
        
        # HEAD TO HEAD (last 5 meetings between these teams)
        h2h = pd.concat([
            df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) & (df['Date'] < match_date)],
            df[(df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team) & (df['Date'] < match_date)]
        ]).sort_values('Date').tail(5)
        
        if len(h2h) > 0:
            # Goals when home_team plays at home vs away_team
            h2h_home_goals = []
            h2h_away_goals = []
            h2h_home_wins = 0
            for _, m in h2h.iterrows():
                if m['HomeTeam'] == home_team:
                    h2h_home_goals.append(m['FTHG'])
                    h2h_away_goals.append(m['FTAG'])
                    if m['FTR'] == 'H':
                        h2h_home_wins += 1
                else:
                    h2h_home_goals.append(m['FTAG'])
                    h2h_away_goals.append(m['FTHG'])
                    if m['FTR'] == 'A':
                        h2h_home_wins += 1
            h2h_avg_home_goals = np.mean(h2h_home_goals)
            h2h_avg_away_goals = np.mean(h2h_away_goals)
            h2h_home_wins = h2h_home_wins / len(h2h)
        else:
            h2h_avg_home_goals = np.nan
            h2h_avg_away_goals = np.nan
            h2h_home_wins = np.nan
        
        # MARKET SIGNALS
        market_home_implied = 1 / row['AvgH'] if pd.notna(row.get('AvgH')) and row['AvgH'] > 0 else np.nan
        market_away_implied = 1 / row['AvgA'] if pd.notna(row.get('AvgA')) and row['AvgA'] > 0 else np.nan
        market_draw_implied = 1 / row['AvgD'] if pd.notna(row.get('AvgD')) and row['AvgD'] > 0 else np.nan
        market_ah_line = row.get('real_ah_line', row.get('AHh', np.nan))
        
        # SEASON CONTEXT
        # Approximate match week from date within season
        season = row.get('Season', '')
        try:
            season_str = str(season)[:4]
            if season_str.isdigit() and len(season_str) == 4:
                season_start_year = int(season_str)
                if season_start_year < 2000 or season_start_year > 2030:
                    season_start_year = match_date.year
            else:
                season_start_year = match_date.year
        except:
            season_start_year = match_date.year
        season_start = pd.Timestamp(f'{season_start_year}-08-01')
        days_into_season = (match_date - season_start).days
        match_week = max(1, min(38, days_into_season // 7 + 1))
        is_early_season = 1 if match_week <= 8 else 0
        is_late_season = 1 if match_week >= 30 else 0
        
        # TARGET: Goal margin
        goal_margin = row['FTHG'] - row['FTAG']
        
        # AH result
        ah_home_covers = np.nan
        if pd.notna(market_ah_line):
            adjusted_margin = goal_margin + market_ah_line  # AH line is negative for favorites
            if adjusted_margin > 0:
                ah_home_covers = 1
            elif adjusted_margin < 0:
                ah_home_covers = 0
            else:
                ah_home_covers = 0.5  # Push
        
        features_list.append({
            # Identifiers
            'Date': match_date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'League': league,
            'Season': season,
            
            # Home team attacking
            'home_goals_scored_h6': home_goals_scored_h6,
            'home_shots_on_target_h6': home_shots_on_target_h6,
            'home_shots_h6': home_shots_h6,
            'home_xg_h6': home_xg_h6,
            
            # Home team defensive
            'home_goals_conceded_h6': home_goals_conceded_h6,
            
            # Away team attacking
            'away_goals_scored_a6': away_goals_scored_a6,
            'away_shots_on_target_a6': away_shots_on_target_a6,
            'away_shots_a6': away_shots_a6,
            'away_xg_a6': away_xg_a6,
            
            # Away team defensive
            'away_goals_conceded_a6': away_goals_conceded_a6,
            
            # Form
            'home_form_pts': home_form_pts,
            'away_form_pts': away_form_pts,
            
            # Head to head
            'h2h_avg_home_goals': h2h_avg_home_goals,
            'h2h_avg_away_goals': h2h_avg_away_goals,
            'h2h_home_wins': h2h_home_wins,
            
            # Market signals
            'market_home_implied': market_home_implied,
            'market_away_implied': market_away_implied,
            'market_draw_implied': market_draw_implied,
            'market_ah_line': market_ah_line,
            
            # Season context
            'match_week': match_week,
            'is_early_season': is_early_season,
            'is_late_season': is_late_season,
            
            # Targets
            'goal_margin': goal_margin,
            'FTHG': row['FTHG'],
            'FTAG': row['FTAG'],
            'FTR': row['FTR'],
            'ah_home_covers': ah_home_covers,
            
            # Keep odds for backtesting
            'ah_home_odds': row.get('ah_home_odds', row.get('AvgAHH', np.nan)),
            'ah_away_odds': row.get('ah_away_odds', row.get('AvgAHA', np.nan)),
        })
        
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(df)} matches...")
    
    features_df = pd.DataFrame(features_list)
    return features_df


if __name__ == "__main__":
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'real_ah_bettable.parquet'
    df = pd.read_parquet(data_path)
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Loaded {len(df)} matches")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Build features
    print("\nBuilding rolling features...")
    features_df = build_rolling_features(df, n_matches=6)
    
    print(f"\nGenerated {len(features_df)} matches with features")
    print(f"Dropped {len(df) - len(features_df)} matches (insufficient history)")
    
    # Save
    output_path = Path(__file__).parent.parent / 'data' / 'features_engineered.parquet'
    features_df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Show sample
    print("\nFeature columns:")
    print(features_df.columns.tolist())
    print("\nSample row:")
    print(features_df.iloc[0].to_dict())
