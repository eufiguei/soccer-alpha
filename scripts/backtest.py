"""
Step 5 & 6: Betting decision logic and backtesting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle


def decide_bet(predicted_margin, ah_line, min_edge=0.4):
    """
    Make betting decision based on predicted margin vs AH line.
    
    AH line interpretation:
    - Negative (e.g., -0.75): Home is favorite, must win by more than line
    - Positive (e.g., +0.75): Home is underdog, can lose by less than line
    
    Returns: 'BET_HOME', 'BET_AWAY', or 'SKIP'
    """
    if pd.isna(predicted_margin) or pd.isna(ah_line):
        return 'SKIP'
    
    # The AH line is added to home team's score
    # If line is -0.75, home needs actual_margin + (-0.75) > 0, so actual_margin > 0.75
    # Our "edge" is predicted_margin - required_margin
    
    # For home to cover: predicted_margin + ah_line > 0
    # Edge for home = predicted_margin + ah_line
    home_edge = predicted_margin + ah_line
    
    # For away to cover: predicted_margin + ah_line < 0
    # Edge for away = -(predicted_margin + ah_line) = -home_edge
    away_edge = -home_edge
    
    if home_edge > min_edge:
        return 'BET_HOME'
    elif away_edge > min_edge:
        return 'BET_AWAY'
    else:
        return 'SKIP'


def calculate_ah_result(actual_margin, ah_line, bet_side):
    """
    Calculate AH bet result.
    
    Returns: win (1.0), loss (0.0), half-win (0.75), half-loss (0.25), push (0.5)
    """
    if pd.isna(actual_margin) or pd.isna(ah_line):
        return np.nan
    
    # Adjusted margin for home (what home "won" by after AH)
    adjusted = actual_margin + ah_line
    
    if bet_side == 'BET_HOME':
        if adjusted > 0.5:  # Clear win
            return 1.0
        elif adjusted == 0.5:  # Half win
            return 0.75
        elif adjusted == 0:  # Push
            return 0.5
        elif adjusted == -0.5:  # Half loss
            return 0.25
        else:  # Clear loss
            return 0.0
    elif bet_side == 'BET_AWAY':
        if adjusted < -0.5:  # Clear win
            return 1.0
        elif adjusted == -0.5:  # Half win
            return 0.75
        elif adjusted == 0:  # Push
            return 0.5
        elif adjusted == 0.5:  # Half loss
            return 0.25
        else:  # Clear loss
            return 0.0
    else:
        return np.nan


def calculate_profit(result, odds):
    """Calculate profit from bet result."""
    if pd.isna(result) or pd.isna(odds):
        return 0
    
    stake = 1.0  # Unit stake
    
    if result == 1.0:  # Full win
        return stake * (odds - 1)
    elif result == 0.75:  # Half win
        return stake * 0.5 * (odds - 1)
    elif result == 0.5:  # Push
        return 0
    elif result == 0.25:  # Half loss
        return -stake * 0.5
    else:  # Full loss
        return -stake


def run_backtest(test_df, model_margin, feature_cols, min_edge=0.4):
    """Run backtest on test data."""
    
    results = []
    
    for idx, row in test_df.iterrows():
        # Prepare features
        features = row[feature_cols].copy()
        for col in feature_cols:
            if pd.isna(features[col]):
                features[col] = test_df[col].median()
        
        # Predict margin
        X = pd.DataFrame([features])
        predicted_margin = model_margin.predict(X)[0]
        
        # Get AH line and odds
        ah_line = row['market_ah_line']
        home_odds = row['ah_home_odds']
        away_odds = row['ah_away_odds']
        
        # Make decision
        decision = decide_bet(predicted_margin, ah_line, min_edge=min_edge)
        
        # Calculate actual result
        actual_margin = row['goal_margin']
        
        if decision == 'BET_HOME':
            bet_result = calculate_ah_result(actual_margin, ah_line, 'BET_HOME')
            profit = calculate_profit(bet_result, home_odds)
            bet_odds = home_odds
        elif decision == 'BET_AWAY':
            bet_result = calculate_ah_result(actual_margin, ah_line, 'BET_AWAY')
            profit = calculate_profit(bet_result, away_odds)
            bet_odds = away_odds
        else:
            bet_result = np.nan
            profit = 0
            bet_odds = np.nan
        
        results.append({
            'Date': row['Date'],
            'HomeTeam': row['HomeTeam'],
            'AwayTeam': row['AwayTeam'],
            'League': row['League'],
            'predicted_margin': predicted_margin,
            'actual_margin': actual_margin,
            'ah_line': ah_line,
            'decision': decision,
            'bet_odds': bet_odds,
            'bet_result': bet_result,
            'profit': profit,
            'FTHG': row['FTHG'],
            'FTAG': row['FTAG'],
        })
    
    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze backtest results."""
    
    # Filter to actual bets
    bets = results_df[results_df['decision'] != 'SKIP']
    
    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Total matches: {len(results_df)}")
    print(f"Bets placed: {len(bets)} ({len(bets)/len(results_df)*100:.1f}% of matches)")
    
    if len(bets) == 0:
        print("No bets placed!")
        return
    
    # Win rate
    wins = bets[bets['bet_result'] >= 0.5]
    full_wins = bets[bets['bet_result'] == 1.0]
    losses = bets[bets['bet_result'] < 0.5]
    
    print(f"\nWin rate: {len(wins)/len(bets)*100:.1f}%")
    print(f"Full wins: {len(full_wins)}, Half wins: {len(bets[bets['bet_result'] == 0.75])}, Pushes: {len(bets[bets['bet_result'] == 0.5])}")
    print(f"Half losses: {len(bets[bets['bet_result'] == 0.25])}, Full losses: {len(bets[bets['bet_result'] == 0])}")
    
    # ROI
    total_profit = bets['profit'].sum()
    total_staked = len(bets)
    roi = total_profit / total_staked * 100
    
    print(f"\nTotal profit: {total_profit:.2f} units")
    print(f"Total staked: {total_staked} units")
    print(f"ROI: {roi:.2f}%")
    
    # By league
    print("\n=== BY LEAGUE ===")
    for league in bets['League'].unique():
        league_bets = bets[bets['League'] == league]
        league_profit = league_bets['profit'].sum()
        league_roi = league_profit / len(league_bets) * 100 if len(league_bets) > 0 else 0
        league_wr = len(league_bets[league_bets['bet_result'] >= 0.5]) / len(league_bets) * 100 if len(league_bets) > 0 else 0
        print(f"{league}: {len(league_bets)} bets, WR: {league_wr:.1f}%, ROI: {league_roi:.2f}%")
    
    # By AH line
    print("\n=== BY AH LINE ===")
    bets['ah_line_group'] = pd.cut(bets['ah_line'], bins=[-3, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 3], labels=['<-1.5', '-1.5 to -0.75', '-0.75 to -0.25', '-0.25 to 0.25', '0.25 to 0.75', '0.75 to 1.5', '>1.5'])
    for group in bets['ah_line_group'].dropna().unique():
        group_bets = bets[bets['ah_line_group'] == group]
        if len(group_bets) > 10:
            group_profit = group_bets['profit'].sum()
            group_roi = group_profit / len(group_bets) * 100
            group_wr = len(group_bets[group_bets['bet_result'] >= 0.5]) / len(group_bets) * 100
            print(f"AH {group}: {len(group_bets)} bets, WR: {group_wr:.1f}%, ROI: {group_roi:.2f}%")
    
    # By bet side
    print("\n=== BY BET SIDE ===")
    for side in ['BET_HOME', 'BET_AWAY']:
        side_bets = bets[bets['decision'] == side]
        if len(side_bets) > 0:
            side_profit = side_bets['profit'].sum()
            side_roi = side_profit / len(side_bets) * 100
            side_wr = len(side_bets[side_bets['bet_result'] >= 0.5]) / len(side_bets) * 100
            print(f"{side}: {len(side_bets)} bets, WR: {side_wr:.1f}%, ROI: {side_roi:.2f}%")
    
    return {
        'total_bets': len(bets),
        'win_rate': len(wins)/len(bets)*100 if len(bets) > 0 else 0,
        'roi': roi,
        'total_profit': total_profit
    }


if __name__ == "__main__":
    # Load features
    data_path = Path(__file__).parent.parent / 'data' / 'features_engineered.parquet'
    df = pd.read_parquet(data_path)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'margin_predictor.pkl'
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model_margin = model_data['model']
    feature_cols = model_data['features']
    
    # Time-based split
    train_cutoff = pd.Timestamp('2024-01-01')
    test_df = df[df['Date'] >= train_cutoff].copy()
    
    print(f"Test set: {len(test_df)} matches")
    
    # Test different edge thresholds
    print("\n" + "="*60)
    print("TESTING DIFFERENT EDGE THRESHOLDS")
    print("="*60)
    
    for min_edge in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"\n{'='*60}")
        print(f"MIN EDGE = {min_edge}")
        print(f"{'='*60}")
        
        results_df = run_backtest(test_df, model_margin, feature_cols, min_edge=min_edge)
        stats = analyze_results(results_df)
    
    # Save best results (0.5 edge)
    print("\n\nSaving results with min_edge=0.5...")
    results_df = run_backtest(test_df, model_margin, feature_cols, min_edge=0.5)
    
    output_path = Path(__file__).parent.parent / 'backtests' / 'per_match_backtest.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
