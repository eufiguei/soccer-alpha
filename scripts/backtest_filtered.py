"""
Backtest with optimized filters: AWAY-only, exclude LaLiga
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from backtest import run_backtest, calculate_ah_result, calculate_profit


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
    
    print("="*60)
    print("FILTERED BACKTEST: AWAY-only, exclude LaLiga")
    print("="*60)
    
    # Run backtest
    results_df = run_backtest(test_df, model_margin, feature_cols, min_edge=0.5)
    
    # Apply filters
    filtered = results_df[
        (results_df['decision'] == 'BET_AWAY') & 
        (results_df['League'] != 'LaLiga')
    ].copy()
    
    print(f"\nFiltered bets: {len(filtered)} out of {len(results_df)} matches")
    
    # Results
    wins = filtered[filtered['bet_result'] >= 0.5]
    win_rate = len(wins) / len(filtered) * 100 if len(filtered) > 0 else 0
    total_profit = filtered['profit'].sum()
    roi = total_profit / len(filtered) * 100 if len(filtered) > 0 else 0
    
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Total profit: {total_profit:.2f} units")
    print(f"ROI: {roi:.2f}%")
    
    # By league
    print("\n=== BY LEAGUE ===")
    for league in filtered['League'].unique():
        league_bets = filtered[filtered['League'] == league]
        if len(league_bets) > 0:
            league_profit = league_bets['profit'].sum()
            league_roi = league_profit / len(league_bets) * 100
            league_wr = len(league_bets[league_bets['bet_result'] >= 0.5]) / len(league_bets) * 100
            print(f"{league}: {len(league_bets)} bets, WR: {league_wr:.1f}%, ROI: {league_roi:.2f}%")
    
    # Monthly breakdown
    print("\n=== MONTHLY PERFORMANCE ===")
    filtered['month'] = filtered['Date'].dt.to_period('M')
    monthly = filtered.groupby('month').agg({
        'profit': ['sum', 'count'],
        'bet_result': lambda x: (x >= 0.5).mean() * 100
    })
    monthly.columns = ['profit', 'bets', 'win_rate']
    monthly['roi'] = monthly['profit'] / monthly['bets'] * 100
    print(monthly.to_string())
    
    # Cumulative curve
    print("\n=== CUMULATIVE PROFIT ===")
    filtered = filtered.sort_values('Date')
    filtered['cum_profit'] = filtered['profit'].cumsum()
    filtered['bet_num'] = range(1, len(filtered) + 1)
    
    checkpoints = [25, 50, 75, 100, 125, 150, 175]
    for cp in checkpoints:
        if len(filtered) >= cp:
            row = filtered.iloc[cp-1]
            print(f"After {cp} bets: {row['cum_profit']:.2f} units ({row['cum_profit']/cp*100:.1f}% ROI)")
    
    # Save filtered results
    output_path = Path(__file__).parent.parent / 'backtests' / 'filtered_away_only.csv'
    filtered.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
