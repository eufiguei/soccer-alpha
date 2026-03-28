# Soccer Alpha - Asian Handicap Prediction System

A per-game goal margin prediction system that generates Asian Handicap betting recommendations.

## Quick Start

```bash
# Rebuild features from raw data
python3 scripts/build_features.py

# Train models
python3 scripts/train_models.py

# Run backtest
python3 scripts/backtest.py

# Run filtered backtest (production strategy)
python3 scripts/backtest_filtered.py

# Predict upcoming match
python3 scripts/predict.py
```

## Results Summary

### Production Strategy
- **Filter:** AWAY bets only, exclude LaLiga, min_edge=0.5
- **Backtest (2024-2025):** 169 bets, 67.5% win rate, **+27.85% ROI**

### By League Performance
| League | Bets | Win Rate | ROI |
|--------|------|----------|-----|
| Ligue 1 | 35 | 77.1% | +47.1% |
| EPL | 47 | 70.2% | +32.8% |
| Bundesliga | 43 | 67.4% | +24.2% |
| Serie A | 44 | 56.8% | +10.9% |

## Architecture

1. **Feature Engineering** - Rolling team stats (home/away split), form, H2H, market signals
2. **Margin Prediction** - LightGBM regressor predicts goal_margin = FTHG - FTAG
3. **AH Classification** - LightGBM classifier predicts P(home covers AH line)
4. **Decision Logic** - Bet when predicted_margin + ah_line > 0.5 (edge threshold)

## Key Findings

- **BET_AWAY is profitable (+20% ROI), BET_HOME loses money (-13% ROI)**
- Market overvalues home advantage → away underdogs are mispriced
- Ligue 1 and Bundesliga show strongest signals
- LaLiga predictions consistently fail (exclude from strategy)

## Files

```
models/
  margin_predictor.pkl   # Goal margin regressor
  ah_cover_classifier.pkl # AH cover probability
  importance_*.csv       # Feature importance

data/
  features_engineered.parquet  # Processed features

backtests/
  per_match_backtest.csv      # All bet decisions
  filtered_away_only.csv      # Production strategy results

research/
  model_architecture.md       # Technical documentation
  per_game_findings.md        # Research findings
```

## Next Steps

1. Deploy AWAY-only strategy for live betting
2. Add xG data from external source (FBref/Understat)
3. Consider adding player availability features
4. Monitor for strategy decay
