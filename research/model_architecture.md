# Soccer Alpha - Model Architecture

## Overview

This system predicts **goal margins** for soccer matches and translates those predictions into **Asian Handicap betting decisions**.

## Data Pipeline

### Input Data
- Source: football-data.co.uk
- 8,408 matches from 2019-2025
- 5 leagues: EPL, LaLiga, Bundesliga, Serie A, Ligue 1
- Features: shots, shots on target, corners, fouls, goals, and betting odds

### Feature Engineering

For each match, we compute rolling statistics (last 6 matches) **separately by venue**:

**Home Team Features (from home matches only):**
- `home_goals_scored_h6` - goals per game at home
- `home_goals_conceded_h6` - goals conceded at home
- `home_shots_on_target_h6` - SOT per game at home
- `home_xg_h6` - xG proxy (SOT × 0.33)

**Away Team Features (from away matches only):**
- `away_goals_scored_a6` - goals per game away
- `away_goals_conceded_a6` - goals conceded away
- `away_shots_on_target_a6` - SOT per game away
- `away_xg_a6` - xG proxy

**Form Features (all venues):**
- `home_form_pts` - points per game last 6
- `away_form_pts` - points per game last 6

**Head-to-Head:**
- `h2h_avg_home_goals` - avg home team goals in last 5 H2H
- `h2h_avg_away_goals` - avg away team goals in last 5 H2H
- `h2h_home_wins` - home win rate in H2H

**Market Signals:**
- `market_home_implied` - 1 / home odds
- `market_away_implied` - 1 / away odds
- `market_draw_implied` - 1 / draw odds

**Season Context:**
- `match_week` - week of season (1-38)
- `is_early_season` - first 8 matches flag
- `is_late_season` - last 8 matches flag

## Models

### Model A: Goal Margin Regressor
- **Target:** `goal_margin = FTHG - FTAG`
- **Algorithm:** LightGBM Regressor
- **Features:** 21 features (no AH line)

Parameters:
```python
n_estimators=300
learning_rate=0.05
max_depth=5
num_leaves=31
```

### Model B: AH Cover Classifier
- **Target:** `ah_home_covers` (binary: did home cover the line?)
- **Algorithm:** LightGBM Classifier
- **Features:** 22 features (includes `market_ah_line`)

## Betting Decision Logic

```python
def decide_bet(predicted_margin, ah_line, min_edge=0.5):
    home_edge = predicted_margin + ah_line
    
    if home_edge > min_edge:
        return 'BET_HOME'
    elif -home_edge > min_edge:
        return 'BET_AWAY'
    else:
        return 'SKIP'
```

**Example:**
- AH line: -0.75 (home favorite, must win by 1+)
- Predicted margin: +1.3 (model thinks home wins by 1.3)
- Edge calculation: 1.3 + (-0.75) = 0.55 > 0.5 → BET_HOME

## Validation

- **Train:** 2019-2023 (5,771 matches)
- **Test:** 2024-2025 (2,068 matches)
- Strictly time-based split (no future leakage)

## Files

- `models/margin_predictor.pkl` - Goal margin model
- `models/ah_cover_classifier.pkl` - AH cover model
- `data/features_engineered.parquet` - Processed features
- `backtests/per_match_backtest.csv` - All bet decisions
