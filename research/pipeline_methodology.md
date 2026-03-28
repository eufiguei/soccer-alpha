# Weekend Predictions Pipeline — Methodology

## Overview
This pipeline generates live betting recommendations for upcoming European football matches (EPL, La Liga, Bundesliga, Serie A) by running them through the calibrated margin prediction model.

## Architecture

### Step 1: Fixture Discovery
**Goal:** Identify upcoming weekend matches across major leagues

**Method:**
- Attempt live fetch via ESPN API for each league (ENG.1, ESP.1, GER.1, ITA.1)
- Filter for SCHEDULED or POSTPONED status matches
- Fallback to demo matches if no real fixtures available

**Output:** List of upcoming matches with team names and dates

### Step 2: Model & Calibration Loading
**Assets:**
- `models/calibrated_margin_model.pkl` — LGBMRegressor (9 features)
- `models/overall_calibration.json` — Calibration lookup table (8 bins)
- `data/real_ah_bettable.parquet` — Historical match data (8,408 matches)

**Features Required:**
```
['home_gf_6', 'home_ga_6', 'home_sot_6',    # Home team L6 avg
 'away_gf_6', 'away_ga_6', 'away_sot_6',    # Away team L6 avg
 'home_implied', 'away_implied',             # Implied win probability
 'real_ah_line']                             # Asian Handicap line
```

### Step 3: Team Rolling Stats (Last 6 Matches)
**Computation:**
- For each team, extract last 6 home matches: GF (goals for), GA (goals against), SOT (shots on target)
- Extract last 6 away matches: GF, GA, SOT
- Compute rolling averages
- Default for unknown teams: home_gf_6=1.5, home_ga_6=1.2, etc.

**Example (Arsenal Home L6):**
- GF: 1.67/game
- GA: 1.17/game  
- SOT: 4.33/game

### Step 4: Implied Probability Estimation
**Method:** Estimate from AH line using empirical mapping
- AH line < -0.5: home_prob = 55%, away_prob = 45%
- AH line > +0.5: home_prob = 45%, away_prob = 55%
- Otherwise: 50/50

**League Defaults (AH line):**
- EPL: -0.75 (home favorite)
- La Liga: -0.50
- Bundesliga: -1.00 (strong home advantage)
- Serie A: -0.50

### Step 5: Prediction
**Process:**
1. Build feature vector from team stats + implied probs + AH line
2. Run through model: `pred_margin = model.predict(features)[0]`
3. Apply calibration lookup
4. Compute AH edge: `edge = calibrated_margin - ah_line`

**Decision Rules:**
- `edge > +0.4`: BET HOME (HIGH if edge > 0.8, else MEDIUM)
- `edge < -0.4`: BET AWAY (HIGH if edge < -0.8, else MEDIUM)
- `else`: PASS (no edge, LOW confidence)

### Step 6: Calibration
**Purpose:** Correct model systematic bias across prediction bins

**Process:**
- Model outputs raw prediction: e.g., +0.980 goals
- Look up bin: 0.5 to 1.0 → mean_actual = 0.602 goals
- Use calibrated value for edge calculation

**Example Bins:**
```json
{
  "<-2": -1.878,
  "-2to-1": -1.075,
  "-1to-0.5": -0.767,
  "0to0.5": 0.405,
  "0.5to1": 0.602,
  "1to2": 1.330,
  ">2": 1.860
}
```

## Outputs

### weekend_predictions.json
Structured data for programmatic consumption:
```json
{
  "match": "Arsenal vs Chelsea",
  "league": "EPL",
  "date": "2026-03-28",
  "predicted_margin": 0.980,
  "calibrated_margin": 0.602,
  "ah_line": -0.75,
  "ah_edge": 1.352,
  "bet_recommendation": "BET HOME (Arsenal -0.75)",
  "confidence": "HIGH",
  "home_form": {
    "gf_6": 1.67,
    "ga_6": 1.17,
    "sot_6": 4.33
  },
  "away_form": {
    "gf_6": 0.83,
    "ga_6": 1.83,
    "sot_6": 4.0
  }
}
```

### weekend_predictions.md
Human-readable betting recommendations with form analysis.

## Limitations & Caveats

### Model Degradation (2024-25 Season)
- **Issue:** Model win rate dropped to 50.5% in recent fold
- **Cause:** Likely seasonal drift in team form, injury impacts, tactical evolution
- **Impact:** All predictions should be treated as lower-confidence
- **Mitigation:** Compare predictions against outcomes to validate

### Missing Real Fixtures
- Demo matches used if ESPN API unavailable (e.g., off-season, scheduled downtime)
- Predictions on demo data are illustrative only

### Feature Limitations
- Rolling stats (L6) may be insufficient for mid-season injuries
- Implied probabilities are estimated, not from live odds
- AH line defaults are league-based, not match-specific

### Data Freshness
- Model trained on historical data through May 2025
- No accommodation for January winter break in some leagues
- Transfer window effects not explicitly modeled

## Integration Points

### Real Fixtures (Future)
```python
# When ESPN API returns live data:
for match in upcoming_matches:
    home_odds = get_market_odds(match)
    ah_line = get_ah_line_from_market(match)
    # Use actual odds instead of estimates
```

### Live Odds Integration
- Fetch Betfair/Pinnacle odds for implied probabilities
- Use match-specific AH lines instead of league defaults
- Validate edge against overround

### Backtest Against Outcomes
- Store predictions with match IDs
- Post-match: record actual goals, compare to predictions
- Track win rate over time for degradation detection

## Running the Pipeline

```bash
cd /root/.openclaw/workspace/projects/soccer-alpha
python3 predict_weekend.py
```

**Output files:**
- `research/weekend_predictions.json`
- `research/weekend_predictions.md`
- `research/pipeline_summary.md`

## Next Steps

1. **Validate degradation thesis** — Compare model WR vs actual outcomes
2. **Integrate live odds** — Replace estimated AH lines with market data
3. **Add head-to-head** — Include historical H2H stats in features
4. **Monitor seasonal drift** — Track feature importance over time
5. **Retrain on 2025+ data** — Update model with latest season information

---

**Last Updated:** 2026-03-28  
**Status:** ✓ Pipeline Operational
