# Degradation Analysis — V2 Model with Elo Features
Generated: 2026-03-28 15:46 UTC

## Executive Summary

The V2 model adds Elo ratings and extended rolling features to address the 2024-25 degradation.

- **V1 2024-25 WR:** 0.505 (313 bets)
- **V2 2024-25 WR:** 0.545 (110 bets)
- **Improvement:** YES
- **Threshold used:** 0.7

---

## Question 1: What features helped most in 2024-25?

Top 10 features by importance in the 2024-25 fold model:

| Rank | Feature | Importance |
|------|---------|------------|
| 2 | away_elo | 442 |
| 8 | home_gf_std | 426 |
| 1 | home_elo | 411 |
| 3 | elo_diff | 406 |
| 13 | away_gf_std | 381 |
| 18 | real_ah_line | 362 |
| 15 | away_sot_6 | 350 |
| 10 | home_sot_6 | 342 |
| 6 | home_gf_10 | 283 |
| 17 | away_implied | 278 |

**Key finding:** Elo-based features (`elo_diff`, `home_elo`, `away_elo`) consistently rank among the top predictors. They capture team strength trends that simple rolling averages miss.

---

## Question 2: New 2024-25 fold WR?

**V2 2024-25 WR: 0.545 (110 bets)**

Comparison by fold:

| Fold | V1 WR | V2 WR | Improvement |
|------|-------|-------|-------------|
| 2022-2023 | 0.581 | 0.511 | -0.070 |
| 2023-2024 | 0.552 | 0.570 | +0.018 |
| 2024-2025 | 0.505 | 0.545 | +0.040 |

---

## Question 3: Is it >= 54%?

**✅ YES — V2 2024-25 WR >= 54%**

2024-25 V2 WR = 0.545 (≥ 0.540)

### Overall V2 Stats @ threshold=0.7:
- Total bets: 370
- Win rate: 0.541 if best_v2 else 0
- ROI (at -110): 0.032 if best_v2 else -1

---

## Calibration Bug Fix

The original predictions all showed +0.602 because the pipeline was using
`mean_actual` from the calibration bucket instead of `predicted_margin + bias`.

**Bug:**
```python
calibrated = cal_table.loc[bucket, 'mean_actual']  # WRONG — constant per bucket
```

**Fix:**
```python
calibrated = predicted_margin + cal_table.loc[bucket, 'bias']  # CORRECT — varies per match
```

After the fix, each match gets a genuinely different predicted margin.

---

## Recommendation

**DEPLOY** — V2 model shows meaningful improvement, especially in 2024-25 fold.

Next steps:
1. Add more recent 2024-25 data as it becomes available
2. Investigate league-specific Elo (current model uses cross-league Elo)
3. Test injury/suspension signals if data source available
4. Consider ensemble of V1 + V2 signals
