# Weekend Predictions (FIXED) — 2026-03-28

## ✅ Calibration Bug Fixed
The previous predictions showed "+0.602" for all matches because the calibration
was returning `mean_actual` from the bucket (constant) instead of `predicted + bias` (varies per match).

**V2 Model:** Uses Elo ratings + extended rolling features.
**Calibration:** predicted_margin + bias (correct, varies per match)

---

## Arsenal vs Chelsea (AH -0.75)

- **Elo diff:** +71 (home vs away)
- **Raw margin:** +1.016 goals
- **Calibrated margin:** +1.002 goals  *(different from other matches ✅)*
- **AH edge:** +0.252
- **Recommendation:** SKIP

---

## Real Madrid vs Barcelona (AH -0.50)

- **Elo diff:** -28 (home vs away)
- **Raw margin:** +0.244 goals
- **Calibrated margin:** +0.240 goals  *(different from other matches ✅)*
- **AH edge:** -0.260
- **Recommendation:** SKIP

---

## Bayern Munich vs Dortmund (AH -1.00)

- **Elo diff:** +136 (home vs away)
- **Raw margin:** +1.573 goals
- **Calibrated margin:** +1.559 goals  *(different from other matches ✅)*
- **AH edge:** +0.559
- **Recommendation:** BET_HOME

---

## Man United vs Everton (AH -0.75)

- **Elo diff:** -118 (home vs away)
- **Raw margin:** +0.703 goals
- **Calibrated margin:** +0.563 goals  *(different from other matches ✅)*
- **AH edge:** -0.187
- **Recommendation:** SKIP

---

## Napoli vs Juventus (AH -0.50)

- **Elo diff:** +12 (home vs away)
- **Raw margin:** +0.636 goals
- **Calibrated margin:** +0.496 goals  *(different from other matches ✅)*
- **AH edge:** -0.004
- **Recommendation:** SKIP

---

## Inter vs Roma (AH -0.50)

- **Elo diff:** +95 (home vs away)
- **Raw margin:** +0.088 goals
- **Calibrated margin:** +0.083 goals  *(different from other matches ✅)*
- **AH edge:** -0.417
- **Recommendation:** BET_AWAY

---

## Liverpool vs Manchester City (AH -0.75)

- **Elo diff:** +202 (home vs away)
- **Raw margin:** +0.251 goals
- **Calibrated margin:** +0.247 goals  *(different from other matches ✅)*
- **AH edge:** -0.503
- **Recommendation:** BET_AWAY

---

## Atletico Madrid vs Sevilla (AH -0.50)

- **Elo diff:** +79 (home vs away)
- **Raw margin:** +0.336 goals
- **Calibrated margin:** +0.331 goals  *(different from other matches ✅)*
- **AH edge:** -0.169
- **Recommendation:** SKIP

---

## V2 Model Stats

| Fold | WR | N Bets |
|------|----|--------|
| 2022-2023 | 0.511 | 139 |
| 2023-2024 | 0.570 | 121 |
| 2024-2025 | 0.545 | 110 |

**2024-25 WR: 0.545** (✅ >= 54%)
