#!/usr/bin/env python3
"""
Soccer Alpha CEO Heartbeat - Autonomous research loop.
Runs every 30min. Reads state, identifies gaps, spawns builders, never stops.
"""
import json, subprocess, sys, os
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path("/root/.openclaw/workspace/projects/soccer-alpha")
LOG = PROJECT / "logs" / "heartbeat.log"
STATE = PROJECT / "memory" / "research_state.json"
PRDS = Path("/root/.openclaw/workspace/memory/prds")

def log(msg):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def load_json(path, default={}):
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except: pass
    return default

def save_json(path, data):
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def check_per_fold_wr():
    """Read calibration CSV and compute per-fold WR honestly."""
    cal = PROJECT / "backtests" / "calibration_validation.csv"
    if not cal.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(cal)
        if 'fold' not in df.columns or 'cal_margin_A' not in df.columns:
            return None
        
        results = {}
        for fold, grp in df.groupby('fold'):
            # Apply AH decision logic
            bets = []
            for _, row in grp.iterrows():
                cal_m = row.get('cal_margin_A', 0)
                ah = row.get('real_ah_line', 0)
                ah_result = row.get('real_ah_result', 0.5)
                if pd.isna(cal_m) or pd.isna(ah): continue
                
                # Decision
                decision = 'SKIP'
                threshold = 0.4
                if ah < 0:
                    req = abs(ah)
                    if cal_m > req + threshold: decision = 'BET_HOME'
                    elif cal_m < req - threshold: decision = 'BET_AWAY'
                elif ah > 0:
                    req = -ah
                    if cal_m < req - threshold: decision = 'BET_AWAY'
                    elif cal_m > req + threshold: decision = 'BET_HOME'
                
                if decision == 'SKIP': continue
                if ah_result == 0.5: continue  # push
                
                won = (decision == 'BET_HOME' and ah_result == 1.0) or \
                      (decision == 'BET_AWAY' and ah_result == 0.0)
                bets.append(won)
            
            if bets:
                wr = sum(bets)/len(bets)
                results[fold] = {'wr': round(wr, 3), 'n': len(bets)}
        
        return results
    except Exception as e:
        log(f"  Per-fold check error: {e}")
        return None

def write_prd(name, content):
    prd_path = PRDS / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}_{name}.md"
    PRDS.mkdir(exist_ok=True)
    with open(prd_path, "w") as f:
        f.write(content)
    log(f"  PRD written: {prd_path.name}")
    return prd_path

def main():
    log("=" * 60)
    log("⚽ Soccer Alpha Heartbeat — Autonomous CEO Loop")
    
    state = load_json(STATE, {"phase": "research", "issues": []})
    issues = []

    # 1. Check per-fold WR (the key metric)
    log("1. Checking per-fold WR...")
    fold_results = check_per_fold_wr()
    if fold_results:
        for fold, res in fold_results.items():
            wr = res['wr']
            n = res['n']
            status = "✅" if wr >= 0.55 else ("⚠️" if wr >= 0.52 else "❌")
            log(f"   {status} {fold}: WR={wr:.1%} n={n}")
            if wr < 0.52:
                issues.append(f"DEGRADATION: {fold} WR={wr:.1%} below threshold")
        state['fold_results'] = fold_results
    else:
        log("   Calibration data not ready yet")

    # 2. Check if v2 model exists (degradation fix)
    v2_exists = (PROJECT / "models" / "margin_predictor_v2.pkl").exists()
    v2_report = (PROJECT / "research" / "degradation_analysis.md").exists()
    log(f"2. V2 model: {'✅' if v2_exists else '❌ NOT YET'} | Report: {'✅' if v2_report else '❌ NOT YET'}")
    
    if not v2_exists and not v2_report:
        issues.append("V2 research not complete — degradation unresolved")

    # 3. Check calibration bug (all same output)
    cal = PROJECT / "backtests" / "calibration_validation.csv"
    if cal.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cal)
            if 'cal_margin_A' in df.columns:
                std = df['cal_margin_A'].std()
                if std < 0.01:
                    log(f"3. ❌ CALIBRATION BUG: all outputs same (std={std:.4f})")
                    issues.append("CALIBRATION BUG: zero variance in cal_margin_A")
                else:
                    log(f"3. ✅ Calibration variance OK (std={std:.3f})")
        except: pass

    # 4. Check if weekend predictions have the bug
    wp = PROJECT / "research" / "weekend_predictions.md"
    if wp.exists():
        content = wp.read_text()
        if content.count("+0.602") > 3:
            log("4. ❌ PREDICTION BUG: repeated +0.602 values — pipeline broken")
            issues.append("PREDICTION PIPELINE BUG: calibration not varying per match")
        else:
            log("4. ✅ Predictions look varied")

    # 5. Check GitHub sync
    try:
        result = subprocess.run(['git', '-C', str(PROJECT), 'status', '--short'], 
                               capture_output=True, text=True, timeout=10)
        uncommitted = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        log(f"5. Git: {uncommitted} uncommitted files")
        if uncommitted > 10:
            issues.append(f"GIT BEHIND: {uncommitted} files not committed")
    except: pass

    # 6. Spawn fixes for identified issues
    if issues:
        log(f"\n🔧 {len(issues)} issues found — writing PRDs and spawning fixes:")
        for issue in issues:
            log(f"   → {issue}")
        
        # Write PRD for the most critical issue
        if any("CALIBRATION BUG" in i or "PREDICTION PIPELINE" in i for i in issues):
            prd = write_prd("fix_calibration_pipeline", f"""# PRD: Fix Calibration Pipeline Bug

## Problem
The calibration pipeline outputs the same value (+0.602) for all matches.
This means the per-match features are not being used correctly.

## Fix needed
In scripts/predict.py or wherever calibration is applied:
- The calibrated margin must vary per match based on actual features
- Debug: print predicted_margin, bucket, bias, cal_margin for 5 sample matches
- Root cause: likely the bucket lookup is always finding the same bucket

## Acceptance criteria
- cal_margin_A std > 0.3 in calibration_validation.csv
- Weekend predictions show different margins for different matches
- Rerun: python3 scripts/run_calibration.py (or equivalent)
- Save fixed backtests/calibration_validation_v2.csv

## Project: /root/.openclaw/workspace/projects/soccer-alpha/
""")

        if any("DEGRADATION" in i or "V2 research" in i for i in issues):
            prd = write_prd("soccer_elo_features", f"""# PRD: Add Elo + Variance Features to Soccer Model

## Problem  
2024-25 fold shows 50.5% WR — degradation unexplained.
Need to add features that capture market efficiency improvements.

## New features to add
1. Elo rating per team (updates after each match, zero leakage)
2. Goals variance last 6 (std, not just mean)
3. xG over/underperformance trend
4. AH line movement (B365 vs market average)
5. Longer form window (10 matches)

## Implementation
- File: scripts/build_features_v2.py
- Reuse exact same walk-forward structure
- Save: data/features_v2.parquet
- Train: models/margin_predictor_v2.pkl
- Backtest: backtests/v2_validation.csv
- Report: research/degradation_analysis.md

## Target
2024-25 fold WR >= 54% with new features

## Project: /root/.openclaw/workspace/projects/soccer-alpha/
""")
    else:
        log("\n✅ No critical issues found")

    # 7. Update state
    state['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
    state['issues'] = issues
    state['v2_ready'] = v2_exists
    save_json(STATE, state)

    log(f"\n📊 Summary: {len(issues)} issues | V2: {'done' if v2_exists else 'pending'}")
    log("=" * 60)

if __name__ == "__main__":
    main()
