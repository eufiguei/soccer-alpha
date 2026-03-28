"""
Full pipeline:
1. Backtest predict_match() on 2023-2025 data
2. Run Man United vs Everton analysis
3. Fetch ESPN weekend fixtures and generate picks
4. Save all outputs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from predict_match import (
    load_data, predict_match, get_team_profile, get_h2h,
    estimate_ah_from_odds, check_ah_result
)

PROJECT_DIR = Path(__file__).parent.parent
BACKTEST_DIR = PROJECT_DIR / 'backtests'
RESEARCH_DIR = PROJECT_DIR / 'research'
BACKTEST_DIR.mkdir(exist_ok=True)
RESEARCH_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 4: BACKTEST
# ============================================================
def run_backtest(df):
    print("\n=== RUNNING BACKTEST (2023-2025) ===")

    test_df = df[df['Date'] >= '2023-01-01'].copy()
    print(f"Test matches: {len(test_df)}")

    results = []
    for i, row in test_df.iterrows():
        pred = predict_match(
            df,
            row['HomeTeam'],
            row['AwayTeam'],
            row['real_ah_line'],
            date=row['Date']
        )

        if pred['recommendation'] == 'SKIP':
            results.append({
                'date': row['Date'],
                'home': row['HomeTeam'],
                'away': row['AwayTeam'],
                'ah_line': row['real_ah_line'],
                'recommendation': 'SKIP',
                'confidence': pred.get('confidence', 'LOW'),
                'predicted_margin': pred.get('predicted_margin', None),
                'edge': pred.get('edge', None),
                'actual_home_goals': row['FTHG'],
                'actual_away_goals': row['FTAG'],
                'actual_margin': row['FTHG'] - row['FTAG'],
                'correct': None,
            })
            continue

        correct = check_ah_result(row, pred['recommendation'])

        results.append({
            'date': row['Date'],
            'home': row['HomeTeam'],
            'away': row['AwayTeam'],
            'ah_line': row['real_ah_line'],
            'recommendation': pred['recommendation'],
            'confidence': pred['confidence'],
            'predicted_margin': pred['predicted_margin'],
            'edge': pred['edge'],
            'actual_home_goals': row['FTHG'],
            'actual_away_goals': row['FTAG'],
            'actual_margin': row['FTHG'] - row['FTAG'],
            'correct': correct,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(BACKTEST_DIR / 'per_game_backtest.csv', index=False)
    print(f"Saved backtest to backtests/per_game_backtest.csv")

    # Summary stats
    bets = results_df[results_df['recommendation'] != 'SKIP'].copy()
    bets_decided = bets[bets['correct'].notna()]

    print(f"\n--- BACKTEST SUMMARY ---")
    print(f"Total matches: {len(results_df)}")
    print(f"Bets placed: {len(bets)}")
    print(f"Pushes (excluded): {len(bets) - len(bets_decided)}")
    print(f"Resolved bets: {len(bets_decided)}")

    if len(bets_decided) > 0:
        overall_wr = bets_decided['correct'].mean()
        print(f"Overall WR: {overall_wr:.1%}")

        # By confidence
        for conf in ['HIGH', 'MEDIUM']:
            sub = bets_decided[bets_decided['confidence'] == conf]
            if len(sub) > 0:
                wr = sub['correct'].mean()
                print(f"  {conf} confidence: {len(sub)} bets, WR {wr:.1%}")

        # By AH line category
        near_even = bets_decided[bets_decided['ah_line'].between(-0.5, 0.5)]
        far_lines = bets_decided[~bets_decided['ah_line'].between(-0.5, 0.5)]
        if len(near_even) > 0:
            print(f"  Near-even lines (-0.5 to +0.5): {len(near_even)} bets, WR {near_even['correct'].mean():.1%}")
        if len(far_lines) > 0:
            print(f"  Far lines: {len(far_lines)} bets, WR {far_lines['correct'].mean():.1%}")

        # HIGH confidence + near-even (the sweet spot)
        sweet_spot = bets_decided[
            (bets_decided['confidence'] == 'HIGH') &
            (bets_decided['ah_line'].between(-0.5, 0.5))
        ]
        if len(sweet_spot) > 0:
            print(f"  HIGH conf + near-even lines: {len(sweet_spot)} bets, WR {sweet_spot['correct'].mean():.1%}")

    return results_df


# ============================================================
# STEP 5: MAN UNITED VS EVERTON ANALYSIS
# ============================================================
def analyze_man_united_everton(df):
    print("\n=== MAN UNITED vs EVERTON ANALYSIS ===")

    home_team = 'Man United'
    away_team = 'Everton'

    # Use the most recent AH line from historical data, or estimate
    recent = df[
        (df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)
    ].tail(1)

    if len(recent) > 0:
        ah_line = float(recent.iloc[0]['real_ah_line'])
        print(f"Using most recent historical AH line: {ah_line:+.2f}")
    else:
        ah_line = -0.25  # Man United slight favourite at home
        print(f"No historical matchup in data, using estimated AH: {ah_line:+.2f}")

    result = predict_match(df, home_team, away_team, ah_line)

    home_p = get_team_profile(df, home_team, as_home=True)
    away_p = get_team_profile(df, away_team, as_home=False)
    h2h = get_h2h(df, home_team, away_team)

    # Build markdown report
    lines = [
        f"# Man United vs Everton — AH Prediction Analysis",
        f"",
        f"**Date generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**AH Line used:** {ah_line:+.2f}",
        f"",
        f"---",
        f"",
        f"## 🏠 Man United (Home) — Last {home_p['n'] if home_p else 'N/A'} home matches",
    ]

    if home_p:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Record | {home_p['record']} |",
            f"| Goals Scored Avg | {home_p['goals_scored_avg']} |",
            f"| Goals Conceded Avg | {home_p['goals_conceded_avg']} |",
            f"| Goal Diff | {home_p['goal_diff']:+.2f} |",
            f"| Form Pts/Game | {home_p['form_pts_per_game']:.2f} |",
        ]
    else:
        lines.append("_Insufficient data_")

    lines += [
        f"",
        f"## ✈️ Everton (Away) — Last {away_p['n'] if away_p else 'N/A'} away matches",
    ]

    if away_p:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Record | {away_p['record']} |",
            f"| Goals Scored Avg | {away_p['goals_scored_avg']} |",
            f"| Goals Conceded Avg | {away_p['goals_conceded_avg']} |",
            f"| Goal Diff | {away_p['goal_diff']:+.2f} |",
            f"| Form Pts/Game | {away_p['form_pts_per_game']:.2f} |",
        ]
    else:
        lines.append("_Insufficient data_")

    lines += [f"", f"## ⚔️ Head-to-Head History"]

    if h2h:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Meetings | {h2h['n_meetings']} |",
            f"| Man United wins | {h2h['home_team_wins']} |",
            f"| Draws | {h2h['draws']} |",
            f"| Everton wins | {h2h['away_team_wins']} |",
            f"| Avg goals (Man United) | {h2h['avg_home_goals']} |",
            f"| Avg goals (Everton) | {h2h['avg_away_goals']} |",
            f"| H2H margin (Man United) | {h2h['predicted_margin']:+.2f} |",
        ]
        if h2h.get('last_3_results'):
            lines += [f"", f"**Last 3 meetings:**"]
            for r in h2h['last_3_results']:
                lines.append(f"- {r}")
    else:
        lines.append("_No H2H data available (< 2 meetings)_")

    lines += [
        f"",
        f"## 🎯 Prediction",
        f"",
        f"| | |",
        f"|---|---|",
        f"| **AH Line** | {result['ah_line']:+.2f} |",
        f"| **Predicted Margin** | {result['predicted_margin']:+.2f} goals |",
        f"| **Edge** | {result.get('edge', 'N/A')} |",
        f"| **Recommendation** | **{result['recommendation']}** |",
        f"| **Confidence** | {result['confidence']} |",
        f"",
        f"### Reasoning",
        f"",
        f"{result.get('reasoning', result.get('reason', 'N/A'))}",
        f"",
        f"---",
        f"_Generated by soccer-alpha predict_match.py_",
    ]

    report = "\n".join(lines)
    out_path = RESEARCH_DIR / 'man_united_everton_analysis.md'
    out_path.write_text(report)
    print(f"Saved to research/man_united_everton_analysis.md")
    print(f"\nFinal: {result['recommendation']} ({result['confidence']}) | Edge: {result.get('edge', 'N/A')}")

    return result


# ============================================================
# STEP 6: WEEKEND FIXTURES
# ============================================================
def fetch_espn_fixtures():
    """Fetch upcoming fixtures from ESPN API."""
    print("\n=== FETCHING ESPN FIXTURES ===")

    # ESPN API for soccer leagues
    leagues = {
        'EPL': 'eng.1',
        'La Liga': 'esp.1',
        'Bundesliga': 'ger.1',
        'Serie A': 'ita.1',
        'Ligue 1': 'fra.1',
    }

    all_fixtures = []

    for league_name, league_code in leagues.items():
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"  {league_name}: HTTP {resp.status_code}")
                continue

            data = resp.json()
            events = data.get('events', [])

            for event in events:
                competitions = event.get('competitions', [])
                for comp in competitions:
                    competitors = comp.get('competitors', [])
                    if len(competitors) < 2:
                        continue

                    home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                    away = next((c for c in competitors if c.get('homeAway') == 'away'), None)

                    if not home or not away:
                        continue

                    home_name = home['team']['displayName']
                    away_name = away['team']['displayName']

                    # Try to get odds
                    odds_data = comp.get('odds', [])
                    home_odds = draw_odds = away_odds = None

                    if odds_data:
                        for odd in odds_data:
                            if isinstance(odd, dict):
                                home_odds = odd.get('homeTeamOdds', {}).get('moneyLine')
                                away_odds = odd.get('awayTeamOdds', {}).get('moneyLine')

                    fixture = {
                        'league': league_name,
                        'home': home_name,
                        'away': away_name,
                        'date': event.get('date', ''),
                        'home_odds': home_odds,
                        'away_odds': away_odds,
                        'status': comp.get('status', {}).get('type', {}).get('name', ''),
                    }
                    all_fixtures.append(fixture)

            print(f"  {league_name}: {len(events)} fixtures")

        except Exception as e:
            print(f"  {league_name}: Error — {e}")

    return all_fixtures


def normalize_team_name(name, df):
    """Try to find a matching team name in our dataset."""
    teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())

    # Direct match
    if name in teams:
        return name

    # Common name mappings for ESPN → football-data.co.uk format
    mappings = {
        'Manchester United': 'Man United',
        'Manchester City': 'Man City',
        'Tottenham Hotspur': 'Tottenham',
        'Newcastle United': 'Newcastle',
        'Nottingham Forest': "Nott'm Forest",
        'Wolverhampton Wanderers': 'Wolves',
        'West Ham United': 'West Ham',
        'Leicester City': 'Leicester',
        'Brighton & Hove Albion': 'Brighton',
        'Sheffield United': 'Sheffield United',
        'Bayer 04 Leverkusen': 'Leverkusen',
        'Borussia Dortmund': 'Dortmund',
        'Borussia Mönchengladbach': "M'gladbach",
        'Eintracht Frankfurt': 'Ein Frankfurt',
        'RB Leipzig': 'RB Leipzig',
        'Internazionale': 'Inter',
        'AC Milan': 'Milan',
        'AS Roma': 'Roma',
        'Atletico Madrid': 'Ath Madrid',
        'Athletic Club': 'Ath Bilbao',
        'Real Sociedad': 'Sociedad',
        'Real Betis': 'Betis',
        'Deportivo Alavés': 'Alaves',
        'Girona FC': 'Girona',
        'Paris Saint-Germain': 'Paris SG',
        'Olympique de Marseille': 'Marseille',
        'Olympique Lyonnais': 'Lyon',
        'Stade Rennais FC': 'Rennes',
        'Stade Brestois 29': 'Brest',
    }

    if name in mappings:
        mapped = mappings[name]
        if mapped in teams:
            return mapped

    # Fuzzy: first word match
    first_word = name.split()[0].lower()
    for t in teams:
        if t.lower().startswith(first_word):
            return t

    return None  # No match


def run_weekend_picks(df, fixtures):
    print(f"\n=== GENERATING WEEKEND PICKS ({len(fixtures)} fixtures) ===")

    picks = []
    no_data = []

    for fix in fixtures:
        home_mapped = normalize_team_name(fix['home'], df)
        away_mapped = normalize_team_name(fix['away'], df)

        if not home_mapped or not away_mapped:
            no_data.append(fix)
            continue

        # Estimate AH line
        if fix.get('home_odds') and fix.get('away_odds'):
            try:
                # ESPN moneyline to decimal
                def ml_to_decimal(ml):
                    if ml is None:
                        return None
                    ml = int(ml)
                    if ml > 0:
                        return (ml / 100) + 1
                    else:
                        return (100 / abs(ml)) + 1

                h_dec = ml_to_decimal(fix['home_odds'])
                a_dec = ml_to_decimal(fix['away_odds'])
                d_dec = 3.2  # Default draw odds
                ah_line = estimate_ah_from_odds(h_dec, d_dec, a_dec)
            except:
                ah_line = 0.0
        else:
            ah_line = 0.0  # Default to pick'em

        result = predict_match(df, home_mapped, away_mapped, ah_line)
        result['league'] = fix['league']
        result['home_display'] = fix['home']
        result['away_display'] = fix['away']
        result['date'] = fix['date']
        picks.append(result)

    # Sort: HIGH confidence first, then MEDIUM, then SKIP
    conf_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2, 'N/A': 3}
    rec_order = {'BET HOME': 0, 'BET AWAY': 1, 'SKIP': 2}
    picks.sort(key=lambda x: (conf_order.get(x['confidence'], 3), rec_order.get(x['recommendation'], 2)))

    return picks, no_data


def build_picks_report(picks, no_data, results_df):
    lines = [
        f"# Weekend Football Picks — AH Prediction System",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Method:** H2H + venue form + goal margin prediction",
        f"**Focus:** AH lines closest to 2.0 odds (fair market)",
        f"",
    ]

    # Backtest summary
    if results_df is not None:
        bets = results_df[results_df['recommendation'] != 'SKIP']
        bets_decided = bets[bets['correct'].notna()]
        if len(bets_decided) > 0:
            wr = bets_decided['correct'].mean()
            high_bets = bets_decided[bets_decided['confidence'] == 'HIGH']
            high_wr = high_bets['correct'].mean() if len(high_bets) > 0 else 0
            lines += [
                f"## 📊 Backtest Performance (2023-2025)",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Resolved bets | {len(bets_decided)} |",
                f"| Overall WR | {wr:.1%} |",
                f"| HIGH confidence WR | {high_wr:.1%} ({len(high_bets)} bets) |",
                f"",
            ]

    # Strong picks
    strong = [p for p in picks if p['recommendation'] != 'SKIP' and p['confidence'] == 'HIGH']
    medium = [p for p in picks if p['recommendation'] != 'SKIP' and p['confidence'] == 'MEDIUM']
    skipped = [p for p in picks if p['recommendation'] == 'SKIP']

    if strong:
        lines += [f"## 🔥 HIGH Confidence Picks ({len(strong)})", f""]
        for p in strong:
            icon = "🏠" if p['recommendation'] == 'BET HOME' else "✈️"
            side = p['home_display'] if p['recommendation'] == 'BET HOME' else p['away_display']
            lines += [
                f"### {icon} {p.get('home_display', p['match'].split(' vs ')[0])} vs {p.get('away_display', p['match'].split(' vs ')[1])} ({p.get('league', '')})",
                f"**Pick:** {p['recommendation']} → **{side}** AH {p['ah_line']:+.2f}",
                f"**Edge:** {p.get('edge', 'N/A'):+.2f} | **Predicted margin:** {p.get('predicted_margin', 'N/A'):+.2f}",
                f"",
                f"> {p.get('reasoning', p.get('reason', ''))}",
                f"",
            ]

    if medium:
        lines += [f"## 📈 MEDIUM Confidence Picks ({len(medium)})", f""]
        for p in medium:
            icon = "🏠" if p['recommendation'] == 'BET HOME' else "✈️"
            side = p['home_display'] if p['recommendation'] == 'BET HOME' else p['away_display']
            lines += [
                f"### {icon} {p.get('home_display', '')} vs {p.get('away_display', '')} ({p.get('league', '')})",
                f"**Pick:** {p['recommendation']} → **{side}** AH {p['ah_line']:+.2f}",
                f"**Edge:** {p.get('edge', 'N/A'):+.2f} | **Predicted margin:** {p.get('predicted_margin', 'N/A'):+.2f}",
                f"",
                f"> {p.get('reasoning', p.get('reason', ''))}",
                f"",
            ]

    if skipped:
        lines += [
            f"## ⏭️ Skipped ({len(skipped)} matches — no clear edge)",
            f"",
        ]
        for p in skipped:
            lines.append(f"- {p.get('home_display', '')} vs {p.get('away_display', '')} ({p.get('league', '')}) — {p.get('reason', 'edge < 0.5')}")

    if no_data:
        lines += [
            f"",
            f"## ❓ No Historical Data ({len(no_data)} matches)",
            f"",
        ]
        for fix in no_data:
            lines.append(f"- {fix['home']} vs {fix['away']} ({fix['league']})")

    lines += [
        f"",
        f"---",
        f"_System: soccer-alpha per-game AH prediction v2_",
        f"_Edge threshold: ±0.5 goals | HIGH confidence: |edge| > 0.8_",
    ]

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    df = load_data()
    print(f"Loaded {len(df)} matches")

    # Step 4: Backtest
    results_df = run_backtest(df)

    # Step 5: Man United vs Everton
    mu_result = analyze_man_united_everton(df)

    # Step 6: Weekend picks
    fixtures = fetch_espn_fixtures()
    picks, no_data = run_weekend_picks(df, fixtures)

    # Build and save picks report
    report = build_picks_report(picks, no_data, results_df)
    picks_path = RESEARCH_DIR / 'weekend_picks_v2.md'
    picks_path.write_text(report)
    print(f"\nSaved picks to research/weekend_picks_v2.md")

    # Summary
    strong = [p for p in picks if p['recommendation'] != 'SKIP' and p['confidence'] == 'HIGH']
    medium = [p for p in picks if p['recommendation'] != 'SKIP' and p['confidence'] == 'MEDIUM']
    print(f"\nPicks summary: {len(strong)} HIGH, {len(medium)} MEDIUM, {len(picks)-len(strong)-len(medium)} SKIP")
    print("\nDone!")
