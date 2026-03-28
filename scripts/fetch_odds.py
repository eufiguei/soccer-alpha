#!/usr/bin/env python3
"""Fetch historical odds data from football-data.co.uk for 5 major leagues, 2019-2024"""

import requests
import pandas as pd
import io
import os
import time

DATA_DIR = '/root/.openclaw/workspace/projects/soccer-alpha/data'
os.makedirs(DATA_DIR, exist_ok=True)

# Season codes and league codes
seasons = ['1920', '2021', '2122', '2223', '2324', '2425']
leagues = {
    'EPL': 'E0',
    'LaLiga': 'SP1', 
    'Bundesliga': 'D1',
    'SerieA': 'I1',
    'Ligue1': 'F1'
}

all_data = []

for season in seasons:
    for league_name, league_code in leagues.items():
        url = f'https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv'
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), on_bad_lines='skip')
                df['Season'] = season
                df['League'] = league_name
                all_data.append(df)
                print(f'{league_name} {season}: {len(df)} matches')
            else:
                print(f'{league_name} {season}: HTTP {r.status_code}')
        except Exception as e:
            print(f'{league_name} {season}: Error - {e}')
        time.sleep(0.5)  # Be nice to the server

# Combine all data
if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_parquet(f'{DATA_DIR}/all_odds_raw.parquet', index=False)
    print(f'\nTotal: {len(combined)} matches saved to all_odds_raw.parquet')
    
    # Show columns available
    print(f'\nColumns: {list(combined.columns)[:30]}...')
else:
    print('No data fetched!')
