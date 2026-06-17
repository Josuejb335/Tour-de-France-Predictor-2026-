"""Post-process cache CSVs to forward-fill gap_pct across time groups.

Cache files don't have year/stage columns, so we detect stage
boundaries from position resets (each stage starts at position 1).
"""
import pandas as pd
import os

CACHE = "cache"
OUT = "tdf_stage_results_2020_2025.csv"
COLS = ['year', 'stage', 'rider_name', 'position', 'won',
        'stage_type', 'elevation', 'distance_km', 'pcs_ranking',
        'rider_type', 'gap_pct', 'hist_stage_wins', 'RCS_ranking']


def fix_year_file(path):
    df = pd.read_csv(path)
    cur = 0.0
    prev_pos = 0
    out_gaps = []
    for _, row in df.iterrows():
        pos = int(row['position'])
        if pos <= prev_pos:
            cur = 0.0
        g = row['gap_pct']
        if g != 0.0:
            cur = g
        out_gaps.append(cur)
        prev_pos = pos
    df = df.copy()
    df['gap_pct'] = out_gaps
    df.to_csv(path, index=False)
    bad = df[(df['position'] > 1) & (df['gap_pct'] == 0.0)]
    return df, len(bad)


def main():
    all_dfs = []
    for fname in sorted(os.listdir(CACHE)):
        if not fname.startswith('results_') or not fname.endswith('.csv'):
            continue
        path = os.path.join(CACHE, fname)
        df, n_bad = fix_year_file(path)
        year = int(fname.replace('results_', '').replace('.csv', ''))
        # Add year/stage columns by detecting position resets
        stage = 1
        prev_pos = 0
        stages = []
        for _, row in df.iterrows():
            pos = int(row['position'])
            if pos <= prev_pos:
                stage += 1
            stages.append(stage)
            prev_pos = pos
        df['year'] = year
        df['stage'] = stages
        all_dfs.append(df)
        print(f"  {fname}: {len(df)} rows, {n_bad} pos>1 still at 0.0")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[COLS]
    combined = combined.sort_values(['year', 'stage', 'position']).reset_index(drop=True)
    combined.to_csv(OUT, index=False)
    print(f"\nSaved {len(combined)} rows to {OUT}")


if __name__ == "__main__":
    main()
