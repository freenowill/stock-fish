"""
Generate deepseek_selections.json — simple top-N + rebalancing strategy.

- Week 1: directly pick top 5 by model score
- Week 2+: keep holdings still in top 50% rank; for those that dropped,
  replace with highest-scoring stocks from the top pool (excluding current holdings)
"""

import json
from pathlib import Path
import pandas as pd

exp_dir = Path('DATA/analysis_outputs/2026-05-27-csi1000')
wf_dir = exp_dir / 'model_predict' / 'walk_forward'
fold_dirs = sorted(
    [d for d in wf_dir.iterdir() if d.is_dir() and (d / 'model_predict' / 'all_scores.csv').exists()],
    key=lambda d: d.name,
)
last = fold_dirs[-1] / 'model_predict'

df = pd.read_csv(last / 'all_scores.csv')
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime', 'instrument', 'score'])
df['code'] = df['instrument'].astype(str).str.replace(r'^[A-Za-z]+', '', regex=True).str.zfill(6)

# Weekly dates
dt_idx = pd.DatetimeIndex(df['datetime']).dropna().sort_values().unique()
weekly_df = (pd.DataFrame({'datetime': dt_idx})
    .assign(iso_year=lambda x: x['datetime'].dt.isocalendar().year,
             iso_week=lambda x: x['datetime'].dt.isocalendar().week)
    .groupby(['iso_year', 'iso_week'], as_index=False)['datetime'].max())
weekly_dates = sorted(pd.to_datetime(weekly_df['datetime']).tolist())

HOLD_NUM = 5
selections = {}
current_holdings: set[str] = set()

print(f"Last fold: {fold_dirs[-1].name}")
print(f"Weekly dates: {len(weekly_dates)}, Hold num: {HOLD_NUM}")
print()

for week_idx, dt in enumerate(weekly_dates, 1):
    pred_date = dt.strftime('%Y-%m-%d')

    day = df[df['datetime'] == dt]
    if day.empty:
        week_end = dt + pd.Timedelta(days=6)
        day = df[(df['datetime'] >= dt) & (df['datetime'] <= week_end)]
        if day.empty:
            continue
        latest = day['datetime'].max()
        day = day[day['datetime'] == latest]

    day = day.sort_values('score', ascending=False).reset_index(drop=True)
    day['rank'] = range(1, len(day) + 1)
    day['code'] = day['code'].str.zfill(6)
    total = len(day)

    if current_holdings:
        prev_codes = {c.zfill(6) for c in current_holdings}
        threshold_rank = total // 2
        prev_rows = day[day['code'].isin(prev_codes)]

        keep = []
        for _, row in prev_rows.iterrows():
            if int(row['rank']) <= threshold_rank:
                keep.append(str(row['code']).zfill(6))

        dropped = prev_codes - set(keep)
        if not dropped:
            selections[pred_date] = ','.join(sorted(current_holdings))
            print(f"  [{week_idx:3d}] {pred_date}: ✓ all held {keep}")
            continue

        # Replace dropped ones with top-scoring stocks (excluding current holdings)
        kept_set = set(keep)
        all_holdings_set = kept_set | {c.zfill(6) for c in current_holdings}
        candidates = day[~day['code'].isin(all_holdings_set)]

        n_replace = len(dropped)
        new_picks = candidates.head(n_replace)['code'].str.zfill(6).tolist()

        result = keep + new_picks
        selections[pred_date] = ','.join(result[:HOLD_NUM])
        current_holdings = set(result[:HOLD_NUM])
        print(f"  [{week_idx:3d}] {pred_date}: 🔄 drop {sorted(dropped)} → add {new_picks} | hold {result[:HOLD_NUM]}")

    else:
        # Week 1: directly pick top 5 by model score
        picks = day.head(HOLD_NUM)['code'].str.zfill(6).tolist()
        selections[pred_date] = ','.join(picks)
        current_holdings = set(picks)
        print(f"  [{week_idx:3d}] {pred_date}: 🆕 top-5 → {picks}")

# Write
out_file = exp_dir / 'deepseek_selections.json'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(selections, f, ensure_ascii=False, indent=2)
print(f"\n✓ Written: {out_file} ({len(selections)} weeks)")
