# Script 05 - Weighted Formation Ranking
# Author: Precious Faseyosan
#
# Ranks formations using a 4-criterion weighted score reflecting acquisition priorities.
# Criteria: remaining life (W1), low decline rate (W2), low water loading (W3), initial rate (W4).
# Weights default to 40/30/20/10 - adjustable in the dashboard via sliders.
# Survivorship bias correction applied: multiply quality score by % active wells per formation.
#
# Inputs: formation_summary.csv, cluster_life_summary.csv (from 04_remaining_life.py)
# Outputs: formation_ranking.csv, ranking_charts/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('ranking_charts', exist_ok=True)

print("=" * 65)
print("  FORMATION RANKING ENGINE — SCRIPT 05")
print("=" * 65)
print()


print("=" * 65)
print("STEP 1: LOADING DATA")
print("=" * 65)
print()

formation_df  = pd.read_csv('formation_summary.csv', index_col=0)
cluster_df    = pd.read_csv('cluster_life_summary.csv', index_col=0)

# Exclude UNKNOWN and OTHER_FORMATION from ranking
formation_df = formation_df[
    ~formation_df.index.isin(['UNKNOWN', 'OTHER_FORMATION'])
].copy()

# Require minimum 30 wells (already filtered in Script 04 but
# applying again to be safe)
formation_df = formation_df[formation_df['well_count'] >= 30]

print(f"✓ Loaded formation_summary.csv: {len(formation_df)} named formations")
print(f"✓ Loaded cluster_life_summary.csv: {len(cluster_df)} clusters")
print()

# Verify required columns exist
REQUIRED = [
    'median_remaining_life_active', 'pct_active',
    'median_decline_ratio', 'median_water_boe_ratio',
    'median_first_yr_pi', 'well_count',
]
missing_cols = [c for c in REQUIRED if c not in formation_df.columns]
if missing_cols:
    print(f"ERROR: Missing columns in formation_summary.csv:")
    for c in missing_cols:
        print(f"  - {c}")
    print("Re-run 04_remaining_life.py first.")
    exit()
print("✓ All required columns present.")
print()


# Weights reflect operator acquisition priorities - users adjust these in the dashboard.
# W1 remaining life (longevity drives asset value) | W2 low decline (stable volumes)
# W3 low water (lower lifting costs) | W4 initial rate (formation productivity potential)
print("=" * 65)
print("STEP 2: RANKING WEIGHTS")
print("=" * 65)
print()

# ↓↓↓ CHANGE THESE WEIGHTS TO REFLECT YOUR ACQUISITION STRATEGY ↓↓↓
W1_remaining_life  = 40   # % weight on remaining productive life
W2_low_decline     = 30   # % weight on low decline rate
W3_low_water       = 20   # % weight on low water loading
W4_initial_rate    = 10   # % weight on high initial productivity

total = W1_remaining_life + W2_low_decline + W3_low_water + W4_initial_rate
assert total == 100, f"Weights must sum to 100 — currently sum to {total}"

print(f"  Remaining Life weight:         {W1_remaining_life}%")
print(f"  Low Decline Rate weight:       {W2_low_decline}%")
print(f"  Low Water Loading weight:      {W3_low_water}%")
print(f"  High Initial Rate weight:      {W4_initial_rate}%")
print(f"  Total:                         {total}%  ✓")
print()
print("  NOTE: Adjust weights to match your acquisition strategy.")
print("        In the dashboard, users set these via sliders.")
print()


# Min-max normalise to [0, 1]. Water loading inverted (lower is better - score = 1 - normalised).
print("=" * 65)
print("STEP 3: NORMALISING CRITERIA")
print("=" * 65)
print()

def normalise(series, higher_is_better=True):
    """Min-max normalise a Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    normalised = (series - mn) / (mx - mn)
    return normalised if higher_is_better else (1 - normalised)

formation_df['score_remaining_life'] = normalise(
    formation_df['median_remaining_life_active'],
    higher_is_better=True
)

formation_df['score_low_decline'] = normalise(
    formation_df['median_decline_ratio'],
    higher_is_better=True      # higher ratio = less declined = better
)

formation_df['score_low_water'] = normalise(
    formation_df['median_water_boe_ratio'],
    higher_is_better=False     # lower water = better
)

formation_df['score_initial_rate'] = normalise(
    formation_df['median_first_yr_pi'],
    higher_is_better=True
)

print("Normalised scores (0 = worst, 1 = best) — sample of 5:")
score_cols = ['score_remaining_life', 'score_low_decline',
              'score_low_water', 'score_initial_rate']
print(formation_df[score_cols].head(5).round(3).to_string())
print()


print("=" * 65)
print("STEP 4: CALCULATING WEIGHTED SCORES")
print("=" * 65)
print()

formation_df['weighted_score'] = (
    (W1_remaining_life / 100) * formation_df['score_remaining_life'] +
    (W2_low_decline    / 100) * formation_df['score_low_decline']    +
    (W3_low_water      / 100) * formation_df['score_low_water']      +
    (W4_initial_rate   / 100) * formation_df['score_initial_rate']
).round(4)

# Survivorship bias correction: multiply quality score by % active.
# Formations with few active wells score well on criteria 2+3 (only resilient wells survive)
# but offer little acquisition inventory. Final Score = Quality Score x (% Active / 100).

formation_df['weighted_score'] = (
    formation_df['weighted_score'] * (formation_df['pct_active'] / 100)
).round(4)

# Also retain the simpler composite score from Script 04
# as a reference column
if 'ranking_score' in formation_df.columns:
    formation_df = formation_df.rename(
        columns={'ranking_score': 'simple_score_script04'}
    )

# Rank
formation_df['rank'] = formation_df['weighted_score'].rank(
    ascending=False, method='min'
).astype(int)

formation_df = formation_df.sort_values('rank')

print(f"  Weights: Remaining Life {W1_remaining_life}%  |  "
      f"Low Decline {W2_low_decline}%  |  "
      f"Low Water {W3_low_water}%  |  "
      f"Initial Rate {W4_initial_rate}%")
print()
print("Top 20 Formations by Weighted Score:")
print()

display_cols = [
    'rank', 'well_count',
    'median_remaining_life_active', 'pct_active',
    'median_decline_ratio', 'median_water_boe_ratio',
    'weighted_score'
]
print(formation_df[display_cols].head(20).to_string())
print()


print("=" * 65)
print("STEP 5: CLUSTER SUMMARY")
print("=" * 65)
print()

CLUSTER_LABELS = {
    1: 'C1: Mature Conventional Gas',
    2: 'C2: Mature Conventional Oil',
    3: 'C3: Legacy End-of-Life',
    4: 'C4: Modern Shale/Tight',
    5: 'C5: Active Horizontal',
}

print("Cluster performance summary (active wells only):")
print()
for c in sorted(cluster_df.index):
    row  = cluster_df.loc[c]
    desc = CLUSTER_LABELS.get(int(c), f'Cluster {c}')
    rl   = row.get('median_remaining_life_active', row['median_remaining_life'])
    act  = row['pct_active']
    print(f"  {desc}")
    print(f"    Active: {act:.0f}%  |  "
          f"Median remaining life: {rl:.0f} months ({rl/12:.1f} years)")
print()


print("=" * 65)
print("STEP 6: GENERATING CHARTS")
print("=" * 65)
print()

# Chart 1: Weighted Formation Ranking - primary output chart for acquisition
print("Generating Chart 1: Weighted formation ranking...")

top20 = formation_df.head(20)

# Component contributions to total weighted score
comp1 = top20['score_remaining_life']  * W1_remaining_life / 100
comp2 = top20['score_low_decline']     * W2_low_decline    / 100
comp3 = top20['score_low_water']       * W3_low_water      / 100
comp4 = top20['score_initial_rate']    * W4_initial_rate   / 100

component_df = pd.DataFrame({
    f'Remaining Life ({W1_remaining_life}%)':  comp1,
    f'Low Decline ({W2_low_decline}%)':        comp2,
    f'Low Water ({W3_low_water}%)':            comp3,
    f'Initial Rate ({W4_initial_rate}%)':      comp4,
}, index=top20.index)

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Left panel: total score bar
bar_colors = [
    '#2ecc71' if v >= 0.5 else
    '#f39c12' if v >= 0.25 else
    '#e74c3c'
    for v in top20['weighted_score']
]

bars = axes[0].barh(
    top20.index[::-1],
    top20['weighted_score'][::-1],
    color=bar_colors[::-1], edgecolor='none', alpha=0.88
)
for bar, score, rl, pct, wells in zip(
    bars,
    top20['weighted_score'][::-1],
    top20['median_remaining_life_active'][::-1],
    top20['pct_active'][::-1],
    top20['well_count'][::-1]
):
    axes[0].text(
        score + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f'{score:.3f}  |  {rl:.0f} mo ({rl/12:.1f} yr)  |  '
        f'{pct:.0f}% active  |  {wells:,} wells',
        va='center', fontsize=7
    )

axes[0].set_xlabel('Weighted Score (0–1)', fontsize=11)
axes[0].set_title(
    f'Formation Ranking — Weighted Score\n'
    f'Remaining Life {W1_remaining_life}%  |  '
    f'Low Decline {W2_low_decline}%  |  '
    f'Low Water {W3_low_water}%  |  '
    f'Initial Rate {W4_initial_rate}%',
    fontsize=10, fontweight='bold'
)
axes[0].set_xlim(0, top20['weighted_score'].max() * 1.55)

# Right panel: stacked score breakdown
component_df.iloc[::-1].plot(
    kind='barh', stacked=True, ax=axes[1],
    color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'],
    edgecolor='none', alpha=0.88
)
axes[1].set_xlabel('Contribution to Weighted Score', fontsize=11)
axes[1].set_title(
    'Score Breakdown by Criterion\n'
    'Shows what drives each formation\'s ranking',
    fontsize=10, fontweight='bold'
)
axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
axes[1].set_xlim(0, 1.0)

plt.suptitle(
    'Formation Acquisition Ranking — Weighted Multi-Criteria Score\n'
    'Score = Quality Score × (% Active)  |  Penalises formations with few surviving wells',
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('ranking_charts/chart01_weighted_ranking.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart01_weighted_ranking.png")

# Chart 2: Formation Performance Map - remaining life vs % active, sized by well count
print("Generating Chart 2: Formation performance map...")

# Use all ranked formations (not just top 20) for the scatter
plot_df = formation_df[
    formation_df['median_remaining_life_active'] > 0
].copy()

fig, ax = plt.subplots(figsize=(12, 9))

sc = ax.scatter(
    plot_df['median_remaining_life_active'],
    plot_df['pct_active'],
    c=plot_df['weighted_score'],
    s=np.sqrt(plot_df['well_count']) * 4,  # size by well count
    cmap='RdYlGn', alpha=0.75,
    edgecolors='grey', linewidths=0.4,
    vmin=0, vmax=plot_df['weighted_score'].max()
)
plt.colorbar(sc, ax=ax, label='Weighted Score')

# Label top 12 formations
for form, row in formation_df.head(12).iterrows():
    if row['median_remaining_life_active'] > 0:
        ax.annotate(
            form[:22],
            xy=(row['median_remaining_life_active'], row['pct_active']),
            xytext=(5, 3), textcoords='offset points',
            fontsize=6.5, alpha=0.9
        )

ax.axvline(x=60,  color='orange', linestyle='--', linewidth=1.2,
           label='5 yr remaining', alpha=0.7)
ax.axvline(x=120, color='green',  linestyle='--', linewidth=1.2,
           label='10 yr remaining', alpha=0.7)
ax.axhline(y=50,  color='steelblue', linestyle=':', linewidth=1.2,
           label='50% active', alpha=0.7)

ax.set_xlabel('Median Remaining Life — Active Wells (months)', fontsize=12)
ax.set_ylabel('% of Formation Wells Still Active', fontsize=12)
ax.set_title(
    'Formation Performance Map\n'
    'Dot size = well count  |  Colour = weighted score\n'
    'Best targets: top-right corner (long life + high % active)',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('ranking_charts/chart02_performance_map.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart02_performance_map.png")

# Chart 3: Cluster remaining life summary (management view)
print("Generating Chart 3: Cluster remaining life (management view)...")

rl_col = 'median_remaining_life_active' \
    if 'median_remaining_life_active' in cluster_df.columns \
    else 'median_remaining_life'

cluster_plot = cluster_df[rl_col].copy()
cluster_plot.index = [
    CLUSTER_LABELS.get(int(c), f'Cluster {c}')
    for c in cluster_plot.index
]

bar_colors_c = [
    '#2ecc71' if v >= 60 else
    '#f39c12' if v >= 24 else
    '#e74c3c'
    for v in cluster_plot.values
]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(
    range(len(cluster_plot)),
    cluster_plot.values,
    color=bar_colors_c, edgecolor='none', alpha=0.88
)
for i, (bar, val) in enumerate(zip(bars, cluster_plot.values)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f'{val:.0f} mo\n({val/12:.1f} yr)',
        ha='center', fontsize=9, fontweight='bold'
    )

ax.set_xticks(range(len(cluster_plot)))
ax.set_xticklabels(cluster_plot.index, fontsize=9, rotation=10, ha='right')
ax.set_ylabel('Median Remaining Life — Active Wells (months)', fontsize=11)
ax.axhline(y=60,  color='green', linestyle='--', linewidth=1.5,
           label='5 years', alpha=0.8)
ax.axhline(y=120, color='teal', linestyle=':', linewidth=1.5,
           label='10 years', alpha=0.8)
ax.set_title(
    'Median Remaining Productive Life by Well Cluster\n'
    'Active wells only — management summary view',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('ranking_charts/chart03_cluster_summary.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart03_cluster_summary.png")
print()


print("=" * 65)
print("STEP 7: SAVING OUTPUTS")
print("=" * 65)
print()

formation_df.to_csv('formation_ranking.csv')
print(f"✓ Saved: formation_ranking.csv  ({len(formation_df)} formations)")
print()
print("  Columns in formation_ranking.csv:")
for col in formation_df.columns:
    print(f"    {col}")
print()


print("=" * 65)
print("FORMATION RANKING COMPLETE — SUMMARY")
print("=" * 65)
print()
print(f"  Weights: Remaining Life {W1_remaining_life}%  |  "
      f"Low Decline {W2_low_decline}%  |  "
      f"Low Water {W3_low_water}%  |  "
      f"Initial Rate {W4_initial_rate}%")
print()
print(f"  Top 10 Formations:")
for rank, (form, row) in enumerate(formation_df.head(10).iterrows(), 1):
    rl  = row['median_remaining_life_active']
    pct = row['pct_active']
    sc  = row['weighted_score']
    print(f"  {rank:>2}. {form:<38}  "
          f"Score: {sc:.3f}  |  "
          f"{rl:.0f} mo ({rl/12:.1f} yr)  |  "
          f"{pct:.0f}% active")
print()
print("  Output files:")
print("    formation_ranking.csv       ← full ranked formation table")
print("    ranking_charts/             ← 3 charts")
print()
print("  NEXT STEP: Run 06_dashboard.py")
print("=" * 65)
