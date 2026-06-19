# Script 04 - Remaining Productive Life Estimation
# Author: Precious Faseyosan
#
# Calculates remaining months of productive life per well using exponential decline.
# Economic limit = when combined gas+oil revenue falls below monthly OPEX.
# Using combined revenue (not fluid-specific) because Enverus Production Type reflects
# filing conventions, not current revenue split - a GAS well may earn more from condensate.
#
# Formula (exponential decline):
#   t_remaining = -ln(OPEX / monthly_revenue) / D_boe
#
# Data note: production values in data_clustered.csv are log-transformed.
# Recovery: raw_rate = np.expm1(log_rate)  [inverse of log1p, exact]
#
# Outputs: data_with_remaining_life.csv, formation_summary.csv,
#          cluster_life_summary.csv, remaining_life_charts/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('remaining_life_charts', exist_ok=True)

print("=" * 65)
print("  REMAINING PRODUCTIVE LIFE ESTIMATION — ENGINE 2")
print("=" * 65)
print()


print("=" * 65)
print("STEP 1: LOADING DATA")
print("=" * 65)
print()

# Place data_clustered.csv in the same folder (output of 03_clustering.py).
df = pd.read_csv('data_clustered.csv', low_memory=False)

print(f"✓ Loaded: {len(df):,} wells  ×  {len(df.columns)} columns")
print()

# Verify cluster labels are 1-5
if 'cluster' not in df.columns:
    print("ERROR: 'cluster' column not found. Run 03_clustering.py first.")
    exit()

cluster_labels_found = sorted(df['cluster'].unique())
print(f"Cluster labels present: {cluster_labels_found}")
print(f"Cluster distribution:")
for c, n in df['cluster'].value_counts().sort_index().items():
    print(f"  Cluster {c}: {n:,} wells ({n/len(df)*100:.1f}%)")
print()


# Operators adjust these in the dashboard via sliders - defaults are conservative estimates.
# Economic limit: revenue = (gas x GAS_PRICE) + (oil x OIL_PRICE) >= MONTHLY_OPEX.
print("=" * 65)
print("STEP 2: ECONOMIC PARAMETERS")
print("=" * 65)
print()

# ↓↓↓ CHANGE THESE VALUES TO MATCH YOUR OPERATING ENVIRONMENT ↓↓↓
GAS_PRICE    = 2.50     # $/MCF   — wellhead or regional gas price
OIL_PRICE    = 65.00    # $/BBL   — wellhead or regional oil price
MONTHLY_OPEX = 3000.00  # $/month — average operating cost per well

# Maximum remaining life cap — prevents unrealistic projections
# for very slow-declining wells. 15 years = 180 months.
MAX_REMAINING_MONTHS = 180

print(f"  Gas Price:     ${GAS_PRICE:.2f}/MCF")
print(f"  Oil Price:     ${OIL_PRICE:.2f}/BBL")
print(f"  Monthly OPEX:  ${MONTHLY_OPEX:,.2f}/well/month")
print()
print(f"  A well is economic when:")
print(f"  (Gas Rate × ${GAS_PRICE:.2f}) + (Oil Rate × ${OIL_PRICE:.2f}) ≥ ${MONTHLY_OPEX:,.2f}/month")
print()
print(f"  Maximum remaining life cap: {MAX_REMAINING_MONTHS} months ({MAX_REMAINING_MONTHS//12} years)")
print()
print("  NOTE: These are defaults. Operators set their own in the dashboard.")
print()


# Log-transformed values in data_clustered.csv (from 02_cleaning.py).
# Recovery: np.expm1(log_Last_12_Gas) = Last 12 Gas annual MCF (exact inverse of log1p).
# Divide by 12 for monthly rates used in revenue calculation.
print("=" * 65)
print("STEP 3: RECOVERING CURRENT PRODUCTION RATES")
print("=" * 65)
print()

# Recover annual totals from log columns
df['last_12_gas_annual'] = np.expm1(df['log_Last_12_Gas'])    # MCF/year
df['last_12_oil_annual'] = np.expm1(df['log_Last_12_Oil'])    # BBL/year
df['last_12_boe_annual'] = np.expm1(df['log_last_12_boe_equiv'])  # BOE/year

# Convert to monthly rates
df['monthly_gas'] = df['last_12_gas_annual'] / 12   # MCF/month
df['monthly_oil'] = df['last_12_oil_annual'] / 12   # BBL/month
df['monthly_boe'] = df['last_12_boe_annual'] / 12   # BOE/month

# Current monthly revenue — combines BOTH gas and oil
# This is the core of the combined-revenue approach
df['monthly_revenue'] = (
    df['monthly_gas'] * GAS_PRICE +
    df['monthly_oil'] * OIL_PRICE
)

print("Current monthly rates (median across all wells):")
print(f"  Gas rate:      {df['monthly_gas'].median():>10,.1f} MCF/month")
print(f"  Oil rate:      {df['monthly_oil'].median():>10,.1f} BBL/month")
print(f"  BOE rate:      {df['monthly_boe'].median():>10,.1f} BOE/month")
print(f"  Monthly revenue: ${df['monthly_revenue'].median():>10,.2f}/month")
print(f"  Monthly OPEX:    ${MONTHLY_OPEX:>10,.2f}/month")
print()

# How many wells are currently generating revenue above OPEX?
currently_economic = (df['monthly_revenue'] >= MONTHLY_OPEX).sum()
print(f"  Wells currently generating revenue >= OPEX: "
      f"{currently_economic:,} ({currently_economic/len(df)*100:.1f}%)")
print()


# D_boe = -ln(decline_ratio_boe) / months_since_peak
# decline_ratio_boe = Last12BOE/PeakBOE from 02_cleaning.py; peak assumed at 12 months.
# D clipped to [0.0001, 0.30]: 0.0001 = virtually flat (capped), 0.30 = max plausible.
print("=" * 65)
print("STEP 4: CALCULATING MONTHLY DECLINE RATES")
print("=" * 65)
print()

# Months since peak
df['months_since_peak'] = (df['Months Produced'] - 12).clip(lower=1)

# Decline rates for all three fluid bases
for fluid, ratio_col in [
    ('boe', 'decline_ratio_boe'),   # PRIMARY — used in remaining life calc
    ('gas', 'decline_ratio_gas'),   # DIAGNOSTIC — informational
    ('oil', 'decline_ratio_oil'),   # DIAGNOSTIC — informational
]:
    safe_ratio     = df[ratio_col].clip(lower=0.001, upper=1.499)
    df[f'D_{fluid}'] = (
        -np.log(safe_ratio) / df['months_since_peak']
    ).clip(lower=0.0001, upper=0.30)

print("Monthly decline rate statistics (D = monthly fractional decline):")
for fluid in ['boe', 'gas', 'oil']:
    col = df[f'D_{fluid}']
    pct = col.median() * 100
    print(f"  D_{fluid}  median: {col.median():.5f}/month  "
          f"({pct:.3f}%/month)  "
          f"  annual equiv: {(1-(1-col.median())**12)*100:.1f}%/year")
print()
print("  PRIMARY: D_boe will be used for all remaining life calculations.")
print("  (D_gas and D_oil retained as diagnostic outputs only.)")
print()


# Three cases: P&A/INACTIVE = 0 (ended), revenue<OPEX = 0 (uneconomic), else project forward.
# SHUT-IN and TA treated as active (expected to resume). D<0.0001 assigns MAX_REMAINING_MONTHS.
print("=" * 65)
print("STEP 5: CALCULATING REMAINING PRODUCTIVE LIFE")
print("=" * 65)
print()

def calculate_remaining_life(row):
    """Return (remaining_months, economic_status) for one well using exponential decline to OPEX."""

    # CASE 1: Well already permanently ended
    status = str(row.get('Well Status', '')).upper().strip()
    if status in ['P & A', 'INACTIVE']:
        return 0.0, 'ended'

    # Current monthly revenue from both gas and oil
    revenue = row['monthly_revenue']

    # CASE 2: Currently below economic limit
    if revenue < MONTHLY_OPEX or revenue <= 0:
        return 0.0, 'uneconomic'

    # CASE 3: Still economic — project forward using exponential decline
    D = row['D_boe']

    if D < 0.0001:
        # Effectively flat decline — assign maximum remaining life
        return float(MAX_REMAINING_MONTHS), 'active'

    try:
        # t = -ln(OPEX / Revenue_current) / D_boe
        remaining = -np.log(MONTHLY_OPEX / revenue) / D
        remaining = max(0.0, min(remaining, float(MAX_REMAINING_MONTHS)))
        return remaining, 'active'
    except Exception:
        return 0.0, 'uneconomic'


print("Calculating remaining life for all wells...")
print("(May take 1-2 minutes on 122,000 wells)")
print()

results = df.apply(calculate_remaining_life, axis=1)
df['remaining_life_months'] = results.apply(lambda x: round(x[0], 1))
df['economic_status']       = results.apply(lambda x: x[1])
df['remaining_life_years']  = (df['remaining_life_months'] / 12).round(2)

print("✓ Remaining life calculated for all wells.")
print()

# Subsets for analysis
ended      = df[df['economic_status'] == 'ended']
uneconomic = df[df['economic_status'] == 'uneconomic']
active     = df[df['economic_status'] == 'active']

print("Economic status breakdown:")
print(f"  Active (revenue > OPEX):      {len(active):>8,}  ({len(active)/len(df)*100:.1f}%)")
print(f"  Uneconomic (revenue < OPEX):  {len(uneconomic):>8,}  ({len(uneconomic)/len(df)*100:.1f}%)")
print(f"  Ended (P&A or Inactive):      {len(ended):>8,}  ({len(ended)/len(df)*100:.1f}%)")
print()
print("Remaining life — ALL wells (including zeros):")
print(f"  Median: {df['remaining_life_months'].median():.1f} months "
      f"({df['remaining_life_months'].median()/12:.1f} years)")
print(f"  Mean:   {df['remaining_life_months'].mean():.1f} months")
print()
print("Remaining life — ACTIVE wells only:")
print(f"  Median: {active['remaining_life_months'].median():.1f} months "
      f"({active['remaining_life_months'].median()/12:.1f} years)")
print(f"  Mean:   {active['remaining_life_months'].mean():.1f} months")
print(f"  Min:    {active['remaining_life_months'].min():.1f} months")
print(f"  Max:    {active['remaining_life_months'].max():.1f} months")
print()


print("=" * 65)
print("STEP 6: CLUSTER-LEVEL SUMMARY")
print("=" * 65)
print()

cluster_life = df.groupby('cluster').agg(
    well_count             = ('remaining_life_months', 'count'),
    median_remaining_life  = ('remaining_life_months', 'median'),
    pct_active             = ('economic_status',
                               lambda x: (x == 'active').mean() * 100),
    pct_uneconomic         = ('economic_status',
                               lambda x: (x == 'uneconomic').mean() * 100),
    pct_ended              = ('economic_status',
                               lambda x: (x == 'ended').mean() * 100),
    median_monthly_revenue = ('monthly_revenue', 'median'),
    median_months_produced = ('Months Produced', 'median'),
).round(2)

# Active-only median is the primary acquisition metric.
# Overall median is zero for most clusters (P&A/inactive dominate) - not useful for screening.
active_only = df[df['economic_status'] == 'active']
cluster_life['median_remaining_life_active'] = (
    active_only.groupby('cluster')['remaining_life_months']
    .median().round(1)
).fillna(0)

print("Cluster summary (all wells vs active-only remaining life):")
print(cluster_life[['well_count', 'pct_active',
                     'median_remaining_life',
                     'median_remaining_life_active']].to_string())
print()
print("  Note: median_remaining_life_active is the primary acquisition metric.")
print()

cluster_life.to_csv('cluster_life_summary.csv')
print("✓ Saved: cluster_life_summary.csv")
print()


print("=" * 65)
print("STEP 7: FORMATION-LEVEL SUMMARY")
print("=" * 65)
print()

MIN_WELLS = 30   # minimum wells for a formation to appear in ranking

if 'Target Formation' not in df.columns:
    print("WARNING: Target Formation column not found.")
    named_formations = pd.DataFrame()
else:
    formation_life = df.groupby('Target Formation').agg(
        well_count             = ('remaining_life_months', 'count'),
        median_remaining_life  = ('remaining_life_months', 'median'),
        mean_remaining_life    = ('remaining_life_months', 'mean'),
        pct_active             = ('economic_status',
                                   lambda x: (x == 'active').mean() * 100),
        median_decline_ratio   = ('decline_ratio_boe', 'median'),
        median_water_boe_ratio = ('water_boe_ratio', 'median'),
        median_first_yr_pi     = ('first_yr_pi', 'median'),
        median_monthly_revenue = ('monthly_revenue', 'median'),
        median_months_produced = ('Months Produced', 'median'),
    ).round(2)

    # Apply minimum well count filter
    formation_life = formation_life[
        formation_life['well_count'] >= MIN_WELLS
    ]

    # Add active-only median - overall median understates life for formations with many P&A wells.
    active_formation = df[df['economic_status'] == 'active']
    formation_life['median_remaining_life_active'] = (
        active_formation.groupby('Target Formation')['remaining_life_months']
        .median().round(1)
    )
    formation_life['median_remaining_life_active'] = (
        formation_life['median_remaining_life_active'].fillna(0)
    )

    # Score = (Remaining Life / Cap) x (% Active / 100).
    # At the 15yr cap, a 3%-active formation still ranks below a 70%-active one.
    formation_life['ranking_score'] = (
        (formation_life['median_remaining_life_active'] / MAX_REMAINING_MONTHS) *
        (formation_life['pct_active'] / 100)
    ).round(4)

    # Named formations only — exclude UNKNOWN and OTHER_FORMATION
    # Sort by composite ranking score
    named_formations = formation_life[
        ~formation_life.index.isin(['UNKNOWN', 'OTHER_FORMATION'])
    ].sort_values('ranking_score', ascending=False)

    print(f"Top 20 formations by composite ranking score (active wells only):")
    print(f"(>= {MIN_WELLS} wells | UNKNOWN and OTHER_FORMATION excluded)")
    print(f"Score = (Remaining Life / 15yr cap) × (% Active)")
    print()
    print(named_formations.head(20)[
        ['well_count', 'ranking_score',
         'median_remaining_life_active', 'pct_active',
         'median_decline_ratio']
    ].to_string())
    print()

    formation_life.to_csv('formation_summary.csv')
    print(f"✓ Saved: formation_summary.csv ({len(formation_life)} formations)")
    print()


print("=" * 65)
print("STEP 8: GENERATING CHARTS")
print("=" * 65)
print()

# Descriptive labels for cluster axes
CLUSTER_LABELS = {
    1: 'C1: Mature\nConventional Gas',
    2: 'C2: Mature\nConventional Oil',
    3: 'C3: Legacy\nEnd-of-Life',
    4: 'C4: Modern\nShale/Tight',
    5: 'C5: Active\nHorizontal',
}

CLUSTER_COLORS = {
    1: '#1abc9c',
    2: '#3498db',
    3: '#e74c3c',
    4: '#f39c12',
    5: '#9b59b6',
}

# Chart 1: Remaining life distribution - all wells vs active only
print("Generating Chart 1: Remaining life distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(
    df['remaining_life_months'].clip(0, 300),
    bins=60, color='steelblue', edgecolor='none', alpha=0.85
)
med_all = df['remaining_life_months'].median()
axes[0].axvline(x=med_all, color='red', linestyle='--', linewidth=2,
                label=f'Median: {med_all:.0f} mo ({med_all/12:.1f} yr)')
axes[0].set_xlabel('Remaining Life (months)', fontsize=11)
axes[0].set_ylabel('Number of Wells', fontsize=11)
axes[0].set_title('All Wells\n(0 = ended or currently uneconomic)',
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)

axes[1].hist(
    active['remaining_life_months'].clip(0, 300),
    bins=60, color='mediumseagreen', edgecolor='none', alpha=0.85
)
med_act = active['remaining_life_months'].median()
axes[1].axvline(x=med_act, color='red', linestyle='--', linewidth=2,
                label=f'Median: {med_act:.0f} mo ({med_act/12:.1f} yr)')
axes[1].set_xlabel('Remaining Life (months)', fontsize=11)
axes[1].set_ylabel('Number of Wells', fontsize=11)
axes[1].set_title('Active Wells Only\n(currently generating revenue above OPEX)',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle(
    f'Remaining Productive Life Distribution\n'
    f'Gas: ${GAS_PRICE:.2f}/MCF  |  Oil: ${OIL_PRICE:.2f}/BBL  |  '
    f'OPEX: ${MONTHLY_OPEX:,.0f}/mo',
    fontsize=12, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('remaining_life_charts/chart01_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart01_distribution.png")

# Chart 2: Remaining life by cluster (active wells only)
# Overall data is dominated by zeros from P&A/inactive - active-only is the useful view.
print("Generating Chart 2: Remaining life by cluster...")

cluster_order = sorted(df['cluster'].unique())
groups = [
    active_only[active_only['cluster'] == c]['remaining_life_months']
    .clip(0, MAX_REMAINING_MONTHS)
    for c in cluster_order
]
tick_labels = [CLUSTER_LABELS.get(c, f'Cluster {c}') for c in cluster_order]
box_colors  = [CLUSTER_COLORS.get(c, 'grey') for c in cluster_order]

fig, ax = plt.subplots(figsize=(13, 7))

bp = ax.boxplot(
    groups, labels=tick_labels,
    patch_artist=True,
    medianprops=dict(color='red', linewidth=2.5),
    flierprops=dict(marker='o', markersize=2, alpha=0.3)
)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.72)

# Median labels above each box
for i, grp in enumerate(groups):
    med = grp.median() if len(grp) > 0 else 0
    q75 = grp.quantile(0.75) if len(grp) > 0 else 0
    ax.text(i + 1, q75 + 3,
            f'{med:.0f} mo\n({med/12:.1f} yr)',
            ha='center', fontsize=8.5, color='darkred', fontweight='bold')

ax.set_ylabel('Remaining Life (months)', fontsize=12)
ax.set_xlabel('Well Cluster', fontsize=12)
ax.set_title('Remaining Productive Life by Cluster — Active Wells Only\n'
             'Red line = median  |  Box = 25th–75th percentile\n'
             '(Active = currently generating combined revenue above OPEX)',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='x', labelsize=9)
plt.tight_layout()
plt.savefig('remaining_life_charts/chart02_by_cluster.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart02_by_cluster.png")

# Chart 3: Formation ranking - primary output chart for acquisition targeting
print("Generating Chart 3: Formation ranking...")

if len(named_formations) > 0:
    top20 = named_formations.head(20)

    # Bar color driven by remaining life (keeps visual meaning)
    bar_colors = [
        '#2ecc71' if v >= 60 else
        '#f39c12' if v >= 24 else
        '#e74c3c'
        for v in top20['median_remaining_life_active']
    ]

    fig, ax = plt.subplots(figsize=(13, 10))

    # Bar LENGTH = composite ranking score (0 to 1)
    # This differentiates formations that all hit the 15yr cap
    # by weighting how many of their wells are still active
    bars = ax.barh(
        top20.index[::-1],
        top20['ranking_score'][::-1],
        color=bar_colors[::-1], edgecolor='none', alpha=0.88
    )

    # Labels: score + remaining life + % active + well count
    for bar, score, rl, pct, wells in zip(
        bars,
        top20['ranking_score'][::-1],
        top20['median_remaining_life_active'][::-1],
        top20['pct_active'][::-1],
        top20['well_count'][::-1]
    ):
        ax.text(
            score + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'Score: {score:.2f}  |  '
            f'{rl:.0f} mo ({rl/12:.1f} yr)  |  '
            f'{pct:.0f}% active  |  {wells:,} wells',
            va='center', fontsize=7.5
        )

    ax.set_xlabel(
        'Formation Quality Score  =  (Remaining Life ÷ 15yr cap)  ×  (% Active)\n'
        'Higher score = better acquisition target',
        fontsize=11
    )
    ax.set_title(
        'Formation Ranking by Composite Quality Score\n'
        'Rewards formations with BOTH long remaining life AND high % of active wells\n'
        f'Gas: ${GAS_PRICE:.2f}/MCF  |  Oil: ${OIL_PRICE:.2f}/BBL  |  '
        f'OPEX: ${MONTHLY_OPEX:,.0f}/month  |  Cap: {MAX_REMAINING_MONTHS} months',
        fontsize=11, fontweight='bold'
    )
    ax.set_xlim(0, top20['ranking_score'].max() * 1.60)
    plt.tight_layout()
    plt.savefig('remaining_life_charts/chart03_formation_ranking.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ chart03_formation_ranking.png")
else:
    print("  Skipped: no named formation data available")

# Chart 4: Economic sensitivity - remaining life vs gas and oil price scenarios
print("Generating Chart 4: Economic sensitivity...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Vary gas price ($1.50 to $5.00/MCF), hold oil price fixed
gas_prices = np.arange(1.50, 5.01, 0.25)
median_rl_gas = []

for gp in gas_prices:
    rev = df['monthly_gas'] * gp + df['monthly_oil'] * OIL_PRICE
    viable = (rev >= MONTHLY_OPEX) & (df['D_boe'] >= 0.0001)
    rl = np.where(
        viable,
        (-np.log(MONTHLY_OPEX / rev.clip(lower=MONTHLY_OPEX + 0.01)) /
         df['D_boe']).clip(0, MAX_REMAINING_MONTHS),
        0.0
    )
    median_rl_gas.append(float(np.median(rl)))

axes[0].plot(gas_prices, [r / 12 for r in median_rl_gas],
             'b-o', linewidth=2.5, markersize=5)
axes[0].axvline(x=GAS_PRICE, color='red', linestyle='--', linewidth=2,
                label=f'Current: ${GAS_PRICE:.2f}/MCF')
axes[0].fill_between(gas_prices, [r / 12 for r in median_rl_gas],
                     alpha=0.12, color='blue')
axes[0].set_xlabel('Gas Price ($/MCF)', fontsize=12)
axes[0].set_ylabel('Portfolio Median Remaining Life (years)', fontsize=11)
axes[0].set_title(f'Sensitivity to Gas Price\n'
                  f'(Oil fixed at ${OIL_PRICE:.2f}/BBL, '
                  f'OPEX ${MONTHLY_OPEX:,.0f}/mo)',
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)

# Right: Vary oil price ($30 to $120/BBL), hold gas price fixed
oil_prices = np.arange(30, 121, 5)
median_rl_oil = []

for op in oil_prices:
    rev = df['monthly_gas'] * GAS_PRICE + df['monthly_oil'] * op
    viable = (rev >= MONTHLY_OPEX) & (df['D_boe'] >= 0.0001)
    rl = np.where(
        viable,
        (-np.log(MONTHLY_OPEX / rev.clip(lower=MONTHLY_OPEX + 0.01)) /
         df['D_boe']).clip(0, MAX_REMAINING_MONTHS),
        0.0
    )
    median_rl_oil.append(float(np.median(rl)))

axes[1].plot(oil_prices, [r / 12 for r in median_rl_oil],
             'g-o', linewidth=2.5, markersize=5, color='darkorange')
axes[1].axvline(x=OIL_PRICE, color='red', linestyle='--', linewidth=2,
                label=f'Current: ${OIL_PRICE:.2f}/BBL')
axes[1].fill_between(oil_prices, [r / 12 for r in median_rl_oil],
                     alpha=0.12, color='darkorange')
axes[1].set_xlabel('Oil Price ($/BBL)', fontsize=12)
axes[1].set_ylabel('Portfolio Median Remaining Life (years)', fontsize=11)
axes[1].set_title(f'Sensitivity to Oil Price\n'
                  f'(Gas fixed at ${GAS_PRICE:.2f}/MCF, '
                  f'OPEX ${MONTHLY_OPEX:,.0f}/mo)',
                  fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle(
    'Economic Sensitivity Analysis — How Remaining Life Changes With Price',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('remaining_life_charts/chart04_economic_sensitivity.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart04_economic_sensitivity.png")

# Chart 5: Economic status by cluster
print("Generating Chart 5: Economic status by cluster...")

status_cluster = pd.crosstab(
    df['cluster'],
    df['economic_status'],
    normalize='index'
) * 100

col_order = [c for c in ['active', 'uneconomic', 'ended']
             if c in status_cluster.columns]
status_cluster = status_cluster[col_order]
status_cluster.index = [CLUSTER_LABELS.get(c, f'Cluster {c}')
                        for c in status_cluster.index]

status_cluster.plot(
    kind='bar', stacked=True, figsize=(12, 6),
    color=['#2ecc71', '#f39c12', '#e74c3c'],
    edgecolor='none', alpha=0.88
)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Percentage of Wells (%)', fontsize=12)
plt.title(
    'Economic Status by Cluster\n'
    'Green = active (revenue > OPEX)  |  '
    'Orange = uneconomic (revenue < OPEX)  |  '
    'Red = ended (P&A / Inactive)',
    fontsize=11, fontweight='bold'
)
plt.legend(title='Economic Status', bbox_to_anchor=(1.01, 1),
           loc='upper left', fontsize=9)
plt.xticks(rotation=15, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('remaining_life_charts/chart05_status_by_cluster.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart05_status_by_cluster.png")

# Chart 6: Remaining life vs well age (portfolio lifecycle view)
print("Generating Chart 6: Remaining life vs well age...")

sample = df.sample(min(8000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(12, 8))

for c in sorted(sample['cluster'].unique()):
    subset = sample[sample['cluster'] == c]
    ax.scatter(
        subset['Months Produced'],
        subset['remaining_life_months'].clip(0, 300),
        c=CLUSTER_COLORS.get(c, 'grey'),
        alpha=0.35, s=8,
        label=CLUSTER_LABELS.get(c, f'Cluster {c}')
    )

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Well Age (Months Produced)', fontsize=12)
ax.set_ylabel('Remaining Life (months, capped at 300)', fontsize=12)
ax.set_title(
    'Portfolio Lifecycle View — Remaining Life vs Well Age\n'
    'Each dot = one well  |  Colour = cluster',
    fontsize=13, fontweight='bold'
)
ax.legend(title='Cluster', fontsize=8, title_fontsize=9,
          bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('remaining_life_charts/chart06_lifecycle_scatter.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart06_lifecycle_scatter.png")
print()


print("=" * 65)
print("STEP 9: SAVING OUTPUT DATASET")
print("=" * 65)
print()

# Drop the intermediate recovery columns — not needed downstream
# Keep everything else including D values for dashboard use
cols_to_drop = ['last_12_gas_annual', 'last_12_oil_annual', 'last_12_boe_annual']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df.to_csv('data_with_remaining_life.csv', index=False)
print(f"✓ Saved: data_with_remaining_life.csv")
print(f"  {len(df):,} wells  ×  {len(df.columns)} columns")
print()
print("  Key new columns added:")
for col in ['monthly_gas', 'monthly_oil', 'monthly_boe',
            'monthly_revenue', 'months_since_peak',
            'D_boe', 'D_gas', 'D_oil',
            'remaining_life_months', 'remaining_life_years',
            'economic_status']:
    if col in df.columns:
        print(f"    + {col}")
print()


print("=" * 65)
print("REMAINING LIFE ESTIMATION COMPLETE — SUMMARY")
print("=" * 65)
print()
print(f"  Total wells:               {len(df):>8,}")
print(f"  Active (revenue > OPEX):   {len(active):>8,}  ({len(active)/len(df)*100:.1f}%)")
print(f"  Uneconomic:                {len(uneconomic):>8,}  ({len(uneconomic)/len(df)*100:.1f}%)")
print(f"  Ended (P&A/Inactive):      {len(ended):>8,}  ({len(ended)/len(df)*100:.1f}%)")
print()
print(f"  Economic parameters:")
print(f"    Gas:  ${GAS_PRICE:.2f}/MCF  |  Oil: ${OIL_PRICE:.2f}/BBL  "
      f"|  OPEX: ${MONTHLY_OPEX:,.0f}/month")
print()
print(f"  Remaining life by cluster (active wells only):")
for c in sorted(df['cluster'].unique()):
    sub_active = active_only[active_only['cluster'] == c]
    med  = sub_active['remaining_life_months'].median() if len(sub_active) > 0 else 0
    act  = (df[df['cluster'] == c]['economic_status'] == 'active').mean() * 100
    desc = CLUSTER_LABELS.get(c, f'Cluster {c}')
    print(f"    {desc:<28}  "
          f"Median (active only): {med:>5.0f} mo ({med/12:>4.1f} yr)  |  "
          f"Active: {act:.0f}%")
print()

if len(named_formations) > 0:
    print(f"  Top 5 formations by remaining life (active wells only):")
    for form, row in named_formations.head(5).iterrows():
        rl = row['median_remaining_life_active']
        print(f"    {form:<30}  "
              f"{rl:>5.0f} mo ({rl/12:.1f} yr)  |  "
              f"{row['well_count']:,} wells  |  "
              f"{row['pct_active']:.0f}% active")
print()
print("  Output files:")
print("    data_with_remaining_life.csv  ← full dataset + remaining life")
print("    formation_summary.csv         ← for 05_formation_ranking.py")
print("    cluster_life_summary.csv      ← for 05_formation_ranking.py")
print("    remaining_life_charts/        ← 6 charts")
print()
print("  NEXT STEP: Run 05_formation_ranking.py")
print("=" * 65)
