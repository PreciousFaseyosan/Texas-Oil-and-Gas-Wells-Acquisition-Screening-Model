# Script 01 - Exploratory Data Analysis
# Author: Precious Faseyosan
#
# Read-only inspection of the raw Enverus export. No data changes - those happen in 02_cleaning.py.
# Focuses on: productive lifespan, decline signals, water loading, formation distribution.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('eda_charts', exist_ok=True)

print("=" * 65)
print("  WELL CLUSTERING PROJECT — PHASE 1: EDA")
print("=" * 65)
print()
print("✓ Libraries imported.")
print("✓ Charts will save to 'eda_charts' folder.")
print()

# Place texas_wells_combined.csv in the same folder before running.
print("(May take 60-120 seconds for large Texas-wide export...)")
print()

df = pd.read_csv('texas_wells_combined.csv', low_memory=False)
print(f"✓ Loaded.")
print()

print("=" * 65)
print("SECTION 1: SIZE AND STRUCTURE")
print("=" * 65)
print()

rows, cols = df.shape
print(f"Total wells (rows):   {rows:,}")
print(f"Total columns:        {cols}")
print()

print("--- All column names ---")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>3}. {col}")
print()

print("--- First 3 rows ---")
print(df.head(3).to_string())
print()

print("=" * 65)
print("SECTION 2: DATA TYPES (df.info())")
print("=" * 65)
print()
df.info(verbose=True, show_counts=True)
print()

print("=" * 65)
print("SECTION 3: MISSING DATA")
print("=" * 65)
print()

missing_count = df.isnull().sum()
missing_pct   = (df.isnull().sum() / len(df) * 100).round(1)
missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %':     missing_pct
}).sort_values('Missing %', ascending=False)

print("--- All columns ---")
print(missing_summary.to_string())
print()

KEY_COLS = [
    'Peak Gas', 'Peak Oil', 'Peak BOE',
    'First 6 Gas', 'First 12 Gas', 'First 12 Oil',
    'Last 12 Gas', 'Last 12 Oil', 'Last 12 Water',
    'Cum Water', 'Months Produced',
    'True Vertical Depth', 'Gross Perforated Interval',
    'Horizontal Length', 'Well Status', 'Production Type',
    'Target Formation', 'DI Basin', 'DI Play', 'Drill Type',
    'API14', 'Well Name', 'Operator Company Name', 'County/Parish',
]

print("--- Key columns only ---")
key_miss = missing_summary[missing_summary.index.isin(KEY_COLS)]
print(key_miss.to_string())
print()

print("=" * 65)
print("SECTION 4: CATEGORICAL COLUMNS")
print("=" * 65)
print()

for col in ['Well Status', 'Production Type', 'DI Basin',
            'DI Play', 'Target Formation', 'Drill Type']:
    if col not in df.columns:
        print(f"  {col}: NOT FOUND"); continue
    print(f"--- {col} ---")
    vc  = df[col].value_counts(dropna=False)
    pct = (vc / len(df) * 100).round(1)
    print(pd.DataFrame({'Count': vc, 'Percent %': pct}).to_string())
    print()

print("=" * 65)
print("SECTION 5: NUMERIC SUMMARY")
print("=" * 65)
print()

NUM_COLS = [c for c in [
    'Peak Gas', 'First 12 Gas', 'Last 12 Gas',
    'First 12 Oil', 'Last 12 Oil', 'Cum Water',
    'Months Produced', 'True Vertical Depth',
    'Gross Perforated Interval'
] if c in df.columns]

print(df[NUM_COLS].describe().round(1).to_string())
print()

print("=" * 65)
print("SECTION 6: PRODUCTIVE LIFESPAN ANALYSIS")
print("=" * 65)
print()

if 'Months Produced' in df.columns and 'Well Status' in df.columns:

    print("--- Months Produced by Well Status ---")
    print(df.groupby('Well Status')['Months Produced'].agg(
        ['count', 'median', 'mean', 'min', 'max']
    ).round(1).to_string())
    print()

    ended = df[df['Well Status'].isin(['P & A', 'INACTIVE', 'TA'])]
    active = df[df['Well Status'] == 'ACTIVE']

    print(f"Ended wells (P&A + Inactive + TA): {len(ended):,}")
    if len(ended) > 0:
        print(f"  Median lifespan: {ended['Months Produced'].median():.0f} months "
              f"({ended['Months Produced'].median()/12:.1f} yrs)")
        print(f"  Max lifespan:    {ended['Months Produced'].max():.0f} months "
              f"({ended['Months Produced'].max()/12:.1f} yrs)")
    print()
    print(f"Active wells: {len(active):,}")
    if len(active) > 0 and 'Last 12 Gas' in df.columns:
        print(f"  Median Last 12 Gas: {active['Last 12 Gas'].median():,.0f} MCF")
    print()

print("=" * 65)
print("SECTION 7: FORMATION ANALYSIS")
print("=" * 65)
print()

if 'Target Formation' in df.columns and 'Months Produced' in df.columns:

    print("--- Top 20 Formations by Well Count ---")
    print(df['Target Formation'].value_counts().head(20).to_string())
    print()

    print("--- Top 15 Formations by Median Lifespan (>=50 wells) ---")
    form_life = df.groupby('Target Formation').agg(
        well_count    = ('Months Produced', 'count'),
        median_months = ('Months Produced', 'median'),
        mean_months   = ('Months Produced', 'mean'),
        pct_active    = ('Well Status',
                         lambda x: (x == 'ACTIVE').mean() * 100)
    ).round(1)
    form_life = form_life[form_life['well_count'] >= 50]
    print(form_life.sort_values('median_months', ascending=False)
          .head(15).to_string())
    print()

print("=" * 65)
print("SECTION 8: DECLINE SIGNAL")
print("=" * 65)
print()

if 'Last 12 Gas' in df.columns and 'Peak Gas' in df.columns:
    gas = df[(df['Last 12 Gas'].notna()) &
             (df['Peak Gas'].notna()) &
             (df['Peak Gas'] > 0)].copy()

    gas['decline_ratio_eda'] = (gas['Last 12 Gas'] / gas['Peak Gas']).clip(0, 2)

    print(f"Wells with both Last 12 Gas and Peak Gas: {len(gas):,}")
    print(f"  Median decline ratio: {gas['decline_ratio_eda'].median():.3f}")
    print(f"  Mean decline ratio:   {gas['decline_ratio_eda'].mean():.3f}")
    print()

    bins   = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    labels = ['<10% remains (near end)',
              '10-25% (late life)',
              '25-50% (declining)',
              '50-75% (mid life)',
              '75-100% (healthy)',
              '>100% (anomaly)']
    gas['decline_bin'] = pd.cut(gas['decline_ratio_eda'],
                                bins=bins, labels=labels)
    print("Decline ratio breakdown:")
    print(gas['decline_bin'].value_counts().to_string())
    print()

print("=" * 65)
print("SECTION 9: GENERATING CHARTS")
print("=" * 65)
print()

# Chart 1: Missing data
print("Chart 1: Missing data...")
top_missing = missing_summary[missing_summary['Missing %'] > 0].head(30)
colors = ['crimson' if x > 50 else 'salmon' if x > 20 else 'steelblue'
          for x in top_missing['Missing %']]
fig, ax = plt.subplots(figsize=(11, 9))
ax.barh(top_missing.index[::-1], top_missing['Missing %'][::-1],
        color=colors[::-1], alpha=0.85)
ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='50%')
ax.axvline(x=20, color='orange', linestyle='--', linewidth=1.5, label='20%')
ax.set_xlabel('% Missing', fontsize=12)
ax.set_title('Missing Data by Column', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('eda_charts/chart01_missing_data.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart01_missing_data.png")

# Chart 2: Well status
print("Chart 2: Well status...")
status_counts = df['Well Status'].value_counts()
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(status_counts.index, status_counts.values,
              color='steelblue', alpha=0.8)
for bar, val in zip(bars, status_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + status_counts.max()*0.01,
            f'{val:,}', ha='center', fontsize=8)
ax.set_xlabel('Well Status', fontsize=12)
ax.set_ylabel('Wells', fontsize=12)
ax.set_title('Well Count by Status', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
plt.savefig('eda_charts/chart02_well_status.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart02_well_status.png")

# Chart 3: Months produced
print("Chart 3: Productive lifespan...")
months_data = df['Months Produced'].dropna()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(months_data.clip(0, 600), bins=60,
             color='mediumpurple', alpha=0.8, edgecolor='none')
axes[0].axvline(x=24, color='orange', linestyle='--', linewidth=2,
                label='24-month min filter')
axes[0].axvline(x=months_data.median(), color='red', linestyle='--',
                linewidth=2, label=f'Median: {months_data.median():.0f} mo')
axes[0].set_xlabel('Months Produced', fontsize=11)
axes[0].set_ylabel('Wells', fontsize=11)
axes[0].set_title('All Wells — Productive Lifespan', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)

if 'Well Status' in df.columns:
    active_m = df[df['Well Status'] == 'ACTIVE']['Months Produced'].dropna()
    ended_m  = df[df['Well Status'].isin(
        ['P & A', 'INACTIVE'])]['Months Produced'].dropna()
    axes[1].hist(active_m.clip(0, 600), bins=50, alpha=0.6,
                 color='steelblue', label=f'Active ({len(active_m):,})', edgecolor='none')
    axes[1].hist(ended_m.clip(0, 600), bins=50, alpha=0.6,
                 color='salmon', label=f'P&A/Inactive ({len(ended_m):,})', edgecolor='none')
    axes[1].set_xlabel('Months Produced', fontsize=11)
    axes[1].set_title('Active vs Ended — Lifespan', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)

plt.suptitle('Productive Lifespan — Texas Wells',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_charts/chart03_months_produced.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart03_months_produced.png")

# Chart 4: Production type
print("Chart 4: Production type...")
prod_counts = df['Production Type'].value_counts()
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(prod_counts.index, prod_counts.values,
       color='teal', edgecolor='darkgreen', alpha=0.8)
ax.set_xlabel('Production Type', fontsize=12)
ax.set_ylabel('Wells', fontsize=12)
ax.set_title('Wells by Production Type', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
plt.savefig('eda_charts/chart04_production_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ chart04_production_type.png")

# Chart 5: Top formations
print("Chart 5: Top formations...")
if 'Target Formation' in df.columns:
    top_20 = df['Target Formation'].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(top_20.index[::-1], top_20.values[::-1],
            color='mediumseagreen', alpha=0.8, edgecolor='none')
    ax.set_xlabel('Well Count', fontsize=12)
    ax.set_title('Top 20 Formations by Well Count',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_charts/chart05_formations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ chart05_formations.png")

# Chart 6: Lifespan by formation
print("Chart 6: Lifespan by formation...")
if 'Target Formation' in df.columns and 'Months Produced' in df.columns:
    fl = df.groupby('Target Formation').agg(
        count  = ('Months Produced', 'count'),
        median = ('Months Produced', 'median')
    )
    fl = fl[fl['count'] >= 100].sort_values('median', ascending=False).head(20)
    colors_life = ['#2ecc71' if v > 200 else '#f39c12' if v > 100 else '#e74c3c'
                   for v in fl['median']]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(fl.index[::-1], fl['median'][::-1],
            color=colors_life[::-1], alpha=0.85, edgecolor='none')
    ax.axvline(x=120, color='orange', linestyle='--', linewidth=1.5, label='10 years')
    ax.axvline(x=240, color='green', linestyle='--', linewidth=1.5, label='20 years')
    ax.set_xlabel('Median Months Produced', fontsize=12)
    ax.set_title('Median Productive Lifespan by Formation\n(>=100 wells)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('eda_charts/chart06_lifespan_by_formation.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  ✓ chart06_lifespan_by_formation.png")

# Chart 7: Decline ratio
print("Chart 7: Decline ratio...")
if 'Last 12 Gas' in df.columns and 'Peak Gas' in df.columns:
    dr = df[(df['Last 12 Gas'].notna()) &
            (df['Peak Gas'].notna()) &
            (df['Peak Gas'] > 0)].copy()
    dr['dr'] = (dr['Last 12 Gas'] / dr['Peak Gas']).clip(0, 1.5)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(dr['dr'], bins=80, color='darkorange', alpha=0.85, edgecolor='none')
    ax.axvline(x=dr['dr'].median(), color='red', linestyle='--', linewidth=2,
               label=f"Median: {dr['dr'].median():.3f}")
    ax.set_xlabel('Decline Ratio (Last 12 Gas / Peak Gas)', fontsize=12)
    ax.set_ylabel('Wells', fontsize=12)
    ax.set_title('Decline Ratio Distribution\n'
                 '1.0 = still at peak  |  0.0 = no longer producing',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('eda_charts/chart07_decline_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ chart07_decline_ratio.png")

# Chart 8: Correlation heatmap
print("Chart 8: Correlation heatmap...")
corr_cols = [c for c in [
    'Peak Gas', 'First 12 Gas', 'Last 12 Gas', 'Cum Water',
    'Months Produced', 'True Vertical Depth', 'Gross Perforated Interval'
] if c in df.columns]
corr_data = df[corr_cols].dropna()
if len(corr_data) > 0:
    corr_matrix = np.log1p(corr_data.clip(lower=0)).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 9})
    ax.set_title('Feature Correlation Heatmap (Log Scale)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('eda_charts/chart08_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ chart08_correlation.png")
print()

print("=" * 65)
print("EDA SUMMARY")
print("=" * 65)
print()

producing_types = ['GAS', 'OIL & GAS', 'OIL']
valid_statuses  = ['ACTIVE', 'INACTIVE', 'P & A', 'SHUT-IN', 'TA', 'COMPLETED']

candidates = df[
    df['Production Type'].isin(producing_types) &
    df['Well Status'].isin(valid_statuses) &
    df['Months Produced'].notna() &
    (df['Months Produced'] >= 24) &
    (df['Peak Gas'].notna() | df['First 12 Gas'].notna())
]

print(f"  Raw dataset:                    {len(df):>10,} wells")
print(f"  Est. usable after cleaning:     {len(candidates):>10,} wells")
print()
print(f"  Months Produced — median: {df['Months Produced'].median():.0f} months "
      f"({df['Months Produced'].median()/12:.1f} years)")
print(f"  Months Produced — max:    {df['Months Produced'].max():.0f} months "
      f"({df['Months Produced'].max()/12:.1f} years)")
print()
print("=" * 65)
print("  EDA COMPLETE. Review charts, then run 02_cleaning.py")
print("=" * 65)
