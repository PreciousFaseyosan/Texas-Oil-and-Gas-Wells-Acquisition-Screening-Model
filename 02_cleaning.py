# Script 02 - Data Cleaning & Preparation
# Author: Precious Faseyosan
#
# Takes the raw Enverus export and produces a clean, feature-engineered dataset for K-Means clustering.
# Key design decisions:
#   - P&A wells kept: they document completed lifecycles and are essential for formation longevity data
#   - Recompletion flag (had_recompletion) derived from the last two digits of the API14
#   - Minimum 24 months of production history required for meaningful decline features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('cleaning_charts', exist_ok=True)

print("=" * 65)
print("  WELL CLUSTERING — PHASE 2: DATA CLEANING")
print("=" * 65)
print()

print("=" * 65)
print("STEP 1: LOADING DATA")
print("=" * 65)
print()

# Place texas_wells_combined.csv in the same folder before running.
df_raw = pd.read_csv('texas_wells_combined.csv', low_memory=False)

print(f"✓ Raw data loaded: {len(df_raw):,} wells, {len(df_raw.columns)} columns")
print()

# Work on a copy — never modify the raw data
df = df_raw.copy()


# Standardise BEFORE filtering - same formation appears under multiple Enverus labels
# (e.g. EAGLEFORD, EAGLE FORD-1, EAGLEFORD SHALE). Corrections must apply before any row drops.
print("=" * 65)
print("STEP 2: STANDARDISING FORMATION NAMES")
print("=" * 65)
print()

FORMATION_MAP = {
    # Eagle Ford variants
    'EAGLE FORD':          'EAGLEFORD',
    'EAGLE FORD-1':        'EAGLEFORD',
    'EAGLE FORD-2':        'EAGLEFORD',
    'EAGLEFORD SHALE':     'EAGLEFORD',
    'EAGLEFORD SHALE GAS': 'EAGLEFORD',
    # Austin Chalk variants
    'AUSTIN CHALK-1':      'AUSTIN CHALK',
    'AUSTIN CHALK-2':      'AUSTIN CHALK',
    'AUSTIN CHALK-3':      'AUSTIN CHALK',
    'AUSTIN CHALK 3':      'AUSTIN CHALK',
    'AUSTIN CHALK, GAS':   'AUSTIN CHALK',
    # Lobo variants
    'LOBO CONS.':          'LOBO',
    'LOBO CONS':           'LOBO',
    'LOBO,WILCOX':         'LOBO',
    # Haynesville variants
    'HAYNESVILLE SHALE':   'HAYNESVILLE',
    'HAYNESVILLE-BOSSIER': 'HAYNESVILLE',
    # Barnett variants
    'BARNETT SHALE':       'BARNETT',
    # Frio variants
    'CONSOLIDATED FRIO':   'FRIO',
    # Cotton Valley variants
    'COTTON VALLEY SAND':  'COTTON VALLEY',
    'COTTON VALLEY-1':     'COTTON VALLEY',
}

affected = df['Target Formation'].isin(FORMATION_MAP.keys()).sum()
df['Target Formation'] = df['Target Formation'].replace(FORMATION_MAP)

print(f"Standardised {len(FORMATION_MAP)} formation name variants")
print(f"Wells affected: {affected:,}")
print()
for old, new in FORMATION_MAP.items():
    print(f"  '{old}'  →  '{new}'")
print()


# Keep only hydrocarbon producers - remove dry holes, disposal, injection, water wells, etc.
print("=" * 65)
print("STEP 3: FILTERING PRODUCTION TYPES")
print("=" * 65)
print()

KEEP_PROD_TYPES = ['GAS', 'OIL & GAS', 'OIL']

before = len(df)
df = df[df['Production Type'].isin(KEEP_PROD_TYPES)]
after = len(df)
print(f"Kept production types: {KEEP_PROD_TYPES}")
print(f"Removed: {before - after:,} wells  |  Remaining: {after:,}")
print()


# P&A kept: completed lifecycle (Months Produced = actual lifespan) is essential formation longevity data.
# INACTIVE kept: most reached economic limit organically - Months Produced ~= true total lifespan.
print("=" * 65)
print("STEP 4: FILTERING WELL STATUSES")
print("=" * 65)
print()

KEEP_STATUSES = ['ACTIVE', 'INACTIVE', 'P & A',
                 'SHUT-IN', 'TA']

before = len(df)
df = df[df['Well Status'].isin(KEEP_STATUSES)]
after = len(df)
print(f"Kept statuses: {KEEP_STATUSES}")
print(f"Removed: {before - after:,} wells  |  Remaining: {after:,}")
print()


# WILDCAT = exploratory, no established formation. EDA: 42% dry holes, median production 8 months.
print("=" * 65)
print("STEP 5: EXCLUDING WILDCAT WELLS")
print("=" * 65)
print()

before = len(df)
df = df[df['Target Formation'].str.upper() != 'WILDCAT']
after = len(df)
print(f"Removed: {before - after:,} WILDCAT wells  |  Remaining: {after:,}")
print()


# API14 last 2 digits = completion code (00=original, 01+=recompletion).
# Enverus creates a separate row per recompletion, but ALL production lives on the 00 row.
# Recompletion rows (01+) are empty registration records - keeping them double-counts the well.
print("=" * 65)
print("STEP 6: HANDLING API14 RECOMPLETION DUPLICATES")
print("=" * 65)
print()

# Check that API14 column exists
if 'API14' not in df.columns:
    print("WARNING: API14 column not found. Skipping deduplication.")
    df['had_recompletion'] = 0
else:
    # Convert API14 to string and clean (remove dashes, spaces)
    df['API14_clean'] = (
        df['API14'].astype(str)
        .str.replace('-', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.strip()
    )

    # Extract base API (first 12 digits = state + county + well number)
    # and completion code (last 2 digits)
    df['API12_base'] = df['API14_clean'].str[:12]
    df['completion_code'] = df['API14_clean'].str[12:14]

    # Convert completion code to numeric for sorting
    # Non-numeric codes default to 0
    df['completion_code_num'] = pd.to_numeric(
        df['completion_code'], errors='coerce'
    ).fillna(0).astype(int)

    # Before dedup stats
    total_before = len(df)
    base_apis = df['API12_base'].nunique()
    recompletion_wells = df[df['completion_code_num'] > 0]['API12_base'].nunique()

    print(f"Before deduplication:")
    print(f"  Total rows:                {total_before:,}")
    print(f"  Unique base APIs:          {base_apis:,}")
    print(f"  Wells with recompletions:  {recompletion_wells:,}")
    print(f"  Duplicate rows to remove:  {total_before - base_apis:,}")
    print()

    # Flag wells that have ANY recompletion
    # (any base API that appears with completion_code > 0)
    apis_with_recompletion = (
        df[df['completion_code_num'] > 0]['API12_base'].unique()
    )
    df['had_recompletion'] = df['API12_base'].isin(
        apis_with_recompletion
    ).astype(int)

    print(f"  Wells flagged as recompleted: {df['had_recompletion'].sum():,}")
    print()

    # Keep only the 00 row: Enverus accumulates ALL production (including post-recompletion)
    # on the original completion. The 01+ rows are empty permit records with no production data.
    df = df[df['completion_code_num'] == 0]

    print(f"After deduplication (kept original completions only):")
    print(f"  Unique wellbores remaining: {len(df):,}")
    print(f"  Rows removed:               {total_before - len(df):,}")
    print()

    # Clean up temporary columns
    df = df.drop(columns=['API14_clean', 'API12_base',
                           'completion_code', 'completion_code_num'],
                 errors='ignore')


# Require 24+ months: enough history for meaningful decline rate and First 12 features.
# Also require at least one production metric (gas OR oil OR BOE) to compute decline features.
print("=" * 65)
print("STEP 7: MINIMUM PRODUCTION HISTORY FILTER")
print("=" * 65)
print()
 
before = len(df)
df = df[
    df['Months Produced'].notna() &
    (df['Months Produced'] >= 24)
]
after = len(df)
print(f"Filter: Months Produced >= 24")
print(f"Removed: {before - after:,}  |  Remaining: {after:,}")
print()
 
before = len(df)
df = df[
    df['Peak Gas'].notna()    | df['First 12 Gas'].notna() |   # gas well data
    df['Peak Oil'].notna()    | df['First 12 Oil'].notna() |   # oil well data
    df['Peak BOE'].notna()    | df['First 12 BOE'].notna()     # combined data
]
after = len(df)
print(f"Filter: Must have production data (gas OR oil OR BOE)")
print(f"Removed: {before - after:,}  |  Remaining: {after:,}")
print()


print("=" * 65)
print("STEP 8: SELECTING COLUMNS")
print("=" * 65)
print()
 
IDENTIFIER_COLS = [
    'API14',
    'Well Name',
    'Operator Company Name',
    'Field',
    'County/Parish',
    'State',
]
 
FEATURE_COLS = [
    # Gas features - mirrored by oil equivalents to avoid biasing toward the 55% gas-well majority
    'Peak Gas',
    'First 6 Gas',
    'First 12 Gas',
    'Last 12 Gas',
    'Cum Gas',
 
    # Oil features (symmetric with gas)
    'Peak Oil',
    'First 6 Oil',
    'First 12 Oil',
    'Last 12 Oil',
    'Cum Oil',
 
    # Water features - No Peak Water in Enverus (not tracked the same way).
    # Rising water cuts accelerate decline and lifting costs.
    'First 6 Water',
    'First 12 Water',
    'Last 12 Water',
    'Cum Water',
 
    # Combined BOE rate features
    'Peak BOE',       # peak combined oil+gas rate in BOE/month
    'First 12 BOE',   # first year combined rate
    'First 6 BOE',    # first 6 months combined rate
 
    'Months Produced',
    'True Vertical Depth',
    'Gross Perforated Interval',
    'Horizontal Length',
    'Production Type',
    'Target Formation',
    'DI Basin',
    'DI Play',
    'Drill Type',
    'Well Status',
    'had_recompletion',  # from Step 6 deduplication
]
 
ALL_COLS = IDENTIFIER_COLS + FEATURE_COLS
keep_cols = [c for c in ALL_COLS if c in df.columns]
skipped   = [c for c in ALL_COLS if c not in df.columns]
 
if skipped:
    print(f"WARNING: Columns not found (will be skipped):")
    for c in skipped:
        print(f"  - {c}")
    print()
 
df = df[keep_cols]
print(f"✓ Kept {len(keep_cols)} columns (from original {len(df_raw.columns)})")
print()


print("=" * 65)
print("STEP 9: HANDLING MISSING VALUES")
print("=" * 65)
print()

# Check missing BEFORE
missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0].sort_values(ascending=False)
print("Missing values BEFORE:")
for col, n in missing_before.items():
    print(f"  {col:<40} {n:>7,}  ({n/len(df)*100:.1f}%)")
print()

# Numeric: fill with median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("Filling numeric columns with MEDIAN:")
for col in numeric_cols:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"  {col:<40} {n_miss:>7,} filled  (median={med:,.1f})")
print()

# Categorical: fill with UNKNOWN
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_features = [c for c in cat_cols if c in FEATURE_COLS]

print("Filling categorical columns with 'UNKNOWN':")
for col in cat_features:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        df[col] = df[col].fillna('UNKNOWN')
        print(f"  {col:<40} {n_miss:>7,} filled")
print()

# Check AFTER
remaining_miss = df.isnull().sum().sum()
print(f"Missing values remaining: {remaining_miss}")
if remaining_miss == 0:
    print("  ✓ No missing values remain.")
print()


# Feature engineering: petroleum engineering concepts in a form the clustering model can learn from.
print("=" * 65)
print("STEP 10: FEATURE ENGINEERING")
print("=" * 65)
print()
 
# Enverus does not publish Last 12 BOE directly - compute from Last 12 Gas + Last 12 Oil.
# Standard conversion: 1 BBL = 6 MCF
 
df['last_12_boe_equiv'] = (
    df['Last 12 Gas'] / 6 + df['Last 12 Oil']
)
print("✓ Computed: last_12_boe_equiv = Last12Gas/6 + Last12Oil")
print(f"  Median: {df['last_12_boe_equiv'].median():.1f} BOE/month")
print()
 
# Feature 1a: Gas Decline Ratio - for pure oil wells this is near 0; captured by 1b instead.
 
df['decline_ratio_gas'] = (
    df['Last 12 Gas'] / (df['Peak Gas'] + 1)
).clip(lower=0.001, upper=1.5)
print(f"✓ decline_ratio_gas  = Last12Gas / Peak Gas")
print(f"  Median: {df['decline_ratio_gas'].median():.3f}")
 
# Feature 1b: Oil Decline Ratio - symmetric with gas; near 0 for pure gas wells.
 
df['decline_ratio_oil'] = (
    df['Last 12 Oil'] / (df['Peak Oil'] + 1)
).clip(lower=0.001, upper=1.5)
print(f"✓ decline_ratio_oil  = Last12Oil / Peak Oil")
print(f"  Median: {df['decline_ratio_oil'].median():.3f}")
 
# Feature 1c: BOE Decline Ratio (primary) - works for ALL well types.
 
df['decline_ratio_boe'] = (
    df['last_12_boe_equiv'] / (df['Peak BOE'] + 1)
).clip(lower=0.001, upper=1.5)
print(f"✓ decline_ratio_boe  = Last12BOE_equiv / Peak BOE  (universal)")
print(f"  Median: {df['decline_ratio_boe'].median():.3f}")
print()
 
# Feature 2a: Water-Gas Ratio - for oil wells (Last 12 Gas near 0) use 2b instead.
 
df['water_gas_ratio'] = (
    df['Cum Water'] / (df['Last 12 Gas'] + 1)
)
cap_wgr = df['water_gas_ratio'].quantile(0.99)
df['water_gas_ratio'] = df['water_gas_ratio'].clip(lower=0, upper=cap_wgr)
print(f"✓ water_gas_ratio    = Cum Water / Last12Gas")
print(f"  Median: {df['water_gas_ratio'].median():.2f}  (capped at {cap_wgr:.2f})")
 
# Feature 2b: Water-Oil Ratio (WOR) - symmetric with 2a; standard petroleum engineering term.
 
df['water_oil_ratio'] = (
    df['Cum Water'] / (df['Last 12 Oil'] + 1)
)
cap_wor = df['water_oil_ratio'].quantile(0.99)
df['water_oil_ratio'] = df['water_oil_ratio'].clip(lower=0, upper=cap_wor)
print(f"✓ water_oil_ratio    = Cum Water / Last12Oil")
print(f"  Median: {df['water_oil_ratio'].median():.2f}  (capped at {cap_wor:.2f})")
 
# Feature 2c: Water-BOE Ratio (primary) - works for ALL well types.
 
df['water_boe_ratio'] = (
    df['Cum Water'] / (df['last_12_boe_equiv'] + 1)
)
cap_wbr = df['water_boe_ratio'].quantile(0.99)
df['water_boe_ratio'] = df['water_boe_ratio'].clip(lower=0, upper=cap_wbr)
print(f"✓ water_boe_ratio    = Cum Water / Last12BOE_equiv  (universal)")
print(f"  Median: {df['water_boe_ratio'].median():.2f}  (capped at {cap_wbr:.2f})")
print()
 
# Feature 3: First-year BOE per perforated foot - equally valid for gas and oil wells.
 
df['first_yr_pi'] = (
    df['First 12 BOE'] / (df['Gross Perforated Interval'] + 1)
)
print(f"✓ first_yr_pi        = First12BOE / Perforated Interval  (BOE/ft)")
print(f"  Median: {df['first_yr_pi'].median():.2f}")
print()
 
# Feature 4: Horizontal flag - different decline characteristics vs vertical wells.
 
df['is_horizontal'] = (df['Drill Type'] == 'H').astype(int)
print(f"✓ is_horizontal      = 1 if Drill Type H, else 0")
print(f"  Horizontal wells: {df['is_horizontal'].sum():,}  ({df['is_horizontal'].mean()*100:.1f}%)")
print()
 
# Feature 5: Log-transform - production spans single digits to millions.
# np.log1p handles zeros safely; applied symmetrically to gas, oil, water, BOE.
 
LOG_COLS = [
    # Gas
    'Peak Gas', 'First 6 Gas', 'First 12 Gas', 'Last 12 Gas', 'Cum Gas',
    # Oil (symmetric with gas)
    'Peak Oil', 'First 6 Oil', 'First 12 Oil', 'Last 12 Oil', 'Cum Oil',
    # Water
    'First 6 Water', 'First 12 Water', 'Last 12 Water', 'Cum Water',
    # Combined BOE
    'Peak BOE', 'First 6 BOE', 'First 12 BOE',
    # Derived
    'last_12_boe_equiv',
]
 
print("Log-transforming production features:")
for col in LOG_COLS:
    if col in df.columns:
        new_col = 'log_' + col.replace(' ', '_')
        df[new_col] = np.log1p(df[col].clip(lower=0))
        print(f"  ✓ {new_col}")
print()



print("=" * 65)
print("STEP 11: ENCODING CATEGORICAL VARIABLES")
print("=" * 65)
print()

# Group rare formations — keep top N, rest → OTHER_FORMATION
N_TOP = 20

top_formations = (
    df['Target Formation'].value_counts()
    .nlargest(N_TOP).index
)
df['Formation_Grouped'] = df['Target Formation'].where(
    df['Target Formation'].isin(top_formations),
    other='OTHER_FORMATION'
)

print(f"✓ Grouped Target Formation: top {N_TOP} + 'OTHER_FORMATION'")
print()
print("Formation groups:")
for f, n in df['Formation_Grouped'].value_counts().items():
    print(f"  {f:<40} {n:>7,} wells")
print()

# One-hot encode categorical features
ENCODE_COLS = ['Production Type', 'Formation_Grouped',
               'DI Basin', 'Drill Type']

# Save distributions before encoding consumes categorical columns
prod_type_dist = df['Production Type'].value_counts()
basin_dist_saved = df['DI Basin'].value_counts() if 'DI Basin' in df.columns else None

df_encoded = pd.get_dummies(
    df, columns=ENCODE_COLS,
    drop_first=False, dtype=int
)

new_dummies = [c for c in df_encoded.columns if c not in df.columns]
print(f"✓ One-hot encoding created {len(new_dummies)} binary columns")
df = df_encoded
print(f"Dataset shape after encoding: {df.shape[0]:,} × {df.shape[1]}")
print()


# Over-represented groups pull cluster centroids and obscure minority patterns.
# Flag if any status or formation dominates >60% of dataset.
print("=" * 65)
print("STEP 12: CLASS IMBALANCE CHECK")
print("=" * 65)
print()

os.makedirs('cleaning_charts', exist_ok=True)

# --- Well Status ---
print("Well Status distribution:")
status_dist = df['Well Status'].value_counts()
for s, n in status_dist.items():
    bar = '█' * int(n / status_dist.max() * 30)
    print(f"  {s:<15} {n:>7,}  ({n/len(df)*100:>5.1f}%)  {bar}")
print()

# --- Formation ---
print("Formation distribution (top 15):")
form_dist = df['Target Formation'].value_counts().head(15)
for f, n in form_dist.items():
    bar = '█' * int(n / form_dist.max() * 30)
    print(f"  {f:<35} {n:>7,}  ({n/len(df)*100:>5.1f}%)  {bar}")
print()

# --- Production Type ---
print("Production Type distribution:")
pt_dist = prod_type_dist
for p, n in pt_dist.items():
    bar = '█' * int(n / pt_dist.max() * 30)
    print(f"  {p:<15} {n:>7,}  ({n/len(df)*100:>5.1f}%)  {bar}")
print()

# --- Basin ---
if 'DI Basin' in df.columns:
    print("Basin distribution:")
    basin_dist = basin_dist_saved
    for b, n in basin_dist.items():
        bar = '█' * int(n / basin_dist.max() * 30)
        print(f"  {b:<30} {n:>7,}  ({n/len(df)*100:>5.1f}%)  {bar}")
    print()

# --- Imbalance flag ---
max_status_pct = status_dist.iloc[0] / len(df) * 100
max_form_pct   = df['Target Formation'].value_counts().iloc[0] / len(df) * 100

print("Imbalance Assessment:")
if max_status_pct > 60:
    print(f"  ⚠ WELL STATUS: dominant category = {max_status_pct:.0f}%")
    print(f"    Recommendation: acknowledge in thesis as dataset characteristic.")
    print(f"    The imbalance reflects the actual Texas well population —")
    print(f"    not a sampling artefact. Report it, do not over-correct it.")
else:
    print(f"  ✓ Well status distribution: acceptable (max {max_status_pct:.0f}%)")

if max_form_pct > 30:
    print(f"  ⚠ FORMATION: dominant category = {max_form_pct:.1f}%")
    print(f"    Monitor clustering output — dominant formation may anchor clusters.")
else:
    print(f"  ✓ Formation distribution: acceptable (max {max_form_pct:.1f}%)")
print()

# --- Imbalance chart ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Well status
ax = axes[0, 0]
status_df = df['Well Status'].value_counts()
ax.bar(status_df.index, status_df.values, color='steelblue', alpha=0.8)
ax.set_title('Well Status Distribution', fontweight='bold')
ax.set_ylabel('Well Count')
ax.tick_params(axis='x', rotation=25)
for i, (idx, val) in enumerate(status_df.items()):
    ax.text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=8)

# Formation top 15
ax = axes[0, 1]
top15 = df['Target Formation'].value_counts().head(15)
ax.barh(top15.index[::-1], top15.values[::-1], color='mediumseagreen', alpha=0.8)
ax.set_title('Top 15 Formations', fontweight='bold')
ax.set_xlabel('Well Count')

# Production type
ax = axes[1, 0]
pt_cols = [c for c in df.columns if c.startswith('Production Type_')]
pt = df[pt_cols].sum().sort_values(ascending=False)
pt.index = pt.index.str.replace('Production Type_', '', regex=False)
ax.bar(pt.index, pt.values, color='darkorange', alpha=0.8)
ax.set_title('Production Type Distribution', fontweight='bold')
ax.set_ylabel('Well Count')
ax.tick_params(axis='x', rotation=25)

# Basin
ax = axes[1, 1]
if 'DI Basin' in df.columns:
    basin_cols = [c for c in df.columns if c.startswith('DI Basin_')]
    basin = df[basin_cols].sum().sort_values(ascending=False)
    basin.index = basin.index.str.replace('DI Basin_', '', regex=False)
    ax.bar(basin.index, basin.values, color='mediumpurple', alpha=0.8)
    ax.set_title('Basin Distribution', fontweight='bold')
    ax.set_ylabel('Well Count')
    ax.tick_params(axis='x', rotation=25)

plt.suptitle('Class Imbalance Check — Clean Dataset',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('cleaning_charts/chart01_imbalance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: cleaning_charts/chart01_imbalance.png")
print()


print("=" * 65)
print("STEP 13: SAVING OUTPUTS")
print("=" * 65)
print()
 
# Define final feature set for clustering
CLUSTER_FEATURES = [
    # --- Decline features (gas, oil, and universal BOE) ---
    'decline_ratio_gas',     # gas decline signal
    'decline_ratio_oil',     # oil decline signal
    'decline_ratio_boe',     # universal decline signal (primary)
    # --- Water loading features (gas, oil, and universal BOE) ---
    'water_gas_ratio',       # water loading for gas wells
    'water_oil_ratio',       # water loading for oil wells (WOR)
    'water_boe_ratio',       # universal water loading (primary)
    # --- Productivity and age ---
    'first_yr_pi',           # first year BOE/ft (universal)
    'Months Produced',       # well age / lifecycle stage
    # --- Current production level (log scale) ---
    'log_last_12_boe_equiv', # current combined rate (universal)
    'log_Last_12_Gas',       # current gas rate
    'log_Last_12_Oil',       # current oil rate
    # --- Well construction ---
    'True Vertical Depth',
    'is_horizontal',
    'had_recompletion',
]
 
# Add production type and basin dummies
DUMMY_COLS = [
    c for c in df.columns
    if any(c.startswith(base + '_')
           for base in ['Production Type', 'Formation_Grouped',
                        'DI Basin', 'Drill Type'])
]
 
CLUSTER_FEATURES_ALL = [
    c for c in CLUSTER_FEATURES + DUMMY_COLS if c in df.columns
]
 
# Save feature list
with open('feature_list.json', 'w') as f:
    json.dump({
        'cluster_features': CLUSTER_FEATURES_ALL,
        'identifier_cols':  [c for c in IDENTIFIER_COLS if c in df.columns],
    }, f, indent=2)
print("✓ Saved: feature_list.json")
 
# Save full clean dataset (with identifiers)
df.to_csv('data_clean_full.csv', index=False)
print(f"✓ Saved: data_clean_full.csv  ({len(df):,} wells × {len(df.columns)} cols)")
 
# Save model-ready dataset (features only, no text identifiers)
id_cols_present = [c for c in IDENTIFIER_COLS if c in df.columns]
keep_for_model  = id_cols_present + ['Well Status', 'Target Formation',
                                     'DI Basin', 'Production Type'] + \
                  CLUSTER_FEATURES_ALL
keep_for_model  = [c for c in keep_for_model if c in df.columns]
keep_for_model  = list(dict.fromkeys(keep_for_model))  # deduplicate
 
df[keep_for_model].to_csv('data_model_ready.csv', index=False)
print(f"✓ Saved: data_model_ready.csv  ({len(df):,} wells × {len(keep_for_model)} cols)")
print()



print("=" * 65)
print("CLEANING COMPLETE — SUMMARY")
print("=" * 65)
print()
print(f"  Raw wells:               {len(df_raw):>10,}")
print(f"  After cleaning:          {len(df):>10,}")
print(f"  Removed:                 {len(df_raw)-len(df):>10,}  "
      f"({(len(df_raw)-len(df))/len(df_raw)*100:.0f}%)")
print()
print(f"  Well Status breakdown:")
for s, n in df['Well Status'].value_counts().items():
    print(f"    {s:<15} {n:>8,}  ({n/len(df)*100:.1f}%)")
print()
print(f"  Production Type breakdown:")
pt_cols = [c for c in df.columns if c.startswith('Production Type_')]
pt_final = df[pt_cols].sum().sort_values(ascending=False)
pt_final.index = pt_final.index.str.replace('Production Type_', '', regex=False)
for p, n in pt_final.items():
    print(f"    {p:<15} {n:>8,}  ({n/len(df)*100:.1f}%)")
print()
print(f"  Months Produced:")
print(f"    Median: {df['Months Produced'].median():.0f} months  "
      f"({df['Months Produced'].median()/12:.1f} years)")
print(f"    Max:    {df['Months Produced'].max():.0f} months  "
      f"({df['Months Produced'].max()/12:.1f} years)")
print()
print(f"  Cluster features ready:  {len(CLUSTER_FEATURES_ALL)}")
print()
print("  Output files:")
print("    data_clean_full.csv    ← full clean dataset with identifiers")
print("    data_model_ready.csv   ← clustering-ready feature set")
print("    feature_list.json      ← feature names for next scripts")
print()
print("  NEXT STEP: Run 04_clustering.py")
print("=" * 65)
