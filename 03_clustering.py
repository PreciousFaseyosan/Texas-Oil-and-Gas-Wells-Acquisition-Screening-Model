# Script 03 - K-Means Well Performance Clustering
# Author: Precious Faseyosan
#
# Clusters Texas wells into 5 behavioural groups using K-Means on production behaviour features.
# Clusters reflect HOW wells behave (decline, water, output) - not their formation labels.
# Formation names become interpretive labels after cluster assignment.
#
# Outputs: cluster_model.joblib, cluster_scaler.joblib, cluster_metadata.json,
#          data_clustered.csv, cluster_profiles.csv, cluster_charts/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import warnings
import os
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('cluster_charts', exist_ok=True)

print("=" * 65)
print("  WELL PERFORMANCE CLUSTERING — ENGINE 1")
print("=" * 65)
print()
print("✓ Libraries imported.")
print()


print("=" * 65)
print("STEP 1: LOADING DATA")
print("=" * 65)
print()

# Place data_model_ready.csv in the same folder (output of 02_cleaning.py).
df = pd.read_csv('data_model_ready.csv', low_memory=False)
print(f"✓ Loaded: {len(df):,} wells  ×  {len(df.columns)} columns")
print()

# Load feature list from cleaning script output
with open('feature_list.json', 'r') as f:
    feature_info = json.load(f)

ALL_FEATURES   = feature_info['cluster_features']
IDENTIFIER_COLS = feature_info['identifier_cols']

print(f"Features from feature_list.json: {len(ALL_FEATURES)}")
print()


# EAGLE FORD SHALE variant was missed in 02_cleaning.py's FORMATION_MAP - merge here.
print("=" * 65)
print("STEP 2: FIXING REMAINING FORMATION INCONSISTENCY")
print("=" * 65)
print()

if ('Formation_Grouped_EAGLE FORD SHALE' in df.columns and
        'Formation_Grouped_EAGLEFORD' in df.columns):
    before = df['Formation_Grouped_EAGLEFORD'].sum()
    df['Formation_Grouped_EAGLEFORD'] = (
        df['Formation_Grouped_EAGLEFORD'] +
        df['Formation_Grouped_EAGLE FORD SHALE']
    ).clip(upper=1)
    df = df.drop(columns=['Formation_Grouped_EAGLE FORD SHALE'])
    after = df['Formation_Grouped_EAGLEFORD'].sum()
    print(f"✓ Merged EAGLE FORD SHALE → EAGLEFORD")
    print(f"  EAGLEFORD wells: {before:,.0f} → {after:,.0f}")
    print()
    # Update feature list to remove merged column
    ALL_FEATURES = [f for f in ALL_FEATURES
                    if f != 'Formation_Grouped_EAGLE FORD SHALE']
    print(f"Updated feature count: {len(ALL_FEATURES)}")
else:
    print("No EAGLE FORD SHALE column found — no merge needed.")
print()
    
# Fix raw Target Formation column for heatmap clarity
df['Target Formation'] = df['Target Formation'].replace(
    'EAGLE FORD SHALE', 'EAGLEFORD'
)
print("✓ Fixed raw Target Formation: EAGLE FORD SHALE → EAGLEFORD")

# 15 behaviour features selected from 46+ available - curse of dimensionality risk with more.
# Formation/basin dummies excluded: clusters should reflect behaviour, not formation labels.
# Including them would anchor clusters to the 44% OTHER_FORMATION + 18% UNKNOWN majority.
print("=" * 65)
print("STEP 3: DEFINING CLUSTERING FEATURES")
print("=" * 65)
print()

CLUSTER_FEATURES = [
    # Decline — gas, oil, and universal BOE
    'decline_ratio_gas',
    'decline_ratio_oil',
    'decline_ratio_boe',
    # Water loading — gas, oil, and universal BOE
    'water_gas_ratio',
    'water_oil_ratio',
    'water_boe_ratio',
    # Productivity and age
    'first_yr_pi',
    'Months Produced',
    # Current production level (log scale — universal)
    'log_last_12_boe_equiv',
    # Geological context
    'True Vertical Depth',
    # Well type
    'is_horizontal',
    'had_recompletion',
    # Production type (gas vs oil is behaviorally meaningful)
    'Production Type_GAS',
    'Production Type_OIL',
    'Production Type_OIL & GAS',
]

# Verify all features exist in the dataframe
missing_feats = [f for f in CLUSTER_FEATURES if f not in df.columns]
if missing_feats:
    print(f"WARNING: Missing features — will be dropped:")
    for f in missing_feats:
        print(f"  - {f}")
    CLUSTER_FEATURES = [f for f in CLUSTER_FEATURES if f in df.columns]

print(f"Clustering features ({len(CLUSTER_FEATURES)} total):")
for f in CLUSTER_FEATURES:
    print(f"  - {f}")
print()

X_cluster = df[CLUSTER_FEATURES].copy().dropna()
valid_idx  = X_cluster.index

print(f"Wells with complete clustering features: {len(X_cluster):,}")
if len(df) - len(X_cluster) > 0:
    print(f"  (Dropped {len(df)-len(X_cluster):,} rows with missing values)")
print()


# StandardScaler (mean=0, std=1): without scaling, Months Produced (range ~1000)
# would dominate distance calculations over is_horizontal (range 1).
# Fit on training data and save - apply the same scaler to any new wells.
print("=" * 65)
print("STEP 4: SCALING FEATURES")
print("=" * 65)
print()

cluster_scaler = StandardScaler()
X_scaled = cluster_scaler.fit_transform(X_cluster)

print(f"✓ StandardScaler applied: mean→0, std→1 per feature")
print(f"  Scaled matrix shape: {X_scaled.shape}")
print()


# Elbow method (inertia) + silhouette score, tested K=2-12 on a 10,000-well sample.
# Final model trains on all wells.
print("=" * 65)
print("STEP 5: FINDING OPTIMAL K")
print("=" * 65)
print()
print("Testing K = 2 to 12 on a sample of 10,000 wells...")
print("(Using sample for speed — final model uses all wells)")
print()

SAMPLE_SIZE = min(10000, len(X_scaled))
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), SAMPLE_SIZE, replace=False)
X_sample   = X_scaled[sample_idx]

K_RANGE    = range(2, 13)
inertias   = []
silhouettes = []

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_sample)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_sample, labels, sample_size=3000, random_state=42)
    silhouettes.append(sil)
    print(f"  K={k}:  Inertia={km.inertia_:>14,.0f}  |  Silhouette={sil:.4f}")

print()
best_k_sil = list(K_RANGE)[np.argmax(silhouettes)]
print(f"Best K by Silhouette Score: K = {best_k_sil}")
print()

# --- Chart 1: Elbow and Silhouette ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_RANGE), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Method\n(Look for where improvement slows)',
                  fontsize=12, fontweight='bold')
axes[0].set_xticks(list(K_RANGE))

axes[1].plot(list(K_RANGE), silhouettes, 'ro-', linewidth=2, markersize=8)
axes[1].axvline(x=best_k_sil, color='green', linestyle='--', linewidth=2,
                label=f'Best K = {best_k_sil}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score (higher = better)', fontsize=12)
axes[1].set_title('Silhouette Score\n(Higher = more distinct clusters)',
                  fontsize=12, fontweight='bold')
axes[1].set_xticks(list(K_RANGE))
axes[1].legend(fontsize=10)

plt.suptitle('Optimal Number of Clusters — Elbow & Silhouette',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('cluster_charts/chart01_optimal_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: cluster_charts/chart01_optimal_k.png")
print()
print("IMPORTANT: Review chart01_optimal_k.png before proceeding.")
print("If the elbow chart suggests a different K than the silhouette,")
print("use your petroleum engineering judgment to decide.")
print(f"Current selection: K = {best_k_sil}")
print("To override, change FINAL_K below before continuing.")
print()


# n_init=20: run 20 times with different starting points, keep the best result.
FINAL_K = 5   # overriding silhouette - elbow and interpretability support K=5

print("=" * 65)
print(f"STEP 6: TRAINING FINAL K-MEANS  (K = {FINAL_K})")
print("=" * 65)
print()
print(f"Training on all {len(X_scaled):,} wells...")
print("(May take 3-6 minutes)")
print()

final_kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=20,
    max_iter=500
)
cluster_labels = final_kmeans.fit_predict(X_scaled)

print(f"✓ K-Means trained.")
print()

unique, counts = np.unique(cluster_labels, return_counts=True)
print("Wells per cluster:")
for c, n in zip(unique, counts):
    pct = n / len(cluster_labels) * 100
    bar = '█' * int(pct / 2)
    print(f"  Cluster {c}: {n:>7,} wells  ({pct:.1f}%)  {bar}")
print()


print("=" * 65)
print("STEP 7: ATTACHING CLUSTER LABELS")
print("=" * 65)
print()

df_clustered = df.loc[valid_idx].copy()
df_clustered['cluster'] = cluster_labels

# Rename clusters: shift from 0-based (0,1,2,3,4) to 1-based (1,2,3,4,5)
df_clustered['cluster'] = df_clustered['cluster'] + 1
cluster_labels = cluster_labels + 1

print(f"✓ Cluster labels attached: {len(df_clustered):,} wells")
print()


# Cluster descriptions below are starting points - review against the profiles and refine.
print("=" * 65)
print("STEP 8: PROFILING CLUSTERS")
print("=" * 65)
print()

PROFILE_COLS = [
    'decline_ratio_boe', 'decline_ratio_gas', 'decline_ratio_oil',
    'water_boe_ratio', 'water_gas_ratio', 'water_oil_ratio',
    'first_yr_pi', 'Months Produced', 'log_last_12_boe_equiv',
    'True Vertical Depth', 'is_horizontal', 'had_recompletion',
    'Production Type_GAS', 'Production Type_OIL',
]
PROFILE_COLS = [c for c in PROFILE_COLS if c in df_clustered.columns]

cluster_profiles = df_clustered.groupby('cluster')[PROFILE_COLS].mean().round(3)
cluster_profiles.index = cluster_profiles.index.astype(int)
cluster_profiles['well_count'] = df_clustered.groupby('cluster').size()

# Well status distribution per cluster
if 'Well Status' in df_clustered.columns:
    active_pct = (
        df_clustered.groupby('cluster')['Well Status']
        .apply(lambda x: (x == 'ACTIVE').mean() * 100)
        .round(1)
    )
    cluster_profiles['pct_active'] = active_pct

# Top formation per cluster
if 'Target Formation' in df_clustered.columns:
    def top_formation(group):
        # Exclude UNKNOWN and OTHER from top formation
        valid = group['Target Formation'][
            ~group['Target Formation'].isin(['UNKNOWN', 'OTHER_FORMATION'])
        ]
        if len(valid) == 0:
            return 'UNKNOWN'
        return valid.value_counts().index[0]

    cluster_profiles['top_formation'] = (
        df_clustered.groupby('cluster')
        .apply(top_formation)
    )

print("Cluster Profiles:")
print(cluster_profiles.to_string())
print()

cluster_descriptions = {
    1: "Cluster 1: Mature Conventional Gas — declining vertical wells, deep formations, East/South Texas",
    2: "Cluster 2: Mature Conventional Oil/Mixed — Gulf Coast formations, further declined, more P&A",
    3: "Cluster 3: Legacy End-of-Life — 40-year-old nearly depleted conventional wells",
    4: "Cluster 4: Modern Shale/Tight — young wells at or near peak, Barnett/Haynesville/Eagle Ford",
    5: "Cluster 5: Active Horizontal — 100% currently producing, high-productivity horizontal wells"
}

print()


print("=" * 65)
print("STEP 9: GENERATING CHARTS")
print("=" * 65)
print()

colors = plt.cm.Set2(np.linspace(0, 1, FINAL_K))

# --- Chart 2: Cluster feature profiles ---
print("Generating Chart 2: Cluster feature profiles...")

PROFILE_PLOT_COLS = [
    'decline_ratio_boe', 'water_boe_ratio', 'first_yr_pi',
    'Months Produced', 'log_last_12_boe_equiv', 'True Vertical Depth',
    'is_horizontal', 'Production Type_GAS', 'Production Type_OIL',
]
PROFILE_PLOT_COLS = [c for c in PROFILE_PLOT_COLS if c in cluster_profiles.columns]

profile_num = cluster_profiles[PROFILE_PLOT_COLS].copy()
# Normalise to 0-1 for comparison
profile_norm = (profile_num - profile_num.min()) / (
    profile_num.max() - profile_num.min() + 1e-9
)

fig, ax = plt.subplots(figsize=(14, 7))
x     = np.arange(len(PROFILE_PLOT_COLS))
width = 0.8 / FINAL_K

for i, (c, color) in enumerate(zip(range(1, FINAL_K + 1), colors)):
    ax.bar(x + i * width - (FINAL_K - 1) * width / 2,
           profile_norm.loc[c],
           width, label=f'Cluster {c}',
           color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(PROFILE_PLOT_COLS, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Normalised Value (0-1)', fontsize=11)
ax.set_title('Cluster Feature Profiles\n'
             '(Higher bar = higher value on that feature relative to other clusters)',
             fontsize=12, fontweight='bold')
ax.legend(title='Cluster', fontsize=9, title_fontsize=10,
          bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('cluster_charts/chart02_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: cluster_charts/chart02_cluster_profiles.png")

# --- Chart 3: PCA visualisation ---
print("Generating Chart 3: PCA cluster visualisation...")

pca    = PCA(n_components=2, random_state=42)
X_pca  = pca.fit_transform(X_scaled)

sample = np.random.choice(len(X_pca), min(8000, len(X_pca)), replace=False)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    X_pca[sample, 0], X_pca[sample, 1],
    c=cluster_labels[sample], cmap='Set2',
    alpha=0.4, s=8
)
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.set_xlabel(
    f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
    fontsize=11
)
ax.set_ylabel(
    f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
    fontsize=11
)
ax.set_title('Well Clusters in 2D (PCA Projection)\n'
             'Each dot = one well. Colour = cluster assignment.',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_charts/chart03_pca_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: cluster_charts/chart03_pca_clusters.png")

# --- Chart 4: Formation-Cluster heatmap ---
print("Generating Chart 4: Formation-Cluster heatmap...")

if 'Target Formation' in df_clustered.columns:
    # Use named formations only — exclude UNKNOWN and OTHER
    named = df_clustered[
        ~df_clustered['Target Formation'].isin(['UNKNOWN', 'OTHER_FORMATION'])
    ]
    top_12 = named['Target Formation'].value_counts().head(12).index

    form_cluster = pd.crosstab(
        named[named['Target Formation'].isin(top_12)]['Target Formation'],
        named[named['Target Formation'].isin(top_12)]['cluster'],
        normalize='index'
    ) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        form_cluster, annot=True, fmt='.0f',
        cmap='YlOrRd', linewidths=0.5, ax=ax,
        cbar_kws={'label': '% of formation wells in each cluster'}
    )
    ax.set_title('Formation × Cluster Distribution\n'
                 '(% of each formation\'s wells in each cluster)\n'
                 'UNKNOWN and OTHER_FORMATION excluded for clarity',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Formation', fontsize=12)
    plt.tight_layout()
    plt.savefig('cluster_charts/chart04_formation_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: cluster_charts/chart04_formation_heatmap.png")

# --- Chart 5: Well status by cluster ---
print("Generating Chart 5: Well status by cluster...")

if 'Well Status' in df_clustered.columns:
    status_by_cluster = pd.crosstab(
        df_clustered['cluster'],
        df_clustered['Well Status'],
        normalize='index'
    ) * 100

    status_by_cluster.plot(
        kind='bar', stacked=True, figsize=(11, 6),
        colormap='Set3', edgecolor='none', alpha=0.9
    )
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Percentage of Wells (%)', fontsize=12)
    plt.title('Well Status Distribution by Cluster\n'
              '(Validates lifecycle interpretation — active wells should '
              'concentrate in healthy clusters)',
              fontsize=12, fontweight='bold')
    plt.legend(title='Well Status', bbox_to_anchor=(1.01, 1),
               loc='upper left', fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('cluster_charts/chart05_status_by_cluster.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: cluster_charts/chart05_status_by_cluster.png")

# --- Chart 6: Months Produced by cluster ---
print("Generating Chart 6: Well age by cluster...")

fig, ax = plt.subplots(figsize=(11, 6))
cluster_groups = [
    df_clustered[df_clustered['cluster'] == c]['Months Produced'].dropna()
    for c in range(1, FINAL_K + 1)
]
bp = ax.boxplot(
    cluster_groups,
    labels=[f'Cluster {c}' for c in range(1, FINAL_K + 1)],
    patch_artist=True,
    medianprops=dict(color='red', linewidth=2)
)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Months Produced', fontsize=12)
ax.set_title('Well Age Distribution by Cluster\n'
             '(Validates lifecycle stage interpretation)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_charts/chart06_age_by_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: cluster_charts/chart06_age_by_cluster.png")

# --- Chart 7: Decline ratio by cluster ---
print("Generating Chart 7: Decline ratio by cluster...")

fig, ax = plt.subplots(figsize=(11, 6))
decline_groups = [
    df_clustered[df_clustered['cluster'] == c]['decline_ratio_boe'].dropna()
    for c in range(1, FINAL_K + 1)
]
bp2 = ax.boxplot(
    decline_groups,
    labels=[f'Cluster {c}' for c in range(1, FINAL_K + 1)],
    patch_artist=True,
    medianprops=dict(color='red', linewidth=2)
)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('BOE Decline Ratio (Last12BOE / Peak BOE)', fontsize=12)
ax.set_title('Decline Ratio (BOE) by Cluster\n'
             '(Higher = less declined, more remaining life)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_charts/chart07_decline_by_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: cluster_charts/chart07_decline_by_cluster.png")
print()


print("=" * 65)
print("STEP 10: SAVING OUTPUTS")
print("=" * 65)
print()

# Trained model
joblib.dump(final_kmeans, 'cluster_model.joblib')
print("✓ Saved: cluster_model.joblib")

# Scaler
joblib.dump(cluster_scaler, 'cluster_scaler.joblib')
print("✓ Saved: cluster_scaler.joblib")

# Full clustered dataset
df_clustered.to_csv('data_clustered.csv', index=False)
print(f"✓ Saved: data_clustered.csv  ({len(df_clustered):,} wells)")

# Cluster profiles
cluster_profiles.to_csv('cluster_profiles.csv')
print("✓ Saved: cluster_profiles.csv")

# Metadata for dashboard
cluster_meta = {
    'n_clusters':         FINAL_K,
    'cluster_features':   CLUSTER_FEATURES,
    'descriptions':       cluster_descriptions,
    'well_counts':        {int(k): int(v) for k, v in zip(unique, counts)},
    'silhouette_scores':  {int(k): round(s, 4) for k, s in zip(K_RANGE, silhouettes)},
    'best_k_silhouette':  int(best_k_sil),
    'pca_variance':       [round(float(v), 4)
                           for v in pca.explained_variance_ratio_],
}
with open('cluster_metadata.json', 'w') as f:
    json.dump(cluster_meta, f, indent=2)
print("✓ Saved: cluster_metadata.json")
print()


print("=" * 65)
print("CLUSTERING COMPLETE — SUMMARY")
print("=" * 65)
print()
print(f"  Wells clustered:      {len(df_clustered):,}")
print(f"  Number of clusters:   {FINAL_K}")
print(f"  Best silhouette K:    {best_k_sil}")
print(f"  PCA variance (2D):    {sum(pca.explained_variance_ratio_)*100:.1f}%")
print()
print("  Cluster summary:")
for c in range(1, FINAL_K + 1):
    p    = cluster_profiles.loc[c]
    desc = cluster_descriptions[c]
    pact = p.get('pct_active', 0)
    print(f"  {desc}")
    print(f"    Wells: {p['well_count']:,.0f}  |  "
          f"Active: {pact:.1f}%  |  "
          f"Median age: {df_clustered[df_clustered['cluster']==c]['Months Produced'].median():.0f} months  |  "
          f"Top formation: {p.get('top_formation','N/A')}")
    print()
print("  Charts saved to: cluster_charts/")
print()
print("  NEXT STEP: Run 05_remaining_life.py")
print("=" * 65)
