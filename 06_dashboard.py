# Well Performance Analyser - Interactive Streamlit Dashboard
# Author: Precious Faseyosan
#
# Upload an Enverus Wells Table CSV to get:
# - Formation ranking by acquisition quality score
# - Cluster profiles and remaining life estimates
# - Formation-cluster heatmap, individual well scoring, economic sensitivity
#
# Required files: cluster_model.joblib, cluster_scaler.joblib, cluster_metadata.json
# Run: streamlit run 06_dashboard.py
#
# Note: Model trained on Texas onshore wells (East Texas, Gulf Coast West,
# Gulf Coast Central). Results are most reliable for these basins.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import warnings
import os
from io import BytesIO

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Well Performance Analyser",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


FORMATION_MAP = {
    'EAGLE FORD':               'EAGLEFORD',
    'EAGLE FORD-1':             'EAGLEFORD',
    'EAGLE FORD-2':             'EAGLEFORD',
    'EAGLE FORD SHALE':         'EAGLEFORD',
    'EAGLEFORD SHALE':          'EAGLEFORD',
    'EAGLEFORD SHALE GAS':      'EAGLEFORD',
    'AUSTIN CHALK-1':           'AUSTIN CHALK',
    'AUSTIN CHALK-2':           'AUSTIN CHALK',
    'AUSTIN CHALK-3':           'AUSTIN CHALK',
    'AUSTIN CHALK 3':           'AUSTIN CHALK',
    'AUSTIN CHALK, GAS':        'AUSTIN CHALK',
    'LOBO CONS.':               'LOBO',
    'LOBO CONS':                'LOBO',
    'LOBO,WILCOX':              'LOBO',
    'HAYNESVILLE SHALE':        'HAYNESVILLE',
    'HAYNESVILLE-BOSSIER':      'HAYNESVILLE',
    'BARNETT SHALE':            'BARNETT',
    'CONSOLIDATED FRIO':        'FRIO',
    'COTTON VALLEY SAND':       'COTTON VALLEY',
    'COTTON VALLEY-1':          'COTTON VALLEY',
}

CLUSTER_LABELS = {
    1: 'C1: Mature Conventional Gas',
    2: 'C2: Mature Conventional Oil/Mixed',
    3: 'C3: Legacy End-of-Life',
    4: 'C4: Modern Shale/Tight',
    5: 'C5: Active Horizontal',
}

CLUSTER_DESCRIPTIONS = {
    1: 'Declining vertical wells in deep conventional gas formations. East and South Texas character. Still many active wells with moderate remaining life.',
    2: 'Mature conventional oil and mixed wells concentrated in Gulf Coast formations. Further declined than C1 with more P&A wells.',
    3: 'Very old wells — median age over 40 years — in near-terminal decline. Limited acquisition value. Document what long-life formations look like at end of life.',
    4: 'Young shale and tight formation wells in early production life, many still near their peak rate. Barnett, Haynesville, Eagle Ford dominated.',
    5: 'Active horizontal wells across multiple formations, 100% currently producing. High productivity per well.',
}

CLUSTER_FEATURES = [
    'decline_ratio_gas', 'decline_ratio_oil', 'decline_ratio_boe',
    'water_gas_ratio', 'water_oil_ratio', 'water_boe_ratio',
    'first_yr_pi', 'Months Produced', 'log_last_12_boe_equiv',
    'True Vertical Depth', 'is_horizontal', 'had_recompletion',
    'Production Type_GAS', 'Production Type_OIL', 'Production Type_OIL & GAS',
]

KEEP_PROD_TYPES = ['GAS', 'OIL & GAS', 'OIL']
KEEP_STATUSES   = ['ACTIVE', 'INACTIVE', 'P & A', 'SHUT-IN', 'TA', 'COMPLETED']
MIN_MONTHS      = 24
MAX_REMAINING   = 180   # 15-year cap


@st.cache_resource
def load_models():
    """Load trained K-Means model, scaler, and metadata."""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model    = joblib.load(os.path.join(model_dir, 'cluster_model.joblib'))
        scaler   = joblib.load(os.path.join(model_dir, 'cluster_scaler.joblib'))
        with open(os.path.join(model_dir, 'cluster_metadata.json')) as f:
            metadata = json.load(f)
        return model, scaler, metadata, None
    except FileNotFoundError as e:
        return None, None, None, str(e)


def clean_data(df_raw):
    """Clean raw Enverus CSV and return (df, status_message)."""
    df = df_raw.copy()
    n0 = len(df)

    # Standardise formation names
    if 'Target Formation' in df.columns:
        df['Target Formation'] = (
            df['Target Formation'].str.upper().str.strip()
            .replace(FORMATION_MAP)
        )

    if 'Production Type' in df.columns:
        df = df[df['Production Type'].isin(KEEP_PROD_TYPES)]

    if 'Well Status' in df.columns:
        df = df[df['Well Status'].isin(KEEP_STATUSES)]

    # Exclude WILDCAT
    if 'Target Formation' in df.columns:
        df = df[df['Target Formation'].str.upper() != 'WILDCAT']

    # Keep only original completion (code 00) - recompletion rows carry no production data
    if 'API14' in df.columns:
        df['_api14_str'] = df['API14'].astype(str).str.replace('-', '').str.strip()
        df['_api12']     = df['_api14_str'].str[:12]
        df['_comp_code'] = pd.to_numeric(
            df['_api14_str'].str[12:14], errors='coerce'
        ).fillna(0).astype(int)

        apis_with_recomplete = (
            df[df['_comp_code'] > 0]['_api12'].unique()
        )
        df['had_recompletion'] = df['_api12'].isin(apis_with_recomplete).astype(int)

        # Keep only original completion (code = 00)
        df = df[df['_comp_code'] == 0]
        df = df.drop(columns=['_api14_str', '_api12', '_comp_code'])
    else:
        df['had_recompletion'] = 0

    if 'Months Produced' in df.columns:
        df = df[df['Months Produced'].notna() & (df['Months Produced'] >= MIN_MONTHS)]

    # Require production data in at least one fluid
    has_gas = df.get('Peak Gas', pd.Series(dtype=float)).notna() | \
              df.get('First 12 Gas', pd.Series(dtype=float)).notna()
    has_oil = df.get('Peak Oil', pd.Series(dtype=float)).notna() | \
              df.get('First 12 Oil', pd.Series(dtype=float)).notna()
    df = df[has_gas | has_oil]

    # Fill missing numerics with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in ['Target Formation', 'Production Type', 'Well Status',
                'DI Basin', 'Drill Type']:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')

    n1 = len(df)
    msg = f"Raw: {n0:,} wells → After cleaning: {n1:,} wells ({n0-n1:,} removed)"
    return df, msg


def engineer_features(df):
    """Reproduce feature engineering from 02_cleaning.py for all clustering features."""
    df = df.copy()

    # Ensure required numeric columns exist (fill 0 if missing)
    for col in ['Peak Gas', 'Peak Oil', 'Peak BOE',
                'Last 12 Gas', 'Last 12 Oil',
                'First 12 Gas', 'First 12 Oil', 'First 12 BOE',
                'Cum Water', 'Months Produced',
                'True Vertical Depth', 'Gross Perforated Interval']:
        if col not in df.columns:
            df[col] = 0.0

    df['months_since_peak'] = (df['Months Produced'] - 12).clip(lower=1)

    # Decline ratios - gas, oil, BOE
    df['last_12_boe_equiv'] = df['Last 12 Gas'] / 6 + df['Last 12 Oil']

    df['decline_ratio_gas'] = (
        df['Last 12 Gas'] / (df['Peak Gas'] + 1)
    ).clip(0.001, 1.5)

    df['decline_ratio_oil'] = (
        df['Last 12 Oil'] / (df['Peak Oil'] + 1)
    ).clip(0.001, 1.5)

    df['decline_ratio_boe'] = (
        df['last_12_boe_equiv'] / (df['Peak BOE'] + 1)
    ).clip(0.001, 1.5)

    # Water ratios - gas, oil, BOE
    for ratio, num, denom in [
        ('water_gas_ratio', 'Cum Water', 'Last 12 Gas'),
        ('water_oil_ratio', 'Cum Water', 'Last 12 Oil'),
        ('water_boe_ratio', 'Cum Water', 'last_12_boe_equiv'),
    ]:
        df[ratio] = df[num] / (df[denom] + 1)
        cap = df[ratio].quantile(0.99)
        df[ratio] = df[ratio].clip(0, cap)

    # First-year productivity index (BOE/ft)
    df['first_yr_pi'] = (
        df['First 12 BOE'] / (df['Gross Perforated Interval'] + 1)
    )

    if 'Drill Type' in df.columns:
        df['is_horizontal'] = (df['Drill Type'] == 'H').astype(int)
    else:
        df['is_horizontal'] = 0

    # Log-transformed BOE rate (primary clustering feature)
    df['log_last_12_boe_equiv'] = np.log1p(df['last_12_boe_equiv'].clip(lower=0))

    for pt in ['GAS', 'OIL', 'OIL & GAS']:
        col = f'Production Type_{pt}'
        if 'Production Type' in df.columns:
            df[col] = (df['Production Type'] == pt).astype(int)
        else:
            df[col] = 0

    return df


def assign_clusters(df, model, scaler):
    """Assign K-Means clusters. Labels shifted to 1-based to match training script."""
    missing = [f for f in CLUSTER_FEATURES if f not in df.columns]
    if missing:
        return df, f"Missing features for clustering: {missing}"

    X = df[CLUSTER_FEATURES].copy().fillna(0)
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled) + 1   # shift to 1-based
    df['cluster'] = labels
    df['cluster_label'] = df['cluster'].map(CLUSTER_LABELS)
    return df, None


def calculate_remaining_life(df, gas_price, oil_price, monthly_opex):
    """Estimate remaining life (months) via exponential decline: t = -ln(OPEX/Revenue) / D_boe."""
    df = df.copy()
    MAX_REMAINING = 180   # 15-year cap

    df['monthly_gas'] = df.get('Last 12 Gas', pd.Series(0, index=df.index)) / 12
    df['monthly_oil'] = df.get('Last 12 Oil', pd.Series(0, index=df.index)) / 12

    df['monthly_revenue'] = (
        df['monthly_gas'] * gas_price +
        df['monthly_oil'] * oil_price
    )

    # BOE decline rate
    df['months_since_peak'] = (df['Months Produced'] - 12).clip(lower=1)
    safe_ratio = df['decline_ratio_boe'].clip(0.001, 1.499)
    df['D_boe'] = (
        -np.log(safe_ratio) / df['months_since_peak']
    ).clip(0.0001, 0.30)

    def calc_rl(row):
        status = str(row.get('Well Status', '')).upper().strip()
        if status in ['P & A', 'INACTIVE']:
            return 0.0, 'ended'
        rev = row['monthly_revenue']
        if rev < monthly_opex or rev <= 0:
            return 0.0, 'uneconomic'
        D = row['D_boe']
        if D < 0.0001:
            return float(MAX_REMAINING), 'active'
        try:
            rl = -np.log(monthly_opex / rev) / D
            return max(0.0, min(float(MAX_REMAINING), rl)), 'active'
        except:
            return 0.0, 'uneconomic'

    results = df.apply(calc_rl, axis=1)
    df['remaining_life_months'] = results.apply(lambda x: round(x[0], 1))
    df['remaining_life_years']  = (df['remaining_life_months'] / 12).round(2)
    df['economic_status']       = results.apply(lambda x: x[1])

    return df


def rank_formations(df, w1, w2, w3, w4):
    """Return formation summary dataframe ranked by weighted acquisition quality score."""
    if 'Target Formation' not in df.columns:
        return pd.DataFrame()

    formation = df.groupby('Target Formation').agg(
        well_count            = ('remaining_life_months', 'count'),
        pct_active            = ('economic_status',
                                  lambda x: (x == 'active').mean() * 100),
        median_decline_ratio  = ('decline_ratio_boe', 'median'),
        median_water_ratio    = ('water_boe_ratio', 'median'),
        median_first_yr_pi    = ('first_yr_pi', 'median'),
        median_months_produced = ('Months Produced', 'median'),
    ).round(2)

    # Active-only remaining life
    active = df[df['economic_status'] == 'active']
    formation['median_remaining_life'] = (
        active.groupby('Target Formation')['remaining_life_months']
        .median().round(1)
    ).fillna(0)

    # Filter: minimum 5 wells, exclude UNKNOWN
    formation = formation[
        (formation['well_count'] >= 5) &
        (~formation.index.isin(['UNKNOWN', 'OTHER_FORMATION', 'WILDCAT']))
    ]

    if len(formation) == 0:
        return pd.DataFrame()

    # Normalise
    def norm(s, higher=True):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        n = (s - mn) / (mx - mn)
        return n if higher else 1 - n

    formation['s_life']    = norm(formation['median_remaining_life'], True)
    formation['s_decline'] = norm(formation['median_decline_ratio'],  True)
    formation['s_water']   = norm(formation['median_water_ratio'],    False)
    formation['s_pi']      = norm(formation['median_first_yr_pi'],    True)

    # Weighted quality score × pct_active (survivorship bias correction)
    quality = (
        (w1/100) * formation['s_life']    +
        (w2/100) * formation['s_decline'] +
        (w3/100) * formation['s_water']   +
        (w4/100) * formation['s_pi']
    )
    formation['weighted_score'] = (
        quality * (formation['pct_active'] / 100)
    ).round(4)

    formation['weighted_score'] = formation['weighted_score'].fillna(0)
    formation['rank'] = formation['weighted_score'].rank(
        ascending=False, method='min'
    ).astype(int)

    return formation.sort_values('rank')


@st.cache_data(show_spinner=False)
def run_full_pipeline(_df_raw, gas_price, oil_price, monthly_opex, w1, w2, w3, w4):
    """Run full pipeline: clean -> feature engineer. Cached for fast price/weight changes."""
    df, clean_msg  = clean_data(_df_raw)
    df             = engineer_features(df)
    return df, clean_msg


with st.sidebar:
    st.markdown("## 🛢️ Well Performance Analyser")
    st.markdown("---")

    st.markdown("### 📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Enverus Wells Table CSV Export",
        type=['csv'],
        help="Upload a raw Enverus DrillingInfo Wells Table CSV export. "
             "No pre-processing needed."
    )

    st.markdown("---")
    st.markdown("### 💰 Economic Parameters")
    st.caption("Set your commodity prices and operating costs.")

    gas_price = st.slider(
        "Gas Price ($/MCF)", 1.00, 8.00, 2.50, 0.25,
        help="Wellhead or regional gas price"
    )
    oil_price = st.slider(
        "Oil Price ($/BBL)", 30.0, 120.0, 65.0, 2.5,
        help="Wellhead or regional oil price"
    )
    monthly_opex = st.slider(
        "Monthly OPEX ($/well/month)", 500, 10000, 3000, 500,
        help="Average operating cost per well per month"
    )

    st.markdown("---")
    st.markdown("### ⚖️ Ranking Weights")
    st.caption("Weights must sum to 100%")

    w1 = st.slider("Remaining Life (%)", 0, 100, 40, 5)
    remaining = 100 - w1
    w2 = st.slider("Low Decline Rate (%)", 0, remaining, min(30, remaining), 5)
    remaining2 = remaining - w2
    w3 = st.slider("Low Water Loading (%)", 0, remaining2, min(20, remaining2), 5)
    w4 = 100 - w1 - w2 - w3

    weight_ok = (w1 + w2 + w3 + w4 == 100)
    if weight_ok:
        st.success(f"Initial Rate: {w4}%  ✓  Total: 100%")
    else:
        st.error(f"Weights sum to {w1+w2+w3+w4}% — must equal 100%")

    st.markdown("---")
    st.caption(
        "Model trained on Texas wells — East Texas, Gulf Coast West, "
        "Gulf Coast Central. Results most reliable for Texas onshore wells."
    )


st.markdown('<p class="main-header">🛢️ Well Performance Analyser</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Formation-Based Well Clustering & Remaining Productive Life Estimation — '
    'Powered by K-Means ML'
    '</p>',
    unsafe_allow_html=True
)

model, scaler, metadata, model_err = load_models()

if model_err:
    st.error(
        f"**Model files not found:** {model_err}\n\n"
        "Ensure `cluster_model.joblib`, `cluster_scaler.joblib`, and "
        "`cluster_metadata.json` are in the same folder as this script."
    )
    st.stop()

if uploaded_file is None:
    # Landing page
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nUpload your Enverus Wells Table CSV export using the sidebar.")
    with col2:
        st.info("**Step 2**\n\nSet your gas price, oil price, and monthly OPEX.")
    with col3:
        st.info("**Step 3**\n\nAdjust ranking weights to reflect your acquisition strategy.")

    st.markdown("---")
    st.markdown("### What this tool does")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Formation Ranking**
        Ranks formations in your uploaded dataset by a composite acquisition
        quality score — combining remaining productive life, decline stability,
        water loading, and initial productivity, weighted by your priorities.

        **Well Clustering**
        Assigns each well to one of five behavioural clusters using a K-Means
        model trained on 122,000 Texas wells. Clusters reflect production
        behaviour patterns, not formation labels.
        """)
    with c2:
        st.markdown("""
        **Remaining Life Estimation**
        Projects how many months each active well will continue generating
        revenue above your stated OPEX, using exponential decline mathematics
        applied to combined gas + oil revenue.

        **Economic Sensitivity**
        Shows how formation rankings and remaining life estimates shift across
        a range of gas and oil price scenarios — critical for acquisition
        underwriting under price uncertainty.
        """)
    st.stop()

with st.spinner("Reading file..."):
    try:
        df_raw = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

# Check for minimum required columns
REQUIRED_COLS = ['Months Produced', 'Production Type', 'Well Status']
missing_required = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_required:
    st.error(
        f"**Missing required columns:** {missing_required}\n\n"
        "Please ensure you are uploading an Enverus Wells Table export "
        "with standard column names."
    )
    st.stop()

with st.spinner("Running pipeline: cleaning → clustering → remaining life..."):
    df_clean, clean_msg = run_full_pipeline(
        df_raw, gas_price, oil_price, monthly_opex, w1, w2, w3, w4
    )

    if len(df_clean) == 0:
        st.error("No wells remaining after cleaning. Check your data format.")
        st.stop()

    # Cluster assignment (not cached - depends on slider changes)
    df_clean, cluster_err = assign_clusters(df_clean, model, scaler)
    if cluster_err:
        st.warning(f"Clustering issue: {cluster_err}. Proceeding without cluster assignment.")

    df_clean = calculate_remaining_life(
        df_clean, gas_price, oil_price, monthly_opex
    )

    formation_ranked = rank_formations(df_clean, w1, w2, w3, w4)

active_wells = df_clean[df_clean['economic_status'] == 'active']
ended_wells  = df_clean[df_clean['economic_status'] == 'ended']

st.success(f"✓ {clean_msg}")
st.markdown("---")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Wells", f"{len(df_clean):,}")
m2.metric("Active (Revenue > OPEX)",
          f"{len(active_wells):,}",
          f"{len(active_wells)/len(df_clean)*100:.1f}%")
m3.metric("Uneconomic",
          f"{(df_clean['economic_status']=='uneconomic').sum():,}")
m4.metric("Ended (P&A/Inactive)", f"{len(ended_wells):,}")
m5.metric("Median Remaining Life (Active)",
          f"{active_wells['remaining_life_months'].median():.0f} mo" if len(active_wells) > 0 else "N/A",
          f"{active_wells['remaining_life_months'].median()/12:.1f} yr" if len(active_wells) > 0 else "")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Formation Ranking",
    "🔬 Cluster Profiles",
    "🗺️ Formation × Cluster",
    "🏭 Well Scoring",
    "📈 Economic Sensitivity",
])


# --- Tab 1: Formation Ranking
with tab1:
    st.markdown("### Formation Ranking by Acquisition Quality Score")
    st.caption(
        f"Score = Quality Score × (% Active wells)  |  "
        f"Weights: Remaining Life {w1}%  |  Low Decline {w2}%  |  "
        f"Low Water {w3}%  |  Initial Rate {w4}%"
    )

    if len(formation_ranked) == 0:
        st.warning("Not enough formation data to rank. Need at least 5 wells per formation.")
    else:
        top_n = min(20, len(formation_ranked))
        top_df = formation_ranked.head(top_n).reset_index()

        fig = px.bar(
            top_df.iloc[::-1],
            x='weighted_score',
            y='Target Formation',
            orientation='h',
            color='weighted_score',
            color_continuous_scale='RdYlGn',
            text=top_df.iloc[::-1].apply(
                lambda r: f"Score: {r['weighted_score']:.3f} | "
                          f"{r['median_remaining_life']:.0f} mo | "
                          f"{r['pct_active']:.0f}% active | "
                          f"{r['well_count']:,} wells",
                axis=1
            ),
            labels={'weighted_score': 'Quality Score (0–1)',
                    'Target Formation': 'Formation'},
            title=f"Top {top_n} Formations by Acquisition Quality Score"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=600,
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Full Formation Ranking Table")
        display_df = formation_ranked[
            ['rank', 'well_count', 'weighted_score',
             'median_remaining_life', 'pct_active',
             'median_decline_ratio', 'median_water_ratio']
        ].copy()
        display_df.columns = [
            'Rank', 'Wells', 'Score',
            'Median Remaining Life (mo)', '% Active',
            'Median Decline Ratio', 'Median Water-BOE Ratio'
        ]
        st.dataframe(display_df, use_container_width=True)

        csv_buffer = BytesIO()
        formation_ranked.to_csv(csv_buffer)
        st.download_button(
            "⬇️ Download Formation Ranking CSV",
            csv_buffer.getvalue(),
            file_name="formation_ranking.csv",
            mime="text/csv"
        )


# --- Tab 2: Cluster Profiles
with tab2:
    st.markdown("### Well Cluster Profiles")
    st.caption(
        "Wells are assigned to clusters based on production behaviour — "
        "decline pattern, water loading, productivity, and well age. "
        "Formation name is not an input to clustering."
    )

    if 'cluster' not in df_clean.columns:
        st.warning("Cluster assignment not available.")
    else:
        cluster_counts = df_clean['cluster'].value_counts().sort_index()

        profile_cols = [
            'decline_ratio_boe', 'water_boe_ratio', 'first_yr_pi',
            'Months Produced', 'log_last_12_boe_equiv', 'True Vertical Depth',
            'is_horizontal'
        ]
        profile_cols = [c for c in profile_cols if c in df_clean.columns]

        profiles = df_clean.groupby('cluster')[profile_cols].mean()
        profiles_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min() + 1e-9)

        st.markdown("#### Cluster Descriptions")
        col_cards = st.columns(5)
        for i, (c, desc_text) in enumerate(CLUSTER_DESCRIPTIONS.items()):
            n = cluster_counts.get(c, 0)
            pct = n / len(df_clean) * 100
            with col_cards[i]:
                st.markdown(
                    f"**{CLUSTER_LABELS[c]}**\n\n"
                    f"_{n:,} wells ({pct:.1f}%)_\n\n"
                    f"{desc_text}"
                )

        st.markdown("---")

        st.markdown("#### Feature Profile by Cluster (Normalised 0–1)")

        fig = go.Figure()
        colors = ['#1abc9c', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        for (c, row), color in zip(profiles_norm.iterrows(), colors):
            fig.add_trace(go.Bar(
                name=CLUSTER_LABELS.get(int(c), f'Cluster {c}'),
                x=profile_cols,
                y=row.values,
                marker_color=color,
                opacity=0.8
            ))

        fig.update_layout(
            barmode='group',
            title='Cluster Feature Profiles — Higher Bar = Higher Value',
            xaxis_tickangle=-30,
            height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Remaining Life by Cluster — Active Wells Only")
        fig2 = go.Figure()
        for c, color in zip(sorted(df_clean['cluster'].unique()), colors):
            data = (
                df_clean[
                    (df_clean['cluster'] == c) &
                    (df_clean['economic_status'] == 'active')
                ]['remaining_life_months']
                .clip(0, MAX_REMAINING)
            )
            fig2.add_trace(go.Box(
                y=data,
                name=CLUSTER_LABELS.get(int(c), f'Cluster {c}'),
                marker_color=color,
                boxmean=True
            ))
        fig2.update_layout(
            height=420,
            yaxis_title='Remaining Life (months)',
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)


# --- Tab 3: Formation x Cluster Heatmap
with tab3:
    st.markdown("### Formation × Cluster Distribution")
    st.caption(
        "Shows what percentage of each formation's wells fall into each cluster. "
        "Concentrated red cells indicate strong formation-cluster alignment. "
        "UNKNOWN and multi-zone names are excluded for clarity."
    )

    if 'cluster' not in df_clean.columns or 'Target Formation' not in df_clean.columns:
        st.warning("Formation or cluster data not available.")
    else:
        # Get top formations (exclude UNKNOWN, require minimum wells)
        top_forms = (
            df_clean[~df_clean['Target Formation'].isin(['UNKNOWN', 'OTHER_FORMATION'])]
            ['Target Formation'].value_counts()
            .head(15).index
        )

        heat_df = df_clean[
            df_clean['Target Formation'].isin(top_forms)
        ]

        if len(heat_df) > 0:
            crosstab = pd.crosstab(
                heat_df['Target Formation'],
                heat_df['cluster'],
                normalize='index'
            ) * 100

            crosstab.columns = [
                CLUSTER_LABELS.get(int(c), f'C{c}')
                for c in crosstab.columns
            ]

            fig = px.imshow(
                crosstab,
                text_auto='.0f',
                color_continuous_scale='YlOrRd',
                title='Formation × Cluster Distribution (% of formation wells per cluster)',
                labels=dict(x='Cluster', y='Formation',
                            color='% of wells in cluster'),
                aspect='auto'
            )
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Read row-wise: each row shows how a formation's wells distribute "
                "across the five clusters. A formation concentrated in C4 (Modern Shale/Tight) "
                "is younger and nearer peak. One concentrated in C1 (Mature Conventional Gas) "
                "is older and declining."
            )
        else:
            st.warning("Not enough named formation data for heatmap.")


# --- Tab 4: Well Scoring
with tab4:
    st.markdown("### Individual Well Scoring")
    st.caption(
        "Every well in your uploaded dataset, scored and ranked. "
        "Score reflects remaining life, decline stability, water loading, and productivity. "
        "Active wells ranked first."
    )

    def norm_series(s, higher=True):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        n = (s - mn) / (mx - mn)
        return n if higher else 1 - n

    df_score = df_clean.copy()
    df_score['ws_life']    = norm_series(df_score['remaining_life_months'], True)
    df_score['ws_decline'] = norm_series(df_score['decline_ratio_boe'],     True)
    df_score['ws_water']   = norm_series(df_score['water_boe_ratio'],       False)
    df_score['ws_pi']      = norm_series(df_score['first_yr_pi'],           True)

    df_score['well_score'] = (
        (w1/100) * df_score['ws_life']    +
        (w2/100) * df_score['ws_decline'] +
        (w3/100) * df_score['ws_water']   +
        (w4/100) * df_score['ws_pi']
    ).round(4)

    df_score['well_rank'] = df_score['well_score'].rank(
        ascending=False, method='min'
    ).astype(int)

    id_cols = [c for c in [
        'API14', 'Well Name', 'Operator Company Name',
        'County/Parish', 'Target Formation', 'Well Status',
        'cluster_label', 'Production Type'
    ] if c in df_score.columns]

    result_cols = id_cols + [
        'remaining_life_months', 'remaining_life_years',
        'economic_status', 'well_score', 'well_rank'
    ]
    result_cols = [c for c in result_cols if c in df_score.columns]

    well_table = df_score[result_cols].sort_values('well_rank')

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        status_filter = st.multiselect(
            "Economic Status",
            ['active', 'uneconomic', 'ended'],
            default=['active']
        )
    with col_f2:
        if 'Target Formation' in df_score.columns:
            formations = sorted(df_score['Target Formation'].unique())
            form_filter = st.multiselect("Formation", formations, default=[])
        else:
            form_filter = []
    with col_f3:
        if 'cluster_label' in df_score.columns:
            clusters = sorted(df_score['cluster_label'].dropna().unique())
            cluster_filter = st.multiselect("Cluster", clusters, default=[])
        else:
            cluster_filter = []

    filtered = well_table[well_table['economic_status'].isin(status_filter)]
    if form_filter:
        filtered = filtered[filtered['Target Formation'].isin(form_filter)]
    if cluster_filter:
        filtered = filtered[filtered['cluster_label'].isin(cluster_filter)]

    st.markdown(f"**Showing {len(filtered):,} wells**")
    st.dataframe(filtered.head(500), use_container_width=True)

    csv_buf = BytesIO()
    well_table.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download Full Well Scoring CSV",
        csv_buf.getvalue(),
        file_name="well_scores.csv",
        mime="text/csv"
    )


# --- Tab 5: Economic Sensitivity
with tab5:
    st.markdown("### Economic Sensitivity Analysis")
    st.caption(
        "Shows how portfolio remaining life and the number of economic wells "
        "change across commodity price scenarios. "
        "All other parameters held at sidebar values."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        # Gas price sensitivity
        gas_range = np.arange(1.50, 5.26, 0.25)
        median_rl_gas, count_active_gas = [], []

        for gp in gas_range:
            rev = df_clean['monthly_gas'] * gp + df_clean['monthly_oil'] * oil_price
            viable = (rev >= monthly_opex) & (df_clean['D_boe'] >= 0.0001)
            rl = np.where(
                viable,
                (-np.log(monthly_opex / rev.clip(lower=monthly_opex + 0.01)) /
                 df_clean['D_boe']).clip(0, MAX_REMAINING),
                0.0
            )
            median_rl_gas.append(float(np.median(rl)))
            count_active_gas.append(int(viable.sum()))

        fig_gas = make_subplots(specs=[[{"secondary_y": True}]])
        fig_gas.add_trace(
            go.Scatter(
                x=gas_range, y=[r/12 for r in median_rl_gas],
                name='Median Remaining Life (yr)',
                line=dict(color='steelblue', width=2.5),
                fill='tozeroy', fillcolor='rgba(70,130,180,0.15)'
            ), secondary_y=False
        )
        fig_gas.add_trace(
            go.Scatter(
                x=gas_range, y=count_active_gas,
                name='Active Wells Count',
                line=dict(color='green', width=2, dash='dot'),
            ), secondary_y=True
        )
        fig_gas.add_vline(
            x=gas_price, line_dash='dash', line_color='red',
            annotation_text=f"Current: ${gas_price:.2f}"
        )
        fig_gas.update_xaxes(title_text='Gas Price ($/MCF)')
        fig_gas.update_yaxes(title_text='Median Remaining Life (years)', secondary_y=False)
        fig_gas.update_yaxes(title_text='Active Well Count', secondary_y=True)
        fig_gas.update_layout(
            title=f'Gas Price Sensitivity<br>(Oil fixed at ${oil_price:.2f}/BBL)',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_gas, use_container_width=True)

    with col_right:
        # Oil price sensitivity
        oil_range = np.arange(30, 121, 5)
        median_rl_oil, count_active_oil = [], []

        for op in oil_range:
            rev = df_clean['monthly_gas'] * gas_price + df_clean['monthly_oil'] * op
            viable = (rev >= monthly_opex) & (df_clean['D_boe'] >= 0.0001)
            rl = np.where(
                viable,
                (-np.log(monthly_opex / rev.clip(lower=monthly_opex + 0.01)) /
                 df_clean['D_boe']).clip(0, MAX_REMAINING),
                0.0
            )
            median_rl_oil.append(float(np.median(rl)))
            count_active_oil.append(int(viable.sum()))

        fig_oil = make_subplots(specs=[[{"secondary_y": True}]])
        fig_oil.add_trace(
            go.Scatter(
                x=oil_range, y=[r/12 for r in median_rl_oil],
                name='Median Remaining Life (yr)',
                line=dict(color='darkorange', width=2.5),
                fill='tozeroy', fillcolor='rgba(255,140,0,0.15)'
            ), secondary_y=False
        )
        fig_oil.add_trace(
            go.Scatter(
                x=oil_range, y=count_active_oil,
                name='Active Wells Count',
                line=dict(color='green', width=2, dash='dot'),
            ), secondary_y=True
        )
        fig_oil.add_vline(
            x=oil_price, line_dash='dash', line_color='red',
            annotation_text=f"Current: ${oil_price:.2f}"
        )
        fig_oil.update_xaxes(title_text='Oil Price ($/BBL)')
        fig_oil.update_yaxes(title_text='Median Remaining Life (years)', secondary_y=False)
        fig_oil.update_yaxes(title_text='Active Well Count', secondary_y=True)
        fig_oil.update_layout(
            title=f'Oil Price Sensitivity<br>(Gas fixed at ${gas_price:.2f}/MCF)',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_oil, use_container_width=True)

    st.markdown("#### OPEX Sensitivity — Active Well Count at Current Prices")
    opex_vals = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000]
    opex_results = []
    for ov in opex_vals:
        rev = df_clean['monthly_gas'] * gas_price + df_clean['monthly_oil'] * oil_price
        n_active = (rev >= ov).sum()
        pct = n_active / len(df_clean) * 100
        opex_results.append({
            'Monthly OPEX ($/well)': f"${ov:,}",
            'Active Wells': f"{n_active:,}",
            '% of Portfolio': f"{pct:.1f}%",
            'Highlighted': ov == monthly_opex
        })

    opex_df = pd.DataFrame(opex_results)
    st.dataframe(
        opex_df[['Monthly OPEX ($/well)', 'Active Wells', '% of Portfolio']],
        use_container_width=True
    )
