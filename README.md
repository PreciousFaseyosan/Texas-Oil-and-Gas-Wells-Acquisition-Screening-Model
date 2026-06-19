# 🛢️ Well Performance Analyser: Texas Oil & Gas Acquisition Screening Model

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://acquisitionscreeningmodel.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

> A machine learning–powered web application for acquisition screening of oil and gas wells — built by a petroleum engineer, for petroleum engineers.

---

## 🔍 Project Overview

Screening acquisition packages in upstream oil and gas is time-consuming, domain-intensive work. As part of my MSc in Data Science, I built this tool to explore whether unsupervised machine learning could accelerate and systematise that process.

The model was trained on **122,000+ Texas onshore wells** sourced from Enverus DrillingInfo, covering three major basins:
- East Texas
- Gulf Coast West
- Gulf Coast Central

Given a raw Enverus Wells Table CSV export, the application clusters wells by production behaviour, estimates remaining productive life using exponential decline mathematics, and ranks formations by a weighted acquisition quality score — all without any manual preprocessing.

---

## ⚙️ What the App Does

### 📊 Tab 1 — Formation Ranking
Ranks formations by a composite acquisition quality score combining:
- Remaining productive life (months above operating breakeven)
- Decline stability (BOE decline ratio)
- Water loading (water-to-BOE ratio)
- Initial productivity (first-year BOE per perforated foot)

Weights are adjustable so operators can tune rankings to their acquisition strategy.

### 🔬 Tab 2 — Cluster Profiles
Assigns each well to one of five behavioural clusters using the trained K-Means model:

| Cluster | Label | Characteristics |
|---------|-------|----------------|
| C1 | Mature Conventional Gas | Declining vertical wells, deep gas, moderate remaining life |
| C2 | Mature Conventional Oil/Mixed | Gulf Coast origin, further declined, more P&A wells |
| C3 | Legacy End-of-Life | Median age 40+ yrs, near-terminal decline, limited acquisition value |
| C4 | Modern Shale/Tight | Young wells near peak — Barnett, Haynesville, Eagle Ford |
| C5 | Active Horizontal | 100% producing horizontal wells, highest per-well productivity |

### 🗺️ Tab 3 — Formation × Cluster Heatmap
Shows how each formation's wells distribute across the five clusters, revealing whether a formation is dominated by legacy declining inventory or active modern completions.

### 🏭 Tab 4 — Individual Well Scoring
Scores and ranks every well in the uploaded dataset. Filterable by economic status, formation, and cluster. Exportable as CSV.

### 📈 Tab 5 — Economic Sensitivity Analysis
Shows how portfolio remaining life and the count of economic wells shift across gas and oil price scenarios, critical for acquisition underwriting under commodity price uncertainty.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Machine Learning | scikit-learn (K-Means, StandardScaler) |
| Feature Engineering | pandas, NumPy |
| Visualisation | Plotly |
| Web Application | Streamlit |
| Model Serialisation | joblib |
| Data Source | Enverus DrillingInfo Wells Table export |

---

## 📁 Repository Structure

```
Texas-Oil-and-Gas-Wells-Acquisition-Screening-Model/
├── 06_dashboard.py          # Main Streamlit application (1,100+ lines)
├── cluster_model.joblib     # Trained K-Means model (5 clusters)
├── cluster_scaler.joblib    # Fitted StandardScaler
├── cluster_metadata.json    # Cluster centroid and label metadata
├── requirements.txt         # Python dependencies
└── README.md
```

> **Note:** Training notebooks (01–05 covering EDA, cleaning, feature engineering, model selection, and evaluation) are not included as the underlying dataset is proprietary. The deployed app accepts any Enverus Wells Table CSV and applies the pre-trained model.

---

## 🚀 Running Locally

```bash
git clone https://github.com/PreciousFaseyosan/Texas-Oil-and-Gas-Wells-Acquisition-Screening-Model.git
cd Texas-Oil-and-Gas-Wells-Acquisition-Screening-Model
pip install -r requirements.txt
streamlit run 06_dashboard.py
```

**Input:** Raw Enverus DrillingInfo Wells Table CSV export.  
Minimum required columns: `Months Produced`, `Production Type`, `Well Status`.

---

## 🔬 Methodology

### Feature Engineering
15 features engineered from raw production data:
- **Decline ratios** (gas, oil, BOE): last-12-month vs peak production
- **Water ratios**: cumulative water relative to recent fluid rates
- **First-year productivity index**: initial BOE per perforated foot
- **Log-transformed current BOE rate**: primary productivity signal
- **Well geometry & history flags**: TVD, is_horizontal, had_recompletion, production type dummies

### Clustering
K-Means (k=5) trained on standardised features across 122,000+ wells. Optimal k selected via elbow method and silhouette analysis.

### Remaining Life Estimation
Exponential decline applied to combined gas + oil revenue:

```
t_remaining = -ln(OPEX / Monthly_Revenue) / D_BOE
```

Remaining life capped at 180 months (15 years).

---

## ⚠️ Limitations & Future Work

**Current limitations:**
- Trained on Texas onshore only (East Texas, Gulf Coast West & Central). Performance on Permian Basin or other plays not validated.
- Exponential decline is a simplification; hyperbolic fits shale wells better but requires inputs not in the Wells Table export.
- Water ratios use cumulative rather than recent water — may overstate risk on older wells with high early-water history.
- Single OPEX assumed per well; real operating costs vary by depth, lift method, and location.

**Future improvements:**
- Retrain on multi-basin dataset for broader generalisability
- Add hyperbolic/harmonic decline curve fitting
- Build supervised classifier for targeted acquisition criteria
- Individual well decline curve visualisation

---

## 👤 Author

**Precious Faseyosan**  
Graduate Petroleum Engineer | MSc Data Science Scholar

- 🔗 [LinkedIn](https://www.linkedin.com/in/precious-faseyosan)
- 💻 [GitHub](https://github.com/PreciousFaseyosan)
- 🌐 [Live App](https://acquisitionscreeningmodel.streamlit.app)

*Built as part of the Machine Learning Fundamentals course, MSc Data Science, Nigerian University of Technology and Management (NUTM), 2024/2025.*
