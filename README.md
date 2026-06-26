# MDL-v5: Modular Deep Learning for Non-Stationary Stock Price Forecasting

> **A Tehran Stock Exchange Case Study with Multi-Modal Integration**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)
![Status](https://img.shields.io/badge/Status-Research-f59e0b?style=flat)
![Market](https://img.shields.io/badge/Market-Tehran_Stock_Exchange-1d4ed8?style=flat)
![Stocks](https://img.shields.io/badge/Stocks-310_|_867-6366f1?style=flat)

Inspired by the state-of-the-art framework of Huang & Yang (2026), this repository presents **MDL-v5**, a modular and robust deep learning pipeline specifically optimized for high-noise, non-stationary markets like the Tehran Stock Exchange (TSE).

---

## 🚀 Key Innovations

### 1. Robustness to TSE Regime Shifts
Unlike standard monolithic models, MDL-v5 utilizes a **Modular Deep Learning (MDL)** architecture that effectively handles the severe distributional drifts observed in the Iranian market (post-1400/2021 regimes). The model is evaluated on a universe of **513–844** actively traded symbols.

### 2. Single-Stock Multimodal Pipeline (v6)
Standard Multi-Stock models often lose significant historical data (~1,400 trading days) due to **"Common Day Intersection"** during symbol suspensions (affecting 391–400 symbols). Our v6 Pipeline overcomes this by implementing a **Single-Stock but Multimodal** approach, preserving **100% of the historical context** for each ticker while integrating **42 heterogeneous features** (733–742 feature-days retained).

### 3. Feature-Rich Representation (N=42)
The model processes a comprehensive **42-dimensional feature vector**, including:

| Category | Count | Features |
|---|---|---|
| **Technical Indicators** | 15 | MACD, EMA, Momentum, RSI, Bollinger Bands, etc. |
| **Fourier / Spectral** | 6 | FFT components to capture market cycles |
| **Fundamentals** | — | Anti-leak processed ROA, ROE, P/E-ttm |
| **Macroeconomic** | 3 | Brent Oil, XAU (Gold), USD/IRR exchange rate |

---

## 🏗️ Architecture: The Four Strategic Layers

The core model follows a **Plug-and-Play** design with four sequential layers:

```
Input (N=42 features)
       │
       ▼
┌─────────────────┐
│   RevIN Layer   │  ← Reversible Instance Normalization (mitigates non-stationarity)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Temporal CNN   │  ← Extracts local micro-patterns & filters market noise
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  iTransformer Block  │  ← Inverted attention: models Cross-Variable correlations
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  BiLSTM Aggregator   │  ← Bidirectional temporal dynamics, prevents memory loss
└────────┬─────────────┘
         │
         ▼
     Output (Price Forecast)
```

---

## 📊 Strategic Evaluation Suite

MDL-v5 includes a comprehensive **9-metric evaluation module**:

| Metric | Type | Purpose |
|---|---|---|
| RMSE | Error | Root Mean Squared Error |
| MAPE | Error | Mean Absolute Percentage Error |
| R² | Fit | Coefficient of Determination |
| RMSRE | Error | Root Mean Squared Relative Error |
| RMSPE | Error | Root Mean Squared Percentage Error |
| MARE | Error | Mean Absolute Relative Error |
| Theil's U2 | Comparative | Forecast vs. naïve benchmark |
| DirAcc | Directional | Directional Accuracy (trend prediction) |
| DM-Test | Statistical | Diebold-Mariano significance test |

### Interpretability (SHAP)
- 🌍 **Global importance** plots across all features
- 🔍 **Local importance** plots for individual predictions

### Regime Shift Analysis
Automated detection of structural market turning points using the **PELT algorithm** (Pruned Exact Linear Time).

---

## 📂 Repository Structure

```
MDL-v5-TSE-Forecasting/
│
├── 📁 data/
│   ├── technical_features_v8.py     # Technical Feature Engineering v8
│   └── master_merger_v4.py          # Master Merger pipeline v4
│
├── 📁 models/
│   └── mdl_architecture.py          # Core MDL model definitions
│
├── 📁 training/
│   └── pipeline_v6_singlestock.py   # v6 Single-Stock Pipeline
│                                    # (Cosine Annealing + SWA optimizer)
│
├── 📁 evaluation/
│   └── metrics_and_viz.py           # Full 9-metric evaluation & visualization
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Mojtaba97sh/MDL-v5-TSE-Forecasting.git
cd MDL-v5-TSE-Forecasting

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
shap>=0.42.0
ruptures>=1.1.7
matplotlib>=3.7.0
seaborn>=0.12.0
ta>=0.10.0
```

---

## 🚀 Quick Start

```python
from models.mdl_architecture import MDLv5
from training.pipeline_v6_singlestock import SingleStockPipeline

# Initialize pipeline for a single TSE ticker
pipeline = SingleStockPipeline(
    ticker="IKCO",       # نماد: ایران خودرو
    seq_len=60,
    pred_len=5,
    n_features=42
)

# Train with Cosine Annealing + SWA
pipeline.train(epochs=100)

# Evaluate with 9-metric suite
results = pipeline.evaluate()
print(results)
```

---

## 📜 CRediT Authorship Statement

| Contributor | Role |
|---|---|
| **Mojtaba Sh** | Methodology, Software, Data Curation, Visualization, Tehran Stock Exchange Adaptation |
| **Dr. Asgar Noorbakhsh** | Supervision, Conceptualization, and Review |

*University of Tehran — Department of Financial Engineering and Risk Management*

---

## ⚖️ License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions regarding this research, please open a GitHub Issue or contact via the University of Tehran.

---

*© 2026 Mojtaba Sh — University of Tehran*
