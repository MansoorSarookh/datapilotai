<!-- <![CDATA[<div align="center"> -->

# 🧠 DataPilot AI

### The AI Data Intelligence Copilot

**Understand · Clean · Model · Decide**

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://datapilot-ai.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-10b981.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-3.0-667eea.svg)](#)

> **DataPilot AI v3.0** is a full-featured, AI-powered data intelligence platform that automates
> exploratory data analysis, intelligent data cleaning, statistical testing, ML model training
> with explainability, and professional report generation — all from a single file upload.

**Live App →** [datapilot-ai.streamlit.app](https://datapilot-ai.streamlit.app)
&nbsp;|&nbsp;
**Repository →** [GitHub.com/MansoorSarookh/DataPilot-AI](https://GitHub.com/MansoorSarookh/DataPilot-AI)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [What's New in v3.0](#-whats-new-in-v30)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🧠 Overview

DataPilot AI is designed to streamline the **entire data analysis lifecycle** — from raw file upload to cleaned dataset, statistical conclusions, trained ML models, and shareable reports — using AI-assisted workflows at every step.

It is built for:

- 🎓 **Students & Researchers** — quick dataset exploration without writing code
- 📊 **Analysts & Data Scientists** — rapid prototyping, statistical testing, and cleaning
- 🏢 **Teams & Decision-Makers** — shareable PDF/HTML reports and Jupyter notebooks
- 🛡️ **Privacy-Conscious Users** — built-in GDPR/PII scanning and audit trails

The platform emphasizes **usability, speed, interpretability, and data quality**.

---

## 🆕 What's New in v3.0

| Area                     | Upgrade                                                                 |
| ------------------------ | ----------------------------------------------------------------------- |
| **Architecture**         | Modular codebase with 30+ dedicated engine modules & UI components      |
| **AI Copilot**           | Multi-provider LLM router (Groq · Gemini · Ollama) with auto-fallback  |
| **ML Studio**            | Train 10+ algorithms, Model Comparison Arena, SHAP explainability       |
| **Hyperparameter Tuning**| Optuna-powered HPO engine with Bayesian optimization                    |
| **Data Cleaning**        | AI-assisted cleaning, fuzzy deduplication, type repair, anomaly removal |
| **Statistics**           | Hypothesis testing, power analysis, regression analysis, time-series    |
| **Privacy**              | GDPR / PII scanner with risk classification and remediation guidance    |
| **Trust Score**          | 5-dimension dataset quality scoring (completeness, consistency, etc.)   |
| **Export**               | PDF reports · Interactive HTML dashboards · Jupyter notebooks · PPTX    |
| **File Support**         | 12 file formats: CSV, Excel, ODS, TSV, JSON, Parquet, Feather, PDF, Word, HTML, and more |

---

## ✨ Features

### 📁 Overview & Trust Score
- **Dataset Trust Score** — multi-dimensional quality assessment across completeness, consistency, uniqueness, validity, and timeliness
- Schema detection, column-type summaries, missing-value heatmaps
- Data dictionary generation with column metadata

### 📊 Interactive Visualizations
- **25+ chart types** — histograms, box plots, violin plots, KDE, scatter, line, bar, pie, heatmaps, 3D scatter, parallel coordinates, bubble, sunburst, treemaps, pair plots
- **Time-series** — line with range slider, area charts, candlestick, animated series, rolling statistics
- **AI Chart Recommender** — suggests the best chart types based on column data characteristics
- Customizable Plotly themes (7 built-in palettes)
- Export charts as PNG, SVG, interactive HTML, or animated GIF

### 🤖 AI Copilot
- Natural-language Q&A about your dataset
- AI-generated insight summaries, trend explanations, and anomaly hints
- **Multi-Provider LLM Router** — automatically routes to the best available provider:
  - Groq (Llama 3.3 70B)
  - Google Gemini Flash
  - Ollama (local models)
  - Rule-based heuristic fallback (works offline)

### 📐 Advanced Statistics
- Descriptive statistics with distribution analysis
- **Hypothesis testing** — t-test, chi-square, ANOVA, and more via Pingouin
- **Power analysis** — sample size and effect size calculations
- **Regression analysis** — OLS and advanced regression modeling
- **AI Stats Narrator** — natural-language interpretation of statistical results
- Outlier detection and categorical analysis

### 🧹 Intelligent Data Cleaning
- **AI-assisted cleaning** — auto-detect issues and suggest fixes
- Missing-value imputation (mean, median, mode, forward/backward fill, ML-based)
- Duplicate detection and **fuzzy deduplication** (RapidFuzz)
- **Type fixer** — automatic type coercion and repair
- **Anomaly detection** — Isolation Forest, LOF, Z-Score, ensemble methods
- **GDPR / PII scanner** — detects emails, phones, SSNs, credit cards, IPs with risk classification
- **Cluster profiling** — auto-segment data and generate persona descriptions
- Full audit trail for all cleaning operations

### 🎯 ML Studio
- **ML Readiness Advisor** — assesses if your data is ML-ready with actionable checks
- **Model Comparison Arena** — train and benchmark 10+ algorithms simultaneously:
  - Random Forest, Logistic/Linear Regression, XGBoost, LightGBM, SVM/SVR, KNN, Decision Tree, Naive Bayes, Gradient Boosting
- **Hyperparameter Optimization** — Optuna-powered Bayesian tuning
- **SHAP Explainability** — feature importance and model explanation
- Supports both classification and regression tasks
- Cross-validation and proper train/test splitting

### 📥 Multi-Format Export
- **PDF Reports** — executive-quality reports with charts, trust scores, and AI summaries
- **Interactive HTML Dashboards** — self-contained dashboards with embedded Plotly charts
- **Jupyter Notebooks** — reproducible `.ipynb` with all analysis steps preserved
- **PowerPoint** — presentation-ready export
- **Data Export** — cleaned datasets in CSV, Excel, JSON, Parquet

### 📂 File Format Support

| Category          | Formats                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| **Spreadsheets**  | CSV · Excel (.xlsx / .xls) · ODS · TSV · TXT                           |
| **Documents**     | PDF (table extraction) · Word (.docx, table extraction) · HTML tables   |
| **Data Formats**  | JSON · Apache Parquet · Feather                                         |
| **Limits**        | Up to **500 MB** per file · auto-encoding detection · smart type inference |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI                            │
│  ┌──────────┬──────────┬────────┬────────┬────────┬───────────┐  │
│  │ Overview │ Visualize│AI Chat │ Stats  │ Clean  │ ML Studio │  │
│  └────┬─────┴────┬─────┴───┬────┴───┬────┴───┬────┴─────┬─────┘  │
│       │          │         │        │        │          │         │
│  ┌────▼──────────▼─────────▼────────▼────────▼──────────▼─────┐  │
│  │                    COMPONENT LAYER                         │  │
│  │  sidebar · data_preview · stats_panel · clean_panel ·      │  │
│  │  ml_studio · ai_chat · report_panel · export · trust_score │  │
│  └────┬───────────────────────────────────────────────────────┘  │
│       │                                                          │
│  ┌────▼───────────────────────────────────────────────────────┐  │
│  │                     ENGINE LAYER (30+ modules)             │  │
│  │  ai_engine · llm_router · eda_engine · viz_engine ·        │  │
│  │  stats_engine · cleaner · ml_advisor · model_arena ·       │  │
│  │  hpo_engine · shap_explainer · anomaly_detector ·          │  │
│  │  trust_score · chart_recommender · gdpr_scanner ·          │  │
│  │  fuzzy_dedup · type_fixer · time_series · report_generator │  │
│  │  notebook_exporter · dashboard_exporter · audit_trail      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Workflow:**

1. User uploads a dataset via the sidebar
2. File parser detects format, encoding, and schema
3. Trust Score engine computes 5-dimension quality assessment
4. User navigates tabs: Overview → Visualize → AI Copilot → Statistics → Clean → ML Studio → Export
5. AI insights, cleaning, and ML training are processed in real-time
6. Results are exported as PDF, HTML dashboards, or Jupyter notebooks

---

## 🛠️ Tech Stack

| Category                | Technologies                                                   |
| ----------------------- | -------------------------------------------------------------- |
| **Core Language**       | Python 3.11                                                    |
| **Web Framework**       | Streamlit ≥ 1.31                                               |
| **Data Processing**     | Pandas · NumPy · PyArrow · Chardet                             |
| **Visualization**       | Plotly 5.20 · Kaleido (static export)                          |
| **Statistics**          | SciPy · Statsmodels · Pingouin                                 |
| **Machine Learning**    | Scikit-learn · XGBoost · LightGBM · Imbalanced-learn           |
| **Explainability**      | SHAP                                                           |
| **Hyperparameter Tuning** | Optuna                                                      |
| **AI / LLM**           | Groq (Llama 3.3 70B) · Google Gemini · Ollama                  |
| **File Parsing**        | OpenPyXL · xlrd · odfpy · pdfplumber · python-docx · BeautifulSoup · lxml |
| **Export & Reports**    | FPDF2 · nbformat · python-pptx · Pillow · Jinja2              |
| **Data Quality**        | RapidFuzz · Pandera                                            |

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://GitHub.com/MansoorSarookh/DataPilot-AI.git
cd DataPilot-AI
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure API keys (optional — for AI features)

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
GEMINI_API_KEY = "your-gemini-api-key"
```

> **Note:** The app works fully without API keys — AI features fall back to rule-based heuristics.

### 5️⃣ Run the app

```bash
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

---

## ▶️ Usage

1. **Launch** the application
2. **Upload** your dataset (CSV, Excel, PDF, Parquet, or any supported format)
3. **Explore** the seven analysis tabs:

| Tab             | What It Does                                                            |
| --------------- | ----------------------------------------------------------------------- |
| 📁 **Overview**  | Trust Score, data preview, column info, missing-value analysis          |
| 📊 **Visualize** | 25+ interactive chart types with AI recommendations                    |
| 🤖 **AI Copilot**| Ask questions about your data in plain English                         |
| 📐 **Statistics**| Hypothesis testing, power analysis, regression, distribution analysis  |
| 🧹 **Clean**     | AI-assisted cleaning, dedup, PII scan, anomaly detection               |
| 🎯 **ML Studio** | Train models, compare algorithms, tune hyperparameters, SHAP analysis  |
| 📥 **Export**     | Download reports (PDF, HTML, Jupyter, PPTX) and cleaned data           |

No configuration required — the pipeline runs automatically on upload.

---

## 📁 Project Structure

```
DataPilot-AI/
│
├── app/
│   ├── main.py                      # Main Streamlit application (v3.0)
│   ├── config.py                    # App settings, chart types, color palettes
│   │
│   ├── components/                  # UI Components
│   │   ├── ai_chat.py               # AI Copilot chat interface
│   │   ├── clean_panel.py           # Data cleaning panel
│   │   ├── data_preview.py          # Dataset preview widget
│   │   ├── export.py                # Export panel (charts & data)
│   │   ├── ml_studio.py             # ML Studio interface
│   │   ├── report_panel.py          # Report generation panel
│   │   ├── sidebar.py               # Sidebar layout
│   │   ├── statistics.py            # Statistics display component
│   │   ├── stats_panel.py           # Advanced statistics panel
│   │   └── trust_score_display.py   # Trust Score widget
│   │
│   └── modules/                     # Backend Engines (30+ modules)
│       ├── ai_engine.py             # AI insight generation
│       ├── llm_router.py            # Multi-provider LLM router
│       ├── eda_engine.py            # Exploratory data analysis engine
│       ├── viz_engine.py            # Visualization engine (25+ chart types)
│       ├── stats_engine.py          # Statistical analysis engine
│       ├── stats_narrator.py        # AI statistics narrator
│       ├── cleaner.py               # Intelligent data cleaning engine
│       ├── type_fixer.py            # Automatic type coercion & repair
│       ├── fuzzy_dedup.py           # Fuzzy deduplication (RapidFuzz)
│       ├── anomaly_detector.py      # Anomaly detection (IF, LOF, Z-Score)
│       ├── gdpr_scanner.py          # GDPR / PII scanner
│       ├── trust_score.py           # Dataset Trust Score engine
│       ├── ml_advisor.py            # ML readiness advisor & algorithms
│       ├── model_arena.py           # Model Comparison Arena
│       ├── hpo_engine.py            # Optuna hyperparameter optimizer
│       ├── shap_explainer.py        # SHAP explainability engine
│       ├── chart_recommender.py     # AI chart recommendation engine
│       ├── cluster_profiler.py      # Auto-clustering & persona generation
│       ├── time_series.py           # Time-series detection & analysis
│       ├── ts_stats_engine.py       # Time-series statistics engine
│       ├── regression_engine.py     # Regression analysis engine
│       ├── power_analysis.py        # Statistical power analysis
│       ├── data_dictionary.py       # Auto data dictionary generation
│       ├── report_generator.py      # PDF & HTML report generator
│       ├── notebook_exporter.py     # Jupyter notebook exporter
│       ├── dashboard_exporter.py    # Interactive HTML dashboard exporter
│       ├── file_parser.py           # Multi-format file parser
│       ├── session_manager.py       # Session state manager
│       └── audit_trail.py           # Cleaning audit trail
│
├── assets/
│   └── styles.css                   # Custom CSS theme (dark mode)
│
├── .streamlit/
│   └── config.toml                  # Streamlit theme & server config
│
├── requirements.txt                 # Python dependencies (57 packages)
├── runtime.txt                      # Python version specification
└── README.md
```

---

## 🗺️ Roadmap

- ✅ Automated EDA with Trust Score
- ✅ 25+ interactive visualization types
- ✅ AI Copilot with multi-provider LLM support
- ✅ Advanced statistical testing & power analysis
- ✅ AI-assisted data cleaning & GDPR/PII scanner
- ✅ ML Studio with 10+ algorithms
- ✅ Model Comparison Arena & SHAP explainability
- ✅ Hyperparameter optimization (Optuna)
- ✅ PDF / HTML Dashboard / Jupyter Notebook / PPTX export
- ✅ Fuzzy deduplication & anomaly detection
- ✅ Time-series analysis & detection
- 🔜 Real-time collaborative analysis
- 🔜 Plugin ecosystem for custom analysis modules
- 🔜 API endpoint for programmatic access
- 🔜 Natural-language SQL query interface
- 🔜 Multi-dataset join and merge support

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

Please ensure your code is documented, tested, and follows the existing module structure.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and modify with attribution.

---

## 👨‍💻 Author

**Mansoor Sarookh**
Computer Science Student & AI Developer

[![GitHub](https://img.shields.io/badge/GitHub-MansoorSarookh-181717?logo=github)](https://GitHub.com/MansoorSarookh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mansoorsarookh-0A66C2?logo=linkedin)](https://linkedin.com/in/mansoorsarookh)
[![YouTube](https://img.shields.io/badge/YouTube-Mansoor--Sarookh-FF0000?logo=youtube)](https://youtube.com/Mansoor-Sarookh)
[![Instagram](https://img.shields.io/badge/Instagram-mansoorsarookh-E4405F?logo=instagram)](https://instagram.com/mansoorsarookh)
[![Facebook](https://img.shields.io/badge/Facebook-mansoorsarookh-1877F2?logo=facebook)](https://facebook.com/mansoorsarookh)
[![X](https://img.shields.io/badge/X-mansoorsarookh-000000?logo=x)](https://x.com/mansoorsarookh)
[![Kaggle](https://img.shields.io/badge/Kaggle-mansoorsarookhh-20BEFF?logo=kaggle)](https://kaggle.com/mansoorsarookhh)

---

<div align="center">

**Made with 🧠 by Mansoor Sarookh**

*DataPilot AI — Transform raw data into intelligence.*

</div>
]]>
