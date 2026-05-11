# 💧 Water Quality Classification for Safe Drinking Prediction
 
> A Real-Time, Region-Aware Water Quality Analysis Platform with Machine Learning, IoT Sensor Simulation, and Streamlit Deployment
 
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-blue?logo=streamlit)](https://predictive-project03-ug6mfxcp9ag8rssuuqvt27.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Predictive--Project03-black?logo=github)](https://github.com/ananthans-Techie/Predictive-Project03)
 
---
 
## 📌 Table of Contents
 
- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Team Members](#-team-members)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Data Science Life Cycle](#-data-science-life-cycle)
- [Models & Results](#-models--results)
- [Feature Engineering](#-feature-engineering)
- [SHAP Explainability](#-shap-explainability)
- [IoT Sensor Simulation](#-iot-sensor-simulation)
- [Streamlit Web App](#-streamlit-web-app)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [GitHub Collaboration](#-github-collaboration)
- [Technologies Used](#-technologies-used)
- [Limitations & Future Work](#-limitations--future-work)
---
 
## 🚀 Project Overview
 
Access to safe drinking water is a fundamental human right, yet billions of people worldwide face risks from contaminated water sources. Traditional lab-based water testing is slow, expensive, and inaccessible in many regions.
 
This project builds a complete **end-to-end machine learning pipeline** that:
 
- Classifies water samples as **Potable (Safe)** or **Non-Potable (Unsafe)** using physicochemical properties
- Calculates a **Water Quality Index (WQI)** for intuitive, interpretable scoring
- Handles **class imbalance** using SMOTE oversampling
- Provides **model explainability** through SHAP (SHapley Additive exPlanations)
- Simulates **real-time IoT sensor monitoring** for continuous water quality tracking
- Delivers a **live Streamlit web application** for interactive predictions
---
 
## 🌐 Live Demo
 
> **Streamlit App:** https://predictive-project03-ug6mfxcp9ag8rssuuqvt27.streamlit.app/

 <img width="1920" height="881" alt="image" src="https://github.com/user-attachments/assets/a22215fc-1a12-4cab-a107-6a19fc16d156" />

 <img width="1920" height="885" alt="image" src="https://github.com/user-attachments/assets/93f161a0-e193-492d-87d8-36cc6c896b95" />

 
---
 
## 👨‍💻 Team Members
 
| Name | GitHub |
|------|--------|
| **Ananthan S** | [@ananthans-Techie](https://github.com/ananthans-Techie) |
| **Muhammed Shahid** | [@muhammedshahidsds25-collab](https://github.com/muhammedshahidsds25-collab) |
 
**Course:** Predictive Analytics — Academic Year 2025–26  
**Project:** Group Project 03
 
---
 
## 🎯 Problem Statement
 
Water contamination is a global crisis. This project addresses the challenge of **automated, rapid water potability classification** using machine learning — reducing dependence on slow laboratory testing.
 
**Goal:** Given 9 physicochemical measurements from a water sample, predict whether the water is **safe to drink** with high recall (minimising false "safe" predictions on unsafe water).
 
**Why it matters:**
- 2 billion people lack access to safely managed drinking water (WHO, 2023)
- Traditional testing takes 24–72 hours; ML inference takes milliseconds
- Interpretable predictions (SHAP) allow environmental engineers to act on specific contaminants
---
 
## 📊 Dataset
 
**Source:** [Kaggle — Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
 
| Property | Value |
|----------|-------|
| Total Samples | 3,276 |
| Features | 9 physicochemical properties |
| Target | `Potability` (0 = Not Potable, 1 = Potable) |
| Class Distribution | 1,998 Not Potable (61%) · 1,278 Potable (39%) |
| Missing Values | `ph`: 491 · `Sulfate`: 781 · `Trihalomethanes`: 162 |
 
### Feature Descriptions
 
| Feature | Unit | WHO Safe Range | Missing |
|---------|------|---------------|---------|
| `ph` | — | 6.5 – 8.5 | 15.0% |
| `Hardness` | mg/L | < 300 | 0% |
| `Solids` | ppm | < 500 | 0% |
| `Chloramines` | ppm | < 4.0 | 0% |
| `Sulfate` | mg/L | < 250 | 23.8% |
| `Conductivity` | μS/cm | < 400 | 0% |
| `Organic_carbon` | ppm | < 2.0 | 0% |
| `Trihalomethanes` | μg/L | < 80 | 4.9% |
| `Turbidity` | NTU | < 1.0 | 0% |
 
**Target:** `Potability` — 1 = Safe to drink, 0 = Not safe
 
---
 
## 🏗️ Project Architecture
 
```
Raw CSV Data
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              Data Preprocessing                      │
│  Group-wise Median Imputation → Winsorization (IQR) │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│           Exploratory Data Analysis                  │
│  Distributions · Correlation · Boxplots · Imbalance │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│           Feature Engineering                        │
│  WQI Score · pH Safety Flag · Hardness/Cond Ratio   │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│     Train/Test Split (80/20) + StandardScaler        │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│        SMOTE Oversampling (on training set only)     │
│        Balances minority class (Potable)             │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Random Forest│  │     SVM      │  │   XGBoost    │
│ (200 trees)  │  │ (RBF kernel) │  │ (300 trees)  │
└──────────────┘  └──────────────┘  └──────────────┘
     │                  │                  │
     └──────────────────┴──────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │  Model Evaluation       │
          │  Accuracy · F1 · ROC    │
          │  Confusion Matrix       │
          └─────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │  SHAP Explainability    │
          │  Feature Impact         │
          └─────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │  Streamlit Web App      │
          │  Live Deployment        │
          └─────────────────────────┘
```
 
---
 
## 🔬 Data Science Life Cycle
 
All 10 stages of the data science life cycle are covered:
 
### Stage 1 — Problem Definition & Literature Review
Defined the potability classification task with reference to WHO guidelines. Reviewed literature on ML-based water quality prediction (Eze et al. 2021, Ismail et al. 2022, Tyagi & Singh 2020).
 
### Stage 2 — Data Collection & Understanding
Loaded the Water Potability dataset (3,276 samples × 10 columns). Analysed data types, missing values, class distribution, and statistical summaries.
 
### Stage 3 — Data Preprocessing & Cleaning
- **Missing values:** Group-wise median imputation (by Potability class) for `ph`, `Sulfate`, `Trihalomethanes` — preserves class signal better than global median
- **Outliers:** Winsorization using IQR (1.5× fence) — caps extreme values without removing rows
### Stage 4 — Exploratory Data Analysis (EDA)
- Feature distributions by class (histograms)
- Correlation heatmap (all 12 features)
- Boxplots — potable vs non-potable per feature
- Class imbalance visualisation (bar + pie chart)
### Stage 5 — Feature Engineering & Selection
Three new features engineered (see [Feature Engineering](#-feature-engineering)):
- `WQI` — Water Quality Index (weighted composite score)
- `pH_safe` — Binary flag (1 if pH in WHO safe range 6.5–8.5)
- `Hard_Cond_ratio` — Hardness to Conductivity ratio
Class imbalance handled using **SMOTE** (Synthetic Minority Over-sampling Technique).
 
### Stage 6 — Model Building & Training
Three classifiers trained on SMOTE-balanced data with 5-fold stratified cross-validation:
- **Random Forest** — 200 estimators, max_depth=12
- **SVM** — RBF kernel, C=10, gamma='scale'
- **XGBoost** — 300 estimators, learning_rate=0.05
### Stage 7 — Model Evaluation & Comparison
Evaluated on held-out test set (20%):
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices (all 3 models)
- ROC curves overlay
- Model comparison bar chart
### Stage 8 — Model Interpretation & Explainability
SHAP analysis on best model (XGBoost):
- Global feature importance (bar plot)
- Beeswarm plot (direction + magnitude per sample)
- Waterfall plot (individual prediction explanation)
### Stage 9 — Deployment
- Best model (XGBoost) saved as `water_model.pkl`
- Scaler saved as `scaler.pkl`
- Feature columns saved as `feature_cols.pkl`
- Deployed as interactive Streamlit web app
### Stage 10 — Documentation
- Detailed README (this file)
- Jupyter notebook with clean, commented cells
- PPT presentation in repository
- IoT sensor simulation for real-time demo
- Individual GitHub contribution profiles
---
 
## 📈 Models & Results
 
### Model Comparison Table
 
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.79 | 0.76 | 0.73 | 0.74 | 0.86 |
| SVM (RBF) | 0.74 | 0.72 | 0.68 | 0.70 | 0.82 |
| **XGBoost** ⭐ | **0.81** | **0.78** | **0.76** | **0.77** | **0.88** |
 
> *Exact values will vary slightly with each run due to SMOTE randomness. Results above are representative.*
 
**Winner: XGBoost** — best across all five metrics, deployed in production.
 
### Why XGBoost Won
- Gradient boosting corrects errors iteratively — better at capturing complex feature interactions
- Regularisation (L1 + L2) prevents overfitting on the small dataset
- Native handling of feature importance aligns well with SHAP
### Evaluation Plots
 
*(Screenshots of confusion matrices, ROC curves, and model comparison charts will be added after deployment)*
 
---
 
## ⚙️ Feature Engineering
 
Three features were engineered beyond the original 9:
 
### 1. Water Quality Index (WQI)
A weighted composite score (0–1) combining the most health-critical parameters:
 
```
WQI = (pH/8.5 × 0.15) + ((1 - Turbidity/6.7) × 0.20) +
      ((1 - Chloramines/13.1) × 0.20) + ((1 - THMs/124) × 0.15) +
      ((1 - Solids/61227) × 0.15) + ((1 - Sulfate/481) × 0.15)
```
 
Higher WQI = safer water. Used directly in Streamlit app display.
 
### 2. pH Safety Flag (`pH_safe`)
Binary indicator: `1` if pH ∈ [6.5, 8.5] (WHO recommended range), `0` otherwise.
 
### 3. Hardness-to-Conductivity Ratio
`Hard_Cond_ratio = Hardness / Conductivity` — captures mineral concentration relative to ion mobility.
 
---
 
## 🔍 SHAP Explainability
 
SHAP (SHapley Additive exPlanations) makes the model transparent — critical when predictions affect public health decisions.
 
| Plot Type | What It Shows |
|-----------|--------------|
| **Bar Plot** | Global feature importance (mean absolute SHAP) |
| **Beeswarm** | Per-sample impact — direction and magnitude per feature |
| **Waterfall** | Single prediction explained step-by-step |
 
**Key Findings from SHAP:**
- `Sulfate` and `Solids` are the most influential raw features
- Engineered `WQI` consistently ranks in the top 3 contributors
- High `Chloramines` strongly pushes predictions toward Not Potable
---
 
## 🤖 IoT Sensor Simulation
 
The notebook includes a real-time IoT monitoring simulation that:
- Generates synthetic sensor readings within realistic physical ranges
- Computes WQI and engineered features on-the-fly
- Runs XGBoost inference per reading
- Outputs a live monitoring table with timestamps and confidence scores
```
 # |     Time |    pH | Turbidity |   WQI |        Result | Conf
------------------------------------------------------------------------
 1 | 01:33:12 |   7.3 |      3.21 | 0.612 |  OK  Potable  | 84.3%
 2 | 01:33:12 |   5.1 |      5.87 | 0.341 | !!! Not Potable | 91.2%
 3 | 01:33:13 |   7.9 |      2.44 | 0.708 |  OK  Potable  | 88.6%
```
 
This simulates how the model would behave connected to physical IoT water quality sensors.
 
---
 
## 🌐 Streamlit Web App
 
The Streamlit app (`app.py`) provides a full interactive interface:
 
### Features
- **Sidebar sliders** for all 9 raw water parameters
- **Automatic WQI computation** from inputs
- **One-click prediction** with XGBoost
- **Confidence score** and probability bar chart
- **Input summary table** with all values including engineered features
- **Graceful error handling** for edge-case inputs
### Running Locally
```bash
streamlit run app.py
```
 
### Deploying to Streamlit Cloud
1. Push all files (including `water_model.pkl`, `scaler.pkl`, `feature_cols.pkl`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set **Main file path:** `app.py`
5. Click **Deploy** — your app will be live in ~2 minutes
---
 
## 🗂️ Project Structure
 
```
Predictive-Project03/
│
├── 📓 Water_Quality_Classification.ipynb   # Main notebook — all 10 stages
├── 🌐 app.py                               # Streamlit web application
├── 🔀 merge_region_data.py                 # Region-specific dataset merging
│
├── 🤖 water_model.pkl                      # Trained XGBoost model
├── ⚖️  scaler.pkl                           # Fitted StandardScaler
├── 📋 feature_cols.pkl                     # Feature column names list
│
├── 📄 requirements.txt                     # Python dependencies
├── 📖 README.md                            # This file
│
├── 📁 individual_profiles/                 # GitHub activity screenshots
│   ├── ananthan_contribution.png
│   └── shahid_contribution.png
│
└── 📁 screenshots/                         # App and results screenshots
    ├── app_input.png
    ├── app_iot.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── shap_importance.png
    └── model_comparison.png
```
 
---
 
## 🛠️ Installation & Setup
 
### Prerequisites
- Python 3.10+
- pip
### Step 1 — Clone the Repository
```bash
git clone https://github.com/ananthans-Techie/Predictive-Project03.git
cd Predictive-Project03
```
 
### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```
 
### Step 3 — Prepare the Dataset
Download the dataset from Kaggle:
[Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
 
Place the downloaded `water_potability.csv` file in the project root directory.
 
### Step 4 — (Optional) Retrain the Model
Open and run the notebook:
```bash
jupyter notebook Water_Quality_Classification.ipynb
```
Run all cells. This will regenerate `water_model.pkl`, `scaler.pkl`, and `feature_cols.pkl`.
 
---
 
## ▶️ Running the App
 
```bash
streamlit run app.py
```
 
The app opens automatically at [http://localhost:8501](http://localhost:8501)
 
**Usage:**
1. Adjust the **9 parameter sliders** in the sidebar to match your water sample
2. Click **🔍 Predict Water Quality**
3. View the prediction, confidence score, and probability chart
4. Review the input summary table including computed WQI
---
 
## 📦 Requirements
 
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
shap>=0.41.0
streamlit>=1.20.0
matplotlib>=3.6.0
seaborn>=0.12.0
```
 
Install all at once:
```bash
pip install -r requirements.txt
```
 
---
 
## 🤝 GitHub Collaboration
 
Both members contributed across all stages with meaningful commits, pull requests, and collaborative development using GitHub.
 
### Individual Contribution Profiles
See the `/individual_profiles/` folder for exported GitHub activity screenshots for each member.
 
---
 
## 🛡️ Technologies Used
 
| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML Models | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Data Processing | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web App | Streamlit |
| Version Control | Git, GitHub |
| Notebook | Jupyter |
 
---
 
## ⚠️ Limitations & Future Work
 
### Current Limitations
- Dataset is relatively small (3,276 samples) — may not generalise across all geographies
- High missing rate in `Sulfate` (23.8%) introduces imputation noise
- Models trained on a single static dataset — no concept drift handling
- Threshold for potability classification is fixed at 0.5 probability
### Future Work
- **Live IoT Integration** — connect to physical sensors (Arduino/Raspberry Pi) via MQTT
- **Regional Dashboard** — SHAP-powered region-specific water quality maps
- **Drift Detection** — monitor model performance on new incoming data
- **Deep Learning** — explore TabNet for tabular classification
- **Multi-class Severity** — extend from binary (Safe/Unsafe) to severity levels (Safe / Borderline / Critical)
- **Mobile App** — React Native app wrapping the Streamlit API
---
 
## 📄 License
 
This project is open-source and available under the [MIT License](LICENSE).
 
---
 

 
---
 
<div align="center">
**Predictive Analytics — Group Project 03 — Academic Year 2025–26**
 
*Built with passion, not just to finish it.*
 
</div>
 
