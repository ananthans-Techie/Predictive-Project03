# Water Quality Classification & Monitoring System

**A Real-Time, Region-Aware Water Quality Analysis Platform with IoT, ML, and Streamlit Web App**

---

## 🚀 Project Overview
This project provides a complete workflow for analyzing, predicting, and monitoring water quality using machine learning, real-time IoT sensor simulation, and a modern web interface.

- **Classifies water as Potable or Not Potable** using 9+ chemical/physical features
- **Calculates Water Quality Index (WQI)** for interpretability
- **Supports region-specific data** for localized analysis
- **Streamlit web app** for interactive use and deployment
- **IoT sensor simulation** for real-time monitoring

---

## 🧩 Features
- Data cleaning, missing value handling, and EDA
- Model training (Random Forest, etc.) and evaluation
- Water potability prediction system
- Water Quality Index (WQI) calculation and display
- Real-time IoT sensor simulation and monitoring
- Region-specific data integration and merging
- Streamlit web application for deployment

---

## 🗂️ Project Structure
```bash
Predictive-Project03/
│
├── app.py                    # Streamlit web app (main entrypoint)
├── Water Quality Classification.ipynb # Jupyter notebook (EDA, ML)
├── merge_region_data.py      # Script to merge region-specific datasets
├── requirements.txt          # Python dependencies
├── scaler.pkl                # Saved feature scaler
├── water_model.pkl           # Trained ML model
├── README.md                 # Project documentation
└── ...
```

---

## 📊 Dataset

The dataset contains water quality indicators and a target label for potability. You can use your own or merge with region-specific datasets.

**Features:**
- pH (acidity/alkalinity)
- Hardness (mg/L)
- Solids (ppm)
- Chloramines (ppm)
- Sulfate (mg/L)
- Conductivity (μS/cm)
- Organic Carbon (ppm)
- Trihalomethanes (μg/L)
- Turbidity (NTU)
- region (optional, for region-aware analysis)

**Target:**
- Potability (1 = Safe, 0 = Not Safe)

**Sample sources:**
- [Kaggle: Water Potability Dataset](https://www.kaggle.com/datasets/nayanack/water-probability)
- Local or government water quality datasets

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ananthans-Techie/Predictive-Project03.git
   cd Predictive-Project03
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare your dataset:**
   - Download and place your CSV (e.g., `water_potability.csv`) in the project folder.
   - (Optional) Merge region-specific data using `merge_region_data.py`.

---

## 🌐 Running the Streamlit Web App

```bash
streamlit run app.py
```

- The app will open in your browser (default: http://localhost:8501)
- Use **Manual Input** or **IoT Simulation** modes
- View predictions, WQI, and feature analysis interactively

---

## 🧬 Workflow & Architecture

1. **Data Preparation:**
   - Clean and merge datasets (optionally by region)
   - Handle missing values
2. **Model Training:**
   - Train Random Forest (or other) classifier
   - Save model and scaler as `.pkl` files
3. **Web App (Streamlit):**
   - Manual input or simulated IoT sensor readings
   - Real-time prediction and WQI calculation
   - Results displayed with confidence and interpretability
4. **Region-Specific Analysis:**
   - Add a `region` column for localized insights
   - Visualize and compare water quality by region

---

## 📦 Key Files
- `app.py` — Streamlit web app (main deployment target)
- `requirements.txt` — All dependencies for local or cloud deployment
- `merge_region_data.py` — Merge/append region-specific datasets
- `Water Quality Classification.ipynb` — EDA, ML, and model training
- `scaler.pkl`, `water_model.pkl` — Saved model artifacts

---

## 🌍 Deployment
- Deploy on [Streamlit Cloud](https://share.streamlit.io/) or any cloud supporting Streamlit
- Set `app.py` as the entrypoint
- Make sure your dataset and model files are included or retrain as needed

---

## 🛡️ Technologies Used
- Python, Pandas, NumPy, scikit-learn
- Streamlit (web app)
- Matplotlib, Seaborn (EDA)

---

## 🤝 Contributing
ANANTHAN S, SHAHID MUHAMMAD

---

## 📄 License
This project is open-source and available under the MIT License.





