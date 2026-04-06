#  Water Quality Analysis & Prediction  
*A Machine Learning-based Environmental Monitoring System*

---

##  Project Overview  
This project focuses on analyzing and predicting **water quality** using Machine Learning techniques. The goal is to determine whether water is **safe for consumption (potable) or not** based on various chemical and physical properties.

The system processes water parameters such as **pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity** to classify water quality.

---

##  Features  
- Data preprocessing and cleaning  
- Handling missing values  
- Exploratory Data Analysis (EDA)  
- Feature correlation analysis  
- Machine Learning classification model  
- Water potability prediction system  
- Model performance evaluation  

---

##  Project Structure
```bash
Water-Quality-Prediction/
│
├── water_quality_analysis.ipynb # Main Kaggle notebook
├── README.md # Project documentation
└── dataset/ # Dataset (external / Kaggle)
```

---

##  Dataset  

The dataset contains various **water quality indicators** used to determine potability.

###  Features:
- **pH** → Acidity/alkalinity of water  
- **Hardness** → Mineral content  
- **Solids (TDS)** → Total dissolved solids  
- **Chloramines** → Disinfectant level  
- **Sulfate** → Mineral concentration  
- **Conductivity** → Electrical conductivity of water  
- **Organic Carbon** → Organic pollutants  
- **Trihalomethanes** → Chemical compounds from disinfection  
- **Turbidity** → Water clarity  

###  Target:
- **Potability**  
  - 1 → Safe to drink  
  - 0 → Not safe
---


##  Dataset

The dataset is not included in this repository due to size limitations.

Download it from:
https://www.kaggle.com/datasets/nayanack/water-probability

### Steps:

1. Download the dataset from the link above
2. Extract the files
3. Place the dataset inside the folder:

```bash
water_potability.csv
```

---
Installation & Setup
1 Clone the repository:

```bash
git clone https://github.com/ananthans-Techie/Predictive-Project03.git
cd Water-Quality-Prediction
```

2 Install required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3 Run the project:
```bash
jupyter notebook water_quality_analysis.ipynb
```

---

### Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook / Kaggle

---

### Model Workflow
Load Dataset
Data Preprocessing
Handle missing values
Feature scaling
Exploratory Data Analysis (EDA)
Distribution plots
Correlation heatmap
Feature Selection
Model Training
Logistic Regression
Decision Tree
Random Forest
Model Evaluation
Accuracy
Confusion Matrix
Precision & Recall
Prediction System
Input parameters
Predict water potability

--- 

### Future Improvements
Use Deep Learning models (ANN, XGBoost)
Integrate real-time IoT sensors
Deploy as a web application (Flask / Streamlit)
Add Water Quality Index (WQI)
Improve dataset with region-specific data

---

### Author

ANANTHAN S





