"""
Water Quality Classification Web App
Streamlit application for real-time water potability prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Page configuration
st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="💧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .potable {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .not-potable {
        background-color: #ffcdd2;
        color: #c62828;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load or train the model"""
    try:
        model = pickle.load(open('water_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except:
        # Train fresh model if pickle files not found
        df = pd.read_csv(r"C:\Users\sanan\OneDrive\Documents\Predictive_analysis\Project 03\water_potability.csv")
        df = df.fillna(df.median(numeric_only=True))
        
        X = df.drop('Potability', axis=1)
        y = df['Potability']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_scaled, y)
        
        return model, scaler

def main():
    # Header
    st.title("💧 Water Quality Classification")
    st.markdown("### Real-time IoT Water Potability Monitoring System")
    
    # Load model
    model, scaler = load_model()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Mode selection
    mode = st.sidebar.radio("Select Mode:", ["Manual Input", "IoT Simulation"])
    
    if mode == "Manual Input":
        st.header("📊 Manual Water Quality Input")
        
        # Create columns for input fields
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Water Properties")
            ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
            hardness = st.slider("Hardness (mg/L)", 47.0, 324.0, 200.0, 1.0)
            solids = st.slider("Solids (ppm)", 320.0, 61227.0, 20000.0, 100.0)
        
        with col2:
            st.markdown("#### Chemical Properties")
            chloramines = st.slider("Chloramines (ppm)", 0.0, 13.1, 7.0, 0.1)
            sulfate = st.slider("Sulfate (mg/L)", 3.0, 481.0, 300.0, 1.0)
            conductivity = st.slider("Conductivity (μS/cm)", 207.0, 564.0, 400.0, 1.0)
        
        with col3:
            st.markdown("#### Additional Properties")
            organic_carbon = st.slider("Organic Carbon (ppm)", 2.0, 65.0, 15.0, 0.1)
            trihalomethanes = st.slider("Trihalomethanes (μg/L)", 1.0, 124.0, 60.0, 1.0)
            turbidity = st.slider("Turbidity (NTU)", 1.4, 6.7, 4.0, 0.1)
        
        # Create input array
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                               conductivity, organic_carbon, trihalomethanes, turbidity]])

        # Water Quality Index (WQI) calculation (simple normalized average method)
        def calculate_wqi(values):
            # Example weights and ideal values (for demo purposes)
            weights = np.array([0.11, 0.10, 0.10, 0.12, 0.11, 0.10, 0.12, 0.12, 0.12])
            ideal = np.array([7.0, 100, 500, 4.0, 250, 300, 10, 50, 2.5])
            # Normalize (closer to ideal is better)
            norm = 1 - (np.abs(values - ideal) / (np.abs(ideal) + 1e-6))
            norm = np.clip(norm, 0, 1)
            wqi = np.sum(norm * weights) * 100
            return wqi

        # Predict button
        if st.button("🔍 Analyze Water Quality", type="primary"):
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            wqi = calculate_wqi(input_data[0])

            st.markdown("---")
            st.subheader("📋 Results")

            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-box potable">
                        ✅ Water is Potable
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-box not-potable">
                        ❌ Water is Not Potable
                    </div>
                    """, unsafe_allow_html=True)

            with col_res2:
                st.metric("Confidence", f"{max(probability)*100:.1f}%")
                st.metric("Potability Score", f"{probability[1]*100:.1f}%")

            with col_res3:
                st.metric("Water Quality Index (WQI)", f"{wqi:.1f}")
                if wqi >= 80:
                    st.success("Excellent")
                elif wqi >= 60:
                    st.info("Good")
                elif wqi >= 40:
                    st.warning("Poor")
                else:
                    st.error("Very Poor")

            # Feature importance
            st.markdown("---")
            st.subheader("📈 Feature Analysis")

            features = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                       'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity']

            # Show feature values with indicators
            for feature, value in zip(features, input_data[0]):
                col_f1, col_f2 = st.columns([1, 2])
                with col_f1:
                    st.write(f"**{feature}:** {value:.2f}")
                with col_f2:
                    st.progress(min(value / 100, 1.0))
    
    else:  # IoT Simulation mode
        st.header("📡 IoT Sensor Simulation")
        
        # Import required for simulation
        import random
        from datetime import datetime
        
        class IoTSensorSimulator:
            def __init__(self):
                self.feature_ranges = {
                    'ph': (0, 14),
                    'Hardness': (47, 324),
                    'Solids': (320, 61227),
                    'Chloramines': (0, 13.1),
                    'Sulfate': (3, 481),
                    'Conductivity': (207, 564),
                    'Organic_carbon': (2, 65),
                    'Trihalomethanes': (1, 124),
                    'Turbidity': (1.4, 6.7)
                }
            
            def read_sensors(self):
                readings = {}
                for feature, (min_val, max_val) in self.feature_ranges.items():
                    readings[feature] = round(random.uniform(min_val, max_val), 2)
                readings['timestamp'] = datetime.now().isoformat()
                return readings
        
        # Initialize sensor
        sensor = IoTSensorSimulator()
        
        # Number of readings
        num_readings = st.slider("Number of Readings", 1, 20, 5)
        
        if st.button("🔄 Start IoT Monitoring", type="primary"):
            st.markdown("---")
            st.subheader(f"📊 Monitoring {num_readings} Sensor Readings")
            
            results_list = []
            
            def calculate_wqi(values):
                weights = np.array([0.11, 0.10, 0.10, 0.12, 0.11, 0.10, 0.12, 0.12, 0.12])
                ideal = np.array([7.0, 100, 500, 4.0, 250, 300, 10, 50, 2.5])
                norm = 1 - (np.abs(values - ideal) / (np.abs(ideal) + 1e-6))
                norm = np.clip(norm, 0, 1)
                wqi = np.sum(norm * weights) * 100
                return wqi

            for i in range(num_readings):
                reading = sensor.read_sensors()
                input_data = np.array([[
                    reading['ph'], reading['Hardness'], reading['Solids'],
                    reading['Chloramines'], reading['Sulfate'], reading['Conductivity'],
                    reading['Organic_carbon'], reading['Trihalomethanes'], reading['Turbidity']
                ]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                wqi = calculate_wqi(input_data[0])
                results_list.append({
                    'Reading': i + 1,
                    'pH': reading['ph'],
                    'Hardness': reading['Hardness'],
                    'Sulfate': reading['Sulfate'],
                    'Prediction': 'Potable ✅' if prediction == 1 else 'Not Potable ❌',
                    'Confidence': f"{max(probability)*100:.1f}%",
                    'WQI': f"{wqi:.1f}"
                })
            # Display results as dataframe
            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df, use_container_width=True)
            # Summary
            potable_count = sum(1 for r in results_list if 'Potable' in r['Prediction'])
            st.markdown("---")
            st.subheader("📋 Summary")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Readings", num_readings)
            with col_s2:
                st.metric("Potable", potable_count)
            with col_s3:
                st.metric("Not Potable", num_readings - potable_count)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Water Quality Classification System | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()