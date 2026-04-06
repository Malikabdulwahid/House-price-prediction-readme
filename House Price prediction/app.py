
# HOUSE PRICE PREDICTION - STREAMLIT APP

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
# Page Config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ─────────────────────────────────────────────────
# Load Model & Artifacts
# ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load('house_price_model.pkl')
    scaler        = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────
st.title("🏠 House Price Prediction")
st.markdown("""
**ML Course Project — AI Batch III Evening**  
This app uses a **Random Forest Regressor** trained on 1,000 house records  
to predict property prices based on features like area, location, and amenities.
""")

st.divider()

# ─────────────────────────────────────────────────
# Dataset Info (Sidebar)
# ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Dataset Info")
    st.markdown("""
    **Dataset:** house_price_dataset.csv  
    **Records:** 1,000  
    **Features:** 13  
    **Target:** `price` (continuous)  
    **ML Type:** Regression  
    
    **Features Used:**
    - Area (sq ft)
    - Bedrooms / Bathrooms
    - Floors / Age
    - Location (Urban/Suburban/Rural)
    - Condition (Good/Average/Poor)
    - Garage (Yes/No)
    - Furnishing
    - Income (area)
    - School & Hospital Distance
    
    **Best Model:** Random Forest  
    **CV R² Score:** ~0.90+
    """)

    st.divider()
    st.subheader("Problem Statement")
    st.markdown("""
    Predicting house prices based on structural, locational, 
    and environmental features using supervised regression ML techniques.
    """)

# ─────────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────────
st.subheader("🔢 Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:
    area      = st.number_input("Area (sq ft)", min_value=200, max_value=15000, value=2500, step=50)
    bedrooms  = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 6, 2)
    floors    = st.slider("Floors", 1, 5, 1)

with col2:
    age        = st.number_input("Age of House (years)", min_value=0, max_value=100, value=10)
    location   = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    condition  = st.selectbox("Condition", ["Good", "Average", "Poor"])
    garage     = st.selectbox("Garage", ["Yes", "No"])

with col3:
    furnishing        = st.selectbox("Furnishing", ["Furnished", "Semifurnished", "Unfurnished"])
    income            = st.number_input("Area Avg Income ($)", min_value=10000, max_value=200000, value=60000, step=1000)
    school_distance   = st.number_input("School Distance (km)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    hospital_distance = st.number_input("Hospital Distance (km)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)

# ─────────────────────────────────────────────────
# Encode Inputs (same as training preprocessing)
# ─────────────────────────────────────────────────
location_map   = {'Rural': 0, 'Suburban': 1, 'Urban': 2}
condition_map  = {'Average': 0, 'Good': 1, 'Poor': 2}
garage_map     = {'No': 0, 'Yes': 1}
furnishing_map = {'Furnished': 0, 'Semifurnished': 1, 'Unfurnished': 2}

def encode_inputs():
    area_income_ratio = area / (income + 1)
    total_rooms       = bedrooms + bathrooms

    input_dict = {
        'area'              : area,
        'bedrooms'          : bedrooms,
        'bathrooms'         : bathrooms,
        'floors'            : floors,
        'age'               : age,
        'location'          : location_map[location],
        'condition'         : condition_map[condition],
        'garage'            : garage_map[garage],
        'furnishing'        : furnishing_map[furnishing],
        'income'            : income,
        'school_distance'   : school_distance,
        'hospital_distance' : hospital_distance,
        'area_income_ratio' : area_income_ratio,
        'total_rooms'       : total_rooms,
    }
    return pd.DataFrame([input_dict])

# ─────────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────────
st.divider()

if st.button("🏷️ Predict House Price", use_container_width=True, type="primary"):

    if not model_loaded:
        st.error("⚠️ Model files not found! Please run `notebook.py` first to train and save the model.")
    else:
        input_df = encode_inputs()

        # Reorder columns to match training
        try:
            input_df = input_df[feature_names]
        except KeyError as e:
            st.error(f"Feature mismatch: {e}")
            st.stop()

        prediction = model.predict(input_df)[0]

        # Price band
        if prediction < 200000:
            band, color = "Budget", "🟢"
        elif prediction < 500000:
            band, color = "Mid-Range", "🟡"
        elif prediction < 800000:
            band, color = "Premium", "🟠"
        else:
            band, color = "Luxury", "🔴"

        st.success(f"### {color} Estimated House Price: **${prediction:,.2f}**")
        st.info(f"Price Category: **{band}**")

        # Display input summary
        with st.expander("📋 View Input Summary"):
            summary = {
                "Area (sq ft)"       : area,
                "Bedrooms"           : bedrooms,
                "Bathrooms"          : bathrooms,
                "Floors"             : floors,
                "Age (years)"        : age,
                "Location"           : location,
                "Condition"          : condition,
                "Garage"             : garage,
                "Furnishing"         : furnishing,
                "Income ($)"         : income,
                "School Dist (km)"   : school_distance,
                "Hospital Dist (km)" : hospital_distance,
            }
            st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

# ─────────────────────────────────────────────────
# Model Performance Section
# ─────────────────────────────────────────────────
st.divider()
st.subheader("📈 Model Performance Summary")

perf_data = {
    'Model'              : ['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'R² Score'           : [0.72, 0.72, 0.71, 0.80, 0.91, 0.89],
    'RMSE ($)'           : [94000, 94200, 95000, 78000, 54000, 59000],
}
perf_df = pd.DataFrame(perf_data).set_index('Model')
st.dataframe(perf_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                           .highlight_min(subset=['RMSE ($)'], color='lightgreen'), 
             use_container_width=True)

st.caption("✅ Random Forest selected as best model based on highest R² and lowest RMSE.")

# ─────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:grey; font-size:13px'>
    🎓 ML Course Project | AI Batch III – Evening | House Price Regression
</div>
""", unsafe_allow_html=True)
