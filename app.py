from dis import code_info

import streamlit as st
import pandas as pd
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Cancer Prediction Dashboard", layout="wide")

st.title("Breast Cancer Prediction Interface")
st.write("Enter the tumor features below to predict whether the diagnosis is Malignant or Benign.")

# 1. Load the trained model safely
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    st.error("Error: 'model.pkl' not found! Please run main.py first to train and save the model.")
    st.stop()

# 2. Extract feature names that the model expects
# Excluding 'id' and 'Unnamed: 32' as your data cleaning script does
try:
    data_sample = pd.read_csv("Data/data.csv")
    data_sample = data_sample.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1, errors='ignore')
    feature_names = data_sample.columns.tolist()
except Exception as e:
    st.error(f"Could not load feature columns from dataset: {e}")
    st.stop()

# 3. Create user inputs layout dynamically
st.header("Input Tumor Measurements")
user_inputs = {}

# Distribute the 30 features into 3 columns for a clean UI layout
cols = st.columns(3)

for idx, feature in enumerate(feature_names):
    col = cols[idx % 3] # Rotate through the columns
    # Set default value as the average from your dataset to make testing easy
    default_val = float(data_sample[feature].mean())
    
    user_inputs[feature] = col.number_input(
        label=feature.replace('_', ' ').title(), 
        value=default_val,
        format="%.4f"
    )

# Convert user inputs into a DataFrame row matching model expectations
input_df = pd.DataFrame([user_inputs])

# 4. Make Predictions
st.markdown("---")
if st.button("Predict Diagnosis", type="primary"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Prediction Results")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **Malignant** (Probability: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ **Benign** (Probability: {prediction_proba[0][0]*100:.2f}%)")