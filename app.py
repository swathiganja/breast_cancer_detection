import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------
# Load model and scaler (cached)
# ---------------------------------------

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("1234ES_Drop_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------------
# UI
# ---------------------------------------

st.title("Breast Cancer Prediction System")
st.write("Enter tumor measurements to predict cancer type")

def num_input(label, value):
    return st.number_input(label, value=float(value))

# ---------------------------------------
# Inputs (30 features)
# ---------------------------------------

inputs = {
    'radius_mean': num_input("Radius Mean", 14.0),
    'texture_mean': num_input("Texture Mean", 20.0),
    'perimeter_mean': num_input("Perimeter Mean", 90.0),
    'area_mean': num_input("Area Mean", 600.0),
    'smoothness_mean': num_input("Smoothness Mean", 0.1),
    'compactness_mean': num_input("Compactness Mean", 0.15),
    'concavity_mean': num_input("Concavity Mean", 0.2),
    'concave_points_mean': num_input("Concave Points Mean", 0.1),
    'symmetry_mean': num_input("Symmetry Mean", 0.2),
    'fractal_dimension_mean': num_input("Fractal Dimension Mean", 0.06),

    'radius_se': num_input("Radius SE", 0.2),
    'texture_se': num_input("Texture SE", 1.0),
    'perimeter_se': num_input("Perimeter SE", 1.5),
    'area_se': num_input("Area SE", 20.0),
    'smoothness_se': num_input("Smoothness SE", 0.005),
    'compactness_se': num_input("Compactness SE", 0.02),
    'concavity_se': num_input("Concavity SE", 0.03),
    'concave_points_se': num_input("Concave Points SE", 0.01),
    'symmetry_se': num_input("Symmetry SE", 0.03),
    'fractal_dimension_se': num_input("Fractal Dimension SE", 0.004),

    'radius_worst': num_input("Radius Worst", 16.0),
    'texture_worst': num_input("Texture Worst", 25.0),
    'perimeter_worst': num_input("Perimeter Worst", 105.0),
    'area_worst': num_input("Area Worst", 800.0),
    'smoothness_worst': num_input("Smoothness Worst", 0.12),
    'compactness_worst': num_input("Compactness Worst", 0.2),
    'concavity_worst': num_input("Concavity Worst", 0.3),
    'concave_points_worst': num_input("Concave Points Worst", 0.15),
    'symmetry_worst': num_input("Symmetry Worst", 0.25),
    'fractal_dimension_worst': num_input("Fractal Dimension Worst", 0.08)
}

# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame([inputs])

    # Apply trained scaler
    input_scaled = scaler.transform(input_df)

    probability = model.predict(input_scaled)[0][0]

    if probability > 0.5:
        st.error(f"Malignant (Confidence: {probability:.2%})")
    else:
        st.success(f"Benign (Confidence: {(1 - probability):.2%})")
