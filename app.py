import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Overdose Prediction App", page_icon="⚕️", layout="centered")

st.title("💊 Drug Overdose Prediction System")
st.write("Predict overdose deaths using trained ML models.")
st.divider()

# Load shared objects
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_cols = pickle.load(open("feature_cols.pkl", "rb"))
train_means = pickle.load(open("train_means.pkl", "rb"))

if isinstance(train_means, dict):
    train_means = pd.Series(train_means)

# Load 3 models
models = {
    "Linear Regression": pickle.load(open("lr_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("dt_model.pkl", "rb")),
    "Random Forest": pickle.load(open("rf_model.pkl", "rb")),
}

# Hide this from UI if present, but keep internally for scaler
HIDDEN_COLS = {"Predicted Value"}
ui_cols = [c for c in feature_cols if c not in HIDDEN_COLS]

st.subheader("Choose Model")
model_name = st.selectbox("Select a model", list(models.keys()))
model = models[model_name]

st.subheader("Enter Input Data")
user_input = {}

for col in ui_cols:
    default_val = float(train_means[col]) if col in train_means.index else 0.0
    if str(col).lower() == "year":
        user_input[col] = st.number_input(label=str(col), value=int(default_val) if default_val else 2024, step=1)
    else:
        user_input[col] = st.number_input(label=str(col), value=float(default_val))

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    # Add hidden cols back so scaler gets exact training columns
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = float(train_means[col]) if col in train_means.index else 0.0

    # Ensure exact order
    input_df = input_df[feature_cols]

    # Numeric + fill missing
    input_df = input_df.apply(pd.to_numeric, errors="coerce")
    for c in input_df.columns:
        input_df[c] = input_df[c].fillna(train_means[c]) if c in train_means.index else input_df[c].fillna(0)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.divider()
    st.success(f"Model: {model_name}")
    st.success(f"Predicted Overdose Deaths: {prediction:.2f}")
    st.caption("Note: one internal feature was required to match the trained scaler input format.")
