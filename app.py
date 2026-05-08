import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

st.title("💰 Loan Default Prediction (Loss Prediction)")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data")
    st.dataframe(data.head())

    # Drop id if exists
    if "id" in data.columns:
        data = data.drop("id", axis=1)

    # Align columns
    data = data.reindex(columns=columns, fill_value=0)

    # Clean data
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)
    data = data.clip(-1e6, 1e6)

    # Prediction
    predictions = model.predict(data)

    data["Predicted_Loss"] = predictions

    st.write("### 📊 Predictions")
    st.dataframe(data.head())

    # 📈 Graph 1: Histogram
    st.write("### 📈 Predicted Loss Distribution")
    fig, ax = plt.subplots()
    ax.hist(predictions, bins=30)
    ax.set_title("Loss Distribution")
    st.pyplot(fig)

    # 📊 Graph 2: Feature importance (top 10)
    st.write("### 🔥 Feature Importance (Top 10)")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]

    fig2, ax2 = plt.subplots()
    ax2.barh(range(len(indices)), importances[indices])
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels([columns[i] for i in indices])
    ax2.set_title("Top Features")
    st.pyplot(fig2)