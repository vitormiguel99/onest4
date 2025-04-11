import streamlit as st
import pandas as pd
from ml import model  # your model.py
from utils.helpers import preprocess_data  # optional

# Title
st.title("Machine Learning Dashboard")

# Upload CSVs or read from folder
data_file = st.file_uploader("Upload your CSV", type=["csv"])
if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("📊 Data Preview", df.head())

    # Optional preprocessing
    df_clean = preprocess_data(df)

    # ML Predictions
    if st.button("Run Prediction"):
        prediction = model.predict(df_clean)
        st.write("🧠 Predictions", prediction)
