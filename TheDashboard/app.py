import streamlit as st
import pandas as pd
from ml import model 
from utils.helpers import preprocess_data  


st.title("Machine Learning Dashboard")


data_file = st.file_uploader("Upload your CSV", type=["csv"])
if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("📊 Data Preview", df.head())

    
    df_clean = preprocess_data(df)

    
    if st.button("Run Prediction"):
        prediction = model.predict(df_clean)
        st.write("🧠 Predictions", prediction)
