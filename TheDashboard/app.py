import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Interactive Data Dashboard")

# 📁 CSV Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())
else:
    # Generate sample data
    st.subheader("🧪 Sample Generated Data")
    df = pd.DataFrame({
        "Date": pd.date_range(start="2025-01-01", periods=30),
        "Value": np.random.normal(loc=100, scale=10, size=30)
    })

# 📊 Filter Slider in Sidebar
st.sidebar.header("Filters")
min_val = int(df["Value"].min())
max_val = int(df["Value"].max())
selected_range = st.sidebar.slider("Select Value Range", min_val, max_val, (min_val, max_val))

# Filter the dataframe
filtered_df = df[(df["Value"] >= selected_range[0]) & (df["Value"] <= selected_range[1])]

# 📈 Plotting
st.subheader("📉 Line Chart (Filtered Data)")
fig, ax = plt.subplots()
ax.plot(filtered_df["Date"], filtered_df["Value"], marker="o", linestyle="-")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.set_title("Filtered Time Series")
plt.xticks(rotation=45)
st.pyplot(fig)
