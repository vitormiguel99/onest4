import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page title
st.title("📈 Sample Line Chart")

# Create sample data
st.subheader("Generated Data")
days = pd.date_range(start="2025-01-01", periods=30)
values = np.random.normal(loc=50, scale=10, size=30)

df = pd.DataFrame({
    "Date": days,
    "Value": values
})

# Show the dataframe
st.dataframe(df)

# Plotting the graph
st.subheader("Line Chart")

fig, ax = plt.subplots()
ax.plot(df["Date"], df["Value"], marker='o', linestyle='-')
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.set_title("Randomly Generated Time Series")
plt.xticks(rotation=45)

# Display the chart
st.pyplot(fig)
