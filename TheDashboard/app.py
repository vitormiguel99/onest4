import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Dashboard with Multiple Datasets")

# 📁 Load both datasets (replace with GitHub raw URLs if needed)
action_url = "https://github.com/vitormiguel99/onest4/blob/main/TheDashboard/action.csv"
session_url = "https://github.com/vitormiguel99/onest4/blob/main/TheDashboard/session.csv"

# Load data
try:
    action_df = pd.read_csv(action_url)
    session_df = pd.read_csv(session_url)
    st.success("✅ Both datasets loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# 🔍 Show previews
st.subheader("📄 Action Data")
st.dataframe(action_df.head())

st.subheader("📄 Session Data")
st.dataframe(session_df.head())

# 🎯 Assume both have a 'date' column and a numeric metric column
# You can adjust this based on your real column names

# Line chart for Action data
st.subheader("📈 Action Chart")
try:
    fig1, ax1 = plt.subplots()
    action_df['date'] = pd.to_datetime(action_df['date'])  # Adjust column name if needed
    ax1.plot(action_df['date'], action_df.iloc[:, 1], marker='o')  # assumes value is in 2nd column
    ax1.set_title("Action Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(action_df.columns[1])
    plt.xticks(rotation=45)
    st.pyplot(fig1)
except Exception as e:
    st.warning(f"Could not plot action.csv: {e}")

# Line chart for Session data
st.subheader("📈 Session Chart")
try:
    fig2, ax2 = plt.subplots()
    session_df['date'] = pd.to_datetime(session_df['date'])  # Adjust column name if needed
    ax2.plot(session_df['date'], session_df.iloc[:, 1], marker='o')
    ax2.set_title("Session Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel(session_df.columns[1])
    plt.xticks(rotation=45)
    st.pyplot(fig2)
except Exception as e:
    st.warning(f"Could not plot session.csv: {e}")
