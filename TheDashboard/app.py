import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mg&DS Dashboard", layout="wide")
st.title("🧠 Website Activity Analysis")

# Tabs
tabs = st.tabs(["🏠 Home", "📈 Overview", "📊 Classification", "🧠 Clustering"])

# 🏠 Home
with tabs[0]:
    st.header("Welcome to the Mg&DS Analysis Dashboard")
    st.markdown("""
        This interactive dashboard helps you explore data regarding the activity in the Website Management & Data Science.
        The analyses are separated between:
        📈 Overview -> That gives you a global view about the activity, prioritizing the visualisation of KPIs and distributions
        📊 Classification -> 
        🧠 Clustering ->

        Use the tabs above to navigate through the analysis.
    """)

# 📈 Overview (still empty)
with tabs[1]:
    st.header("📈 Overview")
    st.info("This section will show data summary and distributions. (Coming soon)")

import streamlit as st
import pandas as pd
import plotly.express as px

# Load the datasets (replace with your raw GitHub URLs)
actions_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/cleaned_action.csv?token=GHSAT0AAAAAADCCBYDF3OUGCSA6ZM57AYZCZ72U46A"
clicks_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/cleaned_click_session.csv?token=GHSAT0AAAAAADCCBYDFHHIJPXWA3232PW2SZ72U5DQ"

actions_df = pd.read_csv(actions_url)
click_sessions_df = pd.read_csv(clicks_url)

# Cast yyyymmdd columns as timestamps (handle invalid entries gracefully)
def safe_to_datetime(series):
    return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')

actions_df['action_yyyymmdd'] = safe_to_datetime(actions_df['action_yyyymmdd'])
click_sessions_df['click_yyyymmdd'] = safe_to_datetime(click_sessions_df['click_yyyymmdd'])
click_sessions_df['session_yyyymmdd'] = safe_to_datetime(click_sessions_df['session_yyyymmdd'])

st.header("📊 Usage & Interaction Analysis")

# -------------------- Q1: Distribution Over Time ---------------------
st.subheader("1. Distribution of Actions, Clicks, and Sessions Over Time")

# Prepare time-based data
actions_by_date = actions_df.groupby('action_yyyymmdd').size().reset_index(name='nb_actions')
clicks_by_date = click_sessions_df.groupby('click_yyyymmdd').size().reset_index(name='nb_clicks')
sessions_by_date = click_sessions_df.drop_duplicates('click_session_id').groupby('session_yyyymmdd').size().reset_index(name='nb_sessions')

fig_actions = px.line(actions_by_date, x='action_yyyymmdd', y='nb_actions', title="Actions Over Time", markers=True)
fig_clicks = px.line(clicks_by_date, x='click_yyyymmdd', y='nb_clicks', title="Clicks Over Time", markers=True)
fig_sessions = px.line(sessions_by_date, x='session_yyyymmdd', y='nb_sessions', title="Sessions Over Time", markers=True)

st.plotly_chart(fig_actions, use_container_width=True)
st.plotly_chart(fig_clicks, use_container_width=True)
st.plotly_chart(fig_sessions, use_container_width=True)

# -------------------- Q2: Top 5 Users ---------------------
st.subheader("2. Top 5 Users by Actions, Clicks, and Sessions")

top_actions = actions_df['action_visitor_id'].value_counts().head(5).reset_index(name='nb_actions')
top_clicks = click_sessions_df['click_visitor_id'].value_counts().head(5).reset_index(name='nb_clicks')
top_sessions = click_sessions_df.drop_duplicates('click_session_id')['click_visitor_id'].value_counts().head(5).reset_index(name='nb_sessions')

col1, col2, col3 = st.columns(3)
with col1:
    st.write("🔝 Actions")
    st.dataframe(top_actions)
with col2:
    st.write("🔝 Clicks")
    st.dataframe(top_clicks)
with col3:
    st.write("🔝 Sessions")
    st.dataframe(top_sessions)

# -------------------- Q3: Avg Interactions per User ---------------------
st.subheader("3. Average Number of Interactions per User")

avg_actions_per_user = actions_df.groupby('action_visitor_id').size().mean()
avg_clicks_per_user = click_sessions_df.groupby('click_visitor_id').size().mean()
avg_sessions_per_user = click_sessions_df.drop_duplicates('click_session_id').groupby('click_visitor_id').size().mean()

st.metric("Avg Actions/User", f"{avg_actions_per_user:.2f}")
st.metric("Avg Clicks/User", f"{avg_clicks_per_user:.2f}")
st.metric("Avg Sessions/User", f"{avg_sessions_per_user:.2f}")

# -------------------- Q4: Clicks per Session ---------------------
st.subheader("4. Clicks per Session")
clicks_per_session = click_sessions_df.groupby('click_session_id').size()
st.write(f"**Avg**: {clicks_per_session.mean():.2f} | **Max**: {clicks_per_session.max()} | **Min**: {clicks_per_session.min()}")

# -------------------- Q5: Actions per Session ---------------------
st.subheader("5. Actions per Session (from Actions File Only)")
actions_per_session = actions_df.groupby('action_session_id').size()
st.write(f"**Avg**: {actions_per_session.mean():.2f} | **Max**: {actions_per_session.max()} | **Min**: {actions_per_session.min()}")

# -------------------- Q6: Actions per Type ---------------------
if "action_name" in actions_df.columns:
    st.subheader("6. Actions per Type")
    type_counts = actions_df.groupby('action_name').size().reset_index(name='count')
    fig_type = px.bar(type_counts, x='action_name', y='count', title="Actions per Type")
    st.plotly_chart(fig_type, use_container_width=True)

    # ---------------- Q7: Actions per Type per Session ----------------
    st.subheader("7. Actions per Type per Session")
    type_session = actions_df.groupby(['action_session_id', 'action_name']).size().reset_index(name='count')
    stats = type_session.groupby('action_name')['count'].agg(['mean', 'max', 'min']).reset_index()
    st.dataframe(stats)
else:
    st.warning("⚠️ 'action_name' column not found in actions dataset.")


# 🧠 Clustering
with tabs[2]:
    st.header("🧠 Clustering")
    st.info("This section will display final results, interpretations, and export options.")

# 📊 Results (still empty)
with tabs[3]:
    st.header("🧠 Clustering")
    st.info("This section will display final results, interpretations, and export options.")
