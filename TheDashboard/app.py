import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Mg&DS Dashboard", layout="wide")
st.title("🧠 Website Activity Analysis")

# Load the datasets (replace with your raw GitHub URLs)
actions_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/cleaned_action.csv"
clicks_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/cleaned_click_session.csv"

actions_df = pd.read_csv(actions_url)
click_sessions_df = pd.read_csv(clicks_url)

# Cast yyyymmdd columns as timestamps (handle invalid entries gracefully)
def safe_to_datetime(series):
    return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')

actions_df['action_yyyymmdd'] = safe_to_datetime(actions_df['action_yyyymmdd'])
click_sessions_df['click_yyyymmdd'] = safe_to_datetime(click_sessions_df['click_yyyymmdd'])
click_sessions_df['session_yyyymmdd'] = safe_to_datetime(click_sessions_df['session_yyyymmdd'])

# Tabs
tabs = st.tabs(["🏠 Home", "📈 Overview", "🫱🏻‍🫲🏼Engagement", "📊 Classification", "🧠 Clustering"])

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

# 📈 Overview
with tabs[1]:
    st.header("📊 Usage & Interaction Analysis")

    # Q1: Distribution Over Time
    st.subheader("1. Distribution of Actions, Clicks, and Sessions Over Time")
    actions_by_date = actions_df.groupby('action_yyyymmdd').size().reset_index(name='nb_actions')
    clicks_by_date = click_sessions_df.groupby('click_yyyymmdd').size().reset_index(name='nb_clicks')
    sessions_by_date = click_sessions_df.drop_duplicates('click_session_id').groupby('session_yyyymmdd').size().reset_index(name='nb_sessions')

    fig_actions = px.line(actions_by_date, x='action_yyyymmdd', y='nb_actions', title="Actions Over Time", markers=True)
    fig_clicks = px.line(clicks_by_date, x='click_yyyymmdd', y='nb_clicks', title="Clicks Over Time", markers=True)
    fig_sessions = px.line(sessions_by_date, x='session_yyyymmdd', y='nb_sessions', title="Sessions Over Time", markers=True)

    st.plotly_chart(fig_actions, use_container_width=True)
    st.plotly_chart(fig_clicks, use_container_width=True)
    st.plotly_chart(fig_sessions, use_container_width=True)

    # Q2: Top 5 Users
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

    # Q3: Avg Interactions per User
    st.subheader("3. Average Number of Interactions per User")
    avg_actions_per_user = actions_df.groupby('action_visitor_id').size().mean()
    avg_clicks_per_user = click_sessions_df.groupby('click_visitor_id').size().mean()
    avg_sessions_per_user = click_sessions_df.drop_duplicates('click_session_id').groupby('click_visitor_id').size().mean()

    st.metric("Avg Actions/User", f"{avg_actions_per_user:.2f}")
    st.metric("Avg Clicks/User", f"{avg_clicks_per_user:.2f}")
    st.metric("Avg Sessions/User", f"{avg_sessions_per_user:.2f}")

    # Q4: Clicks per Session
    st.subheader("4. Clicks per Session")
    clicks_per_session = click_sessions_df.groupby('click_session_id').size()
    st.write(f"**Avg**: {clicks_per_session.mean():.2f} | **Max**: {clicks_per_session.max()} | **Min**: {clicks_per_session.min()}")

    # Q5: Actions per Session
    st.subheader("5. Actions per Session (from Actions File Only)")
    actions_per_session = actions_df.groupby('action_session_id').size()
    st.write(f"**Avg**: {actions_per_session.mean():.2f} | **Max**: {actions_per_session.max()} | **Min**: {actions_per_session.min()}")

    # Q6: Actions per Type
    if "action_name" in actions_df.columns:
        st.subheader("6. Actions per Type")
        type_counts = actions_df.groupby('action_name').size().reset_index(name='count')
        fig_type = px.bar(type_counts, x='action_name', y='count', title="Actions per Type")
        st.plotly_chart(fig_type, use_container_width=True)

        # Q7: Actions per Type per Session
        st.subheader("7. Actions per Type per Session")
        type_session = actions_df.groupby(['action_session_id', 'action_name']).size().reset_index(name='count')
        stats = type_session.groupby('action_name')['count'].agg(['mean', 'max', 'min']).reset_index()
        st.dataframe(stats)
    else:
        st.warning("⚠️ 'action_name' column not found in actions dataset.")

# 🫱🏻‍🫲🏼Engagement
with tabs[2]:
    st.header("🫱🏻‍🫲🏼Engagement Analysis")
    
    # Q1. SCORE D'ENGAGEMENT = Prend en compte les action(calssées par importance) et le parmaétre de régularité de l'utilisateur
    st.subheader("1. Engagement Score: Takes into account the user's actions (ranked by importance) and regularity parameter")
    
    # 🧽 Préparation
    actions_df["action_timestamp"] = pd.to_datetime(actions_df["action_timestamp"], unit="s", errors="coerce")

    # Pondération des groupes d’actions
    poids_action_group = {
        'publish': 5,
        'animate': 4,
        'participate': 3,
        'reaction': 2,
        'user': 1
    }
    actions_df["action_group_weight"] = actions_df["action_group"].map(poids_action_group)
    
    # Supprimer les lignes sans poids
    df_clean = actions_df.dropna(subset=["action_group_weight"])
    
    # Calcul du score brut
    user_scores = df_clean.groupby("action_visitor_id").agg(
        score_brut=("action_group_weight", "sum"),
        frequence=("action_group_weight", "count"),
        is_repeat_visitor=("action_is_repeat_visitor", "max")
    ).reset_index()
    
    # Score pondéré par fréquence (logarithmique)
    user_scores["score_actions"] = user_scores["score_brut"] * np.log1p(user_scores["frequence"])
    
    # Normalisation des actions
    score_max = user_scores["score_actions"].max()
    user_scores["score_action_normalise"] = user_scores["score_actions"] / score_max
    
    # ⚖️ Nouveau calcul du score : 80 % actions, 20 % fidélité
    user_scores["score_engagement"] = (
        0.8 * user_scores["score_action_normalise"] +
        0.2 * user_scores["is_repeat_visitor"]
    ) * 100
    
    # Résultat final
    result = user_scores[["action_visitor_id", "score_engagement"]].sort_values(by="score_engagement", ascending=False).round(2)

    # 🔍 Search box
    visitor_filter = st.text_input("Search for a specific Visitor ID")
    if visitor_filter:
        filtered_result = result[result["action_visitor_id"].astype(str).str.contains(visitor_filter)]
        st.dataframe(filtered_result)
    else:
        st.dataframe(result.head(10))

    #Q2. Analysis of Rebound
    st.subheader("2. Analysis of Rebound")
    total_sessions = click_sessions_df["click_session_id"].nunique()
    total_bounces = click_sessions_df[click_sessions_df["session_is_bounce"] == 1]["click_session_id"].nunique()
    taux_de_rebond = (total_bounces / total_sessions) * 100
    st.metric("Taux de rebond", f"{taux_de_rebond:.2f} %")

    sessions_user = click_session_df.groupby(["session_visitor_id", "session_id"]).agg(
    is_bounce=("session_is_bounce", "max")  # 1 si c'est un rebond, sinon 0
    ).reset_index()

    # Regrouper par utilisateur pour compter les sessions et rebonds
    bounce_stats = sessions_user.groupby("session_visitor_id").agg(
        total_sessions=("session_id", "count"),
        total_bounces=("is_bounce", "sum")
    ).reset_index()
    
    # Calcul du taux de rebond par utilisateur
    bounce_stats["taux_rebond_utilisateur"] = (bounce_stats["total_bounces"] / bounce_stats["total_sessions"]) * 100
    
    # Afficher les 10 premiers résultats triés
    final_result = bounce_stats.sort_values(by="taux_rebond_utilisateur", ascending=False).head(10)
    # 🔍 Search box
    visitor_filter1 = st.text_input("Search for a specific Visitor ID")
    if visitor_filter1:
        filtered_result = result[result["session_visitor_id"].astype(str).str.contains(visitor_filter1)]
        st.dataframe(filtered_result)
    else:
        st.dataframe(final_result.head(10))

    #Q3. Analysis of return of users
    st.subheader("3. Analysis of Returns")
    nb_total_utilisateurs = click_sessions_df["click_visitor_id"].nunique()
    sessions_par_utilisateur = click_sessions_df.groupby("click_visitor_id")["click_session_id"].nunique().reset_index()
    utilisateurs_revenus = sessions_par_utilisateur[sessions_par_utilisateur["click_session_id"] > 1].shape[0]
    taux_de_retour = (utilisateurs_revenus / nb_total_utilisateurs) * 100
    st.metric("Taux de rebond", f"{taux_de_retour:.2f} %")
    
# 📊 Classification
with tabs[3]:
    st.header("📊 Classification")
    st.info("This section will display classification models and performance metrics.")

# 🧠 Clustering
with tabs[4]:
    st.header("🧠 Clustering")
    st.info("This section will display final results, interpretations, and export options.")
