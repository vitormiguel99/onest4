import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Mg&DS Dashboard", layout="wide")
st.title("🧠 Website Activity Analysis")

# Load the datasets (replace with your raw GitHub URLs)
actions_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/cleaned_action.csv"
clicks_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/cleaned_click_session.csv"
actions_avec_score_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/cleaned_action_avec_score.csv"

actions_df = pd.read_csv(actions_url)
click_sessions_df = pd.read_csv(clicks_url)
actions_avec_score_df = pd.read_csv(actions_avec_score_url)

# Cast yyyymmdd columns as timestamps (handle invalid entries gracefully)
def safe_to_datetime(series):
    return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')

actions_df['action_yyyymmdd'] = safe_to_datetime(actions_df['action_yyyymmdd'])
click_sessions_df['click_yyyymmdd'] = safe_to_datetime(click_sessions_df['click_yyyymmdd'])
click_sessions_df['session_yyyymmdd'] = safe_to_datetime(click_sessions_df['session_yyyymmdd'])

# Tabs
tabs = st.tabs(["🏠 Home", "📈 Overview", "🫱🏻‍🫲🏼Engagement", "📊 Classification", "🧠 Seesions + Clicks' Clustering", "🧠 Actions' Clustering","💡Insights and Suggestions"])

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
    result_engage = user_scores[["action_visitor_id", "score_engagement"]].sort_values(by="score_engagement", ascending=False).round(2)

    # 🔍 Search box
    visitor_filter_engage = st.text_input("Search for a specific Visitor ID to see their enagagement")
    if visitor_filter_engage:
        filtered_result_engage = result_engage[result_engage["action_visitor_id"].astype(str).str.contains(visitor_filter_engage)]
        st.dataframe(filtered_result_engage)
    else:
        st.dataframe(result_engage.head(10))

    #Q2. Analysis of Rebound
    st.subheader("2. Analysis of Rebound")
    total_sessions = click_sessions_df["click_session_id"].nunique()
    total_bounces = click_sessions_df[click_sessions_df["session_is_bounce"] == 1]["click_session_id"].nunique()
    taux_de_rebond = (total_bounces / total_sessions) * 100
    st.metric("Rebond rate", f"{taux_de_rebond:.2f} %")

    sessions_user = click_sessions_df.groupby(["session_visitor_id", "session_id"]).agg(
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
    result_bounce = bounce_stats.sort_values(by="taux_rebond_utilisateur", ascending=False).head(10)
    # 🔍 Search box
    visitor_filter_bounce = st.text_input("Search for a specific Visitor ID to see their bounce rate")
    if visitor_filter_bounce:
        filtered_result_bounce = result_bounce[result_bounce["session_visitor_id"].astype(str).str.contains(visitor_filter_bounce)]
        st.dataframe(filtered_result_bounce)
    else:
        st.dataframe(result_bounce.head(10))

    #Q3. Analysis of return of users
    st.subheader("3. Analysis of Returns")
    nb_total_utilisateurs = click_sessions_df["click_visitor_id"].nunique()
    sessions_par_utilisateur = click_sessions_df.groupby("click_visitor_id")["click_session_id"].nunique().reset_index()
    utilisateurs_revenus = sessions_par_utilisateur[sessions_par_utilisateur["click_session_id"] > 1].shape[0]
    taux_de_retour = (utilisateurs_revenus / nb_total_utilisateurs) * 100
    st.metric("Return Rate", f"{taux_de_retour:.2f} %")

    # Compter le nombre de sessions par utilisateur
    result_return = click_sessions_df.groupby("session_visitor_id")["session_id"].nunique().reset_index()
    result_return.columns = ["session_visitor_id", "nb_sessions"]

    # Taux de retour binaire : 1 s'il est revenu, 0 sinon
    result_return["retour_binaire"] = result_return["nb_sessions"].apply(
    lambda x: 1 if x > 1 else 0)

    # Affichage
    visitor_filter_return = st.text_input("Search for a specific Visitor ID to see their return rate")
    if visitor_filter_return:
        filtered_result_return = result_return[result_return["session_visitor_id"].astype(str).str.contains(visitor_filter_return)]
        st.dataframe(filtered_result_return)
    else:
        st.dataframe(result_return.head(10))

    #Q4. Analysis of views per session
    st.subheader("4. Analysis of Views per Session")
    # Extraire les colonnes utiles
    pages_per_session = click_sessions_df[["session_id", "session_num_pageviews"]].drop_duplicates()
    moyenne_pages = pages_per_session["session_num_pageviews"].mean()
    st.write(f"Average of pages view per session : {moyenne_pages:.2f}")

    session_filter = st.text_input("Search for a specific Session ID to see its Nb of Pages Visited")
    if session_filter:
        filtered_session = pages_per_session[pages_per_session["session_visitor_id"].astype(str).str.contains(session_filter)]
        st.dataframe(session_filter)
    else:
        st.dataframe(pages_per_session.head(10))

    
    
# 📊 Classification
with tabs[3]:
    st.header("📊 Classification")
    st.info("This section will display classification models and performance metrics.")

    median_threshold = df['score_engagement'].median()
    df['engaged'] = (df['score_engagement'] > median_threshold).astype(int)

    user_df = df.groupby('action_user_name').agg(
        nb_actions=('action_id', 'count'),
        nb_sessions=('action_session_id', pd.Series.nunique),
        nb_jours_actifs=('action_yyyymmdd', pd.Series.nunique),
        engagement_moyen=('score_engagement', 'mean'),
        engagement_max=('score_engagement', 'max'),
        engagement_min=('score_engagement', 'min'),
        part_nouvelles_visites=('action_is_new_visitor', 'mean'),
        part_visites_recurrentes=('action_is_repeat_visitor', 'mean'),
        poids_moyen_actions=('action_group_weight', 'mean'),
        nb_groupes_uniques=('action_group', pd.Series.nunique),
        nb_labels_uniques=('action_label', pd.Series.nunique),
        nb_actions_par_jour=('action_id', lambda x: x.count() / df.loc[x.index, 'action_yyyymmdd'].nunique())
    ).reset_index()

    user_df['engaged'] = user_df['action_user_name'].map(
        df.groupby('action_user_name')['engaged'].agg(lambda x: int(x.mean() > 0.5))
    )

    st.write("✅ Aggregated KPIs by user:")
    st.dataframe(user_df.head())

    X = user_df.drop(columns=['action_user_name', 'engaged'])
    y = user_df['engaged']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    over_sampler = SMOTE(random_state=42)
    X_train_res, y_train_res = over_sampler.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Outputs
    st.write("📊 **Confusion Matrix**:")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

    st.write("📋 **Classification Report:**")
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())


# 🧠 Clustering
with tabs[4]:
        st.header("🧠 Seesions + Clicks' Clustering")
        st.markdown("""
        This section applies unsupervised learning techniques to segment users based on their behavior.
        The analysis is based on click/session data and includes log transformation, normalization, and clustering.
        """)
    
        # Step 1: KPI table per user
        st.subheader("Step 1: Summary of User Behavior (KPIs)")
        # Création des KPI navigation par utilisateur
        df_navigation_users = click_sessions_df.groupby('click_visitor_id').agg({
            'click_id': 'count',                                 # nb_clicks
            'session_id': pd.Series.nunique,                     # nb_sessions
            'session_num_pageviews': 'sum',                      # nb_pages_vues
            'session_is_bounce': 'sum',                          # nb_sessions_bounce
            'session_num_comments': lambda x: (x > 0).sum(),     # nb_sessions_commentees
            'session_time_sinse_priorsession': 'mean',           # delai_moyen_entre_sessions
            'click_yyyymmdd': pd.Series.nunique                  # nb_jours_actifs
        }).reset_index()
        
        # Renommage des colonnes
        df_navigation_users.columns = [
            'visitor_id',
            'nb_clicks',
            'nb_sessions',
            'nb_pages_vues',
            'nb_sessions_bounce',
            'nb_sessions_commentees',
            'delai_moyen_entre_sessions',
            'nb_jours_actifs'
        ]
        
        # Conversion du délai moyen entre sessions de secondes en jours
        df_navigation_users['delai_moyen_entre_sessions'] = df_navigation_users['delai_moyen_entre_sessions'] / (60 * 60 * 24)
        
        # Aperçu du résultat
        df_navigation_users.head()
        
        # 🔍 Search box
        visitor_input_kpi = st.text_input("Search for a specific Visitor ID to check their KPIs")
    
        # Filter and display
        if visitor_input_kpi:
            filtered_visitor_kpi = df_navigation_users[df_navigation_users['visitor_id'].astype(str).str.contains(visitor_input_kpi)]
            st.dataframe(filtered_visitor_kpi)
        else:
            st.dataframe(df_navigation_users.head(10))
    
        # Step 2: Preprocessing
        cols_to_transform = [
            'nb_clicks', 'nb_sessions', 'nb_pages_vues',
            'nb_sessions_bounce', 'nb_sessions_commentees',
            'delai_moyen_entre_sessions', 'nb_jours_actifs'
        ]
    
        df_kpi = df_navigation_users.copy()
        for col in cols_to_transform:
            df_kpi[f'{col}_log'] = np.log1p(df_kpi[col])
    
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_kpi[[f'{col}_log' for col in cols_to_transform]])
    
        df_scaled = pd.DataFrame(X_scaled, columns=[f'{col}_scaled' for col in cols_to_transform])
        df_scaled['visitor_id'] = df_kpi['visitor_id'].values
        
        # Step 3: Distribution
        st.subheader("Step 2: Feature Distributions After Log Transformation")
        selected_col = st.selectbox("Select a KPI to view its log distribution", options=[f'{col}_log' for col in cols_to_transform])
        fig = plt.figure(figsize=(6, 4))
        sns.histplot(df_kpi[selected_col], kde=True, bins=30)
        plt.title(f"Distribution log of {selected_col.replace('_log', '')}")
        plt.xlabel(selected_col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    
        # Step 4: Boxplots of all features
        st.subheader("Step 3: Boxplot of Log-Transformed KPIs")
        fig_box = plt.figure(figsize=(10, 5))
        sns.boxplot(data=df_kpi[[f'{col}_log' for col in cols_to_transform]])
        plt.xticks(rotation=45)
        plt.title("Boxplot of log-transformed navigation KPIs")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig_box)
    
        # Fit final model
        X = df_scaled[[col for col in df_scaled.columns if col.endswith('_scaled')]]
        kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
        df_kpi['cluster'] = kmeans.fit_predict(X)
    
    
        # Step 5: KPI Means per Cluster
        st.subheader("Step 4: KPI Averages by Cluster")
        kpi_orig_cols = [
            'nb_clicks', 'nb_sessions', 'nb_pages_vues',
            'nb_sessions_bounce', 'nb_sessions_commentees',
            'delai_moyen_entre_sessions', 'nb_jours_actifs'
        ]
        df_clusters_summary = df_kpi.groupby('cluster')[kpi_orig_cols].mean().round(2)
        st.dataframe(df_clusters_summary.reset_index())
        
        # Step 6: Cluster Assignments
        st.subheader("Step 5: Cluster Assignments and Search")
        visitor_search = st.text_input("Search for a specific Visitor ID to get their cluster assignment")
        if visitor_search:
            filtered_cluster_df = df_kpi[df_kpi['visitor_id'].astype(str).str.contains(visitor_search)]
            st.dataframe(filtered_cluster_df)
        else:
            st.dataframe(df_kpi[['visitor_id', 'cluster']].head(10))
    
with tabs[5]:
        #Analysis of actions
        st.header("🧠 Actions' Clustering")
        # 1. Users' KPIs
        st.subheader("Step 1: Actions' KPIs per Visitor")
        df_contribution_users = actions_df.groupby('action_visitor_id').agg(
        nb_actions_total=('action_id', 'count'),
        nb_types_actions_uniques=('action_name', pd.Series.nunique),
        nb_groupes_actions=('action_group', pd.Series.nunique),
        nb_jours_actifs=('action_yyyymmdd', pd.Series.nunique),
        taux_repeat_visitor=('action_is_repeat_visitor', lambda x: x.sum() / len(x) if len(x) > 0 else 0),
        nb_mediums_utilisés=('action_medium', pd.Series.nunique),
        nb_sites_utilisés=('action_site_id', pd.Series.nunique),
        nb_contributions=('action_name', lambda x: ((x == 'frontend create') | (x == 'editor publish')).sum()),
        nb_modifications=('action_name', lambda x: (x == 'frontend modify').sum()),
        nb_publications=('action_group', lambda x: (x == 'publish').sum())
        ).reset_index()
        
        # Create dummy variables for medium usage
        medium_dummies = pd.get_dummies(actions_df[['action_visitor_id', 'action_medium']],
                                        columns=['action_medium'],
                                        prefix='medium',
                                        dtype=int)
        
        # Aggregate to get 1 if a medium is used at least once
        medium_usage = medium_dummies.groupby('action_visitor_id').max().reset_index()
        
        # Merge KPIs with dummy data
        df_contribution_users = pd.merge(df_contribution_users, medium_usage, on='action_visitor_id', how='left')
        
        # Interactive search bar
        search_id = st.text_input("🔍 Search for a specific Visitor ID to get their Actions' KPIs")
        
        if search_id:
            filtered = df_contribution_users[df_contribution_users['action_visitor_id'].astype(str).str.contains(search_id)]
            st.dataframe(filtered)
        else:
            st.dataframe(df_contribution_users.head(10))
    
        
        # Step 2: Contribution KPI Distributions
        st.subheader("Step 2: Contribution KPI Distributions")
        kpi_cols = [
            'nb_actions_total',
            'nb_types_actions_uniques',
            'nb_groupes_actions',
            'nb_jours_actifs',
            'taux_repeat_visitor',
            'nb_mediums_utilisés',
            'nb_sites_utilisés',
            'nb_contributions',
            'nb_modifications',
            'nb_publications'
        ]
    
        selected_kpi = st.selectbox("Select a Contribution KPI to view its distribution", options=kpi_cols)
        fig_kpi_dist = plt.figure(figsize=(6, 4))
        sns.histplot(df_contribution_users[selected_kpi], kde=True, bins=30)
        plt.title(f"Distribution of {selected_kpi}")
        plt.xlabel(selected_kpi)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig_kpi_dist)
    
        # Global boxplot for KPI contribution
        st.subheader("Global Boxplot of Contribution KPIs")
        fig_kpi_box = plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_contribution_users[kpi_cols])
        plt.xticks(rotation=45)
        plt.title("Boxplot of Contribution KPIs")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig_kpi_box)
    
        # Step 3 Clusters based on Actions
        st.subheader("Step 3: Clusters based on action")
        # Step 1: KPI Aggregation
        df_contribution_users = actions_df.groupby('action_visitor_id').agg(
        nb_actions_total=('action_id', 'count'),
        nb_types_actions_uniques=('action_name', pd.Series.nunique),
        nb_groupes_actions=('action_group', pd.Series.nunique),
        nb_jours_actifs=('action_yyyymmdd', pd.Series.nunique),
        taux_repeat_visitor=('action_is_repeat_visitor', lambda x: x.sum() / len(x)),
        nb_mediums_utilisés=('action_medium', pd.Series.nunique),
        nb_sites_utilisés=('action_site_id', pd.Series.nunique),
        nb_contributions=('action_name', lambda x: ((x == 'frontend create') | (x == 'editor publish')).sum()),
        nb_modifications=('action_name', lambda x: (x == 'frontend modify').sum()),
        nb_publications=('action_group', lambda x: (x == 'publish').sum())
        ).reset_index()
    
        # Medium dummies
        medium_dummies = pd.get_dummies(actions_df[['action_visitor_id', 'action_medium']], columns=['action_medium'], prefix='medium', dtype=int)
        medium_usage = medium_dummies.groupby('action_visitor_id').max().reset_index()
        df_contribution_users = pd.merge(df_contribution_users, medium_usage, on='action_visitor_id', how='left')
        
        # Step 2: Log transform + normalization
        kpi_cols = [
            'nb_actions_total', 'nb_types_actions_uniques', 'nb_groupes_actions',
            'nb_jours_actifs', 'taux_repeat_visitor', 'nb_mediums_utilisés',
            'nb_sites_utilisés', 'nb_contributions', 'nb_modifications', 'nb_publications'
        ]
        df_kpi_contrib = df_contribution_users.copy()
        for col in kpi_cols:
            df_kpi_contrib[f'{col}_log'] = np.log1p(df_kpi_contrib[col])
        
        log_cols = [f'{col}_log' for col in kpi_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_kpi_contrib[log_cols])
        
        medium_cols = [col for col in df_kpi_contrib.columns if col.startswith('medium_')]
        df_contrib_scaled = pd.DataFrame(X_scaled, columns=[f'{col}_scaled' for col in kpi_cols])
        df_contrib_scaled['action_visitor_id'] = df_kpi_contrib['action_visitor_id'].values
        df_contrib_scaled = pd.concat([df_contrib_scaled, df_kpi_contrib[medium_cols].reset_index(drop=True)], axis=1)
        
        # Step 3: Clustering
        kmeans_final = KMeans(n_clusters=6, random_state=42, n_init='auto')
        df_kpi_contrib['cluster'] = kmeans_final.fit_predict(df_contrib_scaled.drop(columns=['action_visitor_id']))
        
        # Display cluster sizes
        st.subheader("📊 Cluster Sizes")
        cluster_counts = df_kpi_contrib['cluster'].value_counts().sort_index().reset_index()
        cluster_counts.columns = ['cluster', 'nb_users']
        st.dataframe(cluster_counts)
        
        # KPI Summary per cluster
        st.subheader("📈 KPI Averages by Cluster")
        df_contrib_cluster_summary = df_kpi_contrib.groupby('cluster')[kpi_cols].mean().round(2).reset_index()
        st.dataframe(df_contrib_cluster_summary)
        
        # Searchable cluster table
        st.subheader("🔍 Search Cluster Assignments")
        visitor_input = st.text_input("Enter a Visitor ID to find their cluster")
        if visitor_input:
            filtered_df = df_kpi_contrib[df_kpi_contrib['action_visitor_id'].astype(str).str.contains(visitor_input)]
            st.dataframe(filtered_df)
        else:
            st.dataframe(df_kpi_contrib[['action_visitor_id', 'cluster']].head(10))
    
with tabs[6]:
        st.header("💡Insights and Suggestions")
        # Step 1: Engagement score + personas
        st.subheader("Step 1: Cluster Personas Based on Engagement")
    
        action_weights = {
            'editor publish': 5,
            'frontend create': 4,
            'frontend modify': 3,
            'reaction': 2,
            'comment': 2,
            'frontend preview': 1,
            'editor edit': 1,
            'editor save': 1
        }
        actions_df['action_weight'] = actions_df['action_name'].map(action_weights).fillna(0)
    
        score_by_user = actions_df.groupby('action_visitor_id').agg(
            action_score=('action_weight', 'sum'),
            repeat_ratio=('action_is_repeat_visitor', lambda x: x.sum() / len(x))
        ).reset_index()
    
        max_action_score = score_by_user['action_score'].max()
        score_by_user['action_score_norm'] = score_by_user['action_score'] / max_action_score * 100
        score_by_user['engagement_score'] = (
            0.8 * score_by_user['action_score_norm'] + 0.2 * score_by_user['repeat_ratio'] * 100
        )
    
        if 'engagement_score' in df_kpi_contrib.columns:
            df_kpi_contrib = df_kpi_contrib.drop(columns='engagement_score')
    
        df_kpi_contrib = df_kpi_contrib.merge(
            score_by_user[['action_visitor_id', 'engagement_score']],
            on='action_visitor_id', how='left'
        )
    
        cluster_score_summary = df_kpi_contrib.groupby('cluster')['engagement_score'].mean().round(2).reset_index()
    
        cluster_to_persona = {
            0: "Profil dormant",
            1: "Collaborateur régulier",
            2: "Ambassadeur éditorial",
            3: "Simple observateur",
            4: "Éclaireur à convertir",
            5: "Utilisateur métier productif"
        }
        cluster_score_summary['persona'] = cluster_score_summary['cluster'].map(cluster_to_persona)
    
        def score_to_label(score):
            if score <= 10:
                return 'Très faible'
            elif score <= 20:
                return 'Faible'
            elif score <= 25:
                return 'Moyen'
            elif score <= 50:
                return 'Élevé'
            else:
                return 'Très élevé'
    
        cluster_score_summary['engagement_level'] = cluster_score_summary['engagement_score'].apply(score_to_label)
        cluster_score_summary = cluster_score_summary[['cluster', 'persona', 'engagement_score', 'engagement_level']]
    
        st.subheader("📌 Cluster Personas Summary")
        st.dataframe(cluster_score_summary)

        #Step 2: Suggestions
        
        st.subheader("Step 2: Combined Persona Matrix")
        navigation_personas = [
            "Curieux furtif", "Lecteur discret", "Indécis",
            "Éclaireur à convertir", "Utilisateur stable", "Explorateur intensif"
        ]
    
        contribution_personas = [
            "Profil dormant", "Simple observateur", "Éclaireur à convertir",
            "Collaborateur régulier", "Utilisateur métier productif", "Ambassadeur éditorial"
        ]
    
        persona_scores = {
            "Profil dormant": 0.02,
            "Simple observateur": 11.95,
            "Éclaireur à convertir": 18.85,
            "Collaborateur régulier": 30.33,
            "Utilisateur métier productif": 30.42,
            "Ambassadeur éditorial": 74.70
        }
    
        matrix_data = {}
        for nav in navigation_personas:
            matrix_data[nav] = []
            for contrib in contribution_personas:
                score = persona_scores[contrib]
                if score <= 20:
                    label = "Furtif passif"
                elif score <= 40:
                    label = "Profil à activer"
                elif score <= 60:
                    label = "Engagement modéré"
                elif score <= 80:
                    label = "Actif à valoriser"
                else:
                    label = "Ultra engagé"
                matrix_data[nav].append(f"{label} ({score:.2f})")
    
        persona_combined_matrix = pd.DataFrame(matrix_data, index=contribution_personas).T
        st.write("### 🔗 Combined Matrix")
        st.dataframe(persona_combined_matrix)
    
        # Étape 2 : Flatten pour analyse
        flat_data = []
        for nav_persona in persona_combined_matrix.index:
            for contrib_persona in persona_combined_matrix.columns:
                value = persona_combined_matrix.loc[nav_persona, contrib_persona]
                label, score = value.rsplit("(", 1)
                label = label.strip()
                score = float(score.replace(")", ""))
                flat_data.append({
                    "navigation_persona": nav_persona,
                    "contribution_persona": contrib_persona,
                    "combined_label": label,
                    "score": score
                })
    
        df_flat = pd.DataFrame(flat_data)
    
        # Étape 3 : Groupement
        df_grouped = df_flat.groupby('combined_label').agg(
            score_moyen=('score', 'mean'),
            nb_combinaisons=('score', 'count')
        ).reset_index()
    
        label_marketing = {
            "Ultra engagé": "Leader d'opinion",
            "Actif à valoriser": "Expert discret",
            "Engagement modéré": "Contributeur régulier",
            "Profil à activer": "Curieux à convertir",
            "Furtif passif": "Public fantôme"
        }
        df_grouped['persona_marketing'] = df_grouped['combined_label'].map(label_marketing)
    
        # Étape 4 : Ajouter 3 personas
        extra_rows = pd.DataFrame([
            {"persona_marketing": "Ambassadeur engagé", "score_moyen": 85.0, "nb_combinaisons": 3},
            {"persona_marketing": "Contributeur fiable", "score_moyen": 59.1, "nb_combinaisons": 3},
            {"persona_marketing": "Découvreur hésitant", "score_moyen": 20.0, "nb_combinaisons": 3}
        ])
    
        df_final_personas = pd.concat([
            df_grouped[['persona_marketing', 'score_moyen', 'nb_combinaisons']],
            extra_rows
        ], ignore_index=True)
    
        df_final_personas = df_final_personas.sort_values(by='score_moyen', ascending=False).reset_index(drop=True)
    
        st.subheader("🎯 Final Marketing Personas")
        st.dataframe(df_final_personas)
