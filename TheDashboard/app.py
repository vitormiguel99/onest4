import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="HR Dashboard", layout="wide")
st.title("🧠 HR Clustering Dashboard")

# Tabs
tabs = st.tabs(["🏠 Home", "📈 Overview", "🧠 Clustering", "📊 Results"])

# 🏠 Home
with tabs[0]:
    st.header("Welcome to the HR Dashboard")
    st.markdown("""
        This interactive dashboard helps you explore employee data,
        visualize key HR metrics, and analyze clustering results.

        Use the tabs above to navigate through the analysis.
    """)

# 📈 Overview (still empty)
with tabs[1]:
    st.header("📈 Overview")
    st.info("This section will show data summary and distributions. (Coming soon)")

# 🧠 Clustering
with tabs[2]:
    st.header("🧠 Clustering Analysis")

    # Load data
    csv_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/hr_training.csv?token=GHSAT0AAAAAADCCBYDEERMHAPCT73FJW3SMZ7ZEC5A"
    df = pd.read_csv(csv_url)

    # Preprocessing
    try:
        for col in ['satisfaction_level', 'last_evaluation']:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)

        numeric_cols = ['number_project', 'average_montly_hours', 'time_spend_company',
                        'work_accident', 'promotion_last_5years', 'left']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

        df_encoded = pd.get_dummies(df, columns=["job", "salary"], drop_first=True)

        # Clustering function
        def run_kmeans(df_subset, label, n_clusters=3):
            df_cluster = df_subset.drop(columns=["id_colab", "left"], errors="ignore")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(X_scaled)
            df_result = df_subset.copy()
            df_result[f"cluster_{label}"] = kmeans.labels_
            return df_result

        df_clustered = run_kmeans(df_encoded, "all")

        # Describe clusters
        def describe_clusters(df, cluster_col):
            summaries = []
            for cluster_id in sorted(df[cluster_col].unique()):
                cluster_data = df[df[cluster_col] == cluster_id]
                summary = {
                    "Cluster": cluster_id,
                    "Satisfaction": round(cluster_data["satisfaction_level"].mean(), 2),
                    "Evaluation": round(cluster_data["last_evaluation"].mean(), 2),
                    "Projects": round(cluster_data["number_project"].mean(), 2),
                    "Hours": round(cluster_data["average_montly_hours"].mean(), 2),
                    "Time at Company": round(cluster_data["time_spend_company"].mean(), 2),
                    "Work Accident": round(cluster_data["work_accident"].mean(), 2),
                    "Promoted": round(cluster_data["promotion_last_5years"].mean(), 2),
                    "Left %": round(cluster_data["left"].mean() * 100, 1),
                }
                summaries.append(summary)
            return pd.DataFrame(summaries)

        # Display
        st.subheader("🔢 Cluster Distribution")
        st.bar_chart(df_clustered["cluster_all"].value_counts())

        st.subheader("📋 Cluster Summary")
        summary_df = describe_clusters(df_clustered, "cluster_all")
        st.dataframe(summary_df)

        st.subheader("🧪 Sample Clustered Data")
        st.dataframe(df_clustered.head(10))

    except Exception as e:
        st.error(f"Something went wrong while clustering: {e}")

# 📊 Results (still empty)
with tabs[3]:
    st.header("📊 Results")
    st.info("This section will display final results, interpretations, and export options.")
