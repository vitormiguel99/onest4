import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="HR Clustering Dashboard", layout="wide")
st.title("🧠 HR Clustering Dashboard")

# ✅ 1. Load the CSV from GitHub
csv_url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/hr_training.csv?token=GHSAT0AAAAAADCCBYDEIA5X7QX7ATT7BDPIZ7ZDUHA"  # update if needed
df = pd.read_csv(csv_url)  # assuming comma-separated now

# ✅ 2. Preprocessing
try:
    # Convert numeric fields
    for col in ['satisfaction_level', 'last_evaluation']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)

    numeric_cols = ['number_project', 'average_montly_hours', 'time_spend_company', 
                    'work_accident', 'promotion_last_5years', 'left']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=["job", "salary"], drop_first=True)
except Exception as e:
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

# ✅ 3. Run KMeans Clustering
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

# ✅ 4. Describe Clusters
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

summary_df = describe_clusters(df_clustered, "cluster_all")

# ✅ 5. Display Results
st.subheader("📊 Cluster Distribution")
st.bar_chart(df_clustered["cluster_all"].value_counts())

st.subheader("📋 Cluster Summary")
st.dataframe(summary_df)

st.subheader("🧪 Sample of Clustered Data")
st.dataframe(df_clustered.head(10))
