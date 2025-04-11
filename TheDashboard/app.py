import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🧠 Clustering Analysis - HR Data")

# Load data
url = "https://raw.githubusercontent.com/vitormiguel99/onest4/refs/heads/main/TheDashboard/data/hr_training.csv?token=GHSAT0AAAAAADCCBYDED5P7FLHSBLBAMCO2Z7ZDPNA"
df = pd.read_csv(url, sep=";")

# Preprocess
for col in ['satisfaction_level', 'last_evaluation']:
    df[col] = df[col].str.replace(',', '.').astype(float)

numeric_cols = ['number_project', 'average_montly_hours', 'time_spend_company', 'work_accident', 'promotion_last_5years', 'left']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

df_encoded = pd.get_dummies(df, columns=["job", "salary"], drop_first=True)

# Cluster function
def run_kmeans(df_subset, cluster_label, n_clusters=3):
    df_cluster = df_subset.drop(columns=["id_colab", "left"], errors="ignore")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    df_result = df_subset.copy()
    df_result[f"cluster_{cluster_label}"] = kmeans.labels_
    return df_result

# Run clustering
df_all_clusters = run_kmeans(df_encoded, "all")

# Show cluster counts
st.subheader("🔢 Cluster Distribution (All Employees)")
st.bar_chart(df_all_clusters["cluster_all"].value_counts())

# Optional: Preview the result
st.subheader("📄 Sample Clustered Data")
st.dataframe(df_all_clusters.head())
