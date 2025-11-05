import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import missingno as msno
import io

st.set_page_config(page_title="Clustering Global Development", layout="wide")

st.title("🌍 Clustering Global Development Data")

# --- Upload Data ---
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # --- Drop unwanted column ---
    df.drop(columns=['Number of Records'], errors="ignore", inplace=True)

    # --- Encode Country ---
    if 'Country' in df.columns:
        le = LabelEncoder()
        df['Country_encoded'] = le.fit_transform(df['Country'])
        df.drop(['Country'], axis=1, inplace=True)

    # --- Clean symbols ---
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(r"[\$,%]", "", regex=True)
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    # --- Missing Value Analysis ---
    st.subheader("Missing Value Analysis")
    st.write(f"🔍 Total missing values: {df.isnull().sum().sum()}")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    # --- Impute Missing Values ---
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        cat_cols = [col for col in cat_cols if not df[col].isnull().all()]
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    st.success("✅ Missing values imputed.")

    # --- Visualizations ---
    st.subheader("📈 Feature Distributions")
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, bins=30, color="skyblue", ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.subheader("📊 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])

    # --- PCA ---
    st.subheader("🔍 PCA Analysis")
    pca = PCA()
    data_pca = pca.fit_transform(X_scaled)
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    fig, ax = plt.subplots()
    ax.plot(var, color='red')
    ax.set_xlabel('Index')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title("PCA Variance Explained")
    st.pyplot(fig)

    n_components = st.slider("Select number of PCA components", 2, len(var), 15)
    data_pca = data_pca[:, :n_components]

    # --- Clustering ---
    st.subheader("🔗 Clustering Models")

    # KMeans
    st.markdown("**KMeans Clustering**")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data_pca)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

    k_clusters = st.slider("Select number of clusters for KMeans", 2, 10, 3)
    kmeans = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(data_pca)

    fig, ax = plt.subplots()
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=y_kmeans, palette="Set1", ax=ax)
    ax.set_title("KMeans Clusters")
    st.pyplot(fig)

    # Hierarchical
    st.markdown("**Hierarchical Clustering**")
    h_clusters = st.slider("Select number of clusters for Hierarchical", 2, 10, 4)
    hc = AgglomerativeClustering(n_clusters=h_clusters, linkage='ward')
    y_hc = hc.fit_predict(data_pca)

    fig, ax = plt.subplots()
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=y_hc, palette="Set2", ax=ax)
    ax.set_title("Hierarchical Clusters")
    st.pyplot(fig)

    # DBSCAN
    st.markdown("**DBSCAN Clustering**")
    eps = st.slider("Select epsilon (eps)", 0.1, 1.0, 0.5)
    min_samples = st.slider("Select min_samples", 1, 10, 5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(data_pca)

    fig, ax = plt.subplots()
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=dbscan_labels, palette="Set3", ax=ax)
    ax.set_title("DBSCAN Clusters")
    st.pyplot(fig)

    # --- Evaluation ---
    st.subheader("📌 Model Evaluation")

    def safe_silhouette(data, labels):
        try:
            return silhouette_score(data, labels)
        except:
            return None

    sil_kmeans = safe_silhouette(data_pca, y_kmeans)
    sil_hc = safe_silhouette(data_pca, y_hc)
    sil_dbscan = safe_silhouette(data_pca, dbscan_labels)

    st.write("**Silhouette Scores:**")
    st.write(f"KMeans: {sil_kmeans}")
    st.write(f"Hierarchical: {sil_hc}")
    st.write(f"DBSCAN: {sil_dbscan if sil_dbscan is not None else 'Could not be calculated'}")

    scores = {
        "KMeans": sil_kmeans,
        "Hierarchical": sil_hc,
        "DBSCAN": sil_dbscan if sil_dbscan is not None else -1
    }
    best_method = max(scores, key=scores.get)
    st.success(f"🏆 Best clustering method based on Silhouette Score: **{best_method}**")

else:
    st.info("📂 Please upload a dataset to begin.")
