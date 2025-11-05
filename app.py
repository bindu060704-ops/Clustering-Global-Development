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

# --- Page Config ---
st.set_page_config(page_title="Clustering Global Development", layout="wide")
st.title("🌍 Clustering Global Development Data")

# --- Sidebar Model Selection ---
st.sidebar.header("🔧 Model Settings")
model_choice = st.sidebar.selectbox("Choose Clustering Model", ["KMeans", "Hierarchical", "DBSCAN"])
show_best_model = st.sidebar.button("🏆 Show Best Model")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # --- Preprocessing ---
    df.drop(columns=['Number of Records'], errors="ignore", inplace=True)
    if 'Country' in df.columns:
        le = LabelEncoder()
        df['Country_encoded'] = le.fit_transform(df['Country'])
        df.drop(['Country'], axis=1, inplace=True)

    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(r"[\$,%]", "", regex=True)
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        cat_cols = [col for col in cat_cols if not df[col].isnull().all()]
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # --- Scaling & PCA ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])
    pca = PCA()
    data_pca = pca.fit_transform(X_scaled)
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    st.subheader("📊 PCA Variance Explained")
    fig, ax = plt.subplots()
    ax.plot(var, color='red')
    ax.set_xlabel('Components')
    ax.set_ylabel('Cumulative Variance (%)')
    st.pyplot(fig)

    n_components = st.slider("Select number of PCA components", 2, len(var), 15)
    data_pca = data_pca[:, :n_components]

    # --- Clustering ---
    st.subheader("🔗 Clustering Visualization")

    def plot_clusters(labels, title):
        fig, ax = plt.subplots()
        sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=labels, palette="Set1", ax=ax)
        ax.set_title(title)
        st.pyplot(fig)

    def cluster_summary(labels):
        df['Cluster'] = labels
        st.write("📋 Cluster Summary")
        st.dataframe(df.groupby('Cluster').mean().round(2))

    silhouette_scores = {}

    if model_choice == "KMeans":
        k = st.sidebar.slider("Number of clusters (KMeans)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data_pca)
        sil_score = silhouette_score(data_pca, labels)
        silhouette_scores["KMeans"] = sil_score
        plot_clusters(labels, "KMeans Clustering")
        cluster_summary(labels)

    elif model_choice == "Hierarchical":
        h = st.sidebar.slider("Number of clusters (Hierarchical)", 2, 10, 4)
        hc = AgglomerativeClustering(n_clusters=h, linkage='ward')
        labels = hc.fit_predict(data_pca)
        sil_score = silhouette_score(data_pca, labels)
        silhouette_scores["Hierarchical"] = sil_score
        plot_clusters(labels, "Hierarchical Clustering")
        cluster_summary(labels)

    elif model_choice == "DBSCAN":
        eps = st.sidebar.slider("Epsilon (DBSCAN)", 0.1, 1.0, 0.5)
        min_samples = st.sidebar.slider("Min Samples (DBSCAN)", 1, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_pca)
        try:
            sil_score = silhouette_score(data_pca, labels)
        except:
            sil_score = None
        silhouette_scores["DBSCAN"] = sil_score
        plot_clusters(labels, "DBSCAN Clustering")
        cluster_summary(labels)

    # --- Model Comparison ---
    st.subheader("📈 Model Comparison")
    for model, score in silhouette_scores.items():
        st.write(f"{model}: {score if score is not None else 'Could not be calculated'}")

    # --- Best Model ---
    if show_best_model:
        valid_scores = {k: v for k, v in silhouette_scores.items() if v is not None}
        if valid_scores:
            best_model = max(valid_scores, key=valid_scores.get)
            st.success(f"🏅 Best Model: {best_model} with Silhouette Score {valid_scores[best_model]:.3f}")
        else:
            st.warning("No valid silhouette scores to determine best model.")
else:
    st.info("📂 Please upload a dataset to begin.")
