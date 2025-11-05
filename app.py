import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="Global Development Clustering", layout="wide")
st.title("🌍 Global Development Clustering App")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # --- Load and Clean Data ---
    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file)
        df = df.drop(columns=["Number of Records"], errors="ignore")

        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"[\$,%]", "", regex=True)
            try:
                df[col] = df[col].astype(float)
            except:
                pass

        if 'Country' in df.columns:
            le = LabelEncoder()
            df['Country_encoded'] = le.fit_transform(df['Country'])
            df['Country_encoded'] = df['Country_encoded'].astype(float)
            df.drop('Country', axis=1, inplace=True)

        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        cat_cols = [col for col in cat_cols if col != "Country_encoded"]
        for col in cat_cols:
            df[col] = pd.factorize(df[col])[0].astype(float)

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df

    df = load_data(uploaded_file)

    # --- Preprocessing ---
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    X_scaled = np.nan_to_num(X_scaled)

    # --- Sidebar Options ---
    model_choice = st.sidebar.selectbox("Choose Clustering Model", ["KMeans", "DBSCAN", "Hierarchical"])
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)
    eps = st.sidebar.slider("DBSCAN eps", min_value=0.5, max_value=5.0, value=1.5)
    min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=3, max_value=10, value=5)

    # --- Clustering ---
    if model_choice == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif model_choice == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    elif model_choice == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)

    df['Cluster'] = labels

    # --- Silhouette Score Evaluation ---
    def get_silhouette(X, labels):
        try:
            return round(silhouette_score(X, labels), 3)
        except:
            return 'N/A'

    if st.button("Evaluate Clustering"):
        score = get_silhouette(X_scaled, labels)
        st.subheader("Silhouette Score")
        st.write(f"{model_choice} Score: **{score}**")

    # --- Best Model Selection (Silhouette Only) ---
    if st.button("Select Best Model by Silhouette Score"):
        km = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        hc = AgglomerativeClustering(n_clusters=n_clusters).fit(X_scaled)

        db_valid = len(set(db.labels_)) > 1 and -1 in db.labels_
        db_filtered = db.labels_ != -1 if db_valid else None

        km_sil = get_silhouette(X_scaled, km.labels_)
        db_sil = get_silhouette(X_scaled[db_filtered], db.labels_[db_filtered]) if db_valid else -1
        hc_sil = get_silhouette(X_scaled, hc.labels_)

        scores = {
            'KMeans': km_sil if isinstance(km_sil, float) else -1,
            'DBSCAN': db_sil if isinstance(db_sil, float) else -1,
            'Hierarchical': hc_sil if isinstance(hc_sil, float) else -1
        }

        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]

        st.subheader("Best Model Based on Silhouette Score")
        st.write(f"*Best Model:* {best_model}")
        st.write(f"*Silhouette Score:* {best_score}")

    # --- Cluster Assignments ---
    st.subheader("🌍 Cluster Assignments")
    if 'Country_encoded' in df.columns:
        st.dataframe(df[['Country_encoded', 'Cluster']].sort_values(by='Cluster'))
    else:
        st.dataframe(df[['Cluster']].sort_values(by='Cluster'))

    # --- Cluster Summary ---
    st.subheader("Cluster Summary")
    summary_raw = df[df['Cluster'] != -1].groupby('Cluster')[numeric_df.columns].mean()
    valid_cols = summary_raw.columns[~summary_raw.isnull().any()]
    summary = summary_raw[valid_cols].round(2)
    st.dataframe(summary)

    # --- Cluster Profiling ---
    st.subheader("Cluster Profiles")
    percentiles = df[numeric_df.columns].quantile([0.25, 0.5, 0.75])
    for cluster_id, row in summary.iterrows():
        st.markdown(f"### Cluster {cluster_id}")
        description = []
        for col in valid_cols:
            val = row[col]
            if pd.isna(val): continue
            p25 = percentiles.loc[0.25, col]
            p75 = percentiles.loc[0.75, col]
            if val <= p25:
                label = f"Low {col}"
            elif val <= p75:
                label = f"Moderate {col}"
            else:
                label = f"High {col}"
            description.append(label)
        st.write("• " + "\n• ".join(description) if description else "No interpretable indicators.")

    # --- PCA Visualization ---
    st.subheader("Cluster Visualization (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df['Cluster'],
        cmap='tab10',
        s=60,
        edgecolor='k'
    )
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_title("Clusters Visualized in PCA Space", fontsize=14)
    ax.grid(True)

    unique_clusters = np.unique(df['Cluster'])
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=f'Cluster {int(c)}',
                   markerfacecolor=plt.cm.tab10(c / 10),
                   markersize=10)
        for c in unique_clusters if c != -1
    ]
    if -1 in unique_clusters:
        handles.append(
            plt.Line2D([0], [0], marker='x', color='gray',
                       label='Noise (DBSCAN)',
                       markersize=10)
        )
    ax.legend(handles=handles, title="Clusters", loc="best")
    st.pyplot(fig)

else:
    st.info("📂 Please upload an Excel file to begin.")
