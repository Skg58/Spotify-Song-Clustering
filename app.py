import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Spotify Mood Clustering", layout="wide")
st.title("🎵 Spotify Song Clustering & Recommender")
st.caption("Unsupervised Learning (K-Means + PCA) on 5,000 Popular Tracks")

@st.cache_data
def load_and_cluster():
    df = pd.read_csv("spotify_5000.csv")
    features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo', 'loudness']
    X_scaled = StandardScaler().fit_transform(df[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'], df['pca_2'] = X_pca[:, 0], X_pca[:, 1]

    # Update these names to match your cluster_profile table
    cluster_names = {0: 'Chill / Acoustic', 1: 'High Energy / Workout', 2: 'Happy / Party', 3: 'Mellow / Sad'}
    df['mood'] = df['cluster'].map(cluster_names)
    return df

df = load_and_cluster()

# Interactive Song Selector
song = st.selectbox("Pick a song to find similar vibes:", df['track_name'].values)
target = df[df['track_name'] == song].iloc[0]

st.subheader(f"Detected Vibe: **{target['mood']}** (Cluster {target['cluster']})")
recs = df[(df['cluster'] == target['cluster']) & (df['track_name'] != song)].sample(5)
st.dataframe(recs[['track_name', 'artists', 'mood']], use_container_width=True, hide_index=True)

# 2D PCA Plot
st.subheader("2D PCA Cluster Visualization")
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='mood', palette='viridis', alpha=0.5, ax=ax)
st.pyplot(fig)