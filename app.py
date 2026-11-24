import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Spotify Mood Clustering", layout="wide")


@st.cache_data
def load_and_cluster():
    df = pd.read_csv("dataset.csv")
    features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo', 'loudness']

    # Clean data & reset index so DataFrame indices align 1:1 with X_scaled rows
    df = df.dropna(subset=['track_name', 'artists'] + features)
    df = df.drop_duplicates(subset=['track_name'])
    if 'popularity' in df.columns:
        df = df.sort_values('popularity', ascending=False)
    df = df.reset_index(drop=True)

    X_scaled = StandardScaler().fit_transform(df[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Auto-label clusters based on actual audio averages (never gets swapped)
    profile = df.groupby('cluster')[features].mean()
    remaining = list(profile.index)
    cluster_names = {}

    sleep_c = profile.loc[remaining, 'energy'].idxmin()
    cluster_names[sleep_c] = 'Sleep / Ambient / Soft'
    remaining.remove(sleep_c)

    chill_c = profile.loc[remaining, 'acousticness'].idxmax()
    cluster_names[chill_c] = 'Chill / Mellow Acoustic'
    remaining.remove(chill_c)

    party_c = profile.loc[remaining, 'valence'].idxmax()
    cluster_names[party_c] = 'Happy / Party / Dance'
    remaining.remove(party_c)

    cluster_names[remaining[0]] = 'High Energy / Workout'
    df['mood'] = df['cluster'].map(cluster_names)

    # 2D PCA strictly for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'], df['pca_2'] = X_pca[:, 0], X_pca[:, 1]
    var_ratio = pca.explained_variance_ratio_ * 100

    return df, X_scaled, var_ratio


df, X_scaled, var_ratio = load_and_cluster()

st.title("🎵 Spotify Song Clustering & Recommender")
st.caption(f"Unsupervised Learning (K-Means + PCA) on {len(df):,} Unique Tracks")

# Interactive Song Selector
song = st.selectbox("Pick a song to find similar vibes:", df['track_name'].values)
idx = df[df['track_name'] == song].index[0]
target = df.loc[idx]

st.subheader(f"Detected Vibe: **{target['mood']}** (Cluster {target['cluster']})")

# Find the 5 closest songs in the exact same cluster (Euclidean distance)
same_cluster_idx = df[(df['cluster'] == target['cluster']) & (df.index != idx)].index
distances = np.linalg.norm(X_scaled[same_cluster_idx] - X_scaled[idx], axis=1)
top5_idx = same_cluster_idx[np.argsort(distances)[:5]]

recs = df.loc[top5_idx, ['track_name', 'artists', 'mood']]
st.dataframe(recs, use_container_width=True, hide_index=True)

# 2D PCA Plot (sampled to 5,000 points for instant redraws + selected song highlighted)
st.subheader("2D PCA Cluster Visualization")
plot_df = df.sample(min(5000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    data=plot_df,
    x='pca_1',
    y='pca_2',
    hue='mood',
    palette='viridis',
    alpha=0.45,
    s=15,
    ax=ax,
)
ax.scatter(
    target['pca_1'],
    target['pca_2'],
    color='red',
    s=160,
    marker='*',
    edgecolor='black',
    zorder=5,
    label=f"Selected: {target['track_name']}",
)
ax.set_xlabel(f"PC1 ({var_ratio[0]:.1f}% variance)")
ax.set_ylabel(f"PC2 ({var_ratio[1]:.1f}% variance)")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
st.pyplot(fig)
plt.close(fig)