"""
Spotify Song Clustering & Mood-Based Recommender (Training & Analysis Pipeline)
Unsupervised Machine Learning: K-Means Clustering + 2D PCA + Euclidean Similarity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_PATH = "dataset.csv"
FEATURES = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness"]
TOP_N_SONGS = None        # Set to 5000 for top 5,000 popular songs, or None for all 73,608 unique songs
OPTIMAL_K = 4
RANDOM_STATE = 42


# ==============================================================================
# 1. DATA LOADING & CLEANING
# ==============================================================================
def load_and_clean_data(filepath: str, features: list, top_n: int | None = None) -> pd.DataFrame:
    """Load Spotify dataset, remove missing/duplicate tracks, and sort by popularity."""
    df = pd.read_csv(filepath)
    raw_count = len(df)

    cols = ["track_name", "artists", "popularity"] + features
    df = df[cols].dropna().drop_duplicates(subset=["track_name"])
    clean_count = len(df)

    df = df.sort_values(by="popularity", ascending=False)
    if top_n is not None:
        df = df.head(top_n)

    df = df.reset_index(drop=True)
    print(f"Raw rows: {raw_count:,} -> Unique clean songs: {clean_count:,} -> Using: {len(df):,} songs")
    return df


# ==============================================================================
# 2. FEATURE SCALING
# ==============================================================================
def scale_features(df: pd.DataFrame, features: list):
    """Standardize numerical audio features to mean=0 and std=1."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, scaler


# ==============================================================================
# 3. OPTIMAL K EVALUATION (ELBOW METHOD & SILHOUETTE SCORE)
# ==============================================================================
def evaluate_k_range(X_scaled: np.ndarray, k_range=range(2, 11)):
    """Compute and plot WCSS (Elbow Curve) and Silhouette Score across K values."""
    wcss = []
    sil_scores = []
    sample_size = 5000 if len(X_scaled) > 5000 else None

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(
            silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=RANDOM_STATE)
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, wcss, marker="o", color="blue")
    ax1.set_title("Elbow Method (WCSS)")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("WCSS (Inertia)")

    ax2.plot(k_range, sil_scores, marker="o", color="green")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. MODEL TRAINING & AUTOMATIC MOOD LABELING
# ==============================================================================
def train_kmeans_and_label(df: pd.DataFrame, X_scaled: np.ndarray, features: list, k: int = 4):
    """Fit K-Means and automatically assign mood labels based on cluster audio profiles."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_profile = df.groupby("cluster")[features].mean()
    print("\nCluster Feature Averages (Profile):")
    print(cluster_profile.round(3))

    # Auto-label clusters when k == 4 so labels never get swapped across runs
    if k == 4:
        remaining = list(cluster_profile.index)
        cluster_names = {}

        sleep_c = cluster_profile.loc[remaining, "energy"].idxmin()
        cluster_names[sleep_c] = "Sleep / Ambient / Soft"
        remaining.remove(sleep_c)

        chill_c = cluster_profile.loc[remaining, "acousticness"].idxmax()
        cluster_names[chill_c] = "Chill / Mellow Acoustic"
        remaining.remove(chill_c)

        party_c = cluster_profile.loc[remaining, "valence"].idxmax()
        cluster_names[party_c] = "Happy / Party / Dance"
        remaining.remove(party_c)

        cluster_names[remaining[0]] = "High Energy / Workout"
        df["mood"] = df["cluster"].map(cluster_names)
    else:
        df["mood"] = df["cluster"].apply(lambda c: f"Cluster {c}")

    return df, kmeans


# ==============================================================================
# 5. MODEL EVALUATION METRICS
# ==============================================================================
def print_evaluation_metrics(X_scaled: np.ndarray, df: pd.DataFrame, kmeans: KMeans):
    """Print Silhouette Score and WCSS (Inertia) for the final model."""
    sample_size = 5000 if len(X_scaled) > 5000 else None
    sil = silhouette_score(X_scaled, df["cluster"], sample_size=sample_size, random_state=RANDOM_STATE)

    print("\n--- Final Model Evaluation ---")
    print(f"Silhouette Score: {sil:.3f}  (Higher is better, max = 1.0)")
    print(f"WCSS (Inertia):   {kmeans.inertia_:,.1f}  (Lower is better)")


# ==============================================================================
# 6. 2D PCA VISUALIZATION
# ==============================================================================
def visualize_pca_clusters(df: pd.DataFrame, X_scaled: np.ndarray):
    """Project 6D scaled features into 2D using PCA and plot clusters."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]

    # Sample 5,000 points for plotting if dataset is large so the chart stays fast and readable
    plot_df = df.sample(n=5000, random_state=RANDOM_STATE) if len(df) > 5000 else df

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x="pca_1", y="pca_2", hue="mood", palette="viridis", alpha=0.6)
    plt.title("Spotify Song Clusters (Visualized via 2D PCA)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.legend(title="Mood Cluster")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 7. EUCLIDEAN DISTANCE SONG RECOMMENDER
# ==============================================================================
def recommend_songs(
    song_name: str, df: pd.DataFrame, X_scaled: np.ndarray, num_recommendations: int = 5
):
    """Find the closest songs inside the same mood cluster using 6D Euclidean distance."""
    match = df[df["track_name"].str.contains(song_name, case=False, na=False)]

    if match.empty:
        print(f"Song '{song_name}' not found in dataset. Try another title!")
        return None

    idx = match.index[0]
    target_song = df.loc[idx]
    target_cluster = target_song["cluster"]

    print(f"\nInput Song: '{target_song['track_name']}' by {target_song['artists']}")
    print(f"Detected Vibe: {target_song['mood']} (Cluster {target_cluster})\n")

    same_cluster_idx = df[(df["cluster"] == target_cluster) & (df.index != idx)].index
    distances = np.linalg.norm(X_scaled[same_cluster_idx] - X_scaled[idx], axis=1)
    top_idx = same_cluster_idx[np.argsort(distances)[:num_recommendations]]

    recs = df.loc[top_idx, ["track_name", "artists", "mood"]]
    print(recs.to_string(index=False))
    return recs


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    # Step 1: Load & clean
    df = load_and_clean_data(DATA_PATH, FEATURES, top_n=TOP_N_SONGS)

    # Step 2: Scale features
    X_scaled, _ = scale_features(df, FEATURES)

    # Step 3: Evaluate K = 2..10 (Elbow & Silhouette plots)
    evaluate_k_range(X_scaled, k_range=range(2, 11))

    # Step 4: Train final K-Means model & auto-label clusters
    df, kmeans = train_kmeans_and_label(df, X_scaled, FEATURES, k=OPTIMAL_K)

    # Step 5: Print evaluation metrics
    print_evaluation_metrics(X_scaled, df, kmeans)

    # Step 6: Visualize clusters in 2D via PCA
    visualize_pca_clusters(df, X_scaled)

    # Step 7: Test live song recommendation
    recommend_songs("Until I Found You", df, X_scaled, num_recommendations=5)


if __name__ == "__main__":
    main()
