import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from collections import defaultdict
import seaborn as sns
import os

def generate_visualisations(df, embeddings, PLOTS_DIR):
    """
    Generate visualisations for the processed data.
    """
    # Plot clusters
    plot_cluster_scatter(df, embeddings, PLOTS_DIR)

    # Plot cluster sizes
    plot_cluster_sizes(df, PLOTS_DIR)

    # Plot word clouds for each cluster
    plot_wordclouds(df, PLOTS_DIR)

    # Plot distance to centroid
    plot_distance_to_centroid(df, PLOTS_DIR)

    # Plot similarity heatmap within a cluster
    for cluster_id in df["cluster_id"].unique():
        plot_similarity_heatmap_within_cluster(df, embeddings, cluster_id, PLOTS_DIR)

def plot_cluster_scatter(df, embeddings, PLOTS_DIR):
    """
    Visualise clusters using t-SNE on sentence embeddings.
    """
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    reduced = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=df["cluster_id"].values, palette="tab10", s=40, alpha=0.8, edgecolor="k")
    plt.title("t-SNE Scatter Plot of Clusters (on Sentence Embeddings)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(PLOTS_DIR, "tsne_clusters.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_cluster_sizes(df, PLOTS_DIR):
    """
    Plot the distribution of cluster sizes.
    """
    cluster_counts = df["cluster_id"].value_counts().sort_index()

    plt.figure(figsize=(10, 8))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, hue=cluster_counts.values, palette="tab10", legend=False)
    plt.title("Distribution of Cluster Sizes")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Companies")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cluster_sizes.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_wordclouds(df, PLOTS_DIR):
    """
    Generate word clouds for each cluster based on the lemmatized descriptions.
    """
    # Create a mapping: cluster_id to list of descriptions
    cluster_texts = defaultdict(list)
    for cluster_id, text in zip(df["cluster_id"], df["lemmatized_description"]):
        cluster_texts[cluster_id].append(text)
    
    # Generate word clouds for each cluster
    for cluster_id in sorted(cluster_texts.keys())[:7]:
        text = " ".join(cluster_texts[cluster_id])
        wordcloud = WordCloud(width=800, height=400, background_color="white",
                              max_words=100, colormap="tab10").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Cluster {cluster_id} Word Cloud")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"wordcloud_cluster_{cluster_id}.png"), dpi=300, bbox_inches="tight")
        plt.close()

def plot_distance_to_centroid(df, PLOTS_DIR):
    """
    Plot the distribution of distances to the centroid for each cluster.
    """
    plt.figure(figsize=(10, 8))
    sns.histplot(df["distance_to_centroid"], bins=30, kde=True, color="steelblue")
    plt.title("Distribution of Distance to Centroid")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "distance_to_centroid.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_similarity_heatmap_within_cluster(df, embeddings, cluster_id, PLOTS_DIR):
    """
    Plot a heatmap of cosine similarity within a specific cluster.
    """
    # Filter to chosen cluster
    cluster_df = df[df["cluster_id"] == cluster_id]

    sample_df = cluster_df.sample(15, random_state=42)
    sample_embeddings = embeddings[sample_df.index]
    labels = sample_df["id"].astype(str).tolist()

    # Compute cosine similarity
    sim_matrix = cosine_similarity(sample_embeddings)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels,
                cmap="coolwarm", annot=True, fmt=".2f", square=True)
    plt.title(f"Cosine Similarity Within Cluster {cluster_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"similarity_heatmap_sample_cluster_{cluster_id}.png"), dpi=300, bbox_inches="tight")
    plt.close()
