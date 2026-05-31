
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")

def visualize_clusters():
    with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f:
        dna_map = json.load(f)
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
        clusters = json.load(f)["item_to_cluster"]
    with open(os.path.join(DATA_DIR, "item_names.json"), "r") as f:
        names = json.load(f)

    ids = []
    features = []
    colors = []

    for iid, stats in dna_map.items():
        if iid not in clusters: continue
        feat = [
            stats.get("ad", 0), stats.get("ap", 0), stats.get("hp", 0),
            stats.get("armor", 0), stats.get("mr", 0), stats.get("as", 0),
            stats.get("ms", 0), stats.get("crit", 0), stats.get("lifesteal", 0)
        ]
        ids.append(iid)
        features.append(feat)
        colors.append(clusters[iid])

    X = np.array(features)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("LTA Item Archetype Map (PCA Projection)")
    plt.xlabel("Principal Component 1 (Power/Stats)")
    plt.ylabel("Principal Component 2 (Utility/Type)")
    
    # Label a few key items for context
    targets = {
        "3071": "Black Cleaver",
        "3078": "Trinity Force",
        "3089": "Rabadon's",
        "3075": "Thornmail",
        "3111": "Merc Treads",
        "3181": "Hullbreaker"
    }
    for i, iid in enumerate(ids):
        if iid in targets:
            plt.annotate(targets[iid], (X_pca[i, 0], X_pca[i, 1]), fontsize=9, fontweight='bold')

    output_path = os.path.join(DATA_DIR, "item_clusters_visual.png")
    plt.savefig(output_path)
    print(f"[*] Visual map saved to {output_path}")

if __name__ == "__main__":
    visualize_clusters()
