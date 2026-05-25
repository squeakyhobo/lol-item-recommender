import requests
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def download_champion_data():
    versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
    latest = versions[0]
    print(f"[*] Fetching Data Dragon for version {latest}")
    
    url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json"
    data = requests.get(url).json()["data"]
    return data

def process_data(raw_data):
    rows = []
    for name, info in raw_data.items():
        stats = info["stats"]
        tags = info["tags"]
        row = {
            "name": name,
            "hp": stats["hp"],
            "armor": stats["armor"],
            "spellblock": stats["spellblock"],
            "attackdamage": stats["attackdamage"],
            "attackspeed": stats["attackspeed"],
            "range": stats["attackrange"],
            "is_mage": 1 if "Mage" in tags else 0,
            "is_assassin": 1 if "Assassin" in tags else 0,
            "is_tank": 1 if "Tank" in tags else 0,
            "is_support": 1 if "Support" in tags else 0,
            "is_fighter": 1 if "Fighter" in tags else 0,
            "is_marksman": 1 if "Marksman" in tags else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def cluster_and_visualize(df, df_scaled, k=8):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["cluster"] = kmeans.fit_predict(df_scaled)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)
    df["pca1"] = components[:, 0]
    df["pca2"] = components[:, 1]
    
    plt.figure(figsize=(20,15))
    scatter = plt.scatter(df["pca1"], df["pca2"], c=df["cluster"], cmap="tab10", s=100, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    
    for i in range(len(df)):
        plt.annotate(df["name"][i], (df["pca1"][i], df["pca2"][i]), 
                     fontsize=9, alpha=0.8, xytext=(5,5), textcoords="offset points")
            
    plt.title(f"Champion Archetypes (K={k}) - PCA Projection")
    plot_path = os.path.join(DATA_DIR, "archetype_clusters.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[*] High-res cluster visualization saved to {plot_path}")
    
    summary = {}
    for cluster_id in range(k):
        members = df[df["cluster"] == cluster_id]["name"].tolist()
        summary[f"Cluster_{cluster_id}"] = members
        print(f"\nCluster {cluster_id}: {", ".join(members[:10])}...")

    summary_path = os.path.join(DATA_DIR, "cluster_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    mapping_path = os.path.join(DATA_DIR, "archetypes.json")
    mapping = df.set_index("name")["cluster"].to_dict()
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)

if __name__ == "__main__":
    raw = download_champion_data()
    df = process_data(raw)
    features = df.drop("name", axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    cluster_and_visualize(df, scaled, k=8)

