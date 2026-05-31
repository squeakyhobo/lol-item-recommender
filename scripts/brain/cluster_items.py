
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")

def cluster_items(n_clusters=15):
    with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f:
        dna_map = json.load(f)
    
    item_ids = []
    features = []
    
    # We only cluster "Legendary/Completed" items or significant components
    # For now, let's use anything with a name
    for iid, stats in dna_map.items():
        # Feature vector: [ad, ap, hp, armor, mr, as, ms, crit, lifesteal, anti_heal, lifeline, spellblade, hydra]
        feat = [
            stats.get("ad", 0) / 100,
            stats.get("ap", 0) / 150,
            stats.get("hp", 0) / 1000,
            stats.get("armor", 0) / 100,
            stats.get("mr", 0) / 100,
            stats.get("as", 0) / 1.0,
            stats.get("ms", 0) / 100,
            stats.get("crit", 0) / 1.0,
            stats.get("lifesteal", 0) / 1.0,
            stats.get("anti_heal", 0),
            stats.get("has_lifeline", 0),
            stats.get("is_spellblade", 0),
            stats.get("has_burn", 0)
        ]
        item_ids.append(iid)
        features.append(feat)

    X = np.array(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    item_to_cluster = {item_ids[i]: int(clusters[i]) for i in range(len(item_ids))}
    
    # Create a summary for the HUD
    cluster_to_items = {}
    for iid, cid in item_to_cluster.items():
        if cid not in cluster_to_items: cluster_to_items[cid] = []
        cluster_to_items[cid].append(iid)

    # Save mapping
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "w") as f:
        json.dump({
            "item_to_cluster": item_to_cluster,
            "cluster_to_items": cluster_to_items
        }, f, indent=4)
    
    print(f"[*] Successfully clustered {len(item_ids)} items into {n_clusters} archetypes.")
    
    # Print examples of Cluster 2 (usually Bruiser items)
    with open(os.path.join(DATA_DIR, "item_names.json"), "r") as f:
        names = json.load(f)
    
    for cid in range(n_clusters):
        sample_items = [names.get(iid, iid) for iid in cluster_to_items[cid][:5]]
        print(f"Cluster {cid}: {', '.join(sample_items)}")

if __name__ == "__main__":
    cluster_items()
