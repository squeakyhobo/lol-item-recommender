
import json
import os

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")

def name_clusters():
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
        data = json.load(f)
    
    # Human-readable names based on the PCA and item DNA
    names = {
        "0": "Starting Components",
        "1": "Jungle/Niche Starters",
        "2": "Heavy Health & AD Staples",
        "3": "AP Utility & Health",
        "4": "High AD / Lethality",
        "5": "Spellblade Duelist",
        "6": "Lifeline & Shields",
        "7": "Magic Resist Tank",
        "8": "Niche / Special Items",
        "9": "Grievous Wounds & Pen",
        "10": "Boots & Movement",
        "11": "Attack Speed & On-Hit",
        "12": "Armor Tank",
        "13": "Pure AP Power",
        "14": "Niche Defensive"
    }
    
    data["cluster_names"] = names
    
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "w") as f:
        json.dump(data, f, indent=4)
    print("[*] Clusters named and saved.")

if __name__ == "__main__":
    name_clusters()
