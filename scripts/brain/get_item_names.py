
import requests
import json
import os

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")

def download_item_names():
    # Get latest version
    versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
    latest = versions[0]
    print(f"[*] Fetching Item Names for version {latest}")
    
    url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/item.json"
    items = requests.get(url).json()["data"]
    
    # Map ID -> Name
    id_to_name = {id: info["name"] for id, info in items.items()}
    
    with open(os.path.join(DATA_DIR, "item_names.json"), "w") as f:
        json.dump(id_to_name, f, indent=4)
    print(f"[*] Saved {len(id_to_name)} item names to data/item_names.json")

if __name__ == "__main__":
    download_item_names()

