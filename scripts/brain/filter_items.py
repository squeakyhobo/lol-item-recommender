
import requests
import json
import os

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")

def generate_valid_targets():
    versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
    latest = versions[0]
    url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/item.json"
    items = requests.get(url).json()["data"]
    
    valid_ids = []
    
    for item_id, info in items.items():
        gold = info.get("gold", {}).get("total", 0)
        tags = info.get("tags", [])
        builds_into = info.get("into", [])
        
        # 1. Starter Items
        is_starter = "Starter" in info.get("description", "") or gold <= 450 and "Lane" in info.get("description", "")
        # Specific check for Dorans
        if "Doran" in info["name"]:
            is_starter = True
            
        # 2. Completed Items (Legendaries/Mythics/Boots)
        # Criteria: Expensive (>1500) and doesn"t build into anything
        is_completed = (not builds_into) and (gold >= 1500)
        
        # 3. Boots (Completed versions)
        is_boot = "Boots" in tags and gold > 500
        
        if is_starter or is_completed or is_boot:
            valid_ids.append(int(item_id))

    with open(os.path.join(DATA_DIR, "valid_targets.json"), "w") as f:
        json.dump(valid_ids, f)
    
    print(f"[*] Identified {len(valid_ids)} valid target items (Starters + Legendaries).")

if __name__ == "__main__":
    generate_valid_targets()

