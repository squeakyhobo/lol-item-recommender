
import requests
import json
import os

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")

def extract_champion_dna():
    versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
    latest = versions[0]
    url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json"
    champs = requests.get(url).json()["data"]
    
    dna_map = {}

    for champ_id, info in champs.items():
        stats = info.get("stats", {})
        
        # We store both Base and Growth to calculate live DNA later
        dna = {
            "name": info["name"],
            "hp": {"base": stats.get("hp", 0), "growth": stats.get("hpperlevel", 0)},
            "armor": {"base": stats.get("armor", 0), "growth": stats.get("armorperlevel", 0)},
            "mr": {"base": stats.get("spellblock", 0), "growth": stats.get("spellblockperlevel", 0)},
            "ad": {"base": stats.get("attackdamage", 0), "growth": stats.get("attackdamageperlevel", 0)},
            "as": {"base": stats.get("attackspeed", 0), "growth": stats.get("attackspeedperlevel", 0)},
            "ms": {"base": stats.get("movespeed", 0), "growth": 0} # MS doesn't usually grow per level
        }
        dna_map[champ_id] = dna

    with open(os.path.join(DATA_DIR, "champion_dna.json"), "w") as f:
        json.dump(dna_map, f, indent=4)
    
    print(f"[*] Saved Champion DNA (Base + Growth) for {len(champs)} champions.")

if __name__ == "__main__":
    extract_champion_dna()
