
import requests
import json
import os
import sys
import pandas as pd
import numpy as np

# Calculate root directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
import config
DATA_DIR = config.DATA_DIR

def extract_dna():
    versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
    latest = versions[0]
    url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/item.json"
    items = requests.get(url).json()["data"]
    
    rows = []
    # Keywords to detect passives
    PASSIVE_KEYWORDS = {
        "anti_heal": ["Grievous Wounds", "healing reduction"],
        "has_lifeline": ["Lifeline", "shield when your health", "falling below"],
        "has_burn": ["Burning", "burn", "Torment", "Immolate"],
        "has_pen": ["Armor Penetration", "Magic Penetration", "Armor Shred"],
        "is_spellblade": ["Spellblade", "after using an ability"],
        "has_heal": ["healing", "Omnivamp", "Life Steal"],
        "has_slow": ["Slow", "Icy", "Movement Speed reduction"]
    }

    for item_id, info in items.items():
        stats = info.get("stats", {})
        desc = info.get("description", "")
        
        # 1. Base Stats DNA
        dna = {
            "id": item_id,
            "name": info["name"],
            "ad": stats.get("FlatPhysicalDamageMod", 0),
            "ap": stats.get("FlatMagicDamageMod", 0),
            "hp": stats.get("FlatHPPoolMod", 0),
            "armor": stats.get("FlatArmorMod", 0),
            "mr": stats.get("FlatSpellBlockMod", 0),
            "as": stats.get("PercentAttackSpeedMod", 0),
            "ms": stats.get("FlatMovementSpeedMod", 0) + stats.get("PercentMovementSpeedMod", 0) * 3.5,
            "crit": stats.get("FlatCritChanceMod", 0),
            "lifesteal": stats.get("PercentLifeStealMod", 0)
        }
        
        # 2. Passive DNA (Flagging keywords in description)
        for key, words in PASSIVE_KEYWORDS.items():
            dna[key] = 1 if any(word.lower() in desc.lower() for word in words) else 0

        # Only keep items with stats OR passives
        if sum(list(dna.values())[2:]) > 0:
            rows.append(dna)
            
    df = pd.DataFrame(rows)
    dna_map = df.set_index("id").to_dict(orient="index")
    with open(os.path.join(DATA_DIR, "item_dna.json"), "w") as f:
        json.dump(dna_map, f, indent=4)
    
    print(f"[*] Saved Enhanced DNA (Stats + Passives) for {len(df)} items.")

if __name__ == "__main__":
    extract_dna()

