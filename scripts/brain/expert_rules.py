
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

def generate_expert_masks():
    with open(os.path.join(DATA_DIR, "item_names.json"), "r") as f:
        item_names = json.load(f)
    
    # 1. BANNED: Items top lane NEVER buys
    BANNED_KEYWORDS = ["Support", "Wardstone", "Spellthief", "Relic Shield", "Steel Shoulderguards"]
    banned_ids = [int(id) for id, name in item_names.items() if any(word in name for word in BANNED_KEYWORDS)]
    
    # 2. INCOMPATIBILITY GROUPS (Unique Passives)
    # If you own one, you cannot buy the others.
    GROUPS = {
        "LIFELINE": [3053, 3156, 3153], # Sterak, Maw, Shieldbow
        "SPELLBLADE": [3078, 3100, 6662], # Trinity, Lich, Iceborn
        "HYDRA": [3074, 3748, 6692], # Ravenous, Titanic, Profane
        "BOOTS": [1001, 3006, 3009, 3020, 3047, 3111, 3117, 3158] # All final boots
    }
    
    data = {
        "banned_ids": banned_ids,
        "groups": GROUPS
    }
    
    with open(os.path.join(DATA_DIR, "expert_config.json"), "w") as f:
        json.dump(data, f, indent=4)
    
    print("[*] Expert System Configured: Banned items + 4 Passive Groups.")

if __name__ == "__main__":
    generate_expert_masks()

