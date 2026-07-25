import os
from dotenv import load_dotenv
import json

# 1. Dynamically calculate BASE_DIR (root of the project)
# This file is in LTA/src/core/config.py, so we go up two levels to LTA/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Load .env from the root
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# 3. Environment Variables (Central Source of Truth)
MODEL_VERSION = "V2.8.1_moe"
USE_AUGMENTED_DATA = os.getenv("USE_AUGMENTED_DATA", "False").lower() == "true"
RIOT_API_KEYS = os.getenv("RIOT_API_KEYS", "").split(",")
RIOT_API_KEYS = [k.strip() for k in RIOT_API_KEYS if k.strip()]

# 4. Global Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

V2_EPISODES_PATH = os.path.join(DATA_DIR, "v2_episodes.json")
V2_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_VERSION) 

RUNE_MAP = {
    8005: 1,   # Press the Attack (PTA)
    8008: 2,   # Lethal Tempo
    8021: 3,   # Fleet Footwork
    8010: 4,   # Conqueror
    8112: 5,   # Electrocute
    8128: 6,   # Dark Harvest
    9923: 7,   # Hail of Blades
    8214: 8,   # Summon Aery
    8229: 9,   # Arcane Comet
    8230: 10,  # Phase Rush
    8437: 11,  # Grasp of the Undying
    8439: 12,  # Aftershock
    8465: 13,  # Guardian
    8351: 14,  # Glacial Augment
    8360: 15,  # Unsealed Spellbook
    8369: 16,  # First Strike (Standard ID)
    
}
RUNE_NAMES = {
    1: "Press the Attack",
    2: "Lethal Tempo",
    3: "Fleet Footwork",
    4: "Conqueror",
    5: "Electrocute",
    6: "Dark Harvest",
    7: "Hail of Blades",
    8: "Aery",
    9: "Comet",
    10: "Phase Rush",
    11: "Grasp",
    12: "Aftershock",
    13: "Guardian",
    14: "Glacial Augment",
    15: "Unsealed Spellbook",
    16: "First Strike"
}

# 6. Global Verified Champion Core Items

with open(os.path.join(DATA_DIR, "item_vocab.json"), "r") as f:
            v = json.load(f)
            CORE_MAP = v.get("core_map", {})

NUM_EPOCHS = 60

def ensure_dirs():
    """Utility to make sure project folders exist."""
    for d in [DATA_DIR, AUGMENTED_DIR, MODEL_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

def calculate_threat_score(gold, level, tags):
    """Calculates the standardized threat score for an enemy champion."""
    is_aa = tags.get("is_aa_heavy", 0)
    is_healer = tags.get("is_healer", 0)
    is_cc = tags.get("is_cc_heavy", 0)
    is_tanky = tags.get("is_tanky", 0)
    is_burst = tags.get("is_burst_threat", 0)
    is_ranged = tags.get("is_ranged", 0)
    
    # Exponential level scaling (higher levels are significantly scarier)
    level_threat = (level ** 1.2) * 0.2
    
    # Gold scaling
    gold_threat = (gold / 1000.0) * 1.5
    
    # Class weights: Burst/Assassins are terrifying, Tanks are lower base threat
    class_threat = (is_burst * 2.5) + (is_aa * 1.5) + (is_ranged * 1.2) + (is_cc * 1.0) + (is_healer * 1.0) + (is_tanky * 0.5)
    
    # Synergy: Assassins with a lot of gold are exponentially more dangerous
    synergy_threat = 0
    if is_burst and gold > 3000:
        synergy_threat = (gold / 1000.0) * 0.5

    return gold_threat + level_threat + class_threat + synergy_threat

# 7. Global Supported Champions Mapping
# Maps proper champion names to their Riot internal IDs
TARGET_CHAMPS = {
    "Illaoi": 420,
    #"Shen": 98,
    "Aatrox": 266,
    #"Garen": 86,
    "Mordekaiser": 82,
    #"Trundle": 48,
    "DrMundo": 36,
    #"Sett": 875,
    "Yorick": 83,
    #"Renekton": 58,
    "Darius":122,
    #"Jax": 24,
    #"Malphite": 54,
    "Jayce": 126,
    #"Zaahen": 904, 
    "Ambessa": 799,
    #"Camille": 164,
    #"Chogath": 31,
    #"Kled": 240,
    #"Gnar": 150,
    "Swain": 50,
    #"Rumble": 68,
    #"Warwick": 19,
    #"Urgot": 6,
    #"Volibear": 106,
    #"Kennen": 85,
    #"Pantheon": 80,
    #"Tryndamere": 23,
    #"tahmkench":223
}
# use to cut off /mask items used less than that percent - 
#Note - could do this for core items and have a cut off for what makes a core item
THRESHOLD_PERCENTAGE =0.008 
CORE_THRESHOLD_PERCENTAGE =0.2
