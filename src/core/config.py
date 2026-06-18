import os
from dotenv import load_dotenv

# 1. Dynamically calculate BASE_DIR (root of the project)
# This file is in LTA/src/core/config.py, so we go up two levels to LTA/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Load .env from the root
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# 3. Environment Variables (Central Source of Truth)
MODEL_VERSION = os.getenv("MODEL_VERSION")
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

# 5. Shared Constants
RUNE_MAP = {
    8010: 1, 8437: 2, 8229: 3, 8992: 4, 8230: 5, 8112: 6, 8369: 7, 8214: 8, 8008: 9,
    8128: 10, 8021: 11
}
RUNE_NAMES = {
    1: "Conqueror", 2: "Grasp", 3: "Comet", 4: "First Strike", 
    5: "Phase Rush", 6: "Electrocute", 7: "Inspiration", 8: "Aery", 9: "Lethal Tempo",
    10: "Hail of Blades", 11: "Fleet"
}


NUM_EPOCHS = 180

def ensure_dirs():
    """Utility to make sure project folders exist."""
    for d in [DATA_DIR, AUGMENTED_DIR, MODEL_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
