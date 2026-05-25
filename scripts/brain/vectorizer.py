import json
import os
import numpy as np

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

class LTAVectorizer:
    def __init__(self):
        archetypes_path = os.path.join(DATA_DIR, "archetypes.json")
        try:
            with open(archetypes_path, "r") as f:
                self.archetypes = json.load(f)
        except FileNotFoundError:
            print("[!] Archetypes file not found. Run cluster_champions.py first.")
            self.archetypes = {}

    def vectorize_state(self, log_data):
        active = log_data.get("activePlayer", {})
        stats = active.get("championStats", {})
        
        player_vector = {
            "name": log_data.get("activePlayer", {}).get("summonerName", "Unknown"),
            "gold": active.get("currentGold", 0),
            "level": active.get("level", 0),
            "ad": stats.get("attackDamage", 0),
            "ap": stats.get("abilityPower", 0),
            "armor": stats.get("armor", 0),
            "mr": stats.get("spellblock", 0)
        }

        all_players = log_data.get("allPlayers", [])
        threat_vectors = []
        for p in all_players:
            champ_name = p.get("championName", "")
            archetype_id = self.archetypes.get(champ_name, -1)
            scores = p.get("scores", {})
            items = [item.get("displayName") for item in p.get("items", [])]
            p_vector = {
                "name": champ_name,
                "archetype": archetype_id,
                "kda": f"{scores.get("kills",0)}/{scores.get("deaths",0)}/{scores.get("assists",0)}",
                "cs": scores.get("creepScore", 0),
                "items": items
            }
            threat_vectors.append(p_vector)
        return player_vector, threat_vectors

    def display_tensor_view(self, p_vec, t_vecs):
        print("\n" + "="*50)
        print("--- TRANSFORMER INPUT VIEW (VIRTUAL TENSOR) ---")
        print("="*50)
        print(f"SELF: [Gold: {p_vec["gold"]:>5.0f} | Lvl: {p_vec["level"]:>2} | Armor: {p_vec["armor"]:>3.0f} | MR: {p_vec["mr"]:>3.0f}]")
        print("-" * 50)
        print(f"{"CHAMP":<15} | {"TYPE":<4} | {"K/D/A":<8} | {"CORE ITEMS"}")
        for t in t_vecs:
            items_str = ", ".join(t["items"][:2]) + "..." if t["items"] else "None"
            print(f"{t["name"]:<15} | {t["archetype"]:<4} | {t["kda"]:<8} | {items_str}")
        print("="*50)
        print("[*] Note: The Attention mechanism will now compare your stats")
        print("[*] against the highest KDA archetypes (Types 0-7) in this list.")

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        print(f"[!] Log directory not found at {LOG_DIR}")
    else:
        logs = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith(".json")]
        if not logs:
            print("[!] No logs found. Run scripts/eye/live_scraper.py in-game.")
        else:
            latest_log = max(logs, key=os.path.getctime)
            with open(latest_log, "r") as f:
                data = json.load(f)
            vec = LTAVectorizer()
            p_vec, t_vecs = vec.vectorize_state(data)
            vec.display_tensor_view(p_vec, t_vecs)

