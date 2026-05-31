import os
import requests
import json
import time
from dotenv import load_dotenv

# DYNAMIC BASE_DIR for Cloud/Local compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

raw_keys = os.getenv("RIOT_API_KEYS", os.getenv("RIOT_API_KEY", ""))
INITIAL_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

class InfiniteYorickHunter:
    """
    The 'Infection' Scraper (V5.0).
    Automatically discovers new Yorick players from every match it finds.
    """
    def __init__(self, keys):
        if not keys: raise ValueError("No API keys provided!")
        self.active_keys = list(keys)
        self.current_key_idx = 0
        
        self.data_dir = os.path.join(BASE_DIR, "data")
        self.match_dir = os.path.join(self.data_dir, "yorick_games", "matches")
        self.timeline_dir = os.path.join(self.data_dir, "yorick_games", "timelines")
        self.checkpoint_path = os.path.join(self.data_dir, "scraper_checkpoint.json")
        self.reservoir_path = os.path.join(self.data_dir, "player_reservoir.json")
        
        os.makedirs(self.match_dir, exist_ok=True)
        os.makedirs(self.timeline_dir, exist_ok=True)
        
        # Load Checkpoint
        self.processed_seeds = set()
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    self.processed_seeds = set(json.load(f))
            except: pass

        # Load Reservoir (Discovered Players)
        self.reservoir = []
        if os.path.exists(self.reservoir_path):
            try:
                with open(self.reservoir_path, "r") as f:
                    self.reservoir = json.load(f)
            except: pass
        
        # Core Elite Seeds
        self.seeds = [
            '철 릭#KR1', '릭 철#KR1', 'Slogdog#OCE', 'Krykey#EUW', 'Pokerick#NA1', 
            'Ninetales#OCE', 'T1 Midir#KR1', 'EWI Vizicsacsi#EUW', 'Ndidracian#9528'
        ]

    def get_headers(self):
        return {"X-Riot-Token": self.active_keys[self.current_key_idx]}

    def rotate_key(self, failed_key=None, status_code=None):
        if status_code in [401, 403]:
            print(f"\n[!] KEY EXPIRED ({status_code})")
            if failed_key in self.active_keys: self.active_keys.remove(failed_key)
        if not self.active_keys: return False
        self.current_key_idx = (self.current_key_idx + 1) % len(self.active_keys)
        return True

    def make_request(self, url):
        while self.active_keys:
            current_key = self.active_keys[self.current_key_idx]
            try:
                r = requests.get(url, headers={"X-Riot-Token": current_key})
                time.sleep(0.05 if len(self.active_keys) > 2 else 1.2)
                if r.status_code == 200: return r
                if r.status_code == 429:
                    if not self.rotate_key(): return None
                elif r.status_code in [401, 403]:
                    if not self.rotate_key(current_key, r.status_code): return None
                else: return r
            except: return None
        return None

    def download_match(self, m_id, routing):
        m_path = os.path.join(self.match_dir, f"{m_id}.json")
        if os.path.exists(m_path): return False

        r = self.make_request(f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{m_id}")
        if not r or r.status_code != 200: return False
        
        data = r.json()
        participants = data.get("info", {}).get("participants", [])
        is_yorick_game = any(p.get("championName") == "Yorick" and p.get("teamPosition") == "TOP" for p in participants)
        
        if is_yorick_game:
            # 🧟 INFECTION LOGIC: Grab all players from this match
            for p in participants:
                p_id = p.get("puuid")
                if p_id and p_id not in self.processed_seeds and p_id not in self.reservoir:
                    self.reservoir.append(p_id)
            
            with open(m_path, "w") as f: json.dump(data, f)
            tr = self.make_request(f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{m_id}/timeline")
            if tr and tr.status_code == 200:
                with open(os.path.join(self.timeline_dir, f"{m_id}.json"), "w") as f: json.dump(tr.json(), f)
            
            print(f"[*] SECURED: {m_id} | Discovered {len(self.reservoir)} potential new players")
            return True
        return False

    def run(self):
        print(f"[*] Launching INFINITE HUNTER with {len(self.active_keys)} Keys...")
        
        while True:
            # 1. Prioritize Main Seeds
            current_target = None
            if self.seeds:
                seed = self.seeds.pop(0)
                if seed in self.processed_seeds: continue
                print(f"[*] Harvesting Seed: {seed}")
                name, tag = seed.split("#")
                for reg in ["americas", "europe", "asia"]:
                    url = f"https://{reg}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
                    r = self.make_request(url)
                    if r and r.status_code == 200:
                        current_target = r.json().get("puuid"); break
            
            # 2. Use Reservoir if no seeds left
            elif self.reservoir:
                current_target = self.reservoir.pop(0)
                print(f"[*] Harvesting Discovered Player: {current_target[:10]}...")

            if not current_target:
                print("[!] No targets left. Sleeping 10 mins...")
                time.sleep(600); continue

            # 3. Harvest history
            for routing in ["americas", "europe", "asia"]:
                matches = self.make_request(f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{current_target}/ids?queue=420&count=100")
                if matches and matches.status_code == 200:
                    for m in matches.json():
                        self.download_match(m, routing)
            
            # 4. Cleanup
            self.processed_seeds.add(current_target)
            with open(self.checkpoint_path, "w") as f: json.dump(list(self.processed_seeds), f)
            with open(self.reservoir_path, "w") as f: json.dump(self.reservoir, f)

if __name__ == "__main__":
    if INITIAL_KEYS:
        InfiniteYorickHunter(INITIAL_KEYS).run()
    else:
        print("Set RIOT_API_KEYS in .env")
