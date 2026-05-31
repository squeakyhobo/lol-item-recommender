import os
import requests
import json
import time
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: On Linux/Cloud, we look for .env in the same folder or root
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

raw_keys = os.getenv("RIOT_API_KEYS", os.getenv("RIOT_API_KEY", ""))
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

class CloudYorickHunter:
    """
    The Ultimate Persistent Cloud Scraper.
    Designed to run 24/7 on DigitalOcean.
    """
    def __init__(self, keys):
        self.keys = keys
        self.current_key_idx = 0
        
        # Local folders on the server
        self.data_dir = "data"
        self.match_dir = os.path.join(self.data_dir, "yorick_games", "matches")
        self.timeline_dir = os.path.join(self.data_dir, "yorick_games", "timelines")
        self.checkpoint_path = os.path.join(self.data_dir, "hunter_checkpoint.json")
        
        os.makedirs(self.match_dir, exist_ok=True)
        os.makedirs(self.timeline_dir, exist_ok=True)
        
        self.processed_seeds = set()
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    self.processed_seeds = set(json.load(f))
            except: pass
        
        # Initial Seeds
        self.seeds = ['철 릭#KR1', 'KC NEXT CACAKING#PX01', 'Ndidracian#9528'] # Add more here

    def get_headers(self):
        return {"X-Riot-Token": self.keys[self.current_key_idx]}

    def rotate_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
        print(f"[*] Rotating to Key #{self.current_key_idx + 1}")

    def make_request(self, url):
        for _ in range(len(self.keys) + 1):
            r = requests.get(url, headers=self.get_headers())
            time.sleep(0.05 if len(self.keys) > 2 else 1.2)
            if r.status_code == 429:
                self.rotate_key()
                continue
            return r
        return None

    def download_match(self, m_id, routing):
        m_path = os.path.join(self.match_dir, f"{m_id}.json")
        # --- DEDUPLICATION ---
        # If the file exists on the cloud server, we don't download it again
        if os.path.exists(m_path): return False

        r = self.make_request(f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{m_id}")
        if not r or r.status_code != 200: return False
        
        data = r.json()
        participants = data.get("info", {}).get("participants", [])
        if any(p.get("championName") == "Yorick" and p.get("teamPosition") == "TOP" for p in participants):
            with open(m_path, "w") as f: json.dump(data, f)
            # Also get timeline
            tr = self.make_request(f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{m_id}/timeline")
            if tr and tr.status_code == 200:
                with open(os.path.join(self.timeline_dir, f"{m_id}.json"), "w") as f: json.dump(tr.json(), f)
            print(f"[+] Hunter Secured: {m_id}")
            return True
        return False

    def run_forever(self):
        print(f"[*] Cloud Hunter active with {len(self.keys)} keys. Targeting Yorick Mastery.")
        
        while True:
            for seed in self.seeds:
                if seed in self.processed_seeds: continue
                
                print(f"[*] Processing: {seed}")
                name, tag = seed.split("#")
                # 1. Get PUUID
                url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
                r = self.make_request(url)
                if not r or r.status_code != 200: continue
                puuid = r.json().get("puuid")
                
                # 2. Get Matches
                for region in ["americas", "europe", "asia"]:
                    matches = self.make_request(f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?count=100")
                    if matches and matches.status_code == 200:
                        for m in matches.json():
                            self.download_match(m, region)
                
                # 3. Save Checkpoint
                self.processed_seeds.add(seed)
                with open(self.checkpoint_path, "w") as f:
                    json.dump(list(self.processed_seeds), f)
            
            print("[*] Cycle complete. Sleeping 1 hour before re-scanning...")
            time.sleep(3600)

if __name__ == "__main__":
    if API_KEYS:
        CloudYorickHunter(API_KEYS).run_forever()
    else:
        print("No keys found in .env")
