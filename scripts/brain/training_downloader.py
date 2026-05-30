import os
import requests
import json
import time
from dotenv import load_dotenv

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# Support multiple API keys for 10x faster scraping
# e.g., in .env: RIOT_API_KEYS="RGAPI-111,RGAPI-222,RGAPI-333"
raw_keys = os.getenv("RIOT_API_KEYS", os.getenv("RIOT_API_KEY", ""))
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

ROUTING_MAP = {
    "kr": "asia", "jp1": "asia",
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe"
}

class SeededYorickDownloader:
    def __init__(self, keys):
        if not keys:
            raise ValueError("No API keys provided!")
        self.keys = keys
        self.current_key_idx = 0
        
        self.match_dir = os.path.join(BASE_DIR, "data", "yorick_games", "matches")
        self.timeline_dir = os.path.join(BASE_DIR, "data", "yorick_games", "timelines")
        os.makedirs(self.match_dir, exist_ok=True)
        os.makedirs(self.timeline_dir, exist_ok=True)
        
        # Expanded seeds to hit the 2500 goal
        self.seeds = [
            'Kampsycho#ZzRot', 'unfair12262023#asdf', 'Ratz chef#Los', 'HandHolder77#77777', 
            'Pulgo#NA1', 'Nyjin#80011', 'Tzuyu#Chu', 'squee#tech', 'operat0r#NA1', 
            'Ghouls abuser#NA1', 'OldManNAYNAY#Odysy', 'Sparky#H350', 'MJBlee#SENPI', 
            'Stoney#2311', 'deafened#002', 'SomQuite#NA1', 'alwaysdeafened#mute', 
            'Bananaman1102#NA1', 'Ndidracian#9528', 'Nico#NA5', 'Excorpse#NA1', 
            'Ruzuzu#Kymmi', 'Toojuicebox#NA1', 'Vladimatt#ghoul', 'bEhumble#cantc', 
            'Tilted Toplaner#Hope', 'playtowin#123', 'Suvomaesy#NA1', 'Spruce Prune#NA1', 
            'DrunkenWaffle#NA1', 'M00SE#Berry', 'SlimeRock#NA2', 'Yorickless#Yoric', 
            'Shiny Yorick#2705', 'Kaladyn#CP3K9', 'Crack America#crack', 'HaveDownSyndra#7267', 
            'chatisoff#anti', 'goofy foot#333', 'aiiyL#NA1', 'SenseiTurtle#NA1', 'Mai#pee', 
            'JcTheMan#WoW', 'Daze#Murky', 'BlackLightnings#NA1', 'Aivas#NA1', 'renel7#Monke', 
            'HullbreakEnjoyer#Meow', 'ZombieFro#NA1', 'I am Súnlight#NA1'
        ]

    def get_headers(self):
        return {"X-Riot-Token": self.keys[self.current_key_idx]}

    def rotate_key(self):
        """Switches to the next API key when rate limited."""
        if len(self.keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
            print(f"\n[!] Rate Limited! Rotating to API Key #{self.current_key_idx + 1}...")
        else:
            print(f"\n[!] Rate Limited! Only 1 key available. Sleeping 10s...")
            time.sleep(10)

    def make_request(self, url):
        """Helper to handle requests and key rotation."""
        for _ in range(len(self.keys) + 1): # Try each key once, plus a sleep retry
            r = requests.get(url, headers=self.get_headers())
            
            # If we have multiple keys, we don't need to sleep 1.2s! We can go much faster.
            sleep_time = 0.05 if len(self.keys) > 2 else 1.2
            time.sleep(sleep_time) 
            
            if r.status_code == 429:
                self.rotate_key()
                continue
            return r
        return None

    def get_puuid_by_riot_id(self, game_name, tag_line):
        url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        r = self.make_request(url)
        if r and r.status_code == 200:
            return r.json().get("puuid")
        return None

    def get_active_region_for_puuid(self, puuid):
        for routing in ["americas", "europe", "asia"]:
            url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?count=1"
            r = self.make_request(url)
            if r and r.status_code == 200 and len(r.json()) > 0:
                return routing
        return "americas" # fallback

    def get_match_ids(self, puuid, routing, count=100):
        url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count={count}"
        r = self.make_request(url)
        return r.json() if r and r.status_code == 200 else []

    def download_match(self, m_id, routing):
        m_path = os.path.join(self.match_dir, f"{m_id}.json")
        t_path = os.path.join(self.timeline_dir, f"{m_id}.json")
        
        # PREVENTS DUPLICATES: Checks if file exists before downloading
        if os.path.exists(m_path) and os.path.exists(t_path): 
            return False

        # Match Info
        m_url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{m_id}"
        r = self.make_request(m_url)
        if not r or r.status_code != 200: return False
        
        match_data = r.json()
        participants = match_data.get("info", {}).get("participants", [])
        is_yorick_game = any(p.get("championName") == "Yorick" and p.get("teamPosition") == "TOP" for p in participants)
        if not is_yorick_game: return False

        with open(m_path, "w") as f: json.dump(match_data, f)
            
        # Timeline
        t_url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{m_id}/timeline"
        t_r = self.make_request(t_url)
        if t_r and t_r.status_code == 200:
            with open(t_path, "w") as f: json.dump(t_r.json(), f)
            
            # Count files in the folder for a real-time progress update
            total_on_disk = len(os.listdir(self.match_dir))
            print(f"[*] SECURED: {m_id} | Total Yorick Games: {total_on_disk}")
            return True
        return False

    def run(self):
        print(f"[*] Launching MULTI-KEY Scrape with {len(self.keys)} API Keys...")
        
        total_found = 0
        for seed in self.seeds:
            if "#" not in seed: continue
            name, tag = seed.split("#", 1)
            print(f"\n[*] Processing Seed: {name}#{tag}")
            
            puuid = self.get_puuid_by_riot_id(name, tag)
            if not puuid: continue
                
            routing = self.get_active_region_for_puuid(puuid)
            matches = self.get_match_ids(puuid, routing, count=100) # Increased to 100 matches per player!
            
            print(f"  -> Found {len(matches)} recent ranked games. Downloading...")
            for m in matches:
                if self.download_match(m, routing):
                    total_found += 1
                    
                    if total_found >= 2500:
                        print(f"\n[*] Reached massive 2500 match milestone!")
                        return
                    
        print(f"\n[*] Scrape complete! Successfully downloaded {total_found} pure Yorick matches.")

if __name__ == "__main__":
    if not API_KEYS:
        print("[!] Please set your Riot API Key in the .env file.")
    else:
        SeededYorickDownloader(API_KEYS).run()