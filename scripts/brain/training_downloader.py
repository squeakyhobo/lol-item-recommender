import os
import requests
import json
import time
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("RIOT_API_KEY")
REGION = "euw1"
ROUTING = "europe"

class MatchDownloader:
    def __init__(self, key):
        self.key = key
        self.headers = {"X-Riot-Token": key}
        self.dir = os.path.join(BASE_DIR, "data", "timelines")
        os.makedirs(self.dir, exist_ok=True)

    def get_challenger_puuids(self, count=50):
        url = f"https://{REGION}.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/CHALLENGER/I"
        r = requests.get(url, headers=self.headers)
        if r.status_code == 200:
            return [p.get("puuid") for p in r.json() if p.get("puuid")][:count]
        return []

    def get_matches(self, puuid, count=5):
        url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?count={count}"
        r = requests.get(url, headers=self.headers)
        return r.json() if r.status_code == 200 else []

    def download_timeline(self, m_id):
        p = os.path.join(self.dir, f"{m_id}.json")
        if os.path.exists(p): return
        url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{m_id}/timeline"
        r = requests.get(url, headers=self.headers)
        if r.status_code == 200:
            with open(p, "w") as f: json.dump(r.json(), f)
            print(f"[*] Saved: {m_id}")
        time.sleep(1.2) # Rate limit management

    def run(self):
        puuids = self.get_challenger_puuids(count=50)
        print(f"[*] Found {len(puuids)} players. Starting 250-match download...")
        for p in puuids:
            for m in self.get_matches(p):
                self.download_timeline(m)

if __name__ == "__main__":
    if API_KEY: MatchDownloader(API_KEY).run()
