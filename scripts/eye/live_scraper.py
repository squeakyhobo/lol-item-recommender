import requests
import json
import time
import urllib3
import os

# Suppress insecure request warnings for the self-signed certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LiveScraper:
    def __init__(self, endpoint="https://127.0.0.1:2999/liveclientdata/allgamedata"):
        self.endpoint = endpoint
        # Use absolute path relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.log_dir = os.path.join(base_dir, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def fetch_data(self):
        try:
            # Riot uses a self-signed cert, so verify=False is necessary
            response = requests.get(self.endpoint, verify=False, timeout=2)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[!] Error: API returned status code {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            print("[!] Connection Error: Is League of Legends running and are you in a match?")
            return None
        except Exception as e:
            print(f"[!] Unexpected error: {e}")
            return None

    def save_log(self, data):
        timestamp = int(time.time())
        filename = os.path.join(self.log_dir, f"game_state_{timestamp}.json")
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[*] Data logged to {filename}")

    def run(self, interval=10):
        print(f"[*] Starting scraper. Polling every {interval} seconds...")
        try:
            while True:
                data = self.fetch_data()
                if data:
                    # Extract active player name as a quick check
                    active_player = data.get("activePlayer", {}).get("summonerName", "Unknown")
                    print(f"[*] Captured state for player: {active_player}")
                    self.save_log(data)
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[*] Scraper stopped by user.")

if __name__ == "__main__":
    scraper = LiveScraper()
    scraper.run()
