import time
import subprocess
import os
import requests
import urllib3
import sys
import pyttsx3
import json

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
import config

DATA_DIR = config.DATA_DIR

def find_lcu_access():
    """Finds the Riot lockfile to get port and password."""
    paths = [r"C:\Riot Games\League of Legends\lockfile", r"D:\Riot Games\League of Legends\lockfile"]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                parts = f.read().split(':')
                return {"port": parts[2], "password": parts[3], "url": f"https://127.0.0.1:{parts[2]}"}
    return None

class Watcher:
    def __init__(self):
        self.active_process = None
        self.current_mode = None
        self.lobby_matchup = None
        
        # Voice Engine
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 190)
        
        # Load Knowledge Base for names
        kb_path = os.path.join(DATA_DIR, "champion_knowledge.json")
        with open(kb_path, "r") as f:
            self.kb = json.load(f)

    def speak(self, text):
        print(f"[VOICE]: {text}")
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except: pass

    def get_enemy_names(self, access):
        """Polls the LCU for current enemy team in Champ Select."""
        try:
            r = requests.get(f"{access['url']}/lol-champ-select/v1/session", auth=('riot', access['password']), verify=False)
            if r.status_code != 200: return []
            data = r.json()
            enemies = []
            for p in data.get("theirTeam", []):
                cid = str(p.get("championId", 0))
                # Cross-reference ID with knowledge base to get name
                found_name = "Unknown"
                for name, info in self.kb.items():
                    if info["id"] == cid:
                        found_name = name.capitalize()
                        break
                if cid != "0":
                    enemies.append({"id": cid, "name": found_name})
            return enemies
        except: return []

    def get_my_champ(self, access):
        """Detects which champion the local player has locked in."""
        try:
            r = requests.get(f"{access['url']}/lol-champ-select/v1/session", auth=('riot', access['password']), verify=False)
            if r.status_code != 200: return "Yorick" # Default
            data = r.json()
            local_cell_id = data.get("localPlayerCellId")
            for p in data.get("myTeam", []):
                if p.get("cellId") == local_cell_id:
                    cid = str(p.get("championId", 0))
                    for name, info in self.kb.items():
                        if info["id"] == cid:
                            return name.capitalize()
            return "Yorick"
        except: return "Yorick"

    def get_expert_rune_advice(self, champion, matchup):
        """Uses the multi-champ lookup table for tailored rune advice."""
        lookup_path = os.path.join(DATA_DIR, "rune_lookup.json")
        if not os.path.exists(lookup_path):
            return "Grasp (Default)", {}
            
        with open(lookup_path, "r") as f:
            lookup_table = json.load(f)
            
        # Get the specific sub-table for the current champion
        champ_key = champion.lower()
        champ_lookup = lookup_table.get(champ_key, lookup_table.get("yorick", {}))
        
        data = champ_lookup.get(matchup.lower(), champ_lookup.get("unknown", {}))
        
        # Format stats for printing
        stats_text = f"\n  [{champion} vs {matchup} stats from {data.get('total_games', 0)} games]\n"
        stats_text += f"  Keystones: {', '.join([f'{k}({v})' for k,v in data.get('keystone_stats', {}).items()])}\n"
        stats_text += f"  Secondary: {', '.join([f'{k}({v})' for k,v in data.get('secondary_stats', {}).items()])}\n"
        
        return data.get("advice", "Grasp"), stats_text

    def start_hud(self):
        """Deploys the V2.1 Maiden Brain HUD."""
        self.stop_active()
        self.current_mode = "HUD"
        script = os.path.join(BASE_DIR, "src", "core", "inference_hud.py")
        cmd = ["py", "-3.12", script]
        print("[*] DEPLOYING LIVE HUD...")
        self.active_process = subprocess.Popen(cmd)

    def stop_active(self):
        if self.active_process:
            self.active_process.terminate()
            self.active_process = None

    def run(self):
        print("="*60)
        print(" MAIDEN WATCHER V2.1 - Lobby & Gameflow Monitor")
        print("="*60)
        self.speak("working...")
        while True:
            access = find_lcu_access()
            if not access:
                time.sleep(10); continue

            try:
                r = requests.get(f"{access['url']}/lol-gameflow/v1/session", auth=('riot', access['password']), verify=False)
                phase = r.json().get('phase', 'None')
                
                # MODE 1: LOBBY (PREGAME)
                if phase == "ChampSelect" and self.current_mode != "PREGAME":
                    self.current_mode = "PREGAME"
                    print("[+] LOBBY DETECTED. Analyzing enemies...")
                    
                    while self.current_mode == "PREGAME":
                        enemies = self.get_enemy_names(access)
                        if len(enemies) == 5:
                            print("\n--- ENEMY TEAM ---")
                            for idx, e in enumerate(enemies):
                                print(f" [{idx+1}] {e['name']}")
                            
                            print("\n[?] Who is the TOP LANER? (1-5, or 0 to skip):")
                            try:
                                ui = input(">> ").strip()
                                if ui and ui != "0":
                                    top_idx = int(ui) - 1
                                    self.lobby_matchup = enemies[top_idx]["name"]
                                    
                                    # New: Multi-Champ Rune Advice
                                    my_champ = self.get_my_champ(access)
                                    rune_advice, rune_stats = self.get_expert_rune_advice(my_champ, self.lobby_matchup)
                                    
                                    print(f"\n--- {my_champ.upper()} EXPERT ADVICE ---")
                                    print(rune_stats)
                                    self.speak(f"Playing {my_champ} against {self.lobby_matchup}. Use {rune_advice}.")
                            except: pass
                            break 
                        
                        time.sleep(3)
                        # Check if we left lobby
                        rp = requests.get(f"{access['url']}/lol-gameflow/v1/session", auth=('riot', access['password']), verify=False)
                        if rp.json().get('phase') != "ChampSelect": break
                
                # MODE 2: MATCH START (HUD)
                elif phase == "InProgress" and self.current_mode != "HUD":
                    self.speak("Match started. Loading Brain V2.1.")
                    self.start_hud()
                
                # MODE 3: POST-GAME (CLEANUP)
                elif phase in ["None", "Lobby", "Matchmaking"] and self.current_mode is not None:
                    print("[*] Game ended. Resetting Watcher.")
                    self.stop_active()
                    self.current_mode = None
                    self.lobby_matchup = None

            except Exception as e:
                # Silently catch API errors (Riot API is jittery)
                pass

            time.sleep(5)

if __name__ == "__main__":
    watcher = Watcher()
    try:
        watcher.run()
    except KeyboardInterrupt:
        watcher.stop_active()
        print("\n[*] Watcher terminated.")