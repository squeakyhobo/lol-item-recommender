import requests
import json
import os
import sys
import urllib3
import time

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def inspect_api():
    url = "https://127.0.0.1:2999/liveclientdata/allgamedata"
    
    print("="*60)
    print(" LTA API INSPECTOR - Live Client Data Snapshot")
    print("="*60)
    print("[*] Connecting to League of Legends Live API...")
    
    try:
        response = requests.get(url, verify=False, timeout=3)
        if response.status_code != 200:
            print(f"[!] API Error: Status {response.status_code}")
            return
            
        data = response.json()
        
        # SAVE RAW DATA FOR INSPECTION
        dump_path = "live_api_dump.json"
        with open(dump_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[+] FULL RAW DATA saved to: {dump_path}")
        
        # 1. GAME CONTEXT
        game_data = data.get("gameData", {})
        print(f"\n[1] GAME INFO")
        print(f"  -> Time: {game_data.get('gameTime', 0)/60:.2f} minutes")
        print(f"  -> Map:  {game_data.get('mapName', 'Unknown')} (ID: {game_data.get('mapId', 0)})")

        # 2. ACTIVE PLAYER (YOU)
        active = data.get("activePlayer", {})
        print(f"\n[2] ACTIVE PLAYER (YOU)")
        print(f"  -> Name: {active.get('summonerName', 'Unknown')}")
        print(f"  -> Gold: {active.get('currentGold', 0):.0f}")
        # Note: totalGold in stats is often 0 here!
        stats = active.get("championStats", {})
        print(f"  -> Stats (Raw): AD: {stats.get('attackDamage', 0):.1f}, AP: {stats.get('abilityPower', 0):.1f}, HP: {stats.get('currentHealth', 0):.0f}/{stats.get('maxHealth', 0):.0f}")

        # 3. ALL PLAYERS
        all_players = data.get("allPlayers", [])
        print(f"\n[3] PLAYERS & INVENTORIES ({len(all_players)} found)")
        for p in all_players:
            team = p.get("team")
            role = p.get("teamPosition", "UNKNOWN")
            items = [f"{i.get('displayName')} ({i.get('itemID')})" for i in p.get("items", [])]
            print(f"  [{team}] {p.get('championName'):<15} | Lvl {p.get('level'):<2} | {role:<10}")
            if items:
                print(f"    -> Items: {', '.join(items[:3])}...")
            else:
                print(f"    -> Items: [EMPTY]")

        # 4. RAW SNAPSHOT (First 20 lines)
        print(f"\n[4] RAW DATA PREVIEW (First 30 lines)")
        print("-" * 30)
        raw_lines = json.dumps(data, indent=4).splitlines()
        for line in raw_lines:
            print(line)
        print("... (rest of JSON truncated) ...")

    except requests.exceptions.ConnectionError:
        print("\n[!] ERROR: Could not connect to League of Legends.")
        print("    Make sure:")
        print("    1. You are currently in a match (even a Custom Game).")
        print("    2. The League Client is running.")
    except Exception as e:
        print(f"\n[!] Unexpected Error: {e}")

if __name__ == "__main__":
    inspect_api()
