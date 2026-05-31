import os
import json
from collections import Counter

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
MATCH_DIR = os.path.join(BASE_DIR, "data", "yorick_games", "matches")

def analyze_matchup_history():
    print("🧟 YORICK MATCHUP ANALYST (The Historical Log)")
    print("-" * 50)
    
    match_files = [f for f in os.listdir(MATCH_DIR) if f.endswith(".json")]
    print(f"[*] Scanning {len(match_files)} professional matches...")
    
    matchups = {}
    
    for f in match_files:
        try:
            with open(os.path.join(MATCH_DIR, f), "r") as m:
                data = json.load(m)
                
                # 1. Find Yorick and his Opponent
                participants = data.get("info", {}).get("participants", [])
                yorick = next((p for p in participants if p.get("championName") == "Yorick" and p.get("teamPosition") == "TOP"), None)
                if not yorick: continue
                
                opponent = next((p for p in participants if p.get("teamPosition") == "TOP" and p.get("teamId") != yorick.get("teamId")), None)
                if not opponent: continue
                
                opp_name = opponent.get("championName")
                win = 1 if yorick.get("win") else 0
                
                # 2. Track stats
                if opp_name not in matchups:
                    matchups[opp_name] = {"games": 0, "wins": 0, "items": []}
                
                matchups[opp_name]["games"] += 1
                matchups[opp_name]["wins"] += win
                # Track what yorick bought
                matchups[opp_name]["items"].extend([item.get("itemID") for item in yorick.get("items", []) if item.get("itemID")])
                
        except: continue

    # 3. Print Results
    print(f"{'OPPONENT':<15} | {'GAMES':<6} | {'WIN RATE':<6} | {'POPULAR ITEMS'}")
    print("-" * 60)
    
    # Sort by games played
    sorted_matchups = sorted(matchups.items(), key=lambda x: x[1]['games'], reverse=True)
    
    for name, stats in sorted_matchups[:15]:
        wr = (stats["wins"] / stats["games"]) * 100
        common_items = [i[0] for i in Counter(stats["items"]).most_common(3)]
        print(f"{name:<15} | {stats['games']:<6} | {wr:>5.1f}%  | {common_items}")

    print("\n[+] Analysis Complete. Use this log to decide your counter-picks!")

if __name__ == "__main__":
    analyze_matchup_history()
