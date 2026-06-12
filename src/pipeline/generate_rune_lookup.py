import json
import os
import sys

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
import config

DATA_DIR = config.DATA_DIR
MATCH_DIR = os.path.join(DATA_DIR, "yorick_games", "matches")

RUNE_MAP = config.RUNE_MAP
RUNE_NAMES = config.RUNE_NAMES

STYLE_NAMES = {
    8100: "Domination",
    8000: "Precision",
    8400: "Resolve",
    8200: "Sorcery",
    8300: "Inspiration"
}

def generate_lookup():
    print("[*] Generating Full Rune Page Lookup Table from raw matches...")
    
    matchup_counts = {}
    
    files = [f for f in os.listdir(MATCH_DIR) if f.endswith(".json")]
    
    for f_name in files:
        try:
            with open(os.path.join(MATCH_DIR, f_name), "r") as f:
                data = json.load(f)
            
            info = data.get("info", {})
            participants = info.get("participants", [])
            
            # 1. Find Yorick and the Lane Opponent
            yorick = None
            enemy_team = None
            for p in participants:
                if p.get("championName") == "Yorick":
                    yorick = p
                    enemy_team = 200 if p["teamId"] == 100 else 100
                    break
            
            if not yorick: continue
            
            matchup = "Unknown"
            for p in participants:
                if p["teamId"] == enemy_team and p.get("teamPosition") == "TOP":
                    matchup = p.get("championName", "Unknown").lower()
                    break
            
            # 2. Extract Rune Page
            perks = yorick.get("perks", {})
            styles = perks.get("styles", [])
            
            keystone_id = 0
            secondary_style_id = 0
            
            for s in styles:
                if s.get("description") == "primaryStyle":
                    keystone_id = s.get("selections", [{}])[0].get("perk", 0)
                elif s.get("description") == "subStyle":
                    secondary_style_id = s.get("style", 0)
            
            # 3. Record Data
            if matchup not in matchup_counts:
                matchup_counts[matchup] = {"total": 0, "keystones": {}, "secondaries": {}}
            
            m = matchup_counts[matchup]
            m["total"] += 1
            
            # Record Keystone
            k_name = RUNE_NAMES.get(RUNE_MAP.get(keystone_id, 0), "Unknown")
            m["keystones"][k_name] = m["keystones"].get(k_name, 0) + 1
            
            # Record Secondary
            s_name = STYLE_NAMES.get(secondary_style_id, "Unknown")
            m["secondaries"][s_name] = m["secondaries"].get(s_name, 0) + 1
            
        except: pass

    # 4. Process Votes and Percentages
    final_lookup = {}
    for matchup, stats in matchup_counts.items():
        total = stats["total"]
        if total < 2: continue # Ignore rare data
        
        # Calculate Keystone %
        k_stats = {}
        for name, count in stats["keystones"].items():
            k_stats[name] = f"{int((count/total)*100)}%"
        
        # Calculate Secondary %
        s_stats = {}
        for name, count in stats["secondaries"].items():
            s_stats[name] = f"{int((count/total)*100)}%"
            
        # Determine Winners
        best_keystone = max(stats["keystones"], key=stats["keystones"].get)
        best_secondary = max(stats["secondaries"], key=stats["secondaries"].get)
        
        final_lookup[matchup] = {
            "advice": f"{best_keystone} with {best_secondary} secondary",
            "keystone_stats": k_stats,
            "secondary_stats": s_stats,
            "total_games": total
        }

    # 5. Default Fallback
    final_lookup["unknown"] = {
        "advice": "Grasp with Inspiration secondary",
        "keystone_stats": {"Grasp": "100%"},
        "secondary_stats": {"Inspiration": "100%"},
        "total_games": 0
    }
    
    with open(os.path.join(DATA_DIR, "rune_lookup.json"), "w") as f:
        json.dump(final_lookup, f, indent=4)
        
    print(f"[SUCCESS] Generated full rune lookup for {len(final_lookup)-1} matchups.")
    print(f"[*] Saved to: {os.path.join(DATA_DIR, 'rune_lookup.json')}")

if __name__ == "__main__":
    generate_lookup()
