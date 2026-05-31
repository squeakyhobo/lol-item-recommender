import pandas as pd
import json
import os

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")

def extract_pro_benchmarks():
    # We will use the 2025/2026 data
    files = [
        os.path.join(DATA_DIR, "2025_LoL_esports_match_data_from_OraclesElixir.csv"),
        os.path.join(DATA_DIR, "2026_LoL_esports_match_data_from_OraclesElixir.csv")
    ]
    
    all_pro_top_winners = []
    
    for f in files:
        if not os.path.exists(f): continue
        print(f"[*] Processing: {os.path.basename(f)}")
        
        # Load the data
        df = pd.read_csv(f)
        
        # 1. Filter for TOP lane players who WON their game
        winners = df[(df['position'] == 'top') & (df['result'] == 1)]
        all_pro_top_winners.append(winners)
    
    final_df = pd.concat(all_pro_top_winners)
    
    # 2. Calculate the "Winning Pace" averages
    benchmarks = {
        "minute_10": {
            "avg_gold": float(final_df['goldat10'].mean()),
            "avg_cs": float(final_df['csat10'].mean()),
            "avg_xp": float(final_df['xpat10'].mean())
        },
        "minute_15": {
            "avg_gold": float(final_df['goldat15'].mean()),
            "avg_cs": float(final_df['csat15'].mean()),
            "avg_xp": float(final_df['xpat15'].mean())
        },
        "global": {
            "vision_score_per_min": float(final_df['vspm'].mean()),
            "turret_plates_avg": float(final_df['turretplates'].mean()),
            "damage_per_min": float(final_df['dpm'].mean())
        }
    }
    
    # 3. Save to JSON for the smolagent to read
    output_path = os.path.join(DATA_DIR, "pro_benchmarks.json")
    with open(output_path, "w") as out:
        json.dump(benchmarks, out, indent=4)
        
    print(f"\n[+] SUCCESS: Extracted pro benchmarks from {len(final_df)} winning games.")
    print(f"[*] Gold Standard @ 10m: {benchmarks['minute_10']['avg_gold']:.0f} Gold")
    print(f"[*] CS Standard @ 15m: {benchmarks['minute_15']['avg_cs']:.0f} CS")

if __name__ == "__main__":
    extract_pro_benchmarks()
