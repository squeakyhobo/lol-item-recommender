import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data", "yorick_games")
MATCH_DIR = os.path.join(DATA_DIR, "matches")
TIMELINE_DIR = os.path.join(DATA_DIR, "timelines")

os.makedirs(MATCH_DIR, exist_ok=True)
os.makedirs(TIMELINE_DIR, exist_ok=True)

def harvest_yorick_from_hf():
    """
    Connects to the GPTilt Challenger Archive on Hugging Face.
    Filters 10,000 professional-grade matches for Yorick players.
    Saves them directly into the LTA data pipeline.
    """
    print("[*] Connecting to Hugging Face: gptilt/lol-basic-matches-challenger-10k...")
    
    # Correct splits based on test run
    regions = ['region_americas', 'region_asia', 'region_europe']
    all_yorick_match_ids = []

    try:
        for region in regions:
            print(f"[*] Loading data for {region}...")
            # Load the Participants metadata to find Yorick games (championId 83)
            ds = load_dataset("gptilt/lol-basic-matches-challenger-10k", name="participants", split=region)
            df = ds.to_pandas()
            
            print(f"  -> Scanning {len(df)} participant records...")
            yorick_games = df[df['championId'] == 83]
            match_ids = yorick_games['matchId'].unique().tolist()
            
            print(f"  -> Found {len(match_ids)} Yorick games in {region}!")
            all_yorick_match_ids.extend(match_ids)
        
        print(f"\n[+] GLOBAL SUCCESS: Found {len(all_yorick_match_ids)} Yorick games across all regions!")
        print(f"[*] Sample Match IDs: {all_yorick_match_ids[:10]}")
        
        # Save the list of target IDs for the next phase (bulk download)
        target_list_path = os.path.join(BASE_DIR, "data", "hf_target_matches.json")
        with open(target_list_path, "w") as f:
            json.dump(all_yorick_match_ids, f)
        print(f"[*] Saved target list to: {target_list_path}")
        
    except Exception as e:
        print(f"[!] Hugging Face Error: {e}")

if __name__ == "__main__":
    harvest_yorick_from_hf()
