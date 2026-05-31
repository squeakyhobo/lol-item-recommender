import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# DYNAMIC BASE_DIR for Cloud/Local compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "yorick_games")
MATCH_DIR = os.path.join(DATA_DIR, "matches")
TIMELINE_DIR = os.path.join(DATA_DIR, "timelines")

os.makedirs(MATCH_DIR, exist_ok=True)
os.makedirs(TIMELINE_DIR, exist_ok=True)

def harvest_ultimate_yorick():
    """
    Connects to the GPTilt ULTIMATE Archive (10 Million Events).
    """
    print("[*] STEP 1: Loading Yorick Match IDs from previous scan...")
    target_list_path = os.path.join(BASE_DIR, "data", "hf_target_matches.json")
    if not os.path.exists(target_list_path):
        print("[!] Error: No target matches found.")
        return

    with open(target_list_path, "r") as f:
        target_ids = set(json.load(f))
    
    print(f"[*] Found {len(target_ids)} target Challenger Yorick games.")

    print("[*] STEP 2: Connecting to ULTIMATE Archive (10m Enriched Events)...")
    splits = [
        'train_region_americas', 'train_region_asia', 'train_region_europe',
        'test_region_americas', 'test_region_asia', 'test_region_europe'
    ]
    
    for split in splits:
        print(f"\n[*] Processing ULTIMATE events for {split}...")
        try:
            ds = load_dataset("gptilt/lol-ultimate-events-challenger-10m", split=split, streaming=True)
            found_count = 0
            for row in tqdm(ds, desc=f"Scanning {split}"):
                m_id = row.get("matchId")
                if m_id in target_ids:
                    t_path = os.path.join(TIMELINE_DIR, f"{m_id}_hf.jsonl")
                    with open(t_path, "a") as out:
                        json.dump(row, out)
                        out.write("\n")
                    found_count += 1
            print(f"[+] Split {split}: Secured {found_count} records.")
        except Exception as e:
            print(f"[!] Hugging Face Error: {e}")

    print(f"\n[*] HARVEST COMPLETE!")

if __name__ == "__main__":
    harvest_ultimate_yorick()
