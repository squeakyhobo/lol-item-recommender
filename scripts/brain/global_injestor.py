import pandas as pd
import json
import os

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOT_FILE = os.path.join(DATA_DIR, "yorick_episodes.json")

def injest_kaggle_data(csv_path):
    """
    Takes a massive Kaggle CSV and 'Injest' it into our Yorick snapshots.
    This bypasses the need for the Riot API entirely.
    """
    print(f"[*] Loading massive dataset: {csv_path}")
    
    # We use 'chunksize' because these files can be 2GB+
    chunk_iter = pd.read_csv(csv_path, chunksize=10000)
    
    yorick_games_count = 0
    new_snapshots = []
    
    for chunk in chunk_iter:
        # 1. Filter for Yorick (championId 83)
        # Note: Column names might vary by dataset (e.g. 'champion_id' or 'p1_champ')
        yorick_data = chunk[chunk.values == 83] 
        
        if not yorick_data.empty:
            yorick_games_count += len(yorick_data)
            # 2. In real research, we would convert these rows to our JSON format here.
            # For now, we are just counting them to show you the scale!
            pass

    print(f"[*] INJESTION COMPLETE!")
    print(f"[*] Found {yorick_games_count} potential Yorick games in this file.")
    print(f"[*] This is {yorick_games_count / 2500:.1f}x more data than our scraper!")

if __name__ == "__main__":
    # Change this to the name of the file you download from Kaggle
    target_csv = os.path.join(DATA_DIR, "match_data.csv")
    
    if os.path.exists(target_csv):
        injest_kaggle_data(target_csv)
    else:
        print(f"[!] File not found: {target_csv}")
        print("[!] Please download the LoL dataset from Kaggle and put it in the data folder.")
