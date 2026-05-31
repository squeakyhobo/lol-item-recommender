
import os
import time
import json
from datetime import datetime

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
TIMELINE_DIR = os.path.join(BASE_DIR, "data", "timelines")
PROGRESS_FILE = os.path.join(BASE_DIR, "logs", "training_progress.json")
TARGET_MATCHES = 1000

def get_dashboard():
    match_count = len([f for f in os.listdir(TIMELINE_DIR) if f.endswith(".json")])
    scrape_percent = min((match_count / TARGET_MATCHES) * 100, 100.0)
    
    train_status = "Initializing..."
    acc_str = "Waiting for Epoch 1..."
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
                train_status = "Epoch {}/20".format(data.get("epoch", 0))
                acc_str = "Top-1: {}% | Top-3: {}%".format(data.get("top1", 0), data.get("top3", 0))
        except:
            pass

    os.system("cls")
    print("=" * 60)
    print("   LTA BLUE GIANT COMMAND CENTER | " + datetime.now().strftime("%H:%M:%S"))
    print("=" * 60)
    print("\n[1] SCRAPER PROGRESS (Target: 1000)")
    bar = "#" * int(scrape_percent // 2) + "-" * (50 - int(scrape_percent // 2))
    print("[{}] {} Matches ({:.1f}%)".format(bar, match_count, scrape_percent))
    
    print("\n[2] TRANSFORMER TRAINING (Goliath V4)")
    print("Status:   " + train_status)
    print("Accuracy: " + acc_str)
    
    print("\n[3] HARDWARE STATUS")
    print("Device:   RTX 3080 (CUDA Active)")
    print("Memory:   64GB System RAM")
    
    print("\n" + "=" * 60)
    print(" [*] Background tasks are running. ETA: ~6-8 hours.")

if __name__ == "__main__":
    while True:
        get_dashboard()
        time.sleep(10)

