import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
import numpy as np
from model_v4 import GoliathV4

BASE_DIR = "/Users/lucas/Desktop/lol-item-recommender"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

ROLE_RANK = {"JUNGLE": 0, "MIDDLE": 1, "BOTTOM": 2, "UTILITY": 3}

# Comprehensive Yorick Keystones to Index
RUNE_MAP = {
    8010: 1, 8437: 2, 8229: 3, 8992: 4, 8230: 5, 8112: 6, 8369: 7, 8214: 8, 8008: 9
}

class LTA_Dataset(Dataset):
    def __init__(self, episodes, item_dna, champ_dna, item_to_cluster, vocab):
        self.episodes = episodes
        self.item_dna = item_dna
        self.champ_dna = champ_dna
        self.item_to_cluster = item_to_cluster
        self.vocab = vocab 

    def __len__(self): return len(self.episodes)

    def calculate_total_dna(self, champ_name, level, inventory):
        # Full Research-Grade DNA Logic
        total = np.zeros(15) 
        if champ_name in self.champ_dna:
            c = self.champ_dna[champ_name]
            level_factor = (level - 1)
            # Base + Growth
            total[0] = c["ad"]["base"] + (c["ad"]["growth"] * level_factor)
            total[1] = 0 # Base AP is usually 0
            total[2] = c["hp"]["base"] + (c["hp"]["growth"] * level_factor)
            total[3] = c["armor"]["base"] + (c["armor"]["growth"] * level_factor)
            total[4] = c["mr"]["base"] + (c["mr"]["growth"] * level_factor)
            total[5] = c["as"]["base"] + (c["as"]["growth"] * level_factor / 100.0)
            total[6] = c["ms"]["base"]
            
        # Add Item Bonuses
        for iid in inventory:
            if str(iid) in self.item_dna:
                d = self.item_dna[str(iid)]
                total[0] += d.get("ad", 0); total[1] += d.get("ap", 0)
                total[2] += d.get("hp", 0); total[3] += d.get("armor", 0)
                total[4] += d.get("mr", 0); total[5] += d.get("as", 0)
                total[6] += d.get("ms", 0); total[7] += d.get("crit", 0)
                total[8] += d.get("lifesteal", 0)
                
        # NO MANUAL DIVISION HERE - We let Goliath's LayerNorm handle it!
        return total

    def __getitem__(self, idx):
        e = self.episodes[idx]
        seq = e["sequence"]
        last_frame = seq[-1]
        
        # 1. Player Sequence (RAW NUMBERS)
        p_num = []
        p_dna = []
        for f in seq:
            p_num.append([
                f["gold"], f["total_gold"], f["level"], 
                f["minute"], f.get("kill_pressure", 0), f.get("gold_diff", 0)
            ])
            p_dna.append(self.calculate_total_dna("Yorick", f["level"], f["inventory"]))
            
        # 2. Enemy Team (FIXED ROLE ORDERING + RAW NUMBERS)
        e_num = []
        e_dna = []
        others_raw = []
        lane_enemy = None
        
        for enemy in last_frame["enemy_context"]:
            dna = self.calculate_total_dna(enemy.get("championName"), enemy["level"], enemy.get("inventory", []))
            stats = [enemy["gold"], enemy["level"], 0, 0, 0] # Pad missing fields
            if enemy.get("is_lane_opponent", False):
                lane_enemy = {"dna": dna, "num": stats, "role": "TOP"}
            else:
                others_raw.append({"dna": dna, "num": stats, "role": enemy.get("role", "Unknown")})
        
        # Fallback if no lane opponent found
        if not lane_enemy:
            lane_enemy = {"dna": np.zeros(15), "num": [0,1,0,0,0], "role": "TOP"}

        others_raw.sort(key=lambda x: ROLE_RANK.get(x["role"], 99))
        
        # Team: [Lane, Jungle, Mid, ADC, Supp]
        team_raw = [lane_enemy] + others_raw
        
        for p in team_raw[:5]:
            e_num.append(p["num"])
            e_dna.append(p["dna"])
            
        target_item_id = str(e["target_item"])
        target_item_idx = self.vocab.get(target_item_id, 0)
        
        return (torch.tensor(p_num, dtype=torch.float32), 
                torch.tensor(np.array(p_dna), dtype=torch.float32), 
                torch.tensor(np.array(e_num), dtype=torch.float32), 
                torch.tensor(np.array(e_dna), dtype=torch.float32),
                torch.tensor(target_item_idx, dtype=torch.long))

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] GOLIATH V4 training starting on {device}...")

    # Load Configs
    with open(os.path.join(DATA_DIR, "yorick_episodes.json"), "r") as f: episodes = json.load(f)
    with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f: item_dna = json.load(f)
    with open(os.path.join(DATA_DIR, "champion_dna.json"), "r") as f: champ_dna = json.load(f)
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
        item_to_cluster = json.load(f)["item_to_cluster"]
    with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "r") as f: 
        vocab_data = json.load(f); vocab = vocab_data["item_to_index"]; vocab_size = vocab_data["size"]
        
    dataset = LTA_Dataset(episodes, item_dna, champ_dna, item_to_cluster, vocab)
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = GoliathV4(item_vocab_size=vocab_size, dna_dim=15).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        for p_num, p_dna, e_num, e_dna, target in loader:
            p_num, p_dna, e_num, e_dna, target = p_num.to(device), p_dna.to(device), e_num.to(device), e_dna.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits = model(p_num, p_dna, e_num, e_dna)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 5 == 0:
            # Simple accuracy check
            print(f"Epoch {epoch+1} Complete. Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "goliath_v4_baseline.pth"))
    print("[+] Training Complete. Model saved to models/goliath_v4_baseline.pth")

if __name__ == "__main__":
    train()
