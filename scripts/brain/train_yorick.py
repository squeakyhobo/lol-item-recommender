import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
import numpy as np
from model_yorick import YorickBrain

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Comprehensive Yorick Keystones to Index
RUNE_MAP = {
    8010: 1, # Conqueror
    8437: 2, # Grasp
    8229: 3, # Comet
    8992: 4, # Fleet
    8230: 5, # Phase Rush
    8112: 6, # Electrocute
    8369: 7, # First Strike
    8214: 8, # Summon Aery
    8008: 9  # Lethal Tempo
}

class YorickDataset(Dataset):
    """
    Handles the loading and normalization of Yorick matches.
    It extracts the Direct Lane Opponent DNA for focused weighting.
    """
    def __init__(self, episodes, item_dna, champ_dna, item_to_cluster, vocab):
        self.episodes = episodes
        self.item_dna = item_dna
        self.champ_dna = champ_dna
        self.item_to_cluster = item_to_cluster
        self.vocab = vocab 

    def __len__(self): return len(self.episodes)

    def calculate_total_dna(self, champ_name, level, inventory):
        total = np.zeros(9)
        if champ_name in self.champ_dna:
            c = self.champ_dna[champ_name]
            level_factor = (level - 1)
            total[0] = c["ad"]["base"] + (c["ad"]["growth"] * level_factor)
            total[2] = c["hp"]["base"] + (c["hp"]["growth"] * level_factor)
            total[3] = c["armor"]["base"] + (c["armor"]["growth"] * level_factor)
            total[4] = c["mr"]["base"] + (c["mr"]["growth"] * level_factor)
            total[5] = c["as"]["base"] + (c["as"]["growth"] * level_factor / 100.0)
            total[6] = c["ms"]["base"]
            
        for iid in inventory:
            if str(iid) in self.item_dna:
                d = self.item_dna[str(iid)]
                total[0] += d.get("ad", 0); total[1] += d.get("ap", 0)
                total[2] += d.get("hp", 0); total[3] += d.get("armor", 0)
                total[4] += d.get("mr", 0); total[5] += d.get("as", 0)
                total[6] += d.get("ms", 0); total[7] += d.get("crit", 0)
                total[8] += d.get("lifesteal", 0)
                
        total[0] /= 400.0; total[1] /= 800.0; total[2] /= 5000.0
        total[3] /= 300.0; total[4] /= 300.0; total[5] /= 2.5; total[6] /= 600.0
        return total

    def __getitem__(self, idx):
        e = self.episodes[idx]
        seq = e["sequence"]
        last_frame = seq[-1]
        
        GOLD_SCALE = 25000.0
        LVL_SCALE = 20.0 
        MIN_SCALE = 60.0
        
        # 1. Player Sequence
        p_num = []
        p_dna = []
        for f in seq:
            p_num.append([f["gold"]/5000.0, f["total_gold"]/GOLD_SCALE, f["level"]/LVL_SCALE, f["minute"]/MIN_SCALE, f.get("kill_pressure", 0)/10.0,f["gold_diff"]/5000.0])
            p_dna.append(self.calculate_total_dna("Yorick", f["level"], f["inventory"]))
            
        # 2. Enemy Team DNA (Split Matchup from Others)
        lane_dna = np.zeros(9)
        others_dna = []
        
        for enemy in last_frame["enemy_context"]:
            dna = self.calculate_total_dna(enemy.get("championName"), enemy["level"], enemy.get("inventory", []))
            if enemy.get("is_lane_opponent", False):
                lane_dna = dna # Found the matchup!
            else:
                others_dna.append(dna)
        
        # Pad others to 4
        while len(others_dna) < 4: others_dna.append(np.zeros(9))
        others_dna = others_dna[:4]
            
        target_item_id = str(e["target_item"])
        target_cluster = self.item_to_cluster.get(target_item_id, 0)
        target_item_idx = self.vocab.get(target_item_id, 0)
        
        return (torch.tensor(p_num, dtype=torch.float32), 
                torch.tensor(np.array(p_dna), dtype=torch.float32), 
                torch.tensor(np.array(others_dna), dtype=torch.float32), 
                torch.tensor(lane_dna, dtype=torch.float32),
                torch.tensor(RUNE_MAP.get(last_frame.get("keystone", 0), 0), dtype=torch.long),
                torch.tensor(target_cluster, dtype=torch.long),
                torch.tensor(target_item_idx, dtype=torch.long))

def get_accuracy(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct.float().sum() * 100.0 / target.size(0)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] YORICK MATCHUP-AWARE BRAIN (V4) training starting on {device}...")

    # Load Data
    with open(os.path.join(DATA_DIR, "yorick_episodes.json"), "r") as f: episodes = json.load(f)
    with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f: item_dna = json.load(f)
    with open(os.path.join(DATA_DIR, "champion_dna.json"), "r") as f: champ_dna = json.load(f)
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
        item_to_cluster = json.load(f)["item_to_cluster"]
    with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "r") as f: 
        vocab_data = json.load(f)
        vocab = vocab_data["item_to_index"]
        vocab_size = vocab_data["size"]
        
    dataset = YorickDataset(episodes, item_dna, champ_dna, item_to_cluster, vocab)
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = YorickBrain(num_clusters=15, num_items=vocab_size, dna_dim=9).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 100
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for p_num, p_dna, others_dna, lane_dna, rune, target_cluster, target_item in loader:
            p_num, p_dna, others_dna, lane_dna, rune = p_num.to(device), p_dna.to(device), others_dna.to(device), lane_dna.to(device), rune.to(device)
            target_cluster, target_item = target_cluster.to(device), target_item.to(device)
            
            optimizer.zero_grad()
            cluster_logits, item_logits = model(p_num, p_dna, others_dna, lane_dna, rune)
            
            loss_cluster = criterion(cluster_logits, target_cluster)
            loss_item = criterion(item_logits, target_item)
            loss = (loss_cluster * 0.8) + (loss_item * 0.2) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_acc_item = 0
            with torch.no_grad():
                for p_num, p_dna, others_dna, lane_dna, rune, target_cluster, target_item in val_loader:
                    p_num, p_dna, others_dna, lane_dna, rune = p_num.to(device), p_dna.to(device), others_dna.to(device), lane_dna.to(device), rune.to(device)
                    target_item = target_item.to(device)
                    _, item_logits = model(p_num, p_dna, others_dna, lane_dna, rune)
                    val_acc_item += get_accuracy(item_logits, target_item).item()
            
            avg_acc_item = val_acc_item / len(val_loader)
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(loader):.4f} | Item Acc: {avg_acc_item:.1f}%")
            
            if avg_acc_item > best_acc:
                best_acc = avg_acc_item
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, "yorick_brain_v4.pth"))
                print("  -> New Best Model Saved (V4)!")

    print(f"[*] Training complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
