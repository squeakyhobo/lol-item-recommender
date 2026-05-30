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
# These numbers match the Riot API 'Keystone' ID.
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
    It calculates the 'Total DNA' (Champion + Items) live during training.
    """
    def __init__(self, episodes, item_dna, champ_dna, item_to_cluster, vocab):
        self.episodes = episodes
        self.item_dna = item_dna
        self.champ_dna = champ_dna
        self.item_to_cluster = item_to_cluster
        self.vocab = vocab # Using the pruned 63-item vocab

    def __len__(self): return len(self.episodes)

    def calculate_total_dna(self, champ_name, level, inventory):
        """
        Merges base champion stats with item stats.
        Normalization (division) is key to preventing the 'Dead Brain' issue.
        """
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
                
        # Normalization targets (Standard high-level game stats)
        total[0] /= 400.0   # 400 AD cap
        total[1] /= 800.0   # 800 AP cap
        total[2] /= 5000.0  # 5000 HP cap
        total[3] /= 300.0   # 300 Armor cap
        total[4] /= 300.0   # 300 MR cap
        total[5] /= 2.5     # 2.5 Attack Speed cap
        total[6] /= 600.0   # 600 MS cap
        return total

    def __getitem__(self, idx):
        e = self.episodes[idx]
        seq = e["sequence"]
        last_frame = seq[-1]
        
        # Hyper-Normalization Constants
        GOLD_SCALE = 25000.0
        LVL_SCALE = 20.0 
        MIN_SCALE = 60.0
        
        # Build sequence numeric features
        p_num = []
        p_dna = []
        for f in seq:
            p_num.append([
                f["gold"] / 5000.0,
                f["total_gold"] / GOLD_SCALE,
                f["level"] / LVL_SCALE,
                f["minute"] / MIN_SCALE,
                f.get("kill_pressure", 0) / 10.0
            ])
            p_dna.append(self.calculate_total_dna("Yorick", f["level"], f["inventory"]))
            
        # Build enemy DNA features
        e_dna = []
        for enemy in last_frame["enemy_context"][:5]:
            e_dna.append(self.calculate_total_dna(
                enemy.get("championName"), 
                enemy["level"], 
                enemy.get("inventory", [])
            ))
        while len(e_dna) < 5: e_dna.append(np.zeros(9))
            
        target_item_id = str(e["target_item"])
        target_cluster = self.item_to_cluster.get(target_item_id, 0)
        
        # Map the specific item to its pruned 0-62 index
        # If it's somehow not in the vocab, default to 0
        target_item_idx = self.vocab.get(target_item_id, 0)
        
        return (torch.tensor(p_num, dtype=torch.float32), 
                torch.tensor(np.array(p_dna), dtype=torch.float32), 
                torch.tensor(np.array(e_dna), dtype=torch.float32), 
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
    print(f"[*] YORICK MTL BRAIN (V3) training starting on {device}...")

    # 1. Load Data
    with open(os.path.join(DATA_DIR, "yorick_episodes.json"), "r") as f: episodes = json.load(f)
    with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f: item_dna = json.load(f)
    with open(os.path.join(DATA_DIR, "champion_dna.json"), "r") as f: champ_dna = json.load(f)
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
        item_to_cluster = json.load(f)["item_to_cluster"]
    with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "r") as f: 
        vocab_data = json.load(f)
        vocab = vocab_data["item_to_index"]
        vocab_size = vocab_data["size"]
        
    print(f"[*] Loaded {len(episodes)} Pure Yorick Snapshots")

    # Inject pruned vocab into dataset
    dataset = YorickDataset(episodes, item_dna, champ_dna, item_to_cluster, vocab)

    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    # 3. Setup Brain & Optimizer
    # Update model to use the new smaller vocab_size (63)
    model = YorickBrain(num_clusters=15, num_items=vocab_size, dna_dim=9).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    epochs = 100
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for p_num, p_dna, e_dna, rune, target_cluster, target_item in loader:
            p_num, p_dna, e_dna, rune = p_num.to(device), p_dna.to(device), e_dna.to(device), rune.to(device)
            target_cluster, target_item = target_cluster.to(device), target_item.to(device)
            
            optimizer.zero_grad()
            cluster_logits, item_logits = model(p_num, p_dna, e_dna, None, rune)
            
            # ASYMMETRIC LOSS WEIGHTING
            # 80% focus on Strategy (Clusters), 20% focus on exact item.
            # This prevents the noisy 63-item guess from sabotaging the stable 15-cluster guess.
            loss_cluster = criterion(cluster_logits, target_cluster)
            loss_item = criterion(item_logits, target_item)
            loss = (loss_cluster * 0.8) + (loss_item * 0.2) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_acc_cluster = 0
            val_acc_item = 0
            with torch.no_grad():
                for p_num, p_dna, e_dna, rune, target_cluster, target_item in val_loader:
                    p_num, p_dna, e_dna, rune = p_num.to(device), p_dna.to(device), e_dna.to(device), rune.to(device)
                    target_cluster, target_item = target_cluster.to(device), target_item.to(device)
                    
                    cluster_logits, item_logits = model(p_num, p_dna, e_dna, None, rune)
                    val_acc_cluster += get_accuracy(cluster_logits, target_cluster).item()
                    val_acc_item += get_accuracy(item_logits, target_item).item()
            
            avg_acc_cluster = val_acc_cluster / len(val_loader)
            avg_acc_item = val_acc_item / len(val_loader)
            
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Cluster Acc: {avg_acc_cluster:.1f}% | Item Acc: {avg_acc_item:.1f}%")
            
            # Save the absolute best version based on specific item accuracy
            if avg_acc_item > best_acc:
                best_acc = avg_acc_item
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, "yorick_brain_v3.pth"))
                print("  -> New Best Model Saved!")

    print(f"[*] Training complete. Best Specific Item Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
