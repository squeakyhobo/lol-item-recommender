import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
import sys
import numpy as np

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
import config
from model_v2 import YorickMLP

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR

class V2Dataset(Dataset):
    def __init__(self, episodes, vocab, item_to_cluster):
        self.data = episodes
        self.vocab = vocab
        self.item_to_cluster = item_to_cluster

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        my_id = torch.tensor(row["my_id"], dtype=torch.long)
        opp_id = torch.tensor(row["opp_id"], dtype=torch.long)
        rune_id = torch.tensor(row["rune_id"], dtype=torch.long)
        inv_ids = torch.tensor(row["inventory"], dtype=torch.long)
        
        # Numerical Features (Context [17] + DNA [69] + Dist [vocab_size])
        context = [
            row["minute"], row["total_gold"], row["gold_velocity"], row["core_progress"],
            row["lane_healer"], row["lane_shield"], row["lane_aa"], row["lane_tank"],
            row["lane_cc_heavy"], row["lane_mobile"], row["lane_archetype"],
            row["team_healers"], row["team_aa"], row["team_tanks"], row["team_cc"], row["team_mobile"],
            row["enemy_snowball"]
        ]
        
        num_feats = context + row["my_dna"] + row["enemy_dna"] + row["threat_dna"] + row["p_dist"]
        num_t = torch.tensor(num_feats, dtype=torch.float32)
        
        target_iid = str(row["target_item"])
        target_idx = self.vocab.get(target_iid, 0)
        target_cluster = self.item_to_cluster.get(target_iid, 0)
        
        return my_id, opp_id, rune_id, inv_ids, num_t, torch.tensor(target_idx, dtype=torch.long), torch.tensor(target_cluster, dtype=torch.long)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Brain V2.1 (Ultimate Edition) Training starting on {device}...")

    # Load Support Files
    with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "r") as f:
        vocab_data = json.load(f)
        vocab = vocab_data["item_to_index"]
        vocab_size = vocab_data["size"]
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
        item_to_cluster = json.load(f)["item_to_cluster"]
    with open(config.V2_EPISODES_PATH, "r") as f:
        episodes = json.load(f)

    ds = V2Dataset(episodes, vocab, item_to_cluster)
    train_size = int(0.9 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    # Calculate numerical_dim dynamically: Context(17) + DNA(3*23=69) + Dist(vocab_size)
    dyn_numerical_dim = 17 + 69 + vocab_size
    print(f"[*] Dynamically calculated numerical_dim: {dyn_numerical_dim}")

    # Initialize Model with synchronized dimension
    model = YorickMLP(num_champs=200, num_runes=20, num_items=vocab_size, numerical_dim=dyn_numerical_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for my_id, opp_id, rune_id, inv_ids, num_t, target_i, target_c in train_loader:
            my_id, opp_id, rune_id, inv_ids, num_t = my_id.to(device), opp_id.to(device), rune_id.to(device), inv_ids.to(device), num_t.to(device)
            target_i, target_c = target_i.to(device), target_c.to(device)
            
            optimizer.zero_grad()
            s_logits, i_logits = model(my_id, opp_id, rune_id, inv_ids, num_t)
            
            loss_i = criterion(i_logits, target_i)
            loss_s = criterion(s_logits, target_c)
            loss = loss_i + (loss_s * 1.5)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        v_correct_i = 0
        v_correct_s = 0
        with torch.no_grad():
            for my_id, opp_id, rune_id, inv_ids, num_t, target_i, target_c in val_loader:
                my_id, opp_id, rune_id, inv_ids, num_t = my_id.to(device), opp_id.to(device), rune_id.to(device), inv_ids.to(device), num_t.to(device)
                target_i, target_c = target_i.to(device), target_c.to(device)
                
                s_logits, i_logits = model(my_id, opp_id, rune_id, inv_ids, num_t)
                v_correct_i += (torch.argmax(i_logits, dim=1) == target_i).sum().item()
                v_correct_s += (torch.argmax(s_logits, dim=1) == target_c).sum().item()

        val_acc_i = (v_correct_i / len(val_ds)) * 100
        val_acc_s = (v_correct_s / len(val_ds)) * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Item Acc: {val_acc_i:.1f}% | Strat Acc: {val_acc_s:.1f}%")

        if val_acc_i > best_acc:
            best_acc = val_acc_i
            torch.save(model.state_dict(), config.V2_MODEL_PATH)
            print(f"  -> [+] New Best V2.1 Model Saved ({best_acc:.1f}%)")

if __name__ == "__main__":
    train()
