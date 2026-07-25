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
from model_v2 import YorickMLP,MoEYorickMLP

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR

class V2Dataset(Dataset):
    def __init__(self, episodes, vocab, item_to_cluster):
        print("[*] Pre-converting dataset to tensors for high-performance training...")
        self.my_ids = torch.tensor([row["my_id"] for row in episodes], dtype=torch.long)
        self.opp_ids = torch.tensor([row["opp_id"] for row in episodes], dtype=torch.long)
        self.rune_ids = torch.tensor([row["rune_id"] for row in episodes], dtype=torch.long)
        self.inv_ids = torch.tensor([row["inventory"] for row in episodes], dtype=torch.long)

        num_feats_list = []
        target_idxs = []
        target_clusters = []
        for row in episodes:
            context = [
                row["minute"], row["total_gold"], row["gold_velocity"], row["core_progress"],
                row["lane_healer"], row["lane_shield"], row["lane_aa"], row["lane_tank"],
                row["lane_cc_heavy"], row["lane_mobile"], row["lane_archetype"],
                row["team_healers"], row["team_aa"], row["team_tanks"], row["team_cc"], row["team_mobile"],
                row["enemy_snowball"]
            ]
            context.extend(row.get("enemy_team_composition_tags", [0.0] * 9))
            context.extend(row.get("enemy_team_damage_split", [0.5, 0.5]))
            context.append(row.get("player_snowball_factor", 0.0))
            context.append(row.get("kda_proxy", 0.0))

            num_feats = context + row["my_dna"] + row["enemy_dna"] + row["threat_dna"] + row["p_dist"]
            num_feats_list.append(num_feats)

            target_iid = str(row["target_item"])
            target_idxs.append(vocab.get(target_iid, 0))
            target_clusters.append(item_to_cluster.get(target_iid, 0))

        self.num_feats = torch.tensor(num_feats_list, dtype=torch.float32)
        self.target_idxs = torch.tensor(target_idxs, dtype=torch.long)
        self.target_clusters = torch.tensor(target_clusters, dtype=torch.long)
        print("[*] Dataset conversion complete!")

    def __len__(self):
        return len(self.my_ids)

    def __getitem__(self, idx):
        return (
            self.my_ids[idx],
            self.opp_ids[idx],
            self.rune_ids[idx],
            self.inv_ids[idx],
            self.num_feats[idx],
            self.target_idxs[idx],
            self.target_clusters[idx]
        )
#Note change into a flag with argeparser
def train(is_moe:bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Brain {config.MODEL_VERSION} (Ultimate Edition) Training starting on {device}...")

    # Load Support Files
    with open(os.path.join(DATA_DIR, "item_vocab.json"), "r") as f:
        vocab_data = json.load(f)
        vocab = vocab_data["item_to_index"]
        vocab_size = vocab_data["size"]
    with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
        item_to_cluster = json.load(f)["item_to_cluster"]
    with open(config.V2_EPISODES_PATH, "r") as f:
        episodes = json.load(f)

    # Filter for specific champions (including Aatrox)

    
    
   
    

    ds = V2Dataset(episodes, vocab, item_to_cluster)
    train_size = int(0.9 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    

    # Calculate numerical_dim dynamically from a dataset sample
    _, _, _, _, sample_num_t, _, _ = ds[0]
    dyn_numerical_dim = sample_num_t.shape[0]
    print(f"[*] Dynamically calculated numerical_dim: {dyn_numerical_dim}")

    # Initialize Model with synchronized dimension
    

    if(is_moe):
        model = MoEYorickMLP(num_champs=200, num_runes=20, num_items=vocab_size, numerical_dim=dyn_numerical_dim).to(device)

    else:
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
            if(is_moe):
                s_logits, i_logits, aux_loss, _ = model(my_id, opp_id, rune_id, inv_ids, num_t)
            else:
                s_logits, i_logits = model(my_id, opp_id, rune_id, inv_ids, num_t)
            
            loss_i = criterion(i_logits, target_i)
            loss_s = criterion(s_logits, target_c)
            loss = loss_i + (loss_s * 1.5) #Note what happends if the scalar for aux loss is too high?
            if(is_moe):
                loss+= aux_loss 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        v_correct_i = 0
        v_correct_s = 0
        global_gate_probs = torch.zeros(len(model.experts), device=device) # Hardcoded to 8 experts for tracking
        
        with torch.no_grad():
            for my_id, opp_id, rune_id, inv_ids, num_t, target_i, target_c in val_loader:
                my_id, opp_id, rune_id, inv_ids, num_t = my_id.to(device), opp_id.to(device), rune_id.to(device), inv_ids.to(device), num_t.to(device)
                target_i, target_c = target_i.to(device), target_c.to(device)
                if(is_moe):
                    s_logits, i_logits, aux_loss, batch_gate_probs = model(my_id, opp_id, rune_id, inv_ids, num_t)
                    global_gate_probs += batch_gate_probs.sum(dim=0)
                else:
                    s_logits, i_logits = model(my_id, opp_id, rune_id, inv_ids, num_t)

                v_correct_i += (torch.argmax(i_logits, dim=1) == target_i).sum().item()
                v_correct_s += (torch.argmax(s_logits, dim=1) == target_c).sum().item()

        val_acc_i = (v_correct_i / len(val_ds)) * 100
        val_acc_s = (v_correct_s / len(val_ds)) * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Item Acc: {val_acc_i:.1f}% | Strat Acc: {val_acc_s:.1f}%")
            if is_moe:
                avg_gates = (global_gate_probs / len(val_ds)) * 100
                formatted_probs = [f"{p.item():.1f}%" for p in avg_gates]
                print(f"  -> MoE Global Expert Dist: {formatted_probs}")

        if val_acc_i > best_acc:
            best_acc = val_acc_i
            save_path = os.path.join(MODEL_DIR, config.MODEL_VERSION)
            torch.save(model.state_dict(), save_path)
            print(f"  -> [+] New Best {config.MODEL_VERSION} Model Saved ({best_acc:.1f}%) to {save_path}")

if __name__ == "__main__":
    train(is_moe=True)
