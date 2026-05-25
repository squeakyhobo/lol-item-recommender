
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
import numpy as np
from model import LTATransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

class LTADataset(Dataset):
    def __init__(self, snapshots, item_map):
        self.snapshots = snapshots
        self.item_map = item_map

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        s = self.snapshots[idx]
        p_num = torch.tensor([s["top_gold"], 1, s["minute"]], dtype=torch.float32)
        p_arch = torch.tensor([0], dtype=torch.long)
        e_num = []
        e_arch = []
        enemies = [p for p in s["world_state"] if p["is_enemy"]][:5]
        for e in enemies:
            e_num.append([e["gold"], e["level"], 0])
            e_arch.append(0)
        while len(e_num) < 5:
            e_num.append([0, 0, 0])
            e_arch.append(0)
        e_num = torch.tensor(e_num, dtype=torch.float32)
        e_arch = torch.tensor(e_arch, dtype=torch.long)
        target = torch.tensor(self.item_map.get(str(s["target_item"]), 0), dtype=torch.long)
        return p_num, p_arch, e_num, e_arch, target

def train():
    with open(os.path.join(DATA_DIR, "top_lane_snapshots.json"), "r") as f:
        snapshots = json.load(f)
    unique_items = sorted(list(set(str(s["target_item"]) for s in snapshots)))
    item_map = {item: i for i, item in enumerate(unique_items)}
    
    # SAVE THE ITEM MAP (Crucial for inference!)
    with open(os.path.join(DATA_DIR, "item_map.json"), "w") as f:
        json.dump(item_map, f)
    print(f"[*] Saved item map with {len(item_map)} unique items.")

    dataset = LTADataset(snapshots, item_map)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    model = LTATransformer(item_vocab_size=len(item_map))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    print(f"[*] Training on {train_size} | Validating on {val_size}")
    best_val_loss = float("inf")
    for epoch in range(10):
        model.train()
        train_loss = 0
        for p_num, p_arch, e_num, e_arch, target in train_loader:
            optimizer.zero_grad()
            output = model(p_num, p_arch.squeeze(1), e_num, e_arch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for p_num, p_arch, e_num, e_arch, target in val_loader:
                output = model(p_num, p_arch.squeeze(1), e_num, e_arch)
                loss = criterion(output, target)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                correct += (preds == target).sum().item()
        avg_val = val_loss/len(val_loader)
        print(f"Ep {epoch+1:02d} | Val Loss: {avg_val:.4f} | Acc: {(correct/val_size)*100:.2f}%")
        scheduler.step(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lta_brain.pth"))
    print("[*] Training Complete. Model Saved.")

if __name__ == "__main__":
    train()

