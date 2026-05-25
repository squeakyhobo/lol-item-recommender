
import sys
import os
import json
import time
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Add brain and eye to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain"))
from model import LTATransformer
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eye"))
from live_scraper import LiveScraper

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lta_brain.pth")

class InferenceThread(QThread):
    recommendation_ready = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        with open(os.path.join(DATA_DIR, "item_map.json"), "r") as f:
            self.item_map = json.load(f)
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        with open(os.path.join(DATA_DIR, "item_names.json"), "r") as f:
            self.id_to_name = json.load(f)
        with open(os.path.join(DATA_DIR, "archetypes.json"), "r") as f:
            self.archetypes = json.load(f)
        with open(os.path.join(DATA_DIR, "expert_config.json"), "r") as f:
            self.expert = json.load(f)
        
        self.model = LTATransformer(item_vocab_size=len(self.item_map))
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        self.scraper = LiveScraper()

    def run(self):
        while True:
            data = self.scraper.fetch_data()
            if data:
                active = data.get("activePlayer", {})
                game_stats = data.get("gameData", {})
                game_time = game_stats.get("gameTime", 0) / 60
                p_num = torch.tensor([[active.get("currentGold", 0), active.get("level", 1), game_time]], dtype=torch.float32)
                
                summoner_name = active.get("summonerName")
                all_players = data.get("allPlayers", [])
                active_champ = "Unknown"
                active_team = "ORDER"
                owned_ids = []
                for p in all_players:
                    if p.get("summonerName") == summoner_name:
                        active_champ = p.get("championName")
                        active_team = p.get("team")
                        owned_ids = [item.get("itemId") for item in p.get("items", [])]
                        break
                
                p_arch = torch.tensor([self.archetypes.get(active_champ, 0)], dtype=torch.long)
                
                # --- MASKING LOGIC ---
                mask = torch.zeros(len(self.item_map))
                
                # 1. Mask Owned Items
                for oid in owned_ids:
                    if str(oid) in self.item_map:
                        mask[self.item_map[str(oid)]] = 1
                
                # 2. Mask Banned Items
                for bid in self.expert["banned_ids"]:
                    if str(bid) in self.item_map:
                        mask[self.item_map[str(bid)]] = 1
                
                # 3. Mask Incompatible Groups (Unique Passives)
                for group_name, group_ids in self.expert["groups"].items():
                    if any(oid in group_ids for oid in owned_ids):
                        for gid in group_ids:
                            if str(gid) in self.item_map:
                                mask[self.item_map[str(gid)]] = 1
                
                e_num = []
                e_arch = []
                enemies = [p for p in all_players if p.get("team") != active_team][:5]
                for e in enemies:
                    scores = e.get("scores", {})
                    e_num.append([scores.get("gold", 0), e.get("level", 1), scores.get("creepScore", 0)])
                    e_arch.append(self.archetypes.get(e.get("championName"), 0))
                
                while len(e_num) < 5:
                    e_num.append([0, 0, 0])
                    e_arch.append(0)
                
                e_num_tensor = torch.tensor([e_num], dtype=torch.float32)
                e_arch_tensor = torch.tensor([e_arch], dtype=torch.long)
                
                with torch.no_grad():
                    # Pass the mask to the model
                    output = self.model(p_num, p_arch, e_num_tensor, e_arch_tensor, mask=mask.unsqueeze(0))
                    probs, indices = torch.topk(torch.softmax(output, dim=1), 3)
                    
                    results = []
                    for i in range(3):
                        idx = indices[0][i].item()
                        prob = probs[0][i].item()
                        item_id = self.inv_item_map.get(idx, "Unknown")
                        item_name = self.id_to_name.get(str(item_id), f"ID: {item_id}")
                        results.append(f"{item_name} ({prob*100:.1f}%)")
                    self.recommendation_ready.emit(results)
            time.sleep(5)

class LTAOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.layout = QVBoxLayout()
        self.title = QLabel("LoL ITEM RECOMMENDER V1.1")
        self.title.setStyleSheet("color: #FFA500; font-size: 14px; font-weight: bold; background-color: rgba(0, 0, 0, 180); padding: 5px;")
        self.layout.addWidget(self.title)
        self.recs = [QLabel("Waiting for match...") for _ in range(3)]
        for i, r in enumerate(self.recs):
            r.setStyleSheet(f"color: {"#00FF00" if i==0 else "#AAAAAA"}; font-size: 16px; font-weight: bold; background-color: rgba(0, 0, 0, 150); padding: 5px;")
            self.layout.addWidget(r)
        self.setLayout(self.layout)
        self.setGeometry(50, 50, 350, 150)
        self.thread = InferenceThread()
        self.thread.recommendation_ready.connect(self.display_recs)
        self.thread.start()

    def display_recs(self, rec_list):
        for i, text in enumerate(rec_list):
            if i < len(self.recs):
                self.recs[i].setText(f"{i+1}. {text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = LTAOverlay()
    overlay.show()
    sys.exit(app.exec_())

