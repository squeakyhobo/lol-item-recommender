import sys
import os
import json
import time
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

# Add brain and eye to path
BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
sys.path.append(os.path.join(BASE_DIR, "scripts", "brain"))
sys.path.append(os.path.join(BASE_DIR, "scripts", "eye"))

from model_yorick import YorickBrain
from live_scraper import LiveScraper
from wave_agent import create_wave_coach

DATA_DIR = os.path.join(BASE_DIR, "data")

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

class WaveCoachThread(QThread):
    """
    The High-Level Macro Coach.
    Uses 'smolagents' to analyze the wave manual and give advice.
    Runs every 60 seconds to prevent lag.
    """
    tip_ready = pyqtSignal(str)

    def __init__(self, parent_thread):
        super().__init__()
        self.parent = parent_thread 
        self.coach = create_wave_coach()

    def run(self):
        while True:
            if hasattr(self.parent, 'last_full_data') and self.parent.last_full_data and self.coach:
                try:
                    data = self.parent.last_full_data
                    active = data.get("activePlayer", {})
                    game_time = data.get("gameData", {}).get("gameTime", 0) / 60
                    hp_pct = (active.get("championStats", {}).get("currentHealth", 0) / 
                              max(active.get("championStats", {}).get("maxHealth", 1), 1)) * 100
                    
                    prompt = f"I am Yorick. Minute {game_time:.1f}. HP: {hp_pct:.0f}%. My items: {active.get('items', [])}. Check the wave manual and give me 1 short tip (10 words max)."
                    response = self.coach.run(prompt)
                    self.tip_ready.emit(str(response))
                except Exception as e:
                    print(f"[!] Wave Coach Error: {e}")
            time.sleep(60)

class InferenceThread(QThread):
    """
    The Core Engine of the HUD.
    Handles the "Eye -> Brain -> Oracle -> HUD" data pipeline every 3 seconds.
    """
    recommendation_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.last_full_data = None
        
        # 1. Load Knowledge Bases
        with open(os.path.join(DATA_DIR, "item_map.json"), "r") as f: 
            self.item_map = json.load(f)
        with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "r") as f:
            vocab_data = json.load(f)
            self.yorick_vocab = vocab_data["item_to_index"] # Added this for masking!
            self.inv_yorick_vocab = vocab_data["index_to_item"]
            self.vocab_size = vocab_data["size"]

        with open(os.path.join(DATA_DIR, "item_names.json"), "r") as f: self.id_to_name = json.load(f)
        with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f: self.item_dna = json.load(f)
        with open(os.path.join(DATA_DIR, "champion_dna.json"), "r") as f: self.champ_dna = json.load(f)
        with open(os.path.join(DATA_DIR, "expert_config.json"), "r") as f: self.expert = json.load(f)
        with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
            cdata = json.load(f)
            self.cluster_to_items = cdata["cluster_to_items"]
            self.cluster_names = cdata["cluster_names"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YorickBrain(num_clusters=15, num_items=self.vocab_size, dna_dim=9).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "yorick_brain_v3.pth"), map_location=self.device))
        self.model.eval()
        
        self.scraper = LiveScraper(); self.history = []
        self.log_file = os.path.join(BASE_DIR, "logs", f"hud_session_{int(time.time())}.jsonl")
        if not os.path.exists(os.path.dirname(self.log_file)): os.makedirs(os.path.dirname(self.log_file))

    def calculate_live_dna(self, champ_name, level, inventory):
        total = np.zeros(9)
        if champ_name in self.champ_dna:
            c = self.champ_dna[champ_name]; lf = (level - 1)
            total[0] = c["ad"]["base"] + (c["ad"]["growth"] * lf)
            total[2] = c["hp"]["base"] + (c["hp"]["growth"] * lf)
            total[3] = c["armor"]["base"] + (c["armor"]["growth"] * lf)
            total[4] = c["mr"]["base"] + (c["mr"]["growth"] * lf)
            total[5] = c["as"]["base"] + (c["as"]["growth"] * lf / 100.0)
            total[6] = c["ms"]["base"]
            
        for iid in inventory:
            if str(iid) in self.item_dna:
                d = self.item_dna[str(iid)]
                total[0] += d.get("ad", 0); total[1] += d.get("ap", 0); total[2] += d.get("hp", 0)
                total[3] += d.get("armor", 0); total[4] += d.get("mr", 0); total[5] += d.get("as", 0)
                total[6] += d.get("ms", 0); total[7] += d.get("crit", 0); total[8] += d.get("lifesteal", 0)
                
        total[0] /= 400.0; total[1] /= 800.0; total[2] /= 5000.0
        total[3] /= 300.0; total[4] /= 300.0; total[5] /= 2.5; total[6] /= 600.0
        return total

    def get_explanation(self, cluster_id, enemy_info):
        cid = str(cluster_id); reason = "Strategic power spike"; target = "the team"
        if enemy_info:
            h_ad = max(enemy_info, key=lambda x: x['dna'][0]); h_ap = max(enemy_info, key=lambda x: x['dna'][1])
            h_ar = max(enemy_info, key=lambda x: x['dna'][3])
            if cid == "9":
                if h_ar['dna'][3] > 0.4: reason = "Armor Pen"; target = h_ar['name']
                else: reason = "Healing Cut"; target = "enemy sustain"
            elif cid == "12": reason = "Physical Defense"; target = h_ad['name']
            elif cid == "7": reason = "Magic Defense"; target = h_ap['name']
            elif cid == "4": reason = "Burst Damage"; target = "the squishies"
        return f"Why: {reason} vs {target}"

    def oracle_pick(self, cluster_id, gold, owned_ids, minute):
        items = self.cluster_to_items.get(str(cluster_id), [])
        starters = ["1054", "1055", "1056", "1086"]
        if not any(int(s) in owned_ids for s in starters) and minute < 2: items = self.cluster_to_items.get("0", [])
        
        candidates = []
        for iid in items:
            if str(iid) not in self.item_map or str(iid) in ["1111", "3865", "3866", "3867", "3008", "3172"]: continue 
            if int(iid) in owned_ids or int(iid) in self.expert["banned_ids"]: continue
            conflict = False
            for g, ids in self.expert["groups"].items():
                g_str = [str(x) for x in ids]
                if str(iid) in g_str and any(str(o) in g_str for o in owned_ids): conflict = True; break
            if not conflict: candidates.append(iid)

        if not candidates: return "Check manual"
        if not any(int(s) in owned_ids for s in starters) and (minute < 5 or len(owned_ids) <= 2):
            for s in starters: 
                if s in candidates: return self.id_to_name.get(s)

        scored = []
        for iid in candidates:
            dna = self.item_dna.get(str(iid), {})
            comp = sum([1 for v in dna.values() if isinstance(v, (int, float)) and v > 0])
            scored.append({"id": iid, "leg": comp > 4, "score": comp})
        scored.sort(key=lambda x: (x["leg"], x["score"]), reverse=True)
        return self.id_to_name.get(str(scored[0]["id"]), f"ID: {scored[0]['id']}")

    def run(self):
        while True:
            try:
                data = self.scraper.fetch_data()
                if not data: time.sleep(3); continue
                self.last_full_data = data 
                
                active = data.get("activePlayer", {}); game_time = data.get("gameData", {}).get("gameTime", 0) / 60
                active_runes = active.get("fullRunes", {})
                keystone_id = active_runes.get("keystone", {}).get("id", 0) if active_runes else 0
                rune_idx = RUNE_MAP.get(keystone_id, 0); rune_tensor = torch.tensor([[rune_idx]], dtype=torch.long).to(self.device)
                
                s_name = active.get("summonerName"); all_p = data.get("allPlayers", []); owned_ids = []; a_team = "ORDER"; a_lvl = 1
                for p in all_p:
                    if p.get("summonerName") == s_name or len(all_p) == 1:
                        owned_ids = [item.get("itemID") for item in p.get("items", []) if item.get("itemID")]
                        a_team = p.get("team"); a_lvl = p.get("level", 1); break
                
                # --- HEARTBEAT ---
                print(f"[*] HEARTBEAT | Min: {game_time:.1f} | Gold: {active.get('currentGold', 0):.0f} | Inv: {owned_ids}")
                if self.history and self.history[-1]["inventory"] != owned_ids: print("  -> Inventory Changed! Resetting AI memory...")
                
                state = {"gold": active.get("currentGold", 0), "total_gold": active.get("stats", {}).get("totalGold", 0), "level": a_lvl, "minute": game_time, "inventory": owned_ids}
                self.history.append(state); 
                if len(self.history) > 5: self.history.pop(0)
                p_hist = list(self.history); while len(p_hist) < 5: p_hist.insert(0, p_hist[0])

                p_num = torch.tensor([[[f["gold"]/5000.0, f.get("total_gold", 0)/25000.0, f["level"]/20.0, f["minute"]/60.0, 0] for f in p_hist]], dtype=torch.float32).to(self.device)
                p_dna = torch.tensor(np.array([[self.calculate_live_dna("Yorick", f["level"], f["inventory"]) for f in p_hist]]), dtype=torch.float32).to(self.device)

                e_info = []; e_dna_list = []
                for p in all_p:
                    if p.get("team") != a_team:
                        inv = [item.get("itemID") for item in p.get("items", []) if item.get("itemID")]
                        dna = self.calculate_live_dna(p.get("championName"), p.get("level", 1), inv)
                        e_info.append({"name": p.get("championName"), "dna": dna}); e_dna_list.append(dna)
                while len(e_dna_list) < 5: e_dna_list.append(np.zeros(9))
                e_dna = torch.tensor(np.array([e_dna_list[:5]]), dtype=torch.float32).to(self.device)

                # --- NEW: GENERATE ITEM MASK ---
                item_mask = torch.zeros(self.vocab_size).to(self.device)
                for oid in owned_ids:
                    if str(oid) in self.yorick_vocab:
                        item_mask[self.yorick_vocab[str(oid)]] = 1
                for bid in self.expert["banned_ids"]:
                    if str(bid) in self.yorick_vocab:
                        item_mask[self.yorick_vocab[str(bid)]] = 1

                with torch.no_grad():
                    # Pass the item_mask to the model!
                    c_logits, i_logits = self.model(p_num, p_dna, e_dna, None, rune_tensor, item_mask=item_mask.unsqueeze(0))
                    c_probs, c_idx_t = torch.topk(torch.softmax(c_logits, dim=1), 1)
                    i_probs, i_idx_t = torch.topk(torch.softmax(i_logits, dim=1), 1)
                    
                    c_idx = c_idx_t[0][0].item(); i_idx = i_idx_t[0][0].item()
                    ai_name = self.id_to_name.get(str(self.inv_yorick_vocab.get(str(i_idx), 0)), "Unknown")
                    oracle_name = self.oracle_pick(c_idx, active.get("currentGold", 0), owned_ids, game_time)
                    
                    self.recommendation_ready.emit({
                        "strat": f"{self.cluster_names.get(str(c_idx), 'Cluster '+str(c_idx)).upper()} ({c_probs[0][0].item()*100:.0f}%)",
                        "item": f"AI: {ai_name} | Oracle: {oracle_name}",
                        "why": self.get_explanation(c_idx, e_info),
                        "inv": [self.id_to_name.get(str(x), str(x)) for x in owned_ids]
                    })
            except Exception as e: print(f"[!] Thread Error: {e}")
            time.sleep(3)

class LTAOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.layout = QVBoxLayout()
        self.header = QLabel("THE MAIDEN'S BRAIN (RUNE-AWARE)")
        self.header.setStyleSheet("color: #FF00FF; font-size: 13px; font-weight: bold; background-color: rgba(0, 0, 0, 220); padding: 5px; border-top-left-radius: 5px;")
        self.strat_lbl = QLabel("Strategy..."); self.strat_lbl.setStyleSheet("color: #AAAAAA; font-size: 11px; background-color: rgba(20, 20, 20, 200); padding-left: 5px;")
        self.item_lbl = QLabel("Waiting..."); self.item_lbl.setStyleSheet("color: #00FF00; font-size: 16px; font-weight: bold; background-color: rgba(0, 0, 0, 200); padding: 10px; border-left: 5px solid #00FF00;")
        self.why_lbl = QLabel("Reasoning..."); self.why_lbl.setStyleSheet("color: #00FFFF; font-size: 11px; font-style: italic; background-color: rgba(0, 0, 0, 180); padding: 5px;")
        self.coach_lbl = QLabel("Coach: Initializing..."); self.coach_lbl.setStyleSheet("color: #FFA500; font-size: 12px; font-weight: bold; background-color: rgba(30, 15, 0, 220); padding: 8px; border-left: 5px solid #FFA500;")
        self.inv_lbl = QLabel("Inventory: []"); self.inv_lbl.setStyleSheet("color: #FFFFFF; font-size: 10px; background-color: rgba(50, 0, 0, 200); padding: 5px; border-bottom-left-radius: 5px;")
        self.layout.addWidget(self.header); self.layout.addWidget(self.strat_lbl); self.layout.addWidget(self.item_lbl); self.layout.addWidget(self.why_lbl); self.layout.addWidget(self.coach_lbl); self.layout.addWidget(self.inv_lbl)
        self.setLayout(self.layout); self.setGeometry(50, 150, 450, 220)
        self.inference_thread = InferenceThread()
        self.inference_thread.recommendation_ready.connect(self.update_ui)
        self.inference_thread.start()
        self.coach_thread = WaveCoachThread(self.inference_thread)
        self.coach_thread.tip_ready.connect(self.update_coach)
        self.coach_thread.start()

    def update_ui(self, data):
        self.strat_lbl.setText(data['strat']); self.item_lbl.setText(data['item'])
        self.why_lbl.setText(data['why']); self.inv_lbl.setText(f"DETECTED: {', '.join(data['inv'])}")

    def update_coach(self, tip): self.coach_lbl.setText(f"COACH: {tip}")

if __name__ == "__main__":
    app = QApplication(sys.argv); overlay = LTAOverlay(); overlay.show(); sys.exit(app.exec_())
