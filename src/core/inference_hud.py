import sys
import os
import json
import time
import torch
import numpy as np
import math
import traceback
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
import config

sys.path.append(os.path.join(BASE_DIR, "src", "model"))
sys.path.append(os.path.join(BASE_DIR, "src", "scraper"))

from model_v2 import YorickMLP
from live_scraper import LiveScraper

DATA_DIR = config.DATA_DIR
DEBUG_LOG = os.path.join(config.LOGS_DIR, "hud_debug.jsonl")

class InferenceThread(QThread):
    recommendation_ready = pyqtSignal(dict)
    def __init__(self, locked_matchup=None):
        super().__init__()
        self.locked_matchup = locked_matchup
        self.last_full_data = None
        
        # Threat Inertia Memory
        self.current_boss_name = "Unknown"
        self.current_boss_gold = 0
        
        # Load Knowledge Bases
        with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "r") as f:
            v = json.load(f)
            self.yorick_vocab = v["item_to_index"]
            self.inv_vocab = v["index_to_item"]
            self.vocab_size = v["size"]
            
        with open(os.path.join(DATA_DIR, "item_names.json"), "r") as f: 
            self.id_to_name = json.load(f)
        with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f: 
            self.item_dna = json.load(f)
        with open(os.path.join(DATA_DIR, "champion_dna.json"), "r") as f: 
            self.champ_dna = json.load(f)
        with open(os.path.join(DATA_DIR, "champion_knowledge.json"), "r") as f: 
            self.kb = json.load(f)
        with open(os.path.join(DATA_DIR, "expert_config.json"), "r") as f: 
            self.expert = json.load(f)
        with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
            c = json.load(f)
            self.cluster_names = c["cluster_names"]
        with open(os.path.join(DATA_DIR, "item_costs.json"), "r") as f: 
            self.item_costs = json.load(f)
        
        # Map Champion names to 0-200 for embeddings
        self.champ_to_idx = {name: i for i, name in enumerate(sorted(self.kb.keys()))}
        self.vocab_list = sorted(list(self.yorick_vocab.keys()), key=lambda x: self.yorick_vocab[x])
        self.core_items = ["3078", "6692", "3153", "6631", "3181"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sychronized with V2.1 architecture (61 features)
        self.model = YorickMLP(num_champs=200, num_runes=10, num_items=self.vocab_size, numerical_dim=61).to(self.device)
        
        # Load V2 Weights
        if os.path.exists(config.V2_MODEL_PATH):
            self.model.load_state_dict(torch.load(config.V2_MODEL_PATH, map_location=self.device))
            print(f"[*] HUD: Loaded Maiden Brain V2.1 (Omega Context Edition).")
        else:
            print(f"[!] HUD: V2 Model weights not found at {config.V2_MODEL_PATH}")
        
        self.model.eval()
        self.scraper = LiveScraper()
        self.history = []
        os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
        
        # Session Logging
        self.session_log = []
        self.last_log_min = -1
        self.log_path = os.path.join(BASE_DIR, "data", "my_matches", f"match_{int(time.time())}.json")

    def save_session_log(self):
        try:
            with open(self.log_path, "w") as f:
                json.dump(self.session_log, f, indent=4)
        except: pass

    def calculate_dna(self, name, lvl, inv):
        total = [0.0] * 9
        clean_name = name.replace(" ", "").replace("'", "").lower()
        c_info = self.kb.get(clean_name, {})
        c_id = c_info.get("id")
        if c_id and str(c_id) in self.champ_dna:
            c = self.champ_dna[str(c_id)]
            lf = (lvl - 1)
            total[0] = (c["ad"]["base"] + (c["ad"]["growth"] * lf)) / 150.0
            total[2] = (c["hp"]["base"] + (c["hp"]["growth"] * lf)) / 4000.0
            total[3] = (c["armor"]["base"] + (c["armor"]["growth"] * lf)) / 200.0
            total[4] = (c["mr"]["base"] + (c["mr"]["growth"] * lf)) / 200.0
        for iid in inv:
            if str(iid) in self.item_dna:
                d = self.item_dna[str(iid)]
                total[0] += d.get("ad", 0) / 150.0
                total[1] += d.get("ap", 0) / 400.0
                total[2] += d.get("hp", 0) / 4000.0
                total[3] += d.get("armor", 0) / 200.0
                total[4] += d.get("mr", 0) / 200.0
        return total

    def calculate_dist_vector(self, inv):
        dist = [1.0] * self.vocab_size
        inv_str = [str(i) for i in inv]
        for i_id, i_idx in self.yorick_vocab.items():
            if i_id in inv_str: 
                dist[i_idx] = 0.0
            else:
                c_info = self.item_costs.get(i_id, {})
                val = sum([cost for cid, cost in c_info.get("components", {}).items() if cid in inv_str])
                dist[i_idx] = max(0.0, 1.0 - (val / max(c_info.get("total_cost", 3000), 1)))
        return dist

    def run(self):
        while True:
            try:
                data = self.scraper.fetch_data()
                if not data or "activePlayer" not in data:
                    time.sleep(3)
                    continue
                
                active = data["activePlayer"]
                allp = data["allPlayers"]
                game_time = data.get("gameData", {}).get("gameTime", 0) / 60
                sname = active.get("summonerName")
                me = None
                owned = []
                ateam = "ORDER"
                alvl = 1
                
                for p in allp:
                    if isinstance(p, dict) and (p.get("summonerName") == sname or len(allp) == 1):
                        me = p
                        owned = [i.get("itemID") for i in p.get("items", []) if i.get("itemID")]
                        ateam = p.get("team")
                        alvl = p.get("level", 1)
                        break
                
                # 1. Manual Gold Calculation
                current_gold = active.get("currentGold", 0)
                inv_value = sum([self.item_costs.get(str(iid), {}).get("total_cost", 0) for iid in owned])
                live_total_gold = current_gold + inv_value

                # 2. Matchup Discovery (Smart Swap with Threat Inertia)
                opp_name = self.locked_matchup if self.locked_matchup else "Unknown"
                
                if game_time > 14 or not self.locked_matchup:
                    potential_threat = "Unknown"
                    for p in allp:
                        pos = str(p.get("teamPosition", p.get("individualPosition", "UNKNOWN"))).upper()
                        if (pos in ["TOP", "TOPLANE"]) and p.get("team") != ateam:
                            potential_threat = p.get("championName"); break
                    
                    if game_time > 14:
                        enemies = [p for p in allp if p.get("team") != ateam]
                        if enemies:
                            enemies.sort(key=lambda x: (x.get("level", 0), x.get("stats", {}).get("totalGold", 0)), reverse=True)
                            top_threat = enemies[0]
                            top_threat_name = top_threat.get("championName")
                            top_threat_gold = top_threat.get("stats", {}).get("totalGold", 0)
                            top_threat_lvl = top_threat.get("level", 0)
                            
                            # Threat Inertia Rule
                            if self.current_boss_name == "Unknown":
                                self.current_boss_name = top_threat_name
                                self.current_boss_gold = top_threat_gold
                            else:
                                # Find current boss stats
                                current_boss_stats = next((e for e in enemies if e.get("championName") == self.current_boss_name), None)
                                cb_gold = current_boss_stats.get("stats", {}).get("totalGold", 0) if current_boss_stats else self.current_boss_gold
                                cb_lvl = current_boss_stats.get("level", 0) if current_boss_stats else 0
                                
                                # Only swap if the new threat is +1 Level OR +1500 Gold richer than the current boss
                                if top_threat_name != self.current_boss_name:
                                    if top_threat_lvl > cb_lvl or top_threat_gold > (cb_gold + 1500):
                                        self.current_boss_name = top_threat_name
                                        self.current_boss_gold = top_threat_gold
                                        
                            opp_name = self.current_boss_name
                    else:
                        opp_name = potential_threat if potential_threat != "Unknown" else opp_name
                
                # 3. Rich Feature Extraction
                clean_opp = opp_name.replace(" ", "").replace("'", "").lower()
                opp_tags = self.kb.get(clean_opp, {"is_healer": 0, "has_shields": 0, "is_aa_heavy": 0, "is_tanky": 0})
                
                core_count = sum(1 for iid in owned if str(iid) in self.core_items)
                
                team_healers, team_aa, team_tanks = 0, 0, 0
                enemy_dnas = []
                enemy_golds = []
                for p in allp:
                    if p.get("team") != ateam:
                        p_name = p.get("championName", "")
                        tags = self.kb.get(p_name.lower(), {})
                        team_healers += tags.get("is_healer", 0)
                        team_aa += tags.get("is_aa_heavy", 0)
                        team_tanks += tags.get("is_tanky", 0)
                        inv = [i.get("itemID") for i in p.get("items", []) if i.get("itemID")]
                        enemy_dnas.append(self.calculate_dna(p_name, p.get("level", 1), inv))
                        enemy_golds.append(p.get("stats", {}).get("totalGold", 0))
                
                avg_enemy_dna = [sum(stat) / max(len(enemy_dnas), 1) for stat in zip(*enemy_dnas)] if enemy_dnas else [0.0] * 9
                my_dna = self.calculate_dna("Yorick", alvl, owned)
                
                # Enemy Snowball Awareness
                max_enemy_gold = max(enemy_golds) if enemy_golds else 0
                enemy_snowball_ratio = max_enemy_gold / max(live_total_gold, 1)

                # Gold Velocity
                self.history.append(live_total_gold)
                if len(self.history) > 10: 
                    self.history.pop(0)
                gold_velocity = live_total_gold - self.history[0]

                # 4. Construct Tensors
                my_id_t = torch.tensor([self.champ_to_idx.get("yorick", 0)], dtype=torch.long).to(self.device)
                opp_id_t = torch.tensor([self.champ_to_idx.get(clean_opp, 0)], dtype=torch.long).to(self.device)
                
                keystone = active.get("fullRunes", {}).get("keystone", {}).get("id", 0)
                rune_idx = config.RUNE_MAP.get(keystone, 0)
                rune_t = torch.tensor([rune_idx], dtype=torch.long).to(self.device)
                
                # Context must be exactly 12 items for total dim 61
                context = [
                    game_time / 40.0, 
                    live_total_gold / 20000.0, 
                    gold_velocity / 5000.0, 
                    core_count / 5.0, 
                    opp_tags.get("is_healer", 0), 
                    opp_tags.get("has_shields", 0), 
                    opp_tags.get("is_aa_heavy", 0), 
                    opp_tags.get("is_tanky", 0),
                    team_healers / 5.0, 
                    team_aa / 5.0, 
                    team_tanks / 5.0,
                    enemy_snowball_ratio / 5.0
                ]
                num_feats = context + my_dna + avg_enemy_dna + self.calculate_dist_vector(owned)
                num_t = torch.tensor([num_feats], dtype=torch.float32).to(self.device)

                # 5. Format Inventory for Model (Direct Vision)
                inv_ids = [int(i) for i in owned][:6]
                while len(inv_ids) < 6: inv_ids.append(0)
                inv_ids_t = torch.tensor([inv_ids], dtype=torch.long).to(self.device)

                # 6. Masking Logic
                imask = torch.zeros(self.vocab_size).to(self.device)
                ostrs = [str(o) for o in owned]
                for iids, idx in self.yorick_vocab.items():
                    if iids in ostrs: 
                        imask[idx] = 1
                        continue
                    # Group exclusions (Boots, Hydra, etc.)
                    for gn, gids in self.expert["groups"].items():
                        g_ids_str = [str(x) for x in gids]
                        if iids in g_ids_str and any(o in g_ids_str for o in ostrs): 
                            imask[idx] = 1
                            break

                with torch.no_grad():
                    clog, ilog = self.model(my_id_t, opp_id_t, rune_t, inv_ids_t, num_t, item_mask=imask.unsqueeze(0))
                    
                    probs = torch.nn.functional.softmax(ilog[0], dim=0)
                    ii = torch.argmax(probs).item()
                    ci = torch.argmax(clog, dim=1).item()
                    ai_n = self.id_to_name.get(str(self.inv_vocab.get(str(ii), 0)), "Unknown")
                    
                    res = {
                        "strategy": f"{self.cluster_names.get(str(ci), 'STRAT')}", 
                        "item": f"AI: {ai_n}", 
                        "matchup": opp_name,
                        "confidence": f"{probs[ii].item()*100:.1f}%"
                    }

                    # Session Logging
                    current_int_min = int(game_time)
                    if current_int_min > self.last_log_min:
                        log_entry = {
                            "minute": current_int_min,
                            "gold": current_gold,
                            "total_gold": live_total_gold,
                            "level": alvl,
                            "inventory": [self.id_to_name.get(str(i), str(i)) for i in owned],
                            "matchup": opp_name,
                            "ai_prediction": ai_n,
                            "ai_confidence": res["confidence"],
                            "strat": res["strategy"]
                        }
                        self.session_log.append(log_entry)
                        self.last_log_min = current_int_min
                        self.save_session_log()

                    self.recommendation_ready.emit(res)
            except Exception as e: 
                print(f"[!] HUD Error: {e}")
                traceback.print_exc()
            time.sleep(3)

class LTAOverlay(QWidget):
    def __init__(self, locked_matchup=None):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.layout = QVBoxLayout()
        
        self.header = QLabel("MAIDEN'S BRAIN V2.0 (EXPERT MLP)")
        self.header.setStyleSheet("color: #FF00FF; font-size: 13px; font-weight: bold; background-color: rgba(0, 0, 0, 220); padding: 5px;")
        
        self.strat_lbl = QLabel("Strategy...")
        self.strat_lbl.setStyleSheet("color: #AAAAAA; font-size: 11px; background-color: rgba(20, 20, 20, 200); padding-left: 5px;")
        
        self.item_lbl = QLabel("Waiting...")
        self.item_lbl.setStyleSheet("color: #00FF00; font-size: 16px; font-weight: bold; background-color: rgba(0, 0, 0, 200); padding: 10px; border-left: 5px solid #00FF00;")
        
        self.why_lbl = QLabel("Reasoning...")
        self.why_lbl.setStyleSheet("color: #00FFFF; font-size: 11px; font-style: italic; background-color: rgba(0, 0, 0, 180); padding: 5px;")
        
        self.layout.addWidget(self.header)
        self.layout.addWidget(self.strat_lbl)
        self.layout.addWidget(self.item_lbl)
        self.layout.addWidget(self.why_lbl)
        
        self.setLayout(self.layout)
        self.setGeometry(50, 150, 450, 140)
        
        self.inference_thread = InferenceThread(locked_matchup)
        self.inference_thread.recommendation_ready.connect(self.update_ui)
        self.inference_thread.start()

    def update_ui(self, data):
        self.strat_lbl.setText(f"STRAT: {data['strategy']} | CONF: {data['confidence']}")
        self.item_lbl.setText(data['item'])
        self.why_lbl.setText(f"Targeting Matchup: {data['matchup']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--matchup", type=str, default=None)
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    overlay = LTAOverlay(locked_matchup=args.matchup)
    overlay.show()
    sys.exit(app.exec_())
