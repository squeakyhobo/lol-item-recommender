import sys
import os
import json
import time
import torch
import numpy as np
import pyttsx3
import math
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Add brain and eye to path
BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
sys.path.append(os.path.join(BASE_DIR, "scripts", "brain"))
sys.path.append(os.path.join(BASE_DIR, "scripts", "eye"))

from model_yorick import YorickBrain
from live_scraper import LiveScraper
from wave_agent import create_wave_coach

DATA_DIR = os.path.join(BASE_DIR, "data")
RUNE_MAP = {8010: 1, 8437: 2, 8229: 3, 8992: 4, 8230: 5, 8112: 6, 8369: 7, 8214: 8, 8008: 9}

class WaveCoachThread(QThread):
    """
    The Strategic Watchman (V7.1)
    No-hallucination logic + Minimap Awareness.
    """
    tip_ready = pyqtSignal(str)

    def __init__(self, parent_thread):
        super().__init__()
        self.parent = parent_thread 
        self.coach = create_wave_coach()
        self.dead_objectives = set()

    def get_spatial_context(self, me_data):
        pos = me_data.get("position", {"x": 0, "y": 0})
        x, y = pos.get("x", 0), pos.get("y", 0)
        zone = "JUNGLE"
        if y > 11000 and x < 4000: zone = "TOP LANE"
        elif x > 11000 and y < 4000: zone = "BOT LANE"
        elif abs(x - y) < 2000: zone = "MID LANE"
        danger = "SAFE"
        if x > 10000 or y > 10000: danger = "PUSHED DEEP / OVEREXTENDED"
        return zone, danger

    def get_minimap_eyes(self, data, active_team):
        """Calculates which enemies are visible and if the player is in danger."""
        visible_enemies = []
        missing_enemies = []
        for p in data.get("allPlayers", []):
            if p.get("team") != active_team:
                # In the Riot API, if an enemy is in Fog of War, their stats don't update
                # but we can check if they have a position or were seen recently
                if p.get("isDead"): 
                    continue
                
                # Logic: If 'level' is 0 or 'position' is missing, they are in Fog
                if p.get("level", 0) > 0:
                    visible_enemies.append(p.get("championName"))
                else:
                    missing_enemies.append(p.get("championName"))
        return visible_enemies, missing_enemies

    def run(self):
        while True:
            if hasattr(self.parent, 'last_full_data') and self.parent.last_full_data and self.coach:
                try:
                    data = self.parent.last_full_data
                    active = data.get("activePlayer", {})
                    game_time = data.get("gameData", {}).get("gameTime", 0) / 60
                    
                    # 1. Update Dead Objectives
                    events = data.get("events", {}).get("Events", [])
                    for e in events:
                        n = e.get("EventName", "")
                        if "Horde" in n or "Grubs" in n: self.dead_objectives.add("Void Grubs")
                        if "Herald" in n: self.dead_objectives.add("Rift Herald")
                        if "Dragon" in n: self.dead_objectives.add("Dragon")

                    # 2. Minimap Vision & Overextension
                    s_name = active.get("summonerName")
                    me = next((p for p in data.get("allPlayers", []) if p.get("summonerName") == s_name), {})
                    zone, danger = self.get_spatial_context(me)
                    visible, missing = self.get_minimap_eyes(data, me.get("team"))
                    
                    # 3. Matchup Info
                    matchup = "Unknown"
                    for p in data.get("allPlayers", []):
                        if p.get("teamPosition") == "TOP" and p.get("team") != active.get("team"):
                            matchup = f"Yorick vs {p.get('championName')}"
                            break

                    # 3. Calculate Relative Lead (vs Lane Opponent)
                    me_gold = active.get('currentGold',0) + active.get('stats',{}).get('totalGold',0)
                    enemy_top_gold = 0
                    for p in data.get("allPlayers", []):
                        if p.get("teamPosition") == "TOP" and p.get("team") != active.get("team"):
                            # Estimate enemy gold from their items + kills if possible, 
                            # or just use their level as a proxy if gold isn't available
                            enemy_top_gold = p.get('level', 1) * 500 # Simple proxy for relative power
                            break
                    
                    gold_gap = "Even"
                    if me.get('level',1) > p.get('level',1) + 1: gold_gap = "Massive Lead"
                    elif me.get('level',1) > p.get('level',1): gold_gap = "Slight Lead"
                    elif me.get('level',1) < p.get('level',1): gold_gap = "Behind"

                    # 4. 'Relative Tactical' Prompt
                    prompt = f"""
                    ROLE: Professional LoL Tactical Coach.
                    SITUATION: {matchup}
                    STATE: Minute {game_time:.1f} | HP: {hp_pct:.0f}% | Power Status: {gold_gap}
                    VISION: Visible: {visible} | ⚠️ MISSING: {missing}
                    OBJECTIVES: Dead: {list(self.dead_objectives)}
                    
                    TASK: Check the 'Manuals' for the best move in THIS specific situation.
                    - If Behind: Focus on safe farm/scaling.
                    - If Leading: Focus on taking towers/plates or roaming.
                    - If Jungle Missing & Overextended: Warn to back off.
                    
                    Give 1 short, actionable tactical tip (10 words max).
                    """
                    response = self.coach.run(prompt)
                    self.tip_ready.emit(str(response))
                except Exception as e:
                    print(f"[!] Coach Error: {e}")
            time.sleep(60)

class InferenceThread(QThread):
    recommendation_ready = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.last_full_data = None
        with open(os.path.join(DATA_DIR, "item_map.json"), "r") as f: self.item_map = json.load(f)
        with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "r") as f:
            v = json.load(f); self.yorick_vocab = v["item_to_index"]; self.inv_vocab = v["index_to_item"]; self.vocab_size = v["size"]
        with open(os.path.join(DATA_DIR, "item_names.json"), "r") as f: self.id_to_name = json.load(f)
        with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f: self.item_dna = json.load(f)
        with open(os.path.join(DATA_DIR, "champion_dna.json"), "r") as f: self.champ_dna = json.load(f)
        with open(os.path.join(DATA_DIR, "expert_config.json"), "r") as f: self.expert = json.load(f)
        with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
            c = json.load(f); self.cluster_to_items = c["cluster_to_items"]; self.cluster_names = c["cluster_names"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YorickBrain(num_clusters=15, num_items=self.vocab_size, dna_dim=9).to(self.device)
        mv4 = os.path.join(BASE_DIR, "models", "yorick_brain_v4.pth")
        if os.path.exists(mv4): self.model.load_state_dict(torch.load(mv4, map_location=self.device))
        self.model.eval(); self.scraper = LiveScraper(); self.history = []
        self.log_file = os.path.join(BASE_DIR, "logs", f"hud_session_{int(time.time())}.jsonl")
        if not os.path.exists(os.path.dirname(self.log_file)): os.makedirs(os.path.dirname(self.log_file))

    def calculate_live_dna(self, name, lvl, inv):
        total = np.zeros(9)
        if name in self.champ_dna:
            c = self.champ_dna[name]; lf = (lvl - 1)
            total[0] = c["ad"]["base"] + (c["ad"]["growth"] * lf); total[2] = c["hp"]["base"] + (c["hp"]["growth"] * lf)
            total[3] = c["armor"]["base"] + (c["armor"]["growth"] * lf); total[4] = c["mr"]["base"] + (c["mr"]["growth"] * lf)
            total[5] = c["as"]["base"] + (c["as"]["growth"] * lf / 100.0); total[6] = c["ms"]["base"]
        for iid in inv:
            if str(iid) in self.item_dna:
                d = self.item_dna[str(iid)]
                total[0]+=d.get("ad",0); total[1]+=d.get("ap",0); total[2]+=d.get("hp",0); total[3]+=d.get("armor",0); total[4]+=d.get("mr",0); total[5]+=d.get("as",0); total[6]+=d.get("ms",0); total[7]+=d.get("crit",0); total[8]+=d.get("lifesteal",0)
        total[0]/=400; total[1]/=800; total[2]/=5000; total[3]/=300; total[4]/=300; total[5]/=2.5; total[6]/=600
        return total

    def get_explanation(self, cid, lane_enemy, ldna):
        """Matchup-Aware Reasoning Engine"""
        name = lane_enemy if lane_enemy else "the team"
        # 1. Check for Tankiness
        if ldna[2] > 0.4 or ldna[3] > 0.4: return f"Reason: Shredding TANK stats of {name}"
        # 2. Check for Burst
        if ldna[0] > 0.4: return f"Reason: Survival vs high damage of {name}"
        # 3. Check for AP
        if ldna[1] > 0.3: return f"Reason: Magic resistance vs {name}"
        # 4. Default to Strategy
        return f"Reason: Strategic snowballing vs {name}"

    def oracle_pick(self, cid, gold, owned, minute, ldna=None):
        items = self.cluster_to_items.get(str(cid), [])
        if not any(int(s) in owned for s in ["1054","1055","1056","1086"]) and minute < 2: items = self.cluster_to_items.get("0", [])
        candidates = []
        for iid in items:
            iid_s = str(iid)
            if iid_s not in self.item_map or iid_s in ["1111","3865","3866","3867","3008","3172"]: continue
            if int(iid) in owned or int(iid) in self.expert["banned_ids"]: continue
            conflict = False
            for gn, gids in self.expert["groups"].items():
                gs = [str(x) for x in gids]
                if iid_s in gs and any(str(o) in gs for o in owned): conflict = True; break
            if not conflict: candidates.append(iid)
        if not candidates: return "Check manual"
        scored = []
        for iid in candidates:
            dna = self.item_dna.get(str(iid), {}); bs = sum([1 for v in dna.values() if isinstance(v,(int,float)) and v>0])
            mb = 0
            if ldna is not None:
                if ldna[1]>0.3 and dna.get("mr",0)>0: mb+=5
                if ldna[0]>0.3 and dna.get("armor",0)>0: mb+=5
            scored.append({"id":iid, "leg":bs>4, "score":bs+mb})
        scored.sort(key=lambda x:(x["leg"], x["score"]), reverse=True)
        return self.id_to_name.get(str(scored[0]["id"]), str(scored[0]["id"]))

    def run(self):
        while True:
            try:
                data = self.scraper.fetch_data()
                if not data: time.sleep(3); continue
                self.last_full_data = data
                active = data.get("activePlayer", {}); game_time = data.get("gameData", {}).get("gameTime", 0)/60
                keystone = active.get("fullRunes", {}).get("keystone", {}).get("id", 0)
                rune_t = torch.tensor([[RUNE_MAP.get(keystone, 0)]], dtype=torch.long).to(self.device)
                sname = active.get("summonerName"); allp = data.get("allPlayers", []); owned = []; ateam = "ORDER"; alvl = 1
                for p in allp:
                    if p.get("summonerName") == sname or len(allp)==1:
                        owned = [i.get("itemID") for i in p.get("items", []) if i.get("itemID")]; ateam=p.get("team"); alvl=p.get("level",1); break
                
                state = {"gold": active.get("currentGold",0), "total_gold": active.get("stats",{}).get("totalGold",0), "level":alvl, "minute":game_time, "inventory":owned}
                self.history.append(state); 
                if len(self.history)>5: self.history.pop(0)
                phist = list(self.history); 
                while len(phist)<5: phist.insert(0, phist[0])
                
                pnum = torch.tensor([[[f["gold"]/5000.0, f.get("total_gold",0)/25000.0, f["level"]/20.0, f["minute"]/60.0, 0] for f in phist]], dtype=torch.float32).to(self.device)
                pdna = torch.tensor(np.array([[self.calculate_live_dna("Yorick", f["level"], f["inventory"]) for f in phist]]), dtype=torch.float32).to(self.device)
                
                odna = []; ldna_v = np.zeros(9); matchup_name = "Unknown"
                for p in allp:
                    if p.get("team") != ateam:
                        inv = [i.get("itemID") for i in p.get("items",[]) if i.get("itemID")]
                        dna = self.calculate_live_dna(p.get("championName"), p.get("level",1), inv)
                        if p.get("teamPosition") == "TOP": 
                            ldna_v = dna; matchup_name = p.get("championName"); print(f"  -> Matchup Found: vs {matchup_name}")
                        else: odna.append(dna)
                
                while len(odna)<4: odna.append(np.zeros(9))
                edna = torch.tensor(np.array([odna[:4]]), dtype=torch.float32).to(self.device)
                ldna = torch.tensor(np.array([ldna_v]), dtype=torch.float32).to(self.device)
                
                imask = torch.zeros(self.vocab_size).to(self.device); ostrs = [str(o) for o in owned]
                for iids, idx in self.yorick_vocab.items():
                    if iids in ostrs: imask[idx]=1; continue
                    for gn, gids in self.expert["groups"].items():
                        gs = [str(x) for x in gids]
                        if iids in gs and any(o in gs for o in ostrs): imask[idx]=1; break
                
                with torch.no_grad():
                    clog, ilog = self.model(pnum, pdna, edna, ldna, rune_t, item_mask=imask.unsqueeze(0))
                    cp, ci = torch.topk(torch.softmax(clog, dim=1), 1); ip, ii = torch.topk(torch.softmax(ilog, dim=1), 1)
                    cidx = ci[0][0].item(); iidx = ii[0][0].item()
                    ai_n = self.id_to_name.get(str(self.inv_vocab.get(str(iidx), 0)), "Unknown")
                    ora_n = self.oracle_pick(cidx, active.get("currentGold",0), owned, game_time, ldna=ldna_v)
                    
                    res = {
                        "strategy": f"{self.cluster_names.get(str(cidx), 'Strat').upper()}", 
                        "item": f"AI: {ai_n} | Oracle: {ora_n}", 
                        "prob": ip[0][0].item(), 
                        "explanation": self.get_explanation(cidx, matchup_name, ldna_v), 
                        "inventory": [self.id_to_name.get(str(x), str(x)) for x in owned]
                    }
                    self.recommendation_ready.emit(res)
                    with open(self.log_file, "a") as f: json.dump({"timestamp":time.time(), "results":res}, f); f.write("\n")
            except Exception as e: print(f"[!] Thread Error: {e}")
            time.sleep(3)

class LTAOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.layout = QVBoxLayout()
        self.header = QLabel("MAIDEN'S BRAIN V7.1 (MATCHUP REASONING)")
        self.header.setStyleSheet("color: #FF00FF; font-size: 13px; font-weight: bold; background-color: rgba(0, 0, 0, 220); padding: 5px;")
        self.strat_lbl = QLabel("Strategy..."); self.strat_lbl.setStyleSheet("color: #AAAAAA; font-size: 11px; background-color: rgba(20, 20, 20, 200); padding-left: 5px;")
        self.item_lbl = QLabel("Waiting..."); self.item_lbl.setStyleSheet("color: #00FF00; font-size: 16px; font-weight: bold; background-color: rgba(0, 0, 0, 200); padding: 10px; border-left: 5px solid #00FF00;")
        self.why_lbl = QLabel("Reasoning..."); self.why_lbl.setStyleSheet("color: #00FFFF; font-size: 11px; font-style: italic; background-color: rgba(0, 0, 0, 180); padding: 5px;")
        self.coach_lbl = QLabel("Coach: Initializing..."); self.coach_lbl.setStyleSheet("color: #FFA500; font-size: 12px; font-weight: bold; background-color: rgba(30, 15, 0, 220); padding: 8px; border-left: 5px solid #FFA500;")
        self.inv_lbl = QLabel("Inventory: []"); self.inv_lbl.setStyleSheet("color: #FFFFFF; font-size: 10px; background-color: rgba(50, 0, 0, 200); padding: 5px;")
        self.layout.addWidget(self.header); self.layout.addWidget(self.strat_lbl); self.layout.addWidget(self.item_lbl); self.layout.addWidget(self.why_lbl); self.layout.addWidget(self.coach_lbl); self.layout.addWidget(self.inv_lbl)
        self.setLayout(self.layout); self.setGeometry(50, 150, 450, 180)
        
        self.tts = pyttsx3.init(); self.tts.setProperty('rate', 180)
        self.inference_thread = InferenceThread()
        self.inference_thread.recommendation_ready.connect(self.update_ui)
        self.inference_thread.start()
        self.coach_thread = WaveCoachThread(self.inference_thread)
        self.coach_thread.tip_ready.connect(self.update_coach)
        self.coach_thread.start()

    def update_ui(self, data):
        self.strat_lbl.setText(data['strategy']); self.item_lbl.setText(data['item'])
        self.why_lbl.setText(data['explanation']) # NOW MATCHUP AWARE
        self.inv_lbl.setText(f"DETECTED: {', '.join(data['inventory'])}")

    def update_coach(self, tip):
        self.coach_lbl.setText(f"COACH: {tip}")
        try: self.tts.say(tip); self.tts.runAndWait()
        except: pass

if __name__ == "__main__":
    app = QApplication(sys.argv); overlay = LTAOverlay(); overlay.show(); sys.exit(app.exec_())
