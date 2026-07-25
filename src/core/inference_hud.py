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

from model_v2 import YorickMLP,MoEYorickMLP
from live_scraper import LiveScraper

DATA_DIR = config.DATA_DIR
DEBUG_LOG = os.path.join(config.LOGS_DIR, "hud_debug.jsonl")

class InferenceThread(QThread):
    recommendation_ready = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.last_full_data = None
        self.purchase_history = []
        
        # Threat Inertia Memory
        self.current_boss_name = None
        #NOTE - is this needed 
        self.current_boss_gold = 0
        
        # Load Knowledge Bases
        with open(os.path.join(DATA_DIR, "item_vocab.json"), "r") as f:
            v = json.load(f)
            self.yorick_vocab = v["item_to_index"]
            self.inv_vocab = v["index_to_item"]
            self.vocab_size = v["size"]
            self.champ_masks = v["champ_mask"]
            
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
        
        self.CORE_MAP = config.CORE_MAP
        self.core_items = None
        self.current_champ = None 
        
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate model with the latest version's architecture
        self.model = MoEYorickMLP(num_champs=200, num_runes=20, num_items=self.vocab_size, numerical_dim=99 + self.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(config.V2_MODEL_PATH, map_location=self.device))
          
        
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
        # 12 Stats + 11 Intent Bits = 23 Total DNA Slots
        total = [0.0] * 23
        clean_name = name.replace(" ", "").replace(".", "").replace("'", "").lower()
        c_info = self.kb.get(clean_name, {})
        c_id = c_info.get("id")
        if c_id and str(c_id) in self.champ_dna:
            c = self.champ_dna[str(c_id)]
            lf = (lvl - 1)
            total[0] = (c["ad"]["base"] + (c["ad"]["growth"] * lf)) / 150.0
            total[1] = 0.0 # Base AP is 0
            total[2] = (c["hp"]["base"] + (c["hp"]["growth"] * lf)) / 4000.0
            total[3] = (c["armor"]["base"] + (c["armor"]["growth"] * lf)) / 200.0
            total[4] = (c["mr"]["base"] + (c["mr"]["growth"] * lf)) / 200.0
            total[5] = (c["as"]["base"] + (c["as"]["growth"] * lf)) / 2.0
            total[6] = (c["ms"]["base"]) / 500.0
            total[7] = (c["crit"]["base"] + (c["crit"]["growth"] * lf)) / 1.0
            total[8] = 0.0 # Base lifesteal is 0
            total[9] = 0.0 # Haste
            total[10] = 0.0 # Armor Pen
            total[11] = 0.0 # Magic Pen

        intent_keys = [
            "is_anti_heal", "is_lifeline", "is_burn", "is_penetration", 
            "is_lethality", "is_spellblade", "is_on_hit", "is_slow", 
            "is_shield_breaker", "is_tenacity", "is_aoe"
        ]

        for iid in inv:
            if str(iid) in self.item_dna:
                d = self.item_dna[str(iid)]
                total[0] += d.get("ad", 0) / 150.0
                total[1] += d.get("ap", 0) / 400.0
                total[2] += d.get("hp", 0) / 4000.0
                total[3] += d.get("armor", 0) / 200.0
                total[4] += d.get("mr", 0) / 200.0
                total[5] += d.get("as", 0) / 2.0
                total[6] += d.get("ms", 0) / 500.0
                total[7] += d.get("crit", 0) / 1.0
                total[8] += d.get("lifesteal", 0) / 1.0
                total[9] += d.get("haste", 0) / 100.0
                total[10] += d.get("armor_pen", 0) / 50.0
                total[11] += d.get("magic_pen", 0) / 50.0

                for i, key in enumerate(intent_keys):
                    if d.get(key, 0) == 1:
                        total[12 + i] = 1.0
        return total

    def calculate_dist_vector(self, inv):
        dist = [1.0] * self.vocab_size
        inv_str = [str(i) for i in inv]
        for i_id, i_idx in self.yorick_vocab.items():
            if i_id in inv_str: 
                dist[i_idx] = 0.0
            else:
                c_info = self.item_costs.get(i_id, {})
                # Properly sum duplicate components (e.g. two Long Swords)
                val = sum([c_info.get("components", {}).get(iid, 0) for iid in inv_str])
                dist[i_idx] = max(0.0, 1.0 - (val / max(c_info.get("total_cost", 3000), 1)))
        return dist

    def get_real_gold(self, player_dict):
        """Calculates actual net worth since API totalGold is often bugged at 0."""
        current_gold = player_dict.get("currentGold", 0) # Only works for active player, but safe to default 0
        inv = [i.get("itemID") for i in player_dict.get("items", []) if i.get("itemID")]
        inv_value = sum([self.item_costs.get(str(iid), {}).get("total_cost", 0) for iid in inv])
        
        # If API gives us a real number, use it if it's higher than our estimate
        api_gold = player_dict.get("stats", {}).get("totalGold", 0)
        return max(api_gold, inv_value + current_gold)

    def run(self):
        while True:
            try:
                #print(self.model.dtype())
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
                        raw_owned = [str(i.get("itemID")) for i in p.get("items", []) if i.get("itemID")]
                        
                        # Maintain chronological purchase history (with duplicate awareness)
                        from collections import Counter
                        owned_counts = Counter(raw_owned)
                        new_history = []
                        
                        # 1. Retain already-known purchases in chronological order if still owned
                        for iid in self.purchase_history:
                            if owned_counts[iid] > 0:
                                new_history.append(iid)
                                owned_counts[iid] -= 1
                        
                        # 2. Append newly acquired items/duplicates
                        for iid in raw_owned:
                            if owned_counts[iid] > 0:
                                new_history.append(iid)
                                owned_counts[iid] -= 1
                        
                        self.purchase_history = new_history
                        owned = list(self.purchase_history)
                        
                        ateam = p.get("team")
                        alvl = p.get("level", 1)
                        # New: Detect Current Champion
                        self.current_champ = p.get("championName")
                        clean_champ = self.current_champ.lower().replace(" ", "").replace(".", "").replace("'", "")
                        self.core_items = self.CORE_MAP.get(clean_champ) or []
                        break
                
                # 1. Manual Gold Calculation
                live_total_gold = self.get_real_gold(active)
                current_gold = active.get("currentGold", 0)

                # 2. Matchup Discovery (Automatically Detect enemy Top Laner)
                opp_name = "Unknown"

                # Identify Enemies
                enemies = [p for p in allp if p.get("team") != ateam]
                enemies.sort(key=lambda x: (self.get_real_gold(x), x.get("level", 0)), reverse=True)

                # Try to find specific enemy TOP laner first
                for p in enemies:
                    pos = str(p.get("position", "UNKNOWN")).upper()
                    if pos in ["TOP", "TOPLANE"]:
                        opp_name = p.get("championName")
                        break

                # Fallback: If no TOP found, default to the first enemy
                if opp_name == "Unknown" and enemies:
                    opp_name = enemies[0].get("championName", "Unknown")
                # 3. Rich Feature Extraction
                clean_opp = opp_name.replace(" ", "").replace(".", "").replace("'", "").lower()
                opp_tags = self.kb.get(clean_opp, {"is_healer": 0, "has_shields": 0, "is_aa_heavy": 0, "is_tanky": 0, "is_cc_heavy": 0, "is_mobile": 0, "archetype": 0})
                
                core_count = sum(1 for iid in owned if str(iid) in self.core_items)
                
                team_healers, team_aa, team_tanks, team_cc, team_mobile = 0, 0, 0, 0, 0
                enemy_dnas = []
                enemy_golds = []
                enemy_threat_scores = []
                enemy_team_composition_tags = {
                    "num_tanks": 0, "num_healers": 0, "num_shielders": 0, "num_cc_heavy": 0,
                    "num_assassins": 0, "num_adc": 0, "num_fighters": 0, "num_mages": 0, "num_supports": 0
                }
                enemy_ad_total, enemy_ap_total = 0, 0

                for p in allp:
                    p_name = p.get("championName", "")
                    p_clean = p_name.replace(" ", "").replace("'", "").lower()
                    tags = self.kb.get(p_clean, {})
                    if p.get("team") != ateam:
                        
                        
                        
                        inv = [i.get("itemID") for i in p.get("items", []) if i.get("itemID")]
                        dna = self.calculate_dna(p_name, p.get("level", 1), inv)
                        enemy_dnas.append(dna)
                        
                        egold = self.get_real_gold(p)
                        enemy_golds.append(egold)
                        
                        # Standardized Threat Score from config
                        t_score = config.calculate_threat_score(egold, p.get("level", 1), tags)
                        enemy_threat_scores.append(t_score)

                        # Aggregate enemy team composition tags
                        if tags.get("archetype") == 0: enemy_team_composition_tags["num_tanks"] += 1
                        elif tags.get("archetype") == 1: enemy_team_composition_tags["num_healers"] += 1
                        elif tags.get("archetype") == 2: enemy_team_composition_tags["num_fighters"] += 1
                        elif tags.get("archetype") == 3: enemy_team_composition_tags["num_mages"] += 1
                        elif tags.get("archetype") == 4: enemy_team_composition_tags["num_assassins"] += 1
                        elif tags.get("archetype") == 5: enemy_team_composition_tags["num_adc"] += 1
                        elif tags.get("archetype") == 6: enemy_team_composition_tags["num_supports"] += 1

                        enemy_team_composition_tags["num_healers"] += tags.get("is_healer", 0)
                        enemy_team_composition_tags["num_shielders"] += tags.get("has_shields", 0)
                        enemy_team_composition_tags["num_cc_heavy"] += tags.get("is_cc_heavy", 0)

                        enemy_ad_total += dna[0] * 150.0
                        enemy_ap_total += dna[1] * 400.0
                    # get tags for your team
                    elif(p.get("summonerName") != sname):
                        team_healers += tags.get("is_healer", 0)
                        team_aa += tags.get("is_aa_heavy", 0)
                        team_tanks += tags.get("is_tanky", 0)
                        team_cc += tags.get("is_cc_heavy", 0)
                        team_mobile += tags.get("is_mobile", 0)

                avg_enemy_dna = [sum(stat) / max(len(enemy_dnas), 1) for stat in zip(*enemy_dnas)] if enemy_dnas else [0.0] * 23
                
                # New: Highest Threat DNA (Synchronized with Training)
                threat_dna = [0.0] * 23
                threat_champ = "Unknown"
                if enemy_threat_scores:
                    max_idx = enemy_threat_scores.index(max(enemy_threat_scores))
                    threat_dna = enemy_dnas[max_idx]
                    unsorted_enemies = [p for p in allp if p.get("team") != ateam]
                    if max_idx < len(unsorted_enemies):
                        threat_champ = unsorted_enemies[max_idx].get("championName", "Unknown")

                my_dna = self.calculate_dna(self.current_champ, alvl, owned)
                
                # Enemy Snowball Awareness
                min_expected_gold = 500.0 + (game_time / 60.0) * 122.0
                max_enemy_gold = max(enemy_golds) if enemy_golds else 0
                max_enemy_gold = max(max_enemy_gold, min_expected_gold)
                enemy_snowball_ratio = max_enemy_gold / max(live_total_gold, 1)

                # Gold Velocity (Training uses a 3-minute window)
                self.history.append(live_total_gold)
                if len(self.history) > 60: # 60 ticks * 3 seconds = 180 seconds (3 minutes)
                    self.history.pop(0)
                gold_velocity = live_total_gold - self.history[0]

                # NEW: Hubris / Snowball detection features
                # Enemies' unspent gold is hidden by Riot API, causing avg_enemy_gold to be 0 at minute 0.
                min_expected_gold = 500.0 + (game_time / 60.0) * 122.0
                avg_enemy_gold = sum(enemy_golds) / max(len(enemy_golds), 1)
                avg_enemy_gold = max(avg_enemy_gold, min_expected_gold)
                
                player_snowball_factor = live_total_gold / max(avg_enemy_gold, 1.0)
                kda_proxy = 1.0 if gold_velocity >= 300 else 0.0

                # Damage Split ratios
                total_damage_potential = enemy_ad_total + enemy_ap_total
                enemy_ad_ratio = enemy_ad_total / max(1.0, total_damage_potential)
                enemy_ap_ratio = enemy_ap_total / max(1.0, total_damage_potential)

                # 4. Construct Tensors
                clean_champ = self.current_champ.lower().replace(" ", "").replace(".", "").replace("'", "")
                my_id_t = torch.tensor([self.champ_to_idx.get(clean_champ, 0)], dtype=torch.long).to(self.device)
                opp_id_t = torch.tensor([self.champ_to_idx.get(clean_opp, 0)], dtype=torch.long).to(self.device)
                
                keystone = active.get("fullRunes", {}).get("keystone", {}).get("id", 0)
                rune_idx = config.RUNE_MAP.get(keystone, 0)
                rune_t = torch.tensor([rune_idx], dtype=torch.long).to(self.device)
                core_percent =min(1.0, core_count / 3.0)
                
                context = [
                    game_time / 40.0, 
                    live_total_gold / 20000.0, 
                    gold_velocity / 5000.0, 
                    core_percent,
                    opp_tags.get("is_healer", 0), 
                    opp_tags.get("has_shields", 0), 
                    opp_tags.get("is_aa_heavy", 0), 
                    opp_tags.get("is_tanky", 0),
                    opp_tags.get("is_cc_heavy", 0),
                    opp_tags.get("is_mobile", 0),
                    opp_tags.get("archetype", 0) / 5.0,
                    team_healers / 5.0, 
                    team_aa / 5.0, 
                    team_tanks / 5.0,
                    team_cc / 5.0,
                    team_mobile / 5.0,
                    enemy_snowball_ratio / 5.0
                ]
                
                # Extend context with latest v2.5 features
                context.extend(list(enemy_team_composition_tags.values()))
                context.extend([enemy_ad_ratio, enemy_ap_ratio])
                context.append(player_snowball_factor / 3.0)
                context.append(kda_proxy)

                num_feats = context + my_dna + avg_enemy_dna + threat_dna + self.calculate_dist_vector(owned)
                num_t = torch.tensor([num_feats], dtype=torch.float32).to(self.device)

                # 5. Format Inventory for Model (Direct Vision)
                inv_ids = [int(i) for i in owned][:6]
                while len(inv_ids) < 6: inv_ids.append(0)
                inv_ids_t = torch.tensor([inv_ids], dtype=torch.long).to(self.device)

                # 6. Masking Logic
                imask = torch.zeros(self.vocab_size).to(self.device)
                champ_mask = torch.tensor(self.champ_masks[clean_champ]).to(self.device)
                
                
                ostrs = [str(o) for o in owned]
                starter_items = {"1054", "1055", "1056", "1082", "1083"}
                for iids, idx in self.yorick_vocab.items():
                    if iids in ostrs: 
                        imask[idx] = 1
                        continue
                    if iids in starter_items and game_time > 5.0:
                        imask[idx] = 1
                        continue
                    # Group exclusions (Boots, Hydra, etc.)
                    for gn, gids in self.expert["groups"].items():
                        g_ids_str = [str(x) for x in gids]
                        if gn == "BOOTS":
                            has_upgraded_boots = any(o in g_ids_str and o != "1001" for o in ostrs)
                            if has_upgraded_boots:
                                if iids in g_ids_str:
                                    imask[idx] = 1
                                    break
                            elif "1001" in ostrs and iids == "1001":
                                imask[idx] = 1
                                break
                        else:
                            if iids in g_ids_str and any(o in g_ids_str for o in ostrs): 
                                imask[idx] = 1
                                break
                
                combined_mask = imask.bool() | champ_mask.bool()

                 # --- DEBUG PRINTS ---
                total_banned = combined_mask.sum().item()
                #print(f"[*] DEBUG: Total items masked out: {total_banned} / {self.vocab_size}")

                with torch.no_grad():
                    clog, ilog,_ ,_= self.model(my_id_t, opp_id_t, rune_t, inv_ids_t, num_t, item_mask=combined_mask.unsqueeze(0))
                    print(ilog.shape)
                    #sfot boost core items 
                    if(core_count<min(2,len(self.core_items)) and game_time>3.0):
                        for core_id in self.core_items:
                            #get the item id in core items
                            core_index =self.yorick_vocab[core_id]
                            #get the idnxex
                            #boost the index by a scalar
                            ilog[0][core_index]+=8.0
                    #print(ilog.shape)
                        
                    
                    probs = torch.nn.functional.softmax(ilog[0], dim=0)
                    top_probs, top_indices = torch.topk(probs, k=3)
                    ii = top_indices.tolist()
                    ci = torch.argmax(clog, dim=1).item()
                    ai_ns =[]

                    for index in ii:
                        ai_ns.append(self.id_to_name.get(str(self.inv_vocab.get(str(index), 0)), "Unknown"))
                    
                    
                    res = {
                        "strategy": f"{self.cluster_names.get(str(ci), 'STRAT')}", 
                        "items": f"AI: {ai_ns}", 
                        "matchup": opp_name,
                        "confidence": f"{top_probs.tolist()}%",
                        "biggest threat": threat_champ
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
                            "biggest threat": threat_champ,
                            "ai_prediction": ai_ns,
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
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.layout = QVBoxLayout()

        self.header = QLabel(f"{config.MODEL_VERSION} (MLP)")
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

        self.inference_thread = InferenceThread()
        self.inference_thread.recommendation_ready.connect(self.update_ui)
        self.inference_thread.start()

    def update_ui(self, data):
        self.strat_lbl.setText(f"STRAT: {data['strategy']} | CONF: {data['confidence']}")
        self.item_lbl.setText(data['items'])
        self.why_lbl.setText(f"Targeting Matchup: {data['matchup']} , biigest threat: {data['biggest threat']}")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = LTAOverlay()
    overlay.show()
    sys.exit(app.exec_())
