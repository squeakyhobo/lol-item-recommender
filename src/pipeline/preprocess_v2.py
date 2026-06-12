import os
import json
import sys
import random

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
import config

DATA_DIR = config.DATA_DIR
MATCH_DIR = os.path.join(DATA_DIR, "yorick_games", "matches")
TIMELINE_DIR = os.path.join(DATA_DIR, "yorick_games", "timelines")

class V2Preprocessor:
    def __init__(self):
        print("[*] Initializing V2 Preprocessor...")
        
        # Load Knowledge Base
        kb_path = os.path.join(DATA_DIR, "champion_knowledge.json")
        if not os.path.exists(kb_path):
            print("[!] Fatal: champion_knowledge.json missing. Run generate_knowledge_base.py first.")
            sys.exit(1)
        with open(kb_path, "r") as f:
            self.kb = json.load(f)

        # Load Vocab
        with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "r") as f:
            v = json.load(f)
            self.vocab = v["item_to_index"]
            self.vocab_size = v["size"]
            
        with open(os.path.join(DATA_DIR, "valid_targets.json"), "r") as f:
            self.valid_targets = set(json.load(f))

        with open(os.path.join(DATA_DIR, "item_costs.json"), "r") as f:
            self.costs = json.load(f)

        with open(os.path.join(DATA_DIR, "champion_dna.json"), "r") as f:
            self.champ_dna = json.load(f)
        with open(os.path.join(DATA_DIR, "item_dna.json"), "r") as f:
            self.item_dna = json.load(f)

        # Map Champion names to 0-200 for embeddings
        self.champ_to_idx = {name: i for i, name in enumerate(sorted(self.kb.keys()))}

        # Common Yorick Core Items (To track progress)
        self.core_items = ["3078", "6692", "3153", "6631", "3181"]

    def get_champ_tags(self, champ_name):
        clean_name = champ_name.replace(" ", "").replace("'", "").lower()
        return self.kb.get(clean_name, {
            "is_healer": 0, "has_shields": 0, "is_aa_heavy": 0, 
            "is_burst_threat": 0, "is_ranged": 0, "is_tanky": 0, "archetype": 2
        })

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

    def process_match(self, match_id):
        try:
            with open(os.path.join(MATCH_DIR, f"{match_id}.json"), "r") as f: m_data = json.load(f)
            with open(os.path.join(TIMELINE_DIR, f"{match_id}.json"), "r") as f: t_data = json.load(f)
        except: return []

        yorick_id = None
        enemy_team = None
        champ_map = {}
        enemy_roles = {}
        keystone_id = 0
        
        for p in m_data.get("info", {}).get("participants", []):
            p_id = p["participantId"]
            name = p["championName"]
            champ_map[p_id] = name
            if name == "Yorick":
                yorick_id = p_id
                enemy_team = 200 if p["teamId"] == 100 else 100
                for r in p.get("perks", {}).get("styles", []):
                    if r.get("description") == "primaryStyle":
                        keystone_id = r.get("selections", [{}])[0].get("perk", 0)
            else:
                enemy_roles[p_id] = p.get("teamPosition", "UNKNOWN")

        if not yorick_id: return []

        frames = t_data.get("info", {}).get("frames", [])
        snapshots = []
        inventories = {i: [] for i in range(1, 11)}
        gold_history = [] 

        major_purchases = []
        for f_idx, frame in enumerate(frames):
            for event in frame.get("events", []):
                if event.get("type") == "ITEM_PURCHASED" and event.get("participantId") == yorick_id:
                    iid = str(event.get("itemId"))
                    if iid in self.vocab and int(iid) in self.valid_targets:
                        major_purchases.append({"min": f_idx, "id": iid})

        for frame_idx, frame in enumerate(frames):
            p_frame = frame.get("participantFrames", {}).get(str(yorick_id), {})
            total_gold = p_frame.get("totalGold", 0)
            gold_history.append(total_gold)
            
            gold_velocity = total_gold - gold_history[max(0, frame_idx - 3)]

            target_item = None
            for buy in major_purchases:
                if buy["min"] >= frame_idx:
                    if buy["id"] not in inventories[yorick_id]:
                        target_item = buy["id"]
                        break

            if target_item:
                core_count = sum(1 for iid in inventories[yorick_id] if str(iid) in self.core_items)
                
                lane_tags = {"is_healer": 0, "has_shields": 0, "is_aa_heavy": 0, "is_tanky": 0}
                opp_id = 0
                for i in range(1, 11):
                    if champ_map.get(i) and (100 if i <= 5 else 200) == enemy_team and enemy_roles.get(i) == "TOP":
                        opp_name = champ_map.get(i)
                        opp_id = self.champ_to_idx.get(opp_name.lower(), 0)
                        lane_tags = self.get_champ_tags(opp_name)
                        break

                team_healers, team_aa, team_tanks = 0, 0, 0
                enemy_dnas = []
                enemy_golds = []
                for i in range(1, 11):
                    if champ_map.get(i) and (100 if i <= 5 else 200) == enemy_team:
                        tags = self.get_champ_tags(champ_map.get(i))
                        team_healers += tags["is_healer"]
                        team_aa += tags["is_aa_heavy"]
                        team_tanks += tags["is_tanky"]
                        ef = frame.get("participantFrames", {}).get(str(i), {})
                        enemy_dnas.append(self.calculate_dna(champ_map[i], ef.get("level", 1), inventories[i]))
                        enemy_golds.append(ef.get("totalGold", 0))
                
                avg_enemy_dna = [sum(stat)/max(len(enemy_dnas),1) for stat in zip(*enemy_dnas)] if enemy_dnas else [0]*9
                my_dna = self.calculate_dna("Yorick", p_frame.get("level", 1), inventories[yorick_id])

                # New: Enemy Snowball Aware
                max_enemy_gold = max(enemy_golds) if enemy_golds else 0
                enemy_snowball_ratio = max_enemy_gold / max(total_gold, 1)

                # p_dist
                dist = [1.0] * self.vocab_size
                inv_str = [str(i) for i in inventories[yorick_id]]
                for i_id, i_idx in self.vocab.items():
                    if i_id in inv_str: dist[i_idx] = 0.0
                    else:
                        c_info = self.costs.get(i_id, {})
                        val = sum([cost for cid, cost in c_info.get("components", {}).items() if cid in inv_str])
                        dist[i_idx] = max(0.0, 1.0 - (val / max(c_info.get("total_cost", 3000), 1)))

                # Inventory slots
                inv_ids = [int(iid) for iid in inventories[yorick_id]][:6]
                while len(inv_ids) < 6: inv_ids.append(0)

                row = {
                    "my_id": self.champ_to_idx.get("yorick", 0),
                    "opp_id": opp_id,
                    "rune_id": config.RUNE_MAP.get(keystone_id, 0),
                    "inventory": inv_ids,
                    "minute": frame_idx / 40.0,
                    "total_gold": total_gold / 20000.0,
                    "gold_velocity": gold_velocity / 5000.0,
                    "core_progress": core_count / 5.0,
                    "lane_healer": lane_tags["is_healer"],
                    "lane_shield": lane_tags["has_shields"],
                    "lane_aa": lane_tags["is_aa_heavy"],
                    "lane_tank": lane_tags["is_tanky"],
                    "team_healers": team_healers / 5.0,
                    "team_aa": team_aa / 5.0,
                    "team_tanks": team_tanks / 5.0,
                    "enemy_snowball": enemy_snowball_ratio / 5.0, # Normalizing
                    "my_dna": my_dna,
                    "enemy_dna": avg_enemy_dna,
                    "p_dist": dist,
                    "target_item": int(target_item)
                }
                snapshots.append(row)

            for event in frame.get("events", []):
                p_id = event.get("participantId")
                if p_id and 1 <= p_id <= 10:
                    if event.get("type") == "ITEM_PURCHASED":
                        inventories[p_id].append(str(event.get("itemId")))
                    elif event.get("type") in ["ITEM_SOLD", "ITEM_DESTROYED"]:
                        if str(event.get("itemId")) in inventories[p_id]: inventories[p_id].remove(str(event.get("itemId")))
                    elif event.get("type") == "ITEM_UNDO":
                        if inventories[p_id] and str(event.get("beforeId")) == inventories[p_id][-1]: inventories[p_id].pop()
        return snapshots

    def run(self):
        files = [f for f in os.listdir(MATCH_DIR) if f.endswith(".json")]
        print(f"[*] Processing {len(files)} matches for Brain V2.1...")
        all_snapshots = []
        for f in files:
            all_snapshots.extend(self.process_match(f.replace(".json", "")))
        with open(config.V2_EPISODES_PATH, "w") as f:
            json.dump(all_snapshots, f, indent=4)
        print(f"[SUCCESS] V2.1 Preprocessing Complete! Created {len(all_snapshots)} rich snapshots.")

if __name__ == "__main__":
    V2Preprocessor().run()
