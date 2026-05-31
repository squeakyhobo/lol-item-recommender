import os
import json

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")
MATCH_DIR = os.path.join(DATA_DIR, "yorick_games", "matches")
TIMELINE_DIR = os.path.join(DATA_DIR, "yorick_games", "timelines")

class YorickPreprocessor:
    """
    The Data Chef.
    This class takes raw, unstructured Riot API JSON files and converts them into
    clean, mathematical 'Snapshots' (Tensors) that the Neural Network can learn from.
    """
    def __init__(self, sequence_length=5):
        self.seq_len = sequence_length
        with open(os.path.join(DATA_DIR, "archetypes.json"), "r") as f:
            self.archetypes = json.load(f)
        with open(os.path.join(DATA_DIR, "valid_targets.json"), "r") as f:
            self.valid_targets = set(json.load(f))

    def get_winning_team(self, timeline):
        frames = timeline.get("info", {}).get("frames", [])
        if not frames: return None
        for e in frames[-1].get("events", []):
            if e.get("type") == "GAME_END": return e.get("winningTeam")
        return None

    def get_champion_map(self, timeline):
        champ_map = {}
        frames = timeline.get("info", {}).get("frames", [])
        for frame in frames:
            for event in frame.get("events", []):
                if event.get("type") == "CHAMPION_KILL":
                    for source in ["victimDamageDealt", "victimDamageReceived"]:
                        for damage in event.get(source, []):
                            p_id = damage.get("participantId")
                            name = damage.get("name")
                            if p_id and name and 1 <= p_id <= 10 and "Minion" not in name:
                                champ_map[p_id] = name
        return champ_map

    def process_match(self, match_id):
        m_path = os.path.join(MATCH_DIR, f"{match_id}.json")
        t_path = os.path.join(TIMELINE_DIR, f"{match_id}.json")
        try:
            with open(m_path, "r") as f: match_data = json.load(f)
            with open(t_path, "r") as f: timeline_data = json.load(f)
        except: return []

        winning_team = self.get_winning_team(timeline_data)
        if not winning_team: return []

        yorick_id = None
        yorick_team = None
        keystone_id = 0
        
        participants = match_data.get("info", {}).get("participants", [])
        for p in participants:
            if p.get("championName") == "Yorick" and p.get("teamPosition") == "TOP":
                yorick_id = p.get("participantId")
                yorick_team = p.get("teamId")
                for s in p.get("perks", {}).get("styles", []):
                    if s.get("description") == "primaryStyle":
                        keystone_id = s.get("selections", [{}])[0].get("perk", 0)
                break
                
        if not yorick_id or yorick_team != winning_team:
            return []

        champ_map = self.get_champion_map(timeline_data)
        enemy_team = 200 if yorick_team == 100 else 100
        enemy_roles = {p.get("participantId"): p.get("teamPosition") for p in participants if p.get("teamId") == enemy_team}

        snapshots = []
        frames = timeline_data.get("info", {}).get("frames", [])
        inventories = {i: [] for i in range(1, 11)}
        frame_history = []
        enemy_kills = 0

        for frame_idx, frame in enumerate(frames):
            for event in frame.get("events", []):
                p_id = event.get("participantId")
                if p_id and 1 <= p_id <= 10:
                    if event.get("type") == "ITEM_PURCHASED":
                        inventories[p_id].append(event.get("itemId"))
                    elif event.get("type") in ["ITEM_SOLD", "ITEM_DESTROYED"]:
                        if event.get("itemId") in inventories[p_id]:
                            inventories[p_id].remove(event.get("itemId"))
                    elif event.get("type") == "ITEM_UNDO":
                        if inventories[p_id]: inventories[p_id].pop()

                if event.get("type") == "CHAMPION_KILL":
                    victim_team = 100 if event.get("victimId", 0) <= 5 else 200
                    if victim_team == yorick_team: enemy_kills += 1

            p_frame = frame.get("participantFrames", {}).get(str(yorick_id), {})
            
            # Matchup-Aware Enemy State
            enemy_states = []
            for i in range(1, 11):
                team = 100 if i <= 5 else 200
                if team == enemy_team:
                    ef = frame.get("participantFrames", {}).get(str(i), {})
                    role = enemy_roles.get(i, "Unknown")
                    enemy_states.append({
                        "championName": champ_map.get(i, "Unknown"),
                        "gold": ef.get("totalGold", 0),
                        "level": ef.get("level", 1),
                        "inventory": list(inventories[i]),
                        "archetype": self.archetypes.get(champ_map.get(i, ""), 0),
                        "is_lane_opponent": (role == "TOP")
                    })

            state = {
                "championName": "Yorick",
                "keystone": keystone_id,
                "gold": p_frame.get("currentGold", 0),
                "total_gold": p_frame.get("totalGold", 0),
                "level": p_frame.get("level", 1),
                "minute": frame_idx,
                "inventory": list(inventories[yorick_id]),
                "enemy_context": enemy_states,
                "kill_pressure": enemy_kills / max(frame_idx, 1),
                "gold_diff": p_frame.get("totalGold", 0) - sum([e["gold"] for e in enemy_states])/5
            }
            
            frame_history.append(state)
            if len(frame_history) > self.seq_len: frame_history.pop(0)
            
            for event in frame.get("events", []):
                if event.get("type") == "ITEM_PURCHASED" and event.get("participantId") == yorick_id:
                    item_id = event.get("itemId")
                    if item_id in self.valid_targets and len(frame_history) == self.seq_len:
                        snapshots.append({
                            "sequence": list(frame_history),
                            "target_item": item_id
                        })
                    
        return snapshots

    def run(self):
        all_snapshots = []
        files = [f for f in os.listdir(MATCH_DIR) if f.endswith(".json")]
        print(f"[*] YORICK PREPROCESSOR: Processing {len(files)} matches for V4 Matchup-Awareness...")
        
        for f in files:
            all_snapshots.extend(self.process_match(f.replace(".json", "")))
        
        output_path = os.path.join(DATA_DIR, "yorick_episodes.json")
        with open(output_path, "w") as out:
            json.dump(all_snapshots, out, indent=4)
        print(f"[*] PREPROCESSING COMPLETE: Created {len(all_snapshots)} snapshots.")

if __name__ == "__main__":
    YorickPreprocessor().run()
