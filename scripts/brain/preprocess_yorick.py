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
        """
        We only want the AI to learn from WINNING strategies.
        If Yorick loses, we throw the game away.
        """
        frames = timeline.get("info", {}).get("frames", [])
        if not frames: return None
        for e in frames[-1].get("events", []):
            if e.get("type") == "GAME_END": return e.get("winningTeam")
        return None

    def get_champion_map(self, timeline):
        """
        The Riot API timeline hides champion names. We use kill events 
        to reverse-engineer which player ID corresponds to which champion.
        """
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
        """
        The core engine. It plays through the timeline minute-by-minute.
        Every time Yorick buys an important item, it takes a "Snapshot" of the last 5 minutes.
        """
        m_path = os.path.join(MATCH_DIR, f"{match_id}.json")
        t_path = os.path.join(TIMELINE_DIR, f"{match_id}.json")
        try:
            with open(m_path, "r") as f: match_data = json.load(f)
            with open(t_path, "r") as f: timeline_data = json.load(f)
        except: return []

        winning_team = self.get_winning_team(timeline_data)
        if not winning_team: return []

        # Find Yorick and extract his starting Rune (Keystone)
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
                
        # If Yorick didn't win or wasn't Top, skip this match
        if not yorick_id or yorick_team != winning_team:
            return []

        champ_map = self.get_champion_map(timeline_data)
        enemy_team = 200 if yorick_team == 100 else 100

        snapshots = []
        frames = timeline_data.get("info", {}).get("frames", [])
        
        # Track inventories for ALL 10 players to calculate 'Total Enemy DNA'
        inventories = {i: [] for i in range(1, 11)}
        
        frame_history = []
        enemy_kills = 0

        for frame_idx, frame in enumerate(frames):
            # 1. Update Inventories based on events in this frame (buys, sells, undos)
            for event in frame.get("events", []):
                p_id = event.get("participantId")
                if p_id and 1 <= p_id <= 10:
                    if event.get("type") == "ITEM_PURCHASED":
                        inventories[p_id].append(event.get("itemId"))
                    elif event.get("type") == "ITEM_SOLD":
                        if event.get("itemId") in inventories[p_id]:
                            inventories[p_id].remove(event.get("itemId"))
                    elif event.get("type") == "ITEM_DESTROYED":
                        if event.get("itemId") in inventories[p_id]:
                            inventories[p_id].remove(event.get("itemId"))
                    elif event.get("type") == "ITEM_UNDO":
                        if inventories[p_id]: inventories[p_id].pop()

                # Track global kill pressure (are we winning or losing?)
                if event.get("type") == "CHAMPION_KILL":
                    victim_team = 100 if event.get("victimId", 0) <= 5 else 200
                    if victim_team == yorick_team: enemy_kills += 1

            p_frame = frame.get("participantFrames", {}).get(str(yorick_id), {})
            
            # 2. Build the Enemy State Array
            enemy_states = []
            for i in range(1, 11):
                team = 100 if i <= 5 else 200
                if team == enemy_team:
                    ef = frame.get("participantFrames", {}).get(str(i), {})
                    enemy_states.append({
                        "championName": champ_map.get(i, "Unknown"),
                        "gold": ef.get("totalGold", 0),
                        "level": ef.get("level", 1),
                        "inventory": list(inventories[i]), # SAVE ENEMY ITEMS FOR DNA!
                        "archetype": self.archetypes.get(champ_map.get(i, ""), 0)
                    })

            # 3. Compile the current frame
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
            
            # 4. Check if Yorick bought a valid target item in this frame. 
            # If yes, save the last 5 minutes of history as a "Snapshot Lesson" for the AI.
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
        print(f"[*] YORICK PREPROCESSOR: Extracting Runes and Timelines from {len(files)} matches...")
        
        for f in files:
            all_snapshots.extend(self.process_match(f.replace(".json", "")))
        
        output_path = os.path.join(DATA_DIR, "yorick_episodes.json")
        with open(output_path, "w") as out:
            json.dump(all_snapshots, out, indent=4)
        print(f"[*] YORICK PREPROCESSING COMPLETE: Created {len(all_snapshots)} Rune-Aware snapshots.")

if __name__ == "__main__":
    YorickPreprocessor().run()
