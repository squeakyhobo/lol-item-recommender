
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
TIMELINE_DIR = os.path.join(DATA_DIR, "timelines")

class TimelineProcessor:
    def __init__(self):
        with open(os.path.join(DATA_DIR, "archetypes.json"), "r") as f:
            self.archetypes = json.load(f)
        with open(os.path.join(DATA_DIR, "valid_targets.json"), "r") as f:
            self.valid_targets = set(json.load(f))

    def get_winning_team(self, timeline):
        # The last event usually contains GAME_END with winningTeam
        frames = timeline.get("info", {}).get("frames", [])
        if not frames: return None
        last_events = frames[-1].get("events", [])
        for e in last_events:
            if e.get("type") == "GAME_END":
                return e.get("winningTeam")
        return None

    def get_top_lane_participant(self, timeline):
        frames = timeline.get("info", {}).get("frames", [])
        if len(frames) < 2: return []
        p_frames = frames[1].get("participantFrames", {})
        top_candidates = []
        for p_id, data in p_frames.items():
            pos = data.get("position", {})
            x, y = pos.get("x", 0), pos.get("y", 0)
            if (x < 4000 and y > 10000) or (x > 10000 and y < 4000):
                # Identify team based on participantId (1-5 is Blue/100, 6-10 is Red/200)
                team_id = 100 if int(p_id) <= 5 else 200
                top_candidates.append({"id": int(p_id), "team": team_id})
        return top_candidates

    def process_match(self, match_id):
        path = os.path.join(TIMELINE_DIR, f"{match_id}.json")
        with open(path, "r") as f:
            data = json.load(f)
            
        winning_team = self.get_winning_team(data)
        if not winning_team: return []
        
        top_candidates = self.get_top_lane_participant(data)
        if not top_candidates: return []

        snapshots = []
        frames = data.get("info", {}).get("frames", [])
        
        for cand in top_candidates:
            # V2.0 FEATURE: ONLY TRAIN ON WINNERS
            if cand["team"] != winning_team:
                continue
                
            target_id = cand["id"]
            current_items = []
            
            for frame_idx, frame in enumerate(frames):
                for event in frame.get("events", []):
                    if event.get("type") == "ITEM_PURCHASED" and event.get("participantId") == target_id:
                        item_id = event.get("itemId")
                        if item_id not in self.valid_targets:
                            current_items.append(item_id)
                            continue

                        p_frame = frame.get("participantFrames", {}).get(str(target_id), {})
                        all_p_states = []
                        for i in range(1, 11):
                            other_p = frame.get("participantFrames", {}).get(str(i), {})
                            all_p_states.append({
                                "id": i,
                                "gold": other_p.get("totalGold", 0),
                                "level": other_p.get("level", 0),
                                "is_enemy": (i > 5) if target_id <= 5 else (i <= 5)
                            })
                        
                        snapshot = {
                            "minute": frame_idx,
                            "top_gold": p_frame.get("currentGold", 0),
                            "top_current_items": list(current_items),
                            "world_state": all_p_states,
                            "target_item": item_id
                        }
                        snapshots.append(snapshot)
                        current_items.append(item_id)
        return snapshots

    def run(self):
        all_snapshots = []
        files = [f for f in os.listdir(TIMELINE_DIR) if f.endswith(".json")]
        print(f"[*] Processing {len(files)} matches (WINNERS ONLY)...")
        for f in files:
            all_snapshots.extend(self.process_match(f.replace(".json", "")))
        output_path = os.path.join(DATA_DIR, "top_lane_snapshots.json")
        with open(output_path, "w") as out:
            json.dump(all_snapshots, out, indent=4)
        print(f"[*] Created {len(all_snapshots)} high-quality snapshots in {output_path}")

if __name__ == "__main__":
    TimelineProcessor().run()

