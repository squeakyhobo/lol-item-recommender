import os
import json

BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
DATA_DIR = os.path.join(BASE_DIR, "data")

def build_vocab():
    match_dir = os.path.join(DATA_DIR, "yorick_games", "timelines")
    matches_dir = os.path.join(DATA_DIR, "yorick_games", "matches")
    
    with open(os.path.join(DATA_DIR, "valid_targets.json"), "r") as f:
        valid_targets = set(json.load(f))
        
    with open(os.path.join(DATA_DIR, "item_names.json"), "r") as f:
        names = json.load(f)

    unique_items = set()
    
    # 1. Scan all games to find what Yorick actually buys
    for f in os.listdir(match_dir):
        try:
            with open(os.path.join(matches_dir, f)) as mf:
                match_data = json.load(mf)
                yorick_id = None
                for p in match_data.get('info', {}).get('participants', []):
                    if p.get('championName') == 'Yorick':
                        yorick_id = p.get('participantId')
                        break
            if not yorick_id: continue

            with open(os.path.join(match_dir, f)) as jf:
                data = json.load(jf)
                for frame in data.get('info', {}).get('frames', []):
                    for event in frame.get('events', []):
                        if event.get('type') == 'ITEM_PURCHASED' and event.get('participantId') == yorick_id:
                            iid = int(event.get('itemId'))
                            if iid in valid_targets:
                                unique_items.add(iid)
        except: pass

    # 2. Sort the items to ensure consistent indexing
    vocab_list = sorted(list(unique_items))
    
    # 3. Create bidirectional mapping (Index -> ItemID, ItemID -> Index)
    vocab_map = {str(iid): idx for idx, iid in enumerate(vocab_list)}
    inv_vocab_map = {str(idx): iid for idx, iid in enumerate(vocab_list)}
    
    with open(os.path.join(DATA_DIR, "yorick_vocab.json"), "w") as f:
        json.dump({
            "size": len(vocab_list),
            "item_to_index": vocab_map,
            "index_to_item": inv_vocab_map
        }, f, indent=4)
        
    print(f"[*] Pruned Action Space! Created Yorick Vocab with {len(vocab_list)} items.")
    print("Sample of allowed items:", [names.get(str(i)) for i in vocab_list[:5]])

if __name__ == "__main__":
    build_vocab()
