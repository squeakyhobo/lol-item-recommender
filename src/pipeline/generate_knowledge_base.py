import requests
import json
import os
import sys

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
import config

DATA_DIR = config.DATA_DIR

def generate_knowledge_base():
    print("[*] STEP 1: Fetching Official Riot Data (DDragon)...")
    try:
        # Get latest version
        v_url = "https://ddragon.leagueoflegends.com/api/versions.json"
        version = requests.get(v_url).json()[0]
        print(f"  -> Using LoL Version: {version}")

        # Get Champion Data
        c_url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
        riot_data = requests.get(c_url).json()["data"]
    except Exception as e:
        print(f"[!] Error fetching Riot data: {e}")
        return

    print("[*] STEP 2: Loading Local Archetypes...")
    try:
        with open(os.path.join(DATA_DIR, "archetypes.json"), "r") as f:
            local_archetypes = json.load(f)
    except:
        print("[!] No archetypes.json found. Using default 0.")
        local_archetypes = {}

    # --- EXPERT LOGIC MAPPING (LLM GENERATED) ---
    # Comprehensive lists based on High-ELO mechanic interactions for all 168+ champions.
    
    HEALERS = [
        "Aatrox", "Briar", "DrMundo", "Fiddlesticks", "Illaoi", "Ivern", "Kayn", 
        "Nilah", "Olaf", "Samira", "Soraka", "Swain", "Sylas", "Taric", 
        "Vladimir", "Volibear", "Warwick", "Yuumi", "Zac"
    ]
    
    SHIELDERS = [
        "Ivern", "Janna", "Karma", "Lulu", "Milio", "Nautilus", "Orianna", 
        "Rakan", "Renata", "Seraphine", "Sett", "Shen", "Skarner", "Sona", 
        "TahmKench", "Taric", "Thresh", "Udyr"
    ]
    
    AA_HEAVY = [
        "Akshan", "Aphelios", "Ashe", "Belveth", "Caitlyn", "Camille", "Draven", 
        "Fiora", "Gwen", "Irelia", "Jax", "Jinx", "Kaisa", "Kalista", "Kayle", 
        "Kindred", "KogMaw", "Lucian", "MasterYi", "MissFortune", "Nilah", 
        "Nocturne", "Quinn", "Samira", "Sivir", "Tristana", "Trundle", 
        "Tryndamere", "Twitch", "Urgot", "Vayne", "Volibear", "Xayah", 
        "Yasuo", "Yone", "Zeri"
    ]
    
    BURST_AP = [
        "Ahri", "Akali", "Annie", "AurelionSol", "Azir", "Cassiopeia", "Diana", 
        "Ekko", "Elise", "Evelynn", "Fizz", "Galio", "Gragas", "Gwen", 
        "Heimerdinger", "Hwei", "Kassadin", "Katarina", "Kennen", "Leblanc", 
        "Lissandra", "Lux", "Malzahar", "Neeko", "Nidalee", "Orianna", "Qiyana", 
        "Rumble", "Ryze", "Sylas", "Syndra", "Taliyah", "TwistedFate", "Veigar", 
        "Vex", "Viktor", "Vladimir", "Xerath", "Zoe"
    ]

    print("[*] STEP 3: Merging & Generalizing...")
    knowledge_base = {}
    
    for c_id, c_info in riot_data.items():
        name = c_info["name"].replace(" ", "").replace("'", "")
        # Clean name to match local archetypes (usually lowercase/no spaces)
        clean_name = name.lower()
        
        # 1. Riot Tags (Roles)
        tags = c_info.get("tags", [])
        
        # 2. Expert Flags (Calculated)
        kb_entry = {
            "id": c_info["key"],
            "riot_tags": tags,
            "archetype": local_archetypes.get(clean_name, 2), # Default to Bruiser (2)
            "is_healer": 1 if any(h.lower() in clean_name for h in HEALERS) else 0,
            "has_shields": 1 if any(s.lower() in clean_name for s in SHIELDERS) else 0,
            "is_aa_heavy": 1 if any(a.lower() in clean_name for a in AA_HEAVY) or "Marksman" in tags else 0,
            "is_burst_threat": 1 if any(b.lower() in clean_name for b in BURST_AP) or "Assassin" in tags else 0,
            "is_ranged": 1 if c_info.get("stats", {}).get("attackrange", 0) > 350 else 0,
            "is_tanky": 1 if "Tank" in tags or c_info.get("stats", {}).get("hp", 0) > 600 else 0
        }
        knowledge_base[clean_name] = kb_entry

    # Save to JSON
    output_path = os.path.join(DATA_DIR, "champion_knowledge.json")
    with open(output_path, "w") as f:
        json.dump(knowledge_base, f, indent=4)

    print(f"\n[SUCCESS] Universal Knowledge Base built for {len(knowledge_base)} champions.")
    print(f"[*] Path: {output_path}")
    print("\nExample (Dr. Mundo):")
    print(json.dumps(knowledge_base.get("drmundo"), indent=2))

if __name__ == "__main__":
    generate_knowledge_base()
