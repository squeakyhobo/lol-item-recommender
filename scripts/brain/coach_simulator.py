import sys
import os
import json
import time

# Setup paths
BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
sys.path.append(os.path.join(BASE_DIR, "scripts", "brain"))
from wave_agent import create_wave_coach

def run_simulation():
    print("🧟 LTA GOLIATH: STRATEGIC WATCHMAN SIMULATOR (V7.1)")
    print("-" * 50)
    
    coach = create_wave_coach()
    if not coach:
        print("Error: Could not boot local coach. Is Ollama running?")
        return

    # Scenario: The "Deep Push" Danger Zone
    # You are pushed deep, it is early game, and the Jungler is MISSING.
    test_scenario = {
        "matchup": "Yorick vs Ambessa",
        "time": 4.2, 
        "hp": 90,
        "zone": "TOP LANE",
        "danger": "PUSHED DEEP / OVEREXTENDED",
        "visible": ["Mid", "Bot", "Support"],
        "missing": ["Jungle", "Top"],
        "dead_objectives": []
    }

    prompt = f"""
    ROLE: Challenger Strategic Watchman.
    STATE: {test_scenario['matchup']} | Time: {test_scenario['time']}m | HP: {test_scenario['hp']}%
    MAP: You are in {test_scenario['zone']} ({test_scenario['danger']}).
    VISION: Visible: {test_scenario['visible']} | ⚠️ MISSING: {test_scenario['missing']}
    OBJECTIVES: Already Dead: {test_scenario['dead_objectives']}
    
    YOUR JOB: Act like a pro coach. Warn the player LOUDLY if they are in danger.
    Consult manuals. Give 1 short tactical reminder (10 words max).
    """

    print(f"[*] Simulating: {test_scenario['danger']} with JUNGLE MISSING...")
    print("[*] Coach is scanning the map...")
    
    response = coach.run(prompt)
    
    print("\n" + "="*50)
    print(f"HUD VOCAL OUTPUT: {response}")
    print("="*50)

if __name__ == "__main__":
    run_simulation()
