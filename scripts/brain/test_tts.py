import sys
import os
import time
import pyttsx3

# Setup paths
BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
sys.path.append(os.path.join(BASE_DIR, "scripts", "brain"))

# We won't use the full coach logic here to avoid Ollama overhead, 
# just testing the actual TTS engine and HUD voice quality.

def test_tts():
    print("🧟 LTA GOLIATH: TTS COACH TEST")
    print("-" * 40)
    
    # 1. Initialize the engine exactly like the HUD does
    engine = pyttsx3.init()
    engine.setProperty('rate', 180) # Challenger speed
    
    # 2. Get available voices (optional, just for info)
    voices = engine.getProperty('voices')
    print(f"[*] Using System Voice: {voices[0].name}")

    # 3. Simulate a sequence of tactical tips
    tips = [
        "Welcome back, Yorick. The Maiden is ready.",
        "Watch river. Enemy jungler likely Top side.",
        "Matchup Ambessa is stacking HP. Build shred items.",
        "Grubs are dead. Priority shifted to Mid lane pressure."
    ]

    for tip in tips:
        print(f"\n[COACH SAYS]: {tip}")
        engine.say(tip)
        engine.runAndWait()
        time.sleep(1)

    print("\n[+] TTS Test Complete. If you heard the voice, V7.1 is ready for battle!")

if __name__ == "__main__":
    test_tts()
