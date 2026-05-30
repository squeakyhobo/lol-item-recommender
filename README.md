


> **Made with Gemini**

A real-time AI assistant for League of Legends, specially optimized for **Yorick**. 

This AI analyzes your game in real-time and recommends items based on winning patterns from the top 50 Yorick players in the world.

### 🧠 Key Features
- **Yorick Only (for now):** Trained on 2,500+ high-ELO Yorick matches.
- **Smart Pruning:** The AI only suggests items that pro Yorick players actually buy.
- **Rune Aware:** Recognizes your Keystone (Conqueror, Comet, etc.) and changes your build path to match.
- **Wave Coach:** Includes an AI coach that gives you tips on how to manage your minion waves.

### 🛠️ Setup
1. Install requirements: `pip install -r requirements.txt`
2. Add your API keys to `.env`.
3. Build the data: `python scripts/brain/preprocess_yorick.py`
4. Train the AI: `python scripts/brain/train_yorick.py`

### 🎮 How to Use
Run the overlay while playing:
`python scripts/hud/inference_hud.py`
