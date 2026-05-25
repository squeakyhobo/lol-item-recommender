> This README was made with Gemini

# LoL Item Recommender

A real-time, AI-driven itemization engine for League of Legends, specifically optimized for the Top Lane. It uses a custom PyTorch Transformer architecture to analyze live game states and recommend counter-builds based on Challenger-level winning patterns.

## ?? The Architecture
1. **The Eye (Data Extraction):** Real-time extraction of player stats and gold via the Riot Live Client API.
2. **The Pre-Processor (Clustering):** Maps 160+ champions into 8 core **Archetypes** (e.g., Juggernauts, Skirmishers, Wardens).
3. **The Brain (Transformer):** A 4-layer Transformer Encoder that uses **Attention** to weigh enemy threats.
4. **The HUD (Overlay):** A transparent PyQt5 window that provides Top 3 recommendations without interrupting gameplay.

## ?? Key Innovations
- **Inventory Positional Embeddings:** The model understands the chronological order of your build path.
- **Logit Masking:** An expert system that enforces game rules (e.g., no double boots, unique passive group restrictions).
- **Winners-Only Training:** The model is trained exclusively on the sequences of players who successfully closed out their games.

## ??? Installation & Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Set your Riot API Key in `.env`.
3. Launch the advisor: `python scripts/hud/inference_hud.py`

---
*Created as part of the Blue Giant Development Phase.*
