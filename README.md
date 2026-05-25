# LTA-Transformer: AI Itemization Advisor

A real-time, Transformer-based tactical advisor for League of Legends (Top Lane focus).

## Overview
LTA-Transformer uses a PyTorch Transformer architecture to analyze the live game state—including enemy archetypes, KDA, and current gold—to recommend the optimal next item purchase.

## Key Features
- **Transformer Brain:** Uses Attention mechanisms to weigh enemy threats.
- **Expert System:** Enforces game-legal rules (e.g., no double boots, unique passive groups).
- **Winners-Only Training:** Trained on Challenger-level match timelines from winning teams.
- **Transparent HUD:** A non-intrusive PyQt5 overlay for real-time guidance.

## Architecture
1. **The Eye (Scraper):** Polls the Riot Live Client Data API.
2. **The Pre-Processor:** Clusters 160+ champions into 8 archetypes via K-Means.
3. **The Brain:** 4-layer Transformer Encoder with Inventory Positional Embeddings.
4. **Logit Masking:** Hard-masks items you already own or are ineligible to buy.

---
Developed by Lucas Bubu-Oppenheimer.
