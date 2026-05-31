import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class YorickBrain(nn.Module):
    """
    The Maiden's Brain (V4): Matchup-Aware Multi-Task Learning.
    This model explicitly weights the direct lane opponent's DNA heavier
    than the other 4 enemies using a dedicated Matchup Head.
    """
    def __init__(self, num_clusters=15, num_items=80, dna_dim=9, d_model=256, nhead=4, num_layers=4):
        super(YorickBrain, self).__init__()
        self.d_model = d_model
        self.rune_embedding = nn.Embedding(num_embeddings=15, embedding_dim=d_model)

        # 1. Player & Enemy Projections
        self.numeric_projection = nn.Sequential(
            nn.Linear(5, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU()
        )
        self.dna_projection = nn.Sequential(
            nn.Linear(dna_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU()
        )
        self.enemy_projection = nn.Sequential(
            nn.Linear(dna_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # 2. NEW: Lane Opponent Projection (High Weight)
        self.lane_projection = nn.Sequential(
            nn.Linear(dna_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Heads (Merged dimensions: Backbone + Runes + Lane Opponent)
        # We concat: [Backbone Output (d_model)] + [Rune Emb (d_model)] + [Lane Emb (d_model)]
        self.cluster_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_clusters)
        )
        
        self.item_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_items)
        )

    def forward(self, p_num, p_dna, e_dna, lane_dna, rune_idx, item_mask=None):
        # A. Embed Context
        rune_emb = self.rune_embedding(rune_idx)
        if len(rune_emb.shape) == 3: rune_emb = rune_emb.squeeze(1)
        
        lane_emb = self.lane_projection(lane_dna) # [batch, d_model]
        
        # B. Backbone Processing
        p_emb = torch.cat([self.numeric_projection(p_num), self.dna_projection(p_dna)], dim=2)
        e_emb = self.enemy_projection(e_dna) 
        
        sequence = torch.cat([p_emb, e_emb], dim=1) 
        sequence = self.pos_encoder(sequence)
        transformed = self.transformer(sequence)
        
        # C. Feature Fusion
        final_backbone_state = transformed[:, 4, :] # Last player frame
        
        # Merge Backbone + Strategic Bias (Runes) + Tactical Bias (Matchup)
        combined_state = torch.cat([final_backbone_state, rune_emb, lane_emb], dim=1)
        
        # D. Output
        cluster_logits = self.cluster_head(combined_state)
        item_logits = self.item_head(combined_state)
        
        if item_mask is not None:
            item_logits = item_logits.masked_fill(item_mask.bool(), -1e9)
            
        return cluster_logits, item_logits
