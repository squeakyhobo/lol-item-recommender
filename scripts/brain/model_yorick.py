import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Standard Transformer Positional Encoding.
    Since we look at a sequence of 5 frames, we need to tell the AI 
    which frame came first and which came last.
    """
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
    The Maiden's Brain (V3): Multi-Task Learning (MTL) Transformer.
    This model simultaneously predicts the broad strategic cluster AND the specific item ID.
    The easier cluster task helps stabilize and guide the harder specific item task.
    """
    def __init__(self, num_clusters=15, num_items=528, dna_dim=9, d_model=256, nhead=4, num_layers=4):
        super(YorickBrain, self).__init__()
        self.d_model = d_model
        
        # Rune Embedding: Maps 15 possible keystones into the high-dimensional brain space.
        self.rune_embedding = nn.Embedding(num_embeddings=15, embedding_dim=d_model)

        # 1. Projections: Convert raw game data into a format the Transformer understands.
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
        
        # 2. Enemy Projection: Processes the combined 'Total Threat DNA' of all 5 enemies.
        self.enemy_projection = nn.Sequential(
            nn.Linear(dna_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder: The 'Shared Backbone' of the Multi-Task network.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Multi-Task Heads
        # Head 1: The Strategist (Predicts Cluster)
        self.cluster_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_clusters)
        )
        
        # Head 2: The Tactician (Predicts Exact Item)
        self.item_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_items)
        )

    def forward(self, p_num, p_dna, e_dna, e_arch, rune_idx, item_mask=None):
        # A. Embed Rune
        rune_emb = self.rune_embedding(rune_idx)
        if len(rune_emb.shape) == 3:
            rune_emb = rune_emb.squeeze(1)
        
        # B. Combine player features
        p_emb = torch.cat([self.numeric_projection(p_num), 
                           self.dna_projection(p_dna)], dim=2)
        
        # C. Process enemy features
        e_emb = self.enemy_projection(e_dna) 
        
        # D. Sequence Assembly
        sequence = torch.cat([p_emb, e_emb], dim=1) 
        sequence = self.pos_encoder(sequence)
        
        # E. Attention processing
        transformed = self.transformer(sequence)
        
        # F. Strategic Merge
        final_state = transformed[:, 4, :] 
        combined_state = torch.cat([final_state, rune_emb], dim=1)
        
        # G. Multi-Task Predictions
        cluster_logits = self.cluster_head(combined_state)
        item_logits = self.item_head(combined_state)
        
        if item_mask is not None:
            item_logits = item_logits.masked_fill(item_mask.bool(), -1e9)
            
        return cluster_logits, item_logits
