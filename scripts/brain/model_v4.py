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

class GoliathV4(nn.Module):
    def __init__(self, item_vocab_size, dna_dim=15, d_model=256, nhead=4, num_layers=4):
        super(GoliathV4, self).__init__()
        
        self.d_model = d_model
        
        # 1. Projections with AUTOMATIC NORMALIZATION (LayerNorm)
        # We now accept 6 numeric features (including Gold Diff)
        self.numeric_projection = nn.Sequential(
            nn.Linear(6, d_model // 2),
            nn.LayerNorm(d_model // 2), # THE SELF-GRADING ENGINE
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.dna_projection = nn.Sequential(
            nn.Linear(dna_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Enemy projection now handles DNA too! (Architectural Upgrade)
        self.enemy_projection = nn.Sequential(
            nn.Linear(dna_dim + 5, d_model), # 5 numerics + DNA
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier Head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, item_vocab_size)
        )

    def forward(self, p_num, p_dna, e_num, e_dna, mask=None):
        # Fusion logic
        p_emb = torch.cat([self.numeric_projection(p_num), 
                           self.dna_projection(p_dna)], dim=2)
        
        # Enemy Token Fusion (Numeric + DNA)
        e_emb = self.enemy_projection(torch.cat([e_num, e_dna], dim=2))
        
        sequence = torch.cat([p_emb, e_emb], dim=1) 
        sequence = self.pos_encoder(sequence)
        
        transformed = self.transformer(sequence)
        
        # Pull prediction from the current Player state (Index 4)
        logits = self.output_head(transformed[:, 4, :])
        
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), -1e9)
            
        return logits
