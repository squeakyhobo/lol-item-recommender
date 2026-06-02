
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
        
        
        self.numeric_projection = nn.Sequential(
            nn.Linear(6, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU()
        )
        self.dna_projection = nn.Sequential(
            nn.Linear(dna_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU()
        )
        self.enemy_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer (Smaller for 1000 samples to prevent overfitting)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=1024, 
            dropout=0.2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier Head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, item_vocab_size)
        )

    def forward(self, player_seq_numeric, player_seq_dna, enemy_numeric, enemy_archetypes, mask=None):
        # player_seq_numeric: [batch, 5, 5]
        # player_seq_dna: [batch, 5, 15]
        
        # Player embedding
        p_emb = torch.cat([self.numeric_projection(player_seq_numeric), 
                           self.dna_projection(player_seq_dna)], dim=2)
        
        # Enemy embedding
        e_emb = self.enemy_projection(enemy_numeric) # [batch, 5, d_model]
        
        sequence = torch.cat([p_emb, e_emb], dim=1) # [batch, 10, d_model]
        sequence = self.pos_encoder(sequence)
        
        transformed = self.transformer(sequence)
        
        # Use the final player state for prediction
        logits = self.output_head(transformed[:, 4, :])
        
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), -1e9)
            
        return logits

