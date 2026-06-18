import torch
import torch.nn as nn
import torch.nn.functional as F

class YorickMLP(nn.Module):
    """
    Maiden Brain V2.3 (Ultra-Brain): Optimized for Multi-Champion Mastery.
    Supports Yorick, Sett, Mundo, Trundle, Morde, Garen.
    Scaled to 2048 neurons with heavy regularization.
    """
    def __init__(self, num_champs=200, num_runes=20, num_items=30, numerical_dim=113):
        super(YorickMLP, self).__init__()
        
        # 1. Categorical Embeddings
        self.champ_emb = nn.Embedding(num_champs, 64) # Increased embedding size
        self.rune_emb = nn.Embedding(num_runes, 32)
        self.inv_emb = nn.Embedding(10000, 16) 
        
        # 2. Dense Layers for Numerical Features
        # Dimensions: Champ(64) + Opp(64) + Rune(32) + Inventory(6*16=96) + Stats(dim)
        input_size = 64 + 64 + 32 + 96 + numerical_dim
        self.input_layer = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        
        self.hidden_layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # 3. Output Heads
        self.strategy_head = nn.Linear(256, 8) 
        self.item_head = nn.Linear(256, num_items)

    def forward(self, my_id, opp_id, rune_id, inv_ids, numerical_feats, item_mask=None):
        # 1. Embed Categoricals
        me_e = self.champ_emb(my_id)
        opp_e = self.champ_emb(opp_id)
        rune_e = self.rune_emb(rune_id)
        inv_e = self.inv_emb(inv_ids).view(inv_ids.size(0), -1)
        
        # 2. Combine all features
        x = torch.cat([me_e, opp_e, rune_e, inv_e, numerical_feats], dim=1)
        
        # 3. Process through Scaled MLP
        x = F.relu(self.bn1(self.input_layer(x)))
        x = self.hidden_layers(x)
        
        # 4. Predict
        s_logits = self.strategy_head(x)
        i_logits = self.item_head(x)
        
        if item_mask is not None:
            i_logits = i_logits.masked_fill(item_mask.bool(), -1e9)
            
        return s_logits, i_logits
