import torch
import torch.nn as nn
import torch.nn.functional as F

class YorickMLP(nn.Module):
    """
    Maiden Brain V2.1: The Ultimate Expert MLP.
    Features: Direct Inventory Vision + Expert Tags.
    """
    def __init__(self, num_champs=200, num_runes=10, num_items=30, numerical_dim=61):
        super(YorickMLP, self).__init__()
        
        # 1. Categorical Embeddings
        self.champ_emb = nn.Embedding(num_champs, 32)
        self.rune_emb = nn.Embedding(num_runes, 16)
        self.inv_emb = nn.Embedding(10000, 16) # Embed all possible Item IDs
        
        # 2. Dense Layers for Numerical Features
        # Dimensions: Champ(32) + Opp(32) + Rune(16) + Inventory(6*16=96) + Stats(61) = 237
        self.input_layer = nn.Linear(32 + 32 + 16 + 96 + numerical_dim, 512)
        
        self.hidden_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
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
        
        # 2. Embed Inventory (6 slots -> flattened)
        # [batch, 6] -> [batch, 6, 16] -> [batch, 96]
        inv_e = self.inv_emb(inv_ids).view(inv_ids.size(0), -1)
        
        # 3. Combine all features
        x = torch.cat([me_e, opp_e, rune_e, inv_e, numerical_feats], dim=1)
        
        # 4. Process through MLP
        x = F.relu(self.input_layer(x))
        x = self.hidden_layers(x)
        
        # 4. Predict
        s_logits = self.strategy_head(x)
        i_logits = self.item_head(x)
        
        # Apply mask in inference if needed
        if item_mask is not None:
            i_logits = i_logits.masked_fill(item_mask.bool(), -1e9)
            
        return s_logits, i_logits
