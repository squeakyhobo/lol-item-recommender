
import torch
import torch.nn as nn
import torch.nn.functional as F

class LTATransformer(nn.Module):
    def __init__(self, item_vocab_size, archetype_vocab_size=8, d_model=128, nhead=8, num_layers=4):
        super(LTATransformer, self).__init__()
        
        self.item_vocab_size = item_vocab_size
        self.item_embedding = nn.Embedding(item_vocab_size + 1, d_model, padding_idx=0)
        self.archetype_embedding = nn.Embedding(archetype_vocab_size, d_model)
        self.numeric_projection = nn.Linear(3, d_model)
        
        # Positional Encoding for Inventory (Order matters!)
        self.inv_pos_emb = nn.Embedding(7, d_model) # 6 item slots + 1
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Linear(d_model, item_vocab_size)

    def forward(self, player_numeric, player_archetype, enemy_numeric, enemy_archetypes, current_inventory=None, mask=None):
        """
        mask: [batch, item_vocab_size] tensor of 1s and 0s
        """
        p_num_emb = self.numeric_projection(player_numeric).unsqueeze(1)
        p_arch_emb = self.archetype_embedding(player_archetype)
        
        if current_inventory is not None:
            # LLM-STYLE: Embed items AND their "positions" (slots)
            # current_inventory: [batch, 6]
            positions = torch.arange(current_inventory.size(1), device=current_inventory.device).unsqueeze(0)
            inv_emb = self.item_embedding(current_inventory) + self.inv_pos_emb(positions)
            # Combine all owned items into the player token
            inv_context = inv_emb.mean(dim=1, keepdim=True)
            player_token = p_num_emb + p_arch_emb + inv_context
        else:
            player_token = p_num_emb + p_arch_emb
        
        e_num_emb = self.numeric_projection(enemy_numeric)
        e_arch_emb = self.archetype_embedding(enemy_archetypes)
        enemy_tokens = e_num_emb + e_arch_emb
        
        sequence = torch.cat([player_token, enemy_tokens], dim=1)
        transformed = self.transformer(sequence)
        
        logits = self.output_head(transformed[:, 0, :])
        
        # LLM-STYLE LOGIT MASKING
        if mask is not None:
            # mask is 1 where we want to BLOCK the item
            logits = logits.masked_fill(mask.bool(), -1e9)
        
        return logits

