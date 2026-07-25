import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class YorickMLP(nn.Module):
    """
    Maiden Brain V2.3 (Ultra-Brain): Optimized for Multi-Champion Mastery.
    Supports Yorick, Sett, Mundo, Trundle, Morde, Garen.
    Scaled to 2048 neurons with heavy regularization.
    """
    def __init__(self, num_champs=200, num_runes=20, num_items=65, numerical_dim=162):
        super(YorickMLP, self).__init__()

        # 1. Categorical Embeddings 
        #Note is champ embed and DNA both NEEDED?
        self.champ_emb = nn.Embedding(num_champs, 64) # Increased embedding size
        #Note - is this that important? main rune I feel like not really for picking items 
        self.rune_emb = nn.Embedding(num_runes, 32)
        
        self.inv_emb = nn.Embedding(10000, 16)
        

        # 2. Dense Layers for Numerical Features
        # Dimensions: Champ(64) + Opp(64) + Rune(32) + Inventory(6*16=96) + Numerical Features (162)
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


class Expert(nn.Module):
    def __init__(self,input_size,num_items):
        super(Expert,self).__init__()
        #1024 -> 512 -> 256
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size,1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.4),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),

            nn.Linear(512,256),
            nn.ReLU(),
        )
    
    def forward(self,x):
        return(self.hidden_layers(x))
    
        
    
class MoEYorickMLP(nn.Module):
    def __init__(self, num_champs=200, num_runes=20, num_items=65, numerical_dim=162,num_experts=4):
        super(MoEYorickMLP, self).__init__()
    
        #Note should I also use embeddings here- i feel like I shoiuld 
        #Node soft vs spaerse MOE 
        self.champ_emb = nn.Embedding(num_champs, 128) 
        # 
        self.rune_emb = nn.Embedding(num_runes, 32)
        
        self.inv_emb = nn.Embedding(10000, 16)
        
        input_size = self.champ_emb.embedding_dim + self.champ_emb.embedding_dim + 32 + 96 + numerical_dim
        self.gate = nn.Sequential(nn.Linear(input_size, num_experts), 
                                  nn.Softmax(dim=-1))
        
        self.experts = nn.ModuleList([Expert(input_size,num_items) for _ in range(num_experts)])

        self.strategy_head = nn.Linear(256, 8) 
        self.item_head = nn.Linear(256, num_items)

        
    
    





    def forward(self, my_id, opp_id, rune_id, inv_ids, numerical_feats, item_mask=None):

        #get the embeds 
        my_embed = self.champ_emb(my_id)
        opp_embed = self.champ_emb(opp_id)
        rune_embed = self.rune_emb(rune_id)#Note maybe for opponent too?
        inv_embed = self.inv_emb(inv_ids).view(inv_ids.size(0), -1)

        #the numerical stuff so my dna opp dna thread dna moinute gold etc
        x = torch.cat([my_embed, opp_embed, rune_embed, inv_embed, numerical_feats], dim=1)

        # 1. Get Top 2 Experts
        gate_probs = self.gate(x) # shape [batch_size, expert_num]
        top2_probs, top2_idx = torch.topk(gate_probs, k=2, dim=1) # shape [batch_size, 2]
        
        # 2. Re-normalize (with keepdim=True)
        normalised_probs = top2_probs / top2_probs.sum(dim=1, keepdim=True) # shape [batch_size, 2]
        
        # 3. Create a blank canvas to store the final blended answers
        # The experts output 256 features
        combined = torch.zeros(x.size(0), 256, device=x.device)
        
        # 4. True Sparse Routing: Only run the experts that were actually selected!
        for expert_id, expert in enumerate(self.experts):
            # Find which rows in the batch picked this specific expert
            batch_rows, topk_pos = torch.where(top2_idx == expert_id)
            
            if batch_rows.numel() > 0:
                # Extract ONLY the data for those specific rows
                expert_inputs = x[batch_rows]
                
                # Run the expert (this saves massive CPU/GPU time by skipping ignored experts!)
                expert_outputs = expert(expert_inputs)
                
                # Grab the multipliers for these specific rows
                expert_weights = normalised_probs[batch_rows, topk_pos].unsqueeze(-1)
                
                # Add the weighted answers into our final canvas
                combined[batch_rows] += expert_outputs * expert_weights
        #Note -implement my onw mask from scratch myself 
        s_logits = self.strategy_head(combined)
        i_logits = self.item_head(combined)
        if item_mask is not None:
            i_logits = i_logits.masked_fill(item_mask.bool(), -1e9)
        aux_loss = torch.var(gate_probs.mean(dim=0))
        return s_logits, i_logits, aux_loss, gate_probs


    
        