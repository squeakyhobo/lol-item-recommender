import torch
import json
import os
import sys
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
import statistics

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
import config
from model_v2 import MoEYorickMLP
from train_v2 import V2Dataset

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(os.path.join(DATA_DIR, "item_clusters.json"), "r") as f:
        item_to_cluster = json.load(f)["item_to_cluster"]
with open(config.V2_EPISODES_PATH, "r") as f:
        episodes = json.load(f)
with open(os.path.join(DATA_DIR, "item_vocab.json"), "r") as f:
            v = json.load(f)
            vocab = v["item_to_index"]
            inv_vocab = v["index_to_item"]
            vocab_size = v["size"]
            champ_masks = v["champ_mask"]
with open(os.path.join(DATA_DIR,"champion_dna.json"),'r') as f:
       champ_dna = json.load(f)
with open(os.path.join(DATA_DIR,"item_names.json"),'r') as f:
       id_to_item_name = json.load(f)
with open(os.path.join(DATA_DIR, "champion_knowledge.json"), "r") as f: 
            kb = json.load(f)
            idx_to_champ = {i: name for i, name in enumerate(sorted(kb.keys()))}

model = MoEYorickMLP(num_champs=200, num_runes=20, num_items=vocab_size, numerical_dim=99 + vocab_size).to(device)
model.load_state_dict(torch.load(config.V2_MODEL_PATH, map_location=device))
model.eval()
          


def analyse_experts():

    
    expert_analysis ={}
    champ_t =[]
    opp_champ_t =[]
    expert_t =[]
    item_t =[]
    rune_t =[]
    minute_t =[]
    
    gold_t =[]

    i =0
    
    dataset = V2Dataset(episodes, vocab, item_to_cluster)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    for my_id, opp_id, rune_id, inv_ids, num_t, _ ,_ in loader:
        my_id, opp_id, rune_id, inv_ids, num_t = my_id.to(device), opp_id.to(device), rune_id.to(device), inv_ids.to(device), num_t.to(device)
        with torch.no_grad():
            _,i_logits,_,gate_probs = model(my_id, opp_id, rune_id, inv_ids, num_t)
        
        expert_idx = torch.argmax(gate_probs,dim=1) #shape batch_size,
        item_idx = torch.argmax(i_logits,dim=1)
        minute =num_t[:,0]
        total_gold =num_t[:,1]
        
        

        champ_t.append(my_id)
        opp_champ_t.append(opp_id)
        item_t.append(item_idx)
        rune_t.append(rune_id)
        minute_t.append(minute)
        gold_t.append(total_gold)
        expert_t.append(expert_idx)


# add gate distrubution 

      
        
       
    
    expert_t =torch.cat(expert_t)
    item_t = torch.cat(item_t)
    rune_t =torch.cat(rune_t)
    minute_t =torch.cat(minute_t)
    gold_t = torch.cat(gold_t)
    champ_t = torch.cat(champ_t)
    opp_champ_t = torch.cat(opp_champ_t)

   
    
       

    # for each expert , i will make a mask
    for i in range(len(model.experts)):
           expert_dict ={}
           expert_mask = (expert_t == i)
           expert_item_t =(torch.masked_select(item_t,expert_mask))
           expert_champ_t =(torch.masked_select(champ_t,expert_mask))
           

           if len(expert_item_t)!=0:
                  
                common_item_idx = torch.mode(expert_item_t).values.item()
                
                item_id = inv_vocab[str(common_item_idx)]
                item_name = id_to_item_name[str(item_id)]

                expert_dict["common item"] = item_name
             
           if len(expert_champ_t)!=0:
                 common_champ_idx = torch.mode(expert_champ_t).values.item()
                 champ_name = idx_to_champ[common_champ_idx]
                 expert_dict["commom champ"] = champ_name
            
           expert_analysis[i] =expert_dict
                 



    with open(os.path.join(DATA_DIR,"expert_analysis.json"),'w') as f:
        json.dump(expert_analysis,f)

    

    

        

  

if __name__ == '__main__':
    analyse_experts()
