import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import pandas as pd
import torch
import h5py
from esm import pretrained
from netam.sequences import (
    AA_STR_SORTED,
)
from netam.common import pick_device

output_dir = "/home/mjohnso4/epam/data/gcreplay"
replay_igh = "EVQLQESGPSLVKPSQTLSLTCSVTGDSITSGYWNWIRKFPGNKLEYMGYISYSGSTYYNPSLKSRISITRDTSKNQYYLQLNSVTTEDTATYYCARDFDVWGAGTTVTVSS"
replay_igk = "DIVMTQSQKFMSTSVGDRVSVTCKASQNVGTNVAWYQQKPGQSPKALIYSASYRYSGVPDRFTGSGSGTDFTLTISNVQSEDLAEYFCQQYNSYPLTFGSGTKLEIKR"


def write_naive_probs(model_number, output_hdf5):
    
    model_location = f"esm1v_t33_650M_UR90S_{model_number}"
    
    device = pick_device()
    
    # Initialize the model
    print(f"Loading model {model_location}...")
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    model = model.to(device)
    aa_idxs = [alphabet.get_idx(aa) for aa in AA_STR_SORTED]
    
    batch_converter = alphabet.get_batch_converter()
    
    sequences_aa = [replay_igh, replay_igk]
    protein_ids = [f"protein{i}" for i in range(len(sequences_aa))]
    
    with h5py.File(output_hdf5, "w") as outfile:
        # attributes related to PCP data file
        outfile.attrs["data_set"] = "replay_igh_igk"
        outfile.attrs["model_name"] = f"masked_{model_location}"
        
        for seq in range(len(sequences_aa)):
            print(f"Processing sequence {seq+1}/{len(sequences_aa)} with model {model_number}...")
            
            data = list(zip([protein_ids[seq]], [sequences_aa[seq]]))
            batch_tokens = batch_converter(data)[2]
            
            # Mask each site in the sequence to get token probabilities before softmax.
            all_token_probs = []
            for site in range(batch_tokens.size(1)):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, site] = alphabet.mask_idx
                
                with torch.no_grad():
                    batch_tokens_masked = batch_tokens_masked.to(device)
                    token_probs_pre_softmax = model(batch_tokens_masked)["logits"]
                
                all_token_probs.append(token_probs_pre_softmax[:, site])
            
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            
            aa_probs = torch.softmax(token_probs[..., aa_idxs], dim=-1)
            
            aa_probs_np = aa_probs.cpu().numpy().squeeze()
            
            len_seq = len(sequences_aa[seq])
            matrix = aa_probs_np[1 : len_seq + 1, :]
            
            parent = sequences_aa[seq]
            
            outfile.create_dataset(
                f"{parent}", data=matrix, compression="gzip", compression_opts=4
            )


for model_number in range(1, 6):
    print(f"Processing model number {model_number}...")
    
    output_hdf5 = f"{output_dir}/replay_igh_igk_esm{model_number}.hdf5"
    
    write_naive_logits(model_number, output_hdf5)
    
