# Write a dataframe of substitution accuracy for all models on Tang et al.
# (Figure 4A)

import h5py
import numpy as np
import pandas as pd
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.evaluation import (
    highest_k_substitutions,
    locate_child_substitutions,
)
from netam.oe_plot import (
    pcp_sites_regions,
)
from netam.sequences import (
    translate_sequences,
)

epam_results_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2"
top_k = 1

dataset = "tang-deepshm-prod_pcp_2024-08-08_MASKED_NI_noN_no-naive"
dsname = "tang"

outfname = f"{dsname}_subacc.csv"
output_df = pd.DataFrame(columns=['model','All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4'])

model_list = [
    "S5F", "S5FESM_mask", "S5FBLOSUM",
    "ThriftyHumV0.2-59", "ThriftyESM_mask", "ThriftyBLOSUM",
    "ThriftyProdHumV0.2-59",
    "ESM1v_mask", "AbLang2_mask", "AbLang1"
]

modelname_list = [
    "S5F", "S5F + ESM-1v", "S5F + BLOSUM62",
    "Thrifty-SHM", "Thrifty-SHM + ESM-1v", "Thrifty-SHM + BLOSUM62",
    "Thrifty-prod",
    "ESM-1v", "AbLang2", "AbLang1"
]

for model, modelname in zip(model_list, modelname_list):
    print("Model:", model)
    
    aaprob_path = f"{epam_results_dir}/{dataset}/{model}/combined_aaprob.hdf5"
    
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    pcp_sub_locations = [
        locate_child_substitutions(parent, child)
        for parent, child in zip(parent_aa_seqs, child_aa_seqs)
    ]

    pcp_sub_correct = []
    pcp_regions = []
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            pcp_row = pcp_df.loc[pcp_index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            parent_aa = parent_aa_seqs[index]
            child_aa = child_aa_seqs[index]
            pcp_sub_correct.append(
                [
                    child_aa[j]
                    in highest_k_substitutions(top_k, matrix[j, :], parent_aa, j)
                    for j in range(len(parent_aa))
                    if parent_aa[j] != child_aa[j]
                ]
            )
            
            regions_anno = pcp_sites_regions(pcp_row)
            pcp_regions.append(
                [
                    regions_anno[j]
                    for j in range(len(parent_aa))
                    if parent_aa[j] != child_aa[j]
                ]
            )
    
    df = pd.DataFrame(columns=['correct','region'])
    df['correct'] = np.concatenate(pcp_sub_correct)
    df['region'] = np.concatenate(pcp_regions)
    
    output_df.loc[len(output_df)] = [
        modelname,
        sum(df['correct'].to_numpy())/df.shape[0],
        sum(df[df['region']=='FWR1']['correct'].to_numpy())/df[df['region']=='FWR1'].shape[0],
        sum(df[df['region']=='CDR1']['correct'].to_numpy())/df[df['region']=='CDR1'].shape[0],
        sum(df[df['region']=='FWR2']['correct'].to_numpy())/df[df['region']=='FWR2'].shape[0],
        sum(df[df['region']=='CDR2']['correct'].to_numpy())/df[df['region']=='CDR2'].shape[0],
        sum(df[df['region']=='FWR3']['correct'].to_numpy())/df[df['region']=='FWR3'].shape[0],
        sum(df[df['region']=='CDR3']['correct'].to_numpy())/df[df['region']=='CDR3'].shape[0],
        sum(df[df['region']=='FWR4']['correct'].to_numpy())/df[df['region']=='FWR4'].shape[0],
    ]

print(output_df)
output_df.to_csv(outfname,index=False)
