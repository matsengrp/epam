import os
import h5py
import numpy as np
import pandas as pd
import glob
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.sequences import translate_sequences
from epam.evaluation import calculate_site_substitution_probabilities


modelname = 'GCReplayOptSHMDMSSigmoid'
chain = 'igk'

output_dir = f'pseudo_{chain}'
os.makedirs(output_dir, exist_ok=True)

pseudodirs = glob.glob(f'output_pseudo/{chain}/{chain}_pcp_*')

for resdir in pseudodirs:
    print(resdir)
    seed = resdir.split('_')[-1]
    aaprob_path = f'{resdir}/{modelname}_{chain}/aaprob.hdf5'
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    pcp_index_col = []
    sites_col = []
    site_sub_probs = []
    site_sub_flags = []
    with h5py.File(aaprob_path, "r") as matfile:
        model_name = matfile.attrs["model_name"]
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]
            
            parent = parent_aa_seqs[index]
            child = child_aa_seqs[index]
            
            pcp_index_col = np.concatenate((pcp_index_col, [pcp_index]*len(parent)))
            sites_col = np.concatenate((sites_col, np.arange(len(parent))))
            site_sub_probs = np.concatenate((site_sub_probs, calculate_site_substitution_probabilities(matrix, parent)))
            site_sub_flags = np.concatenate((site_sub_flags, [p!=c for p,c in zip(parent, child)]))

    #print(len(pcp_index_col), len(sites_col), len(site_sub_probs), len(site_sub_flags))
    output_df = pd.DataFrame(columns=['pcp_index','site','prob','mutation'])
    output_df['pcp_index'] = pcp_index_col
    output_df['site'] = sites_col
    output_df['prob'] = site_sub_probs
    output_df['mutation'] = site_sub_flags
    output_df.to_csv(f'{output_dir}/{chain}_{seed}_site_sub_probs.csv', index=False)
