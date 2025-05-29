import os
import pandas as pd
from epam.df_for_plots import (
    get_site_mutabilities_df,
    get_subs_and_preds_from_aaprob,
    get_site_substitutions_df,
    get_sub_acc_from_aaprob,
    get_site_subs_acc_df,
    get_site_csp_df,
)

epam_results_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2/gcreplay"
epam_esm_results_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2/gcreplay_esm"
output_dir = "dataframes"
os.makedirs(output_dir, exist_ok=True)

model_list = [
    "GCReplaySHM", "GCReplaySHMDMSSigmoid", "GCReplayAbLang2", "GCReplayESM",
    "GCReplaySHMBLOSUMSigmoid", "GCReplaySHMESMSigmoid"
]

for chain in ['igh','igk']:
    print('chain:',chain)
    
    dataset = f"gctrees_2025-01-10-full_{chain}_pcp_NoBackMuts"
    
    for model in model_list:
        print("Model:", model)
        
        if "ESM" in model:
            aaprob_path = f"{epam_esm_results_dir}/{chain}/{dataset}/{model}_{chain}/ensemble_esm/aaprob.hdf5"
        else:
            aaprob_path = f"{epam_results_dir}/{chain}/{dataset}/{model}_{chain}/aaprob.hdf5"
        
        # dataframe marking sites of observed and/or predicted (i.e. top-k) substitutions
        muts_obs_pred_df = get_site_substitutions_df(get_subs_and_preds_from_aaprob(aaprob_path))
        muts_obs_pred_df.to_csv(f"{output_dir}/{model}_{chain}_site_subs_df.csv.gz")
        
        # dataframe of site substitution probabilities (SSPs)
        sitemuts_df = get_site_mutabilities_df(aaprob_path)
        sitemuts_df.to_csv(f"{output_dir}/{model}_{chain}_ssp_df.csv.gz")
        
        # dataframe of substitution sites and whether the correct amino acid was predicted (i.e. top CSP)
        subacc_df = get_site_subs_acc_df(get_sub_acc_from_aaprob(aaprob_path, 1))
        subacc_df.to_csv(f"{output_dir}/{model}_{chain}_site_subacc_df.csv.gz")
        
        # dataframe of conditional subsitution probabilities (CSPs)
        csp_df = get_site_csp_df(aaprob_path)
        csp_df.to_csv(f"{output_dir}/{model}_{chain}_csp_df.csv.gz")