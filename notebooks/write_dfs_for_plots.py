import os
import pandas as pd
import pickle
from pathlib import Path
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.df_for_plots import (
    get_site_mutabilities_df,
    get_subs_and_preds_from_aaprob,
    get_sub_acc_from_aaprob,
    get_site_csp_df,
)
from epam.oe_plot import (
    get_numbering_dict,
    get_site_substitutions_df,
    get_site_subs_acc_df,
)

epam_results_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2"
anarci_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci"
output_dir = "dataframes"
os.makedirs(output_dir, exist_ok=True)

dataset_list = [
("ford-flairr-seq-prod_pcp_2024-07-26_MASKED_NI_noN_no-naive",
 "ford",
 f"{anarci_dir}/ford-flairr-seq-prod_imgt.csv"),

("rodriguez-airr-seq-race-prod_pcp_2024-07-28_MASKED_NI_noN_no-naive",
 "rodriguez",
 f"{anarci_dir}/rodriguez-airr-seq-race-prod_imgt.csv"),

("tang-deepshm-prod_pcp_2024-08-08_MASKED_NI_noN_no-naive",
 "tang",
 f"{anarci_dir}/tang-deepshm-prod_imgt.csv"),

("wyatt-10x-1p5m_paired-igh_fs-all_pcp_2024-11-22_NI_noN_no-naive",
 "wyatt",
 f"{anarci_dir}/wyatt-10x-1p5m_paired-igh_fs-all_imgt.csv"),
]

model_list = [
    "S5F", "S5FESM_mask", "S5FBLOSUM",
    "ThriftyHumV0.2-59", "ThriftyProdHumV0.2-59", "ThriftyESM_mask", "ThriftyBLOSUM",
    "ESM1v_mask", "AbLang2_mask", "AbLang1"
]

for dsinfo in dataset_list:
    print("dataset:", dsinfo[0])
    dataset = dsinfo[0]
    dsname = dsinfo[1]
    anarci_path = dsinfo[2]
    
    pcp_path = pcp_path_of_aaprob_path(f"{epam_results_dir}/{dataset}/S5F/combined_aaprob.hdf5")
    pcp_df = load_and_filter_pcp_df(pcp_path)

    nb_path = Path(f'{output_dir}/{dsname}_numbering.pkl')
    if nb_path.exists() and nb_path.is_file():
        with open(nb_path, 'rb') as f:
            numbering = pickle.load(f)
    else:
        numbering, excluded = get_numbering_dict(anarci_path, pcp_df, True, "imgt")
        with open(nb_path, 'wb') as f:
            pickle.dump(numbering, f, pickle.HIGHEST_PROTOCOL)
    
    
    for model in model_list:
        print("Model:", model)
        
        aaprob_path = f"{epam_results_dir}/{dataset}/{model}/combined_aaprob.hdf5"

        # dataframe marking sites of observed and/or predicted (i.e. top-k) substitutions
        muts_obs_pred_df = get_site_substitutions_df(get_subs_and_preds_from_aaprob(aaprob_path), numbering)
        muts_obs_pred_df.to_csv(f"{output_dir}/{dsname}_{model}_site_subs_df.csv.gz")
        
        # dataframe of site substitution probabilities (SSPs)
        sitemuts_df = get_site_mutabilities_df(aaprob_path, numbering)
        sitemuts_df.to_csv(f"{output_dir}/{dsname}_{model}_ssp_df.csv.gz")
        
        # dataframe of substitution sites and whether the correct amino acid was predicted (i.e. top CSP)
        subacc_df = get_site_subs_acc_df(get_sub_acc_from_aaprob(aaprob_path, 1), numbering)
        subacc_df.to_csv(f"{output_dir}/{dsname}_{model}_site_subacc_df.csv.gz")
        
        # dataframe of conditional subsitution probabilities (CSPs)
        csp_df = get_site_csp_df(aaprob_path, numbering)
        csp_df.to_csv(f"{output_dir}/{dsname}_{model}_csp_df.csv.gz")
