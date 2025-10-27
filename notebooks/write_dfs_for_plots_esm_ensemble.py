import os
import pandas as pd
import pickle
from pathlib import Path
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.df_for_plots import get_site_mutabilities_df
from epam.oe_plot import (
    get_numbering_dict,
    get_site_substitutions_df,
    get_site_subs_acc_df,
)

epam_results_dir = "epam_output"
anarci_dir = "anarci"
output_dir = "dataframes"
os.makedirs(output_dir, exist_ok=True)

dataset_info = (
    "rodriguez-airr-seq-race-prod_pcp_2024-07-28_MASKED_NI_noN_no-naive",
    "rodriguez",
    f"{anarci_dir}/rodriguez-airr-seq-race-prod_imgt.csv"
)

model_list = [
    "S5FESM_mask", "ThriftyESM_mask", "ESM1v_mask"
]

esm_ensemble_models = [
    "esm1", "esm2", "esm3", "esm4", "esm5", "ensemble_set"
]

dataset = dataset_info[0]
dsname = dataset_info[1]
anarci_path = dataset_info[2]

pcp_path = pcp_path_of_aaprob_path(f"{epam_results_dir}/{dataset}/esm1/S5FESM_mask/combined_aaprob.hdf5")
pcp_df = load_and_filter_pcp_df(pcp_path)

nb_path = Path(f'{output_dir}/{dsname}_numbering.pkl')
if nb_path.exists() and nb_path.is_file():
    with open(nb_path, 'rb') as f:
        numbering = pickle.load(f)
else:
    numbering, excluded = get_numbering_dict(anarci_path, pcp_df, True, "imgt")
    with open(nb_path, 'wb') as f:
        pickle.dump(numbering, f, pickle.HIGHEST_PROTOCOL)
    
for esm_model in esm_ensemble_models:
    print("ESM Model:", esm_model)

    for model in model_list:
        
        print("Model:", model)
        
        aaprob_path = f"{epam_results_dir}/{dataset}/{esm_model}/{model}/combined_aaprob.hdf5"
        
        # dataframe of site substitution probabilities (SSPs)
        sitemuts_df = get_site_mutabilities_df(aaprob_path, numbering)
        sitemuts_df.to_csv(f"{output_dir}/{dsname}_{model}_{esm_model}_ssp_df.csv.gz")
