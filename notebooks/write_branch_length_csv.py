# Plot plots for the distribution of optimized branch lengths for all models and datasets.
# (Supplementary figures)

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

path_to_bl_data = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2"
path_to_pcp_data = "pcp_inputs/"
output_dir = "tables/"

dataset_list = [
    ("ford", "Ford et al.", "ford-flairr-seq-prod_pcp_2024-07-26_MASKED_NI_noN_no-naive"),
    ("rodriguez", "Rodriguez et al.", "rodriguez-airr-seq-race-prod_pcp_2024-07-28_MASKED_NI_noN_no-naive"),
    ("tang", "Tang et al.", "tang-deepshm-prod_pcp_2024-08-08_MASKED_NI_noN_no-naive"),
    ("wyatt", "Jaffe et al.", "wyatt-10x-1p5m_paired-igh_fs-all_pcp_2024-11-22_NI_noN_no-naive"),
]

model_list = [
    "S5F", "S5FESM_mask", "S5FBLOSUM",
    "ThriftyHumV0.2-59", "ThriftyProdHumV0.2-59", "ThriftyESM_mask", "ThriftyBLOSUM",
    "ESM1v_mask", "AbLang2_mask", "AbLang1"
]


def create_combined_branch_lengths(model_path, opt_bl_path):

    batch_pattern = os.path.join(model_path, "batch*", "optimized_branch_lengths.csv")
    batch_files = glob.glob(batch_pattern)

    dataframes = []
    for batch_file in sorted(batch_files):
        df = pd.read_csv(batch_file)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(opt_bl_path, index=False)


for dsinfo in dataset_list:
    print("Dataset:", dsinfo[0])
    dsname = dsinfo[0]
    dstitle = dsinfo[1]
    dsfilename = dsinfo[2]
    dspath = os.path.join(path_to_bl_data, dsfilename)

    dataset_model_data = []

    for model in model_list:
        mpath = os.path.join(dspath, model)
        print(model)

        combined_file = os.path.join(output_dir, f"{dsname}_{model}_optimized_branch_lengths.csv.gz")
        if not os.path.exists(combined_file):
            create_combined_branch_lengths(mpath, combined_file)

