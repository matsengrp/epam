# Plot plots for the distribution of optimized branch lengths for all models and datasets.
# (Supplementary figures)

import os
import glob
import numpy as np
import pandas as pd
# import h5py
# from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
# from epam.evaluation import calculate_site_substitution_probabilities
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

path_to_bl_data = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2"
path_to_ssp_data = "/fh/fast/matsen_e/ksung2/epam/notebooks/dataframes"
output_dir = "/home/mjohnso4/epam/output/plots/branch_lengths"

dataset_list = [
    ("tang", "Tang et al.", "tang-deepshm-prod_pcp_2024-08-08_MASKED_NI_noN_no-naive"),
    ("wyatt", "Jaffe et al.", "wyatt-10x-1p5m_paired-igh_fs-all_pcp_2024-11-22_NI_noN_no-naive"),
    ("rodriguez", "Rodriguez et al.", "rodriguez-airr-seq-race-prod_pcp_2024-07-28_MASKED_NI_noN_no-naive"),
    ("ford", "Ford et al.", "ford-flairr-seq-prod_pcp_2024-07-26_MASKED_NI_noN_no-naive")
]

model_list = [
    "S5F", "ThriftyHumV0.2-59"
]

modelname_list = [
    "S5F", "Thrifty-SHM"
]

# p = 1 - exp(- \lambda * t_opt) 
# \lambda from S5F or Thrifty-SHM

def calculate_expected_mutations(ssp_df):
    """
    Calculate expected number of mutations per pcp_index.
    Returns exp_mut and exp_mut_norm.
    """
    exp_mut_df = ssp_df.groupby('pcp_index').agg(
        exp_mut=('prob', 'sum'),
        num_sites=('prob', 'count')
    ).reset_index()
    
    exp_mut_df['exp_mut_norm'] = exp_mut_df['exp_mut'] / exp_mut_df['num_sites']
    
    # Drop num_sites if you don't need it in the final output
    exp_mut_df = exp_mut_df[['pcp_index', 'exp_mut', 'exp_mut_norm']]
    
    return exp_mut_df

def plot_exp_mut_v_bl_grid(all_data, dataset_list, modelname_list, output_dir):
    # Create 4x2 subplot figure
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    for row_idx, dsinfo in enumerate(dataset_list):
        dsname = dsinfo[0]
        dstitle = dsinfo[1]
        
        # First pass: determine the max count for this row
        hexbins = []
        for col_idx, modelname in enumerate(modelname_list):
            ax = axes[row_idx, col_idx]
            
            # Get the data for this dataset-model combination
            if dsname in all_data and modelname in all_data[dsname]:
                merged_df = all_data[dsname][modelname]
                
                # Create hexbin plot with log scales
                hexbin = ax.hexbin(merged_df['exp_mut_norm'], merged_df['opt_branch_length'], 
                                   xscale='log', yscale='log', 
                                   gridsize=30, cmap='viridis', mincnt=1)
                hexbins.append(hexbin)
            else:
                hexbins.append(None)
        
        # Get the maximum count across both models in this row
        vmax = max([hb.get_array().max() for hb in hexbins if hb is not None])
        
        # Second pass: redraw with shared scale and add colorbar
        for col_idx, modelname in enumerate(modelname_list):
            ax = axes[row_idx, col_idx]
            ax.clear()  # Clear the first plot
            
            # Get the data for this dataset-model combination
            if dsname in all_data and modelname in all_data[dsname]:
                merged_df = all_data[dsname][modelname]
                
                # Create hexbin plot with shared vmax
                hexbin = ax.hexbin(merged_df['exp_mut_norm'], merged_df['opt_branch_length'], 
                                   xscale='log', yscale='log', 
                                   gridsize=30, cmap='viridis', mincnt=1, 
                                   vmin=1, vmax=vmax)
                
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                # Add colorbar for the second column
                if col_idx == 1:
                    cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('count', fontsize=15)
                    cbar.ax.tick_params(labelsize=12)
                
                ax.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.3)
            
            # Add title for top row (model names)
            if row_idx == 0:
                ax.set_title(modelname, fontsize=20)
            
            # Add y-label for first column (dataset names)
            if col_idx == 0:
                ax.set_ylabel(f'{dstitle}\noptimized branch length', fontsize=20)
    
    # Add centered x-axis label for bottom row
    fig.text(0.5, 0.02, 'expected number of mutations / sequence length', 
             ha='center', fontsize=20)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space at bottom for x-label
    
    output_path = os.path.join(output_dir, 'all_datasets_exp_mut_vs_opt_bl.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    
    print(f"\nPlot saved to {output_path}")


all_data = {}

for dsinfo in dataset_list:
    print("Dataset:", dsinfo[0])
    dsname = dsinfo[0]
    dstitle = dsinfo[1]
    dsfilename = dsinfo[2]
    dspath = os.path.join(path_to_bl_data, dsfilename)
    
    all_data[dsname] = {}

    for model, modelname in zip(model_list, modelname_list):
        bl_path = os.path.join(dspath, model)
        print(model)

        combined_file = os.path.join(bl_path, "combined_optimized_branch_lengths.csv")
        if not os.path.exists(combined_file):
            print(f"File does not exist! Need to run create_combined_branch_lengths({bl_path}) in plot_model_tau.py")

        bl_df = pd.read_csv(combined_file)
        bl_df['model'] = modelname

        ssp_path = os.path.join(path_to_ssp_data, f"{dsname}_{model}_ssp_df.csv.gz")
        ssp_df = pd.read_csv(ssp_path, index_col=0, dtype={'site':'object'})
        exp_mut_df = calculate_expected_mutations(ssp_df)
        print(exp_mut_df.head())

        merged_df = bl_df.merge(exp_mut_df, on='pcp_index', how='inner')
        merged_df = merged_df[['model', 'pcp_index', 'exp_mut', 'exp_mut_norm', 'opt_branch_length']]

        all_data[dsname][modelname] = merged_df

plot_exp_mut_v_bl_grid(all_data, dataset_list, modelname_list, output_dir)


