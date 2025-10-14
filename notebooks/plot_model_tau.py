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
path_to_pcp_data = "/home/mjohnso4/epam/pcp_inputs"
output_dir = "/home/mjohnso4/epam/output/plots/branch_lengths"

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

modelname_list = [
    "S5F", "S5F + ESM-1v", "S5F + BLOSUM62",
    "Thrifty-SHM", "Thrifty-prod", "Thrifty-SHM + ESM-1v", "Thrifty-SHM + BLOSUM62",
    "ESM-1v", "AbLang2", "AbLang1"
]

modelname_plot_list = [
    "IQ-TREE",
    "S5F", "S5F + ESM-1v", "S5F + BLOSUM62",
    "Thrifty-SHM", "Thrifty-prod", "Thrifty-SHM + ESM-1v", "Thrifty-SHM + BLOSUM62",
    "ESM-1v", "AbLang2", "AbLang1"
]


def create_combined_branch_lengths(mpath):

    combined_file_path = os.path.join(mpath, "combined_optimized_branch_lengths.csv")

    batch_pattern = os.path.join(mpath, "batch*", "optimized_branch_lengths.csv")
    batch_files = glob.glob(batch_pattern)
    
    dataframes = []
    for batch_file in sorted(batch_files):
        df = pd.read_csv(batch_file)
        # batch_name = os.path.basename(os.path.dirname(batch_file))
        # df['batch'] = batch_name
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(combined_file_path, index=False)


def create_violin_plots(dataset_data, dsname, output_dir):
    # Get unique datasets
    datasets = [ds[1] for ds in dataset_list]
    n_datasets = len(datasets)
    
    # Create figure with single row and datasets as columns (log scale only)
    fig, axes = plt.subplots(1, n_datasets, figsize=(8*n_datasets, 8))
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ds_data = dataset_data[dataset_data['dataset'] == dataset]
        log_data = ds_data[ds_data['opt_branch_length'] > 0]
        
        violin_parts = ax.violinplot([log_data[log_data['model'] == model]['opt_branch_length'].dropna() 
                                     for model in modelname_list], 
                                    positions=range(len(modelname_list)), 
                                    vert=False,  # Horizontal violins
                                    showmeans=False, showmedians=True)
        
        ax.set_yticks(range(len(modelname_list)))
        if idx == 0:  # Only show labels on leftmost panel
            ax.set_yticklabels(modelname_list)
            ax.set_ylabel('Model')
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        ax.set_xlabel('Optimized Branch Length (log scale)')
        ax.set_title(f'{dataset}')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Color customization
        colors = sns.color_palette("husl", len(modelname_list))
        for i, vp in enumerate(violin_parts['bodies']):
            vp.set_facecolor(colors[i])
            vp.set_alpha(0.7)
        
        # Set lines to black
        for part_name in ['cbars', 'cmins', 'cmaxes', 'cquartiles', 'cmedians', 'cmeans']:
            if part_name in violin_parts:
                violin_parts[part_name].set_color('black')
    
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f"{dsname}_branch_length_distributions.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()


def create_iqtree_violin_plots(dataset_data, dsname, output_dir):
    # Reorder datasets to match: Tang, Jaffe, Rodriguez, Ford
    dataset_order = ["Tang et al.", "Jaffe et al.", "Rodriguez et al.", "Ford et al."]
    n_datasets = len(dataset_order)
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    for idx, dataset in enumerate(dataset_order):
        ax = axes[idx]
        ds_data = dataset_data[dataset_data['dataset'] == dataset]
        log_data = ds_data[ds_data['opt_branch_length'] > 0]
        
        violin_parts = ax.violinplot([log_data[log_data['model'] == model]['opt_branch_length'].dropna() 
                                     for model in modelname_plot_list], 
                                    positions=range(len(modelname_plot_list)), 
                                    vert=False,  # Horizontal violins
                                    showmeans=False, showmedians=True)
        
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_yticks(range(len(modelname_plot_list)))
        # Show labels only on left column (idx 0 and 2)
        if idx % 2 == 0:
            ax.set_yticklabels(modelname_plot_list, fontsize=24)
            # ax.set_ylabel('Model', fontsize=30)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        # ax.set_xlabel('Optimized Branch Length (log scale)', fontsize=30)
        ax.set_title(f'{dataset}', fontsize=24)
        ax.set_xscale('log')
        ax.tick_params(axis="x", labelsize=24)
        ax.grid(True, alpha=0.3)
        
        # Colors and styling
        colors = sns.color_palette("husl", len(modelname_plot_list))
        for i, vp in enumerate(violin_parts['bodies']):
            vp.set_facecolor(colors[i])
            vp.set_alpha(0.7)
        
        for part_name in ['cbars', 'cmins', 'cmaxes', 'cquartiles', 'cmedians', 'cmeans']:
            if part_name in violin_parts:
                violin_parts[part_name].set_color('black')
    
        # Add a single centered x-axis label at the bottom
    fig.text(0.6, 0.02, 'optimized branch length (log scale)', ha='center', fontsize=30)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for label
    output_filename = os.path.join(output_dir, f"{dsname}_bl_opt_v_iqtree.pdf")
    output_filename2 = os.path.join(output_dir, f"{dsname}_bl_opt_v_iqtree.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.savefig(output_filename2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()
    

all_datasets_data = []

for dsinfo in dataset_list:
    print("Dataset:", dsinfo[0])
    dsname = dsinfo[0]
    dstitle = dsinfo[1]
    dsfilename = dsinfo[2]
    dspath = os.path.join(path_to_bl_data, dsfilename)
    
    dataset_model_data = []

    for model, modelname in zip(model_list, modelname_list):
        mpath = os.path.join(dspath, model)
        print(model)

        combined_file = os.path.join(mpath, "combined_optimized_branch_lengths.csv")
        if not os.path.exists(combined_file):
            create_combined_branch_lengths(mpath)

        df = pd.read_csv(combined_file)
        df['model'] = modelname
        df['dataset'] = dstitle  # Add dataset identifier

        dataset_model_data.append(df)

    combined_dataset = pd.concat(dataset_model_data, ignore_index=True)
    
    # Add IQ-TREE data
    pcp_df = pd.read_csv(os.path.join(path_to_pcp_data, f"{dsfilename}.csv"))
    pcp_bl_df = pcp_df[['branch_length']].copy()
    pcp_bl_df = pcp_bl_df.rename(columns={'branch_length': 'opt_branch_length'})
    pcp_bl_df['model'] = "IQ-TREE"
    pcp_bl_df['pcp_index'] = pcp_df.index
    pcp_bl_df['dataset'] = dstitle
    
    combined_dataset = pd.concat([combined_dataset, pcp_bl_df], ignore_index=True)
    all_datasets_data.append(combined_dataset)

# Combine all datasets
full_data = pd.concat(all_datasets_data, ignore_index=True)

# Create the combined plots
# create_violin_plots(full_data, "all_datasets", output_dir)
create_iqtree_violin_plots(full_data, "all_datasets", output_dir)