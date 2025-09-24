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


def create_violin_plots(dataset_data, dsname, dstitle, output_dir):
    """
    Create violin plots for branch length distributions (linear and log scale)
    
    Args:
        dataset_data (pd.DataFrame): Combined data for all models in a dataset
        dsname (str): Dataset short name
        dstitle (str): Dataset title for plot
        output_dir (str): Directory to save figures
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with two subplots (linear and log scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Linear scale violin plot
    violin_parts1 = ax1.violinplot([dataset_data[dataset_data['model'] == model]['opt_branch_length'].dropna() 
                                   for model in modelname_list], 
                                  positions=range(len(modelname_list)), 
                                  showmeans=False, showmedians=True)
    
    ax1.set_xticks(range(len(modelname_list)))
    ax1.set_xticklabels(modelname_list, rotation=45, ha='right')
    ax1.set_ylabel('Optimized Branch Length')
    ax1.set_title(f'{dstitle} - Linear Scale')
    ax1.grid(True, alpha=0.3)
    
    # Log scale violin plot
    # Filter out zero and negative values for log scale
    log_data = dataset_data[dataset_data['opt_branch_length'] > 0]
    
    violin_parts2 = ax2.violinplot([log_data[log_data['model'] == model]['opt_branch_length'].dropna() 
                                   for model in modelname_list], 
                                  positions=range(len(modelname_list)), 
                                  showmeans=False, showmedians=True)
    
    ax2.set_xticks(range(len(modelname_list)))
    ax2.set_xticklabels(modelname_list, rotation=45, ha='right')
    ax2.set_ylabel('Optimized Branch Length (log scale)')
    ax2.set_title(f'{dstitle} - Log Scale')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Customize violin colors
    colors = sns.color_palette("husl", len(modelname_list))
    for i, (vp1, vp2) in enumerate(zip(violin_parts1['bodies'], violin_parts2['bodies'])):
        vp1.set_facecolor(colors[i])
        vp1.set_alpha(0.7)
        vp2.set_facecolor(colors[i])
        vp2.set_alpha(0.7)

    # Set internal lines to black (instead of default red)
    for part_name in ['cbars', 'cmins', 'cmaxes', 'cquartiles', 'cmedians', 'cmeans']:
        if part_name in violin_parts1:
            violin_parts1[part_name].set_color('black')
        if part_name in violin_parts2:
            violin_parts2[part_name].set_color('black')
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = os.path.join(output_dir, f"{dsname}_branch_length_distributions.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()


def create_iqtree_violin_plots(dataset_data, dsname, dstitle, output_dir):
    """
    Create violin plots for branch length distributions in log scale, comparing to the IQ-TREE model
    
    Args:
        dataset_data (pd.DataFrame): Combined data for all models in a dataset
        dsname (str): Dataset short name
        dstitle (str): Dataset title for plot
        output_dir (str): Directory to save figures
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Filter out zero and negative values for log scale
    log_data = dataset_data[dataset_data['opt_branch_length'] > 0]
    
    # Create violin plot for all models
    violin_parts = ax.violinplot([log_data[log_data['model'] == model]['opt_branch_length'].dropna() 
                                 for model in modelname_plot_list], 
                                positions=range(len(modelname_plot_list)), 
                                showmeans=False, showmedians=True)
    
    # Add vertical separation line after IQ-TREE (position 0)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xticks(range(len(modelname_plot_list)))
    ax.set_xticklabels(modelname_plot_list, rotation=45, ha='right')
    ax.set_ylabel('Optimized Branch Length (log scale)')
    ax.set_title(f'{dstitle}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Customize violin colors
    colors = sns.color_palette("husl", len(modelname_plot_list))
    for i, vp in enumerate(violin_parts['bodies']):
        vp.set_facecolor(colors[i])
        vp.set_alpha(0.7)

    # Set internal lines to black (instead of default red)
    for part_name in ['cbars', 'cmins', 'cmaxes', 'cquartiles', 'cmedians', 'cmeans']:
        if part_name in violin_parts:
            violin_parts[part_name].set_color('black')
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = os.path.join(output_dir, f"{dsname}_bl_opt_v_iqtree.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()


def create_bl_diff_violin_plots(dataset_data, dsname, dstitle, output_dir):
    """
    Create violin plots for the difference in branch lengths relative to IQ-TREE
    
    Args:
        dataset_data (pd.DataFrame): Combined data for all models in a dataset (must include IQ-TREE)
        dsname (str): Dataset short name
        dstitle (str): Dataset title for plot
        output_dir (str): Directory to save figures
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get IQ-TREE data as reference
    iqtree_data = dataset_data[dataset_data['model'] == 'IQ-TREE'].copy()
    iqtree_data = iqtree_data.set_index('pcp_index')
    
    # Calculate differences for each model
    diff_data = []
    for model in modelname_list:
        model_data = dataset_data[dataset_data['model'] == model].copy()
        model_data = model_data.set_index('pcp_index')
        
        # Merge with IQ-TREE data on pcp_index
        merged = model_data.merge(iqtree_data[['opt_branch_length']], 
                                 left_index=True, right_index=True, 
                                 suffixes=('_model', '_iqtree'))
        
        # Calculate difference
        merged['bl_diff'] = merged['opt_branch_length_model'] - merged['opt_branch_length_iqtree']
        
        diff_data.append(merged['bl_diff'])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create violin plot
    violin_parts = ax.violinplot(diff_data, 
                                positions=range(len(modelname_list)), 
                                showmeans=False, showmedians=True)
    
    ax.set_xticks(range(len(modelname_list)))
    ax.set_xticklabels(modelname_list, rotation=45, ha='right')
    ax.set_ylabel('Branch Length Difference vs IQ-TREE')
    ax.set_title(f'{dstitle}')
    # ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Customize violin colors
    colors = sns.color_palette("husl", len(modelname_list))
    for i, vp in enumerate(violin_parts['bodies']):
        vp.set_facecolor(colors[i])
        vp.set_alpha(0.7)

    # Set internal lines to black
    for part_name in ['cbars', 'cmins', 'cmaxes', 'cquartiles', 'cmedians', 'cmeans']:
        if part_name in violin_parts:
            violin_parts[part_name].set_color('black')
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = os.path.join(output_dir, f"{dsname}_bl_differences.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()


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

        dataset_model_data.append(df)

    combined_dataset = pd.concat(dataset_model_data, ignore_index=True)
    # create_violin_plots(combined_dataset, dsname, dstitle, output_dir)

    pcp_df = pd.read_csv(os.path.join(path_to_pcp_data, f"{dsfilename}.csv"))
    pcp_bl_df = pcp_df[['branch_length']].copy()
    pcp_bl_df = pcp_bl_df.rename(columns={'branch_length': 'opt_branch_length'})
    pcp_bl_df['model'] = "IQ-TREE"
    pcp_bl_df['pcp_index'] = pcp_df.index

    full_combined_dataset = pd.concat([combined_dataset, pcp_bl_df], ignore_index=True)
    # create_iqtree_violin_plots(full_combined_dataset, dsname, dstitle, output_dir)
    create_bl_diff_violin_plots(full_combined_dataset, dsname, dstitle, output_dir)





