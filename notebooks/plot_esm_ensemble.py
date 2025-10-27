# Comparing performance of individiual vs ensembled ESM-1v models as standalone and selection factors in Rodriguez et al.
# Supplemental Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import pickle
from epam.utils import load_and_filter_pcp_df
from epam.oe_plot import (
    get_numbering_dict,
    plot_sites_observed_vs_expected,
)
from epam.df_for_plots import (
    get_site_mutabilities_df,
)

# Okabe-Ito colors
oi_black         = '#000000'
oi_orange        = '#E69F00'
oi_skyblue       = '#56B4E9'
oi_bluishgreen   = '#009E73'
oi_yellow        = '#F0E442'
oi_blue          = '#0072B2'
oi_vermillion    = '#D55E00'
oi_reddishpurple = '#CC79A7'


numbering_path = "dataframes/rodriguez_numbering.pkl"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
metrics_file_path = "tables/ensemble_combined_performance.csv"

models=["ESM1v_mask", "ThriftyESM_mask", "S5FESM_mask"]
esm_numbers = ['esm1', 'esm2', 'esm3', 'esm4', 'esm5', 'ensemble_set']

def collect_esm_ensemble_results():
    
    with open(numbering_path, 'rb') as f:
        numbering = pickle.load(f)

    results_df = pd.DataFrame(columns=[
        'model', 'ensemble_member', 'overlap', 'residual'
    ])
    
    for esm_number in esm_numbers:
        for model in models:
            model_esm_ssp = f"dataframes/rodriguez_{model}_{esm_number}_ssp_df.csv.gz"
            sitemuts_df = pd.read_csv(model_esm_ssp, index_col=0, dtype={'site':'object'})
            model_esm_results = get_overlap_and_residual(sitemuts_df, numbering)
            
            new_row = [
                {'model': model, 'ensemble_member': esm_number, 'overlap': model_esm_results['overlap'], 'residual': model_esm_results['residual']},
            ]
            
            results_df = pd.concat([results_df, pd.DataFrame(new_row)], ignore_index=True)
    
    return results_df


def get_overlap_and_residual(site_sub_probs_df, numbering):
    results = plot_sites_observed_vs_expected(site_sub_probs_df, None, numbering)
    return {
        'overlap': results['overlap'],
        'residual': results['residual']
    }


def plot_ensemble_performance():
    # Load data
    results_df = pd.read_csv(metrics_file_path)
    oe_results_df = collect_esm_ensemble_results()
    
    # Simplify model names by removing "_mask" suffix
    results_df['simplified_model'] = results_df['model'].str.replace('_mask', '')
    oe_results_df['simplified_model'] = oe_results_df['model'].str.replace('_mask', '')

    # Rename ensemble_set
    oe_results_df['ensemble_member'] = oe_results_df['ensemble_member'].replace(
        {
            'ensemble_set': 'Ensemble',
            'esm1': '1',
            'esm2': '2',
            'esm3': '3',
            'esm4': '4',
            'esm5': '5'
        }
    )
    
    # Color mapping
    version_colors = {
        'Ensemble': oi_black, 
        '1': oi_reddishpurple,#oi_orange, 
        '2': oi_blue, 
        '3': oi_bluishgreen, 
        '4': oi_yellow, 
        '5': oi_vermillion
    }
    
    fig = plt.figure(figsize=(10, 3.25))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 3)
    axs = gs.subplots(sharex=False, sharey=True)

    point_size = 90
    
    axs[0].scatter(
        oe_results_df['overlap'], 
        oe_results_df['simplified_model'],  
        c=oe_results_df['ensemble_member'].map(version_colors), 
        alpha=.75, 
        marker='d',
        s=point_size
    )
    axs[0].set_title("Overlap")
    
    axs[1].scatter(
        results_df['r_precision'], 
        results_df['simplified_model'],  
        c=results_df['ensemble_member'].map(version_colors), 
        alpha=.75, 
        marker='d',
        s=point_size
    )
    axs[1].set_title("R-Precision")
    
    axs[2].scatter(
        results_df['sub_accuracy'], 
        results_df['simplified_model'],  
        c=results_df['ensemble_member'].map(version_colors), 
        alpha=.75, 
        marker='d',
        s=point_size
    )
    axs[2].set_title("Substitution accuracy")

    for ax in axs:
        ax.set_ylim(-0.4, 3 - 0.6)
        
        ax.set_yticks(range(3))
        ax.set_yticklabels(results_df['simplified_model'].unique())
    
    legend_patches = [
        mpatches.Patch(color=color, label=label) 
        for label, color in version_colors.items()
    ]
    
    fig.legend(
        handles=legend_patches, 
        title='ESM version', 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0), 
        ncol=6 
    )

    plt.subplots_adjust(
        left=0.1,      
        right=0.95,    
        top=0.9,       
        bottom=0.25,   
        wspace=0.1     
    )

    # Save and show
    plt.savefig(f"{output_dir}/ensemble_performance.png")
    plt.savefig(f"{output_dir}/ensemble_performance.pdf")
    plt.show()

plot_ensemble_performance()
