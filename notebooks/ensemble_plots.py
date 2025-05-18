import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
from epam.utils import load_and_filter_pcp_df
from epam.oe_plot import (
    get_numbering_dict,
    get_site_mutabilities_df, 
    plot_sites_observed_vs_expected,
    get_subs_and_preds_from_aaprob,
    get_site_substitutions_df, 
    plot_sites_observed_vs_top_k_predictions,
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

local_dir = "/home/mjohnso4/epam"
epam_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2"
pcp_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2"
race_filename = "rodriguez-airr-seq-race-prod_pcp_2024-07-28_MASKED_NI_noN_no-naive"
ensemble_output_dir = f"{local_dir}/output/{race_filename}"
anarci_race = f"{pcp_dir}/anarci/rodriguez-airr-seq-race-prod_imgt.csv"
plot_output_dir = f"{local_dir}/output/plots/"
pcp_path = f"{local_dir}/pcp_inputs/{race_filename}.csv"
metrics_file_path = f"{ensemble_output_dir}/ensemble_combined_performance.csv"
oe_results_path = f"{ensemble_output_dir}/oe_metrics.csv"


def collect_esm_ensemble_results(models=["ESM1v_mask", "ThriftyESM_mask", "S5FESM_mask"]):
    
    pcp_df = load_and_filter_pcp_df(pcp_path)
    numbering, excluded = get_numbering_dict(anarci_race, pcp_df, True, "imgt")

    results_df = pd.DataFrame(columns=[
        'model', 'ensemble_member', 'overlap', 'residual'
    ])
    
    for model in models:
        aaprob_esm1 = f"{ensemble_output_dir}/esm1/{model}/combined_aaprob.hdf5"
        aaprob_esm2 = f"{ensemble_output_dir}/esm2/{model}/combined_aaprob.hdf5"
        aaprob_esm3 = f"{ensemble_output_dir}/esm3/{model}/combined_aaprob.hdf5"
        aaprob_esm4 = f"{ensemble_output_dir}/esm4/{model}/combined_aaprob.hdf5"
        aaprob_esm5 = f"{ensemble_output_dir}/esm5/{model}/combined_aaprob.hdf5"
        aaprob_ensemble = f"{ensemble_output_dir}/ensemble_set/{model}/combined_aaprob.hdf5"
        
        site_sub_probs_df_1 = get_site_mutabilities_df(aaprob_esm1, numbering)
        results_1 = get_overlap_and_residual(site_sub_probs_df_1, numbering)
        
        site_sub_probs_df_2 = get_site_mutabilities_df(aaprob_esm2, numbering)
        results_2 = get_overlap_and_residual(site_sub_probs_df_2, numbering)
        
        site_sub_probs_df_3 = get_site_mutabilities_df(aaprob_esm3, numbering)
        results_3 = get_overlap_and_residual(site_sub_probs_df_3, numbering)
        
        site_sub_probs_df_4 = get_site_mutabilities_df(aaprob_esm4, numbering)
        results_4 = get_overlap_and_residual(site_sub_probs_df_4, numbering)
        
        site_sub_probs_df_5 = get_site_mutabilities_df(aaprob_esm5, numbering)
        results_5 = get_overlap_and_residual(site_sub_probs_df_5, numbering)
        
        site_sub_probs_df_all = get_site_mutabilities_df(aaprob_ensemble, numbering)
        results_all = get_overlap_and_residual(site_sub_probs_df_all, numbering)
        
        new_rows = [
            {'model': model, 'ensemble_member': '1', 'overlap': results_1['overlap'], 'residual': results_1['residual']},
            {'model': model, 'ensemble_member': '2', 'overlap': results_2['overlap'], 'residual': results_2['residual']},
            {'model': model, 'ensemble_member': '3', 'overlap': results_3['overlap'], 'residual': results_3['residual']},
            {'model': model, 'ensemble_member': '4', 'overlap': results_4['overlap'], 'residual': results_4['residual']},
            {'model': model, 'ensemble_member': '5', 'overlap': results_5['overlap'], 'residual': results_5['residual']},
            {'model': model, 'ensemble_member': 'Ensemble', 'overlap': results_all['overlap'], 'residual': results_all['residual']}
        ]
        
        results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)

    results_df.to_csv(oe_results_path, index=False)
    print(f"Results saved to {oe_results_path}")
    
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
    oe_results_df = pd.read_csv(oe_results_path)
    
    # Simplify model names by removing "_mask" suffix
    results_df['simplified_model'] = results_df['model'].str.replace('_mask', '')
    oe_results_df['simplified_model'] = oe_results_df['model'].str.replace('_mask', '')
    
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
    plt.savefig(f"{plot_output_dir}/ensemble_performance.png")
    plt.savefig(f"{plot_output_dir}/ensemble_performance.pdf")
    plt.show()

plot_ensemble_performance()
