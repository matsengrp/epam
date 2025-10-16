# Plot OE plots for SSPs, site substitutions, and CSPs for all models and datasets.
# Generates CSV file of model performance metrics for each dataset.
# (Supplementary figures)

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from epam.evaluation import perplexity_of_probs
from epam.oe_plot import (
    plot_sites_observed_vs_top_k_predictions,
    plot_observed_vs_expected,
    plot_sites_observed_vs_expected,
    plot_sites_subs_acc,
)
from matplotlib.patches import Rectangle


dfs_dir = "dataframes"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

model_list = [
    "S5F", "ThriftyHumV0.2-59",
    "S5FBLOSUM", "ThriftyBLOSUM",
    "S5FESM_mask", "ThriftyESM_mask",
    "ThriftyProdHumV0.2-59", "ESM1v_mask",
    "AbLang1", "AbLang2_mask",
]

modelname_list = [
    "S5F", "Thrifty-SHM",
    "S5F + BLOSUM62", "Thrifty-SHM + BLOSUM62",
    "S5F + ESM-1v", "Thrifty-SHM + ESM-1v",
    "Thrifty-prod", "ESM-1v",
    "AbLang1", "AbLang2",
]

dsname = 'tang'
dstitle = 'Tang et al.'


with open(f'{dfs_dir}/{dsname}_numbering.pkl', 'rb') as f:
    numbering = pickle.load(f)

iplot=0
for iplot in [1,2]:
    if iplot==1:
        figheight = 48
        nrows = 3
        fig_model_list = model_list[:6]
        fig_modelname_list = modelname_list[:6]
    else:
        figheight = 32
        nrows = 2
        fig_model_list = model_list[-4:]
        fig_modelname_list = modelname_list[-4:]

    fig = plt.figure(constrained_layout=True, figsize=[32, figheight])
    fig.patch.set_facecolor('white')
    subfigs = fig.subfigures(nrows,2,wspace=0.1,hspace=0.1)

    isubfig=0
    for model, modelname in zip(fig_model_list, fig_modelname_list):
        print("Model:", model)
        
        irow = isubfig // 2
        icol = isubfig % 2

        (subfig_t, subfig_m, subfig_b) = subfigs[irow, icol].subfigures(3,1,height_ratios=[4,8,4])
        
        #
        # Plot SSP observed vs expected
        #
        topax = subfig_t.subplots()
        
        sitemuts_df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_ssp_df.csv.gz', index_col=0, dtype={'site':'object'})
        ssp_results = plot_observed_vs_expected(sitemuts_df,None,topax,None,binning=np.linspace(-4.5, 0, 101),model_color='#0C0C0C')
        topax.text(
            0.02, 0.73,
            f'overlap: {ssp_results["overlap"]:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = topax.transAxes,
            fontsize=14
        )
        topax.legend(loc='upper left', fontsize=12)
        topax.set_ylabel("no. of substitutions", fontsize=20, labelpad=10)
        topax.set_xlabel("$\log_{10}$(site substitution probability)", fontsize=20, labelpad=10)
        
        #
        # Plot per-site observed vs expected
        #
        gs = subfig_m.add_gridspec(2)
        ax1 = subfig_m.add_subplot(gs[0,:])
        ax2 = subfig_m.add_subplot(gs[1,:], sharex=ax1)
        
        muts_obs_pred_df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_site_subs_df.csv.gz', index_col=0, dtype={'site':'object'})
        results = plot_sites_observed_vs_top_k_predictions(muts_obs_pred_df, None, numbering)
        r_prec = results['r-precision']
        
        sitemuts_results = plot_sites_observed_vs_expected(sitemuts_df, ax1, numbering)
        ax1.text(
            0.02, 0.60,
            f'overlap: {sitemuts_results["overlap"]:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = ax1.transAxes,
            fontsize=14
        )
        ax1.text(
            0.02, 0.52,
            f'R-precision: {r_prec:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = ax1.transAxes,
            fontsize=14
        )
        ax1.axes.get_xaxis().get_label().set_visible(False)
        ax1.set_ylabel("no. of substitutions", fontsize=20, labelpad=10)
        ax1.legend(loc='upper left', fontsize=12)
        plt.setp(ax1.get_xticklabels()[1::3], visible=False)
        plt.setp(ax1.get_xticklabels()[1::2], visible=False)
        ax1.tick_params(axis="x", labelsize=12, labelrotation=90)
        
        #
        # Plot per-site substitution accuracy
        #
        subacc_df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_site_subacc_df.csv.gz', index_col=0, dtype={'site':'object'})
        subacc_results = plot_sites_subs_acc(subacc_df, None, ax2, numbering)
        subaccs = subacc_results['site_subacc']
        
        csp_df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_csp_df.csv.gz', index_col=0, dtype={'site':'object'})
        csp_perplexity = perplexity_of_probs(csp_df[csp_df['mutation']==True]['prob'].to_numpy())

        subaccs = np.clip(subaccs, a_min=0, a_max=None)

        ax2.text(
            0.02, 0.94,
            f'sub. acc.: {subacc_results["total_subacc"]:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = ax2.transAxes,
            fontsize=14
        )
        
        ax2.text(
            0.02, 0.85,
            f'CSP perp.: {csp_perplexity:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = ax2.transAxes,
            fontsize=14
        )
        
        ax2.set_ylim(ymax=1.15)
        
        for cdr_bounds in [("27", "38"), ("56", "65"), ("105", "117")]:
            xlower = numbering[("reference", 0)].index(cdr_bounds[0])
            xupper = numbering[("reference", 0)].index(cdr_bounds[1])
            ax2.add_patch(
                Rectangle(
                    (xlower - 0.5, 0),
                    xupper - xlower + 1,
                    ax2.get_ylim()[1],
                    color="#E69F00",
                    alpha=0.2,
                )
            )

        ax2.set_xlabel("IMGT position", fontsize=20, labelpad=10)
        ax2.set_ylabel(f"sub. acc.", fontsize=20, labelpad=10)
        ax2.axhline(y=1/19, color='black', linestyle='--', linewidth=2)
        plt.setp(ax2.get_xticklabels()[1::3], visible=False)
        plt.setp(ax2.get_xticklabels()[1::2], visible=False)
        ax2.tick_params(axis="x", labelsize=12, labelrotation=90)

        #
        # Plot CSP observed vs expected
        #
        bottomax = subfig_b.subplots()
        
        csp_df = csp_df[csp_df['prob']!=0]
        csp_results = plot_observed_vs_expected(csp_df,None,bottomax,None,binning=np.linspace(-4.5, 0, 101),model_color='#0C0C0C')
        bottomax.text(
            0.02, 0.74,
            f'overlap: {csp_results["overlap"]:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = bottomax.transAxes,
            fontsize=14
        )
        bottomax.legend(loc='upper left', fontsize=12)
        bottomax.set_ylabel("no. of substitutions", fontsize=20, labelpad=10)
        bottomax.set_xlabel("$\log_{10}$(conditional substitution probability)", fontsize=20, labelpad=10)
        
        subfigs[irow, icol].suptitle(f'{dstitle}, {modelname}', fontsize=20, x=0.12, ha='left')
        
        isubfig += 1


    outfname = f"{output_dir}/{dsname}_oe_supp{iplot}"
    print(f"{outfname}.png",'created!')
    plt.savefig(f"{outfname}.png")
    print(f"{outfname}.pdf",'created!')
    plt.savefig(f"{outfname}.pdf")
    plt.close()
