# Plot OE plots for SSPs, site substitutions, and CSPs for all models on Replay.
# Generates CSV file of model performance metrics for IgH and Igk.
# (Supplementary figures)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from epam.evaluation import perplexity_of_probs
from netam.oe_plot import (
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
    "GCReplaySHM", "GCReplaySHMDMSSigmoid", "GCReplayAbLang2", "GCReplayESM",
    "GCReplaySHMBLOSUMSigmoid", "GCReplaySHMESMSigmoid"
]

modelname_list = [
    "ReplaySHM", "ReplaySHM + DMS", "AbLang2", "ESM-1v",
    "ReplaySHM + BLOSUM62", "ReplaySHM + ESM-1v"
]

for chain in ['igh','igk']:
    print('chain:',chain)
    
    METRICS_COLUMNS = ["model","name","subs_overlap","r_precision","sub_acc","csp_perplexity","ssp_overlap","csp_overlap"]
    metrics_df = pd.DataFrame(columns=METRICS_COLUMNS)
    coldata={}
    for metric in METRICS_COLUMNS:
        coldata[metric]=[]
    
    if chain=='igh':
        title = 'GCReplay IgH'
        cdr_bounds = [(25,32), (49,56), (95,100)]
    else:
        title = f"GCReplay Ig$\kappa$"
        cdr_bounds = [(26,31), (49,51), (88,96)]
    
    dataset = f"gctrees_2025-01-10-full_{chain}_pcp_NoBackMuts"

    for model, modelname in zip(model_list, modelname_list):
        print("Model:", model)
        
        fig = plt.figure(constrained_layout=True, figsize=[16,16])
        fig.patch.set_facecolor('white')

        (subfig_t, subfig_m, subfig_b) = fig.subfigures(3,1,height_ratios=[4,8,4])
        
        #
        # Plot SSP observed vs expected
        #
        topax = subfig_t.subplots()
        
        sitemuts_df = pd.read_csv(f"{dfs_dir}/{model}_{chain}_ssp_df.csv.gz", index_col=0)
        ssp_results = plot_observed_vs_expected(sitemuts_df,None,topax,None,binning=np.linspace(-4.5, 0, 101),model_color='#0C0C0C')
        topax.text(
            0.02, 0.73,
            f'overlap: {ssp_results["overlap"]:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = topax.transAxes,
            fontsize=14
        )
        topax.legend(loc='upper left', fontsize=14)
        topax.set_ylabel("no. of substitutions", fontsize=20, labelpad=10)
        topax.set_xlabel("$\log_{10}$(site substitution probability)", fontsize=20, labelpad=10)
        
        #
        # Plot per-site observed vs expected
        #
        gs = subfig_m.add_gridspec(2)
        ax1 = subfig_m.add_subplot(gs[0,:])
        ax2 = subfig_m.add_subplot(gs[1,:], sharex=ax1)
        
        muts_obs_pred_df = pd.read_csv(f"{dfs_dir}/{model}_{chain}_site_subs_df.csv.gz", index_col=0)
        results = plot_sites_observed_vs_top_k_predictions(muts_obs_pred_df, None)
        r_prec = results['r-precision']
        
        #sitemuts_df = pd.read_csv(f"{dfs_dir}/{model}_{chain}_ssp_df.csv.gz", index_col=0)
        sitemuts_results = plot_sites_observed_vs_expected(sitemuts_df, ax1)
        ax1.text(
            0.02, 0.65,
            f'overlap: {sitemuts_results["overlap"]:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = ax1.transAxes,
            fontsize=14
        )
        ax1.text(
            0.02, 0.57,
            f'R-precision: {r_prec:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = ax1.transAxes,
            fontsize=14
        )
        ax1.axes.get_xaxis().get_label().set_visible(False)
        ax1.set_ylabel("no. of substitutions", fontsize=20, labelpad=10)
        ax1.legend(loc='upper left', fontsize=14)
        
        #
        # Plot per-site substitution accuracy
        #
        subacc_df = pd.read_csv(f"{dfs_dir}/{model}_{chain}_site_subacc_df.csv.gz", index_col=0)
        subacc_results = plot_sites_subs_acc(subacc_df, None, ax2)
        subaccs = subacc_results['site_subacc']
        
        numbering = np.arange(np.min(subacc_df["site"]), np.max(subacc_df["site"]) + 1)
        
        csp_df = pd.read_csv(f"{dfs_dir}/{model}_{chain}_csp_df.csv.gz", index_col=0)
        csp_perplexity = perplexity_of_probs(csp_df[csp_df['mutation']==True]['prob'].to_numpy())

        subaccs = np.clip(subaccs, a_min=0, a_max=None)

        ax2.text(
            0.02, 0.93,
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
        ax2.set_ylim(ymax=1.28)
        
        for bounds in cdr_bounds:
            xlower = bounds[0]
            xupper = bounds[1]
            ax2.add_patch(
                Rectangle(
                    (xlower - 0.5, 0),
                    xupper - xlower + 1,
                    ax2.get_ylim()[1],
                    color="#E69F00",
                    alpha=0.2,
                )
            )

        ax2.set_xlabel("amino acid position", fontsize=20, labelpad=10)
        ax2.set_ylabel(f"sub. acc.", fontsize=20, labelpad=10)
        ax2.axhline(y=1/19, color='black', linestyle='--', linewidth=2)

        #
        # Plot CSP observed vs expected
        #
        bottomax = subfig_b.subplots()
        
        csp_df = csp_df[csp_df['prob']!=0]
        csp_results = plot_observed_vs_expected(csp_df,None,bottomax,None,binning=np.linspace(-4.5, 0, 101),model_color='#0C0C0C')
        bottomax.text(
            0.02, 0.73,
            f'overlap: {csp_results["overlap"]:.3g}',
            verticalalignment ='top', 
            horizontalalignment ='left', 
            transform = bottomax.transAxes,
            fontsize=14
        )
        bottomax.legend(loc='upper left', fontsize=14)
        bottomax.set_ylabel("no. of substitutions", fontsize=20, labelpad=10)
        bottomax.set_xlabel("$\log_{10}$(conditional substitution probability)", fontsize=20, labelpad=10)
        
        fig.suptitle(f'{title}, {modelname}', fontsize=20, x=0.12, ha='left')
        
        outfname = f"{output_dir}/gcreplay_{chain}_{model}_oe"
        print(f"{outfname}.png",'created!')
        plt.savefig(f"{outfname}.png")
        print(f"{outfname}.pdf",'created!')
        plt.savefig(f"{outfname}.pdf")
        plt.close()
        
        coldata["model"].append(model)
        coldata["name"].append(modelname)
        coldata["subs_overlap"].append(sitemuts_results["overlap"])
        coldata["r_precision"].append(r_prec)
        coldata["sub_acc"].append(subacc_results["total_subacc"])
        coldata["csp_perplexity"].append(csp_perplexity)
        coldata["ssp_overlap"].append(ssp_results["overlap"])
        coldata["csp_overlap"].append(csp_results["overlap"])
        
    for metric in METRICS_COLUMNS:
        metrics_df[metric] = coldata[metric]    
    metrics_df.to_csv(f"gcreplay_{chain}_metrics.csv")