# Plot comparisons of Overlap and R-precision for all models on Replay. (Figure 5A and 6A)
# Plot substitution accuracy for all models on Replay (Figure 5B and 6B)
# Plot CSP perplexity for all models on Replay (Figure 5C and 6C)

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tab_dir = "tables"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


models = [
    'GCReplaySHM', 'GCReplaySHMBLOSUMSigmoid', 'GCReplaySHMDMSSigmoid', 'GCReplaySHMESMSigmoid',
    'GCReplayESM', 'GCReplayAbLang2'
]

subacc_vlimits={}
subacc_vlimits['igh'] = (0.15, 0.53)
subacc_vlimits['igk'] = (0.095, 0.75)

cspperp_vlimits={}
cspperp_vlimits['igh'] = (4, 12)
cspperp_vlimits['igk'] = (2, 10)

chain_label={}
chain_label['igh'] = 'IgH'
chain_label['igk'] = r'Ig$\kappa$'


for chain in ['igh','igk']:
    print("chain:",chain)
    
    fig = plt.figure(constrained_layout=False, figsize=(15,15))
    fig.patch.set_facecolor('white')

    (subfig_t, subfig_m, subfig_b) = fig.subfigures(3, 1, hspace = 0.1)
    
    #
    # Mutability metrics
    #
    subfig_t.subplots_adjust(left=0.23, right=0.98, wspace=0.1)
    metrics_ax = subfig_t.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
    
    metrics_df = pd.read_csv(f"{tab_dir}/gcreplay_{chain}_metrics.csv", index_col=0)
    indices_list = []
    for model in models:
        indices_list.append(metrics_df.index[metrics_df['model']==model].tolist())
    indices = np.array(indices_list).flatten()
    metrics_df = metrics_df.loc[indices]

    modelnames = metrics_df["name"].to_numpy()
    overlaps = metrics_df["subs_overlap"].to_numpy()
    r_precs = metrics_df["r_precision"].to_numpy()
    yvals = np.arange(len(modelnames))
    
    height=0.8

    metrics_ax[0].barh(yvals, overlaps, height=height, color='#AAAAAA', edgecolor='black')
    metrics_ax[0].tick_params(axis="x", labelsize=18)
    metrics_ax[0].set_xlabel(f"Overlap ({chain_label[chain]})", fontsize=22, labelpad=5)
    metrics_ax[0].grid(axis='x')
    metrics_ax[0].set_yticks(ticks=yvals, labels=modelnames, fontsize=18)
    metrics_ax[0].invert_yaxis()
    metrics_ax[0].set_xlim(0, 0.85)

    metrics_ax[1].barh(yvals, r_precs, height=height, color='#AAAAAA', edgecolor='black')
    metrics_ax[1].tick_params(axis="x", labelsize=18)
    metrics_ax[1].set_xlabel(f"R-precision ({chain_label[chain]})", fontsize=24, labelpad=5)
    metrics_ax[1].grid(axis='x')
    metrics_ax[1].set_xlim(0, 0.154)
    
    
    #
    # Substitution accuracy
    #
    subfig_m.subplots_adjust(left=0.23, right=0.99, wspace=0.03)
    subacc_ax = subfig_m.subplots(nrows=1, ncols=2, sharex=False, sharey=True, width_ratios=[1,8])
    
    subacc_df = pd.read_csv(f'{tab_dir}/gcreplay_{chain}_subacc.csv')
    plot_df = subacc_df[['model','All']].set_index('model')
    sns.heatmap(plot_df, cmap='Greens',
                ax = subacc_ax[0],
                vmin = subacc_vlimits[chain][0], vmax = subacc_vlimits[chain][1],
                annot = True, annot_kws={"fontsize":14}, fmt=".3g",
                square=False, linewidths=0.1,
                cbar=False
                )
    subacc_ax[0].tick_params(axis="x", labelsize=18)
    subacc_ax[0].tick_params(axis="y", labelsize=18)
    subacc_ax[0].axes.get_yaxis().get_label().set_visible(False)
    
    plot_df = subacc_df[['model','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']].set_index('model')
    sns.heatmap(plot_df, cmap='Greens',
                ax = subacc_ax[1],
                vmin = subacc_vlimits[chain][0], vmax = subacc_vlimits[chain][1],
                annot = True, annot_kws={"fontsize":14}, fmt=".3g",
                square=False, linewidths=0.1,
                cbar_kws={"aspect": 10, "pad": 0.01}
                )
    cbar = subacc_ax[1].collections[0].colorbar
    cbar.set_label(f"substitution accuracy ({chain_label[chain]})", labelpad=18)
    subacc_ax[1].tick_params(axis="x", labelsize=18)
    subacc_ax[1].tick_params(axis="y", left=False)
    subacc_ax[1].axes.get_yaxis().get_label().set_visible(False)

    cax = subacc_ax[1].figure.axes[-1]
    cax.tick_params(labelsize=16)
    cax.yaxis.label.set_size(20)
    
    
    #
    # CSP perplexity
    #
    subfig_b.subplots_adjust(left=0.23, right=0.99, wspace=0.03)
    cspperp_ax = subfig_b.subplots(nrows=1, ncols=2, sharex=False, sharey=True, width_ratios=[1,8])
    
    cspperp_df = pd.read_csv(f'{tab_dir}/gcreplay_{chain}_cspperp.csv')
    plot_df = cspperp_df[['model','All']].set_index('model')
    sns.heatmap(plot_df, cmap='Blues_r',
                ax = cspperp_ax[0],
                vmin = cspperp_vlimits[chain][0], vmax = cspperp_vlimits[chain][1],
                annot = True, annot_kws={"fontsize":14}, fmt='.3g',
                square=False, linewidths=0.1,
                cbar=False
                )
    cspperp_ax[0].tick_params(axis="x", labelsize=18)
    cspperp_ax[0].tick_params(axis="y", labelsize=18)
    cspperp_ax[0].axes.get_yaxis().get_label().set_visible(False)
    
    plot_df = cspperp_df[['model','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']].set_index('model')
    sns.heatmap(plot_df, cmap='Blues_r',
                ax = cspperp_ax[1],
                vmin = cspperp_vlimits[chain][0], vmax = cspperp_vlimits[chain][1],
                annot = True, annot_kws={"fontsize":14}, fmt='.3g',
                square=False, linewidths=0.1,
                cbar_kws={"aspect": 10, "pad": 0.01}
                )
    cbar = cspperp_ax[1].collections[0].colorbar
    cbar.set_label(f"CSP perplexity ({chain_label[chain]})", labelpad=18)
    cspperp_ax[1].tick_params(axis="x", labelsize=18)
    cspperp_ax[1].tick_params(axis="y", left=False)
    cspperp_ax[1].axes.get_yaxis().get_label().set_visible(False)

    cax = cspperp_ax[1].figure.axes[-1]
    cax.tick_params(labelsize=16)
    cax.yaxis.label.set_size(20)
    
    
    plt.gcf().text(0, 0.95, "A", fontsize=40)
    plt.gcf().text(0, 0.62, "B", fontsize=40)
    plt.gcf().text(0, 0.28, "C", fontsize=40)

    outfname = f"{output_dir}/gcreplay_{chain}_mutability_subacc_cspperp"
    print(f"{outfname}.png",'created!')
    plt.savefig(f"{outfname}.png")
    print(f"{outfname}.pdf",'created!')
    plt.savefig(f"{outfname}.pdf")
    plt.close()