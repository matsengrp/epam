import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


output_dir = "plots/nsubs"
os.makedirs(output_dir, exist_ok=True)

dsname = "tang"

for pcptype in ['','anc','leaf']:
    
    if len(pcptype)>0:
        df = pd.read_csv(f'{dsname}_{pcptype}_nsubs_subacc.csv')
    else:
        df = pd.read_csv(f'{dsname}_nsubs_subacc.csv')
    df = df.sort_values('nsubs',ascending=False)
    df = df[df['nsubs']<6]

    fig = plt.figure(figsize=[15,6])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(ncols=2, width_ratios=[1,8])
    axs = gs.subplots(sharex=False, sharey=True)

    plot_df = df[['nsubs','All']].set_index('nsubs')
    sns.heatmap(plot_df, cmap='Greens',
                ax = axs[0],
                vmin = 0.14, vmax = 0.44,
                annot = True, annot_kws={"fontsize":14}, fmt='.3g',
                square=False, linewidths=0.1,
                cbar=False
                )
    axs[0].tick_params(axis="x", labelsize=18)
    axs[0].tick_params(axis="y", labelsize=18, rotation=0)
    axs[0].set_ylabel("number of substitutions", fontsize=18, labelpad=10)
    #axs[0].axes.get_yaxis().get_label().set_visible(False)

    plot_df = df[['nsubs','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']].set_index('nsubs')
    sns.heatmap(plot_df, cmap='Greens',
                ax = axs[1],
                vmin = 0.14, vmax = 0.44,
                annot = True, annot_kws={"fontsize":14}, fmt='.3g',
                square=False, linewidths=0.1,
                cbar_kws={"aspect": 14, "pad": 0.01}
                )
    cbar = axs[1].collections[0].colorbar
    cbar.set_label("substitution accuracy", labelpad=18)
    axs[1].tick_params(axis="x", labelsize=18)
    axs[1].tick_params(axis="y", left=False)
    axs[1].axes.get_yaxis().get_label().set_visible(False)

    cax = axs[1].figure.axes[-1]
    cax.tick_params(labelsize=16)
    cax.yaxis.label.set_size(20)

    fig.suptitle(f'Tang et al., Thrifty-prod', fontsize=20, x=0.12, ha='left')
    plt.tight_layout()

    if len(pcptype)>0:
        outfname = f"{output_dir}/{dsname}_subacc_{pcptype}_nsubs"
    else:
        outfname = f"{output_dir}/{dsname}_subacc_nsubs"
    print(f"{outfname}.png",'created!')
    plt.savefig(f"{outfname}.png")
    print(f"{outfname}.pdf",'created!')
    plt.savefig(f"{outfname}.pdf")
    plt.close()