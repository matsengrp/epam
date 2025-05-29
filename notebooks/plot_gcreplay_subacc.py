# Plot substitution accuracy for all models on Replay
# (Figure 5B and Supplementary)

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

vlimits={}
vlimits['igh'] = (0.15, 0.53)
vlimits['igk'] = (0.095, 0.75)

for chain in ['igh','igk']:
    print("chain:",chain)

    df = pd.read_csv(f'gcreplay_{chain}_subacc.csv')

    fig = plt.figure(figsize=[15,4])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(ncols=2, width_ratios=[1,8])
    axs = gs.subplots(sharex=False, sharey=True)

    plot_df = df[['model','All']].set_index('model')
    sns.heatmap(plot_df, cmap='Greens',
                ax = axs[0],
                vmin = vlimits[chain][0], vmax = vlimits[chain][1],
                annot = True, annot_kws={"fontsize":14}, fmt=".3g",
                square=False, linewidths=0.1,
                cbar=False
                )
    axs[0].tick_params(axis="x", labelsize=18)
    axs[0].tick_params(axis="y", labelsize=18)
    axs[0].axes.get_yaxis().get_label().set_visible(False)
    
    plot_df = df[['model','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']].set_index('model')
    sns.heatmap(plot_df, cmap='Greens',
                ax = axs[1],
                vmin = vlimits[chain][0], vmax = vlimits[chain][1],
                annot = True, annot_kws={"fontsize":14}, fmt=".3g",
                square=False, linewidths=0.1,
                cbar_kws={"aspect": 10, "pad": 0.01}
                )
    cbar = axs[1].collections[0].colorbar
    cbar.set_label("substitution accuracy", labelpad=18)
    axs[1].tick_params(axis="x", labelsize=18)
    axs[1].tick_params(axis="y", left=False)
    axs[1].axes.get_yaxis().get_label().set_visible(False)

    cax = axs[1].figure.axes[-1]
    cax.tick_params(labelsize=16)
    cax.yaxis.label.set_size(20)

    plt.gcf().text(0, 0.89, "B", fontsize=40)
    plt.tight_layout()

    outfname = f"{output_dir}/gcreplay_{chain}_subacc"
    print(f"{outfname}.png",'created!')
    plt.savefig(f"{outfname}.png")
    print(f"{outfname}.pdf",'created!')
    plt.savefig(f"{outfname}.pdf")
    plt.close()