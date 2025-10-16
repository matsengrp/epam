# Plot substitution accuracy for all models on Tang et al. (Figure 4A)
# Plot CSP perplexity for all models on Tang et al. (Figure 4B)

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tab_dir = "tables"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

dsname = "tang"

subacc_df = pd.read_csv(f'{tab_dir}/{dsname}_subacc.csv')
subacc_df = subacc_df[subacc_df['model']!='AbLang1']

cspperp_df = pd.read_csv(f'{tab_dir}/{dsname}_cspperp.csv')
cspperp_df = cspperp_df[cspperp_df['model']!='AbLang1']


fig = plt.figure(constrained_layout=False, figsize=(15,12))
fig.patch.set_facecolor('white')

(subfig_t, subfig_b) = fig.subfigures(2, 1, hspace = 0.05)

#
# substitution accuracy
#
subfig_t.subplots_adjust(left=0.23, right=0.99, wspace=0.03)
subacc_ax = subfig_t.subplots(nrows=1, ncols=2, sharex=False, sharey=True, width_ratios=[1,8])

plot_df = subacc_df[['model','All']].set_index('model')
sns.heatmap(plot_df, cmap='Greens',
            ax = subacc_ax[0],
            vmin = 0.14, vmax = 0.44,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar=False
            )
subacc_ax[0].tick_params(axis="x", labelsize=18)
subacc_ax[0].tick_params(axis="y", labelsize=18)
subacc_ax[0].axes.get_yaxis().get_label().set_visible(False)

plot_df = subacc_df[['model','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']].set_index('model')
sns.heatmap(plot_df, cmap='Greens',
            ax = subacc_ax[1],
            vmin = 0.14, vmax = 0.44,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar_kws={"aspect": 14, "pad": 0.01}
            )
cbar = subacc_ax[1].collections[0].colorbar
cbar.set_label("substitution accuracy", labelpad=18)
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

plot_df = cspperp_df[['model','All']].set_index('model')
sns.heatmap(plot_df, cmap='Blues_r',
            ax = cspperp_ax[0],
            vmin = 4, vmax = 14,
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
            vmin = 4, vmax = 14,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar_kws={"aspect": 14, "pad": 0.01}
            )
cbar = cspperp_ax[1].collections[0].colorbar
cbar.set_label("CSP perplexity", labelpad=18)
cspperp_ax[1].tick_params(axis="x", labelsize=18)
cspperp_ax[1].tick_params(axis="y", left=False)
cspperp_ax[1].axes.get_yaxis().get_label().set_visible(False)

cax = cspperp_ax[1].figure.axes[-1]
cax.tick_params(labelsize=16)
cax.yaxis.label.set_size(20)


plt.gcf().text(0, 0.95, "A", fontsize=40)
plt.gcf().text(0, 0.45, "B", fontsize=40)

outfname = f"{output_dir}/subacc_and_cspperp"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()