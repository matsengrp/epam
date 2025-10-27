# Plot overlap and R-precision split by regions for all models on Tang et al.
# (Supplementary)

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tab_dir = "tables"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

dsname = "tang"

overlap_df = pd.read_csv(f'{tab_dir}/{dsname}_overlap.csv')
overlap_df = overlap_df[overlap_df['model'].isin(['S5F','Thrifty-SHM','Thrifty-prod','AbLang2'])]

rprec_df = pd.read_csv(f'{tab_dir}/{dsname}_rprec.csv')
rprec_df = rprec_df[rprec_df['model'].isin(['S5F','Thrifty-SHM','Thrifty-prod','AbLang2'])]

fig = plt.figure(constrained_layout=False, figsize=(15,8))
fig.patch.set_facecolor('white')

(subfig_t, subfig_b) = fig.subfigures(2, 1, hspace = 0.05)

#
# overlap
#
subfig_t.subplots_adjust(left=0.12, right=0.99, wspace=0.03)
overlap_ax = subfig_t.subplots(nrows=1, ncols=2, sharex=False, sharey=True, width_ratios=[1,8])

plot_df = overlap_df[['model','All']].set_index('model')
sns.heatmap(plot_df, cmap='Oranges',
            ax = overlap_ax[0],
            vmin = 0.7, vmax = 1,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar=False
            )
overlap_ax[0].tick_params(axis="x", labelsize=18)
overlap_ax[0].tick_params(axis="y", labelsize=18)
overlap_ax[0].axes.get_yaxis().get_label().set_visible(False)

plot_df = overlap_df[['model','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']].set_index('model')
sns.heatmap(plot_df, cmap='Oranges',
            ax = overlap_ax[1],
            vmin = 0.7, vmax = 1,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar_kws={"aspect": 14, "pad": 0.01}
            )
cbar = overlap_ax[1].collections[0].colorbar
cbar.set_label("overlap", labelpad=18)
overlap_ax[1].tick_params(axis="x", labelsize=18)
overlap_ax[1].tick_params(axis="y", left=False)
overlap_ax[1].axes.get_yaxis().get_label().set_visible(False)

cax = overlap_ax[1].figure.axes[-1]
cax.tick_params(labelsize=16)
cax.yaxis.label.set_size(20)

#
# R-precision
#
subfig_b.subplots_adjust(left=0.12, right=0.99, wspace=0.03)
rprec_ax = subfig_b.subplots(nrows=1, ncols=2, sharex=False, sharey=True, width_ratios=[1,8])

plot_df = rprec_df[['model','All']].set_index('model')
sns.heatmap(plot_df, cmap='Purples',
            ax = rprec_ax[0],
            vmin = 0.09, vmax = 0.42,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar=False
            )
rprec_ax[0].tick_params(axis="x", labelsize=18)
rprec_ax[0].tick_params(axis="y", labelsize=18)
rprec_ax[0].axes.get_yaxis().get_label().set_visible(False)

plot_df = rprec_df[['model','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']].set_index('model')
sns.heatmap(plot_df, cmap='Purples',
            ax = rprec_ax[1],
            vmin = 0.09, vmax = 0.42,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar_kws={"aspect": 14, "pad": 0.01}
            )
cbar = rprec_ax[1].collections[0].colorbar
cbar.set_label("R-precision", labelpad=18)
rprec_ax[1].tick_params(axis="x", labelsize=18)
rprec_ax[1].tick_params(axis="y", left=False)
rprec_ax[1].axes.get_yaxis().get_label().set_visible(False)

cax = rprec_ax[1].figure.axes[-1]
cax.tick_params(labelsize=16)
cax.yaxis.label.set_size(20)


plt.gcf().text(0, 0.95, "A", fontsize=30)
plt.gcf().text(0, 0.45, "B", fontsize=30)

outfname = f"{output_dir}/overlap_and_rprec"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()