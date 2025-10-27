# Plot observed vs expected number of substitutions over IMGT positions for a few models on Tang et al. (Figure 2A)
# Plot comparisons of Overlap and R-precision for all models on Tang et al. (Figure 2B)

import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from epam.oe_plot import (
    plot_sites_observed_vs_expected,
    plot_sites_observed_vs_top_k_predictions,
)

dfs_dir = "dataframes"
tab_dir = "tables"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

oe_model_list = [
    "ThriftyHumV0.2-59", "ThriftyESM_mask", "AbLang2_mask"
]

oe_modelname_list = [
    "Thrifty-SHM", "Thrifty-SHM + ESM-1v", "AbLang2"
]

metrics_models = [
    "S5F", "S5FBLOSUM", "S5FESM_mask",
    "ThriftyHumV0.2-59", "ThriftyBLOSUM", "ThriftyESM_mask",
    "ThriftyProdHumV0.2-59", "ESM1v_mask", "AbLang2_mask"
]

dsname  = "tang"
dstitle = "Tang et al."


with open(f'{dfs_dir}/{dsname}_numbering.pkl', 'rb') as f:
    numbering = pickle.load(f)

site_sub_probs_df = {}
r_prec = {}

for model in oe_model_list:
    print("Model:", model)
    
    # load dataframe of SSPs
    site_sub_probs_df[model] = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_ssp_df.csv.gz', index_col=0, dtype={'site':'object'})
    
    # load dataframe of observed and predicted (i.e. top-k) substitutions, to compute R-precision
    muts_obs_pred_df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_site_subs_df.csv.gz', index_col=0, dtype={'site':'object'})
    results = plot_sites_observed_vs_top_k_predictions(muts_obs_pred_df, None, numbering)
    r_prec[model] = results['r-precision']


metrics_df = pd.read_csv(f"{tab_dir}/{dsname}_metrics.csv", index_col=0)
indices_list = []
for model in metrics_models:
    indices_list.append(metrics_df.index[metrics_df['model']==model].tolist())
indices = np.array(indices_list).flatten()
metrics_df = metrics_df.loc[indices]

modelnames = metrics_df["name"].to_numpy()
overlaps = metrics_df["subs_overlap"].to_numpy()
r_precs = metrics_df["r_precision"].to_numpy()
yvals = np.arange(len(modelnames))


fig = plt.figure(constrained_layout=False, figsize=(15,20))
fig.patch.set_facecolor('white')

(subfig_t, subfig_b) = fig.subfigures(2, 1, height_ratios=[3, 2], hspace = 0.05)

#
# Per site OE plots
# 
subfig_t.subplots_adjust(left=0.12,right=0.99)
oe_ax = subfig_t.subplots(nrows=len(oe_model_list), ncols=1, sharex=True, sharey=False)

for i in range(len(oe_model_list)):
    model = oe_model_list[i]
    modelname = oe_modelname_list[i]
    print("Model:", modelname)

    results = plot_sites_observed_vs_expected(site_sub_probs_df[model], oe_ax[i], numbering)
    oe_ax[i].text(0.02, 0.9, modelname, verticalalignment ='top', horizontalalignment ='left', transform = oe_ax[i].transAxes, fontsize=15)
    oe_ax[i].text(
        0.02, 0.8,
        f'overlap: {results["overlap"]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = oe_ax[i].transAxes,
        fontsize=15
    )
    oe_ax[i].text(
        0.02, 0.7,
        f'R-precision: {r_prec[model]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = oe_ax[i].transAxes,
        fontsize=15
    )
    oe_ax[i].axes.get_xaxis().get_label().set_visible(False)
    oe_ax[i].axes.get_yaxis().get_label().set_visible(False)
    oe_ax[i].legend().set_visible(False)
    plt.setp(oe_ax[i].get_xticklabels()[1::3], visible=False)
    plt.setp(oe_ax[i].get_xticklabels()[1::2], visible=False)
    oe_ax[i].tick_params(axis="x", labelsize=12, labelrotation=90)

oe_ax[0].legend().set_visible(True)
oe_ax[0].legend(ncol=3, fontsize=15, bbox_to_anchor=(0.98, 1.2))

oe_ax[-1].axes.get_xaxis().get_label().set_visible(True)
oe_ax[-1].set_xlabel("IMGT position", fontsize=20, labelpad=10)

subfig_t.suptitle(dstitle, fontsize=20, x=0.14, y=0.92, ha='left')
subfig_t.supylabel('number of substitutions', fontsize=20)

#
# Mutability metrics
#
subfig_b.subplots_adjust(left=0.23, right=0.99, wspace=0.1)
metrics_ax = subfig_b.subplots(nrows=1, ncols=2, sharex=False, sharey=True)

height=0.8

metrics_ax[0].barh(yvals, overlaps, height=height, color='#AAAAAA', edgecolor='black')
metrics_ax[0].tick_params(axis="x", labelsize=18)
metrics_ax[0].set_xlabel("Overlap", fontsize=22, labelpad=5)
metrics_ax[0].grid(axis='x')
metrics_ax[0].set_yticks(ticks=yvals, labels=modelnames, fontsize=18)
metrics_ax[0].invert_yaxis()

metrics_ax[1].barh(yvals, r_precs, height=height, color='#AAAAAA', edgecolor='black')
metrics_ax[1].tick_params(axis="x", labelsize=18)
metrics_ax[1].set_xlabel("R-precision", fontsize=24, labelpad=5)
metrics_ax[1].grid(axis='x')

plt.gcf().text(0, 0.95, "A", fontsize=40)
plt.gcf().text(0, 0.37, "B", fontsize=40)


outfname = f"{output_dir}/oe_and_mutability"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()