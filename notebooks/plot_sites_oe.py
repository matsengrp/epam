# Plot observed vs expected number of substitutions
# over IMGT positions for a few models on Tang et al.
# (Figure 2A)

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from epam.oe_plot import (
    plot_sites_observed_vs_expected,
    plot_sites_observed_vs_top_k_predictions,
)

dfs_dir = "dataframes"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


model_list = [
    "ThriftyHumV0.2-59", "ThriftyESM_mask", "AbLang2_mask"
]

modelname_list = [
    "Thrifty-SHM", "Thrifty-SHM + ESM-1v", "AbLang2"
]


dsname  = "tang"
dstitle = "Tang et al."

with open(f'{dfs_dir}/{dsname}_numbering.pkl', 'rb') as f:
    numbering = pickle.load(f)

site_sub_probs_df = {}
r_prec = {}

for model in model_list:
    print("Model:", model)
    
    # load dataframe of SSPs
    site_sub_probs_df[model] = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_ssp_df.csv.gz', index_col=0, dtype={'site':'object'})
    
    # load dataframe of observed and predicted (i.e. top-k) substitutions, to compute R-precision
    muts_obs_pred_df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_site_subs_df.csv.gz', index_col=0, dtype={'site':'object'})
    results = plot_sites_observed_vs_top_k_predictions(muts_obs_pred_df, None, numbering)
    r_prec[model] = results['r-precision']
    

fig = plt.figure(figsize=[15,12])
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(3, height_ratios=[4,4,4])
axs = gs.subplots(sharex=True, sharey=False)

for i in range(len(model_list)):
    model = model_list[i]
    modelname = modelname_list[i]
    print("Model:", modelname)

    results = plot_sites_observed_vs_expected(site_sub_probs_df[model], axs[i], numbering)
    axs[i].text(0.02, 0.9, modelname, verticalalignment ='top', horizontalalignment ='left', transform = axs[i].transAxes, fontsize=15)
    axs[i].text(
        0.02, 0.8,
        f'overlap: {results["overlap"]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = axs[i].transAxes,
        fontsize=15
    )
    axs[i].text(
        0.02, 0.7,
        f'R-precision: {r_prec[model]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = axs[i].transAxes,
        fontsize=15
    )
    axs[i].axes.get_xaxis().get_label().set_visible(False)
    axs[i].axes.get_yaxis().get_label().set_visible(False)
    axs[i].legend().set_visible(False)
    plt.setp(axs[i].get_xticklabels()[1::3], visible=False)
    plt.setp(axs[i].get_xticklabels()[1::2], visible=False)
    axs[i].tick_params(axis="x", labelsize=12, labelrotation=90)

axs[0].legend().set_visible(True)
axs[0].legend(ncol=3, fontsize=15, bbox_to_anchor=(0.98, 1.17))

axs[-1].axes.get_xaxis().get_label().set_visible(True)
axs[-1].set_xlabel("IMGT position", fontsize=20, labelpad=10)

fig.suptitle(dstitle, fontsize=20, x=0.12, y=0.96, ha='left')
fig.supylabel('number of substitutions', fontsize=20)

plt.gcf().text(0, 0.95, "A", fontsize=40)
plt.tight_layout()

outfname = f"{output_dir}/sites_oe"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()
