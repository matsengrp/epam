# Plot OE comparsion between different AbLang models and implementations with Rodriguez et al.
# Supplemental Figure
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from netam.oe_plot import (
    get_numbering_dict,
    plot_sites_observed_vs_expected,
    plot_sites_observed_vs_top_k_predictions,
    get_site_substitutions_df,
)
from epam.df_for_plots import (
    get_site_mutabilities_df,
    get_subs_and_preds_from_aaprob,
)
from matplotlib.patches import Rectangle

race_file_version = "rodriguez-airr-seq-race-prod_pcp_2024-07-28_MASKED_NI_noN_no-naive"
epam_results_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2"
anarci_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci"
local_dir = "/home/mjohnso4/epam"
output_dir = f"{local_dir}/output/plots"
anarci_path = f"{anarci_dir}/rodriguez-airr-seq-race-prod_imgt.csv"
pcp_path = f"{local_dir}/pcp_inputs/{race_file_version}.csv"

model_list = [
    "AbLang1", "AbLang2_wt", "AbLang2_mask",
]

modelname_list = [
    "AbLang1", "AbLang2 (wt)", "AbLang2 (mask)"
]

dataset  = race_file_version
dsname   = "rodriguez"
dstitle  = "Rodriguez et al."

pcp_df = load_and_filter_pcp_df(pcp_path)

numbering, excluded = get_numbering_dict(anarci_path, pcp_df, True, "imgt")

site_sub_probs_df = {}
r_prec = {}

for model in model_list:
    print("Model:", model)
    aaprob = f"{epam_results_dir}/{dataset}/{model}/combined_aaprob.hdf5"
    df = get_site_mutabilities_df(aaprob, numbering)
    df.to_csv(f"{output_dir}/{dsname}_{model}_imgt.csv.tar.gz", index=False)
    
    site_sub_probs_df[model] = df
    
    muts_obs_pred_df = get_site_substitutions_df(get_subs_and_preds_from_aaprob(aaprob), numbering)
    results = plot_sites_observed_vs_top_k_predictions(muts_obs_pred_df, None, numbering)
    r_prec[model] = results['r-precision']
    

fig = plt.figure(figsize=[15,12])
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(3, height_ratios=[4,4,4])
axs = gs.subplots(sharex=True, sharey=False)

for i in range(len(model_list)):
    model = model_list[i]
    modelname = modelname_list[i]

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
    plt.setp(axs[i].get_xticklabels()[1::2], visible=False)
    
axs[0].legend().set_visible(True)
axs[0].legend(ncol=3, fontsize=15, bbox_to_anchor=(0.98, 1.17))

axs[-1].axes.get_xaxis().get_label().set_visible(True)
axs[-1].set_xlabel("IMGT position", fontsize=20, labelpad=10)

fig.suptitle(dstitle, fontsize=20, x=0.12, y=0.96, ha='left')
fig.supylabel('number of substitutions', fontsize=20)
plt.tight_layout()

outfname = f"{output_dir}/ablang_comp_{dsname}"
# print(f"{output_dir}.png",'created!')
# plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.png")
plt.savefig(f"{outfname}.pdf")
plt.close()
