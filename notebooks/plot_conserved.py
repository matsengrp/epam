# Plot observed vs expected number of substitutions at known conserved sites
# (Figure 3)

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from epam.oe_plot import (
    plot_sites_observed_vs_expected,
    plot_sites_observed_vs_top_k_predictions,
)
from matplotlib.patches import Rectangle

dfs_dir = "dataframes"
tab_dir = "tables"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

dsname = 'tang'
dstitle = 'Tang et al.'

model = "ThriftyProdHumV0.2-59"
modelname = "Thrifty-prod"


conserved_summary_path = f"{tab_dir}/{dsname}_conserved.csv"
modnames = [
    "S5F", "Thrifty-SHM", "ESM-1v", "Thrifty-SHM + ESM-1v", "AbLang2", "Thrifty-prod"
]

with open(f'{dfs_dir}/{dsname}_numbering.pkl', 'rb') as f:
    numbering = pickle.load(f)

conserved_xvals = ["23", "41", "43", "98", "102", "104", "118"]
conserved_xpos = []
for site in conserved_xvals:
    conserved_xpos.append(numbering[("reference", 0)].index(site) - 0.5)

cons_df = pd.read_csv(conserved_summary_path)
cons_df = cons_df[cons_df['site']!=75]
observed = cons_df['observed'].to_numpy()


def plot_sites(modelname, expected, observed, ax):    
    xvals = np.arange(len(expected))

    ax.set_title(modelname, fontsize=18)
    ax.bar(xvals,
           expected,
           width=0.8,
           linewidth=2,
           facecolor="white",
           edgecolor="#0072B2",
           label="Expected",
           )
    ax.plot(xvals,
            observed,
            marker="o",
            markersize=10,
            linewidth=0,
            color="#000000",
            label="Observed",
            )
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(xvals, conserved_xvals)
        

fig = plt.figure(constrained_layout=False, figsize=(16, 12))
fig.patch.set_facecolor('white')

(subfig_t, subfig_b) = fig.subfigures(2, 1, height_ratios=[2, 3], hspace = 0.15)

#
# Per site OE plots
#
oe_ax = subfig_t.subplots()

muts_obs_pred_df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_site_subs_df.csv.gz', index_col=0, dtype={'site':'object'})
results = plot_sites_observed_vs_top_k_predictions(muts_obs_pred_df, None, numbering)
r_prec = results['r-precision']

site_sub_probs_df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_ssp_df.csv.gz', index_col=0, dtype={'site':'object'})
results = plot_sites_observed_vs_expected(site_sub_probs_df, oe_ax, numbering)
oe_ax.text(0.02, 0.9, modelname, verticalalignment ='top', horizontalalignment ='left', transform = oe_ax.transAxes, fontsize=15)
oe_ax.text(
    0.02, 0.8,
    f'overlap: {results["overlap"]:.3g}',
    verticalalignment ='top', 
    horizontalalignment ='left', 
    transform = oe_ax.transAxes,
    fontsize=15
)
oe_ax.text(
    0.02, 0.7,
    f'R-precision: {r_prec:.3g}',
    verticalalignment ='top', 
    horizontalalignment ='left', 
    transform = oe_ax.transAxes,
    fontsize=15
)
for xpos in conserved_xpos:
    oe_ax.add_patch(Rectangle((xpos,0),1,oe_ax.get_ylim()[1],color='red',alpha=0.08))
oe_ax.set_xlabel("IMGT position", fontsize=20, labelpad=10)
oe_ax.set_ylabel("number of substitutions", fontsize=18, labelpad=10)
oe_ax.legend(ncol=3, fontsize=15, bbox_to_anchor=(0.98, 1.15))
oe_ax.set_title(dstitle, fontsize=20, x=0.01, y=1.03, ha='left')
plt.setp(oe_ax.get_xticklabels()[1::3], visible=False)
plt.setp(oe_ax.get_xticklabels()[1::2], visible=False)
oe_ax.tick_params(axis="x", labelsize=12, labelrotation=90)

#
# OE at conserved sites
#
gs = subfig_b.add_gridspec(2, 3)
axs = gs.subplots(sharex=True, sharey=False)

plot_sites('S5F', cons_df['S5F'].to_numpy(), observed, axs[0,0])
plot_sites("Thrifty-SHM", cons_df["Thrifty-SHM"].to_numpy(), observed, axs[0,1])
plot_sites('ESM-1v', cons_df['ESM-1v'].to_numpy(), observed, axs[0,2])

plot_sites("Thrifty-SHM + ESM-1v", cons_df["Thrifty-SHM + ESM-1v"].to_numpy(), observed, axs[1,0])
plot_sites('AbLang2', cons_df['AbLang2'].to_numpy(), observed, axs[1,1])
plot_sites("Thrifty-prod", cons_df["Thrifty-prod"].to_numpy(), observed, axs[1,2])


subfig_b.supylabel('number of substitutions', fontsize=18, x=0.05)

plt.gcf().text(0.01, 0.965, "A", fontsize=36)
plt.gcf().text(0.01, 0.525, "B", fontsize=36)

outfname = f"{output_dir}/conserved"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png", bbox_inches='tight')
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf", bbox_inches='tight')
plt.close()
