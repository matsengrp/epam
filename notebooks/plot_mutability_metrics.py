# Plot comparisons of Overlap and R-precision for all models on Tang et al.
# (Figure 2B)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dsname = "tang"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

models = [
    "S5F", "S5FBLOSUM", "S5FESM_mask",
    "ThriftyHumV0.2-59", "ThriftyBLOSUM", "ThriftyESM_mask",
    "ThriftyProdHumV0.2-59", "ESM1v_mask", "AbLang2_mask"
]

df = pd.read_csv(f"{dsname}_metrics.csv", index_col=0)
indices_list = []
for model in models:
    indices_list.append(df.index[df['model']==model].tolist())
indices = np.array(indices_list).flatten()
df = df.loc[indices]

modelnames = df["name"].to_numpy()
overlaps = df["subs_overlap"].to_numpy()
r_precs = df["r_precision"].to_numpy()
yvals = np.arange(len(modelnames))

fig = plt.figure(figsize=[12,6])
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(ncols=2)
axs = gs.subplots(sharex=False, sharey=True)

height=0.8

axs[0].barh(yvals, overlaps, height=height, color='#AAAAAA', edgecolor='black')
axs[0].tick_params(axis="x", labelsize=18)
axs[0].set_xlabel("Overlap", fontsize=22, labelpad=5)
axs[0].grid(axis='x')
axs[0].set_yticks(ticks=yvals, labels=modelnames, fontsize=18)
axs[0].invert_yaxis()

axs[1].barh(yvals, r_precs, height=height, color='#AAAAAA', edgecolor='black')
axs[1].tick_params(axis="x", labelsize=18)
axs[1].set_xlabel("R-precision", fontsize=24, labelpad=5)
axs[1].grid(axis='x')

plt.gcf().text(0, 0.93, "B", fontsize=32)
plt.tight_layout()

outfname = f"{output_dir}/overlap_rprec"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()