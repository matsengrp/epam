# Plot comparisons of Overlap and R-precision for all models on Replay.
# (Figure 5A and Supplementary)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


models = [
    'GCReplaySHM', 'GCReplaySHMBLOSUMSigmoid', 'GCReplaySHMDMSSigmoid', 'GCReplaySHMESMSigmoid',
    'GCReplayESM', 'GCReplayAbLang2'
]

for chain in ['igh','igk']:
    print("chain:",chain)
    
    df = pd.read_csv(f"gcreplay_{chain}_metrics.csv", index_col=0)
    indices_list = []
    for model in models:
        indices_list.append(df.index[df['model']==model].tolist())
    indices = np.array(indices_list).flatten()
    df = df.loc[indices]

    modelnames = df["name"].to_numpy()
    overlaps = df["subs_overlap"].to_numpy()
    r_precs = df["r_precision"].to_numpy()
    yvals = np.arange(len(modelnames))
    
    fig = plt.figure(figsize=[12,4])
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
    axs[0].set_xlim(0, 0.85)

    axs[1].barh(yvals, r_precs, height=height, color='#AAAAAA', edgecolor='black')
    axs[1].tick_params(axis="x", labelsize=18)
    axs[1].set_xlabel("R-precision", fontsize=24, labelpad=5)
    axs[1].grid(axis='x')
    axs[1].set_xlim(0, 0.154)

    plt.gcf().text(0, 0.91, "A", fontsize=32)
    plt.tight_layout()

    outfname = f"{output_dir}/gcreplay_{chain}_overlap_rprec"
    print(f"{outfname}.png",'created!')
    plt.savefig(f"{outfname}.png")
    print(f"{outfname}.pdf",'created!')
    plt.savefig(f"{outfname}.pdf")
    plt.close()