import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

output_dir = "plots/conscount"
os.makedirs(output_dir, exist_ok=True)

dsname = "tang"

df = pd.read_csv(f'{dsname}_nsubs_conscount_subacc.csv')

conscats = ['cons3','cons4','cons5','cons6_10','cons11_100','consmax']
xticks = np.arange(len(conscats))
xticklabels = ['3','4','5','6-10','11-100','101+']

yvals = {}
npcps = {}
for nsubs in [1,2,3]:
    yvals[nsubs] = []
    npcps[nsubs] = []
    for cat in conscats:
        yvals[nsubs].append(df[(df['nsubs']==nsubs) & (df['conscat']==cat)]['All'].item())
        npcps[nsubs].append(df[(df['nsubs']==nsubs) & (df['conscat']==cat)]['nPCP'].item())

colors={}
colors[1] = '#E69F00'
colors[2] = '#009E73'
colors[3] = '#56B4E9'

markers = {}
markers[1] = 'o'
markers[2] = '^'
markers[3] = 's'


fig = plt.figure(figsize=[10,12])
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(4,height_ratios=[3,1,1,1])
axs = gs.subplots(sharex=True, sharey=False)
for nsubs in [1,2,3]:
    axs[0].plot(xticks, yvals[nsubs],
                color=colors[nsubs],
                linewidth=2,
                marker=markers[nsubs], markersize=12, markeredgewidth=2, fillstyle='full',
                label=f'N={nsubs}')

axs[0].set_ylabel("substitution accuracy", fontsize=26)
axs[0].tick_params(axis="y", labelsize=24)
axs[0].grid()
axs[0].legend(fontsize=18)

for nsubs in [1,2,3]:
    axs[nsubs].bar(xticks,
                   npcps[nsubs],
                   width=0.9,
                   color=colors[nsubs],
                   edgecolor='k',
                )

    axs[nsubs].tick_params(axis="x", labelsize=24)
    axs[nsubs].set_ylabel("PCPs", fontsize=26)
    axs[nsubs].tick_params(axis="y", labelsize=20)
    axs[nsubs].grid()

axs[-1].set_xlabel("CONSCOUNT", fontsize=26)
axs[-1].set_xticks(xticks)
axs[-1].set_xticklabels(xticklabels)

fig.suptitle('Thrifty-prod, Tang et al., leaf', fontsize=26)

plt.tight_layout()

outfname = f"{output_dir}/subacc_vs_conscount"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()