import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "plots/cfsize"
os.makedirs(output_dir, exist_ok=True)

dsname = "tang"

df = pd.read_csv(f'{dsname}_nsubs_conscount_cfsize.csv')

colors={}
colors[1] = '#E69F00'
colors[2] = '#009E73'
colors[3] = '#56B4E9'


cfsize={}
conscount={}
for nsubs in [1,2,3]:
    cfsize[nsubs] = df[df['nsubs']==nsubs]['cfsize'].to_numpy()
    conscount[nsubs] = df[df['nsubs']==nsubs]['conscount'].to_numpy()

fig = plt.figure(figsize=[24,8])
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(ncols=3)
axs = gs.subplots(sharex=True, sharey=True)
for nsubs in [1,2,3]:
    axs[nsubs-1].scatter(cfsize[nsubs], conscount[nsubs], alpha=0.1, facecolor=colors[nsubs], edgecolors='k', s=100)
    axs[nsubs-1].set_xlabel("CF size", fontsize=26)
    axs[nsubs-1].tick_params(axis="x", labelsize=24)
    axs[nsubs-1].grid()
    axs[nsubs-1].set_title(f"N = {nsubs}", fontsize=24)

axs[0].set_ylabel("CONSCOUNT", fontsize=26)
axs[0].tick_params(axis="y", labelsize=24)
axs[0].set_yscale('log')

plt.tight_layout()

outfname = f"{output_dir}/conscount_cfsize"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()


conscount={}
cfsize={}
for nsubs in [1,2,3]:
    nsubs_df = df[df['nsubs']==nsubs]
    cfgroups = nsubs_df.groupby(['sample_id','family'])
    conscount[nsubs] = []
    cfsize[nsubs] = []
    for cfname, cf in cfgroups:
        cfsize[nsubs].append(cf['cfsize'].head(1).item())
        conscount[nsubs].append(np.median(cf['conscount']))

fig = plt.figure(figsize=[24,8])
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(ncols=3)
axs = gs.subplots(sharex=True, sharey=True)
for nsubs in [1,2,3]:
    axs[nsubs-1].scatter(cfsize[nsubs], conscount[nsubs], alpha=0.1, facecolor=colors[nsubs], edgecolors='k', s=100)
    axs[nsubs-1].set_xlabel("CF size", fontsize=26)
    axs[nsubs-1].tick_params(axis="x", labelsize=24)
    axs[nsubs-1].grid()
    axs[nsubs-1].set_title(f"N = {nsubs}", fontsize=24)

axs[0].set_ylabel("median CONSCOUNT", fontsize=26)
axs[0].tick_params(axis="y", labelsize=24)
axs[0].set_yscale('log')

plt.tight_layout()

outfname = f"{output_dir}/median_conscount_cfsize_per_cf"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()