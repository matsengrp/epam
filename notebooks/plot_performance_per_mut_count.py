import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

df_dict={}
df_dict['tang'] = pd.read_csv('tang_thriftyprod_eval.csv', index_col=0)
df_dict['gcreplay_igh'] = pd.read_csv('gcreplay_igh_shmdms_eval.csv', index_col=0)
df_dict['gcreplay_igk'] = pd.read_csv('gcreplay_igk_shmdms_eval.csv', index_col=0)

XMAX_HUMANS=25
XMAX_REPLAY=8
xvals={}
xvals['tang'] = np.arange(XMAX_HUMANS)+1
xvals['gcreplay_igh'] = np.arange(XMAX_REPLAY)+1
xvals['gcreplay_igk'] = np.arange(XMAX_REPLAY)+1

colors={}
colors['tang'] = '#E69F00'
colors['gcreplay_igh'] = '#009E73'
colors['gcreplay_igk'] = '#CC79A7'

markers={}
markers['tang'] = '^'
markers['gcreplay_igh'] = 'o'
markers['gcreplay_igk'] = 'o'

markerfill={}
markerfill['tang'] = 'full'
markerfill['gcreplay_igh'] = 'full'
markerfill['gcreplay_igk'] = 'none'

labels={}
labels['tang'] = 'Thrifty-prod (Tang et al.)'
labels['gcreplay_igh'] = 'ReplaySHM + DMS (IgH)'
labels['gcreplay_igk'] = f'ReplaySHM + DMS (Ig$\kappa$)'

datalabels={}
datalabels['tang'] = 'Tang et al.'
datalabels['gcreplay_igh'] = 'Replay IgH'
datalabels['gcreplay_igk'] = f'Replay Ig$\kappa$'

linsty={}
linsty['tang'] = 'solid'
linsty['gcreplay_igh'] = 'solid'
linsty['gcreplay_igk'] = 'dashed'

yvals={}
nmuts={}
for key in ['tang','gcreplay_igh','gcreplay_igk']:
    #print(key)
    yvals[key] = []
    df = df_dict[key]
    df = df[df['r_prec']>-1]
    for i in xvals[key]:
        dfi = df[df['mut_count']==i]
        rprec = sum(dfi['r_prec'])/dfi.shape[0]
        #print(i, rprec)
        yvals[key].append(rprec)
    
    nmuts[key] = df['mut_count'].to_numpy()

fig = plt.figure(figsize=[8,10])
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(2,height_ratios=[3,2])
axs = gs.subplots(sharex=True, sharey=False)
for key in ['gcreplay_igh', 'gcreplay_igk','tang']:
    axs[0].plot(xvals[key], yvals[key], 
                color=colors[key], 
                linewidth=2,
                marker=markers[key], markersize=12, markeredgewidth=3, fillstyle=markerfill[key], 
                label=labels[key])

axs[0].set_ylabel("R-precision", fontsize=30)
axs[0].tick_params(axis="y", labelsize=24)
axs[0].grid()
axs[0].legend(fontsize=18)

for key in ['gcreplay_igh', 'gcreplay_igk','tang']:
    axs[1].hist(nmuts[key], bins=(xvals['tang'] - 0.5), 
                color=colors[key], 
                linewidth=3,
                linestyle=linsty[key],
                density=True,
                histtype='step',
                label=datalabels[key])

axs[1].set_xlabel("number of substitutions", fontsize=30)
axs[1].tick_params(axis="x", labelsize=24)
axs[1].set_ylabel("proportion", fontsize=30)
axs[1].tick_params(axis="y", labelsize=20)
axs[1].legend(fontsize=18)
axs[1].set_yscale('log')
axs[1].grid()

plt.tight_layout()

outfname = f"{output_dir}/rprec_vs_nmuts"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()