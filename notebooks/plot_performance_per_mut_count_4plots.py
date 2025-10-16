# Plot metrics vs number of substitutions for all data sets.
# (Supplementary)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dsinfo_list = [
    ("tang", "Tang et al."),
    ("wyatt", "Jaffe et al."),
    ("rodriguez", "Rodriguez et al."),
    ("ford", "Ford et al."),
]

metricsinfo_list = [
    ('subs_overlap', "overlap", "overlap"),
    ('r_precision', "R-precision", "rprec"),
    ('sub_acc', "substitution accuracy", "subacc"),
    ('csp_perplexity', "CSP perplexity", "cspperp"),
]

XMAX_HUMANS=25
xvals={}
xvals['thriftyprod'] = np.arange(XMAX_HUMANS)+1
xvals['ablang2'] = np.arange(XMAX_HUMANS)+1
xvals['esm'] = np.arange(XMAX_HUMANS)+1
xvals['s5f'] = np.arange(XMAX_HUMANS)+1
xvals['thriftyesm'] = np.arange(XMAX_HUMANS)+1
xvals['thriftyshm'] = np.arange(XMAX_HUMANS)+1

colors={}
colors['thriftyprod'] = '#E69F00'
colors['ablang2'] = '#CC79A7'
colors['esm'] = '#009E73'
colors['s5f'] = '#D55E00'
colors['thriftyesm'] = '#56B4E9'
colors['thriftyshm'] = '#0072B2'

markers={}
markers['thriftyprod'] = '^'
markers['ablang2'] = 'v'
markers['esm'] = 'd'
markers['s5f'] = 'o'
markers['thriftyesm'] = 's'
markers['thriftyshm'] = '*'

markerfill={}
markerfill['thriftyprod'] = 'full'
markerfill['ablang2'] = 'full'
markerfill['esm'] = 'full'
markerfill['s5f'] = 'full'
markerfill['thriftyesm'] = 'full'
markerfill['thriftyshm'] = 'full'

labels={}
labels['thriftyprod'] = 'Thrifty-prod'
labels['ablang2'] = 'AbLang2'
labels['esm'] = 'ESM-1v'
labels['s5f'] = 'S5F'
labels['thriftyesm'] = 'Thrifty-SHM + ESM-1v'
labels['thriftyshm'] = 'Thrifty-SHM'

linsty={}
linsty['thriftyprod'] = 'solid'
linsty['ablang2'] = 'solid'
linsty['esm'] = (0,(1,1))
linsty['s5f'] = (0,(3,1,1,1))
linsty['thriftyesm'] = 'solid'
linsty['thriftyshm'] = 'dashed'


ylims={}
ylims['subs_overlap'] = (0.4, 0.98)
ylims['r_precision'] = (None, None)
ylims['sub_acc'] = (0.16, 0.52)
ylims['csp_perplexity'] = (3.8, 20.8)


tab_dir = "tables"

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for metricsinfo in metricsinfo_list:
    metrics_col = metricsinfo[0]
    metrics_ylabel = metricsinfo[1]
    metrics_label = metricsinfo[2]

    fig = plt.figure(figsize=[24,24])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2, left=0.06, right=0.98, top=0.98, bottom=0.05)
    axs = gs.subplots(sharex=False, sharey=False)

    iax=0
    for dsinfo in dsinfo_list:
        dataset = dsinfo[0]
        dsname = dsinfo[1]
        metrics_df = pd.read_csv(f"{tab_dir}/{dataset}_nsubs_metrics.csv",index_col=0)
        
        irow = iax // 2
        icol = iax % 2

        df_dict={}
        df_dict['thriftyprod'] = metrics_df[metrics_df['name']=='Thrifty-prod']
        df_dict['ablang2'] = metrics_df[metrics_df['name']=='AbLang2']
        df_dict['esm'] = metrics_df[metrics_df['name']=='ESM-1v']
        df_dict['s5f'] = metrics_df[metrics_df['name']=='S5F']
        df_dict['thriftyesm'] = metrics_df[metrics_df['name']=='Thrifty-SHM + ESM-1v']
        df_dict['thriftyshm'] = metrics_df[metrics_df['name']=='Thrifty-SHM']

        yvals={}
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            #print(key)
            yvals[key] = df_dict[key][metrics_col].to_numpy()


        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            axs[irow,icol].plot(xvals[key], yvals[key], 
                                color=colors[key], 
                                linewidth=2,
                                marker=markers[key], markersize=8, markeredgewidth=1, fillstyle=markerfill[key], linestyle=linsty[key],
                                label=labels[key])
        axs[irow,icol].set_xlabel("number of substitutions", fontsize=30)
        axs[irow,icol].set_ylabel(metrics_ylabel, fontsize=30)
        axs[irow,icol].tick_params(axis="x", labelsize=24)
        axs[irow,icol].tick_params(axis="y", labelsize=24)
        axs[irow,icol].grid()
        axs[irow,icol].set_title(dsname, fontsize=30)
        axs[irow,icol].legend(fontsize=18)
        axs[irow,icol].set_ylim(ylims[metrics_col][0], ylims[metrics_col][1])
        
        iax += 1

    outfname = f"{output_dir}/humans_{metrics_label}_vs_nmuts"
    print(f"{outfname}.png",'created!')
    plt.savefig(f"{outfname}.png")
    print(f"{outfname}.pdf",'created!')
    plt.savefig(f"{outfname}.pdf")
    plt.close()