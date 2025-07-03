import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dsinfo_list = [
    ("ford", "Ford et al."),
    ("rodriguez", "Rodriguez et al."),
    ("wyatt", "Jaffe et al."),
    ("tang", "Tang et al."),
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


for pcptype in ['','anc','leaf']:
    output_dir = "plots"
    if len(pcptype)>0:
        output_dir = f"plots/{pcptype}"
    os.makedirs(output_dir, exist_ok=True)

    for dsinfo in dsinfo_list:
        dataset = dsinfo[0]
        dsname = dsinfo[1]
        if pcptype=='anc':
            dsname = dsname + ", ancestral"
        elif pcptype=='leaf':
            dsname = dsname + ", leaf"
        
        if len(pcptype)>0:
            metrics_df = pd.read_csv(f"{dataset}_{pcptype}_nsubs_metrics.csv",index_col=0)
        else:
            metrics_df = pd.read_csv(f"{dataset}_nsubs_metrics.csv",index_col=0)

        df_dict={}
        df_dict['thriftyprod'] = metrics_df[metrics_df['name']=='Thrifty-prod']
        df_dict['ablang2'] = metrics_df[metrics_df['name']=='AbLang2']
        df_dict['esm'] = metrics_df[metrics_df['name']=='ESM-1v']
        df_dict['s5f'] = metrics_df[metrics_df['name']=='S5F']
        df_dict['thriftyesm'] = metrics_df[metrics_df['name']=='Thrifty-SHM + ESM-1v']
        df_dict['thriftyshm'] = metrics_df[metrics_df['name']=='Thrifty-SHM']

        yvals={}
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            print(key)
            yvals[key] = df_dict[key]['subs_overlap'].to_numpy()

        fig, ax = plt.subplots(figsize=[10,10])
        fig.patch.set_facecolor('white')
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            ax.plot(xvals[key], yvals[key], 
                    color=colors[key], 
                    linewidth=2,
                    marker=markers[key], markersize=8, markeredgewidth=1, fillstyle=markerfill[key], linestyle=linsty[key],
                    label=labels[key])
        ax.set_xlabel("number of substitutions", fontsize=30)
        ax.set_ylabel("overlap", fontsize=30)
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.grid()
        ax.set_title(dsname, fontsize=30)
        ax.legend(fontsize=18)
        ax.set_ylim(0.4,0.98)

        plt.tight_layout()

        outfname = f"{output_dir}/{dataset}_overlap_vs_nmuts"
        print(f"{outfname}.png",'created!')
        plt.savefig(f"{outfname}.png")
        print(f"{outfname}.pdf",'created!')
        plt.savefig(f"{outfname}.pdf")
        plt.close()


        yvals={}
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            print(key)
            yvals[key] = df_dict[key]['r_precision'].to_numpy()

        fig, ax = plt.subplots(figsize=[10,10])
        fig.patch.set_facecolor('white')
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            ax.plot(xvals[key], yvals[key], 
                    color=colors[key], 
                    linewidth=2,
                    marker=markers[key], markersize=8, markeredgewidth=1, fillstyle=markerfill[key], linestyle=linsty[key],
                    label=labels[key])
        ax.set_xlabel("number of substitutions", fontsize=30)
        ax.set_ylabel("R-precision", fontsize=30)
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.grid()
        ax.set_title(dsname, fontsize=30)
        ax.legend(fontsize=18)

        plt.tight_layout()

        outfname = f"{output_dir}/{dataset}_rprec_vs_nmuts"
        print(f"{outfname}.png",'created!')
        plt.savefig(f"{outfname}.png")
        print(f"{outfname}.pdf",'created!')
        plt.savefig(f"{outfname}.pdf")
        plt.close()


        yvals={}
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            print(key)
            yvals[key] = df_dict[key]['sub_acc'].to_numpy()

        fig, ax = plt.subplots(figsize=[10,10])
        fig.patch.set_facecolor('white')
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            ax.plot(xvals[key], yvals[key], 
                    color=colors[key], 
                    linewidth=2,
                    marker=markers[key], markersize=8, markeredgewidth=1, fillstyle=markerfill[key], linestyle=linsty[key],
                    label=labels[key])
        ax.set_xlabel("number of substitutions", fontsize=30)
        ax.set_ylabel("substitution accuracy", fontsize=30)
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.grid()
        ax.set_title(dsname, fontsize=30)
        ax.legend(fontsize=18)
        ax.set_ylim(0.16,0.52)

        plt.tight_layout()

        outfname = f"{output_dir}/{dataset}_subacc_vs_nmuts"
        print(f"{outfname}.png",'created!')
        plt.savefig(f"{outfname}.png")
        print(f"{outfname}.pdf",'created!')
        plt.savefig(f"{outfname}.pdf")
        plt.close()


        yvals={}
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            print(key)
            yvals[key] = df_dict[key]['csp_perplexity'].to_numpy()

        fig, ax = plt.subplots(figsize=[10,10])
        fig.patch.set_facecolor('white')
        for key in ['thriftyprod','thriftyshm','thriftyesm','s5f','ablang2','esm']:
            ax.plot(xvals[key], yvals[key], 
                    color=colors[key], 
                    linewidth=2,
                    marker=markers[key], markersize=8, markeredgewidth=1, fillstyle=markerfill[key], linestyle=linsty[key],
                    label=labels[key])
        ax.set_xlabel("number of substitutions", fontsize=30)
        ax.set_ylabel("CSP perplexity", fontsize=30)
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.grid()
        ax.set_title(dsname, fontsize=30)
        ax.legend(fontsize=18, ncol=2)
        #ax.set_xlim(0,2)
        ax.set_ylim(3.8,20.8)

        plt.tight_layout()

        outfname = f"{output_dir}/{dataset}_cspperp_vs_nmuts"
        print(f"{outfname}.png",'created!')
        plt.savefig(f"{outfname}.png")
        print(f"{outfname}.pdf",'created!')
        plt.savefig(f"{outfname}.pdf")
        plt.close()