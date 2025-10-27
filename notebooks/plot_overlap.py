import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

tab_dir = "tables"
dsname = "tang"

df = pd.read_csv(f'{tab_dir}/{dsname}_overlap.csv')
df = df[df['model']!='AbLang1']
#df = df[df['model'].isin(['S5F','Thrifty-SHM','Thrifty-prod','AbLang2'])]

fig = plt.figure(figsize=[15,6])
#fig = plt.figure(figsize=[15,4])
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(ncols=2, width_ratios=[1,8])
axs = gs.subplots(sharex=False, sharey=True)

plot_df = df[['model','All']].set_index('model')
sns.heatmap(plot_df, cmap='Oranges',
            ax = axs[0],
            vmin = 0.5, vmax = 1,
            #vmin = 0.7, vmax = 1,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar=False
            )
axs[0].tick_params(axis="x", labelsize=18)
axs[0].tick_params(axis="y", labelsize=18)
axs[0].axes.get_yaxis().get_label().set_visible(False)

plot_df = df[['model','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']].set_index('model')
sns.heatmap(plot_df, cmap='Oranges',
            ax = axs[1],
            vmin = 0.5, vmax = 1,
            #vmin = 0.7, vmax = 1,
            annot = True, annot_kws={"fontsize":14}, fmt='.3g',
            square=False, linewidths=0.1,
            cbar_kws={"aspect": 14, "pad": 0.01}
            )
cbar = axs[1].collections[0].colorbar
cbar.set_label("overlap", labelpad=18)
axs[1].tick_params(axis="x", labelsize=18)
axs[1].tick_params(axis="y", left=False)
axs[1].axes.get_yaxis().get_label().set_visible(False)

cax = axs[1].figure.axes[-1]
cax.tick_params(labelsize=16)
cax.yaxis.label.set_size(20)

#plt.gcf().text(0, 0.9, "A", fontsize=40)
plt.tight_layout()

outfname = f"{output_dir}/overlap"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()
