import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from netam.sequences import (
    translate_sequence,
)

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

pcp_dir = "pcp_inputs"

dsinfo_list = [
    ("Tang et al.",f"{pcp_dir}/tang-deepshm-prod_pcp_2024-08-08_MASKED_NI_noN_no-naive.csv",'#E69F00','solid'),
    ("Jaffe et al.",f"{pcp_dir}/wyatt-10x-1p5m_paired-igh_fs-all_pcp_2024-11-22_NI_noN_no-naive.csv",'#009E73','dashed'),
    ("Rodriguez et al.",f"{pcp_dir}/rodriguez-airr-seq-race-prod_pcp_2024-07-28_MASKED_NI_noN_no-naive.csv",'#CC79A7','dashdot'),
    ("Ford et al.",f"{pcp_dir}/ford-flairr-seq-prod_pcp_2024-07-26_MASKED_NI_noN_no-naive.csv",'#0072B2','dotted'),
]

nsubs={}
colors={}
linsty={}
for dsinfo in dsinfo_list:
    dsname = dsinfo[0]
    nsubs[dsname] = []
        
    pcpfile = dsinfo[1]
    df = pd.read_csv(pcpfile, index_col=0)
    for i,row in df.iterrows():
        parent_aa = translate_sequence(row['parent'])
        child_aa = translate_sequence(row['child'])
        
        n = sum([p!=c for p,c in zip(parent_aa,child_aa)])
        if n>0:
            nsubs[dsname].append(n)
    
    colors[dsname] = dsinfo[2]
    linsty[dsname] = dsinfo[3]


fig, ax = plt.subplots(figsize=[10,6])
fig.patch.set_facecolor('white')

for key in nsubs.keys():
    ax.hist(nsubs[key], bins=(np.arange(30) + 0.5),
            color=colors[key],
            linewidth=2,
            linestyle=linsty[key],
            density=True,
            histtype='step',
            label=key)

ax.set_xlabel("number of substitutions", fontsize=30)
ax.tick_params(axis="x", labelsize=24)
ax.set_ylabel("proportion", fontsize=30)
ax.tick_params(axis="y", labelsize=20)
ax.legend(fontsize=18)
ax.set_yscale('log')
ax.grid()

plt.tight_layout()

outfname = f"{output_dir}/nsubs"
print(f"{outfname}.png",'created!')
plt.savefig(f"{outfname}.png")
print(f"{outfname}.pdf",'created!')
plt.savefig(f"{outfname}.pdf")
plt.close()    