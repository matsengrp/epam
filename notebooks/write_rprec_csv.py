# Write a dataframe of R-precision for all models on Tang et al.
# (Supplementary)

import os
import numpy as np
import pandas as pd

dfs_dir = "dataframes"
dsname = "rodriguez"

cdr_bounds = [(27,38), (56,65), (104,117)]

output_dir = "tables"
os.makedirs(output_dir, exist_ok=True)

outfname = f"{output_dir}/{dsname}_rprec.csv"
output_df = pd.DataFrame(columns=['model','All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4'])

model_list = [
    "S5F", "S5FESM_mask", "S5FBLOSUM",
    "ThriftyHumV0.2-59", "ThriftyESM_mask", "ThriftyBLOSUM",
    "ThriftyProdHumV0.2-59",
    "ESM1v_mask", "AbLang2_mask", "AbLang1"
]

modelname_list = [
    "S5F", "S5F + ESM-1v", "S5F + BLOSUM62",
    "Thrifty-SHM", "Thrifty-SHM + ESM-1v", "Thrifty-SHM + BLOSUM62",
    "Thrifty-prod",
    "ESM-1v", "AbLang2", "AbLang1"
]

for model, modelname in zip(model_list, modelname_list):
    print("Model:", model)
    
    ssp_df = pd.read_csv(f"{dfs_dir}/{dsname}_{model}_ssp_df.csv.gz", index_col=0)
    
    pcp_groups = ssp_df.groupby('pcp_index')
    region_rprec = {}
    region_k_subs = {}
    for region in ['All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']:
        region_rprec[region] = []
        region_k_subs[region] = []
    for pcp_index, df in pcp_groups:
        for region in ['All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']:
            if region == 'FWR1':
                region_df = df[df['site']<cdr_bounds[0][0]]
            elif region == 'CDR1':
                region_df = df[(df['site']>=cdr_bounds[0][0]) & (df['site']<=cdr_bounds[0][1])]
            elif region == 'FWR2':
                region_df = df[(df['site']>cdr_bounds[0][1]) & (df['site']<cdr_bounds[1][0])]
            elif region == 'CDR2':
                region_df = df[(df['site']>=cdr_bounds[1][0]) & (df['site']<=cdr_bounds[1][1])]
            elif region == 'FWR3':
                region_df = df[(df['site']>cdr_bounds[1][1]) & (df['site']<cdr_bounds[2][0])]
            elif region == 'CDR3':
                region_df = df[(df['site']>=cdr_bounds[2][0]) & (df['site']<=cdr_bounds[2][1])]
            elif region == 'FWR4':
                region_df = df[df['site']>cdr_bounds[2][1]]
            else:
                region_df = df
            
            k_subs = region_df[region_df['mutation']==True].shape[0]
            if k_subs==0:
                continue
            region_k_subs[region].append(k_subs)
            
            sorted_region_df = region_df.sort_values('prob', ascending=False)
            n_corr = sum(sorted_region_df.head(k_subs)['mutation'].to_numpy())
            rprec = n_corr/k_subs
            region_rprec[region].append(rprec)            
    
    print(np.mean(region_rprec['All']))   
    
    output_df.loc[len(output_df)] = [
        modelname,
        np.mean(region_rprec['All']),
        np.mean(region_rprec['FWR1']),
        np.mean(region_rprec['CDR1']),
        np.mean(region_rprec['FWR2']),
        np.mean(region_rprec['CDR2']),
        np.mean(region_rprec['FWR3']),
        np.mean(region_rprec['CDR3']),
        np.mean(region_rprec['FWR4']),
    ]

print(output_df)
output_df.to_csv(outfname,index=False)
print(ssp_df.groupby('pcp_index').ngroups)
