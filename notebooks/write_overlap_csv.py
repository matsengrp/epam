# Write a dataframe of overlap for all models on Tang et al.
# (Supplementary)

import os
import numpy as np
import pandas as pd

dfs_dir = "dataframes"
dsname = "wyatt"

cdr_bounds = [(27,38), (56,65), (104,117)]

output_dir = "tables"
os.makedirs(output_dir, exist_ok=True)

outfname = f"{output_dir}/{dsname}_overlap.csv"
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
    
    region_overlap = {}

    for region in ['All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']:
        if region == 'FWR1':
            region_df = ssp_df[ssp_df['site']<cdr_bounds[0][0]]
        elif region == 'CDR1':
            region_df = ssp_df[(ssp_df['site']>=cdr_bounds[0][0]) & (ssp_df['site']<=cdr_bounds[0][1])]
        elif region == 'FWR2':
            region_df = ssp_df[(ssp_df['site']>cdr_bounds[0][1]) & (ssp_df['site']<cdr_bounds[1][0])]
        elif region == 'CDR2':
            region_df = ssp_df[(ssp_df['site']>=cdr_bounds[1][0]) & (ssp_df['site']<=cdr_bounds[1][1])]
        elif region == 'FWR3':
            region_df = ssp_df[(ssp_df['site']>cdr_bounds[1][1]) & (ssp_df['site']<cdr_bounds[2][0])]
        elif region == 'CDR3':
            region_df = ssp_df[(ssp_df['site']>=cdr_bounds[2][0]) & (ssp_df['site']<=cdr_bounds[2][1])]
        elif region == 'FWR4':
            region_df = ssp_df[ssp_df['site']>cdr_bounds[2][1]]
        else:
            region_df = ssp_df

        oe_df = region_df[['site','prob','mutation']].groupby('site').sum()
        region_overlap[region] = sum([min(row['prob'], row['mutation']) for i,row in oe_df.iterrows()])/0.5/(oe_df['prob'].sum() + oe_df['mutation'].sum())
    
    print(region_overlap['All'])
    
    output_df.loc[len(output_df)] = [
        modelname,
        region_overlap['All'],
        region_overlap['FWR1'],
        region_overlap['CDR1'],
        region_overlap['FWR2'],
        region_overlap['CDR2'],
        region_overlap['FWR3'],
        region_overlap['CDR3'],
        region_overlap['FWR4'],
    ]

print(output_df)
output_df.to_csv(outfname,index=False)
print(ssp_df.groupby('pcp_index').ngroups)
