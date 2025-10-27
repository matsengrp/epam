import numpy as np
import pandas as pd
from epam.oe_plot import (
    plot_sites_observed_vs_expected,
)


epam_results_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2/gcreplay"
epam_esm_results_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2/gcreplay_esm"
top_k = 1


model_list = [
    'GCReplaySHM', 'GCReplaySHMBLOSUMSigmoid', 'GCReplaySHMDMSSigmoid', 'GCReplaySHMESMSigmoid',
    'GCReplayESM', 'GCReplayAbLang2'
]

modelname_list = [
    "ReplaySHM", "ReplaySHM + BLOSUM62", "ReplaySHM + DMS", "ReplaySHM + ESM-1v",
    "ESM-1v", "AbLang2"
]

output_dir = "tables"
os.makedirs(output_dir, exist_ok=True)

dfs_dir = "dataframes"

only_1sub = False
only_naive_parent = False
only_leaf_child = False

cdr_bounds = {}
cdr_bounds['igh'] = [(25,32), (49,56), (95,100)]
cdr_bounds['igk'] = [(26,31), (49,51), (88,96)]

for chain in ['igh','igk']:
    print('chain:',chain)
    
    outfname = f"{output_dir}/gcreplay_{chain}"
    if only_naive_parent==True:
        outfname = outfname + "_naive"
    if only_leaf_child==True:
        outfname = outfname + "_leaf"
    if only_1sub==True:
        outfname = outfname + "_1sub"
    outfname = outfname + "_overlap.csv"
    output_df = pd.DataFrame(columns=['model','All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4'])
    
    pcp_path = f"pcp_gcreplay_inputs/{chain}/gctrees_2025-01-10-full_{chain}_pcp_NoBackMuts.csv"
    pcp_df = pd.read_csv(pcp_path, index_col=0)
    
    for model, modelname in zip(model_list, modelname_list):
        print("Model:", model)
        
        full_ssp_df = pd.read_csv(f"{dfs_dir}/{model}_{chain}_ssp_df.csv.gz", index_col=0)
        
        if (only_1sub==False) and (only_naive_parent==False) and (only_leaf_child==False):
            ssp_df = full_ssp_df
        else:
            pcp_groups = full_ssp_df.groupby('pcp_index')
            keep_pcp_indices = []
            for pcp_index, df in pcp_groups:
                if (only_naive_parent==True) and (pcp_df.loc[pcp_index]['parent_is_naive']!=True):
                    continue
                if (only_leaf_child==True) and (pcp_df.loc[pcp_index]['child_is_leaf']!=True):
                    continue
                nsubs = sum(df['mutation'].to_numpy())
                if (only_1sub==True) and (nsubs!=1):
                    continue
                keep_pcp_indices.append(pcp_index)
            ssp_df = full_ssp_df[full_ssp_df['pcp_index'].isin(keep_pcp_indices)]
                
        region_overlap = {}

        for region in ['All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']:
            if region == 'FWR1':
                region_df = ssp_df[ssp_df['site']<cdr_bounds[chain][0][0]]
            elif region == 'CDR1':
                region_df = ssp_df[(ssp_df['site']>=cdr_bounds[chain][0][0]) & (ssp_df['site']<=cdr_bounds[chain][0][1])]
            elif region == 'FWR2':
                region_df = ssp_df[(ssp_df['site']>cdr_bounds[chain][0][1]) & (ssp_df['site']<cdr_bounds[chain][1][0])]
            elif region == 'CDR2':
                region_df = ssp_df[(ssp_df['site']>=cdr_bounds[chain][1][0]) & (ssp_df['site']<=cdr_bounds[chain][1][1])]
            elif region == 'FWR3':
                region_df = ssp_df[(ssp_df['site']>cdr_bounds[chain][1][1]) & (ssp_df['site']<cdr_bounds[chain][2][0])]
            elif region == 'CDR3':
                region_df = ssp_df[(ssp_df['site']>=cdr_bounds[chain][2][0]) & (ssp_df['site']<=cdr_bounds[chain][2][1])]
            elif region == 'FWR4':
                region_df = ssp_df[ssp_df['site']>cdr_bounds[chain][2][1]]
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
