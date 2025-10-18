import numpy as np
import pandas as pd


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

dfs_dir = "dataframes"

output_dir = "tables"
os.makedirs(output_dir, exist_ok=True)

only_1sub = False
only_naive_parent = False
only_leaf_child = True

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
    outfname = outfname + "_rprec.csv"
    output_df = pd.DataFrame(columns=['model','All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4'])
    
    pcp_path = f"pcp_gcreplay_inputs/{chain}/gctrees_2025-01-10-full_{chain}_pcp_NoBackMuts.csv"
    pcp_df = pd.read_csv(pcp_path, index_col=0)
    
    for model, modelname in zip(model_list, modelname_list):
        print("Model:", model)
        
        full_ssp_df = pd.read_csv(f"{dfs_dir}/{model}_{chain}_ssp_df.csv.gz", index_col=0)
        
        if (only_1sub==False) and (only_naive_parent==False) and (only_leaf_child==False):
            ssp_df = full_ssp_df
        else:
            full_pcp_groups = full_ssp_df.groupby('pcp_index')
            keep_pcp_indices = []
            for pcp_index, df in full_pcp_groups:
                if (only_naive_parent==True) and (pcp_df.loc[pcp_index]['parent_is_naive']!=True):
                    continue
                if (only_leaf_child==True) and (pcp_df.loc[pcp_index]['child_is_leaf']!=True):
                    continue
                nsubs = sum(df['mutation'].to_numpy())
                if (only_1sub==True) and (nsubs!=1):
                    continue
                keep_pcp_indices.append(pcp_index)
            ssp_df = full_ssp_df[full_ssp_df['pcp_index'].isin(keep_pcp_indices)]

        pcp_groups = ssp_df.groupby('pcp_index')
        region_rprec = {}
        region_k_subs = {}
        region_size = {}
        for region in ['All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']:
            region_rprec[region] = []
            region_k_subs[region] = []
        for pcp_index, df in pcp_groups:
            assert df.iloc[-1]['site'] == (df.shape[0]-1)
            for region in ['All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']:
                if region == 'FWR1':
                    region_df = df[df['site']<cdr_bounds[chain][0][0]]
                    region_size[region] = cdr_bounds[chain][0][0]
                elif region == 'CDR1':
                    region_df = df[(df['site']>=cdr_bounds[chain][0][0]) & (df['site']<=cdr_bounds[chain][0][1])]
                    region_size[region] = cdr_bounds[chain][0][1] - cdr_bounds[chain][0][0] + 1
                elif region == 'FWR2':
                    region_df = df[(df['site']>cdr_bounds[chain][0][1]) & (df['site']<cdr_bounds[chain][1][0])]
                    region_size[region] = cdr_bounds[chain][1][0] - cdr_bounds[chain][0][1] - 1
                elif region == 'CDR2':
                    region_df = df[(df['site']>=cdr_bounds[chain][1][0]) & (df['site']<=cdr_bounds[chain][1][1])]
                    region_size[region] = cdr_bounds[chain][1][1] - cdr_bounds[chain][1][0] + 1
                elif region == 'FWR3':
                    region_df = df[(df['site']>cdr_bounds[chain][1][1]) & (df['site']<cdr_bounds[chain][2][0])]
                    region_size[region] = cdr_bounds[chain][2][0] - cdr_bounds[chain][1][1] - 1
                elif region == 'CDR3':
                    region_df = df[(df['site']>=cdr_bounds[chain][2][0]) & (df['site']<=cdr_bounds[chain][2][1])]
                    region_size[region] = cdr_bounds[chain][2][1] - cdr_bounds[chain][2][0] + 1
                elif region == 'FWR4':
                    region_df = df[df['site']>cdr_bounds[chain][2][1]]
                    region_size[region] = df.shape[0] - cdr_bounds[chain][2][1]
                else:
                    region_df = df
                    region_size[region] = df.shape[0]
                
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
    for region in ['All','FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4']:
        print(region, np.median(region_k_subs[region]), np.mean(region_k_subs[region]), region_size[region], np.mean(region_k_subs[region])/region_size[region])
