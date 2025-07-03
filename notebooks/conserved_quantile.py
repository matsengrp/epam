import numpy as np
import pandas as pd

conserved_xvals = ['23', '41', '43', '98', '102', '104', '118']

outfname = "tang_conserved_quantile.csv"
dsname = "tang"
dfs_dir = "dataframes"

model_list = [
    "S5F", "S5FESM_mask", "S5FBLOSUM",
    "ThriftyHumV0.2-59", "ThriftyProdHumV0.2-59", "ThriftyESM_mask", "ThriftyBLOSUM",
    "ESM1v_mask", "AbLang2_mask", "AbLang1"
]

modelname_list = [
    "S5F", "S5F + ESM-1v", "S5F + BLOSUM62",
    "Thrifty", "Thrifty-prod", "Thrifty + ESM-1v", "Thrifty + BLOSUM62",
    "ESM-1v", "AbLang2", "AbLang1"
]

output_df = pd.DataFrame(columns=(["model"] + conserved_xvals))

output_data={}
for colname in output_df.columns:
    output_data[colname] = []

print("Dataset:", dsname)
for model, modname in zip(model_list, modelname_list):
    
    print("Processing",model)
    df = pd.read_csv(f'{dfs_dir}/{dsname}_{model}_ssp_df.csv.gz', index_col=0, dtype={'site':'object'})
    
    #df = df[(df['site']<111.5) | (df['site']>112.5)]
    
    sites_df = df[['site','prob','mutation']].groupby("site").sum()
    
    if len(output_data["model"])==0:
        data = sites_df['mutation'].to_numpy()
        output_data["model"].append("observed")
        for site in conserved_xvals:
            threshold = sites_df.loc[site]['mutation'].item()
            output_data[site].append(sum(data < threshold)/len(data))

    output_data["model"].append(modname)
    for site in conserved_xvals:
        threshold = sites_df.loc[site]['prob'].item()
        output_data[site].append(sum(data < threshold)/len(data))

for colname in output_df.columns:
    output_df[colname] = output_data[colname]

print(output_df)
output_df.to_csv(outfname,index=False)