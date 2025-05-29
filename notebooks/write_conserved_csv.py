# Write a dataframe of observed and expected number of substitutions
# at known conserved sites for several models
# (Figure 3)

import numpy as np
import pandas as pd

conserved_xvals = ['23', '41', '43', '98', '102', '104', '118']

dsname = "tang"
outfname = f"{dsname}_conserved.csv"
inputs = [
    (f"dataframes/{dsname}_S5F_ssp_df.csv.gz", "S5F"),
    (f"dataframes/{dsname}_ThriftyHumV0.2-59_ssp_df.csv.gz", "Thrifty-SHM"),
    (f"dataframes/{dsname}_ESM1v_mask_ssp_df.csv.gz", "ESM-1v"),
    (f"dataframes/{dsname}_ThriftyESM_mask_ssp_df.csv.gz", "Thrifty-SHM + ESM-1v"),
    (f"dataframes/{dsname}_AbLang2_mask_ssp_df.csv.gz", "AbLang2"),
    (f"dataframes/{dsname}_ThriftyProdHumV0.2-59_ssp_df.csv.gz", "Thrifty-prod"),    
]


output_df = pd.DataFrame()
output_df['site'] = conserved_xvals

observed = []

for info in inputs:
    infile = info[0]
    colname = info[1]
    
    print("Processing",infile)
    df = pd.read_csv(infile, index_col=0, dtype={'site':'object'})

    coldata = []
    colerr = []
    for site in conserved_xvals:
        if "observed" not in output_df.columns:
            obs = df[(df["mutation"] == 1) & (df["site"] == site)].shape[0]
            observed.append(obs)
            
        site_probs = df[df["site"] == site]["prob"].to_numpy()
        exp = np.sum(site_probs)
        err = np.sqrt(np.sum(site_probs * (1 - site_probs)))
        coldata.append(exp)
        colerr.append(err)
        
    if "observed" not in output_df.columns:
        output_df['observed'] = observed
        
    output_df[colname] = coldata
    output_df[f'{colname}_err'] = colerr

print(output_df)
output_df.to_csv(outfname,index=False)
