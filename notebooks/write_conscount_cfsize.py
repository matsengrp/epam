
import h5py
import numpy as np
import pandas as pd
from epam.utils import load_and_filter_pcp_df
from netam.sequences import (
    translate_sequences,
)

epam_results_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2"
nsubsmax = 3

dataset = "tang-deepshm-prod_pcp_2024-08-08_MASKED_NI_noN_no-naive"
dsname = "tang"

conscount_df = pd.read_csv(f"{dsname}_conscount.csv",index_col=0,dtype={'seqname':'object'})
cfsize_df = pd.read_csv("/fh/fast/matsen_e/ksung2/pcp-pipeline/ASR/tang-deepshm/cfsize.csv")

outfname = f"{dsname}_nsubs_conscount_cfsize.csv"
output_df = pd.DataFrame(columns=['pcp_index','sample_id','family','nsubs','conscount','cfsize'])

pcp_path = "pcp_inputs/tang-deepshm-prod_pcp_2024-08-08_MASKED_NI_noN_no-naive.csv"
pcp_df = load_and_filter_pcp_df(pcp_path)
pcp_df = pcp_df[pcp_df['child_is_leaf']==True]


coldata={}
for colname in output_df.columns:
    coldata[colname] = []

for i,row in pcp_df.iterrows():
    parent = row['parent']
    child = row['child']
    
    parent_aa, child_aa = translate_sequences([parent, child])
    nsubs = sum([p!=c for p,c in zip(parent_aa, child_aa)])
    if nsubs<1 or nsubs>nsubsmax:
        continue
    
    conscount = conscount_df[conscount_df['seqname']==row['child_name']]['CONSCOUNT'].item()
    
    sample_id = row['sample_id']
    family = row['family']
    fname = f"tang-deepshm_{sample_id}_sizeAll-cluster-{family}.fa"
    cfsize = cfsize_df[cfsize_df['file']==fname]['size'].item() - 1
    
    coldata['pcp_index'].append(i)
    coldata['sample_id'].append(sample_id)
    coldata['family'].append(family)
    coldata['nsubs'].append(nsubs)
    coldata['conscount'].append(conscount)
    coldata['cfsize'].append(cfsize)


for colname in output_df.columns:
    output_df[colname] = coldata[colname]
output_df.to_csv(outfname,index=False)
