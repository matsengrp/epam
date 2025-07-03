import numpy as np
import pandas as pd

datadir = '/fh/fast/matsen_e/data/tang-deepshm/from_our_db/post_mod'
samples_list = [
    'AG391',
    'AG445',
    'AG476',
    'B10',
    'B11',
    'B12',
    'B13',
    'B14',
    'B16',
    'B17',
    'B18',
    'B19',
    'B20',
    'B21',
    'CLL1374',
    'CLL1437',
    'CLL1697',
    'CLL1729',
    'CLL1790',
    'CLL2056',
    'CLL2057'
]

colnames = ['sample_id','seqname','CONSCOUNT','DUPCOUNT']
coldata = {}
for col in colnames:
    coldata[col] = []

for sam in samples_list:
    print(sam)
    df = pd.read_csv(f'{datadir}/{sam}_ourDB_mod_db-pass_clone-pass_celltype-pass.tsv',delimiter='\t')
    for i,row in df.iterrows():
        coldata['sample_id'].append(sam)
        coldata['seqname'].append(row['SEQUENCE_ID'])
        coldata['CONSCOUNT'].append(row['CONSCOUNT'])
        coldata['DUPCOUNT'].append(row['DUPCOUNT'])

output_df = pd.DataFrame(columns=colnames)
for col in colnames:
    output_df[col] = coldata[col]
output_df.to_csv('tang_conscount.csv')
