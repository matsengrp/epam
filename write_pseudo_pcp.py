import numpy as np
import pandas as pd
from Bio.Seq import Seq

ss_entropy = 98765
nseeds = 100

naive_seqs ={
    'igh' : 'GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGCAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCA',
    'igk' : 'GACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCAGCAATATAACAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA'
}

infnames = {
    'igh' : 'pcp_gcreplay_inputs/igh/gctrees_2022-06-16_igh_pcp_NoBackMuts.csv',
    'igk' : 'pcp_gcreplay_inputs/igk/gctrees_2022-06-16_igk_pcp_NoBackMuts.csv'
}


for chain in ['igh', 'igk']:
    print(chain)
    nstops=0
    kk=0
    for seed in np.random.SeedSequence(ss_entropy).generate_state(nseeds):
        if kk % 10 == 0:
            print(kk)
        kk += 1
        
        df = pd.read_csv(infnames[chain], index_col=0)
        rng = np.random.default_rng(seed)
        
        for i,row in df.iterrows():
            parent = row['parent']
            child = row['child']
            child_nts = list(parent)
            nmuts = sum([p!=c for p,c in zip(parent, child)])
            muts_sites = rng.choice(len(parent), nmuts, replace=False)
            for site in muts_sites:
                naive_nt = naive_seqs[chain][site]
                parent_nt = parent[site]
                nt_pool = [nt for nt in list('ACGT') if nt!=naive_nt and nt!=parent_nt]
                child_nts[site] = rng.choice(nt_pool,1)[0]
            new_child = ''.join(child_nts)
            if '*' in Seq.translate(new_child):
                nstops += 1
                continue
            df.loc[i,'child'] = new_child

        df.to_csv(f'pcp_gcreplay_pseudo_inputs/{chain}/{chain}_pcp_{seed}.csv')
    print('N stops:',nstops)