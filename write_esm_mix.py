import h5py
import numpy as np
import pandas as pd
from epam.esm_precompute import load_and_convert_to_dict


esm_versions = ['1','2','3','4','5']

for chain in ['igh','igk']:
    output_hdf5 = f"pcp_gcreplay_inputs/{chain}/gctrees_2022-06-16_{chain}_pcp_masked_esm_mix.hdf5"

    with h5py.File(f"pcp_gcreplay_inputs/{chain}/gctrees_2022-06-16_{chain}_pcp_masked_esm_1.hdf5", "r") as matfile:
        checksum = matfile.attrs['checksum']
        pcp_path = matfile.attrs['pcp_filename']
        model_name = matfile.attrs['model_name']

    matrices = {}
    for member in esm_versions:
        matrices[member] = load_and_convert_to_dict(f"pcp_gcreplay_inputs/{chain}/gctrees_2022-06-16_{chain}_pcp_masked_esm_{member}.hdf5")

    parents = matrices[esm_versions[0]].keys()

    with h5py.File(output_hdf5, "w") as outfile:
        # attributes related to PCP data file
        outfile.attrs["checksum"] = checksum
        outfile.attrs["pcp_filename"] = pcp_path
        outfile.attrs["model_name"] = model_name
        
        for seq in parents:
            mat_list = []
            for member in esm_versions:  
                mat_list.append(matrices[member][seq])
            matrix = sum(mat_list)/len(mat_list)
            
            outfile.create_dataset(
                f"{seq}", data=matrix, compression="gzip", compression_opts=4
            )
    
    print(f" <> {output_hdf5} created")
