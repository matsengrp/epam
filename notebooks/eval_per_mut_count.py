import h5py
import pandas as pd
import numpy as np
from netam.sequences import AA_STR_SORTED
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from netam.sequences import translate_sequences
from epam.evaluation import (
    calculate_site_substitution_probabilities,
    highest_k_substitutions,
    locate_top_k_substitutions,
    locate_child_substitutions,
    identify_child_substitutions
)


dataset_list = [
("/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2/tang-deepshm-prod_pcp_2024-08-08_MASKED_NI_noN_no-naive/ThriftyProdHumV0.2-59/combined_aaprob.hdf5",
 'tang_thriftyprod_eval.csv'),

("/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2/gcreplay/igh/gctrees_2025-01-10-full_igh_pcp_NoBackMuts/GCReplaySHMDMSSigmoid_igh/aaprob.hdf5",
 'gcreplay_igh_shmdms_eval.csv'),

("/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2/gcreplay/igk/gctrees_2025-01-10-full_igk_pcp_NoBackMuts/GCReplaySHMDMSSigmoid_igk/aaprob.hdf5",
 'gcreplay_igk_shmdms_eval.csv'),
]

for dsinfo in dataset_list:
    aaprob_path = dsinfo[0]
    outfname = dsinfo[1]

    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    print(pcp_path)

    pcp_df = load_and_filter_pcp_df(pcp_path)
    print(pcp_df.shape)

    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))

    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]

    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    #
    # Precalculations for performance metrics
    #
    pcp_sub_locations = [
        locate_child_substitutions(parent, child)
        for parent, child in zip(parent_aa_seqs, child_aa_seqs)
    ]

    pcp_sub_aa_ids = [
        identify_child_substitutions(parent, child)
        for parent, child in zip(parent_aa_seqs, child_aa_seqs)
    ]

    # k represents the number of substitutions observed in each PCP, top k substitutions will be evaluated for r-precision
    k_subs = [len(pcp_sub_location) for pcp_sub_location in pcp_sub_locations]

    site_sub_probs = []
    model_sub_aa_ids = []

    with h5py.File(aaprob_path, "r") as matfile:
        model_name = matfile.attrs["model_name"]
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            site_sub_probs.append(
                calculate_site_substitution_probabilities(matrix, parent_aa_seqs[index])
            )

            pred_aa_sub = [
                highest_k_substitutions(1, matrix[j, :], parent_aa_seqs[index], j)[0]
                for j in range(len(parent_aa_seqs[index]))
                if parent_aa_seqs[index][j] != child_aa_seqs[index][j]
            ]

            model_sub_aa_ids.append(pred_aa_sub)
            
    top_k_sub_locations = [
        locate_top_k_substitutions(site_sub_prob, k_sub)
        for site_sub_prob, k_sub in zip(site_sub_probs, k_subs)
    ]


    aa_mut_count_data = []
    aa_seq_length_data = []
        
    for pc in aa_seqs:
        parent = pc[0]
        child = pc[1]
        
        aa_mut_count = sum([p!=c for p,c in zip(list(parent), list(child))])
        aa_mut_count_data.append(aa_mut_count)
        aa_seq_length_data.append(len(parent))


    correct_site_predictions = [
        np.intersect1d(pcp_sub_location, top_k_sub_location)
        for pcp_sub_location, top_k_sub_location in zip(
            pcp_sub_locations, top_k_sub_locations
        )
    ]

    k_subs_correct = [
        len(correct_site_prediction)
        for correct_site_prediction in correct_site_predictions
    ]

    pcp_r_precision = [
        k_correct / k_total if k_total > 0 else -1 for k_correct, k_total in zip(k_subs_correct, k_subs)
    ]

    num_sub_correct = [
        np.sum(pcp_sub_aa_ids[i] == model_sub_aa_ids[i])
        for i in range(len(model_sub_aa_ids))
    ]


    print(len(aa_mut_count_data))
    print(len(aa_seq_length_data))
    print(len(pcp_r_precision))
    print(len(num_sub_correct))
    print(len(k_subs))

    print(sum([r for r in pcp_r_precision if r>-1])/len([r for r in pcp_r_precision if r>-1]))
    print(sum(num_sub_correct)/sum(k_subs))

    df = pd.DataFrame()
    df['mut_count'] = aa_mut_count_data
    df['seq_length'] = aa_seq_length_data
    df['r_prec'] = pcp_r_precision
    df['n_subs_correct'] = num_sub_correct
    df['n_subs'] = k_subs
    df.to_csv(outfname)
