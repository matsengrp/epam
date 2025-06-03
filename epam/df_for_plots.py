# Functions to create dataframes for various observed vs expected plots

import h5py
import numpy as np
import pandas as pd
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.evaluation import (
    highest_k_substitutions,
    locate_child_substitutions,
    calculate_site_substitution_probabilities,
    locate_top_k_substitutions,
    locate_child_substitutions,
    identify_child_substitutions,
)
from epam.oe_plot import(
    pcp_sites_cdr_annotation,
)
from netam.sequences import (
    AA_STR_SORTED,
    translate_sequences,
)


def get_site_mutabilities_df(
    aaprob_path,
    numbering_dict=None,
):
    """
    Computes the amino acid site mutability probabilities
    for every site of every parent in a dataset.
    Returns a dataframe that annotates for each site
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    whether a mutation is observed in the child sequence,
    and whether the site is in a CDR.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with columns pcp_index, site, prob, mutation, is_cdr.
    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    pcp_index_col = []
    sites_col = []
    site_sub_probs = []
    site_sub_flags = []
    is_cdr_col = []
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            pcp_row = pcp_df.loc[pcp_index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            parent = parent_aa_seqs[index]
            child = child_aa_seqs[index]

            if numbering_dict is None:
                sites_col.append(np.arange(len(parent)))
            else:
                nbkey = tuple(pcp_row[["sample_id", "family"]])
                if nbkey in numbering_dict:
                    sites_col.append(numbering_dict[nbkey])
                else:
                    continue

            pcp_index_col.append([pcp_index] * len(parent))
            site_sub_probs.append(
                calculate_site_substitution_probabilities(matrix, parent)
            )
            site_sub_flags.append([p != c for p, c in zip(parent, child)])
            is_cdr_col.append(pcp_sites_cdr_annotation(pcp_row))

    output_df = pd.DataFrame(
        columns=["pcp_index", "site", "prob", "mutation", "is_cdr"]
    )
    output_df["pcp_index"] = np.concatenate(pcp_index_col)
    output_df["site"] = np.concatenate(sites_col)
    output_df["prob"] = np.concatenate(site_sub_probs)
    output_df["mutation"] = np.concatenate(site_sub_flags)
    output_df["is_cdr"] = np.concatenate(is_cdr_col)

    return output_df


def get_subs_and_preds_from_aaprob(aaprob_path):
    """
    Determines the sites of observed and predicted substitutions of every PCP in a dataset,
    from the aaprob file of matrices.
    Predicted substitutions are the sites in the top-k of mutability,
    where k is the number of observed substition in the PCP.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.

    Returns tuple with:
    pcp_indices (list): indices to the reference PCP file.
    pcp_sub_locations (list): per-PCP lists of substitution locations (positions along the sequence string).
    top_k_sub_locations (list): per-PCP lists of top-k mutability locations (positions along the sequence string).
    pcp_sample_family_dict (dict): mapping PCP index to (sample_id, family) 2-tuple.
    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    pcp_sub_locations = [
        locate_child_substitutions(parent, child)
        for parent, child in zip(parent_aa_seqs, child_aa_seqs)
    ]

    # k represents the number of substitutions observed in each PCP, top k substitutions will be evaluated for r-precision
    k_subs = [len(pcp_sub_location) for pcp_sub_location in pcp_sub_locations]

    pcp_indices = []
    site_sub_probs = []
    pcp_sample_family_dict = {}
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            pcp_indices.append(pcp_index)

            site_sub_probs.append(
                calculate_site_substitution_probabilities(matrix, parent_aa_seqs[index])
            )

            pcp_sample_family_dict[pcp_index] = tuple(
                pcp_df.loc[pcp_index][["sample_id", "family"]]
            )

    top_k_sub_locations = [
        locate_top_k_substitutions(site_sub_prob, k_sub)
        for site_sub_prob, k_sub in zip(site_sub_probs, k_subs)
    ]

    return (pcp_indices, pcp_sub_locations, top_k_sub_locations, pcp_sample_family_dict)


def get_sub_acc_from_aaprob(aaprob_path, top_k=1):
    """
    Determines the sites of observed substitutions and whether the amino acid substitution is predicted
    among the top-k most probable for every PCP in a dataset, using the aaprob file of matrices.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.
    top_k (int): the number of top substitutions to consider for matching to observed.

    Returns tuple with:
    pcp_indices (list): indices to the reference PCP file.
    pcp_sub_locations (list): per-PCP lists of substitution locations (positions along the sequence string).
    pcp_sub_correct (list): per-PCP lists of True/False whether substitution prediction contains the correct amino acid.
    pcp_is_cdr (list): per-PCP lists of True/False whether amino acid site is in a CDR.
    pcp_sample_family_dict (dict): mapping PCP index to (sample_id, family) 2-tuple.
    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    pcp_sub_locations = [
        locate_child_substitutions(parent, child)
        for parent, child in zip(parent_aa_seqs, child_aa_seqs)
    ]

    pcp_indices = []
    pcp_sub_correct = []
    pcp_is_cdr = []
    pcp_sample_family_dict = {}
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            pcp_row = pcp_df.loc[pcp_index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            pcp_indices.append(pcp_index)

            parent_aa = parent_aa_seqs[index]
            child_aa = child_aa_seqs[index]
            pcp_sub_correct.append(
                [
                    child_aa[j]
                    in highest_k_substitutions(top_k, matrix[j, :], parent_aa, j)
                    for j in range(len(parent_aa))
                    if parent_aa[j] != child_aa[j]
                ]
            )

            cdr_anno = pcp_sites_cdr_annotation(pcp_row)
            pcp_is_cdr.append(
                [
                    cdr_anno[j]
                    for j in range(len(parent_aa))
                    if parent_aa[j] != child_aa[j]
                ]
            )

            pcp_sample_family_dict[pcp_index] = tuple(
                pcp_df.loc[pcp_index][["sample_id", "family"]]
            )

    return (
        pcp_indices,
        pcp_sub_locations,
        pcp_sub_correct,
        pcp_is_cdr,
        pcp_sample_family_dict,
    )


def get_site_csp_df(
    aaprob_path,
    numbering_dict=None,
):
    """
    Computes the site conditional substitution probabilities (CSP)
    for every substitution in a dataset.
    There are 20 CSPs for each site and each row in the output dataframe
    corresponds to a site-CSP pair.
    Returns a dataframe that annotates for each site-CSP
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    the amino acid of the CSP,
    whether a substitution to the amino acid is observed in the child sequence,
    and whether the site is in a CDR.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with columns pcp_index, site, prob, aa, mutation, is_cdr.
    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    pcp_index_col = []
    sites_col = []
    site_csp = []
    site_sub_flags = []
    site_aa_col = []
    is_cdr_col = []
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            pcp_row = pcp_df.loc[pcp_index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            parent = parent_aa_seqs[index]
            child = child_aa_seqs[index]

            pcp_subs_locations = locate_child_substitutions(parent, child)
            pcp_subs_aa = identify_child_substitutions(parent, child)

            if numbering_dict is None:
                numbering = np.arange(len(parent))
            else:
                nbkey = tuple(pcp_row[["sample_id", "family"]])
                if nbkey in numbering_dict:
                    numbering = numbering_dict[nbkey]
                else:
                    continue

            pcp_is_cdr = pcp_sites_cdr_annotation(pcp_row)

            for sub_loc, sub_aa in zip(pcp_subs_locations, pcp_subs_aa):
                csp = matrix[sub_loc, :]
                csp[AA_STR_SORTED.index(parent[sub_loc])] = 0
                csp = csp / np.sum(csp)

                pcp_index_col.append([pcp_index] * 20)
                sites_col.append([numbering[sub_loc]] * 20)
                site_csp.append(csp)
                site_aa_col.append(list(AA_STR_SORTED))
                site_sub_flags.append(
                    [True if aa == sub_aa else False for aa in AA_STR_SORTED]
                )
                is_cdr_col.append([pcp_is_cdr[sub_loc]] * 20)

    output_df = pd.DataFrame(
        columns=["pcp_index", "site", "prob", "mutation", "is_cdr"]
    )
    output_df["pcp_index"] = np.concatenate(pcp_index_col)
    output_df["site"] = np.concatenate(sites_col)
    output_df["prob"] = np.concatenate(site_csp)
    output_df["aa"] = np.concatenate(site_aa_col)
    output_df["mutation"] = np.concatenate(site_sub_flags)
    output_df["is_cdr"] = np.concatenate(is_cdr_col)

    return output_df
