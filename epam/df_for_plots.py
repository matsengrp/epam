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
from netam.sequences import (
    AA_STR_SORTED,
    translate_sequences,
    translate_sequence,
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


def get_site_substitutions_df(
    subs_and_preds_tuple,
    numbering_dict=None,
):
    """
    Returns a dataframe that annotates for each site of observed and/or predicted substitution:
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    whether the site has an observed substutition,
    and whether the site is predicted to have a substitution.

    Parameters:
    subs_and_preds_tuple (tuple): 4-tuple of
                                  pcp_indices - list of indices to the reference PCP file,
                                  pcp_sub_locations - list of per-PCP lists of substitution locations,
                                  top_k_sub_locations - list of per-PCP lists of top-k mutability locations,
                                  pcp_sample_family_dict - dictionary mapping PCP index to (sample_id, family) 2-tuple.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with columns 'pcp_index', 'site', 'obs', 'pred'.
    """
    pcp_indices = subs_and_preds_tuple[0]
    pcp_sub_locations = subs_and_preds_tuple[1]
    top_k_sub_locations = subs_and_preds_tuple[2]
    pcp_sample_family_dict = subs_and_preds_tuple[3]

    pcp_index_col = []
    site_col = []
    obs_col = []
    pred_col = []
    for i in range(len(pcp_indices)):
        obs_pred_sites = np.union1d(top_k_sub_locations[i], pcp_sub_locations[i])

        nbkey = pcp_sample_family_dict[pcp_indices[i]]
        if (numbering_dict is not None) and (nbkey not in numbering_dict):
            continue

        for site in obs_pred_sites:
            pcp_index_col.append(pcp_indices[i])
            if numbering_dict is None:
                site_col.append(site)
            else:
                site_col.append(numbering_dict[nbkey][site])
            obs_col.append(site in pcp_sub_locations[i])
            pred_col.append(site in top_k_sub_locations[i])

    output_df = pd.DataFrame(columns=["pcp_index", "site", "obs", "pred"])
    output_df["pcp_index"] = pcp_index_col
    output_df["site"] = site_col
    output_df["obs"] = obs_col
    output_df["pred"] = pred_col

    return output_df


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


def get_site_subs_acc_df(
    sub_acc_tuple,
    numbering_dict=None,
):
    """
    Returns a dataframe that annotates for each site of observed substitution:
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    whether the predicted amino acid substitution is correct.

    Parameters:
    subs_and_preds_tuple (tuple): 5-tuple of
                                  pcp_indices - list of indices to the reference PCP file,
                                  pcp_sub_locations - list of per-PCP lists of substitution locations,
                                  pcp_sub_correct - list of per-PCP lists whether amino acid prediction is correct,
                                  pcp_is_cdr = list of per-PCP lists whether amino acid site is in a CDR,
                                  pcp_sample_family_dict - dictionary mapping PCP index to (sample_id, family) 2-tuple.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with columns 'pcp_index', 'site', 'correct', 'is_cdr'.
    """

    pcp_indices = sub_acc_tuple[0]
    pcp_sub_locations = sub_acc_tuple[1]
    pcp_sub_correct = sub_acc_tuple[2]
    pcp_is_cdr = sub_acc_tuple[3]
    pcp_sample_family_dict = sub_acc_tuple[4]

    pcp_index_col = []
    site_col = []
    pred_col = []
    is_cdr_col = []
    for i in range(len(pcp_indices)):
        nbkey = pcp_sample_family_dict[pcp_indices[i]]
        if (numbering_dict is not None) and (nbkey not in numbering_dict):
            continue

        for j in range(len(pcp_sub_locations[i])):
            site = pcp_sub_locations[i][j]
            pcp_index_col.append(pcp_indices[i])
            if numbering_dict is None:
                site_col.append(site)
            else:
                site_col.append(numbering_dict[nbkey][site])
            pred_col.append(pcp_sub_correct[i][j])
            is_cdr_col.append(pcp_is_cdr[i][j])

    output_df = pd.DataFrame(columns=["pcp_index", "site", "correct", "is_cdr"])
    output_df["pcp_index"] = pcp_index_col
    output_df["site"] = site_col
    output_df["correct"] = pred_col
    output_df["is_cdr"] = is_cdr_col

    return output_df


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


def get_numbering_dict(anarci_path, pcp_df=None, verbose=False, checks="imgt"):
    """
    Process ANARCI output to make site numbering lists for each clonal family.

    Parameters:
    anarci_path (str): path to ANARCI output for sequence numbering.
    pcp_df (pd.Dataframe): PCP file to filter for relevant clonal families and check ANARCI sequence lengths.
    verbose (bool): whether to print (sample ID, family ID) info when clonal family is excluded.
    checks (str): perform checks and updates for a specified numbering scheme.
                  Currently, 'imgt' is the only input that has an effect.

    Returns:
    Two dictionaries where the keys are 2-tuples of (sample_id, family).
    The first dictionary have values that are lists of numberings for each site in the clonal family.
    Note that the numberings are lists of strings.
    The dictionary also has an entry with key ('reference', 0) and value the list of all site numberings,
    to be used for setting x-axis tick labels when plotting.
    The second dictionary consists of clonal families that excluded from the first due to issues with the ANARCI output.
    The values describe the reasons for exclusion.
    """
    if anarci_path is None:
        return None

    numbering_dict = {}
    exclusion_dict = {}

    anarci_df = pd.read_csv(anarci_path)

    # assumes numbering starts at column 13 in ANARCI output
    numbering_cols = list(anarci_df.columns[13:])
    numbering_used = [False] * len(numbering_cols)

    for i, row in anarci_df.iterrows():
        # assumes clonal family ID has format "{sample_id}|{family}|{seq_name}"
        cfinfos = row["Id"].split("|")
        sample_id = cfinfos[0]
        # try-block to cast family ID to integer if appropriate
        try:
            int(cfinfos[1])
        except ValueError:
            family = cfinfos[1]
        else:
            family = int(cfinfos[1])

        seqlist = [row[col] for col in numbering_cols]
        numbering = [nn for nn, aa in zip(numbering_cols, seqlist) if aa != "-"]

        if checks == "imgt":
            # For IMGT, numbered insertions can only be 111.* or 112.*.
            # Other numbered insertions come from ANARCI and the clonal family will be excluded
            exclude = False
            for nn in numbering:
                if "." in nn and nn[:3] != "111" and nn[:3] != "112":
                    exclusion_dict[(sample_id, family)] = (
                        f"Invalid IMGT insertion: {nn}"
                    )
                    if verbose == True:
                        print(f"Invalid IMGT insertion: {nn}", sample_id, family)
                    exclude = True
                    break
            if exclude == True:
                continue

        if pcp_df is not None:
            # Check if clonal family is in PCP file, and that ANARCI preserved sequence length.
            # If not, exclude clonal family from output.
            test_df = pcp_df[
                (pcp_df["sample_id"] == sample_id) & (pcp_df["family"] == family)
            ]
            if test_df.shape[0] == 0:
                continue
            else:
                pcp_row = test_df.iloc[0]
                test_seq = translate_sequence(pcp_row["parent"])
                if len(test_seq) != len(numbering):
                    exclusion_dict[(sample_id, family)] = "ANARCI seq length mismatch!"
                    if verbose == True:
                        print("ANARCI seq length mismatch!", sample_id, family)
                    continue

                if checks == "imgt":
                    # Check CDR annotation in PCP file is consistent with IMGT numbering.
                    # If not, exclude the clonal family.
                    cdr_anno = pcp_sites_cdr_annotation(pcp_row)

                    exclude = False
                    for nn, is_cdr in zip(numbering, cdr_anno):
                        if is_imgt_cdr(nn) != is_cdr:
                            exclusion_dict[(sample_id, family)] = (
                                "IMGT mismatch with CDR annotation!"
                            )
                            if verbose == True:
                                print(
                                    "IMGT mismatch with CDR annotation!",
                                    sample_id,
                                    family,
                                )
                            exclude = True
                            break
                    if exclude == True:
                        continue

        numbering_dict[(sample_id, family)] = numbering

        # keep track of which site numbers are used
        for nn in numbering:
            numbering_used[numbering_cols.index(nn)] = True

    # make a numbering reference of site numbers that are used
    numbering_dict[("reference", 0)] = [
        nn for nn, used in zip(numbering_cols, numbering_used) if used == True
    ]

    return (numbering_dict, exclusion_dict)


def is_imgt_cdr(site):
    """
    Determines whether an amino acid site is in a CDR according to IMGT numbering.

    Parameters:
    site (str): IMGT number of an amino acid site.

    Returns:
    True or False whether the site is in a CDR.
    """
    IMGT_CDR1 = (27, 38)
    IMGT_CDR2 = (56, 65)
    IMGT_CDR3 = (105, 117)

    # Note: IMGT uses decimals for insertions (e.g. '111.3')
    if "." in site:
        sitei = int(site.split(".")[0])
    else:
        sitei = int(site)

    return (
        (sitei >= IMGT_CDR1[0] and sitei <= IMGT_CDR1[1])
        or (sitei >= IMGT_CDR2[0] and sitei <= IMGT_CDR2[1])
        or (sitei >= IMGT_CDR3[0] and sitei <= IMGT_CDR3[1])
    )


def pcp_sites_cdr_annotation(pcp_row):
    """
    Annotations for CDR or not for all sites in a PCP.

    Parameters:
    pcp_row (pd.Series): A row from the corresponding PCP file

    Returns:
    An (ordered) list of booleans for whether each site is in the CDR or not. The list is the same length as the sequence.
    """
    cdr1 = (
        pcp_row["cdr1_codon_start"] // 3,
        pcp_row["cdr1_codon_end"] // 3,
    )
    cdr2 = (
        pcp_row["cdr2_codon_start"] // 3,
        pcp_row["cdr2_codon_end"] // 3,
    )
    cdr3 = (
        pcp_row["cdr3_codon_start"] // 3,
        pcp_row["cdr3_codon_end"] // 3,
    )

    return [
        (
            True
            if (i >= cdr1[0] and i <= cdr1[1])
            or (i >= cdr2[0] and i <= cdr2[1])
            or (i >= cdr3[0] and i <= cdr3[1])
            else False
        )
        for i in range(len(pcp_row["parent"]) // 3)
    ]
