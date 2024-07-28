"""Code for evaluating model performance."""

import h5py
import pandas as pd
import numpy as np
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.annotate_pcps import get_cdr_fwk_seqs
from netam.common import SMALL_PROB
from netam.sequences import (
    AA_STR_SORTED,
    translate_sequences,
    translate_sequence,
    generic_mutation_frequency,
)


def evaluate(aaprob_paths, model_performance_path):
    """
    Wrapper function for evaluate_dataset() that takes in a list of aaprob matrices and outputs a CSV of model performance metrics.
    Outputs to CSV file with columns for the different metrics and a row per model/data set combo.

    Parameters:
    aaprob_paths (list): List of paths to evaluate. Each aaprob matrix corresponds to predictions for one model on a given data set.
    model_performance_path (str): Path to output for model performance metrics.

    """
    model_performances = [evaluate_dataset(aaprob_path) for aaprob_path in aaprob_paths]

    all_model_performances = pd.DataFrame(model_performances)

    all_model_performances.to_csv(model_performance_path, index=False)


def evaluate_dataset(aaprob_path):
    """
    Evaluate model predictions against reality for a set of parent-child pairs (PCPs).
    Function is model-agnositic and currently calculates substitution accuracy, r-precision, and cross entropy loss.
    All metrics are reported for the full sequence, as well as for the framework and CDR regions separately.
    Returns evaluation metrics for a single aaprob matrix (generated from one model on one data set).

    Parameters:
    aaprob_path (str): Path to aaprob matrix for parent-child pairs.

    Returns:
    model_performance (dict): Dictionary of model performance metrics for a single aaprob matrix.

    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)

    pcp_df = load_and_filter_pcp_df(pcp_path)

    pcp_df["parent_aa"] = pcp_df.apply(
        lambda row: translate_sequence(row["parent"]), axis=1
    )
    pcp_df["child_aa"] = pcp_df.apply(
        lambda row: translate_sequence(row["child"]), axis=1
    )
    (
        pcp_df["parent_fwk_seq"],
        pcp_df["parent_cdr_seq"],
        pcp_df["child_fwk_seq"],
        pcp_df["child_cdr_seq"],
    ) = zip(*pcp_df.apply(get_cdr_fwk_seqs, axis=1))

    region_parent_aa_seqs = {}
    region_child_aa_seqs = {}
    region_sub_locations = {}
    region_sub_aa_ids = {}
    region_k_subs = {}

    for region in ["full", "fwk", "cdr"]:
        if region == "full":
            region_parent_aa_seqs[region] = pcp_df["parent_aa"].to_numpy()
            region_child_aa_seqs[region] = pcp_df["child_aa"].to_numpy()
        else:
            region_parent_aa_seqs[region] = pcp_df[f"parent_{region}_seq"].to_numpy()
            region_child_aa_seqs[region] = pcp_df[f"child_{region}_seq"].to_numpy()

        region_sub_locations[region] = [
            locate_child_substitutions(parent, child)
            for parent, child in zip(
                region_parent_aa_seqs[region], region_child_aa_seqs[region]
            )
        ]

        region_sub_aa_ids[region] = [
            identify_child_substitutions(parent, child)
            for parent, child in zip(
                region_parent_aa_seqs[region], region_child_aa_seqs[region]
            )
        ]

        # k represents the number of substitutions observed in each PCP, top k substitutions will be evaluated for r-precision
        region_k_subs[region] = [
            len(sub_location) for sub_location in region_sub_locations[region]
        ]

    region_site_sub_probs = {}
    region_model_sub_aa_ids = {}

    with h5py.File(aaprob_path, "r") as matfile:
        model_name = matfile.attrs["model_name"]
        for region in ["full", "fwk", "cdr"]:
            region_site_sub_probs[region] = []
            region_model_sub_aa_ids[region] = []

            for index in range(len(region_parent_aa_seqs[region])):
                pcp_index = pcp_df.index[index]
                grp = matfile[
                    "matrix" + str(pcp_index)
                ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
                matrix = grp["data"]

                region_site_sub_probs[region].append(
                    calculate_site_substitution_probabilities(
                        matrix, region_parent_aa_seqs[region][index]
                    )
                )

                def find_highest_ranked_substitutions(matrix, parent, child):
                    return [
                        highest_ranked_substitution(matrix[j, :], parent, j)
                        for j in range(len(parent))
                        if parent[j] != child[j]
                    ]

                region_model_sub_aa_ids[region].append(
                    find_highest_ranked_substitutions(
                        matrix,
                        region_parent_aa_seqs[region][index],
                        region_child_aa_seqs[region][index],
                    )
                )

    region_top_k_sub_locations = {}
    region_sub_acc = {}
    region_r_prec = {}
    region_cross_ent = {}
    region_aa_sub_freq = {}

    for region in ["full", "fwk", "cdr"]:
        region_top_k_sub_locations[region] = [
            locate_top_k_substitutions(region_site_sub_prob, k_sub)
            for region_site_sub_prob, k_sub in zip(
                region_site_sub_probs[region], region_k_subs[region]
            )
        ]

        region_sub_acc[region] = calculate_sub_accuracy(
            region_sub_aa_ids[region],
            region_model_sub_aa_ids[region],
            region_k_subs[region],
        )
        region_r_prec[region] = calculate_r_precision(
            region_sub_locations[region],
            region_top_k_sub_locations[region],
            region_k_subs[region],
        )
        region_cross_ent[region] = calculate_cross_entropy_loss(
            region_sub_locations[region], region_site_sub_probs[region]
        )

        if region == "full":
            region_aa_sub_freq[region] = [
                generic_mutation_frequency("X", parent, child)
                for parent, child in zip(
                    region_parent_aa_seqs[region], region_child_aa_seqs[region]
                )
            ]
        else:
            parent_only_aa_seqs = [
                seq.replace("-", "") for seq in region_parent_aa_seqs[region]
            ]
            child_only_aa_seqs = [
                seq.replace("-", "") for seq in region_child_aa_seqs[region]
            ]
            region_aa_sub_freq[region] = [
                calculate_aa_substitution_frequencies_by_region(parent, child)
                for parent, child in zip(parent_only_aa_seqs, child_only_aa_seqs)
            ]

    model_performance = {
        "data_set": pcp_path,
        "pcp_count": len(pcp_df),
        "model": model_name,
        "sub_accuracy": region_sub_acc["full"],
        "r_precision": region_r_prec["full"],
        "cross_entropy": region_cross_ent["full"],
        "fwk_sub_accuracy": region_sub_acc["fwk"],
        "fwk_r_precision": region_r_prec["fwk"],
        "fwk_cross_entropy": region_cross_ent["fwk"],
        "cdr_sub_accuracy": region_sub_acc["cdr"],
        "cdr_r_precision": region_r_prec["cdr"],
        "cdr_cross_entropy": region_cross_ent["cdr"],
        "avg_k_subs": np.mean(region_k_subs["full"]),
        "avg_aa_sub_freq": np.mean(region_aa_sub_freq["full"]),
        "aa_sub_freq_range": (
            np.min(region_aa_sub_freq["full"]),
            np.max(region_aa_sub_freq["full"]),
        ),
        "fwk_avg_k_subs": np.mean(region_k_subs["fwk"]),
        "fwk_avg_aa_sub_freq": np.mean(region_aa_sub_freq["fwk"]),
        "fwk_aa_sub_freq_range": (
            np.min(region_aa_sub_freq["fwk"]),
            np.max(region_aa_sub_freq["fwk"]),
        ),
        "cdr_avg_k_subs": np.mean(region_k_subs["cdr"]),
        "cdr_avg_aa_sub_freq": np.mean(region_aa_sub_freq["cdr"]),
        "cdr_aa_sub_freq_range": (
            np.min(region_aa_sub_freq["cdr"]),
            np.max(region_aa_sub_freq["cdr"]),
        ),
    }

    return model_performance


def locate_child_substitutions(parent_aa, child_aa):
    """
    Return the location of the amino acid substitutions for a given parent-child pair.

    Parameters:
    parent_aa (str): Amino acid sequence of parent.
    child_aa (str): Amino acid sequence of child.

    Returns:
    child_sub_sites (np.array): Location of substitutions in parent-child pair.

    """
    child_sub_sites = [i for i in range(len(parent_aa)) if parent_aa[i] != child_aa[i]]

    child_sub_sites = np.array(child_sub_sites)

    return child_sub_sites


def identify_child_substitutions(parent_aa, child_aa):
    """
    Return the identity of the amino acid substitutions for a given parent-child pair.

    Parameters:
    parent_aa (str): Amino acid sequence of parent.
    child_aa (str): Amino acid sequence of child.

    Returns:
    child_aa_subs (np.array): Amino acid substitutions in parent-child pair.

    """
    child_aa_subs = [
        child_aa[i] for i in range(len(parent_aa)) if parent_aa[i] != child_aa[i]
    ]

    child_aa_subs = np.array(child_aa_subs)

    return child_aa_subs


def calculate_site_substitution_probabilities(aaprobs, parent_aa):
    """
    Calculate the probability of substitution at each site for a parent sequence.

    Parameters:
    aaprobs (np.ndarray): A 2D array containing the normalized probabilities of the amino acids by site for a parent sequence.
    parent_aa (str): Amino acid sequence of parent.

    Returns:
    site_sub_probs (np.array): 1D array containing probability of substitution at each site for a parent sequence.

    """
    site_sub_probs = []

    for i in range(len(parent_aa)):
        # assign 0 probability of substitution to sites outside region of interest in CDR and FWK sequences
        if parent_aa[i] == "-":
            site_sub_probs.append(0.0)
        else:
            sub_prob = np.sum(
                aaprobs[i, :][
                    [
                        AA_STR_SORTED.index(aa)
                        for aa in AA_STR_SORTED
                        if aa != parent_aa[i]
                    ]
                ]
            )
            site_sub_probs.append(sub_prob)

    site_sub_probs = np.array(site_sub_probs)

    assert site_sub_probs.size == len(
        parent_aa
    ), "The number of substitution probabilities does not match the number of amino acid sites."

    return site_sub_probs


def highest_ranked_substitution(matrix_i, parent_aa, i):
    """
    Return the highest ranked substitution for site i in a given parent-child pair.

    Parameters:
    matrix_i (np.array): aaprob matrix for parent-child pair at aa site i.
    parent_aa (str): Parent amino acid sequence.
    i (int): Index of amino acid site substituted.

    Returns:
    pred_aa_sub (str): Predicted amino acid substitution (most likely non-parent aa).

    """
    prob_sorted_aa_indices = matrix_i.argsort()[::-1]

    pred_aa_ranked = "".join((np.array(list(AA_STR_SORTED))[prob_sorted_aa_indices]))

    # skip most likely aa if it is the parent aa (enforce substitution)
    if pred_aa_ranked[0] == parent_aa[i]:
        pred_aa_sub = pred_aa_ranked[1]
    elif pred_aa_ranked[0] != parent_aa[i]:
        pred_aa_sub = pred_aa_ranked[0]

    return pred_aa_sub


def locate_top_k_substitutions(site_sub_probs, k_sub):
    """
    Return the top k substitutions predicted for a parent-child pair given precalculated site substitution probabilities.

    Parameters:
    site_sub_probs (np.array): Probability of substition at each site for a parent sequence.
    k_sub (int): Number of substitutions observed in PCP.

    Returns:
    pred_sub_sites (np.array): Location of top-k predicted substitutions by model (unordered).

    """
    if k_sub == 0:
        return []

    # np.argpartition returns indices of top k elements in unsorted order
    pred_sub_sites = np.argpartition(site_sub_probs, -k_sub)[-k_sub:]

    assert (
        pred_sub_sites.size == k_sub
    ), "The number of predicted substitution sites does not match the number of actual substitution sites."

    return pred_sub_sites


def calculate_sub_accuracy(pcp_sub_aa_ids, model_sub_aa_ids, k_subs):
    """
    Calculate substitution accuracy for all PCPs in one data set/HDF5 file.
    Returns substitution accuracy score for use in evaluate() and output files.

    Parameters:
    pcp_aa_sub_ids (list of np.array): Amino acid substitutions in each PCP.
    model_sub_aa_ids (list of np.array): Amino acid substitutions predicted by model at substituted sites in each PCP.
    k_subs (list): Number of substitutions observed in each PCP.

    Returns:
    sub_accuracy (float): Calculated substitution accuracy for data set of PCPs.

    """
    num_sub_correct = [
        np.sum(pcp_sub_aa_ids[i] == model_sub_aa_ids[i])
        for i in range(len(model_sub_aa_ids))
    ]

    sub_accuracy = sum(num_sub_correct) / sum(k_subs)

    return sub_accuracy


def calculate_r_precision(pcp_sub_locations, top_k_sub_locations, k_subs):
    """
    Calculate r-precision for all PCPs in one data set/HDF5 file.
    Returns r-precision score for use in evaluate() and output files.

    Parameters:
    pcp_sub_locations (list of np.array): Location of substitutions in parent-child pairs.
    top_k_sub_locations (list of np.array): Location of top-k predicted substitutions by model (unordered for each PCP).
    k_subs (list): Number of substitutions observed in each PCP.

    Returns:
    r_precision (float): Calculated r-precision for data set of PCPs.

    """
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
        k_correct / k_total
        for k_correct, k_total in zip(k_subs_correct, k_subs)
        if k_total > 0
    ]

    r_precision = sum(pcp_r_precision) / len(pcp_r_precision)

    return r_precision


def calculate_cross_entropy_loss(pcp_sub_locations, site_sub_probs):
    """
    Calculate cross entropy loss for all PCPs in one data set/HDF5 file.

    Parameters:
    pcp_sub_locations (list of np.array): Location of substitutions in parent-child pairs.
    site_sub_probs (list of np.array): Probability of substition at each site for parent sequences.

    Returns:
    cross_entropy_loss (float): Calculated cross entropy loss for data set of PCPs.

    """
    log_probs_substitution = []
    for i in range(len(site_sub_probs)):
        if pcp_sub_locations[i].size > 0:
            assert any(
                pcp_sub_locations[i][j] < len(site_sub_probs[i])
                for j in range(len(pcp_sub_locations[i]))
            ), "The location of a substitution is greater than the number of sites in the parent sequence."
        for idx, p_i in np.ndenumerate(site_sub_probs[i]):
            if idx in pcp_sub_locations[i]:
                log_probs_substitution.append(np.log(p_i if p_i != 0 else SMALL_PROB))
            else:
                log_probs_substitution.append(
                    np.log(1 - p_i) if p_i != 1 else np.log(1 - SMALL_PROB)
                )

    cross_entropy_loss = (
        -1 / len(log_probs_substitution) * np.sum(log_probs_substitution)
    )

    return cross_entropy_loss


def calculate_aa_substitution_frequencies_by_region(parent_aa, child_aa):
    """
    Calculate the fraction of sites that differ between the parent and child FWK or CDR sequences.

    Parameters:
    parent_aa (str): Amino acid sequence of parent. FWK sequences will have CDR sites masked with '-' and vice versa.
    child_aa (str): Amino acid sequence of child.

    Returns:
    aa_sub_frequency (float): Fraction of sites that differ between the parent and child FWK or CDR sequences.

    """
    parent = parent_aa.replace("-", "")
    child = child_aa.replace("-", "")

    assert len(parent) == len(
        child
    ), "Parent and child FWK/CDR sequences must be the same length."

    aa_sub_frequency = sum(1 for p, c in zip(parent, child) if p != c) / len(parent)

    return aa_sub_frequency
