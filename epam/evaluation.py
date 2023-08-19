"""Code for evaluating model performance."""

import h5py
import pandas as pd
from epam.sequences import aa_str_sorted
from epam.utils import pcp_path_of_prob_mat_path
from epam.models import *
from epam.sequences import translate_sequences


def evaluate(prob_mat_path, model_performance_path):
    """
    Evaluate model predictions against reality in parent-child pairs (PCPs).
    Function is model-agnositic and currently calculates substitution accuracy, r-precision, and cross loss entropy.
    Output to CSV with columns for the different metrics and a row per data set.

    Parameters:
    prob_mat_path (str): Path to probability matrix for parent-child pairs.
    model_performance_path (str): Path to output for model performance metrics.

    """
    pcp_path = pcp_path_of_prob_mat_path(prob_mat_path)

    sub_acc = calculate_sub_accuracy(prob_mat_path)

    pcp_df = pd.read_csv(pcp_path, index_col=0)

    nt_seqs = list(zip(pcp_df['parent'], pcp_df['child'])) # list of tuples

    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs] # list of tuples

    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs) # unzip list of tuples into two lists

    site_sub_probs = []

    with h5py.File(prob_mat_path, "r") as matfile:
        for i in range(len(pcp_df)):
            grp = matfile["matrix" + str(i)] # assumes that the keys are named "matrix0", "matrix1", etc.
            matrix = grp["data"]
            index = grp.attrs["pcp_index"]

            site_sub_probs.append(
                calculate_site_substitution_probabilities(
                    matrix, parent_aa_seqs[index]
                )
            )

    pcp_sub_locations = [locate_child_substitutions(parent, child) for parent, child in zip(parent_aa_seqs, child_aa_seqs)]

    k_subs = [len(pcp_sub_location) for pcp_sub_location in pcp_sub_locations]

    model_sub_locations = [locate_top_k_substitutions(site_sub_prob, k_sub) for site_sub_prob, k_sub in zip(site_sub_probs, k_subs)]

    correct_site_predictions = [np.intersect1d(pcp_sub_location, model_sub_location) for pcp_sub_location, model_sub_location in zip(pcp_sub_locations, model_sub_locations)]

    k_subs_correct = [len(correct_site_prediction) for correct_site_prediction in correct_site_predictions]

    r_prec = sum(k_subs_correct) / sum(k_subs)

    print("r-precision: ", r_prec)
    
    # r_prec = calculate_r_precision(prob_mat_path)
    
    cross_ent = None

    model_performance = pd.DataFrame(
        {
            "data_set": [pcp_path],
            "model": ["ablang"],  # Issue 8: hard coded for the moment
            "sub_accuracy": [sub_acc],
            "r_precision": [r_prec],
            "cross_entropy": [cross_ent],
        }
    )

    model_performance.to_csv(model_performance_path, index=False)


def calculate_sub_accuracy(prob_mat_path):
    """
    Calculate substitution accuracy for all PCPs in one data set/HDF5 file.
    Returns substitution accuracy score for use in evaluate() and output files.

    Parameters:
    prob_mat_path (str): Path to probability matrices for parent-child pairs.

    Returns:
    sub_accuracy (float): Calculated substitution accuracy for data set of PCPs.

    """
    with h5py.File(prob_mat_path, "r") as matfile:
        pcp_path = pcp_path_of_prob_mat_path(prob_mat_path)
        pcp_df = pd.read_csv(pcp_path, index_col=0)

        num_sub_total = 0
        num_sub_correct = 0

        for matname in matfile.keys():
            grp = matfile[matname]
            matrix = grp["data"]
            index = grp.attrs["pcp_index"]

            parent_nt, child_nt = pcp_df.loc[index, ["parent", "child"]]
            [parent_aa, child_aa] = translate_sequences([parent_nt, child_nt])

            assert len(parent_aa) == len(
                child_aa
            ), "The parent and child amino acid sequences are not the same length."

            for i in range(len(parent_aa)):
                if parent_aa[i] != child_aa[i]:
                    num_sub_total += 1

                    pred_aa_sub = highest_ranked_substitution(
                        matrix[:, i], parent_aa, i
                    )

                    if pred_aa_sub == child_aa[i]:
                        num_sub_correct += 1

        sub_accuracy = num_sub_correct / num_sub_total

        return sub_accuracy


def highest_ranked_substitution(matrix_i, parent_aa, i):
    """
    Return the highest ranked substitution for a given parent-child pair.

    Parameters:
    matrix_i (np.array): Probability matrix for parent-child pair at aa site i.
    parent_aa (str): Parent amino acid sequence.
    i (int): Index of amino acid site substituted.

    Returns:
    pred_aa_sub (str): Predicted amino acid substitution (most likely non-parent aa).

    """
    prob_sorted_aa_indices = matrix_i.argsort()[::-1]

    pred_aa_ranked = "".join((np.array(list(aa_str_sorted))[prob_sorted_aa_indices]))

    if pred_aa_ranked[0] == parent_aa[i]:
        pred_aa_sub = pred_aa_ranked[1]
    elif pred_aa_ranked[0] != parent_aa[i]:
        pred_aa_sub = pred_aa_ranked[0]

    return pred_aa_sub


def calculate_r_precision(prob_mat_path):
    """
    Calculate r-precision for all PCPs in one data set/HDF5 file.
    Returns r-precision score for use in evaluate() and output files.

    Parameters:
    prob_mat_path (str): Path to probability matrices for parent-child pairs.

    Returns:
    r_precision (float): Calculated r-precision for data set of PCPs.

    """


def locate_child_substitutions(parent_aa, child_aa):
    """
    Return the location of the amino acid substitutions for a given parent-child pair.

    Parameters:
    parent_aa (str): Amino acid sequence of parent.
    child_aa (str): Amino acid sequence of child.

    Returns:
    child_sub_sites (np.array): Location of substitutions in parent-child pair.

    """
    child_sub_sites = []

    for i in range(len(parent_aa)):
        if parent_aa[i] != child_aa[i]:
            child_sub_sites.append(i)

    child_sub_sites = np.array(child_sub_sites)

    return child_sub_sites


def calculate_site_substitution_probabilities(prob_matrix, parent_aa):
    """
    Calculate the probability of substitution at each site for a parent sequence.

    Parameters:
    prob_matrix (np.ndarray): A 2D array containing the normalized probabilities of the amino acids by site for a parent sequence.
    parent_aa (str): Amino acid sequence of parent.

    Returns:
    site_sub_probs (np.array): 1D array containing probability of substitution at each site for a parent sequence.

    """
    site_sub_probs = []

    for i in range(len(parent_aa)):
        site_sub_probs.append(1 - prob_matrix[:, i][aa_str_sorted.index(parent_aa[i])])

    site_sub_probs = np.array(site_sub_probs)

    assert site_sub_probs.size == len(
        parent_aa
    ), "The number of substitution probabilities does not match the number of amino acid sites."

    return site_sub_probs


def locate_top_k_substitutions(site_sub_probs, k_sub):
    """
    Return the top k substitutions predicted for a parent-child pair given precalculated site substitution probabilities.

    Parameters:
    site_sub_probs (str?): Probability of substition at each site for a parent sequence.
    k_sub (int): Number of substitutions observed in PCP.

    Returns:
    pred_sub_sites (np.array): Location of top-k predicted substitutions by model (unordered).

    """
    if k_sub == 0:
        return []

    pred_sub_sites = np.argpartition(site_sub_probs, -k_sub)[-k_sub:]

    assert (
        pred_sub_sites.size == k_sub
    ), "The number of predicted substitution sites does not match the number of actual substitution sites."

    return pred_sub_sites


def calculate_cross_loss_entropy():
    """
    Calculate cross loss entropy for all PCPs in one data set/HDF5 file.

    Parameters:

    Returns:


    """
