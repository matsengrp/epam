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
    prob_mat_path (str): path to probability matrix for parent-child pairs.
    outfilename (str): path to output for model performance metrics.

    """
    pcp_path = pcp_path_of_prob_mat_path(prob_mat_path)

    sub_acc = calculate_sub_accuracy(prob_mat_path)
    r_prec = None
    cross_ent = None

    model_performance = pd.DataFrame(
        {
            "data_set": [pcp_path],
            "model": ["ablang"],  # Issue 8: hard coded for the moment
            "sub_accuracy": [sub_acc],
            "r_precision": [r_prec],
            "cross_entropy": [cross_ent]
        }
    )

    model_performance.to_csv(model_performance_path, index=False)


def calculate_sub_accuracy(prob_mat_path):
    """
    Calculate substitution accuracy for all PCPs in one data set/HDF5 file.
    Returns substitution accuracy score for use in evaluate() and output files.

    Parameters:
    prob_mat_path (str): path to probability matrices for parent-child pairs.

    Returns:
    sub_accuracy (float): calculated substitution accuracy for data set of PCPs.

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
    matrix_i (np.array): probability matrix for parent-child pair at aa site i.
    parent_aa (str): parent amino acid sequence.
    i (int): index of amino acid site substituted.

    Returns:
    pred_aa_sub (str): predicted amino acid substitution (most likely non-parent aa).

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
    prob_mat_path (str): path to probability matrices for parent-child pairs.

    Returns:
    r_precision (float): calculated r-precision for data set of PCPs.

    """
    with h5py.File(prob_mat_path, "r") as matfile:
        pcp_path = pcp_path_of_prob_mat_path(prob_mat_path)
        pcp_df = pd.read_csv(pcp_path, index_col=0)

        num_sub_total = 0
        num_sub_location_correct = 0

        for matname in matfile.keys():
            grp = matfile[matname]
            matrix = grp["data"]
            index = grp.attrs["pcp_index"]

            parent_nt, child_nt = pcp_df.loc[index, ["parent", "child"]]
            [parent_aa, child_aa] = translate_sequences([parent_nt, child_nt])

            

        r_precision = num_sub_location_correct / num_sub_total

        return r_precision


def locate_child_substitutions():
    """
    Return the location of the amino acid substitutions for a given parent-child pair.

    Parameters:

    Returns:

    """

def locate_top_k_substitutions():
    """
    Return the top k substitutions for a given parent-child pair.

    Parameters:

    Returns:

    """


def calculate_cross_loss_entropy():
    """
    Calculate cross loss entropy for all PCPs in one data set/HDF5 file.

    Parameters:

    Returns:


    """