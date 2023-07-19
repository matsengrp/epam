"""Code for evaluating model performance."""

import h5py
import pandas as pd
from epam.sequences import aa_str_sorted
from epam.utils import generate_file_checksum
from epam.models import *
from epam.sequences import translate_sequences


def evaluate(prob_mat_filename, outfilename):
    """
    Evaluate model predictions against reality in parent-child pairs (PCPs)
    Function should be model-agnositic and currently limited to substitution accuracy
    Output to CSV with columns for the different metrics and a row per data set

    Parameters:
    prob_mat_filename (str): file name of probability matrix for parent-child pairs
    outfilename (str): file name for output of model performance metrics
    """
    with h5py.File(prob_mat_filename, "r") as prob_mat_file:
        pcp_filename = prob_mat_file.attrs["pcp_filename"]

        # make sure probablity matrix matches PCP
        if prob_mat_file.attrs["checksum"] != generate_file_checksum(pcp_filename):
            raise ValueError(f"checksum failed for {pcp_filename}.")

        # call metric functions
        sub_acc = calculate_sub_accuracy(prob_mat_filename)

        # output metrics to csv
        model_performance = pd.DataFrame(
            {
                "data_set": [pcp_filename], 
                "model": ["ablang"],  # Issue 8: hard coded for the moment
                "sub_accuracy": [sub_acc],
            }
        )
        model_performance.to_csv(outfilename, index=False)


def calculate_sub_accuracy(prob_mat_file):
    """
    Calculate substitution accuracy for all PCPs in one data set/HDF5 file
    Returns substitution accuracy score for use in evaluate() and reporting all files

    Parameters:
    prob_mat_filename (str): file name of probability matrices for parent-child pairs

    Returns:
    sub_accuracy (float): calculated substitution accuracy for data set of PCPs
    """
    with h5py.File(prob_mat_file, "r") as matfile:
        pcp_filename = matfile.attrs["pcp_filename"]
        pcp_df = pd.read_csv(pcp_filename, index_col=0)

        num_sub_total = 0
        num_sub_correct = 0

        for matname in matfile.keys():
            grp = matfile[matname]
            matrix = grp["data"]
            index = grp.attrs["pcp_index"]

            parent_nt, child_nt = pcp_df.loc[index, ["orig_seq", "mut_seq"]]
            [parent_aa, child_aa] = translate_sequences([parent_nt, child_nt])

            assert len(parent_aa) == len(child_aa)

            for i in range(len(parent_aa)):
                if parent_aa[i] != child_aa[i]:
                    num_sub_total += 1
                    # get the column of the probability matrix corresponding to amino acid site of interest
                    # sort the probabilities (lowest to highest) and return the result as ordered list of indices
                    # reverse the order of the list of indices (argsort() only sorts by lowest to highest)
                    prob_sorted_aa_indices = matrix[:, i].argsort()[
                        ::-1
                    ]  # code from Kevin
                    # string ranked aa string for easy eval
                    pred_aa_ranked = "".join(
                        (np.array(list(aa_str_sorted))[prob_sorted_aa_indices])
                    )
                    # get AA sequence predicted by model
                    # force a substitution if model predicts same aa as parent
                    # if highest prob aa doesn't match parent, use it
                    if pred_aa_ranked[0] == parent_aa[i]:
                        pred_aa_sub = pred_aa_ranked[1]
                    elif pred_aa_ranked[0] != parent_aa[i]:
                        pred_aa_sub = pred_aa_ranked[0]
                    # if aa matches predicted aa (given a substitution), add to num_sub_correct
                    if pred_aa_sub == child_aa[i]:
                        num_sub_correct += 1

        sub_accuracy = num_sub_correct / num_sub_total
        return sub_accuracy
