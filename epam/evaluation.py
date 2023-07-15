import h5py
import pandas as pd
from epam.utils import generate_file_checksum
from epam.models import *
from epam.sequences import translate_sequences


def evaluate(prob_mat_filename):
    """
    Evaluate model predictions against reality in parent-child pairs (PCPs)
    Function should be model-agnositic and currently limited to substitution accuracy
    Output to CSV with columns for the different metrics, and a row for each PCP

    Parameters:
    prob_mat_filename (str): file name of probability matrix for parent-child pairs
    """
    with h5py.File(prob_mat_filename, "r") as prob_mat_file:
        pcp_filename = prob_mat_file.attrs["pcp_filename"]

        # make sure probablity matrix matches PCP
        if prob_mat_file.attrs["checksum"] != generate_file_checksum(pcp_filename):
            raise ValueError(f"checksum failed for {pcp_filename}.")

        # call metric functions

        # output metrics in csv


# calculated for full data set - loop through data set for each metric
def calculate_sub_accuracy(prob_mat_file):
    """
    Calculate substitution accuracy for all PCPs in one data set/HDF5 file
    Returns substitution accuracy score for use in evaluate() and reporting all files

    Parameters:
    prob_mat_filename (str): file name of probability matrices for parent-child pairs

    Returns:
    sub_accuracy (float): calculated substitution accuracy for data set of PCPs
    """
    # open file for reading
    with h5py.File(prob_mat_file, "r") as matfile:
        # get path to PCP sequence data
        pcp_filename = matfile.attrs["pcp_filename"]
        # read in PCPs as a pandas df
        pcp_df = pd.read_csv(pcp_filename, index_col=0)
        # set empty counters for num_sub_total and num_sub_correct
        num_sub_total = 0. # floats to avoid issues with integer division??
        num_sub_correct = 0.
        # read in probablity matrix for one PCP at a time, iterate over all pairs
        for matrix in matfile.keys():
            # get on probablity matrix
            grp = matfile[matrix]
            # get identity for pairing with sequences
            index = grp.attrs["pcp_index"]
            # pull corresponding PCP info
            row = pcp_df.loc[index]
            parent = row["orig_seq"]
            child = row["mut_seq"]
            # translate nt sequences to AA sequences
            [parent_aa, child_aa] = translate_sequences([parent, child])
            # add check to make sure len(parent_aa) = len(child_aa) ??
            # get AA sequence predicted by model
            pred_aa_sub = matrix.max(axis='columns') # PROBABLY WRONG
            # scan through sites and find locations of substitutions
            for i in range(len(parent_aa)):
                # if mutated, add to num_sub_total
                if parent_aa[i] != child_aa[i]:
                    num_sub_total += 1
                    # if aa matches highest prob aa in matrix, add to num_sub_correct
                    if pred_aa_sub[i] == child_aa[i]:
                        num_sub_correct += 1
        # calculate substitution accuracy (ratio of num_sub_correct/num_sub_total)
        sub_accuracy = num_sub_correct/num_sub_total
        # return value
        return sub_accuracy

