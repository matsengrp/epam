import h5py
import pandas as pd
from epam.utils import generate_file_checksum
from epam.models import *


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

        # call function

        # output metrics in csv


# calculated for full data set - loop through data set for each metric
def sub_accuracy(prob_mat_file):
    """
    Calculate substitution accuracy for all PCPs in one data set/HDF5 file
    Returns substitution accuracy score for use in evaluate() and reporting all files

    Parameters:


    Returns:


    """
    # probably need to open file for reading first (?)
    # set empty counters for num_mut_total and num_mut_correct
    # read in probablity matrix for one PCP at a time, iterate over all pairs
    for matrix in prob_mat_file.keys():
        # read in corresponding parent (orig_seq) and child (mut_seq) sequence
        # translate nt sequences to aa sequences
        # scan through sites and find locations of substitutions
        # if site mutated, add to some counters for num_mut_total
        # if site mutated, check if aa matches aa with highest prob in matrix
        print
    # calculate substitution accuracy (ratio of num_mut_correct/num_mut_total)
    # return value

