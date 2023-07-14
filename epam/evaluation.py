import h5py
import pandas as pd
from epam.utils import generate_file_checksum
from epam.models import *


def evaluate(prob_mat_filename):
    """
    Evaluate model predictions against reality in parent-child pairs (PCPs)
    Function should be model-agnositic and currently includes substitution accuracy
    Output to CSV with columns for the different metrics, and a row for each PCP

    Parameters:
    prob_mat_filename (string?): 
    """
    with h5py.File(prob_mat_filename, "r") as prob_mat_file:
        pcp_filename = prob_mat_file.attrs["pcp_filename"]

        # make sure probablity matrix matches PCP
        if prob_mat_file.attrs["checksum"] != generate_file_checksum(pcp_filename):
            raise ValueError(f"checksum failed for {pcp_filename}.")

        # call function

        # output metrics in csv


# calculated for full data set - loop through data set for each metric
def sub_accuracy



# read in prob_match for one PCP
# read in sequence data for child sequence and parent sequence
# translate child sequence from nuc to aa,  
# location/number of mutations in child (as scanning, track substitions)
# evaluate child sequence:
    # for sub accuracy:
    # find sites that mutated in predicted sequence
    # find number of mutated sites that match child aa
    # take ratio of number of matches/number of mutated sites


#for matrix in prob_mat_file.keys() # gives group name / 1 matrix

# given n substitutions observed in the child, find top n probable substitutions/create predicted seq
# compare predicted sequence with child sequence: