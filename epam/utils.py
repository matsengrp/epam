"""Utilities for various tasks."""

import hashlib
import h5py
import torch
import pandas as pd
import numpy as np
from netam.sequences import pcp_criteria_check


def generate_file_checksum(filename, buf_size=65536):
    """
    Generate checksum of a file.

    Parameters:
    filename (str): file name.

    buf_size (int): buffer size for reading the file in chunks (default: 64kB)

    Returns:
    integer: checksum in hex.

    """
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for data in iter(lambda: f.read(buf_size), b""):
            sha256.update(data)
    return sha256.hexdigest()


def pcp_path_of_aaprob_path(aaprob_path):
    """
    Return the path to the PCP file that matches the aaprob matrix.
    Check SHAs to ensure the PCP file matches the aaprob matrix.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.

    Returns:
    pcp_path (str): path to parent-child pairs.

    """
    with h5py.File(aaprob_path, "r") as aaprob_file:
        pcp_path = aaprob_file.attrs["pcp_filename"]

        if aaprob_file.attrs["checksum"] != generate_file_checksum(pcp_path):
            raise ValueError(f"checksum failed for {pcp_path}.")

    return pcp_path


def load_and_filter_pcp_df(pcp_path):
    """
    Load PCP data and filter out PCPs that do not meet criteria.

    Parameters:
    pcp_path (str): path to parent-child pairs.

    Returns:
    pcp_df (pd.DataFrame): PCP data frame.

    """
    full_pcp_df = pd.read_csv(pcp_path, index_col=0)

    # Filter out PCPs that do not meet criteria: 0% < mutation rate < 30% (default value of max_mut_freq)
    pcp_df = full_pcp_df[
        full_pcp_df.apply(lambda x: pcp_criteria_check(x["parent"], x["child"]), axis=1)
    ]

    return pcp_df


def ratios_to_sigmoid(ratio_sel_matrix, scale_const=1):
    """
    Convert selection factors or probability ratios to sigmoid.
    The log (base-e) of the selection factor is passed to the sigmoid function.

    Parameters:
    ratio_sel_matrix (torch.Tensor): selection factors. Expected range of each element is [0, infty].
    scale_const (float): exponent applied to each selection factor.

    Returns:
    sigmoid_sel_matrix (torch.Tensor): sigmoid of selection factors or probability ratios.

    """
    scaled_ratio = torch.pow(ratio_sel_matrix, scale_const)
    sigmoid_sel_matrix = 1 / (1 + (1 / scaled_ratio))
    return sigmoid_sel_matrix
