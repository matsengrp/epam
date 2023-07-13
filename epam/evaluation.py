import h5py
import pandas as pd
from epam.utils import generate_file_checksum
from epam.models import *


def evaluate(prob_mat_filename):
    with h5py.File(prob_mat_filename, "r") as prob_mat_file:
        pcp_filename = prob_mat_file.attrs["pcp_filename"]
        if prob_mat_file.attrs["checksum"] != generate_file_checksum(pcp_filename):
            raise ValueError(f"checksum failed for {pcp_filename}.")
