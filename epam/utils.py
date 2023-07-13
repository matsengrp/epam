"""Utilities for various tasks."""

import hashlib
import h5py
import pandas as pd
from epam.models import AbLang
from epam.sequences import translate_sequences

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
    with open(filename, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


def produce_probability_matrices(model, pcp_filename, output_filename):
    """
    Produce a probability matrix for each parent-child pair (PCP) of nucleotide sequences with a substitution model.

    An HDF5 output file is created that includes the file path to the PCP data and a checksum for verification.

    Parameters:
    model (epam.BaseModel): model for predicting substitution probabilities.

    pcp_filename (str): file name of parent-child pair data.

    output_filename (str): output file name.

    """
    checksum = generate_file_checksum(pcp_filename)
    pcp_df = pd.read_csv(pcp_filename, index_col=0)

    with h5py.File(output_filename, 'w') as outfile:
        # attributes related to PCP data file
        outfile.attrs['checksum'] = checksum
        outfile.attrs['pcp_filename'] = pcp_filename

        for i, row in pcp_df.iterrows():
            parent = row['orig_seq']
            child = row['mut_seq']
            [parent_aa, child_aa] = translate_sequences([parent, child])
            matrix = model.prob_matrix_of_parent_child_pair(parent_aa, child_aa)
            #matrix = model.probability_array_of_seq(parent_aa)  # AbLang placeholder until generic model class is implemented

            # create a group for each matrix
            grp = outfile.create_group(f'matrix{i}')
            grp.attrs['pcp_index'] = i

            # enable gzip compression
            grp.create_dataset('data', data=matrix, compression='gzip', compression_opts=4)
