"""Utilities for various tasks."""

import hashlib
import h5py


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
