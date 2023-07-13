"""Utilities for various tasks."""

import hashlib


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
