"""Code for handling sequences and sequence files."""

import itertools

import torch
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"
NT_STR_SORTED = "ACGT"
CODONS = [
    "".join(codon_list)
    for codon_list in itertools.product(["A", "C", "G", "T"], repeat=3)
]
STOP_CODONS = ["TAA", "TAG", "TGA"]


def nt_idx_array_of_str(nt_str):
    """Return the indices of the nucleotides in a string."""
    return np.array([NT_STR_SORTED.index(nt) for nt in nt_str])


def nt_idx_tensor_of_str(nt_str):
    """Return the indices of the nucleotides in a string."""
    return torch.tensor([NT_STR_SORTED.index(nt) for nt in nt_str])


def read_fasta_sequences(file_path):
    with open(file_path, "r") as handle:
        sequences = [str(record.seq) for record in SeqIO.parse(handle, "fasta")]
    return sequences


def translate_sequences(nt_sequences):
    aa_sequences = []
    for seq in nt_sequences:
        if len(seq) % 3 != 0:
            raise ValueError(f"The sequence '{seq}' is not a multiple of 3.")
        aa_seq = str(Seq(seq).translate())
        if "*" in aa_seq:
            raise ValueError(f"The sequence '{seq}' contains a stop codon.")
        aa_sequences.append(aa_seq)
    return aa_sequences


def aa_index_of_codon(codon):
    """Return the index of the amino acid encoded by a codon."""
    aa = translate_sequences([codon])[0]
    return AA_STR_SORTED.index(aa)


def assert_pcp_lengths(parent, child):
    """Assert that the lengths of the parent and child sequences are
    the same and that they are multiples of 3.
    """
    if len(parent) != len(child):
        raise ValueError(
            f"The parent and child sequences are not the same length: "
            f"{len(parent)} != {len(child)}"
        )
    if len(parent) % 3 != 0:
        raise ValueError(f"Found a PCP with length not a multiple of 3: {len(parent)}")


def generate_codon_aa_indicator_matrix():
    """Generate a matrix that maps codons (rows) to amino acids (columns)."""

    matrix = np.zeros((len(CODONS), len(AA_STR_SORTED)))

    for i, codon in enumerate(CODONS):
        try:
            aa = translate_sequences([codon])[0]
            aa_idx = AA_STR_SORTED.index(aa)
            matrix[i, aa_idx] = 1
        except ValueError:  # Handle STOP codon
            pass

    return matrix


CODON_AA_INDICATOR_MATRIX = torch.tensor(generate_codon_aa_indicator_matrix())
