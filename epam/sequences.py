"""Code for handling sequences and sequence files."""

import itertools

import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable

AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"
NT_STR_SORTED = "ACGT"
CODONS = [
    "".join(codon_list)
    for codon_list in itertools.product(["A", "C", "G", "T"], repeat=3)
]


def nucleotide_indices_of_codon(codon):
    """Return the indices of the nucleotides in a codon."""
    return [NT_STR_SORTED.index(nt) for nt in codon]


def read_fasta_sequences(file_path):
    with open(file_path, "r") as handle:
        sequences = [str(record.seq) for record in SeqIO.parse(handle, "fasta")]
    return sequences


def translate_sequences(nt_sequences):
    aa_sequences = []
    for seq in nt_sequences:
        aa_seq = str(Seq(seq).translate())
        if "*" in aa_seq:
            raise ValueError(f"The sequence '{seq}' contains a stop codon.")
        aa_sequences.append(aa_seq)
    return aa_sequences

def truncate_sequence_at_codon_boundary(seq):
    """Truncate a sequence to the nearest codon boundary."""
    if len(seq) % 3 == 0:
        return seq
    else:
        return seq[:-(len(seq) % 3)]


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


CODON_AA_INDICATOR_MATRIX = generate_codon_aa_indicator_matrix()
