"""Code for handling sequences and sequence files."""

import itertools

from Bio import SeqIO
from Bio.Seq import Seq

AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"
NT_STR_SORTED = "ACGT"
CODONS = [
    "".join(codon_list)
    for codon_list in itertools.product(["A", "C", "G", "T"], repeat=3)
]


def read_fasta_sequences(file_path):
    with open(file_path, "r") as handle:
        sequences = [str(record.seq) for record in SeqIO.parse(handle, "fasta")]
    return sequences


def translate_sequences(nt_sequences):
    aa_sequences = []
    for seq in nt_sequences:
        aa_seq = Seq(seq).translate()
        if "*" in str(aa_seq):
            raise ValueError(f"The sequence '{seq}' contains a stop codon.")
        aa_sequences.append(str(aa_seq))
    return aa_sequences
