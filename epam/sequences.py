"""Code for handling sequences and sequence files."""


from Bio import SeqIO
from Bio.Seq import Seq

global aa_str_sorted
aa_str_sorted = "ACDEFGHIKLMNPQRSTVWY"


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
