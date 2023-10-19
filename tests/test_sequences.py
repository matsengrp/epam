import pytest
import numpy as np
from Bio.Seq import Seq
from Bio.Data import CodonTable
from epam.sequences import (
    AA_STR_SORTED,
    CODONS,
    nt_idx_array_of_str,
    translate_sequences,
    CODON_AA_INDICATOR_MATRIX,
)


def test_nucleotide_indices_of_codon():
    assert nt_idx_array_of_str("AAA").tolist() == [0, 0, 0]
    assert nt_idx_array_of_str("TAC").tolist() == [3, 0, 1]
    assert nt_idx_array_of_str("GCG").tolist() == [2, 1, 2]


def test_translate_sequences():
    # sequence without stop codon
    seq_no_stop = ["AGTGGTGGTGGTGGTGGT"]
    assert translate_sequences(seq_no_stop) == [str(Seq(seq_no_stop[0]).translate())]

    # sequence with stop codon
    seq_with_stop = ["TAAGGTGGTGGTGGTAGT"]
    with pytest.raises(ValueError):
        translate_sequences(seq_with_stop)


def test_indicator_matrix():
    reconstructed_codon_table = {}

    for i, codon in enumerate(CODONS):
        row = CODON_AA_INDICATOR_MATRIX[i]
        if np.any(row):
            amino_acid = AA_STR_SORTED[np.argmax(row)]
            reconstructed_codon_table[codon] = amino_acid

    table = CodonTable.unambiguous_dna_by_id[1]  # 1 is for the standard table

    assert reconstructed_codon_table == table.forward_table
