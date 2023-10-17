import pytest
import numpy as np
from Bio.Seq import Seq
from Bio.Data import CodonTable
from epam.sequences import (
    AA_STR_SORTED,
    CODONS,
    STOP_CODONS,
    nucleotide_indices_of_codon,
    translate_sequences,
    CODON_AA_INDICATOR_MATRIX,
)


def test_nucleotide_indices_of_codon():
    assert nucleotide_indices_of_codon("AAA") == [0, 0, 0]
    assert nucleotide_indices_of_codon("TAC") == [3, 0, 1]
    assert nucleotide_indices_of_codon("GCG") == [2, 1, 2]


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


def test_indicator_matrix_again():
    """Check that the indicator matrix maps a normalized probability
    distribution on non-stop codons to a probability distribution on amino
    acids."""
    # Initialize a random vector of size 64
    random_vector = np.random.rand(64)

    # Set the entries corresponding to stop codons to zero
    for stop_codon in STOP_CODONS:
        index = CODONS.index(stop_codon)
        random_vector[index] = 0.0

    # Normalize the remaining entries so that they sum to 1
    random_vector /= random_vector.sum()

    assert np.isclose((random_vector @ CODON_AA_INDICATOR_MATRIX).sum(), 1)
