"""Routines for molecular evolution computation.

"""

import numpy as np

from epam.sequences import (
    NT_STR_SORTED,
    AA_STR_SORTED,
    CODON_AA_INDICATOR_MATRIX,
    assert_pcp_lengths,
    translate_sequences,
)


def normalize_sub_probs(parent: str, sub_probs: np.ndarray) -> np.ndarray:
    """
    Normalize substitution probabilities.

    Given a parent DNA sequence and a 2D numpy array representing substitution
    probabilities, this function sets the probability of the actual nucleotide
    in the parent sequence to zero and then normalizes each row to form a valid
    probability distribution.

    Parameters:
    parent (str): The parent sequence.
    sub_probs (np.ndarray): A 2D numpy array representing substitution
                            probabilities. Rows correspond to sites, and columns
                            correspond to "ACGT" bases.

    Returns:
    np.ndarray: A 2D numpy array with normalized substitution probabilities.
    """
    for i, base in enumerate(parent):
        idx = NT_STR_SORTED.index(base)
        sub_probs[i, idx] = 0.0

    row_sums = np.sum(sub_probs, axis=1, keepdims=True)
    return sub_probs / row_sums


def build_mutation_matrix(parent_codon, mut_probs, sub_probs):
    """Generate a matrix that represents the mutation probability for each
    site in a given parent codon. So, the ijth entry of the matrix is the
    probability of the ith position mutating to the jth nucleotide.

    See codon_to_aa_probabilities for a description of the arguments, and
    tests for an example."""

    result_matrix = np.empty((len(parent_codon), len(NT_STR_SORTED)))

    for site, nucleotide in enumerate(parent_codon):
        try:
            parent_nt_idx = NT_STR_SORTED.index(nucleotide)
        except ValueError:
            raise ValueError(f"Invalid nucleotide {nucleotide} in codon {parent_codon}")

        for j, nt in enumerate(NT_STR_SORTED):
            if j == parent_nt_idx:
                result_matrix[site, j] = 1 - mut_probs[site]
            else:
                result_matrix[site, j] = mut_probs[site] * sub_probs[site][j]

    return result_matrix


def codon_probs_of_mutation_matrix(mut_matrix):
    """
    Compute the probability tensor for mutating to the codon ijk.

    This method calculates the tensor where the ijk-th entry represents
    the probability of mutating to the codon formed by the i-th, j-th,
    and k-th nucleotide in the nucleotide list. It uses numpy outer
    products to construct this tensor.

    Parameters:
    mut_matrix (numpy.ndarray): A 3D array representing the mutation
                                matrix. The mutation matrix should be
                                formatted as per _build_mutation_matrix.

    Returns:
    numpy.ndarray: A 3D array where the ijk-th entry is the probability
                of mutating to the codon ijk.
    """
    return (
        mut_matrix[0][:, np.newaxis, np.newaxis]
        * mut_matrix[1][np.newaxis, :, np.newaxis]
        * mut_matrix[2][np.newaxis, np.newaxis, :]
    )


def codon_to_aa_probabilities(parent_codon, mut_probs, sub_probs):
    """
    For a specified codon and given nucleotide mutability and substitution probabilities,
    compute the amino acid substitution probabilities.

    Following terminology from Yaari et al 2013, "mutability" refers to the probability
    of a nucleotide mutating at a given site, while "substitution" refers to the probability
    of a nucleotide mutating to another nucleotide at a given site conditional on having
    a mutation.

    We assume that the mutation and substitution probabilities already take branch length
    into account. Here we translate those into amino acid probabilities, which are normalized.
    Probabilities to stop codons are dropped, but self probabilities are kept.

    Parameters:
    parent_codon (str): The specified codon.
    mut_probs (list): The mutability probabilities for each site in the codon.
    sub_probs (list): The substitution probabilities for each site in the codon.

    Returns:
    list: An array of probabilities for all 20 amino acids.

    """
    mut_matrix = build_mutation_matrix(parent_codon, mut_probs, sub_probs)
    codon_probs = codon_probs_of_mutation_matrix(mut_matrix)
    # reshape(-1) flattens the array
    aa_probs = codon_probs.reshape(-1) @ CODON_AA_INDICATOR_MATRIX
    aa_probs /= aa_probs.sum()
    return aa_probs


def aaprobs_of_parent_rates_and_sub_probs(parent, rates, sub_probs) -> np.ndarray:
    """Calculate per-site amino acid probabilities from per-site rates and
    substitution probabilities.

    Args:
        rates (np.ndarray): Poisson rates of mutation per site.
        sub_probs (np.ndarray): Substitution probabilities per site: a 2D
                                array with rows corresponding to sites and
                                columns corresponding to nucleotides.

    Returns:
        np.ndarray: A 2D array with rows corresponding to sites and columns
                    corresponding to amino acids.
    """

    # This `mut_probs` is the probability of at least one mutation at each site.
    # So here we are interpreting the probability in the correctly-specified way rather than the mis-specified
    # way. This is helpful because we'd like normalized probabilities.
    mut_probs = 1.0 - np.exp(-rates)

    # keep track of probabilities as a row per amino acid site, then take transpose before returning output
    aaprobs = []

    for i in range(0, len(parent), 3):
        parent_codon = parent[i : i + 3]
        codon_mut_probs = mut_probs[i : i + 3]
        codon_subs = sub_probs[i : i + 3]

        site_probs = codon_to_aa_probabilities(
            parent_codon, codon_mut_probs, codon_subs
        )
        aaprobs.append(site_probs)

    return np.array(aaprobs)
