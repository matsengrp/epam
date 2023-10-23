"""Free functions for molecular evolution computation.

We will follow terminology from Yaari et al 2013, where "mutability" refers to
the probability of a nucleotide mutating at a given site, while "substitution"
refers to the probability of a nucleotide mutating to another nucleotide at a
given site conditional on having a mutation.

We assume that the mutation and substitution probabilities already take branch
length into account.  
"""

import numpy as np

from epam.sequences import CODON_AA_INDICATOR_MATRIX


def normalize_sub_probs(parent_idxs: np.ndarray, sub_probs: np.ndarray) -> np.ndarray:
    """
    Normalize substitution probabilities.

    Given a parent DNA sequence and a 2D numpy array representing substitution
    probabilities, this function sets the probability of the actual nucleotide
    in the parent sequence to zero and then normalizes each row to form a valid
    probability distribution.

    Parameters:
    parent_idxs (np.ndarray): The parent sequence indices.
    sub_probs (np.ndarray): A 2D numpy array representing substitution
                            probabilities. Rows correspond to sites, and columns
                            correspond to "ACGT" bases.

    Returns:
    np.ndarray: A 2D numpy array with normalized substitution probabilities.
    """

    # Create an array of row indices that matches the shape of `parent_idxs`.
    row_indices = np.arange(len(parent_idxs))

    # Set the entries corresponding to the parent sequence to zero.
    sub_probs[row_indices, parent_idxs] = 0.0

    # Normalize the probabilities.
    row_sums = np.sum(sub_probs, axis=1, keepdims=True)
    return sub_probs / row_sums


def build_mutation_matrices(
    parent_codon_idxs_v: np.ndarray, mut_probs_v: np.ndarray, sub_probs_v: np.ndarray
) -> np.ndarray:
    """
    Generate a sequence of 3x4 mutation matrices for parent codons along a sequence.

    Given indices for parent codons, mutation probabilities, and substitution probabilities for
    each parent codon along the sequence, this function constructs a sequence of 3x4 matrices. Each
    matrix in the sequence represents the mutation probabilities for each nucleotide position in a parent
    codon. The ijkth entry of the resulting tensor corresponds to the probability of the jth nucleotide
    in the ith parent codon mutating to the kth nucleotide (in indices).

    Parameters:
    parent_codon_idxs_v (np.ndarray): 2D array with each row containing indices representing
                                      the parent codon's nucleotides at each site along the sequence.
                                      Shape should be (codon_count, 3).
    mut_probs_v (np.ndarray): 2D array representing the mutation probabilities for each site in the codon,
                              for each codon along the sequence. Shape should be (codon_count, 3).
    sub_probs_v (np.ndarray): 3D array representing substitution probabilities for each codon along the
                              sequence for each site.
                              Shape should be (codon_count, 3, 4).

    Returns:
    np.ndarray: A 4D array with shape (codon_count, 3, 4) where the ijkth entry is the mutation probability
                of the jth position in the ith parent codon mutating to the kth nucleotide.
    """
    codon_count = parent_codon_idxs_v.shape[0]
    assert parent_codon_idxs_v.shape[1] == 3, "Each parent codon must be of length 3"

    result_matrices = np.empty((codon_count, 3, 4))

    # Create a mask with the shape (codon_count, 3, 4) to identify where
    # nucleotides in the parent codon are the same as the nucleotide positions
    # in the new codon. Each row in the third dimension contains a boolean
    # value, which is True if the nucleotide position matches the parent codon
    # nucleotide. How it works: newaxis adds one more dimension to the array, so
    # that the shape of the array is (codon_count, 3, 1) instead of
    # (codon_count, 3). Then broadcasting automatically expands dimensions where
    # needed. So the arange(4) array is automatically expanded to match the
    # (codon_count, 3, 1) shape by implicitly turning it into a (1, 1, 4) shape
    # array, where it is then broadcasted to the shape (codon_count, 3, 4) to
    # match the shape of parent_codon_idxs_v[:, :, np.newaxis] for equality
    # testing.
    mask_same_nt = np.arange(4) == parent_codon_idxs_v[:, :, np.newaxis]

    # Find the multi-dimensional indices where the nucleotide in the parent
    # codon is the same as the nucleotide in the mutation outcome (i.e., no
    # mutation occurs).
    same_nt_indices = np.nonzero(mask_same_nt)

    # Using the multi-dimensional indices obtained from the boolean mask, update
    # the mutation probability in result_matrices to be "1.0 -
    # mutation_probability" at these specific positions. This captures the
    # probability of a given nucleotide not mutating.
    result_matrices[same_nt_indices] = 1.0 - mut_probs_v[same_nt_indices[:-1]]

    # Assign values where the nucleotide is different via broadcasting.
    mask_diff_nt = ~mask_same_nt
    # To understand this line, remember that * is element-wise multiplication.
    # The newaxis repeats the array along a new axis, so that the shape of the
    # array is (codon_count, 3, 4); the mut_probs don't depend on the new base.
    result_matrices[mask_diff_nt] = (mut_probs_v[:, :, np.newaxis] * sub_probs_v)[
        mask_diff_nt
    ]

    return result_matrices


def codon_probs_of_mutation_matrices(mut_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the probability tensor for mutating to the codon ijk along the entire sequence.

    Parameters:
    mut_matrix (numpy.ndarray): A 3D array representing the mutation matrix for the entire sequence.
                                The shape should be (n_sites, 3, 4), where n_sites is the number of sites,
                                3 is the number of positions in a codon, and 4 is the number of nucleotides.

    Returns:
    numpy.ndarray: A 4D array where the first axis represents different sites in the sequence and
                   the ijk-th entry of the remaining 3D tensor is the probability of mutating to the codon ijk.
    """
    assert (
        mut_matrix.shape[1] == 3
    ), "The second dimension of the input mut_matrix should be 3 to represent the 3 positions in a codon"
    assert (
        mut_matrix.shape[2] == 4
    ), "The last dimension of the input mut_matrix should be 4 to represent the 4 nucleotides"

    # The key to understanding how this works is that when these arrays are
    # multiplied, NumPy broadcasts them into a common shape (n_sites, 4, 4, 4),
    # performing element-wise multiplication for each slice along the first axis
    # (i.e., for each site).
    return (
        mut_matrix[:, 0, :, np.newaxis, np.newaxis]
        * mut_matrix[:, 1, np.newaxis, :, np.newaxis]
        * mut_matrix[:, 2, np.newaxis, np.newaxis, :]
    )


def aaprobs_of_codon_probs_v(codon_probs_v: np.ndarray) -> np.ndarray:
    """
    Compute the probability of each amino acid from the probability of each codon, for each parent codon along the sequence.

    Parameters:
    codon_probs_v (numpy.ndarray): A 4D array representing the probability of mutating
                                   to each codon for each parent codon along the sequence.
                                   Shape should be (codon_count, 4, 4, 4).

    Returns:
    numpy.ndarray: A 2D array with shape (codon_count, 20) where the ij-th entry is the probability
                   of mutating to the amino acid j from the codon i for each parent codon along the sequence.
    """
    codon_count = codon_probs_v.shape[0]
    # Reshape such that we merge the last three dimensions into a single dimension while keeping
    # the `codon_count` dimension intact. This prepares the tensor for matrix multiplication.
    reshaped_probs = codon_probs_v.reshape(codon_count, -1)

    # Perform matrix multiplication to get unnormalized amino acid probabilities.
    aaprobs = reshaped_probs @ CODON_AA_INDICATOR_MATRIX

    # Normalize probabilities along the amino acid dimension.
    row_sums = aaprobs.sum(axis=1, keepdims=True)
    aaprobs /= row_sums

    return aaprobs


def aaprob_of_mut_and_sub_v(
    parent_codon_idxs_v: np.ndarray, mut_probs_v: np.ndarray, sub_probs_v: np.ndarray
) -> np.ndarray:
    """
    For a sequence of parent codons and given nucleotide mutability and substitution probabilities,
    compute the amino acid substitution probabilities for each codon along the sequence.

    Parameters:
    parent_codon_idxs_v (np.ndarray): A 2D array where each row contains indices representing 
                                      the parent codon's nucleotides at each site along the sequence.
                                      Shape should be (codon_count, 3).
    mut_probs_v (np.ndarray): A 2D array representing the mutation probabilities for each site in the codon,
                              for each codon along the sequence. Shape should be (codon_count, 3).
    sub_probs_v (np.ndarray): A 3D array representing substitution probabilities for each codon along the 
                              sequence for each site.
                              Shape should be (codon_count, 3, 4).

    Returns:
    np.ndarray: A 2D array with shape (codon_count, 20) where the ij-th entry is the probability
                of mutating to the amino acid j from the codon i for each parent codon along the sequence.
    """
    mut_matrices = build_mutation_matrices(
        parent_codon_idxs_v, mut_probs_v, sub_probs_v
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)
    return aaprobs_of_codon_probs_v(codon_probs)


def reshape_for_codons(array):
    """
    Reshape an array to add a codon dimension by taking groups of 3 sites.

    Parameters:
    array (np.ndarray): Original array.

    Returns:
    np.ndarray: Reshaped array with an added codon dimension.
    """
    site_count = array.shape[0]
    assert site_count % 3 == 0, "Site count must be a multiple of 3"
    codon_count = site_count // 3
    return array.reshape(codon_count, 3, *array.shape[1:])


def aaprobs_of_parent_rates_and_sub_probs(
    parent_idxs: np.ndarray, rates: np.ndarray, sub_probs: np.ndarray
) -> np.ndarray:
    """
    Calculate per-site amino acid probabilities from per-site nucleotide rates
    and substitution probabilities.

    Args:
        parent_idxs (np.ndarray): Parent nucleotide indices. Shape should be (site_count,).
        rates (np.ndarray): Poisson rates of mutation per site. Shape should be (site_count,).
        sub_probs (np.ndarray): Substitution probabilities per site: a 2D
                                array with shape (site_count, 4).

    Returns:
        np.ndarray: A 2D array with rows corresponding to sites and columns
                    corresponding to amino acids.
    """
    # Calculate the probability of at least one mutation at each site.
    mut_probs = 1.0 - np.exp(-rates)

    # Reshape the inputs to include a codon dimension.
    parent_codon_idxs_v = reshape_for_codons(parent_idxs)
    codon_mut_probs_v = reshape_for_codons(mut_probs)
    codon_subs_v = reshape_for_codons(sub_probs)

    # Vectorized calculation of amino acid probabilities.
    return aaprob_of_mut_and_sub_v(parent_codon_idxs_v, codon_mut_probs_v, codon_subs_v)


def build_codon_mutsel_v(
    parent_codon_idxs_v: np.ndarray,
    codon_mut_probs_v: np.ndarray,
    codon_sub_probs_v: np.ndarray,
    aa_sel_matrix_v: np.ndarray,
) -> np.ndarray:
    """
    Build a sequence of codon mutation-selection matrices for codons along a sequence.

    Args:
        parent_codon_idxs_v (np.ndarray): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs_v (np.ndarray): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_sub_probs_v (np.ndarray): The substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)
        aa_sel_matrix_v (np.ndarray): The amino-acid selection matrices for each sequence. Shape: (codon_count, 20)

    Returns:
        np.ndarray: The probability of mutating to each codon, for each sequence. Shape: (codon_count, 4, 4, 4)
    """
    mut_matrix_v = build_mutation_matrices(
        parent_codon_idxs_v, codon_mut_probs_v, codon_sub_probs_v
    )
    codon_probs_v = codon_probs_of_mutation_matrices(mut_matrix_v)

    # Calculate the codon selection matrix for each sequence via Einstein
    # summation, in which we sum over the repeated indices.
    # So, for each site (s) and codon (c), sum over amino acids (a):
    # codon_sel_matrix_v[s, c] = sum_a(CODON_AA_INDICATOR_MATRIX[c, a] * aa_sel_matrix_v[s, a])
    # Resulting shape is (S, C) where S is the number of sites and C is the number of codons.
    codon_sel_matrix_v = np.einsum(
        "ca,sa->sc", CODON_AA_INDICATOR_MATRIX, aa_sel_matrix_v
    )

    # Multiply the codon probabilities by the selection matrices
    codon_mutsel_v = codon_probs_v * codon_sel_matrix_v.reshape(-1, 4, 4, 4)

    # Normalize to get a probability distribution for each sequence
    codon_count = parent_codon_idxs_v.shape[0]

    # Now we need to recalculate the probability of staying in the same codon.
    # In our setup, this is the probability of nothing happening.
    # To calculate this, we zero out the previously calculated probabilities...
    codon_mutsel_v[(np.arange(codon_count), *parent_codon_idxs_v.T)] = 0.0
    # sum together their probabilities...
    sums = codon_mutsel_v.sum(axis=(1, 2, 3))
    # then set the parent codon probabilities to 1 minus the sum.
    codon_mutsel_v[(np.arange(codon_count), *parent_codon_idxs_v.T)] = 1.0 - sums

    return codon_mutsel_v
