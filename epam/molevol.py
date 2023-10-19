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


def build_mutation_matrix(
    parent_codon_idxs: np.ndarray, mut_probs: np.ndarray, sub_probs: np.ndarray
) -> np.ndarray:
    """
    Generate a 3x4 mutation matrix for a given parent codon.
    
    Given indices for a parent codon, Poisson mutation rates, and substitution probabilities, 
    this function constructs a 3x4 matrix. The matrix represents the mutation probabilities for 
    each nucleotide position in the parent codon. The ijth entry of the matrix corresponds to the 
    probability of the ith nucleotide mutating to the jth nucleotide (in indices).
    
    Parameters:
    parent_codon_idxs (np.ndarray): Indices representing the parent codon's nucleotides. 
                                    Must be a 1D array of length 3.
    mut_probs (np.ndarray): 1D array representing the mutation probabilities for each site in the codon.
    sub_probs (np.ndarray): 2D array representing substitution probabilities for each site.
                            Rows correspond to sites, and columns correspond to nucleotide indices.
                            
    Returns:
    np.ndarray: A 3x4 array where the ijth entry is the mutation probability of the ith position in the 
                parent codon mutating to the jth nucleotide.
                
    Example:
    See tests for an example.
    """
    assert len(parent_codon_idxs) == 3, "Parent codon must be length 3"
    result_matrix = np.empty((3, 4))

    # Create a mask where each row has True at the position matching parent_codon_idxs.
    mask_same_nt = np.arange(4) == parent_codon_idxs[:, np.newaxis]

    # Assign values where the nucleotide is the same.
    result_matrix[mask_same_nt] = 1.0 - mut_probs

    # Assign values where the nucleotide is different via broadcasting.
    mask_diff_nt = ~mask_same_nt
    result_matrix[mask_diff_nt] = (mut_probs[:, np.newaxis] * sub_probs)[mask_diff_nt]

    return result_matrix



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
    assert mut_matrix.shape[1] == 3, "The second dimension of the input mut_matrix should be 3 to represent the 3 positions in a codon"
    assert mut_matrix.shape[2] == 4, "The last dimension of the input mut_matrix should be 4 to represent the 4 nucleotides"
    
    # The key to understanding how this works is that when these arrays are
    # multiplied, NumPy broadcasts them into a common shape (n_sites, 4, 4, 4),
    # performing element-wise multiplication for each slice along the first axis
    # (i.e., for each site).
    
    return (
        mut_matrix[:, 0, :, np.newaxis, np.newaxis]
        * mut_matrix[:, 1, np.newaxis, :, np.newaxis]
        * mut_matrix[:, 2, np.newaxis, np.newaxis, :]
    )


def aaprobs_of_codon_probs(codon_probs: np.ndarray) -> np.ndarray:
    """
    Compute the probability of each amino acid from the probability of each codon.

    Parameters:
    codon_probs (numpy.ndarray): A 3D array representing the probability of mutating
                                to each codon. The codon probability array should
                                be formatted as per _codon_probs_of_mutation_matrix.

    Returns:
    numpy.ndarray: A 2D array where the ij-th entry is the probability of mutating
                to the amino acid j from the codon i.
    """
    # reshape(-1) flattens the array
    aaprobs = codon_probs.reshape(-1) @ CODON_AA_INDICATOR_MATRIX
    aaprobs /= aaprobs.sum()
    return aaprobs


def aaprob_of_mut_and_sub_v(
    parent_codon_idxs_v: np.ndarray, mut_probs_v: np.ndarray, sub_probs_v: np.ndarray
) -> np.ndarray:
    """
    For a specified codon and given nucleotide mutability and substitution probabilities,
    compute the amino acid substitution probabilities.

    Here we translate those into amino acid probabilities, which are normalized.
    Probabilities to stop codons are dropped, but self probabilities are kept.

    Parameters:
    parent_codon_idxs (np.ndarray): The specified codon.
    mut_probs (np.ndarray): The mutability probabilities for each site in the codon.
    sub_probs (np.ndarray): The substitution probabilities for each site in the codon.

    Returns:
    np.ndarray: An array of probabilities for all 20 amino acids.

    """
    mut_matrix = build_mutation_matrices(parent_codon_idxs_v, mut_probs_v, sub_probs_v)
    codon_probs = codon_probs_of_mutation_matrix(mut_matrix)
    return aaprobs_of_codon_probs(codon_probs)


def aaprobs_of_parent_rates_and_sub_probs(
    parent_idxs: np.ndarray, rates: np.ndarray, sub_probs: np.ndarray
) -> np.ndarray:
    """Calculate per-site amino acid probabilities from per-site rates and
    substitution probabilities.

    Args:
        parent_idxs (np.ndarray): Parent nucleotide indices.
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

    aaprobs = []

    for i in range(0, len(parent_idxs), 3):
        parent_codon_idxs = parent_idxs[i : i + 3]
        codon_mut_probs = mut_probs[i : i + 3]
        codon_subs = sub_probs[i : i + 3]

        site_probs = aaprob_of_mut_and_sub(
            parent_codon_idxs, codon_mut_probs, codon_subs
        )
        aaprobs.append(site_probs)

    return np.array(aaprobs)


def build_codon_mutsel(
    parent_codon_idxs: np.ndarray,
    codon_mut_probs: np.ndarray,
    codon_sub_probs: np.ndarray,
    aa_sel_matrix: np.ndarray,
) -> np.ndarray:
    """Build a codon mutation-selection matrix from mutation and substitution
    matrices on the nucleotide level, as well as a selection matrix on the amino
    acid level.

    Args:
        parent_codon_idxs (np.ndarray): The parent codon.
        codon_mut_probs (np.ndarray): The mutation probabilities for each site in the codon.
        codon_sub_probs (np.ndarray): The substitution matrices for each site in the codon.
        aa_sel_matrix (n.ndarray): The amino-acid selection matrix.

    Returns:
        np.ndarray: The probability of mutating to each codon, expressed as a 4x4x4 array.
    """

    # This implementation is somewhat inefficient because we do the
    # calculation for all of the possible codons every time even
    # though we only use it for the indicated child codon. However,
    # most of the time the parent and the child codons will be the
    # same, and we need to calculate probabilities for every codon
    # in that case (see below).

    mut_matrix = build_mutation_matrix(
        parent_codon_idxs, codon_mut_probs, codon_sub_probs
    )
    codon_probs = codon_probs_of_mutation_matrix(mut_matrix)

    # Note that because there are no nonzero entries that correspond
    # to stop, these will have selection probability 0.
    codon_sel_matrix = CODON_AA_INDICATOR_MATRIX @ aa_sel_matrix
    codon_mutsel = codon_probs * codon_sel_matrix.reshape(4, 4, 4)

    # Now we need to calculate the probability of no change in the
    # codon so that we can normalize to get a probability
    # distribution.

    codon_mutsel[tuple(parent_codon_idxs)] = 0.0
    codon_mutsel[tuple(parent_codon_idxs)] = 1.0 - codon_mutsel.sum()

    return codon_mutsel
