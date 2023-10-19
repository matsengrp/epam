import numpy as np
import epam.molevol as molevol
from epam.models import SHMple

from epam.sequences import (
    nt_idx_array_of_str,
    translate_sequences,
    AA_STR_SORTED,
    CODONS,
    NT_STR_SORTED,
)

# These happen to be the same as some examples in test_models.py but that's fine.
# If it was important that they were shared, we should put them in a conftest.py.
ex_mut_probs = np.array([0.01, 0.02, 0.03])
ex_sub_probs = np.array(
    [[0.0, 0.3, 0.5, 0.2], [0.4, 0.0, 0.1, 0.5], [0.2, 0.3, 0.0, 0.5]]
)
ex_parent_codon_idxs = nt_idx_array_of_str("ACG")
parent_nt_seq = "CAGGTGCAGCTGGTGGAG"  # QVQLVE
weights_path = "data/shmple_weights/my_shmoof"


def test_build_mutation_matrix():
    correct_tensor = np.array(
        [
            # probability of mutation to each nucleotide (first entry in the first row
            # is probability of no mutation)
            [0.99, 0.003, 0.005, 0.002],
            [0.008, 0.98, 0.002, 0.01],
            [0.006, 0.009, 0.97, 0.015],
        ]
    )
    assert np.allclose(
        correct_tensor,
        molevol.build_mutation_matrix(ex_parent_codon_idxs, ex_mut_probs, ex_sub_probs),
    )


def test_normalize_sub_probs():
    parent_idxs = nt_idx_array_of_str("AC")
    sub_probs = np.array([[0.2, 0.3, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4]])

    expected_normalized = np.array([[0.0, 0.375, 0.5, 0.125], [0.125, 0.0, 0.375, 0.5]])

    normalized_sub_probs = molevol.normalize_sub_probs(parent_idxs, sub_probs)

    assert normalized_sub_probs.shape == (2, 4), "Result has incorrect shape"
    np.testing.assert_allclose(
        normalized_sub_probs, expected_normalized, rtol=1e-6
    ), "Unexpected normalized values"


def iterative_aaprob_of_mut_and_sub(parent_codon, mut_probs, sub_probs):
    """
    Original version of codon_to_aa_probabilities, used for testing.
    """
    aa_probs = {}
    for aa in AA_STR_SORTED:
        aa_probs[aa] = 0

    # iterate through all possible child codons
    for child_codon in CODONS:
        try:
            aa = translate_sequences([child_codon])[0]
        except ValueError:  # check for STOP codon
            continue

        # iterate through codon sites and compute total probability of potential child codon
        child_prob = 1.0
        for isite in range(3):
            if parent_codon[isite] == child_codon[isite]:
                child_prob *= 1.0 - mut_probs[isite]
            else:
                child_prob *= mut_probs[isite]
                child_prob *= sub_probs[isite][NT_STR_SORTED.index(child_codon[isite])]

        aa_probs[aa] += child_prob

    # need renormalization factor so that amino acid probabilities sum to 1,
    # since probabilities to STOP codon are dropped
    psum = np.sum([aa_probs[aa] for aa in aa_probs.keys()])

    return np.array([aa_probs[aa] / psum for aa in AA_STR_SORTED])


def test_aaprob_of_mut_and_sub():
    shmple_shmoof = SHMple(weights_directory=weights_path)
    [rates], [subs] = shmple_shmoof.model.predict_mutabilities_and_substitutions(
        [parent_nt_seq], [0.1]
    )
    mut_probs = 1.0 - np.exp(-rates.squeeze())
    parent_codon = parent_nt_seq[0:3]
    codon_mut_probs = mut_probs[0:3]
    codon_subs = subs[0:3]

    assert np.allclose(
        iterative_aaprob_of_mut_and_sub(parent_codon, codon_mut_probs, codon_subs),
        molevol.aaprob_of_mut_and_sub(
            nt_idx_array_of_str(parent_codon), codon_mut_probs, codon_subs
        ),
    )
