import numpy as np
import pytest
from epam.sequences import translate_sequences, AA_STR_SORTED, CODONS, NT_STR_SORTED
from epam.models import AbLang, SHMple, OptimizableSHMple, RandomMutSel, MutSel, ESM1v, SHMpleESM

parent_seqs = [
    "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVLGWGSMDVWGQGTTVTVSS",
]


def test_ablang():
    ablang_heavy = AbLang()
    parent = parent_seqs[1]
    terrible_child = "A" * 50 + parent[50:]
    prob_arr = ablang_heavy.probability_array_of_seq(parent)
    prob_vec = ablang_heavy.probability_vector_of_child_seq(prob_arr, terrible_child)
    assert np.sum(prob_vec[:50]) < np.sum(prob_vec[50:100])


parent_nt_seq = "CAGGTGCAGCTGGTGGAG"  # QVQLVE
child_nt_seq = "CAGGCGCAGCCGGCGGAG"  # QAQPAE
weights_path = "data/shmple_weights/my_shmoof"


def test_shmple():
    shmple_shmoof = SHMple(weights_directory=weights_path)
    aaprobs = shmple_shmoof.aaprobs_of_parent_child_pair(parent_nt_seq, child_nt_seq)
    child_aa_seq = translate_sequences([child_nt_seq])[0]
    prob_vec = shmple_shmoof.probability_vector_of_child_seq(aaprobs, child_aa_seq)
    assert np.sum(prob_vec[:3]) > np.sum(prob_vec[3:])

    # When we optimize the branch length, we should have a higher probability overall.
    optimizable_shmple = OptimizableSHMple(weights_directory=weights_path)
    opt_aaprobs = optimizable_shmple.aaprobs_of_parent_child_pair(
        parent_nt_seq, child_nt_seq
    )
    opt_prob_vec = optimizable_shmple.probability_vector_of_child_seq(
        opt_aaprobs, child_aa_seq
    )
    assert opt_prob_vec.prod() > prob_vec.prod()


ex_mut_probs = np.array([0.01, 0.02, 0.03])
ex_sub_probs = np.array(
    [[0.0, 0.3, 0.5, 0.2], [0.4, 0.0, 0.1, 0.5], [0.2, 0.3, 0.0, 0.5]]
)
ex_parent_codon = "ACG"


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
        SHMple._build_mutation_matrix(ex_parent_codon, ex_mut_probs, ex_sub_probs),
    )


def test_normalize_sub_probs():
    parent = "AC"
    sub_probs = np.array([[0.2, 0.3, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4]])

    expected_normalized = np.array([[0.0, 0.375, 0.5, 0.125], [0.125, 0.0, 0.375, 0.5]])

    normalized_sub_probs = SHMple._normalize_sub_probs(parent, sub_probs)

    assert normalized_sub_probs.shape == (2, 4), "Result has incorrect shape"
    np.testing.assert_allclose(
        normalized_sub_probs, expected_normalized, rtol=1e-6
    ), "Unexpected normalized values"


def iterative_codon_to_aa_probabilities(parent_codon, mut_probs, sub_probs):
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


def test_codon_to_aa_probabilities():
    shmple_shmoof = SHMple(weights_directory=weights_path)
    [rates], [subs] = shmple_shmoof.model.predict_mutabilities_and_substitutions(
        [parent_nt_seq], [0.1]
    )
    mut_probs = 1.0 - np.exp(-rates.squeeze())
    parent_codon = parent_nt_seq[0:3]
    codon_mut_probs = mut_probs[0:3]
    codon_subs = subs[0:3]

    assert np.allclose(
        iterative_codon_to_aa_probabilities(parent_codon, codon_mut_probs, codon_subs),
        SHMple.codon_to_aa_probabilities(parent_codon, codon_mut_probs, codon_subs),
    )


def test_mutsel():
    mutsel = RandomMutSel(weights_directory=weights_path)
    opt_aaprobs = mutsel.aaprobs_of_parent_child_pair(parent_nt_seq, child_nt_seq)


class MutSelThreonine(MutSel):

    """A mutation selection model with a selection matrix that loves Threonine."""

    def __init__(self, weights_directory, modelname="SillyMutSel"):
        super().__init__(weights_directory, modelname)

    def build_selection_matrix_from_parent(self, parent):
        matrix = np.zeros((1, 20))
        # Set just the entry for tryptophan to be 0.3
        matrix[0, 18] = 0.3
        return matrix


def test_mut_sel_probability():
    mutsel = MutSelThreonine(weights_path)
    # Note we're dividing by two here
    ex_mut_rates = -np.log(1.0 - ex_mut_probs) / 2.0
    # This is an ACG -> TGG mutation.
    neg_pcp_prob = mutsel._build_neg_pcp_probability(
        ex_parent_codon, "TGG", ex_mut_rates, ex_sub_probs
    )
    #               A->T    C->G    G->G   P(Tryp)
    correct_prob = -0.002 * 0.002 * 0.97 * 0.3
    # Here we're using a branch scaling of two to accomodate the scaling by two above.
    assert correct_prob == pytest.approx(neg_pcp_prob(np.log(2.0)))


def test_esm():
    esm_v1 = ESM1v()
    aaprobs = esm_v1.aaprobs_of_parent_child_pair(parent_nt_seq)
    child_aa_seq = translate_sequences([child_nt_seq])[0]
    prob_vec = esm_v1.probability_vector_of_child_seq(aaprobs, child_aa_seq)
    assert np.sum(prob_vec[:3]) > np.sum(prob_vec[3:])

def test_shmple_esm():
    shmple_esm = SHMpleESM(weights_directory=weights_path)
    aaprobs = shmple_esm.aaprobs_of_parent_child_pair(parent_nt_seq, child_nt_seq)
    child_aa_seq = translate_sequences([child_nt_seq])[0]
    prob_vec = shmple_esm.probability_vector_of_child_seq(aaprobs, child_aa_seq)
    assert np.sum(prob_vec[:3]) > np.sum(prob_vec[3:])