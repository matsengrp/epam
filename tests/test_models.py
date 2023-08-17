import numpy as np
import pytest
from epam.sequences import translate_sequences
from epam.models import AbLang
from epam.models import SHMple

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


def test_shmple():
    shmple_shmoof = SHMple(weights_directory="data/shmple_weights/my_shmoof")
    prob_matrix = shmple_shmoof.prob_matrix_of_parent_child_pair(
        parent_nt_seq, child_nt_seq
    )
    child_aa_seq = translate_sequences([child_nt_seq])[0]
    prob_vec = shmple_shmoof.probability_vector_of_child_seq(prob_matrix, child_aa_seq)
    assert np.sum(prob_vec[:3]) > np.sum(prob_vec[3:])
