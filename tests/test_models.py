import numpy as np
import pytest
from epam.models import AbLang

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
