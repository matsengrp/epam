import h5py
import numpy as np
import torch
import pytest
import os
import epam.models
from epam.sequences import translate_sequences
from epam.models import (
    AbLang,
    SHMple,
    OptimizableSHMple,
    MutSel,
    ESM1v,
    SHMpleESM,
)

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


class MutSelThreonine(MutSel):

    """A mutation selection model with a selection matrix that loves Threonine."""

    def __init__(self, weights_directory, modelname="SillyMutSel"):
        super().__init__(weights_directory, modelname)

    def build_selection_matrix_from_parent(self, parent):
        matrix = torch.zeros((1, 20))
        # Set just the entry for tryptophan to be 0.3
        matrix[0, 18] = 0.3
        return matrix


ex_mut_probs = torch.tensor([0.01, 0.02, 0.03])
ex_sub_probs = torch.tensor(
    [[0.0, 0.3, 0.5, 0.2], [0.4, 0.0, 0.1, 0.5], [0.2, 0.3, 0.0, 0.5]]
)
ex_parent_codon = "ACG"


def test_mut_sel_probability():
    mutsel = MutSelThreonine(weights_path)
    # Note we're dividing by two here
    ex_mut_rates = -torch.log(1.0 - ex_mut_probs) / 2.0
    # This is an ACG -> TGG mutation.
    log_pcp_prob = mutsel._build_log_pcp_probability(
        ex_parent_codon, "TGG", ex_mut_rates, ex_sub_probs
    )
    #               A->T    C->G    G->G   P(Tryp)
    correct_prob = np.log(0.002 * 0.002 * 0.97 * 0.3)
    # Here we're using a branch scaling of two to accomodate the scaling by two above.
    calculated_prob = log_pcp_prob(torch.log(torch.tensor(2.0)))
    assert correct_prob == pytest.approx(calculated_prob, rel=1e-5)


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


def hdf5_files_identical(path_1, path_2, tol=1e-4):
    """Return if two HDF5 files are identical."""
    with h5py.File(path_1, "r") as f1, h5py.File(path_2, "r") as f2:
        for key in f1.keys():
            if key not in f2:
                print(f"Key {key} not in second file")
                return False

            d1 = f1[key]["data"]
            d2 = f2[key]["data"]

            if not np.allclose(d1, d2, rtol=tol):
                print(f"Data for key {key} not matching: {d1[...] - d2[...]}")
                return False

    return True


def test_snapshot():
    """Test that the current code produces the same results as a previously-built snapshot."""
    os.makedirs("_ignore", exist_ok=True)
    for model_name, model_class_str, model_args in epam.models.FULLY_SPECIFIED_MODELS:
        print(f"Snapshot testing {model_name}")
        source = "10-random-from-10x"
        ModelClass = getattr(epam.models, model_class_str)
        model = ModelClass(**model_args)
        # Because we're using a snapshot, we don't want to optimize:
        # optimization is fiddly and we want to be able to change it without
        # breaking the snapshot test.
        if isinstance(model, OptimizableSHMple):
            model.max_optimization_steps = 0
        out_file = f"_ignore/{source}-{model_name}.hdf5"
        model.write_aaprobs(f"data/{source}.csv", out_file)
        compare_file = f"tests/test-data/{source}-{model_name}.hdf5"
        assert hdf5_files_identical(out_file, compare_file)
