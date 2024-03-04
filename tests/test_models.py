import h5py
import numpy as np
import torch
import pytest
import os
from importlib import resources
import epam.models
from epam.esm_precompute import precompute_and_save
from epam.esm_precompute import load_and_convert_to_dict
from epam.sequences import translate_sequence
from epam.models import (
    AbLang,
    SHMple,
    OptimizableSHMple,
    MutSel,
    SHMpleESM,
    WrappedBinaryMutSel,
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
    child_aa_seq = translate_sequence(child_nt_seq)
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


def test_cached_esm():
    source = "10-random-from-10x"
    pcp_file = f"data/{source}.csv"
    hdf5_file = f"_ignore/{source}_cached.hdf5"
    compare_file = f"data/{source}.hdf5"

    precompute_and_save(pcp_file, hdf5_file)
    cached_esm_dict = load_and_convert_to_dict(hdf5_file)
    ref_esm_dict = load_and_convert_to_dict(compare_file)

    for key in cached_esm_dict.keys():
        assert np.allclose(ref_esm_dict[key], cached_esm_dict[key])


def test_nasty_shmple_esm():
    bad_pcp_file = "data/wyatt_10x_loss_nan.csv"
    bad_esm_file = "_ignore/wyatt_10x_loss_nan_cached_esm.hdf5"
    bad_out_file = "_ignore/wyatt_10x_loss_nan.hdf5"
    precompute_and_save(bad_pcp_file, bad_esm_file)
    shmple_esm = SHMpleESM(weights_directory=weights_path)
    shmple_esm.preload_esm_data(bad_esm_file)
    shmple_esm.write_aaprobs(bad_pcp_file, bad_out_file)


class ConserveEverythingExceptTyrosine:
    """
    A fake selection model in which everything exept tyrosine is perfectly
    conserved: Y is assigned a selection coefficient of 1, and all other amino
    acids are assigned a selection coefficient of 0.
    """

    def selection_factors_of_aa_str(self, aa_str):
        return torch.tensor(
            [1.0 if aa_str[i] == "Y" else 0.0 for i in range(len(aa_str))]
        )


@pytest.fixture
def wrapped_not_tyrosine():
    not_tyrosine = ConserveEverythingExceptTyrosine()
    return WrappedBinaryMutSel(not_tyrosine, weights_directory=weights_path)


def test_wrapped_binary_mut_sel(wrapped_not_tyrosine):
    nt_seq = "GCTTAT"
    assert translate_sequence(nt_seq) == "AY"
    selection_matrix = wrapped_not_tyrosine.build_selection_matrix_from_parent(nt_seq)
    assert torch.allclose(selection_matrix.sum(axis=1), torch.tensor([1., 20.]))
    assert selection_matrix[0, 0] == 1.0
    assert (selection_matrix <= 1.0).all()


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


with resources.path("epam", "__init__.py") as p:
    pcp_hdf5_path = str(p.parent.parent) + "/data/10-random-from-10x.hdf5"


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
        if isinstance(model, (OptimizableSHMple, AbLang)):
            model.max_optimization_steps = 0
        out_file = f"_ignore/{source}-{model_name}.hdf5"
        if model_name in ("ESM1v_default", "SHMple_ESM1v"):
            model.preload_esm_data(pcp_hdf5_path)
        model.write_aaprobs(f"data/{source}.csv", out_file)
        compare_file = f"tests/test-data/{source}-{model_name}.hdf5"
        assert hdf5_files_identical(out_file, compare_file)
