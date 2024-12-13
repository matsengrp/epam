import h5py
import numpy as np
import torch
import pytest
import os
from importlib import resources
import epam.models
import epam.gcreplay_models
from epam.esm_precompute import precompute_and_save, process_esm_output
from epam.esm_precompute import load_and_convert_to_dict
from netam.sequences import translate_sequence
from epam.models import MLMBase, AbLang1, MutModel, NetamSHM, MutSelModel, NetamSHMESM

parent_seqs = [
    "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVLGWGSMDVWGQGTTVTVSS",
]


def test_ablang():
    ablang_heavy = AbLang1()
    parent = parent_seqs[1]
    terrible_child = "A" * 50 + parent[50:]
    prob_arr = ablang_heavy.probability_array_of_seq(parent)
    prob_vec = ablang_heavy.probability_vector_of_child_seq(prob_arr, terrible_child)
    assert np.sum(prob_vec[:50]) < np.sum(prob_vec[50:100])


parent_nt_seq = "CAGGTGCAGCTGGTGGAG"  # QVQLVE
child_nt_seq = "CAGGCGCAGCCGGCGGAG"  # QAQPAE
crepe_path = "/fh/fast/matsen_e/shared/bcr-mut-sel/netam-shm/trained_models/cnn_ind_med-shmoof_small-full-0"


def test_netam():
    netam_oof = NetamSHM(model_path_prefix=crepe_path, optimize=False)
    aaprobs = netam_oof.aaprobs_of_parent_child_pair(parent_nt_seq, child_nt_seq)
    child_aa_seq = translate_sequence(child_nt_seq)
    prob_vec = netam_oof.probability_vector_of_child_seq(aaprobs, child_aa_seq)
    assert np.sum(prob_vec[:3]) > np.sum(prob_vec[3:])

    # When we optimize the branch length, we should have a higher probability overall.
    optimizable_netam = NetamSHM(model_path_prefix=crepe_path)
    opt_aaprobs = optimizable_netam.aaprobs_of_parent_child_pair(
        parent_nt_seq, child_nt_seq
    )
    opt_prob_vec = optimizable_netam.probability_vector_of_child_seq(
        opt_aaprobs, child_aa_seq
    )
    assert opt_prob_vec.prod() > prob_vec.prod()


class MutSelThreonine(MutSelModel):
    """A mutation selection model with a selection matrix that loves Threonine."""

    def __init__(self, crepe_path, modelname="SillyMutSel"):
        super().__init__(crepe_path, modelname)

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
    mutsel = MutSelThreonine(crepe_path)
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


def test_cached_esm_wt(tol=1e-4):
    source = "10-random-from-10x"
    pcp_file = f"data/{source}.csv"
    logit_file = f"_ignore/{source}_cached_wt.hdf5"
    prob_file = f"_ignore/{source}-wt_prob.hdf5"
    compare_file = f"data/{source}-wt_prob.hdf5"

    precompute_and_save(pcp_file, logit_file, "wt-marginals")
    process_esm_output(logit_file, prob_file, "wt-marginals")
    cached_esm_dict = load_and_convert_to_dict(prob_file)
    ref_esm_dict = load_and_convert_to_dict(compare_file)

    for key in cached_esm_dict.keys():
        assert np.allclose(ref_esm_dict[key], cached_esm_dict[key], rtol=tol)


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


pcp_hdf5_wt_path = str(resources.files("epam").parent) + "/data/10-random-from-10x-wt_prob.hdf5"
pcp_hdf5_mask_path = str(resources.files("epam").parent) + "/data/10-random-from-10x-mask_prob_ratio.hdf5"


def test_snapshot():
    """Test that the current code produces the same results as a previously-built snapshot."""
    os.makedirs("_ignore", exist_ok=True)

    for source in ["10-random-from-10x", "10-random-from-gcreplay"]:
        if source == "10-random-from-10x":
            models_list = epam.models.FULLY_SPECIFIED_MODELS
        elif source == "10-random-from-gcreplay":
            models_list = epam.gcreplay_models.GCREPLAY_MODELS

        for model_name, model_class_str, model_args in models_list:
            print(f"Snapshot testing {model_name}")
            if source == "10-random-from-10x":
                ModelClass = getattr(epam.models, model_class_str)
            elif source == "10-random-from-gcreplay":
                ModelClass = getattr(epam.gcreplay_models, model_class_str)
            model = ModelClass(**model_args)
            # Because we're using a snapshot, we don't want to optimize:
            # optimization is fiddly and we want to be able to change it without
            # breaking the snapshot test.
            if isinstance(model, (MutModel, MLMBase)):
                model.max_optimization_steps = 0
            out_file = f"_ignore/{source}-{model_name}.hdf5"
            if model_name in ("ESM1v_wt"):
                model.preload_esm_data(pcp_hdf5_wt_path)
            if model_name in (
                "ESM1v_mask",
                "S5FESM_mask",
                "NetamESM_mask",
            ):
                model.preload_esm_data(pcp_hdf5_mask_path)
            model.write_aaprobs(f"data/{source}.csv", out_file)
            compare_file = f"tests/test-data/{source}-{model_name}.hdf5"
            assert hdf5_files_identical(out_file, compare_file)
