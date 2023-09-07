import pandas as pd
import os
import numpy as np
from epam.utils import generate_file_checksum
from epam.models import AbLang
from epam.evaluation import *

example_pcp_path = "data/parent-child-example.csv"
example_prob_mat_path = "tests/matrices.hdf5"
example_model_eval_path = "tests/model-performance.csv"
correct_model_eval_path = "tests/eval-model-performance.csv"


def test_evaluate():
    ablang_heavy = AbLang(chain="heavy")
    ablang_heavy.write_probability_matrices(example_pcp_path, example_prob_mat_path)

    evaluate(example_prob_mat_path, example_model_eval_path)
    assert os.path.exists(example_model_eval_path) == True

    assert generate_file_checksum(example_model_eval_path) == generate_file_checksum(
        correct_model_eval_path
    )


def test_locate_child_substitutions():
    parent_aa = "QVQLVE"
    child_aa = "QVRLSE"

    # case with observed substitutions
    assert np.array_equal(
        locate_child_substitutions(parent_aa, child_aa), np.array([2, 4])
    )

    # case with no observed substitutions
    assert np.array_equal(
        locate_child_substitutions(parent_aa, parent_aa), np.array([])
    )


def test_identify_child_substitutions():
    parent_aa = "QVQLVE"
    child_aa = "QVRLSE"

    # case with observed substitutions
    assert np.array_equal(
        identify_child_substitutions(parent_aa, child_aa), np.array(["R", "S"])
    )

    # case with no observed substitutions
    assert np.array_equal(
        identify_child_substitutions(parent_aa, parent_aa), np.array([])
    )


def test_calculate_site_substitution_probabilities():
    example_matrix = np.array(
        [
            [0.8, 0.1],  # A
            [0, 0],
            [0.1, 0],  # D
            [0, 0],
            [0, 0],
            [0, 0.5],  # G
            [0, 0],
            [0, 0],
            [0.05, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0.4],  # Q
            [0, 0],
            [0.05, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    )

    parent_aa = "AQ"

    assert np.allclose(
        calculate_site_substitution_probabilities(example_matrix, parent_aa),
        np.array([0.2, 0.6]),
    )


def test_highest_ranked_substitution():
    example_matrix = np.array(
        [
            [0.6, 0.1],  # A
            [0, 0],
            [0.3, 0],  # D
            [0, 0],
            [0, 0],
            [0.1, 0.5],  # G
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0.4],  # Q
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    )

    parent_aa = "AQ"

    # case when highest predicted aa is parent aa, select next highest
    assert highest_ranked_substitution(example_matrix[:, 0], parent_aa, 0) == "D"

    # case when highest predicted aa is not parent aa, select highest
    assert highest_ranked_substitution(example_matrix[:, 1], parent_aa, 1) == "G"


def test_locate_top_k_substitutions():
    example_site_sub_probs = np.array([0.2, 0.6, 0.1, 0.5, 0.9])

    # note argpartition returns indices of top k elements in unsorted order
    # case when k_sub > 0
    assert np.array_equal(
        np.sort(locate_top_k_substitutions(example_site_sub_probs, 2)), np.array([1, 4])
    )

    # case when k_sub = 0
    assert np.array_equal(
        locate_top_k_substitutions(example_site_sub_probs, 0), np.array([])
    )


def test_calculate_substitution_accuracy():
    pcp_sub_aa_ids = [
        np.array(["R", "S"]),
        np.array(["Q", "V", "D"]),
        np.array(["P"]),
        np.array([]),
    ]

    model_sub_aa_ids = [
        np.array(["S", "S"]),
        np.array(["Q", "M", "W"]),
        np.array(["P"]),
        np.array([]),
    ]

    k_subs = [len(pcp_sub_aa_id) for pcp_sub_aa_id in pcp_sub_aa_ids]

    assert calculate_sub_accuracy(pcp_sub_aa_ids, model_sub_aa_ids, k_subs) == 0.5


def test_calculate_r_precision():
    pcp_sub_locations = [
        np.array([1, 3]),
        np.array([0, 4, 6]),
        np.array([30]),
        np.array([]),
    ]

    model_sub_locations = [
        np.array([1, 5]),
        np.array([2, 4, 18]),
        np.array([54]),
        np.array([]),
    ]

    k_subs = [len(pcp_sub_loc) for pcp_sub_loc in pcp_sub_locations]

    assert (
        calculate_r_precision(pcp_sub_locations, model_sub_locations, k_subs) == 2 / 6
    )


def test_calculate_cross_entropy_loss():
    pcp_sub_locations = [
        np.array([1, 3]),
        np.array([0, 2, 4]),
        np.array([]),
    ]

    site_sub_probs = [
        np.array([0.8, 0.3, 0.2, 0.5, 0.9]),
        np.array([0.2, 0.2, 0.9, 0.7, 0.1]),
        np.array([0.6, 0.2, 0.4, 0.6, 0.1]),
    ]

    assert (
        calculate_cross_entropy_loss(pcp_sub_locations, site_sub_probs)
        == 0.943246504856046
    )
