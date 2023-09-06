import pandas as pd
import os
from epam.utils import generate_file_checksum
from epam.models import AbLang
from epam.evaluation import *

example_pcp_path = "data/parent-child-example.csv" # "data/eval-parent-child-example.csv"
example_prob_mat_path = "tests/matrices.hdf5" # "tests/eval-matrices.hdf5"
example_model_eval_path = "tests/model-performance.csv"
correct_model_eval_path = "tests/eval-model-performance.csv"


def test_calculate_substitution_accuracy():
    ablang_heavy = AbLang(chain="heavy")
    ablang_heavy.write_probability_matrices(example_pcp_path, example_prob_mat_path)

    assert calculate_sub_accuracy(example_prob_mat_path) == 4 / 13


def test_evaluate():
    ablang_heavy = AbLang(chain="heavy")
    ablang_heavy.write_probability_matrices(example_pcp_path, example_prob_mat_path)

    evaluate(example_prob_mat_path, example_model_eval_path)
    assert os.path.exists(example_model_eval_path) == True

    assert generate_file_checksum(example_model_eval_path) == generate_file_checksum(
        correct_model_eval_path
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
