import pytest
#import h5py
#import numpy as np
#import pandas as pd
from epam.evaluation import *

prob_mat_file = "tests/matrices.hdf5"
aa_str_sorted = 'ACDEFGHIKLMNPQRSTVWY'

def test_evaluate_substitution_accuracy():
    assert calculate_sub_accuracy(prob_mat_file) == 41/99

# need to think of way to test main function def test_evaluate():
# maybe test that file was written: https://stackoverflow.com/questions/20531072/writing-a-pytest-function-to-check-outputting-to-a-file-in-python
