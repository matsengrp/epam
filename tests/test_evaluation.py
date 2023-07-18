import pandas as pd
import os
from epam.utils import generate_file_checksum
from epam.models import AbLang
from epam.evaluation import *

infilename = "data/parent-child-example.csv"
outfilename = "tests/matrices.hdf5"
evalfilename = "tests/model_performance.csv"

# this test is probably not necessary
def test_calculate_substitution_accuracy():
    # encode
    ablang_heavy = AbLang(chain="heavy")
    ablang_heavy.write_probability_matrices(infilename, outfilename)

    # evaluate metrics:
    # substitution accuracy
    assert calculate_sub_accuracy(outfilename) == 41/99

# simple test of main function, should replace with something smarter and check file contents
def test_evaluate():
    # encode example
    ablang_heavy = AbLang(chain="heavy")
    ablang_heavy.write_probability_matrices(infilename, outfilename)

    # ensure evaluation script produces csv:
    evaluate(outfilename, evalfilename)
    assert os.path.exists(evalfilename) == True
    