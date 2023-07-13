import h5py
import pandas as pd
from epam.utils import generate_file_checksum, produce_probability_matrices
from epam.models import AbLang

infilename = "../data/parent-child-example.csv"
outfilename = "matrices.hdf5"

ablang_heavy = AbLang(chain="heavy")
produce_probability_matrices(ablang_heavy, infilename, outfilename)

test_site = 23
test_aa = "P"
prob_aa_index = ablang_heavy.aa_str_sorted.index(test_aa)

print(ablang_heavy.aa_str_sorted, prob_aa_index)
with h5py.File(outfilename, "r") as testfile:
    pcp_filename = testfile.attrs["pcp_filename"]
    if testfile.attrs["checksum"] != generate_file_checksum(pcp_filename):
        raise ValueError("checksum fail!")

    pcp_filename = testfile.attrs["pcp_filename"]
    pcp_df = pd.read_csv(pcp_filename, index_col=0)

    for key in testfile.keys():
        grp = testfile[key]
        index = grp.attrs["pcp_index"]
        row = pcp_df.loc[index]
        print(
            index,
            row["sample_id"],
            row["family"],
            row["v_gene"],
            f"prob({test_aa},{test_site}) =",
            grp["data"][prob_aa_index, test_site],
        )
