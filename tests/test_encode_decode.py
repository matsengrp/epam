import h5py
from epam.utils import generate_file_checksum
from epam.models import AbLang1

infilename = "data/parent-child-example.csv"
outfilename = "tests/matrices.hdf5"


def test_encode_decode():
    # encode
    ablang_heavy = AbLang1(chain="heavy")
    ablang_heavy.write_aaprobs(infilename, outfilename)

    # decode
    with h5py.File(outfilename, "r") as testfile:
        pcp_filename = testfile.attrs["pcp_filename"]
        assert testfile.attrs["checksum"] == generate_file_checksum(pcp_filename)
