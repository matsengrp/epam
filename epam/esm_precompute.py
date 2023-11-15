"""
This module enables precomputation of ESM1v selection factors for a set of PCPs in bulk, and then loading those saved values into a dictionary.
"""
import numpy as np
import pandas as pd
import h5py
from epam import utils
from epam.sequences import (
    AA_STR_SORTED,
    translate_sequences,
    pcp_criteria_check,
    assert_pcp_lengths,
)
from esm import pretrained
import torch

model_location = "esm1v_t33_650M_UR90S_1"


def precompute_and_save(pcp_path, output_hdf5):
    """
    Precompute ESM1v selection factors for a full set of PCPs and save to an HDF5 file.

    pcp_path (str): Path to a CSV file containing PCP data.
    output_hdf5 (str): Path to the output HDF5 file.
    """

    # Check that CUDA is usable
    def check_CUDA():
        try:
            torch._C._cuda_init()
            return True
        except:
            return False

    # Set device based on system availability
    if torch.backends.cudnn.is_available() and check_CUDA():
        print("Using CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    # Initialize the model
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    model = model.to(device)
    aa_idxs = [alphabet.get_idx(aa) for aa in AA_STR_SORTED]

    batch_converter = alphabet.get_batch_converter()

    # Load in PCP data
    full_pcp_df = pd.read_csv(pcp_path, index_col=0)

    # Remove PCPs that do not meet criteria: 0% < mutation rate < 30% or mismatched lengths
    pcp_df = full_pcp_df[
        full_pcp_df.apply(
            lambda row: pcp_criteria_check(row["parent"], row["child"]), axis=1
        )
    ]

    pcp_df.apply(lambda row: assert_pcp_lengths(row["parent"], row["child"]), axis=1)

    # Keep only unique parent sequences (remove duplicate computations and entries in HDF5 file)
    pcp_df = pcp_df.drop_duplicates(subset=["parent"])

    # Translate parent sequences and format for ESM1v input
    sequences = pcp_df["parent"].tolist()
    sequences_aa = [translate_sequences([parent])[0] for parent in sequences]
    protein_ids = [f"protein{i}" for i in range(len(sequences))]

    data = list(zip(protein_ids, sequences_aa))
    batch_tokens = batch_converter(data)[2]

    # Get token probabilities before softmax so we can restrict to 20 amino
    # acids in softmax calculation.
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        token_probs_pre_softmax = model(batch_tokens)["logits"]

    aa_probs = torch.softmax(token_probs_pre_softmax[..., aa_idxs], dim=-1)

    aa_probs_np = aa_probs.cpu().numpy().squeeze()

    # Drop first and last elements, which are the probability of the start
    # and end token.
    prob_matrix = aa_probs_np[:, 1:-1, :]

    # Save model output to HDF5 file
    checksum = utils.generate_file_checksum(pcp_path)

    with h5py.File(output_hdf5, "w") as outfile:
        # attributes related to PCP data file
        outfile.attrs["checksum"] = checksum
        outfile.attrs["pcp_filename"] = pcp_path
        outfile.attrs["model_name"] = "ESM1v_bulk"

        for i in range(len(sequences)):
            matrix = prob_matrix[i, :, :]
            parent = sequences[i]
            outfile.create_dataset(
                f"{parent}", data=matrix, compression="gzip", compression_opts=4
            )


def load_and_convert_to_dict(hdf5_path):
    """
    Load precomputed ESM1v selection factors from an HDF5 file and convert to a dictionary.

    hdf5_path (str): Path to the HDF5 file.
    """
    with h5py.File(hdf5_path, "r") as infile:
        # initialize dictionary
        parent_esm_dict = {}

        # iterate through parent sequences and add to dictionary
        for parent in infile:
            parent_esm_dict[parent] = infile[parent][:]

    return parent_esm_dict
