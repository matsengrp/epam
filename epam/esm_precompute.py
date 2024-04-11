"""
This module enables precomputation of ESM-1v selection factors for a set of PCPs in bulk, and then loading those saved values into a dictionary. Different scoring schemes can be used to compute ESM-1v selection factors. See Meier et al. (2021) and https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L192-L225 for more details.
"""
import numpy as np
import pandas as pd
import torch
import h5py

from esm import pretrained

from epam.sequences import (
    AA_STR_SORTED,
    translate_sequences,
    assert_pcp_lengths,
    aa_idx_array_of_str,
)
from epam.torch_common import pick_device
from epam.utils import load_and_filter_pcp_df, generate_file_checksum

model_location = "esm1v_t33_650M_UR90S_1"


def precompute_and_save(pcp_path, output_hdf5, scoring_strategy):
    """
    Precompute ESM-1v selection factors for a full set of PCPs and save to an HDF5 file.

    pcp_path (str): Path to a CSV file containing PCP data.
    output_hdf5 (str): Path to the output HDF5 file.
    scoring_strategy (str): Scoring strategy to use for ESM-1v. Currently 'wt-marginals' and 'masked-marginals' are supported.
    """

    device = pick_device()

    # Initialize the model
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    model = model.to(device)
    aa_idxs = [alphabet.get_idx(aa) for aa in AA_STR_SORTED]

    batch_converter = alphabet.get_batch_converter()

    pcp_df = load_and_filter_pcp_df(pcp_path)

    pcp_df.apply(lambda row: assert_pcp_lengths(row["parent"], row["child"]), axis=1)

    # Keep only unique parent sequences (remove duplicate computations and entries in HDF5 file)
    pcp_df = pcp_df.drop_duplicates(subset=["parent"])

    # Translate parent sequences and format for ESM1v input
    sequences = pcp_df["parent"].tolist()
    sequences_aa = translate_sequences(sequences)
    protein_ids = [f"protein{i}" for i in range(len(sequences))]

    # Generate checksum for PCP file
    checksum = generate_file_checksum(pcp_path)

    if scoring_strategy == "wt-marginals":
        data = list(zip(protein_ids, sequences_aa))
        batch_tokens = batch_converter(data)[2]

        # Single forward pass on the full sequence.
        # Get token probabilities before softmax so we can restrict to 20 amino
        # acids in softmax calculation.
        with torch.no_grad():
            batch_tokens = batch_tokens.to(device)
            token_probs_pre_softmax = model(batch_tokens)["logits"]

        aa_probs = torch.softmax(token_probs_pre_softmax[..., aa_idxs], dim=-1)

        aa_probs_np = aa_probs.cpu().numpy().squeeze()

        # Save model output to HDF5 file
        with h5py.File(output_hdf5, "w") as outfile:
            # attributes related to PCP data file
            outfile.attrs["checksum"] = checksum
            outfile.attrs["pcp_filename"] = pcp_path
            outfile.attrs["model_name"] = f"ESM1v_bulk_{scoring_strategy}"

            for i in range(len(sequences_aa)):
                # Drop first and last element (adjusted for sequence length as ESM pads to largest seq len), which are the probability of the start
                # and end token.
                len_seq = len(sequences_aa[i])
                matrix = aa_probs_np[i, 1 : len_seq + 1, :]
                parent = sequences[i]
                outfile.create_dataset(
                    f"{parent}", data=matrix, compression="gzip", compression_opts=4
                )

    elif scoring_strategy == "masked-marginals":
        # Save model output to HDF5 file
        with h5py.File(output_hdf5, "w") as outfile:
            # attributes related to PCP data file
            outfile.attrs["checksum"] = checksum
            outfile.attrs["pcp_filename"] = pcp_path
            outfile.attrs["model_name"] = f"ESM1v_bulk_{scoring_strategy}"

            for seq in range(len(sequences_aa)):
                data = list(zip([protein_ids[seq]], [sequences_aa[seq]]))
                batch_tokens = batch_converter(data)[2]

                # Mask each site in the sequence to get token probabilities before softmax.
                all_token_probs = []
                for site in range(batch_tokens.size(1)):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, site] = alphabet.mask_idx

                    with torch.no_grad():
                        batch_tokens_masked = batch_tokens_masked.to(device)
                        token_probs_pre_softmax = model(batch_tokens_masked)["logits"]

                    all_token_probs.append(token_probs_pre_softmax[:, site])

                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

                aa_probs = torch.softmax(token_probs[..., aa_idxs], dim=-1)

                aa_probs_np = aa_probs.cpu().numpy().squeeze()

                # Drop first and last element (adjusted for sequence length as ESM pads to largest seq len), which are the probability of the start
                # and end token.
                len_seq = len(sequences_aa[seq])
                non_norm_matrix = aa_probs_np[1 : len_seq + 1, :]

                # Normalize by the probability of the parent AA at each site.
                parent_idx = aa_idx_array_of_str(sequences_aa[seq])
                parent_probs = non_norm_matrix[np.arange(len_seq), parent_idx]
                matrix = non_norm_matrix / parent_probs[:, None]

                parent = sequences[seq]

                outfile.create_dataset(
                    f"{parent}", data=matrix, compression="gzip", compression_opts=4
                )
    else:
        raise ValueError(
            f"Invalid scoring strategy: {scoring_strategy}. Must be 'wt-marginals' or 'masked-marginals'."
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
