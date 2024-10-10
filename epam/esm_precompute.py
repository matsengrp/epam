"""
This module enables precomputation of ESM-1v selection factors for a set of PCPs in bulk, and then loading those saved values into a dictionary. Different scoring schemes can be used to compute ESM-1v selection factors. See Meier et al. (2021) and https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L192-L225 for more details.
"""

import numpy as np
import pandas as pd
import torch
import h5py

from esm import pretrained

from netam.sequences import (
    AA_STR_SORTED,
    translate_sequence,
    assert_pcp_lengths,
    aa_idx_array_of_str,
)
from netam.common import pick_device
from epam.utils import load_and_filter_pcp_df, generate_file_checksum


def precompute_and_save(pcp_path, output_hdf5, scoring_strategy, model_number=1):
    """
    Precompute ESM-1v logits for a full set of PCPs and save to an HDF5 file.

    pcp_path (str): Path to a CSV file containing PCP data.
    output_hdf5 (str): Path to the output HDF5 file.
    scoring_strategy (str): Scoring strategy to use for ESM-1v. Currently 'wt-marginals' and 'masked-marginals' are supported.
    model_number (int): Identifier of model used from ESM1v ensemble. Must be between 1 and 5.
    """

    assert model_number in {
        1,
        2,
        3,
        4,
        5,
    }, "Invalid ESM1v model number. Select a number between 1 and 5."
    model_location = f"esm1v_t33_650M_UR90S_{model_number}"

    print(f"Using ESM1v model number: {model_number}")

    device = pick_device()

    # Initialize the model
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    model = model.to(device)
    aa_idxs = [alphabet.get_idx(aa) for aa in AA_STR_SORTED]

    batch_converter = alphabet.get_batch_converter()

    pcp_df = load_and_filter_pcp_df(pcp_path)

    pcp_df.apply(lambda row: assert_pcp_lengths(row["parent"], row["child"]), axis=1)

    pcp_df["parent_aa"] = pcp_df.apply(
        lambda row: translate_sequence(row["parent"]), axis=1
    )

    # Keep only unique parent sequences (remove duplicate computations and entries in HDF5 file)
    pcp_df = pcp_df.drop_duplicates(subset=["parent_aa"])

    # Translate parent sequences and format for ESM1v input
    sequences = pcp_df["parent"].tolist()
    sequences_aa = pcp_df["parent_aa"].tolist()
    assert len(sequences) == len(sequences_aa)
    protein_ids = [f"protein{i}" for i in range(len(sequences))]

    # Generate checksum for PCP file
    checksum = generate_file_checksum(pcp_path)

    if scoring_strategy == "wt-marginals":
        data = list(zip(protein_ids, sequences_aa))
        batch_tokens = batch_converter(data)[2]

        # Single forward pass on the full sequence.
        # Get token logits before softmax so we can restrict to 20 amino
        # acids.
        with torch.no_grad():
            batch_tokens = batch_tokens.to(device)
            token_probs_pre_softmax = model(batch_tokens)["logits"]

        aa_logits = token_probs_pre_softmax[..., aa_idxs]

        aa_logits_np = aa_logits.cpu().numpy().squeeze()

        # Save model output to HDF5 file
        with h5py.File(output_hdf5, "w") as outfile:
            # Attributes related to PCP data file
            outfile.attrs["checksum"] = checksum
            outfile.attrs["pcp_filename"] = pcp_path
            outfile.attrs["model_name"] = f"ESM1v_{model_number}_{scoring_strategy}"

            for i in range(len(sequences_aa)):
                # Drop first and last element (adjusted for sequence length as ESM pads to largest seq len), which are the probability of the start
                # and end token.
                len_seq = len(sequences_aa[i])
                matrix = aa_logits_np[i, 1 : len_seq + 1, :]
                parent = sequences_aa[i]
                outfile.create_dataset(
                    f"{parent}", data=matrix, compression="gzip", compression_opts=4
                )

    elif scoring_strategy == "masked-marginals":
        # Save model output to HDF5 file
        with h5py.File(output_hdf5, "w") as outfile:
            # Attributes related to PCP data file
            outfile.attrs["checksum"] = checksum
            outfile.attrs["pcp_filename"] = pcp_path
            outfile.attrs["model_name"] = f"ESM1v_{model_number}_{scoring_strategy}"

            for seq in range(len(sequences_aa)):
                data = list(zip([protein_ids[seq]], [sequences_aa[seq]]))
                batch_tokens = batch_converter(data)[2]

                # Mask each site in the sequence to get token probabilities before softmax.
                all_token_logits = []
                for site in range(batch_tokens.size(1)):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, site] = alphabet.mask_idx

                    with torch.no_grad():
                        batch_tokens_masked = batch_tokens_masked.to(device)
                        token_probs_pre_softmax = model(batch_tokens_masked)["logits"]

                    all_token_logits.append(token_probs_pre_softmax[:, site])

                token_logits = torch.cat(all_token_logits, dim=0).unsqueeze(0)

                aa_logits = token_logits[..., aa_idxs]

                aa_logits_np = aa_logits.cpu().numpy().squeeze()

                # Drop first and last element (adjusted for sequence length as ESM pads to largest seq len), which are the probability of the start
                # and end token.
                len_seq = len(sequences_aa[seq])
                matrix = aa_logits_np[1 : len_seq + 1, :]

                parent = sequences_aa[seq]

                outfile.create_dataset(
                    f"{parent}", data=matrix, compression="gzip", compression_opts=4
                )
    else:
        raise ValueError(
            f"Invalid scoring strategy: {scoring_strategy}. Must be 'wt-marginals' or 'masked-marginals'."
        )


def process_esm_output(logit_hdf5_path, probability_hdf5_path, scoring_strategy):
    """
    Take ESM-1v logits and convert to probabilities or probability ratios based on scoring strategy.

    logit_hdf5_path (str): Path to the HDF5 file containing ESM-1v logits.
    probability_hdf5_path (str): Path to the output HDF5 file.
    scoring_strategy (str): Scoring strategy to use for ESM-1v. Currently 'wt-marginals' and 'masked-marginals' are supported.
    """

    # Load logits from HDF5 file
    parent_logit_dict = load_and_convert_to_dict(logit_hdf5_path)

    # Get attributes from logit HDF5 file
    with h5py.File(logit_hdf5_path, "r") as infile:
        checksum = infile.attrs["checksum"]
        pcp_filename = infile.attrs["pcp_filename"]
        model_name = infile.attrs["model_name"]

    with h5py.File(probability_hdf5_path, "w") as outfile:
        # Attributes related to PCP data file
        outfile.attrs["checksum"] = checksum
        outfile.attrs["pcp_filename"] = pcp_filename
        outfile.attrs["model_name"] = model_name

        for parent in parent_logit_dict:
            logit_matrix = parent_logit_dict[parent]

            # Convert logits to probabilities with softmax
            logit_tensor = torch.tensor(logit_matrix)
            prob_tensor = torch.softmax(logit_tensor, dim=-1)
            prob_matrix = prob_tensor.numpy().squeeze()

            if scoring_strategy == "wt-marginals":
                outfile.create_dataset(
                    f"{parent}",
                    data=prob_matrix,
                    compression="gzip",
                    compression_opts=4,
                )

            elif scoring_strategy == "masked-marginals":
                # Normalize by the probability of the parent AA at each site.
                parent_idx = aa_idx_array_of_str(parent)
                parent_probs = prob_matrix[np.arange(len(parent_idx)), parent_idx]
                prob_ratio_matrix = prob_matrix / parent_probs[:, None]

                outfile.create_dataset(
                    f"{parent}",
                    data=prob_ratio_matrix,
                    compression="gzip",
                    compression_opts=4,
                )

            else:
                raise ValueError(
                    f"Invalid scoring strategy: {scoring_strategy}. Must be 'wt-marginals' or 'masked-marginals'."
                )


def ensemble_esm_models(hdf5_files, output_hdf5):
    """
    Ensemble ESM-1v probabilities from multiple HDF5 files and save to a new HDF5 file.

    hdf5_files (list): List of paths to HDF5 files containing ESM-1v logits for models 1-5.
    output_hdf5 (str): Path to the output HDF5 file.
    """

    # Check that all HDF5 files exist and have same scoring strategy
    assert len(hdf5_files) == 5, "Must provide 5 HDF5 files for each ESM1v model."
    if "wt" in hdf5_files[0]:
        assert all(
            "wt" in hdf5_file for hdf5_file in hdf5_files
        ), "All HDF5 files must use the same scoring strategy."
        scoring_strategy = "wt-marginals"
    elif "mask" in hdf5_files[0]:
        assert all(
            "mask" in hdf5_file for hdf5_file in hdf5_files
        ), "All HDF5 files must use the same scoring strategy."
        scoring_strategy = "masked-marginals"
    else:
        raise ValueError(
            "No scoring strategy recognized in filenames. Must include 'wt' or 'mask'."
        )

    # Get attributes from first HDF5 file
    with h5py.File(hdf5_files[0], "r") as infile:
        checksum = infile.attrs["checksum"]
        pcp_filename = infile.attrs["pcp_filename"]

    # Load ESM1v logits from each HDF5 file
    parent_esm_dicts = []
    for hdf5_file in hdf5_files:
        parent_esm_dicts.append(load_and_convert_to_dict(hdf5_file))

    # Check that all parent sequences are the same in each dictionary
    parent_seqs = [set(parent_esm_dict.keys()) for parent_esm_dict in parent_esm_dicts]
    assert all(
        parent_seqs[0] == parent_seq for parent_seq in parent_seqs
    ), "Parent sequences are not the same in each HDF5 file."

    # Initialize ensemble dictionary
    ensemble_dict = {}

    # Iterate through parent sequences and average ESM1v probabilities or probability ratios
    for parent in parent_seqs[0]:
        ensemble_matrix = np.mean(
            [parent_esm_dict[parent] for parent_esm_dict in parent_esm_dicts], axis=0
        )
        ensemble_dict[parent] = ensemble_matrix

    # Save ensemble dictionary to HDF5 file
    with h5py.File(output_hdf5, "w") as outfile:
        # attributes related to PCP data file
        outfile.attrs["checksum"] = checksum
        outfile.attrs["pcp_filename"] = pcp_filename
        outfile.attrs["model_name"] = f"ESM1v_ensemble_{scoring_strategy}"

        for parent in ensemble_dict:
            outfile.create_dataset(
                f"{parent}",
                data=ensemble_dict[parent],
                compression="gzip",
                compression_opts=4,
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
