"""
This module enables precomputation of SHMple rates and substitution
probabilities, and then loading of those saved values into tensors.
"""

import logging

import h5py
import numpy as np
import pandas as pd
import torch

from shmple import AttentionModel


def precompute_and_save(weights_path, csv_path, output_hdf5):
    # Initialize the model
    model = AttentionModel(weights_dir=weights_path, log_level=logging.WARNING)

    # Load the CSV file
    df = pd.read_csv(csv_path)
    parents = df["parent"].tolist()

    # Predict rates and substitution probabilities
    all_rates, all_subs_probs = model.predict_mutabilities_and_substitutions(
        parents, [1.0] * len(parents)
    )

    with h5py.File(output_hdf5, "w") as h5f:
        # Store each set of rates and substitution probabilities in the HDF5 file
        for i, (rates, subs_probs) in enumerate(zip(all_rates, all_subs_probs)):
            rates = rates.squeeze().astype(np.float32)
            subs_probs = subs_probs.astype(np.float32)
            h5f.create_dataset(f"rates_{i}", data=rates)
            h5f.create_dataset(f"subs_probs_{i}", data=subs_probs)

            # Store references to the HDF5 datasets in the dataframe
            df.at[i, "rates"] = f"rates_{i}"
            df.at[i, "subs_probs"] = f"subs_probs_{i}"

        # Save the dataframe with references to the datasets in the HDF5 file
        df.to_hdf(output_hdf5, key="metadata", mode="a")


def load_and_convert_to_tensors(hdf5_path):
    with pd.HDFStore(hdf5_path, "r") as store:
        # Load the dataframe which contains the references to the datasets
        df = store["metadata"]
        df.set_index("Unnamed: 0", inplace=True)
        rates_tensors = []
        subs_probs_tensors = []

        # Use the references in the dataframe to load the actual data from the HDF5 datasets
        with h5py.File(hdf5_path, "r") as h5f:
            for _, row in df.iterrows():
                rates = torch.tensor(h5f[row["rates"]][:], dtype=torch.float)
                subs_probs = torch.tensor(h5f[row["subs_probs"]][:], dtype=torch.float)
                rates_tensors.append(rates)
                subs_probs_tensors.append(subs_probs)

        df["rates"] = rates_tensors
        df["subs_probs"] = subs_probs_tensors

    return df
