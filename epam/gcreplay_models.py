from abc import ABC, abstractmethod
from importlib import resources
import numpy as np
import pandas as pd
import torch
import epam.models as models
import epam.molevol as molevol
import epam.sequences as sequences
import epam.utils

from epam.torch_common import optimize_branch_length
from typing import Tuple

from epam.esm_precompute import load_and_convert_to_dict


with resources.path("epam", "__init__.py") as p:
    DATA_DIR = str(p.parent.parent) + "/data/"

# Here's a list of the models and configurations we will use in our tests.

GCREPLAY_MODELS = [
    (
        "GCReplayDMS_igh",
        "GCReplayDMS",
        {
            "dms_data_file": DATA_DIR + "gcreplay/final_variant_scores.csv",
            "chain": "heavy",
        },
    ),
    (
        "GCReplayDMSSigmoid_igh",
        "GCReplayDMS",
        {
            "dms_data_file": DATA_DIR + "gcreplay/final_variant_scores.csv",
            "chain": "heavy",
            "sf_rescale": "sigmoid",
        },
    ),
    (
        "GCReplaySHM_igh",
        "GCReplaySHM",
        {
            "shm_data_file": DATA_DIR + "gcreplay/chigy_hc_mutation_rates_nt.csv",
            "init_branch_length": 1,
        },
    ),
    (
        "GCReplaySHMDMSSigmoid_igh",
        "GCReplaySHMDMS",
        {
            "shm_data_file": DATA_DIR + "gcreplay/chigy_hc_mutation_rates_nt.csv",
            "dms_data_file": DATA_DIR + "gcreplay/final_variant_scores.csv",
            "chain": "heavy",
            "sf_rescale": "sigmoid",
            "init_branch_length": 1,
        },
    ),
    (
        "GCReplaySHMpleDMSSigmoid_igh",
        "GCReplaySHMpleDMS",
        {
            "weights_directory": DATA_DIR + "shmple_weights/greiff_size2",
            "dms_data_file": DATA_DIR + "gcreplay/final_variant_scores.csv",
            "chain": "heavy",
            "sf_rescale": "sigmoid",
        },
    ),
]


class GCReplayDMS(models.BaseModel):
    def __init__(
        self,
        dms_data_file: str,
        chain="heavy",
        model_name=None,
        sf_rescale=None,
        scaling=1.0,
    ):
        """
        Initialize a selection model from GCReplay DMS data.

        Parameters:
        dms_data_file (str): File path to the DMS measurements data.
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        sf_rescale (str, optional): The selection factor rescaling approach.
        scaling (float): The multiplicative factor on the parent-child binding difference.
        """
        super().__init__(model_name=model_name)
        self.sf_rescale = sf_rescale
        self.scaling = scaling
        self.chain = chain[0].capitalize()
        dms_df = pd.read_csv(dms_data_file)
        self.dms_chain_df = dms_df[dms_df["chain"] == self.chain]

        # Issue: When 'mutant' amino acid is the wildtype, binding information is not provided.
        # Patch: Recover wildtype binding at a site from a non-wildtype mutant data: [bind_CGG] - [delta_bind_CGG]
        for site in self.dms_chain_df["position"].drop_duplicates():
            site_df = self.dms_chain_df[self.dms_chain_df["position"] == site]
            non_wt_row_df = site_df[site_df["wildtype"] != site_df["mutant"]].head(1)
            wt_bind = (
                non_wt_row_df["bind_CGG"].item()
                - non_wt_row_df["delta_bind_CGG"].item()
            )
            wt_iloc = site_df[site_df["wildtype"] == site_df["mutant"]].index.item()
            self.dms_chain_df.loc[wt_iloc, "bind_CGG"] = wt_bind
            self.dms_chain_df.loc[wt_iloc, "delta_bind_CGG"] = 0.0

        # Issue: Light chain DMS measurements at positions 129 (F,L), 134 (R) are missing.
        # Patch: Set the missing binding values to the wildtype value.
        if self.chain == "L":
            patch_sites = [129, 134]
            for site in patch_sites:
                site_df = self.dms_chain_df[self.dms_chain_df["position"] == site]
                wt_bind = site_df[site_df["wildtype"] == site_df["mutant"]][
                    "bind_CGG"
                ].item()
                for i, row in site_df.iterrows():
                    if np.isnan(row["bind_CGG"]):
                        self.dms_chain_df.loc[i, "bind_CGG"] = wt_bind
                        self.dms_chain_df.loc[i, "delta_bind_CGG"] = 0.0

    def _get_dms_ratios(self, parent_aa: str, site: int) -> np.ndarray:
        """
        Generate a numpy array of the ratios of association constants (K_A), according to DMS measurements,
        between each amino acid to the parent amino acid at a specified site.

        Parameters:
        parent_aa (str): The parent amino acid sequence.
        site (int): The site in the parent sequence to get K_A ratios for (0-based index).

        Returns:
        numpy.ndarray: An array containing the K_A ratios for the site.
        """
        if self.chain == "H":
            # heavy chain has amino acid positions [1, 112] in DMS data
            position = site + 1
        else:
            # light chain has amino acid positions [128, 235] in DMS data
            position = site + 128

        site_dms_df = self.dms_chain_df[self.dms_chain_df["position"] == position]
        ref_bind = site_dms_df[site_dms_df["mutant"] == parent_aa[site]][
            "bind_CGG"
        ].item()
        assert ~np.isnan(ref_bind)

        # log(10) because binding is log10[K_A]
        return np.exp(site_dms_df["bind_CGG"].to_numpy() * np.log(10)) / np.exp(
            ref_bind * np.log(10)
        )

    def aaprobs_of_parent_child_pair(self, parent, child=None) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to DMS measurements.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent nucleotide sequence for which we want the array of probabilities.
        child (str): The child nucleotide sequence (ignored).

        Returns:
        numpy.ndarray: A 2D array containing the selection factors of the amino acids by site.

        """
        parent_aa = sequences.translate_sequence(parent)
        matrix = []

        for i in range(len(parent_aa)):
            dms_ratios = self._get_dms_ratios(parent_aa, i)
            assert True not in np.isnan(dms_ratios)
            if self.sf_rescale == "sigmoid":
                sel_factors = epam.utils.ratios_to_sigmoid(
                    torch.tensor(dms_ratios), scale_const=self.scaling
                ).numpy()
            else:
                sel_factors = np.power(dms_ratios, self.scaling)

            # DMS data lists amino acid mutants in alphabetical order (convenient!)
            # Note: sel_factors results is float64, but seems like einsum wants float32
            #       (see: build_codon_mutsel in molevol.py)
            matrix.append(sel_factors.astype(np.float32))

        return np.array(matrix)


class GCReplaySHM(models.MutModel):
    def __init__(self, shm_data_file: str, *args, **kwargs):
        """
        Initialize a neutral mutation model from GCReplay passenger mouse data.

        Parameters:
        shm_data_file (str): File path to the mutation rates from passenger mouse data.
        """
        super().__init__(*args, **kwargs)
        shm_df = pd.read_csv(shm_data_file)
        cols = list("ACGT")

        # remove the last row because (?) it's not part of the sequence, also not multiple of 3 so truncated anyway
        shm_df = shm_df[cols].drop(shm_df.index[-1], axis=0, inplace=False)

        self.mut_probs = shm_df[cols].sum(axis=1).to_numpy()  # mutability probabilities
        self.sub_probs = (
            shm_df[cols].div(self.mut_probs, axis=0).to_numpy()
        )  # substitution probabilities given mutation has occurred

    def predict_rates_and_normed_subs_probs(
        self, parent: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mutability rates and (normalized) substitution probabilities predicted
        by the SHM model, given a parent nucleotide sequence.

        Parameters:
        parent (str): The parent sequence.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the rates and
            substitution probabilities as Torch tensors.
        """
        # Passenger mouse analysis gives mutation probabilities;
        # derive the Poisson rates (corresponding to branch length of 1).
        rates = torch.tensor(-np.log(1 - self.mut_probs), dtype=torch.float)
        sub_probs = torch.tensor(self.sub_probs, dtype=torch.float)
        return rates, sub_probs

    def _aaprobs_of_parent_and_branch_length(
        self, parent: str, branch_length: float
    ) -> torch.Tensor:
        """
        Calculate the amino acid probabilities for a given parent and branch length.

        Parameters:
        parent (str): The parent nucleotide sequence.
        branch_length (float): The length of the branch.

        Returns:
        np.ndarray: The aaprobs for every codon of the parent sequence.
        """
        rates, sub_probs = self.predict_rates_and_normed_subs_probs(parent)
        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        return molevol.aaprobs_of_parent_scaled_rates_and_sub_probs(
            parent_idxs, rates * branch_length, sub_probs
        )


class GCReplaySHMDMS(models.MutSelModel):
    def __init__(
        self,
        shm_data_file: str,
        dms_data_file: str,
        chain="heavy",
        sf_rescale=None,
        scaling=1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize a mutation-selection model from GCReplay passenger mouse and DMS data selection factors.

        Parameters:
        shm_data_file (str): File path to the mutation rates from passenger mouse data.
        dms_data_file (str): File path to the DMS measurements data.
        chain (str): Name of the chain, default is "heavy".
        sf_rescale (str, optional): The selection factor rescaling approach.
        scaling (float): The multiplicative factor on the parent-child binding difference.
        """
        super().__init__(*args, **kwargs)
        self.mutation_model = GCReplaySHM(shm_data_file)
        self.selection_model = GCReplayDMS(
            dms_data_file, chain=chain, sf_rescale=sf_rescale, scaling=scaling
        )

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))


class GCReplaySHMESM(models.MutSelModel):
    def __init__(
        self,
        shm_data_file: str,
        sf_rescale=None,
        *args,
        **kwargs,
    ):
        """
        Initialize a mutation-selection model from GCReplay passenger mouse and ESM1v selection factors.
        Branch optimization is performed.

        Parameters:
        shm_data_file (str): File path to the mutation rates from passenger mouse data.
        sf_rescale (str, optional): The selection factor rescaling approach.
        """
        super().__init__(*args, **kwargs)
        self.mutation_model = GCReplaySHM(shm_data_file)
        self.selection_model = models.CachedESM1v(sf_rescale=sf_rescale)

    def preload_esm_data(self, hdf5_path):
        """
        Preload ESM1v data from HDF5 file.

        Parameters:
        hdf5_path (str): Path to HDF5 file containing pre-computed selection matrices.
        """
        self.selection_model.preload_esm_data(hdf5_path)

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))


class GCReplaySHMpleDMS(models.MutSelModel):
    def __init__(
        self,
        weights_directory: str,
        dms_data_file: str,
        chain="heavy",
        sf_rescale=None,
        scaling=1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize a mutation-selection model for GC-Replay data using SHMple for the mutation part and
        DMS measurements for the selection part.

        Parameters:
        weights_directory (str): Directory path to trained SHMple model weights.
        dms_data_file (str): File path to the DMS measurements data.
        chain (str): Name of the chain, default is "heavy".
        sf_rescale (str, optional): The selection factor rescaling approach.
        scaling (float): The multiplicative factor on the parent-child binding difference.
        """
        super().__init__(*args, **kwargs)
        self.mutation_model = models.SHMple(weights_directory)
        self.selection_model = GCReplayDMS(
            dms_data_file, chain=chain, sf_rescale=sf_rescale, scaling=scaling
        )

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))
