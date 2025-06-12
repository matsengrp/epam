from abc import ABC, abstractmethod
from importlib import resources
import logging
import time
from typing import Tuple

import torch
import torch.optim as optim
from torch import Tensor
from epam.esm_precompute import load_and_convert_to_dict
import h5py
import numpy as np
import pandas as pd
from scipy.special import softmax

import ablang
import ablang2

import netam.framework
import netam.molevol as molevol
import netam.sequences as sequences
from netam.sequences import (
    AA_STR_SORTED,
    assert_pcp_lengths,
    assert_full_sequences,
    translate_sequence,
    aa_idx_array_of_str,
)
import epam.utils as utils
from netam.common import pick_device, SMALL_PROB
from netam.molevol import optimize_branch_length, mutsel_log_pcp_probability_of

import Bio.Align.substitution_matrices

# explictly set number of threads to 1 to avoid slowdowns during branch length optimization
torch.set_num_threads(1)

DATA_DIR = str(resources.files("epam").parent) + "/data/"
THRIFTY_DIR = str(resources.files("epam").parent) + "/thrifty-models/models/"

# Here's a list of the models and configurations we will use in our tests and
# pipeline.

FULLY_SPECIFIED_MODELS = [
    ("AbLang1", "AbLang1", {"chain": "heavy"}),
    # ("AbLang2_wt", "AbLang2", {"version": "ablang2-paired", "chain": "heavy", "masking": False}),
    (
        "AbLang2_mask",
        "AbLang2",
        {"version": "ablang2-paired", "chain": "heavy", "masking": True},
    ),
    # ("ESM1v_wt", "CachedESM1v", {}),
    ("ESM1v_mask", "CachedESM1v", {"scoring_strategy": "masked"}),
    (
        "S5F",
        "S5F",
        {
            "muts_file": DATA_DIR + "S5F/hh_s5f_muts.csv",
            "subs_file": DATA_DIR + "S5F/hh_s5f_subs.csv",
            "init_branch_length": 1,
        },
    ),
    (
        "S5FESM_mask",
        "S5FESM",
        {
            "muts_file": DATA_DIR + "S5F/hh_s5f_muts.csv",
            "subs_file": DATA_DIR + "S5F/hh_s5f_subs.csv",
            "sf_rescale": "sigmoid",
            "init_branch_length": 1,
        },
    ),
    (
        "S5FBLOSUM",
        "S5FBLOSUM",
        {
            "muts_file": DATA_DIR + "S5F/hh_s5f_muts.csv",
            "subs_file": DATA_DIR + "S5F/hh_s5f_subs.csv",
            "sf_rescale": "sigmoid",
            "init_branch_length": 1,
        },
    ),
    (
        "ThriftyHumV0.2-59",
        "NetamSHM",
        {
            "model_path_prefix": THRIFTY_DIR + "ThriftyHumV0.2-59",
        },
    ),
    (
        "ThriftyProdHumV0.2-59",
        "NetamSHM",
        {
            "model_path_prefix": THRIFTY_DIR + "cnn_ind_lrg-v1wyatt-simple-0",
        },
    ),
    (
        "ThriftyESM_mask",
        "NetamSHMESM",
        {
            "model_path_prefix": THRIFTY_DIR + "ThriftyHumV0.2-59",
            "sf_rescale": "sigmoid",
        },
    ),
    (
        "ThriftyBLOSUM",
        "NetamSHMBLOSUM",
        {
            "model_path_prefix": THRIFTY_DIR + "ThriftyHumV0.2-59",
            "sf_rescale": "sigmoid",
        },
    ),
]


class BaseModel(ABC):
    def __init__(self, model_name=None):
        """
        Initializes a new instance of the BaseModel class.

        Parameters:
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        """
        if model_name is None:
            model_name = self.__class__.__name__
        self.model_name = model_name

    @abstractmethod
    def aaprobs_of_parent_child_pair(
        self, parent: str, child: str
    ) -> Tuple[np.ndarray, float, bool]:
        pass

    def probability_vector_of_child_seq(self, prob_arr: np.ndarray, child_seq: str):
        """
        Calculate the sitewise probability of a child sequence given a probability array.

        Parameters:
        prob_arr (numpy.ndarray): A 2D array containing the normalized probabilities of the amino acids by site.
        child_seq (str): The child sequence for which we want the probability vector.

        Returns:
        numpy.ndarray: A 1D array containing the sitewise probability of the child sequence.

        """
        assert (
            len(child_seq) == prob_arr.shape[0]
        ), "The child sequence length does not match the probability array length."

        return np.array(
            [prob_arr[i, AA_STR_SORTED.index(aa)] for i, aa in enumerate(child_seq)]
        )

    def write_aaprobs(self, pcp_path: str, output_path: str, log_file: str = None):
        """
        Write an aaprob matrix for each parent-child pair (PCP) of nucleotide sequences with a substitution model.

        An HDF5 output file is created that includes the file path to the PCP data and a checksum for verification.

        Parameters:
        pcp_path (str): file name of parent-child pair data.
        output_path (str): output file name.
        log_file (str, optional): file name for logging branch optimization results. Default is None.

        """
        checksum = utils.generate_file_checksum(pcp_path)
        pcp_df = pd.read_csv(pcp_path, index_col=0)

        if log_file is not None:
            csv_file = open(log_file, "w")
            csv_file.write(
                "pcp_index,parent,child,opt_branch_length,fail_to_converge\n"
            )

        with h5py.File(output_path, "w") as outfile:
            # attributes related to PCP data file
            outfile.attrs["checksum"] = checksum
            outfile.attrs["pcp_filename"] = pcp_path
            outfile.attrs["model_name"] = self.model_name

            for i, row in pcp_df.iterrows():
                parent = row["parent"]
                child = row["child"]
                assert_pcp_lengths(parent, child)
                assert_full_sequences(parent, child)
                if utils.pcp_criteria_check(parent, child):

                    matrix, opt_branch_length, converge_status = (
                        self.aaprobs_of_parent_child_pair(parent, child)
                    )

                    # create a group for each matrix + add to hdf5 file
                    grp = outfile.create_group(f"matrix{i}")
                    grp.attrs["pcp_index"] = i
                    grp.create_dataset(
                        "data", data=matrix, compression="gzip", compression_opts=4
                    )

                    # log branch length optimization results to csv
                    if log_file is not None:
                        csv_file.write(
                            f"{i},{parent},{child},{opt_branch_length},{converge_status}\n"
                        )


class MutModel(BaseModel):
    """
    Abstract base class for models of neutral nucleotide mutations.
    """

    def __init__(
        self,
        model_name=None,
        optimize=True,
        init_branch_length=None,
        max_optimization_steps=1000,
        optimization_tol=1e-4,
        learning_rate=0.1,
    ):
        """
        Initialize a new instance of the MutModel for neutral nucleotide mutations.

        Parameters:
        model_name : str, optional
            Model name. Default is None, setting the model name to the class name.
        optimize : bool, optional
            Whether to perform branch length optimization. Default is True.
        init_branch_length : float, optional
            Initial branch length before optimization. If None, the mutation frequency of the PCP is the initial branch length.
        max_optimization_steps : int, optional
            Maximum number of gradient descent steps. Default is 1000. Ignored if optimize is False.
        optimization_tol : float, optional
            Tolerance for optimization of log(branch length). Default is 1e-4.
        learning_rate : float, optional
            Learning rate for torch's SGD. Default is 0.1.
        """
        super().__init__(model_name=model_name)
        self.optimize = optimize
        self.max_optimization_steps = max_optimization_steps
        self.init_branch_length = init_branch_length
        self.optimization_tol = optimization_tol
        self.learning_rate = learning_rate

    @abstractmethod
    def predict_rates_and_normed_subs_probs(
        self, parent: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mutability rates and (normalized) substitution probabilities predicted
        by the SHM model for a given parent nucleotide sequence.

        Parameters:
        parent (str): The parent nucleotide sequence.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the rates and
            substitution probabilities as Torch tensors.
        """
        pass

    @abstractmethod
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
        pass

    def _build_log_pcp_probability(
        self, parent: str, child: str, rates: Tensor, sub_probs: Tensor
    ):
        """Constructs the log_pcp_probability function specific to given rates and sub_probs.

        This function takes log_branch_length as input and returns the log
        probability of the child sequence. It uses log of branch length to
        ensure non-negativity."""

        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        child_idxs = sequences.nt_idx_tensor_of_str(child)

        def log_pcp_probability(log_branch_length):
            branch_length = torch.exp(log_branch_length)
            mut_probs = 1.0 - torch.exp(-branch_length * rates)
            no_mutation_sites = parent_idxs == child_idxs

            same_probs = 1.0 - mut_probs[no_mutation_sites]
            diff_probs = (
                mut_probs[~no_mutation_sites]
                * sub_probs[~no_mutation_sites, child_idxs[~no_mutation_sites]]
            )
            child_log_prob = torch.log(torch.cat([same_probs, diff_probs])).sum()

            return child_log_prob

        return log_pcp_probability

    def _find_optimal_branch_length(self, parent, child, starting_branch_length):
        """
        Find the optimal branch length for a parent-child pair in terms of
        nucleotide likelihood.

        Parameters:
        parent (str): The parent sequence.
        child (str): The child sequence.
        starting_branch_length (float): The branch length used to initialize the optimization.

        Returns:
        float: The optimal branch length.
        """

        rates, sub_probs = self.predict_rates_and_normed_subs_probs(parent)

        log_pcp_probability = self._build_log_pcp_probability(
            parent, child, rates, sub_probs
        )

        if self.optimize == True:
            return optimize_branch_length(
                log_prob_fn=log_pcp_probability,
                starting_branch_length=starting_branch_length,
                learning_rate=self.learning_rate,
                max_optimization_steps=self.max_optimization_steps,
                optimization_tol=self.optimization_tol,
            )
        else:
            return starting_branch_length, False

    def aaprobs_of_parent_child_pair(self, parent, child) -> np.ndarray:
        if self.init_branch_length is None:
            base_branch_length = sequences.nt_mutation_frequency(parent, child)
        else:
            base_branch_length = self.init_branch_length
        branch_length, converge_status = self._find_optimal_branch_length(
            parent, child, base_branch_length
        )
        if self.init_branch_length is None and branch_length > 0.5:
            print(f"Warning: branch length of {branch_length} is surprisingly large.")
        aaprob = self._aaprobs_of_parent_and_branch_length(
            parent, branch_length
        ).numpy()
        return aaprob, branch_length, converge_status


class MutSelModel(MutModel):
    """A mutation selection model.

    Note that stop codons are assumed to have zero selection probability.

    Parameters:
    mutation_model (MutModel): A model for SHM.
    selection_model (BaseModel): A model for computing selection factors.
    """

    def __init__(self, mutation_model, selection_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_model = mutation_model
        self.selection_model = selection_model
        # This is a diagnostic generating data for netam issue #7.
        # self.csv_file = open(
        #     f"prob_sums_too_big_{int(time.time())}.csv", "w"
        # )
        # self.csv_file.write("parent,child,branch_length,sums_too_big\n")

    @abstractmethod
    def build_selection_matrix_from_parent(self, parent: str) -> Tensor:
        """Build the selection matrix (i.e. F matrix) from a parent nucleotide
        sequence.

        The shape of this numpy array should be (len(parent) // 3, 20).
        """
        pass

    def predict_rates_and_normed_subs_probs(
        self, parent: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mutation_model.predict_rates_and_normed_subs_probs(parent)

    def _build_log_pcp_probability(
        self, parent: str, child: str, rates: Tensor, sub_probs: Tensor
    ):
        sel_matrix = self.build_selection_matrix_from_parent(parent)
        return mutsel_log_pcp_probability_of(
            sel_matrix, parent, child, rates, sub_probs
        )

    def _aaprobs_of_parent_and_branch_length(self, parent, branch_length) -> Tensor:
        rates, sub_probs = self.predict_rates_and_normed_subs_probs(parent)

        sel_matrix = self.build_selection_matrix_from_parent(parent)
        mut_probs = 1.0 - torch.exp(-branch_length * rates)

        parent_idxs = sequences.nt_idx_tensor_of_str(parent)

        codon_mutsel, sums_too_big = molevol.build_codon_mutsel(
            parent_idxs.reshape(-1, 3),
            mut_probs.reshape(-1, 3),
            sub_probs.reshape(-1, 3, 4),
            sel_matrix,
        )

        if sums_too_big is not None:
            print(
                "Warning: some of the codon probability sums were too big for the codon mutsel calculation."
            )

        return molevol.aaprobs_of_codon_probs(codon_mutsel)


class RandomMutSel(MutSelModel):
    """A mutation selection model with a random selection matrix."""

    def __init__(self, model_path_prefix: str, *args, **kwargs):
        super().__init__(
            mutation_model=NetamSHM(model_path_prefix=model_path_prefix),
            selection_model=None,
            *args,
            **kwargs,
        )

    def build_selection_matrix_from_parent(self, parent: str) -> Tensor:
        matrix = torch.rand(len(parent) // 3, 20)
        matrix /= matrix.sum(dim=1, keepdim=True)
        return matrix


class MLMBase(BaseModel):
    def __init__(
        self,
        model_name=None,
        optimize=True,
        max_optimization_steps=1000,
        optimization_tol=1e-4,
        learning_rate=0.1,
    ):
        """
        This is an abstract class with shared functionality for Masked Language Models (MLMs; i.e., AbLang1, AbLang2, ESM). All models rescales amino acid probabilities from MLMs with an optimized branch length for each parent-child pair for comparison with CTMC models.

        Parameters:
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        optimize (bool, optional): Whether to perform branch length optimization. Default is True.
        max_optimization_steps (int, optional): Maximum number of gradient descent steps. Default is 1000. Ignored if optimize is False.
        optimization_tol (float, optional): Tolerance for optimization of log(branch length). Default is 1e-4.
        learning_rate (float, optional): Learning rate for torch's SGD. Default is 0.1.

        """
        super().__init__(model_name=model_name)
        self.optimize = optimize
        self.max_optimization_steps = max_optimization_steps
        self.optimization_tol = optimization_tol
        self.learning_rate = learning_rate

    @abstractmethod
    def probability_array_of_seq(self, seq: str) -> np.ndarray:
        pass

    def _build_log_pcp_probability(
        self, parent: str, child: str, child_aa_probs: Tensor
    ):
        """
        Constructs the log_pcp_probability function specific to given aa_probs for the child sequence from a MLM.

        This function takes log_branch_length as input and returns the log
        probability of the child sequence. It uses log of branch length to
        ensure non-negativity. The probability of the child sequence is scaled
        here by e^{-tau} (scaling_factor), which is bounded between 0 and 1. We assume
        that MLM probabilities correspond to a branch length much larger than those
        observed in our PCPs, and interpolate between no evolutionary time and the larger
        time scales in MLM training data.

        """

        parent_idx = sequences.aa_idx_tensor_of_str(parent)
        child_idx = sequences.aa_idx_tensor_of_str(child)

        def log_pcp_probability(log_branch_length):
            branch_length = torch.exp(log_branch_length)
            scaling_factor = torch.exp(-branch_length)
            sub_probs = scaling_factor * child_aa_probs

            no_sub_sites = parent_idx == child_idx

            # Rescaling each site based on whether a substitution event occurred or not.
            same_probs = (
                scaling_factor + child_aa_probs[no_sub_sites] - sub_probs[no_sub_sites]
            )

            diff_probs = child_aa_probs[~no_sub_sites] - sub_probs[~no_sub_sites]

            # Clip probabilities to avoid numerical issues.
            same_probs = torch.clamp(same_probs, min=SMALL_PROB, max=(1 - SMALL_PROB))
            diff_probs = torch.clamp(diff_probs, min=SMALL_PROB, max=(1 - SMALL_PROB))

            child_log_prob = torch.log(torch.cat([same_probs, diff_probs])).sum()

            return child_log_prob

        return log_pcp_probability

    def _find_optimal_branch_length(
        self, parent, child, starting_branch_length, prob_arr
    ):
        """
        Find the optimal branch length for a parent-child pair in terms of
        amino acid likelihood.

        Parameters:
        parent (str): The parent AA sequence.
        child (str): The child AA sequence.
        starting_branch_length (float): The branch length used to initialize the optimization.
        prob_arr (numpy.ndarray): A 2D array containing the unscaled probabilities of the amino acids by site computed by a MLM.

        """
        child_prob = self.probability_vector_of_child_seq(prob_arr, child)
        prob_tensor = torch.tensor(child_prob, dtype=torch.float)
        log_pcp_probability = self._build_log_pcp_probability(
            parent, child, prob_tensor
        )

        if self.optimize == True:
            return optimize_branch_length(
                log_prob_fn=log_pcp_probability,
                starting_branch_length=starting_branch_length,
                learning_rate=self.learning_rate,
                max_optimization_steps=self.max_optimization_steps,
                optimization_tol=self.optimization_tol,
            )
        else:
            return starting_branch_length, False

    def scale_probability_array(
        self, prob_arr: np.ndarray, parent: str, branch_length: float
    ) -> np.ndarray:
        """
        Rescale the amino acid probability matrix from a MLM with the optimized "branch length".

        For fair comparison with CTMC models, we apply a linear rescaling of the amino acid probabilities. By itself,
        MLMs do not have any notion of branch length and will make the same predicition regardless of evolutionary
        time between the parent and child sequence. We rescale each prob_arr with scaling_factor, where the probability of
        no substitution is scaling_factor + (1 - scaling_factor) * prob_arr and the probability of substitution is
        scaling_factor * prob_arr. For each PCP, the value of scaling_factor is optimized to maximize the likelihood of
        the child sequence. This is more or less equivalent to scaling the branch length in SHMple mut-sel models.


        Parameters:
        prob_arr (numpy.ndarray): A 2D array containing the normalized probabilities of the amino acids by site.
        parent (str): The parent sequence.
        branch_length (float): The branch length.

        Returns:
        numpy.ndarray: A 2D array containing the scaled probabilities of the amino acids by site.

        """
        scaling_factor = np.exp(-branch_length)
        scaled_prob_arr = np.zeros(prob_arr.shape)

        parent_idx = sequences.aa_idx_array_of_str(parent)
        mask_parent = np.eye(20, dtype=bool)[parent_idx]

        scaled_prob_arr[mask_parent] = scaling_factor + (
            (1 - scaling_factor) * prob_arr[mask_parent]
        )
        scaled_prob_arr[~mask_parent] = (1 - scaling_factor) * prob_arr[~mask_parent]

        # Clip probabilities to avoid numerical issues.
        scaled_prob_arr = np.clip(
            scaled_prob_arr, a_min=SMALL_PROB, a_max=(1 - SMALL_PROB)
        )

        # Assert that each row/probability distribution sums to 1.
        if not np.allclose(np.sum(scaled_prob_arr, axis=1), 1.0, atol=1e-5):
            print(
                f"Warning: rowsums of scaled_prob_arr do not sum to 1 with optimized branch length {branch_length}."
            )

        return scaled_prob_arr

    def aaprobs_of_parent_child_pair(
        self, parent: str, child: str
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the MLM model with a branch length optimization.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent nucleotide sequence for which we want the array of probabilities.
        child (str): The child nucleotide sequence (ignored for MLM model).

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        base_branch_length = sequences.nt_mutation_frequency(parent, child)
        parent_aa = translate_sequence(parent)
        child_aa = translate_sequence(child)

        unscaled_aaprob = self.probability_array_of_seq(parent_aa)

        branch_length, converge_status = self._find_optimal_branch_length(
            parent_aa, child_aa, base_branch_length, unscaled_aaprob
        )

        return (
            self.scale_probability_array(unscaled_aaprob, parent_aa, branch_length),
            branch_length,
            converge_status,
        )


class AbLang1(MLMBase):
    def __init__(
        self,
        chain="heavy",
        model_name=None,
    ):
        """
        Initialize AbLang1 model with specified chain and create amino acid string. This model rescales amino acid probabilities from AbLang with an optimized branch length for each parent-child pair for comparison with CTMC models.

        Parameters:
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.

        """
        super().__init__(model_name=model_name)
        self.device = pick_device()
        self.model = ablang.pretrained(chain, device=self.device)
        self.model.freeze()
        vocab_dict = self.model.tokenizer.vocab_to_aa
        self.aa_str = "".join([vocab_dict[i + 1] for i in range(20)])
        self.aa_str_sorted_indices = np.argsort(list(self.aa_str))
        assert AA_STR_SORTED == "".join(
            np.array(list(self.aa_str))[self.aa_str_sorted_indices]
        )

    def probability_array_of_seq(self, seq: str) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the AbLang model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        seq (str): The sequence for which we want the array of probabilities.

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        likelihoods = self.model([seq], mode="likelihood")

        # Apply softmax to the second dimension, and skip the first and last
        # elements (which are the probability of the start and end token).
        arr = np.apply_along_axis(softmax, 1, likelihoods[0, 1:-1])

        # Sort the second dimension according to the sorted amino acid string.
        arr_sorted = arr[:, self.aa_str_sorted_indices]
        assert len(seq) == arr_sorted.shape[0]

        return arr_sorted


class AbLang2(MLMBase):
    def __init__(
        self,
        version="ablang2-paired",
        chain="heavy",
        masking=False,
        model_name=None,
    ):
        """
        Initialize AbLang2 model with or without masking. This model rescales amino acid probabilities from AbLang with an optimized branch length for each parent-child pair for comparison with CTMC models.

        Parameters:
        version (str, optional): Version of the AbLang model. Currently limited to 'ablang2-paired' but could theoretically support 'ablang1-heavy' and 'ablang1-light'.
        chain (str, optional): Name of the chain, default is "heavy".
        masking (bool, optional): Whether to use masking in the model. Default is False.
        model_name (str, optional): The name of the model. If not specified, the class name is used.

        """
        super().__init__(model_name=model_name)
        self.version = version
        self.chain = chain
        self.device = pick_device()
        self.model = ablang2.pretrained(model_to_use=self.version, device=self.device)
        self.model.freeze()
        self.masking = masking
        vocab_dict = self.model.tokenizer.aa_to_token
        self.aa_sorted_indices = [vocab_dict[aa] for aa in AA_STR_SORTED]
        assert AA_STR_SORTED == "".join(
            [
                key
                for value in self.aa_sorted_indices
                for key, v in vocab_dict.items()
                if v == value
            ]
        )

    def probability_array_of_seq(self, seq: str) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the AbLang model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        seq (str): The sequence for which we want the array of probabilities.

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        # Get log likelihoods for the sequence and softmax to get probabilities
        if self.chain == "heavy":
            likelihoods = self.model(
                [seq, ""], mode="likelihood", stepwise_masking=self.masking
            )
            seq_likelihoods = likelihoods[0]

            # Apply softmax to the second dimension. Skipping the first and last
            # elements (which are the probability of the start, end, and heavy|light divider token),
            # as well as all tokens not corresponding to the 20 AAs.
            arr_sorted = np.apply_along_axis(
                softmax, 1, seq_likelihoods[1:-2, self.aa_sorted_indices]
            )
        elif self.chain == "light":
            likelihoods = self.model(
                ["", seq], mode="likelihood", stepwise_masking=self.masking
            )
            seq_likelihoods = likelihoods[0]

            # Apply softmax to the second dimension. Skipping the first and last
            # elements (which are the probability of the heavy|light divider, start, and stop token),
            # as well as all tokens not corresponding to the 20 AAs.
            arr_sorted = np.apply_along_axis(
                softmax, 1, seq_likelihoods[2:-1, self.aa_sorted_indices]
            )
        else:
            raise ValueError("chain must be set to 'heavy' or 'light'")

        # Return probabilies (masked-marginals probabilities are not parent-dependent)
        assert len(seq) == arr_sorted.shape[0]
        return arr_sorted


class CachedESM1v(MLMBase):
    def __init__(self, model_name=None, scoring_strategy="masked"):
        """
        Initialize ESM1v with cached selection matrices generated in esm_precompute.py for standalone model with scaling.

        Parameters:
        model_name (str, optional): The name of the model.
        """
        super().__init__(model_name=model_name)
        self.scoring_strategy = scoring_strategy

    def preload_esm_data(self, hdf5_path):
        """
        Preload ESM1v data from HDF5 file.

        Parameters:
        hdf5_path (str): Path to HDF5 file containing pre-computed selection matrices.
        """
        self.selection_matrices = load_and_convert_to_dict(hdf5_path)

    def probability_array_of_seq(self, parent, child=None) -> np.ndarray:
        """
        Find probability matrix corresponding to parent sequence via lookup table. Use in MLMBase class scaling.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence (ignored for ESM1v model)

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.
        """
        assert (
            parent in self.selection_matrices.keys()
        ), f"{parent} not present in CachedESM."

        # Selection matrix precomputed for parent sequence
        # probabilities for wt-marginals, probability ratios for masked-marginals
        sel_matrix = self.selection_matrices[parent]

        # Normalize the probability ratios to sum to 1.
        if self.scoring_strategy == "masked":
            sel_matrix = sel_matrix / np.sum(sel_matrix, axis=1, keepdims=True)

        # Assert that each row/probability distribution sums to 1.
        if not np.allclose(np.sum(sel_matrix, axis=1), 1.0, atol=1e-5):
            print(f"Warning: rowsums of ESM sel_matrix do not sum to 1.")

        return sel_matrix


class ESM1vSelModel(BaseModel):
    def __init__(self, model_name=None, sf_rescale=None):
        """
        Initialize ESM1v with cached selection matrices generated in esm_precompute.py. Use as selection factors in MutSel classes.

        If sf_rescale is set to "sigmoid", the selection factors are rescaled using a sigmoid transformation.

        Parameters:
        model_name (str, optional): The name of the model.
        sf_rescale (str, optional): Selection factor rescaling approach used for ratios produced under mask-marginals scoring strategy. Ignoring for wt-marginals selection factors.
        """
        super().__init__(model_name=model_name)
        self.sf_rescale = sf_rescale

    def preload_esm_data(self, hdf5_path):
        """
        Preload ESM1v data from HDF5 file.

        Parameters:
        hdf5_path (str): Path to HDF5 file containing pre-computed selection matrices.
        """
        self.selection_matrices = load_and_convert_to_dict(hdf5_path)

    def aaprobs_of_parent_child_pair(self, parent, child=None) -> np.ndarray:
        """
        Find probability matrix corresponding to parent sequence via lookup table.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence (ignored for ESM1v model)

        Returns:
        numpy.ndarray: A 2D array containing the selection factors of the amino acids by site.
        """
        assert (
            parent in self.selection_matrices.keys()
        ), f"{parent} not present in precomputed ESM dictionary."
        if self.sf_rescale == "sigmoid" or self.sf_rescale == "sigmoid-normalize":
            # Sigmoid transformation for selection factors with some values greater than 1.
            ratio_sel_matrix = torch.tensor(self.selection_matrices[parent])
            sel_tensor = utils.ratios_to_sigmoid(ratio_sel_matrix)

            if self.sf_rescale == "sigmoid-normalize":
                # Normalize the selection matrix.
                row_sums = sel_tensor.sum(dim=1, keepdim=True)
                sel_tensor /= row_sums

            sel_matrix = sel_tensor.numpy()
        else:
            sel_matrix = self.selection_matrices[parent]
        return sel_matrix


class NetamSHM(MutModel):
    def __init__(self, model_path_prefix: str, *args, **kwargs):
        """
        Initialize a Netam SHM model with specified path prefix to trained model weights.

        Parameters:
        model_path_prefix (str): directory path prefix (i.e. without file name extension) to trained Netam SHM model weights.
        """
        super().__init__(*args, **kwargs)
        assert netam.framework.crepe_exists(model_path_prefix)
        self.model = netam.framework.load_crepe(model_path_prefix, device=pick_device())

    def predict_rates_and_normed_subs_probs(
        self, parent: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mutability rates and (normalized) substitution probabilities predicted
        by the Netam SHM model, given a parent nucleotide sequence.

        Parameters:
        parent (str): The parent sequence.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the rates and
            substitution probabilities as Torch tensors.
        """
        [rates], [csp_logits] = self.model([parent])
        sub_probs = torch.softmax(csp_logits, dim=1)
        return rates.detach()[: len(parent)], sub_probs.detach()[: len(parent)]

    def _aaprobs_of_parent_and_branch_length(
        self, parent: str, branch_length: float
    ) -> torch.Tensor:
        """
        Calculate the amino acid probabilities for a given parent and branch length.

        This is the key function that needs to be overridden for implementing a new model.

        Parameters:
        parent (str): The parent nucleotide sequence.
        branch_length (float): The length of the branch.

        Returns:
        np.ndarray: The aaprobs for every codon of the parent sequence.
        """
        rates, subs = self.predict_rates_and_normed_subs_probs(parent)
        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        return molevol.aaprobs_of_parent_scaled_rates_and_csps(
            parent_idxs, rates * branch_length, subs
        )


class NetamSHMESM(MutSelModel):
    def __init__(self, model_path_prefix: str, sf_rescale=None, *args, **kwargs):
        """
        Initialize a mutation-selection model using Netam SHM for the mutation part and ESM-1v for the selection part.

        Parameters:
        model_path_prefix (str): directory path prefix (i.e. without file name extension) to trained Netam SHM model weights.
        sf_rescale (str, optional): Selection factor rescaling approach used for ratios produced under mask-marginals scoring strategy (see CachedESM1v).
        """
        super().__init__(
            mutation_model=NetamSHM(model_path_prefix=model_path_prefix),
            selection_model=ESM1vSelModel(sf_rescale=sf_rescale),
            *args,
            **kwargs,
        )

    def preload_esm_data(self, hdf5_path):
        """
        Preload ESM1v data from HDF5 file.

        Parameters:
        hdf5_path (str): Path to HDF5 file containing pre-computed selection matrices.
        """
        self.selection_model.preload_esm_data(hdf5_path)

    def build_selection_matrix_from_parent(self, parent):
        parent_aa = translate_sequence(parent)
        return torch.tensor(
            self.selection_model.aaprobs_of_parent_child_pair(parent_aa)
        )


class S5F(MutModel):
    def __init__(self, muts_file: str, subs_file: str, *args, **kwargs):
        """
        Initialize S5F model with specified file paths to trained model probabilities.

        Parameters:
        muts_file (str): file of mutabilities per 5-mer motif.
        subs_file (str): file of substitution probabilities per 5-mer motif.
        """
        super().__init__(*args, **kwargs)
        self.motif_mutability = {}
        df = pd.read_csv(muts_file)
        for i, row in df.iterrows():
            self.motif_mutability[row.motifs] = row.muts

        self.motif_substitution = {}
        df = pd.read_csv(subs_file)
        for i, row in df.iterrows():
            self.motif_substitution[row.motif] = np.array(
                [row.Asub, row.Csub, row.Gsub, row.Tsub]
            )

    def predict_rates_and_normed_subs_probs(
        self, parent: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mutability rates and (normalized) substitution probabilities predicted
        by S5F model, given a parent nucleotide sequence.

        Parameters:
        parent (str): The parent sequence.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the rates and
            substitution probabilities as Torch tensors.
        """
        [motifs] = self._motif_list([parent])
        mut_probs = np.array([self._motif_mutability(motif) for motif in motifs])

        # S5F gives mutability probabilities; set the Poisson rates (corresponding to branch length of 1).
        rates = torch.tensor(mut_probs, dtype=torch.float)

        sub_probs = torch.tensor(
            np.stack([self._motif_substitution(motif) for motif in motifs]),
            dtype=torch.float,
        )

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
        return molevol.aaprobs_of_parent_scaled_rates_and_csps(
            parent_idxs, rates * branch_length, sub_probs
        )

    def _motif_list(self, sequences: list[str]):
        """Parse a list of sequence strings to get at the underlying motifs."""
        lists = []
        for seq in sequences:
            motifs = []
            padded = "NN" + seq + "NN"
            for i in range(len(seq)):
                motifs.append(padded[i : i + 5])
            lists.append(motifs)
        return lists

    def _motif_substitution(self, motif: str):
        """Computes the subtitution vector for a motif. Performs a lookup with
        disambiguation, if necessary. Distributions are averaged near the boundaries."""
        if "N" in motif:
            motifs = self._disambiguate(motif)
            return sum([self.motif_substitution[mot] for mot in motifs]) / len(motifs)
        else:
            return self.motif_substitution[motif]

    def _motif_mutability(self, motif: str):
        """
        Computes the mutability of a motif. Mostly a lookup except
        for near the sequence boundaries, where we will resolve the N bases
        by averaging over the mutability of all matching motifs.
        """
        if "N" in motif:
            motifs = self._disambiguate(motif)
            return sum([self.motif_mutability[mot] for mot in motifs]) / len(motifs)
        else:
            return self.motif_mutability[motif]

    def _disambiguate(self, motif: str):
        """Expands ambiguous motif to a list of concrete motifs"""
        idx = motif.find("N")
        if idx < 0:
            return [motif]
        else:
            motifs = []
            for l in self._disambiguate(motif[:idx]):
                for r in self._disambiguate(motif[idx + 1 :]):
                    for ch in "ACGT":
                        motifs.append(l + ch + r)
            return motifs


class S5FESM(MutSelModel):
    def __init__(
        self,
        muts_file: str,
        subs_file: str,
        sf_rescale=None,
        *args,
        **kwargs,
    ):
        """
        Initialize a mutation-selection model from S5F and DMS data selection factors.

        Parameters:
        muts_file (str): file of mutabilities per 5-mer motif.
        subs_file (str): file of substitution probabilities per 5-mer motif.
        sf_rescale (str, optional): The selection factor rescaling approach.
        """
        super().__init__(
            mutation_model=S5F(muts_file=muts_file, subs_file=subs_file),
            selection_model=ESM1vSelModel(sf_rescale=sf_rescale),
            *args,
            **kwargs,
        )

    def preload_esm_data(self, hdf5_path):
        """
        Preload ESM1v data from HDF5 file.

        Parameters:
        hdf5_path (str): Path to HDF5 file containing pre-computed selection matrices.
        """
        self.selection_model.preload_esm_data(hdf5_path)

    def build_selection_matrix_from_parent(self, parent):
        parent_aa = sequences.translate_sequence(parent)
        return torch.tensor(
            self.selection_model.aaprobs_of_parent_child_pair(parent_aa)
        )


class BLOSUM(BaseModel):
    def __init__(
        self,
        matrix_name="BLOSUM62",
        model_name=None,
        sf_rescale=None,
        scaling=1.0,
    ):
        """
        Initialize a selection model from a BLOSUM matrix. Use as selection factors in MutSel classes.

        If sf_rescale is set to "sigmoid", the selection factors are rescaled using a sigmoid transformation.

        Parameters:
        matrix_name (str): Name of BLOSUM matrix (e.g. "BLOSUM45", "BLOSUM62", "BLOSUM80", "BLOSUM90")
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        sf_rescale (str, optional): The selection factor rescaling approach.
        scaling (float): Exponent on the BLOSUM observed-expected ratio.
        """
        super().__init__(model_name=model_name)
        self.substitution_matrix = Bio.Align.substitution_matrices.load(matrix_name)
        self.sf_rescale = sf_rescale
        self.scaling = scaling

    def aaprobs_of_parent_child_pair(self, parent, child=None) -> np.ndarray:
        """
        Generate a matrix of selection factors from BLOSUM matrix entries.

        Parameters:
        parent (str): The parent nucleotide sequence for which we want the array of probabilities.
        child (str): The child nucleotide sequence (ignored).

        Returns:
        numpy.ndarray: A 2D array containing the selection factors of the amino acids by site.

        """
        parent_aa = sequences.translate_sequence(parent)
        matrix = []

        # Note: amino acid order of the BLOSUM matrix is not in alphabetical order
        aa_sorted_indices = [
            self.substitution_matrix.alphabet.index(aa) for aa in AA_STR_SORTED
        ]

        for aa in parent_aa:
            blosum_entries = np.array(
                [self.substitution_matrix[aa, :][i] for i in aa_sorted_indices]
            )
            ratios = np.power(2, blosum_entries / 2)
            assert True not in np.isnan(ratios)
            if self.sf_rescale == "sigmoid":
                sel_factors = utils.ratios_to_sigmoid(
                    torch.tensor(ratios), scale_const=self.scaling
                ).numpy()
            else:
                sel_factors = np.power(ratios, self.scaling)

            # Note: sel_factors results is float64, but seems like einsum wants float32
            #       (see: build_codon_mutsel in molevol.py)
            matrix.append(sel_factors.astype(np.float32))

        return np.array(matrix)


class NetamSHMBLOSUM(MutSelModel):
    def __init__(
        self,
        model_path_prefix: str,
        matrix_name="BLOSUM62",
        sf_rescale=None,
        *args,
        **kwargs,
    ):
        """
        Initialize a mutation-selection model using Netam SHM for the mutation part and a BLOSUM matrix for the selection part.

        Parameters:
        model_path_prefix (str): directory path prefix (i.e. without file name extension) to trained Netam SHM model weights.
        matrix_name (str): Name of BLOSUM matrix (e.g. "BLOSUM45", "BLOSUM62", "BLOSUM80", "BLOSUM90")
        sf_rescale (str, optional): Selection factor rescaling approach used for ratios produced under mask-marginals scoring strategy (see CachedESM1v).
        """
        super().__init__(
            mutation_model=NetamSHM(model_path_prefix=model_path_prefix),
            selection_model=BLOSUM(matrix_name=matrix_name, sf_rescale=sf_rescale),
            *args,
            **kwargs,
        )

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))


class S5FBLOSUM(MutSelModel):
    def __init__(
        self,
        muts_file: str,
        subs_file: str,
        matrix_name="BLOSUM62",
        sf_rescale=None,
        *args,
        **kwargs,
    ):
        """
        Initialize a mutation-selection model from S5F and a BLOSUM matrix.

        Parameters:
        muts_file (str): file of mutabilities per 5-mer motif.
        subs_file (str): file of substitution probabilities per 5-mer motif.
        matrix_name (str): Name of BLOSUM matrix (e.g. "BLOSUM45", "BLOSUM62", "BLOSUM80", "BLOSUM90")
        sf_rescale (str, optional): The selection factor rescaling approach.
        """
        super().__init__(
            mutation_model=S5F(muts_file=muts_file, subs_file=subs_file),
            selection_model=BLOSUM(matrix_name=matrix_name, sf_rescale=sf_rescale),
            *args,
            **kwargs,
        )

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))
