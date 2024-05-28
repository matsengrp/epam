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

import shmple
import epam.molevol as molevol
import epam.sequences as sequences
from epam.sequences import (
    AA_STR_SORTED,
    assert_pcp_lengths,
    translate_sequence,
    pcp_criteria_check,
    aa_idx_array_of_str,
)
from epam.torch_common import pick_device, optimize_branch_length, SMALL_PROB
import epam.utils as utils

with resources.path("epam", "__init__.py") as p:
    DATA_DIR = str(p.parent.parent) + "/data/"

# Here's a list of the models and configurations we will use in our tests and
# pipeline.

FULLY_SPECIFIED_MODELS = [
    ("AbLang1", "AbLang1", {"chain": "heavy"}),
    ("AbLang2_wt", "AbLang2", {"version": "ablang2-paired", "masking": False}),
    ("AbLang2_mask", "AbLang2", {"version": "ablang2-paired", "masking": True}),
    (
        "SHMple_default",
        "SHMple",
        {"weights_directory": DATA_DIR + "shmple_weights/my_shmoof"},
    ),
    (
        "SHMple_productive",
        "SHMple",
        {"weights_directory": DATA_DIR + "shmple_weights/prod_shmple"},
    ),
    ("ESM1v_wt", "CachedESM1v", {}),
    ("ESM1v_mask", "CachedESM1v", {"sf_rescale": "sigmoid"}),
    (
        "SHMpleESM_wt",
        "SHMpleESM",
        {"weights_directory": DATA_DIR + "shmple_weights/my_shmoof"},
    ),
    (
        "SHMpleESM_mask",
        "SHMpleESM",
        {
            "weights_directory": DATA_DIR + "shmple_weights/my_shmoof",
            "sf_rescale": "sigmoid",
        },
    ),
]


class BaseModel(ABC):
    def __init__(self, model_name=None, logging=False):
        """
        Initializes a new instance of the BaseModel class.

        Parameters:
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        """
        if model_name is None:
            model_name = self.__class__.__name__
        self.model_name = model_name
        self.logging = logging
        if self.logging == True:
            self.csv_file = open(f"{self.model_name}_branch_opt_fails_{int(time.time())}.csv", "w")
            self.csv_file.write("pcp_index,parent,child,mut_freq,opt_branch_length,fail_to_converge\n")

    @abstractmethod
    def aaprobs_of_parent_child_pair(self, parent: str, child: str) -> np.ndarray:
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

    def write_aaprobs(self, pcp_path: str, output_path: str):
        """
        Write an aaprob matrix for each parent-child pair (PCP) of nucleotide sequences with a substitution model.

        An HDF5 output file is created that includes the file path to the PCP data and a checksum for verification.

        Parameters:
        pcp_filename (str): file name of parent-child pair data.
        output_filename (str): output file name.

        """
        checksum = utils.generate_file_checksum(pcp_path)
        pcp_df = pd.read_csv(pcp_path, index_col=0)

        with h5py.File(output_path, "w") as outfile:
            # attributes related to PCP data file
            outfile.attrs["checksum"] = checksum
            outfile.attrs["pcp_filename"] = pcp_path
            outfile.attrs["model_name"] = self.model_name

            for i, row in pcp_df.iterrows():
                parent = row["parent"]
                child = row["child"]
                assert_pcp_lengths(parent, child)
                if pcp_criteria_check(parent, child):
                    if self.logging == True:
                        self.csv_file.write(f"{i},")
                                        
                    matrix = self.aaprobs_of_parent_child_pair(parent, child)

                    # create a group for each matrix
                    grp = outfile.create_group(f"matrix{i}")
                    grp.attrs["pcp_index"] = i
                    grp.create_dataset(
                        "data", data=matrix, compression="gzip", compression_opts=4
                    )


class SHMple(BaseModel):
    def __init__(self, weights_directory: str, model_name=None):
        """
        Initialize a SHMple model with specified directory to trained model weights.

        Parameters:
        weights_directory (str): directory path to trained model weights.
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        """
        super().__init__(model_name=model_name)
        # It's a little strange to have no shmple model, but that's useful for
        # cases when we've pre-recorded the mutabilities for our likelihood
        # function and are just using this as a framework for branch length
        # optimization. In any case this is going to change once we shift over
        # to using netam models.
        if weights_directory is None:
            self.model = None
        else:
            self.model = shmple.AttentionModel(
                weights_dir=weights_directory, log_level=logging.WARNING
            )

    def predict_rates_and_normed_subs_probs(
        self, parent: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A wrapper for the predict_mutabilities_and_substitutions method of the
        SHMple model that normalizes the substitution probabilities, as well as
        unpacking and squeezing the results.

        We have to do this because the SHMple model returns substitution
        probabilities that are nearly normalized, but not quite.

        Parameters:
        parent (str): The parent sequence.
        branch_length (float): The branch length.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the rates and
            substitution probabilities as Torch tensors.
        """
        [rates], [subs] = self.model.predict_mutabilities_and_substitutions(
            [parent], [1.0]
        )
        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        return torch.tensor(
            rates.squeeze(), dtype=torch.float
        ), molevol.normalize_sub_probs(
            parent_idxs, torch.tensor(subs, dtype=torch.float)
        )

    def _aaprobs_of_parent_and_branch_length(
        self, parent: str, branch_length: float
    ) -> torch.Tensor:
        """
        Calculate the amino acid probabilities for a given parent and branch length.

        This is the key function that needs to be overridden for implementing a new model.

        Parameters:
        parent: str
            The parent nucleotide sequence.
        branch_length: float
            The length of the branch.

        Returns:
        np.ndarray: The aaprobs for every codon of the parent sequence.
        """
        rates, subs = self.predict_rates_and_normed_subs_probs(parent)
        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        return molevol.aaprobs_of_parent_scaled_rates_and_sub_probs(
            parent_idxs, rates * branch_length, subs
        )

    def aaprobs_of_parent_child_pair(self, parent: str, child: str) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to a SHMple model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence.

        Returns:
        np.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.
        """
        branch_length = np.mean([a != b for a, b in zip(parent, child)])
        return self._aaprobs_of_parent_and_branch_length(parent, branch_length).numpy()


class OptimizableSHMple(SHMple):
    def __init__(
        self,
        weights_directory,
        model_name=None,
        max_optimization_steps=1000,
        optimization_tol=1e-4,
        learning_rate=0.1,
        sf_rescale=None,
    ):
        """
        Initialize a SHMple model that optimizes branch length for each parent-child pair.

        Parameters:
        weights_directory : str
            Directory containing the trained model weights.
        model_name : str, optional
            Model name. Default is None, setting the model name to the class name.
        max_optimization_steps : int, optional
            Maximum number of gradient descent steps. Default is 1000.
        optimization_tol : float, optional
            Tolerance for optimization of log(branch length). Default is 1e-4.
        learning_rate : float, optional
            Learning rate for torch's SGD. Default is 0.1.
        sf_rescale : str, optional
            Selection factor rescaling approach used in SHMpleESM for ratios
            produced under mask-marginals scoring strategy. Using sigmoid transformation
            currently and nothing for wt-marginals selection factors.
        """
        super().__init__(weights_directory, model_name)
        self.max_optimization_steps = max_optimization_steps
        self.optimization_tol = optimization_tol
        self.learning_rate = learning_rate
        self.sf_rescale = sf_rescale

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
        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length,
            self.learning_rate,
            self.max_optimization_steps,
            self.optimization_tol,
        )

    def aaprobs_of_parent_child_pair(self, parent, child) -> np.ndarray:
        base_branch_length = sequences.nt_mutation_frequency(parent, child)
        branch_length = self._find_optimal_branch_length(
            parent, child, base_branch_length
        )
        if branch_length > 0.5:
            print(f"Warning: branch length of {branch_length} is surprisingly large.")
        return self._aaprobs_of_parent_and_branch_length(parent, branch_length).numpy()


class MutSel(OptimizableSHMple):
    """A mutation selection model using SHMple for the mutation part.

    Note that stop codons are assumed to have zero selection probability.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def _build_log_pcp_probability(
        self, parent: str, child: str, rates: Tensor, sub_probs: Tensor
    ):
        """
        Constructs the log_pcp_probability function specific to given rates and sub_probs.

        This function takes log_branch_length as input and returns the log
        probability of the child sequence. It uses log of branch length to
        ensure non-negativity.
        """

        assert len(parent) % 3 == 0
        sel_matrix = self.build_selection_matrix_from_parent(parent)
        assert sel_matrix.shape == (len(parent) // 3, 20)

        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        child_idxs = sequences.nt_idx_tensor_of_str(child)

        def log_pcp_probability(log_branch_length: torch.Tensor):
            branch_length = torch.exp(log_branch_length)
            mut_probs = 1.0 - torch.exp(-branch_length * rates)

            codon_mutsel, sums_too_big = molevol.build_codon_mutsel(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                sub_probs.reshape(-1, 3, 4),
                sel_matrix,
            )

            # This is a diagnostic generating data for netam issue #7.
            # if sums_too_big is not None:
            #     self.csv_file.write(f"{parent},{child},{branch_length},{sums_too_big}\n")

            reshaped_child_idxs = child_idxs.reshape(-1, 3)
            child_prob_vector = codon_mutsel[
                torch.arange(len(reshaped_child_idxs)),
                reshaped_child_idxs[:, 0],
                reshaped_child_idxs[:, 1],
                reshaped_child_idxs[:, 2],
            ]

            child_prob_vector = torch.clamp(child_prob_vector, min=1e-10)

            result = torch.sum(torch.log(child_prob_vector))

            assert torch.isfinite(result)

            return result

        return log_pcp_probability

    def _aaprobs_of_parent_and_branch_length(self, parent, branch_length) -> Tensor:
        rates, sub_probs = self.predict_rates_and_normed_subs_probs(parent)

        # Apply a sigmoid transformation for selection factors with some values greater than
        # 1. This occurs when using ratios under ESM mask-marginals.
        if self.sf_rescale == "sigmoid":
            ratio_sel_matrix = self.build_selection_matrix_from_parent(parent)
            sel_matrix = utils.selection_factor_ratios_to_sigmoid(ratio_sel_matrix)
        else:
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


class RandomMutSel(MutSel):
    """A mutation selection model with a random selection matrix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_selection_matrix_from_parent(self, parent: str) -> Tensor:
        matrix = torch.rand(len(parent) // 3, 20)
        matrix /= matrix.sum(dim=1, keepdim=True)
        return matrix


class AbLang1(BaseModel):
    def __init__(
        self,
        chain="heavy",
        model_name=None,
        optimize=True,
        max_optimization_steps=1000,
        optimization_tol=1e-4,
        learning_rate=0.1,
    ):
        """
        Initialize AbLang model with specified chain and create amino acid string. This model rescales amino acid probabilities from AbLang with an optimized branch length for each parent-child pair for comparison with CTMC models.

        Parameters:
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        max_optimization_steps (int, optional): Maximum number of gradient descent steps. Default is 1000.
        optimization_tol (float, optional): Tolerance for optimization of log(branch length). Default is 1e-4.
        learning_rate (float, optional): Learning rate for torch's SGD. Default is 0.1.

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
        self.optimize = optimize
        self.max_optimization_steps = max_optimization_steps
        self.optimization_tol = optimization_tol
        self.learning_rate = learning_rate
       
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

    def _build_log_pcp_probability(
        self, parent: str, child: str, child_aa_probs: Tensor
    ):
        """
        Constructs the log_pcp_probability function specific to given aa_probs for the child sequence from AbLang.

        This function takes log_branch_length as input and returns the log
        probability of the child sequence. It uses log of branch length to
        ensure non-negativity. The probability of the child sequence is scaled
        here by the probability of no substitution event (p_no_event), which is
        equivalent to e^{-t} and bounded between 0 and 1.

        """

        parent_idx = sequences.aa_idx_tensor_of_str(parent)
        child_idx = sequences.aa_idx_tensor_of_str(child)

        def log_pcp_probability(log_branch_length):
            branch_length = torch.exp(log_branch_length)
            p_no_event = torch.exp(-branch_length)
            sub_probs = p_no_event * child_aa_probs

            no_sub_sites = parent_idx == child_idx

            # Rescaling each site based on whether a substitution event occurred or not.
            same_probs = p_no_event + child_aa_probs[no_sub_sites] - sub_probs[no_sub_sites]
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
        prob_arr (numpy.ndarray): A 2D array containing the unscaled probabilities of the amino acids by site computed by AbLang.

        """
        child_prob = self.probability_vector_of_child_seq(prob_arr, child)
        prob_tensor = torch.tensor(child_prob, dtype=torch.float)
        log_pcp_probability = self._build_log_pcp_probability(
            parent, child, prob_tensor
        )
        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length,
            self.learning_rate,
            self.max_optimization_steps,
            self.optimization_tol,
        )

    def scale_probability_array(
        self, prob_arr: np.ndarray, parent: str, branch_length: float
    ) -> np.ndarray:
        """
        Rescale the amino acid probability matrix from AbLang with the optimized "branch length".

        For fair comparison with CTMC models, we apply a linear rescaling of the amino acid probabilities. By itself,
        AbLang does not any notion of branch length and will make the same predicition regardless of evolutionary
        time between the parent and child sequence. We rescale each prob_arr with p_no_event, where the probability of
        no subsititution is (1 - p_no_event) + p_no_event * prob_arr and the probability of subsitution is p_no_event * prob_arr.
        For each PCP, the value of p_no_event is optimized to maximize the likelihood of the child sequence. This is
        more or less equivalent to scaling the branch length in SHMple mut-sel models.


        Parameters:
        prob_arr (numpy.ndarray): A 2D array containing the normalized probabilities of the amino acids by site.
        parent (str): The parent sequence.
        branch_length (float): The branch length.

        Returns:
        numpy.ndarray: A 2D array containing the scaled probabilities of the amino acids by site.

        """
        p_no_event = np.exp(-branch_length)
        scaled_prob_arr = np.zeros(prob_arr.shape)

        parent_idx = sequences.aa_idx_array_of_str(parent)
        mask_parent = np.eye(20, dtype=bool)[parent_idx]

        scaled_prob_arr[mask_parent] = (
            p_no_event + ((1 - p_no_event) * prob_arr[mask_parent])
        )
        scaled_prob_arr[~mask_parent] = (1 - p_no_event) * prob_arr[~mask_parent]

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

    def aaprobs_of_parent_child_pair(self, parent: str, child: str) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the AbLang model with a branch length optimization.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent nucleotide sequence for which we want the array of probabilities.
        child (str): The child nucleotide sequence (ignored for AbLang model).

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        base_branch_length = sequences.nt_mutation_frequency(parent, child)
        parent_aa = translate_sequence(parent)
        child_aa = translate_sequence(child)

        unscaled_aaprob = self.probability_array_of_seq(parent_aa)
        if self.optimize == False:
            return unscaled_aaprob
        
        branch_length, converge_status = self._find_optimal_branch_length(
            parent_aa, child_aa, base_branch_length, unscaled_aaprob
        )
        if self.logging == True:
            self.csv_file.write(f"{parent_aa},{child_aa},{base_branch_length},{branch_length},{converge_status}\n")

        return self.scale_probability_array(unscaled_aaprob, parent_aa, branch_length)


class AbLang2(BaseModel):
    def __init__(
        self,
        version="ablang2-paired",
        masking=False,
        optimize=True,
        model_name=None,
        max_optimization_steps=1000,
        optimization_tol=1e-4,
        learning_rate=0.1,
    ):
        """
        Initialize AbLang2 model with or without masking. This model rescales amino acid probabilities from AbLang with an optimized branch length for each parent-child pair for comparison with CTMC models.

        Parameters:
        version (str, optional): Version of the AbLang model. Options currently limited to 'ablang2-paired' but could theoretically support 'ablang1-heavy' and 'ablang1-light'.
        masking (bool, optional): Whether to use masking in the model. Default is False.
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        optimize (bool, optional): Whether to optimize branch length for each parent-child pair. Default is True.
        max_optimization_steps (int, optional): Maximum number of gradient descent steps. Default is 1000.
        optimization_tol (float, optional): Tolerance for optimization of log(branch length). Default is 1e-4.
        learning_rate (float, optional): Learning rate for torch's SGD. Default is 0.1.

        """
        super().__init__(model_name=model_name)
        self.version = version
        self.device = pick_device()
        self.model = ablang2.pretrained(model_to_use=self.version, device=self.device)
        self.model.freeze()
        self.masking = masking
        vocab_dict = self.model.tokenizer.aa_to_token
        self.aa_sorted_indices = [vocab_dict[aa] for aa in AA_STR_SORTED]
        assert AA_STR_SORTED == "".join(
            [key for value in self.aa_sorted_indices for key, v in vocab_dict.items() if v == value]
        )
        self.optimize = optimize
        self.max_optimization_steps = max_optimization_steps
        self.optimization_tol = optimization_tol
        self.learning_rate = learning_rate

    def probability_array_of_seq(self, seq: str) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the AbLang model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        seq (str): The sequence for which we want the array of probabilities.

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        assert self.masking in [True, False], "masking must be set to True or False"
        likelihoods = self.model([seq, ""], mode="likelihood", stepwise_masking=self.masking)
        seq_likelihoods = likelihoods[0]

        # Apply softmax to the second dimension. Skipping the first and last
        # elements (which are the probability of the start, end, and heavy|light divider token),
        # as well as all tokens not corresponding to the 20 AAs.
        arr_sorted = np.apply_along_axis(softmax, 1, seq_likelihoods[1:-2, self.aa_sorted_indices])

        if self.masking == False:
            assert len(seq) == arr_sorted.shape[0]
            return arr_sorted
        elif self.masking == True:
            # Take the ratio of the probabilities relative to the parent AA.
            parent_idx = aa_idx_array_of_str(seq)
            parent_probs = arr_sorted[np.arange(len(seq)), parent_idx]
            arr_prob_ratio = arr_sorted / parent_probs[:, None]

            # Sigmoid transformation for probability ratios with some values greater than 1.
            arr_ratio_sig = utils.probability_ratios_to_sigmoid(arr_prob_ratio)

            # Normalize the probabilities to sum to 1.
            row_sums = np.sum(arr_ratio_sig, axis=1, keepdims=True)
            arr_ratio_norm = arr_ratio_sig / row_sums

            assert len(seq) == arr_ratio_norm.shape[0]

            return arr_ratio_norm
        else:
            raise ValueError("masking must be set to True or False")

    def _build_log_pcp_probability(
        self, parent: str, child: str, child_aa_probs: Tensor
    ):
        """
        Constructs the log_pcp_probability function specific to given aa_probs for the child sequence from AbLang.

        This function takes log_branch_length as input and returns the log
        probability of the child sequence. It uses log of branch length to
        ensure non-negativity. The probability of the child sequence is scaled
        here by the probability of no substitution event (p_no_event), which is
        equivalent to e^{-t} and bounded between 0 and 1.

        """

        parent_idx = sequences.aa_idx_tensor_of_str(parent)
        child_idx = sequences.aa_idx_tensor_of_str(child)

        def log_pcp_probability(log_branch_length):
            branch_length = torch.exp(log_branch_length)
            p_no_event = torch.exp(-branch_length)
            sub_probs = p_no_event * child_aa_probs

            no_sub_sites = parent_idx == child_idx

            # Rescaling each site based on whether a substitution event occurred or not.
            same_probs = p_no_event + child_aa_probs[no_sub_sites] - sub_probs[no_sub_sites]
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
        prob_arr (numpy.ndarray): A 2D array containing the unscaled probabilities of the amino acids by site computed by AbLang.

        """
        child_prob = self.probability_vector_of_child_seq(prob_arr, child)
        prob_tensor = torch.tensor(child_prob, dtype=torch.float)
        log_pcp_probability = self._build_log_pcp_probability(
            parent, child, prob_tensor
        )
        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length,
            self.learning_rate,
            self.max_optimization_steps,
            self.optimization_tol,
        )

    def scale_probability_array(
        self, prob_arr: np.ndarray, parent: str, branch_length: float
    ) -> np.ndarray:
        """
        Rescale the amino acid probability matrix from AbLang with the optimized "branch length".

        For fair comparison with CTMC models, we apply a linear rescaling of the amino acid probabilities. By itself,
        AbLang does not any notion of branch length and will make the same predicition regardless of evolutionary
        time between the parent and child sequence. We rescale each prob_arr with p_no_event, where the probability of
        no subsititution is (1 - p_no_event) + p_no_event * prob_arr and the probability of subsitution is p_no_event * prob_arr.
        For each PCP, the value of p_no_event is optimized to maximize the likelihood of the child sequence. This is
        more or less equivalent to scaling the branch length in SHMple mut-sel models.


        Parameters:
        prob_arr (numpy.ndarray): A 2D array containing the normalized probabilities of the amino acids by site.
        parent (str): The parent sequence.
        branch_length (float): The branch length.

        Returns:
        numpy.ndarray: A 2D array containing the scaled probabilities of the amino acids by site.

        """
        p_no_event = np.exp(-branch_length)
        scaled_prob_arr = np.zeros(prob_arr.shape)

        parent_idx = sequences.aa_idx_array_of_str(parent)
        mask_parent = np.eye(20, dtype=bool)[parent_idx]

        scaled_prob_arr[mask_parent] = (
            p_no_event + ((1 - p_no_event) * prob_arr[mask_parent])
        )
        scaled_prob_arr[~mask_parent] = (1 - p_no_event) * prob_arr[~mask_parent]

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

    def aaprobs_of_parent_child_pair(self, parent: str, child: str) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the AbLang model with a branch length optimization.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence (ignored for AbLang model).

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        base_branch_length = sequences.nt_mutation_frequency(parent, child)
        parent_aa = translate_sequence(parent)
        child_aa = translate_sequence(child)

        unscaled_aaprob = self.probability_array_of_seq(parent_aa)
        if self.optimize == False:
            return unscaled_aaprob
        
        branch_length, converge_status = self._find_optimal_branch_length(
            parent_aa, child_aa, base_branch_length, unscaled_aaprob
        )
        if self.logging == True:
            self.csv_file.write(f"{parent_aa},{child_aa},{base_branch_length},{branch_length},{converge_status}\n")

        return self.scale_probability_array(unscaled_aaprob, parent_aa, branch_length)


class CachedESM1v(BaseModel):
    def __init__(self, model_name=None, sf_rescale=None):
        """
        Initialize ESM1v with cached selection matrices generated in esm_precompute.py.

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
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.
        """
        assert (
            parent in self.selection_matrices.keys()
        ), f"{parent} not present in CachedESM."
        if self.sf_rescale == "sigmoid":
            # Sigmoid transformation for selection factors with some values greater than 1.
            ratio_sel_matrix = torch.tensor(self.selection_matrices[parent])
            sel_tensor = utils.selection_factor_ratios_to_sigmoid(ratio_sel_matrix)

            # Normalize the selection matrix.
            row_sums = sel_tensor.sum(dim=1, keepdim=True)
            sel_tensor /= row_sums
            sel_matrix = sel_tensor.numpy()
        else:
            sel_matrix = self.selection_matrices[parent]
        return sel_matrix


class SHMpleESM(MutSel):
    def __init__(self, *args, **kwargs):
        """
        Initialize a mutation-selection model using SHMple for the mutation part and ESM-1v_1 for the selection part.

        Parameters:
        weights_directory (str): Directory path to trained SHMple model weights.
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        """
        super().__init__(*args, **kwargs)

    def preload_esm_data(self, hdf5_path):
        """
        Preload ESM1v data from HDF5 file.

        Parameters:
        hdf5_path (str): Path to HDF5 file containing pre-computed selection matrices.
        """
        self.selection_model = CachedESM1v()
        self.selection_matrices = self.selection_model.preload_esm_data(hdf5_path)

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))


class WrappedBinaryMutSel(MutSel):
    """A mutation selection model that is built from a model that has a `selection_factors_of_aa_str` method."""

    def __init__(self, selection_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_model = selection_model

    def build_selection_matrix_from_parent(self, parent: str):
        parent = translate_sequence(parent)
        selection_factors = self.selection_model.selection_factors_of_aa_str(parent)
        selection_matrix = torch.zeros((len(selection_factors), 20), dtype=torch.float)
        # Every "off-diagonal" entry of the selection matrix is set to the selection
        # factor, where "diagonal" means keeping the same amino acid.
        selection_matrix[:, :] = selection_factors[:, None]
        # Set "diagonal" elements to one.
        parent_idxs = sequences.aa_idx_array_of_str(parent)
        selection_matrix[torch.arange(len(parent_idxs)), parent_idxs] = 1.0

        return selection_matrix
