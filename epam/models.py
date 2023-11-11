from abc import ABC, abstractmethod
from importlib import resources
import logging
from typing import Tuple
import ablang
import shmple
import torch
import torch.optim as optim
from torch import Tensor

from esm import (
    pretrained,
)
import h5py
import numpy as np
import pandas as pd
from scipy.special import softmax
import epam.molevol as molevol
import epam.sequences as sequences
from epam.sequences import (
    AA_STR_SORTED,
    assert_pcp_lengths,
    translate_sequences,
    pcp_criteria_check,
)
import epam.utils as utils

with resources.path("epam", "__init__.py") as p:
    DATA_DIR = str(p.parent.parent) + "/data/"

# Here's a list of the models and configurations we will use in our tests and
# pipeline.

FULLY_SPECIFIED_MODELS = [
    ("AbLang_heavy", "AbLang", {"chain": "heavy"}),
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
    ("ESM1v_default", "ESM1v", {}),
    (
        "SHMple_ESM1v",
        "SHMpleESM",
        {"weights_directory": DATA_DIR + "shmple_weights/my_shmoof"},
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
        self.model = shmple.AttentionModel(
            weights_dir=weights_directory, log_level=logging.WARNING
        )

    def predict_rates_and_normed_sub_probs(
        self, parent: str, branch_length: float
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
            [parent], [branch_length]
        )
        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        return torch.tensor(rates.squeeze()), molevol.normalize_sub_probs(
            parent_idxs, torch.tensor(subs)
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
        rates, subs = self.predict_rates_and_normed_sub_probs(parent, branch_length)
        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        return molevol.aaprobs_of_parent_rates_and_sub_probs(parent_idxs, rates, subs)

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
        """
        super().__init__(weights_directory, model_name)
        self.max_optimization_steps = max_optimization_steps
        self.optimization_tol = optimization_tol
        self.learning_rate = learning_rate

    def _build_log_pcp_probability(
        self, parent: str, child: str, rates: Tensor, sub_probs: Tensor
    ):
        """Constructs the log_pcp_probability function specific to given rates and sub_probs.

        This function takes log_branch_scaling as input and returns the log
        probability of the child sequence. It uses log of branch scaling to
        ensure non-negativity of the branch length."""

        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        child_idxs = sequences.nt_idx_tensor_of_str(child)

        def log_pcp_probability(log_branch_scaling):
            branch_scaling = torch.exp(log_branch_scaling)
            mut_probs = 1.0 - torch.exp(-rates * branch_scaling)
            no_mutation_sites = parent_idxs == child_idxs

            same_probs = 1.0 - mut_probs[no_mutation_sites]
            diff_probs = (
                mut_probs[~no_mutation_sites]
                * sub_probs[~no_mutation_sites, child_idxs[~no_mutation_sites]]
            )
            child_log_prob = torch.log(torch.cat([same_probs, diff_probs])).sum()

            return child_log_prob

        return log_pcp_probability

    def _find_optimal_branch_length(self, parent, child):
        """
        Find the optimal branch length for a parent-child pair in terms of
        nucleotide likelihood.

        Parameters:
        parent (str): The parent sequence.
        child (str): The child sequence.

        Returns:
        float: The optimal branch length.
        """

        base_branch_length = np.mean([a != b for a, b in zip(parent, child)])
        rates, sub_probs = self.predict_rates_and_normed_sub_probs(
            parent, base_branch_length
        )

        log_pcp_probability = self._build_log_pcp_probability(
            parent, child, rates, sub_probs
        )

        log_branch_scaling = torch.tensor(0.0, requires_grad=True)

        #optimizer = optim.SGD([log_branch_scaling], lr=self.learning_rate)
        optimizer = optim.Adam([log_branch_scaling], lr=self.learning_rate)
        prev_log_branch_scaling = log_branch_scaling.clone()

        for _ in range(self.max_optimization_steps):
            optimizer.zero_grad()

            loss = -log_pcp_probability(log_branch_scaling)
            assert not torch.isnan(
                loss
            ), "Loss is NaN: perhaps selection has given a probability of zero?"
            loss.backward()
            torch.nn.utils.clip_grad_norm_([log_branch_scaling], max_norm=2.5)
            optimizer.step()

            change_in_log_branch_scaling = torch.abs(
                log_branch_scaling - prev_log_branch_scaling
            )
            if change_in_log_branch_scaling < self.optimization_tol:
                break

            prev_log_branch_scaling = log_branch_scaling.clone()

        branch_scaling = torch.exp(log_branch_scaling.detach())
        return branch_scaling * base_branch_length

    def aaprobs_of_parent_child_pair(self, parent, child) -> np.ndarray:
        branch_length = self._find_optimal_branch_length(parent, child)
        if branch_length > 0.5:
            print(f"Warning: branch length of {branch_length} is surprisingly large.")
        return self._aaprobs_of_parent_and_branch_length(parent, branch_length).numpy()


class MutSel(OptimizableSHMple):
    """A mutation selection model using SHMple for the mutation part.

    Note that stop codons are assumed to have zero selection probability.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        This function takes log_branch_scaling as input and returns the log
        probability of the child sequence. It uses log of branch scaling to
        ensure non-negativity of the branch length.
        """

        assert len(parent) % 3 == 0
        sel_matrix = self.build_selection_matrix_from_parent(parent)
        assert sel_matrix.shape == (len(parent) // 3, 20)

        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        child_idxs = sequences.nt_idx_tensor_of_str(child)

        def log_pcp_probability(log_branch_scaling: torch.Tensor):
            branch_scaling = torch.exp(log_branch_scaling)
            mut_probs = 1.0 - torch.exp(-rates * branch_scaling)

            codon_mutsel_v = molevol.build_codon_mutsel_v(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                sub_probs.reshape(-1, 3, 4),
                sel_matrix,
            )

            reshaped_child_idxs = child_idxs.reshape(-1, 3)
            child_prob_vector = codon_mutsel_v[
                torch.arange(len(reshaped_child_idxs)),
                reshaped_child_idxs[:, 0],
                reshaped_child_idxs[:, 1],
                reshaped_child_idxs[:, 2],
            ]

            return torch.sum(torch.log(child_prob_vector))

        return log_pcp_probability

    def _aaprobs_of_parent_and_branch_length(self, parent, branch_length) -> Tensor:
        rates, sub_probs = self.predict_rates_and_normed_sub_probs(
            parent, branch_length
        )

        sel_matrix = self.build_selection_matrix_from_parent(parent)
        mut_probs = 1.0 - torch.exp(-rates)

        parent_idxs = sequences.nt_idx_tensor_of_str(parent)

        codon_mutsel_v = molevol.build_codon_mutsel_v(
            parent_idxs.reshape(-1, 3),
            mut_probs.reshape(-1, 3),
            sub_probs.reshape(-1, 3, 4),
            sel_matrix,
        )

        return molevol.aaprobs_of_codon_probs_v(codon_mutsel_v)


class RandomMutSel(MutSel):
    """A mutation selection model with a random selection matrix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_selection_matrix_from_parent(self, parent: str) -> Tensor:
        matrix = torch.rand(len(parent) // 3, 20)
        matrix /= matrix.sum(dim=1, keepdim=True)
        return matrix


class TorchModel(BaseModel):
    def __init__(self, model_name=None):
        """
        Initialize a PyTorch model and select device.

        Parameters:
        model_name (str, optional): The name of the model. If not specified, the class name is used.

        """
        super().__init__(model_name=model_name)

        # check that CUDA is usable
        def check_CUDA():
            try:
                torch._C._cuda_init()
                return True
            except:
                return False

        if torch.backends.cudnn.is_available() and check_CUDA():
            print("Using CUDA")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Metal Performance Shaders")
            self.device = torch.device("mps")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")


class AbLang(TorchModel):
    def __init__(self, chain="heavy", model_name=None):
        """
        Initialize AbLang model with specified chain and create amino acid string.

        Parameters:
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.

        """
        super().__init__(model_name=model_name)
        self.model = ablang.pretrained(chain, device = self.device)
        self.model.freeze()
        vocab_dict = self.model.tokenizer.vocab_to_aa
        self.aa_str = "".join([vocab_dict[i + 1] for i in range(20)])
        self.aa_str_sorted_indices = np.argsort(list(self.aa_str))
        assert AA_STR_SORTED == "".join(
            np.array(list(self.aa_str))[self.aa_str_sorted_indices]
        )

    def probability_array_of_seq(self, seq: str):
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

    def aaprobs_of_parent_child_pair(self, parent: str, child=None) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the AbLang model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence (ignored for AbLang model).

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        parent_aa = translate_sequences([parent])[0]
        return self.probability_array_of_seq(parent_aa)


class ESM1v(TorchModel):
    def __init__(self, model_name=None):
        """
        Initialize ESM1v model; currently using #1 of 5 models in ensemble.

        Parameters:
        model_name (str, optional): The name of the model. If not specified, the class name is used.

        """
        super().__init__(model_name=model_name)
        self.model, self.alphabet = pretrained.load_model_and_alphabet(
            "esm1v_t33_650M_UR90S_1"
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.aa_idxs = [self.alphabet.get_idx(aa) for aa in AA_STR_SORTED]

    def aaprobs_of_parent_child_pair(self, parent, child=None) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the ESM-1v_1 model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence (ignored for AbLang model)

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        batch_converter = self.alphabet.get_batch_converter()

        parent_aa = translate_sequences([parent])[0]
        data = [
            ("protein1", parent_aa),
        ]

        batch_tokens = batch_converter(data)[2]

        # Get token probabilities before softmax so we can restrict to 20 amino
        # acids in softmax calculation.
        with torch.no_grad():
            batch_tokens = batch_tokens.to(self.device)
            token_probs_pre_softmax = self.model(batch_tokens)["logits"]

        aa_probs = torch.softmax(token_probs_pre_softmax[..., self.aa_idxs], dim=-1)

        aa_probs_np = aa_probs.cpu().numpy().squeeze()

        # Drop first and last elements, which are the probability of the start
        # and end token.
        prob_matrix = aa_probs_np[1:-1, :]

        assert prob_matrix.shape[0] == len(parent_aa)

        return prob_matrix


class SHMpleESM(MutSel):
    def __init__(self, *args, **kwargs):
        """
        Initialize a mutation-selection model using SHMple for the mutation part and ESM-1v_1 for the selection part.
        Parameters:
        weights_directory (str): Directory path to trained SHMple model weights.
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        """
        super().__init__(*args, **kwargs)
        self.selection_model = ESM1v()

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))
