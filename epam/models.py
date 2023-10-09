from abc import ABC, abstractmethod
import logging
import ablang
import shmple
import torch
from esm import (
    Alphabet,
    FastaBatchedDataset,
    ProteinBertModel,
    pretrained,
    MSATransformer,
)
import h5py
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import epam.sequences as sequences
from epam.sequences import (
    NT_STR_SORTED,
    AA_STR_SORTED,
    CODON_AA_INDICATOR_MATRIX,
    truncate_sequence_at_codon_boundary,
    translate_sequences,
)
import epam.utils as utils


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
    def aaprobs_of_parent_child_pair(self, parent, child) -> np.ndarray:
        pass

    def probability_vector_of_child_seq(self, prob_arr, child_seq):
        """
        Calculate the sitewise probability of a child sequence given a probability array.

        Parameters:
        prob_arr (numpy.ndarray): A 2D array containing the normalized probabilities of the amino acids by site.
        child_seq (str): The child sequence for which we want the probability vector.

        Returns:
        numpy.ndarray: A 1D array containing the sitewise probability of the child sequence.

        """
        assert (
            len(child_seq) == prob_arr.shape[1]
        ), "The child sequence length does not match the probability array length."

        return np.array(
            [prob_arr[AA_STR_SORTED.index(aa), i] for i, aa in enumerate(child_seq)]
        )

    def write_aaprobs(self, pcp_path, output_path):
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
                # NOTE: this shouldn't be necessary after Kevin fixes the data
                parent = truncate_sequence_at_codon_boundary(row["parent"])
                child = truncate_sequence_at_codon_boundary(row["child"])
                matrix = self.aaprobs_of_parent_child_pair(parent, child)

                # create a group for each matrix
                grp = outfile.create_group(f"matrix{i}")
                grp.attrs["pcp_index"] = i

                # enable gzip compression
                grp.create_dataset(
                    "data", data=matrix, compression="gzip", compression_opts=4
                )

    def plot_sequences(self, seqs):
        """
        Plot the normalized probabilities of the various amino acids by site for
        each sequence in seqs.

        Parameters:
        seqs (list): List of sequences to plot.

        """
        plt.figure(figsize=(10, 6))

        for seq in seqs:
            # get probability array for this sequence
            arr = self.probability_array_of_seq(seq)
            # create a line plot for this sequence
            plt.plot(self.probability_vector_of_child_seq(arr, seq), label=seq)

        plt.legend()
        plt.xlabel("Site")
        plt.ylabel("Probability")
        plt.title("Sequence Probabilities")
        plt.show()


class AbLang(BaseModel):
    def __init__(self, chain="heavy", model_name=None):
        """
        Initialize AbLang model with specified chain and create amino acid string.

        Parameters:
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.

        """
        super().__init__(model_name=model_name)
        self.model = ablang.pretrained(chain)
        self.model.freeze()
        vocab_dict = self.model.tokenizer.vocab_to_aa
        self.aa_str = "".join([vocab_dict[i + 1] for i in range(20)])
        self.aa_str_sorted_indices = np.argsort(list(self.aa_str))
        assert AA_STR_SORTED == "".join(
            np.array(list(self.aa_str))[self.aa_str_sorted_indices]
        )

    def probability_array_of_seq(self, seq):
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
        arr = np.apply_along_axis(softmax, 1, likelihoods[0, 1:-1]).T

        # Sort rows according to the sorted amino acid string.
        arr_sorted = arr[self.aa_str_sorted_indices]
        assert len(seq) == arr_sorted.shape[1]

        return arr_sorted

    def aaprobs_of_parent_child_pair(self, parent, child=None) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the AbLang model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence (ignored for AbLang model)

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        parent_aa = translate_sequences([parent])[0]
        return self.probability_array_of_seq(parent_aa)


class SHMple(BaseModel):
    def __init__(self, weights_directory, model_name=None):
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

    @staticmethod
    def _normalize_sub_probs(parent: str, sub_probs: np.ndarray) -> np.ndarray:
        """
        Normalize substitution probabilities.

        Given a parent DNA sequence and a 2D numpy array representing substitution
        probabilities, this function sets the probability of the actual nucleotide
        in the parent sequence to zero and then normalizes each row to form a valid
        probability distribution.

        Parameters:
        parent (str): The parent sequence.
        sub_probs (np.ndarray): A 2D numpy array representing substitution
                                probabilities. Rows correspond to sites, and columns
                                correspond to "ACGT" bases.

        Returns:
        np.ndarray: A 2D numpy array with normalized substitution probabilities.
        """
        for i, base in enumerate(parent):
            idx = NT_STR_SORTED.index(base)
            sub_probs[i, idx] = 0.0

        row_sums = np.sum(sub_probs, axis=1, keepdims=True)
        return sub_probs / row_sums

    def predict_rates_and_normed_sub_probs(self, parent, branch_length):
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
        tuple: A tuple containing the rates and substitution probabilities.
        """
        [rates], [subs] = self.model.predict_mutabilities_and_substitutions(
            [parent], [branch_length]
        )
        return rates.squeeze(), self._normalize_sub_probs(parent, subs)

    @staticmethod
    def _build_mutation_matrix(parent_codon, mut_probs, sub_probs):
        """Generate a matrix that represents the mutation probability for each
        site in a given parent codon. So, the ijth entry of the matrix is the
        probability of the ith position mutating to the jth nucleotide.

        See codon_to_aa_probabilities for a description of the arguments, and
        tests for an example."""

        result_matrix = np.empty((len(parent_codon), len(NT_STR_SORTED)))

        for site, nucleotide in enumerate(parent_codon):
            try:
                parent_nt_idx = NT_STR_SORTED.index(nucleotide)
            except ValueError:
                raise ValueError(
                    f"Invalid nucleotide {nucleotide} in codon {parent_codon}"
                )

            for j, nt in enumerate(NT_STR_SORTED):
                if j == parent_nt_idx:
                    result_matrix[site, j] = 1 - mut_probs[site]
                else:
                    result_matrix[site, j] = mut_probs[site] * sub_probs[site][j]

        return result_matrix

    @staticmethod
    def _codon_probs_of_mutation_matrix(mut_matrix):
        """
        Compute the probability tensor for mutating to the codon ijk.

        This method calculates the tensor where the ijk-th entry represents
        the probability of mutating to the codon formed by the i-th, j-th,
        and k-th nucleotide in the nucleotide list. It uses numpy outer
        products to construct this tensor.

        Parameters:
        mut_matrix (numpy.ndarray): A 3D array representing the mutation
                                    matrix. The mutation matrix should be
                                    formatted as per _build_mutation_matrix.

        Returns:
        numpy.ndarray: A 3D array where the ijk-th entry is the probability
                    of mutating to the codon ijk.
        """
        return (
            mut_matrix[0][:, np.newaxis, np.newaxis]
            * mut_matrix[1][np.newaxis, :, np.newaxis]
            * mut_matrix[2][np.newaxis, np.newaxis, :]
        )

    @staticmethod
    def codon_to_aa_probabilities(parent_codon, mut_probs, sub_probs):
        """
        For a specified codon and given nucleotide mutability and substitution probabilities,
        compute the amino acid substitution probabilities.

        Following terminology from Yaari et al 2013, "mutability" refers to the probability
        of a nucleotide mutating at a given site, while "substitution" refers to the probability
        of a nucleotide mutating to another nucleotide at a given site conditional on having
        a mutation.

        We assume that the mutation and substitution probabilities already take branch length
        into account. Here we translate those into amino acid probabilities, which are normalized.
        Probabilities to stop codons are dropped, but self probabilities are kept.

        Parameters:
        parent_codon (str): The specified codon.
        mut_probs (list): The mutability probabilities for each site in the codon.
        sub_probs (list): The substitution probabilities for each site in the codon.

        Returns:
        list: An array of probabilities for all 20 amino acids.

        """
        mut_matrix = SHMple._build_mutation_matrix(parent_codon, mut_probs, sub_probs)
        codon_probs = SHMple._codon_probs_of_mutation_matrix(mut_matrix)
        aa_probs = codon_probs.reshape(-1) @ CODON_AA_INDICATOR_MATRIX
        aa_probs /= aa_probs.sum()
        return aa_probs

    def _aaprobs_of_parent_and_branch_length(
        self, parent, branch_length
    ) -> np.ndarray:
        rates, subs = self.predict_rates_and_normed_sub_probs(parent, branch_length)

        # This `mut_probs` is the probability of at least one mutation at each site.
        # So here we are interpreting the probability in the correctly-specified way rather than the mis-specified
        # way. This is helpful because we'd like normalized probabilities.
        mut_probs = 1.0 - np.exp(-rates)

        # keep track of probabilities as a row per amino acid site, then take transpose before returning output
        aaprobs = []

        for i in range(0, len(parent), 3):
            parent_codon = parent[i : i + 3]
            codon_mut_probs = mut_probs[i : i + 3]
            codon_subs = subs[i : i + 3]

            site_probs = self.codon_to_aa_probabilities(
                parent_codon, codon_mut_probs, codon_subs
            )
            aaprobs.append(site_probs)

        return np.array(aaprobs).transpose()

    def aaprobs_of_parent_child_pair(self, parent, child) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to a SHMple model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence.

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.
        """
        branch_length = np.mean([a != b for a, b in zip(parent, child)])
        return self._aaprobs_of_parent_and_branch_length(parent, branch_length)


class OptimizableSHMple(SHMple):
    def __init__(self, weights_directory, model_name=None):
        super().__init__(weights_directory, model_name)

    def _build_neg_pcp_probability(self, parent, child, rates, sub_probs):
        """Constructs the neg_pcp_probability function specific to given rates and sub_probs.

        This function takes log_branch_scaling as input and returns the negative
        probability of the child sequence. It uses log of branch scaling to
        ensure non-negativity of the branch length."""

        def neg_pcp_probability(log_branch_scaling):
            branch_scaling = np.exp(log_branch_scaling)
            mut_probs = 1.0 - np.exp(-rates * branch_scaling)

            child_prob = 1.0

            for isite in range(len(parent)):
                if parent[isite] == child[isite]:
                    child_prob *= 1.0 - mut_probs[isite]
                else:
                    child_prob *= (
                        mut_probs[isite]
                        * sub_probs[isite][NT_STR_SORTED.index(child[isite])]
                    )

            return -child_prob  # Return the negative probability

        return neg_pcp_probability

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

        neg_pcp_probability = self._build_neg_pcp_probability(
            parent, child, rates, sub_probs
        )

        initial_guess = np.log(1.0)
        result = minimize(
            neg_pcp_probability,
            initial_guess,
            method="BFGS",
            options={"maxiter": 1000, "gtol": 1e-6},
        )
        optimized_branch_scaling_log = result.x[0]
        return np.exp(optimized_branch_scaling_log) * base_branch_length

    def aaprobs_of_parent_child_pair(self, parent, child) -> np.ndarray:
        branch_length = self._find_optimal_branch_length(parent, child)
        return self._aaprobs_of_parent_and_branch_length(parent, branch_length)


class MutSel(OptimizableSHMple):
    """A mutation selection model using SHMple for the mutation part.

    Note that stop codons are assumed to have zero selection probability.
    """

    def __init__(self, weights_directory, model_name=None):
        super().__init__(weights_directory, model_name)

    @abstractmethod
    def build_selection_matrix_from_parent(self, parent):
        """Build the selection matrix (i.e. F matrix) from a parent nucleotide
        sequence.

        The shape of this numpy array should be (len(parent) // 3, 20).
        """
        pass

    def _build_neg_pcp_probability(self, parent, child, rates, sub_probs):
        """Constructs the neg_pcp_probability function specific to given rates and sub_probs.

        This function takes log_branch_scaling as input and returns the negative
        probability of the child sequence. It uses log of branch scaling to
        ensure non-negativity of the branch length."""

        assert len(parent) % 3 == 0
        sel_matrix = self.build_selection_matrix_from_parent(parent)
        assert sel_matrix.shape == (len(parent) // 3, 20)

        def neg_pcp_probability(log_branch_scaling):
            branch_scaling = np.exp(log_branch_scaling)
            mut_probs = 1.0 - np.exp(-rates * branch_scaling)
            child_prob = 1.0

            for i in range(0, len(parent), 3):
                # This implementation is somewhat inefficient because we do the
                # calculation for all of the possible codons every time even
                # though we only use it for the indicated child codon. However,
                # most of the time the parent and the child codons will be the
                # same, and we need to calculate probabilities for every codon
                # in that case (see below).

                parent_codon = parent[i : i + 3]
                child_codon = child[i : i + 3]
                codon_mut_probs = mut_probs[i : i + 3]
                codon_sub_probs = sub_probs[i : i + 3]

                mut_matrix = SHMple._build_mutation_matrix(
                    parent_codon, codon_mut_probs, codon_sub_probs
                )
                codon_probs = SHMple._codon_probs_of_mutation_matrix(mut_matrix)

                # Note that because there are no nonzero entries that correspond to stop, these will have selection probability 0.
                codon_sel_matrix = CODON_AA_INDICATOR_MATRIX @ sel_matrix[i // 3]
                codon_mutsel = codon_probs * codon_sel_matrix.reshape(4, 4, 4)

                # Now we need to calculate the probability of no change in the codon.
                [par0, par1, par2] = sequences.nucleotide_indices_of_codon(parent_codon)
                codon_mutsel[par0, par1, par2] = 0.0
                codon_mutsel[par0, par1, par2] = 1.0 - codon_mutsel.sum()

                [chi0, chi1, chi2] = sequences.nucleotide_indices_of_codon(child_codon)
                child_prob *= codon_mutsel[chi0, chi1, chi2]

            return -child_prob  # Return the negative probability

        return neg_pcp_probability


class RandomMutSel(MutSel):
    """A mutation selection model with a random selection matrix."""

    def __init__(self, weights_directory, model_name=None):
        super().__init__(weights_directory, model_name)

    def build_selection_matrix_from_parent(self, parent):
        matrix = np.random.rand(len(parent) // 3, 20)
        matrix /= matrix.sum(axis=1, keepdims=True)
        return matrix



class TorchModel(BaseModel):
    def __init__(self, model_name=None):
        """
        Initialize a pytorch model and select device.

        Parameters:
        model_name (str, optional): The name of the model. If not specified, the class name is used.

        """
        super().__init__(model_name=model_name)
        if torch.backends.cudnn.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Metal Performance Shaders")
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


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

        # get token probabilities before softmax so we can restrict to 20 amino acids in softmax calculation
        with torch.no_grad():
            batch_tokens = batch_tokens.to(self.device)
            token_probs_pre_softmax = self.model(batch_tokens)["logits"]

        aa_probs = torch.softmax(token_probs_pre_softmax[..., self.aa_idxs], dim=-1)

        aa_probs_np = aa_probs.cpu().numpy().squeeze()

        # drop first and last elements, which are the probability of the start and end token
        prob_matrix = aa_probs_np[1:-1, :].transpose()

        assert prob_matrix.shape[1] == len(parent_aa)

        return prob_matrix


class SHMpleESM(MutSel, ESM1v):
    # this seems not right - check
    def __init__(self, weights_directory, model_name=None):
        """
        Initialize a mutation-selection model using SHMple for the mutation part and ESM-1v_1 for the selection part.
        
        Parameters:
        model_name (str, optional): The name of the model. If not specified, the class name is used.

        """
        super().__init__(weights_directory, model_name)

    def build_selection_matrix_from_parent(self, parent):
        matrix = np.random.rand(len(parent) // 3, 20)
        matrix /= matrix.sum(axis=1, keepdims=True)
        return matrix
