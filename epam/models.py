from abc import ABC, abstractmethod
import ablang
import shmple
import h5py
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import itertools
from epam.sequences import translate_sequences
import epam.utils as utils


class BaseModel(ABC):
    @abstractmethod
    def prob_matrix_of_parent_child_pair(self, parent, child) -> np.ndarray:
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
            [
                prob_arr[self.aa_str_sorted.index(aa), i]
                for i, aa in enumerate(child_seq)
            ]
        )

    def write_probability_matrices(self, pcp_path, output_path):
        """
        Produce a probability matrix for each parent-child pair (PCP) of nucleotide sequences with a substitution model.

        An HDF5 output file is created that includes the file path to the PCP data and a checksum for verification.

        Parameters:
        model (epam.BaseModel): model for predicting substitution probabilities.
        pcp_filename (str): file name of parent-child pair data.
        output_filename (str): output file name.

        """
        checksum = utils.generate_file_checksum(pcp_path)
        pcp_df = pd.read_csv(pcp_path, index_col=0)

        with h5py.File(output_path, "w") as outfile:
            # attributes related to PCP data file
            outfile.attrs["checksum"] = checksum
            outfile.attrs["pcp_filename"] = pcp_path

            for i, row in pcp_df.iterrows():
                parent = row["parent"]
                child = row["child"]
                [parent_aa, child_aa] = translate_sequences([parent, child])
                matrix = self.prob_matrix_of_parent_child_pair(parent_aa, child_aa)

                # create a group for each matrix
                grp = outfile.create_group(f"matrix{i}")
                grp.attrs["pcp_index"] = i

                # enable gzip compression
                grp.create_dataset(
                    "data", data=matrix, compression="gzip", compression_opts=4
                )

    def plot_sequences(self, seqs):
        """
        Plot the normalized probabilities of the various amino acids by site for each sequence in seqs.

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
    def __init__(self, chain="heavy"):
        """
        Initialize AbLang model with specified chain and create amino acid string.

        Parameters:
        chain (str): Name of the chain, default is "heavy".
        """
        self.model = ablang.pretrained(chain)
        self.model.freeze()
        vocab_dict = self.model.tokenizer.vocab_to_aa
        self.aa_str = "".join([vocab_dict[i + 1] for i in range(20)])
        self.aa_str_sorted_indices = np.argsort(list(self.aa_str))
        self.aa_str_sorted = "".join(
            np.array(list(self.aa_str))[self.aa_str_sorted_indices]
        )
        assert self.aa_str_sorted == "".join(sorted(self.aa_str_sorted))

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

        # Apply softmax to the second dimension, and skip the first and last elements (which are the probability of the start and end token).
        arr = np.apply_along_axis(softmax, 1, likelihoods[0, 1:-1]).T

        # Sort rows according to the sorted amino acid string.
        arr_sorted = arr[self.aa_str_sorted_indices]
        assert len(seq) == arr_sorted.shape[1]

        return arr_sorted

    def prob_matrix_of_parent_child_pair(self, parent, child=None) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to the AbLang model.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent sequence for which we want the array of probabilities.
        child (str): The child sequence (ignored for AbLang model)

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        return self.probability_array_of_seq(parent)


class SHMple(BaseModel):
    def __init__(self, weights_directory):
        """
        Initialize a SHMple model with specified directory to trained model weights.

        Parameters:
        weights_directory (str): directory path to trained model weights.
        """
        self.model = shmple.AttentionModel(weights_dir=weights_directory)
        self.nt_str_sorted = "ACGT"
        self.aa_str_sorted = "ACDEFGHIKLMNPQRSTVWY"

    def codon_to_aa_probabilities(self, parent_codon, mut_probs, sub_probs):
        """
        For a specified codon and given nucleotide mutability and substitution probabilities, 
        compute the amino acid substitution probabilities.

        Following terminology from Yaari et al 2013, "mutability" refers to the probability 
        of a nucleotide mutating at a given site, while "substitution" refers to the probability 
        of a nucleotide mutating to another nucleotide at a given site conditional on having 
        a mutation.

        Parameters:
        parent_codon (str): The specified codon.
        mut_probs (list): The mutability probabilities for each site in the codon.
        sub_probs (list): The substitution probabilities for each site in the codon.

        Returns:
        list: An array of probabilities for all 20 amino acids.

        """
        aa_probs = {}
        for aa in self.aa_str_sorted:
            aa_probs[aa] = 0

        # iterate through all possible child codons
        for codon_list in itertools.product(["A", "C", "G", "T"], repeat=3):
            child_codon = "".join(codon_list)

            try:
                aa = translate_sequences([child_codon])[0]
            except ValueError:  # check for STOP codon
                continue

            # iterate through codon sites and compute total probability of potential child codon
            child_prob = 1
            for isite in range(3):
                if parent_codon[isite] == child_codon[isite]:
                    child_prob *= 1 - mut_probs[isite]
                else:
                    child_prob *= mut_probs[isite]
                    child_prob *= sub_probs[isite][
                        self.nt_str_sorted.index(child_codon[isite])
                    ]

            aa_probs[aa] += child_prob

        # need renormalization factor so that amino acid probabilities sum to 1,
        # since probabilities to STOP codon are dropped
        psum = np.sum([aa_probs[aa] for aa in aa_probs.keys()])

        return [aa_probs[aa] / psum for aa in self.aa_str_sorted]

    def prob_matrix_of_parent_child_pair(self, parent, child) -> np.ndarray:
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
        muts, subs = self.model.predict_mutabilities_and_substitutions(
            [parent], [branch_length]
        )

        # keep track of probabilities as a row per amino acid site, then take transpose before returning output
        prob_matrix = []

        for i in range(0, len(parent), 3):
            parent_codon = parent[i : i + 3]
            codon_muts = muts[0][i : i + 3].squeeze()
            codon_subs = subs[0][i : i + 3]

            site_probs = self.codon_to_aa_probabilities(
                parent_codon, codon_muts, codon_subs
            )
            prob_matrix.append(site_probs)

        return np.array(prob_matrix).transpose()
