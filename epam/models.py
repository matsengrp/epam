import ablang
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt


class AbLang:
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
