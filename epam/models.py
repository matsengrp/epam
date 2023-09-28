from abc import ABC, abstractmethod
import ablang
import shmple
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import h5py
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
from epam.sequences import translate_sequences, NT_STR_SORTED, AA_STR_SORTED, CODONS
import epam.utils as utils


class BaseModel(ABC):
    @property
    def model_name(self):
        return self.modelname

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
            [prob_arr[AA_STR_SORTED.index(aa), i] for i, aa in enumerate(child_seq)]
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
            outfile.attrs["model_name"] = self.modelname

            for i, row in pcp_df.iterrows():
                parent = row["parent"]
                child = row["child"]
                matrix = self.prob_matrix_of_parent_child_pair(parent, child)

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
    def __init__(self, chain="heavy", modelname="AbLang_heavy"):
        """
        Initialize AbLang model with specified chain and create amino acid string.

        Parameters:
        chain (str): Name of the chain, default is "heavy".
        modelname (str): Name of the model, default is "AbLang_heavy".

        """
        self.model = ablang.pretrained(chain)
        self.model.freeze()
        self.modelname = modelname
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
        parent_aa = translate_sequences([parent])[0]
        return self.probability_array_of_seq(parent_aa)


class SHMple(BaseModel):
    def __init__(self, weights_directory, modelname="SHMple"):
        """
        Initialize a SHMple model with specified directory to trained model weights.

        Parameters:
        weights_directory (str): directory path to trained model weights.
        modelname (str): Name of the model, default is "SHMple".

        """
        self.model = shmple.AttentionModel(weights_dir=weights_directory)
        self.modelname = modelname

    def codon_to_aa_probabilities(self, parent_codon, mut_probs, sub_probs):
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
        aa_probs = {}
        for aa in AA_STR_SORTED:
            aa_probs[aa] = 0

        # iterate through all possible child codons
        for child_codon in CODONS:
            try:
                aa = translate_sequences([child_codon])[0]
            except ValueError:  # check for STOP codon
                continue

            # iterate through codon sites and compute total probability of potential child codon
            child_prob = 1.0
            for isite in range(3):
                if parent_codon[isite] == child_codon[isite]:
                    child_prob *= 1.0 - mut_probs[isite]
                else:
                    child_prob *= mut_probs[isite]
                    child_prob *= sub_probs[isite][
                        NT_STR_SORTED.index(child_codon[isite])
                    ]

            aa_probs[aa] += child_prob

        # need renormalization factor so that amino acid probabilities sum to 1,
        # since probabilities to STOP codon are dropped
        psum = np.sum([aa_probs[aa] for aa in aa_probs.keys()])

        return [aa_probs[aa] / psum for aa in AA_STR_SORTED]

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
        [rates], [subs] = self.model.predict_mutabilities_and_substitutions(
            [parent], [branch_length]
        )

        # This `mut_probs` is the probability of at least one mutation at each site.
        # So here we are interpreting the probability in the correctly-specified way rather than the mis-specified
        # way. This is helpful because we'd like normalized probabilities.
        mut_probs = 1.0 - np.exp(-rates)

        # keep track of probabilities as a row per amino acid site, then take transpose before returning output
        prob_matrix = []

        for i in range(0, len(parent), 3):
            parent_codon = parent[i : i + 3]
            codon_muts = mut_probs[i : i + 3].squeeze()
            codon_subs = subs[i : i + 3]

            site_probs = self.codon_to_aa_probabilities(
                parent_codon, codon_muts, codon_subs
            )
            prob_matrix.append(site_probs)

        return np.array(prob_matrix).transpose()


class ESM1v(BaseModel):
    def __init__(self, modelname="ESM1v_1"):
        """
        Initialize ESM1v ensemble model; currently using #1 of 5.

        Parameters:
        modelname (str): Name of the model, default is "ESM1v_1".

        """
        if torch.backends.cudnn.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Metal Performance Shaders")
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model, self.alphabet = pretrained.load_model_and_alphabet("esm1v_t33_650M_UR90S_1")
        self.model.eval()
        self.model = self.model.to(self.device)
        self.modelname = modelname

    def prob_matrix_of_parent_child_pair(self, parent, child=None) -> np.ndarray:
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

        data = [
            ("protein1", parent),
        ]

        batch_tokens = batch_converter(data)[2]

        with torch.no_grad():
            batch_tokens = batch_tokens.to(self.device)
            token_probs_pre_softmax = self.model(batch_tokens)["logits"]

        aa_idxs = [self.alphabet.get_idx(aa) for aa in AA_STR_SORTED]

        aa_probs = torch.softmax(token_probs_pre_softmax[..., aa_idxs], dim=-1)

        # check aa_probs type, add AA label if necessary

        # drop 1st and last token, assert correct length
        
        return aa_probs
