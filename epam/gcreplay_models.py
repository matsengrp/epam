from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
import epam.models as models
import epam.molevol as molevol
import epam.sequences as sequences
import epam.utils

from epam.torch_common import optimize_branch_length


class GCReplayDMS(models.BaseModel):
    def __init__(self, dms_data_file: str, chain="heavy", model_name=None, scaling=1.0):
        """
        Initialize a selection model from GCReplay DMS data.

        Parameters:
        dms_data_file (str): File path to the DMS measurements data.
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        scaling (float): multiplicative factor on the parent-child binding difference.
        """
        super().__init__(model_name=model_name)
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
                wt_bind = site_df[site_df["wildtype"] == site_df["mutant"]]["bind_CGG"].item()
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
        ref_bind = site_dms_df[site_dms_df["mutant"] == parent_aa[site]]["bind_CGG"].item()
        assert(~np.isnan(ref_bind))

        # log(10) because binding is log10[K_A]
        return np.exp(site_dms_df["bind_CGG"].to_numpy() * np.log(10)) / np.exp(ref_bind * np.log(10))  

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
            assert(True not in np.isnan(dms_ratios))
            sel_factors = np.power(dms_ratios, self.scaling)

            # DMS data lists amino acid mutants in alphabetical order (convenient!)
            # Note: sel_factors results is float64, but seems like einsum wants float32
            #       (see: build_codon_mutsel in molevol.py)
            matrix.append(sel_factors.astype(np.float32))

        return np.array(matrix)


class GCReplayDMSSigmoid(GCReplayDMS):
    def __init__(self, dms_data_file: str, chain="heavy", model_name=None, scaling=1.0):
        """
        Initialize a selection model from GCReplay DMS data that feed into a sigmoid function

        Parameters:
        dms_data_file (str): File path to the DMS measurements data.
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        scaling (float): multiplicative factor on the parent-child binding difference.
        """
        super().__init__(
            dms_data_file=dms_data_file,
            chain=chain,
            model_name=model_name,
            scaling=scaling,
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
            assert(True not in np.isnan(dms_ratios))
            sel_factors = epam.utils.selection_factor_ratios_to_sigmoid(
                torch.tensor(dms_ratios), scale_const=self.scaling
            )

            # DMS data lists amino acid mutants in alphabetical order (convenient!)
            # Note: sel_factors results is float64, but seems like einsum wants float32
            #       (see: build_codon_mutsel in molevol.py)
            matrix.append(sel_factors.numpy().astype(np.float32))

        return np.array(matrix)


class GCReplaySHM(models.BaseModel):
    def __init__(self, shm_data_file: str, model_name=None):
        """
        Initialize a neutral mutation model from GCReplay passenger mouse data.

        Parameters:
        shm_data_file (str): File path to the mutation rates from passenger mouse data.
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        """
        super().__init__(model_name=model_name)
        shm_df = pd.read_csv(shm_data_file)
        cols = list("ACGT")

        # remove the last row because (?) it's not part of the sequence, also not multiple of 3 so truncated anyway
        shm_df = shm_df[cols].drop(shm_df.index[-1], axis=0, inplace=False)

        self.mut_probs = shm_df[cols].sum(axis=1).to_numpy()  # mutability probabilities
        self.sub_probs = (
            shm_df[cols].div(self.mut_probs, axis=0).to_numpy()
        )  # substitution probabilities given mutation has occurred

    def aaprobs_of_parent_child_pair(self, parent, child=None) -> np.ndarray:
        """
        Generate a numpy array of the normalized probability of the various amino acids by site according to DMS measurements.

        The rows of the array correspond to the amino acids sorted alphabetically.

        Parameters:
        parent (str): The parent nucleotide sequence for which we want the array of probabilities.
        child (str): The child nucleotide sequence. (ignored)

        Returns:
        numpy.ndarray: A 2D array containing the normalized probabilities of the amino acids by site.

        """
        parent_idxs = sequences.nt_idx_tensor_of_str(parent)

        # Reshape the inputs to include a codon dimension.
        parent_codon_idxs = molevol.reshape_for_codons(parent_idxs)
        codon_mut_probs = molevol.reshape_for_codons(
            torch.tensor(self.mut_probs, dtype=torch.float)
        )
        codon_sub_probs = molevol.reshape_for_codons(
            torch.tensor(self.sub_probs, dtype=torch.float)
        )

        # Vectorized calculation of amino acid probabilities.
        return molevol.aaprob_of_mut_and_sub(
            parent_codon_idxs, codon_mut_probs, codon_sub_probs
        ).numpy()


class GCReplayOptSHM(GCReplaySHM):
    def __init__(
        self,
        shm_data_file,
        model_name=None,
        max_optimization_steps=1000,
        optimization_tol=1e-4,
        learning_rate=0.1,
    ):
        """
        Initialize a GCReplaySHM model that optimizes branch length for each parent-child pair.

        Parameters:
        shm_data_file : str
            File path to the mutation rates from passenger mouse data.
        model_name : str, optional
            Model name. Default is None, setting the model name to the class name.
        max_optimization_steps : int, optional
            Maximum number of gradient descent steps. Default is 1000.
        optimization_tol : float, optional
            Tolerance for optimization of log(branch length). Default is 1e-4.
        learning_rate : float, optional
            Learning rate for torch's SGD. Default is 0.1.
        """
        super().__init__(shm_data_file=shm_data_file, model_name=model_name)
        self.max_optimization_steps = max_optimization_steps
        self.optimization_tol = optimization_tol
        self.learning_rate = learning_rate

    def _build_log_pcp_probability(
        self, parent: str, child: str, rates: torch.Tensor, sub_probs: torch.Tensor
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
        torch.Tensor: The optimal branch length.
        """
        # Passenger mouse analysis gives mutation probabilities;
        # derive the Poisson rates (corresponding to branch length of 1).
        rates = torch.tensor(-np.log(1 - self.mut_probs), dtype=torch.float)
        sub_probs = torch.tensor(self.sub_probs, dtype=torch.float)
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
        rates = torch.tensor(-np.log(1 - self.mut_probs), dtype=torch.float)
        sub_probs = torch.tensor(self.sub_probs, dtype=torch.float)
        parent_idxs = sequences.nt_idx_tensor_of_str(parent)
        return molevol.aaprobs_of_parent_scaled_rates_and_sub_probs(
            parent_idxs, rates * branch_length, sub_probs
        )

    def aaprobs_of_parent_child_pair(self, parent, child) -> np.ndarray:
        base_branch_length = 1
        branch_length = self._find_optimal_branch_length(
            parent, child, base_branch_length
        )

        # if branch_length > 0.5:
        #     print(f"Warning: branch length of {branch_length} is surprisingly large.")
        return self._aaprobs_of_parent_and_branch_length(parent, branch_length).numpy()


class GCReplayMutSel(GCReplayOptSHM):
    """
    A mutation-selection model using passenger mouse data for the mutation part.

    Note that stop codons are assumed to have zero selection probability.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def build_selection_matrix_from_parent(self, parent: str) -> torch.Tensor:
        """Build the selection matrix (i.e. F matrix) from a parent nucleotide
        sequence.

        The shape of this numpy array should be (len(parent) // 3, 20).
        """
        pass

    def _build_log_pcp_probability(
        self, parent: str, child: str, rates: torch.Tensor, sub_probs: torch.Tensor
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

            reshaped_child_idxs = child_idxs.reshape(-1, 3)
            child_prob_vector = codon_mutsel[
                torch.arange(len(reshaped_child_idxs)),
                reshaped_child_idxs[:, 0],
                reshaped_child_idxs[:, 1],
                reshaped_child_idxs[:, 2],
            ]

            result = torch.sum(torch.log(child_prob_vector))

            assert not torch.isnan(result)

            return result

        return log_pcp_probability

    def _aaprobs_of_parent_and_branch_length(
        self, parent, branch_length
    ) -> torch.Tensor:
        rates = torch.tensor(-np.log(1 - self.mut_probs), dtype=torch.float)
        sub_probs = torch.tensor(self.sub_probs, dtype=torch.float)

        sel_matrix = self.build_selection_matrix_from_parent(parent)
        mut_probs = 1.0 - torch.exp(-branch_length * rates)

        parent_idxs = sequences.nt_idx_tensor_of_str(parent)

        codon_mutsel, sums_too_big = molevol.build_codon_mutsel(
            parent_idxs.reshape(-1, 3),
            mut_probs.reshape(-1, 3),
            sub_probs.reshape(-1, 3, 4),
            sel_matrix,
        )

        return molevol.aaprobs_of_codon_probs(codon_mutsel)


class GCReplayOptSHMDMSSigmoid(GCReplayMutSel):
    def __init__(
        self,
        shm_data_file: str,
        dms_data_file: str,
        chain="heavy",
        model_name=None,
        scaling=1.0,
    ):
        """
        Initialize a mutation-selection model from GCReplay passenger mouse and DMS data with sigmoid function.
        Branch optimization is performed.

        Parameters:
        shm_data_file (str): File path to the mutation rates from passenger mouse data.
        dms_data_file (str): File path to the DMS measurements data.
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        scaling (float): multiplicative factor on the parent-child binding difference.
        """
        super().__init__(shm_data_file, model_name)
        self.selection_model = GCReplayDMSSigmoid(
            dms_data_file, chain, model_name, scaling
        )

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))


class GCReplaySHMpleDMS(models.MutSel):
    def __init__(
        self,
        weights_directory: str,
        dms_data_file: str,
        chain="heavy",
        model_name=None,
        scaling=1.0,
    ):
        """
        Initialize a mutation-selection model for GC-Replay data using SHMple for the mutation part and
        DMS measurements with sigmoid function for the selection part.
        Branch optimization is performed.

        Parameters:
        weights_directory (str): Directory path to trained SHMple model weights.
        dms_data_file (str): File path to the DMS measurements data.
        chain (str): Name of the chain, default is "heavy".
        model_name (str, optional): The name of the model. If not specified, the class name is used.
        """
        super().__init__(weights_directory, model_name)
        self.selection_model = GCReplayDMSSigmoid(
            dms_data_file, chain, model_name, scaling
        )

    def build_selection_matrix_from_parent(self, parent):
        return torch.tensor(self.selection_model.aaprobs_of_parent_child_pair(parent))
