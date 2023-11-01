import numpy as np
from epam.sequences import translate_sequence, NT_STR_SORTED, STOP_CODONS


def parent_position_is_hydrophobic(aa_str, pos, new_aa):
    hydrophobic_aas = set(list("AILMFWV"))
    return aa_str[pos] in hydrophobic_aas


class MutSelSimulator:
    def __init__(self, accept_mut, shmple_model):
        """
        Initializes the MutSelSimulator with a function for accepting mutations
        and a model for predicting mutabilities and substitutions.

        Parameters
        ----------
        accept_mut : callable
            A function that determines whether a proposed mutation should be accepted.
            It takes three parameters:
            - parent_aa_sequence (str): The amino acid sequence of the parent.
            - aa_site (int): The amino acid site where the mutation occurs.
            - new_aa (str): The proposed new amino acid at the site.
            The function should return a boolean indicating whether the mutation
            should be accepted. This function can be random.

        shmple_model : object
            An instance of the AttentionModel from the shmple package, used for predicting
            mutabilities and substitutions.
        """
        self.accept_mut = accept_mut
        self.shmple_model = shmple_model

    def apply_mutation(self, dna_sequence, nt_position, new_nt):
        return dna_sequence[:nt_position] + new_nt + dna_sequence[nt_position + 1 :]

    def simulate_child_sequence(
        self, parent_dna_sequence, target_mut_count, max_tries=10000
    ):
        """
        Simulates a child DNA sequence from a given parent sequence based on a
        mutation-selection process. The method uses the `shmple_model` to compute
        the initial mutabilities for the parent sequence and uses them throughout
        the simulation.

        Parameters
        ----------
        parent_dna_sequence : str
            The parent DNA sequence. Must be a string of nucleotides (A, C, G, T)
            with length divisible by 3. Should not contain stop codons.
        target_mut_count : int
            The target number of mutations to introduce in the child sequence.
        max_tries : int
            The maximum number of attempts to generate a mutation. If this number is
            exceeded, the method will return the current sequence.

        Returns
        -------
        str
            The simulated child DNA sequence.

        Raises
        ------
        ValueError
            If the length of the parent sequence is not divisible by 3.

        Notes
        -----
        - The mutabilities for the parent sequence are computed once at the beginning
        using the `shmple_model` and are used for the rest of the simulation.
        - Mutations are drawn based on these initial mutabilities and are subject to
        the `accept_mut` function to decide if they should be accepted.
        - The method aims to generate exactly `target_mut_count` mutations, avoiding
        duplicates at the same nucleotide site.
        - Mutations to stop are discarded.
        """

        if len(parent_dna_sequence) % 3 != 0:
            raise ValueError("Parent sequence length must be divisible by 3")

        stop_codons = set(STOP_CODONS)
        mutated_sites = set()
        branch_length = target_mut_count / len(parent_dna_sequence)

        [rates], [subs] = self.shmple_model.predict_mutabilities_and_substitutions(
            [parent_dna_sequence], [branch_length]
        )

        rates = rates.squeeze()
        subs = subs.squeeze()
        mutation_prob = 1 - np.exp(-np.array(rates))

        running_dna_sequence = parent_dna_sequence
        running_aa_sequence = translate_sequence(running_dna_sequence)

        try_count = 0

        while len(mutated_sites) < target_mut_count:
            nt_site = np.random.choice(
                len(mutation_prob), p=mutation_prob / sum(mutation_prob)
            )

            if try_count > max_tries:
                return running_dna_sequence

            try_count += 1

            # Skip if this site has already been mutated
            if nt_site in mutated_sites:
                continue

            new_nt = np.random.choice(list(NT_STR_SORTED), p=subs[nt_site, :])

            # Discard if the new nt is the same as the original
            if new_nt == running_dna_sequence[nt_site]:
                continue

            aa_site = nt_site // 3
            proposed_dna_sequence = self.apply_mutation(
                running_dna_sequence, nt_site, new_nt
            )

            # Reject stop codons.
            if proposed_dna_sequence[aa_site * 3 : (aa_site + 1) * 3] in stop_codons:
                continue

            new_aa = translate_sequence(proposed_dna_sequence)[aa_site]

            if self.accept_mut(running_aa_sequence, aa_site, new_aa):
                running_dna_sequence = proposed_dna_sequence
                running_aa_sequence = translate_sequence(running_dna_sequence)
                mutated_sites.add(nt_site)

        return running_dna_sequence
