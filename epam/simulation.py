import pandas as pd
import random

from epam.sequences import AA_STR_SORTED


def mimic_aa_mutations(sequence_mutator_fn, aa_parents, aa_sub_counts):
    """
    Mimics amino acid mutations for a series of parent sequences.

    Parameters
    ----------
    sequence_mutator_fn : function
        Function that takes an amino acid sequence and an integer, returns mutated sequence with that many mutations.
    aa_parents : pd.Series
        Series containing parent amino acid sequences as strings.
    aa_sub_counts : pd.Series
        Series containing the number of substitutions for each parent sequence.

    Returns
    -------
    pd.Series
        Series containing mutated sequences as strings.
    """

    mutated_sequences = []

    for aa_seq, sub_count in zip(aa_parents, aa_sub_counts):
        mutated_seq = sequence_mutator_fn(aa_seq, sub_count)
        mutated_sequences.append(mutated_seq)

    return pd.Series(mutated_sequences)


def general_mutator(aa_seq, sub_count, mut_criterion):
    """
    General function to mutate an amino acid sequence based on a criterion function.
    The function first identifies positions in the sequence that satisfy the criterion
    specified by `mut_criterion`. If the number of such positions is less than or equal
    to the number of mutations needed (`sub_count`), then mutations are made at those positions.
    If `sub_count` is greater than the number of positions satisfying the criterion, the function
    mutates all those positions and then randomly selects additional positions to reach `sub_count`
    total mutations. All mutations change the amino acid to a randomly selected new amino acid,
    avoiding a mutation to the same type.

    Parameters
    ----------
    aa_seq : str
        Original amino acid sequence.
    sub_count : int
        Number of substitutions to make.
    mut_criterion : function
        Function that takes a sequence and a position, returns True if position should be mutated.

    Returns
    -------
    str
        Mutated amino acid sequence.
    """

    def draw_new_aa_for_pos(pos):
        return random.choice([aa for aa in AA_STR_SORTED if aa != aa_seq_list[pos]])

    aa_seq_list = list(aa_seq)

    # find all positions that satisfy the mutation criterion
    mut_positions = [
        pos for pos, aa in enumerate(aa_seq_list) if mut_criterion(aa_seq, pos)
    ]

    # if fewer criterion-satisfying positions than required mutations, randomly add more
    if len(mut_positions) < sub_count:
        extra_positions = random.choices(
            [pos for pos in range(len(aa_seq_list)) if pos not in mut_positions],
            k=sub_count - len(mut_positions),
        )
        mut_positions += extra_positions

    # if more criterion-satisfying positions than required mutations, randomly remove some
    elif len(mut_positions) > sub_count:
        mut_positions = random.sample(mut_positions, sub_count)

    # perform mutations
    for pos in mut_positions:
        aa_seq_list[pos] = draw_new_aa_for_pos(pos)

    return "".join(aa_seq_list)


# Criterion functions
def tyrosine_mut_criterion(aa_seq, pos):
    return aa_seq[pos] == "Y"


def hydrophobic_mut_criterion(aa_seq, pos):
    hydrophobic_aa = set("AILMFVWY")  
    return aa_seq[pos] in hydrophobic_aa


[tyrosine_mutator, hydrophobic_mutator] = [
    lambda aa_seq, sub_count, crit=crit: general_mutator(aa_seq, sub_count, crit)
    for crit in [tyrosine_mut_criterion, hydrophobic_mut_criterion]
]
