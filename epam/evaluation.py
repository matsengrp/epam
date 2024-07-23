"""Code for evaluating model performance."""

import h5py
import pandas as pd
import numpy as np
import bisect
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.annotate_pcps import get_cdr_fwk_seqs
from netam.common import SMALL_PROB
from netam.sequences import (
    AA_STR_SORTED,
    translate_sequences,
    translate_sequence,
    generic_mutation_frequency,
)
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def evaluate(aaprob_paths, model_performance_path):
    """
    Wrapper function for evaluate_dataset() that takes in a list of aaprob matrices and outputs a CSV of model performance metrics.
    Outputs to CSV file with columns for the different metrics and a row per model/data set combo.

    Parameters:
    aaprob_paths (list): List of paths to evaluate. Each aaprob matrix corresponds to predictions for one model on a given data set.
    model_performance_path (str): Path to output for model performance metrics.

    """
    model_performances = [evaluate_dataset(aaprob_path) for aaprob_path in aaprob_paths]

    all_model_performances = pd.DataFrame(model_performances)

    all_model_performances.to_csv(model_performance_path, index=False)


def evaluate_dataset(aaprob_path):
    """
    Evaluate model predictions against reality for a set of parent-child pairs (PCPs).
    Function is model-agnositic and currently calculates substitution accuracy, r-precision, and cross entropy loss.
    All metrics are reported for the full sequence, as well as for the framework and CDR regions separately.
    Returns evaluation metrics for a single aaprob matrix (generated from one model on one data set).

    Parameters:
    aaprob_path (str): Path to aaprob matrix for parent-child pairs.

    Returns:
    model_performance (dict): Dictionary of model performance metrics for a single aaprob matrix.

    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)

    pcp_df = load_and_filter_pcp_df(pcp_path)

    pcp_df["parent_aa"] = pcp_df.apply(
        lambda row: translate_sequence(row["parent"]), axis=1
    )
    pcp_df["child_aa"] = pcp_df.apply(
        lambda row: translate_sequence(row["child"]), axis=1
    )
    (
        pcp_df["parent_fwk_seq"],
        pcp_df["parent_cdr_seq"],
        pcp_df["child_fwk_seq"],
        pcp_df["child_cdr_seq"],
    ) = zip(*pcp_df.apply(get_cdr_fwk_seqs, axis=1))

    region_parent_aa_seqs = {}
    region_child_aa_seqs = {}
    region_sub_locations = {}
    region_sub_aa_ids = {}
    region_k_subs = {}

    for region in ["full", "fwk", "cdr"]:
        if region == "full":
            region_parent_aa_seqs[region] = pcp_df["parent_aa"].to_numpy()
            region_child_aa_seqs[region] = pcp_df["child_aa"].to_numpy()
        else:
            region_parent_aa_seqs[region] = pcp_df[f"parent_{region}_seq"].to_numpy()
            region_child_aa_seqs[region] = pcp_df[f"child_{region}_seq"].to_numpy()

        region_sub_locations[region] = [
            locate_child_substitutions(parent, child)
            for parent, child in zip(
                region_parent_aa_seqs[region], region_child_aa_seqs[region]
            )
        ]

        region_sub_aa_ids[region] = [
            identify_child_substitutions(parent, child)
            for parent, child in zip(
                region_parent_aa_seqs[region], region_child_aa_seqs[region]
            )
        ]

        # k represents the number of substitutions observed in each PCP, top k substitutions will be evaluated for r-precision
        region_k_subs[region] = [
            len(sub_location) for sub_location in region_sub_locations[region]
        ]

    region_site_sub_probs = {}
    region_model_sub_aa_ids = {}

    with h5py.File(aaprob_path, "r") as matfile:
        model_name = matfile.attrs["model_name"]
        for region in ["full", "fwk", "cdr"]:
            region_site_sub_probs[region] = []
            region_model_sub_aa_ids[region] = []

            for index in range(len(region_parent_aa_seqs[region])):
                pcp_index = pcp_df.index[index]
                grp = matfile[
                    "matrix" + str(pcp_index)
                ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
                matrix = grp["data"]

                region_site_sub_probs[region].append(
                    calculate_site_substitution_probabilities(
                        matrix, region_parent_aa_seqs[region][index]
                    )
                )

                def find_highest_ranked_substitutions(matrix, parent, child):
                    return [
                        highest_ranked_substitution(matrix[j, :], parent, j)
                        for j in range(len(parent))
                        if parent[j] != child[j]
                    ]

                region_model_sub_aa_ids[region].append(
                    find_highest_ranked_substitutions(
                        matrix,
                        region_parent_aa_seqs[region][index],
                        region_child_aa_seqs[region][index],
                    )
                )

    region_top_k_sub_locations = {}
    region_sub_acc = {}
    region_r_prec = {}
    region_cross_ent = {}
    region_aa_sub_freq = {}

    for region in ["full", "fwk", "cdr"]:
        region_top_k_sub_locations[region] = [
            locate_top_k_substitutions(region_site_sub_prob, k_sub)
            for region_site_sub_prob, k_sub in zip(
                region_site_sub_probs[region], region_k_subs[region]
            )
        ]

        region_sub_acc[region] = calculate_sub_accuracy(
            region_sub_aa_ids[region],
            region_model_sub_aa_ids[region],
            region_k_subs[region],
        )
        region_r_prec[region] = calculate_r_precision(
            region_sub_locations[region],
            region_top_k_sub_locations[region],
            region_k_subs[region],
        )
        region_cross_ent[region] = calculate_cross_entropy_loss(
            region_sub_locations[region], region_site_sub_probs[region]
        )

        if region == "full":
            region_aa_sub_freq[region] = [
                generic_mutation_frequency("X", parent, child)
                for parent, child in zip(
                    region_parent_aa_seqs[region], region_child_aa_seqs[region]
                )
            ]
        else:
            parent_only_aa_seqs = [
                seq.replace("-", "") for seq in region_parent_aa_seqs[region]
            ]
            child_only_aa_seqs = [
                seq.replace("-", "") for seq in region_child_aa_seqs[region]
            ]
            region_aa_sub_freq[region] = [
                calculate_aa_substitution_frequencies_by_region(parent, child)
                for parent, child in zip(parent_only_aa_seqs, child_only_aa_seqs)
            ]

    model_performance = {
        "data_set": pcp_path,
        "pcp_count": len(pcp_df),
        "model": model_name,
        "sub_accuracy": region_sub_acc["full"],
        "r_precision": region_r_prec["full"],
        "cross_entropy": region_cross_ent["full"],
        "fwk_sub_accuracy": region_sub_acc["fwk"],
        "fwk_r_precision": region_r_prec["fwk"],
        "fwk_cross_entropy": region_cross_ent["fwk"],
        "cdr_sub_accuracy": region_sub_acc["cdr"],
        "cdr_r_precision": region_r_prec["cdr"],
        "cdr_cross_entropy": region_cross_ent["cdr"],
        "avg_k_subs": np.mean(region_k_subs["full"]),
        "avg_aa_sub_freq": np.mean(region_aa_sub_freq["full"]),
        "aa_sub_freq_range": (
            np.min(region_aa_sub_freq["full"]),
            np.max(region_aa_sub_freq["full"]),
        ),
        "fwk_avg_k_subs": np.mean(region_k_subs["fwk"]),
        "fwk_avg_aa_sub_freq": np.mean(region_aa_sub_freq["fwk"]),
        "fwk_aa_sub_freq_range": (
            np.min(region_aa_sub_freq["fwk"]),
            np.max(region_aa_sub_freq["fwk"]),
        ),
        "cdr_avg_k_subs": np.mean(region_k_subs["cdr"]),
        "cdr_avg_aa_sub_freq": np.mean(region_aa_sub_freq["cdr"]),
        "cdr_aa_sub_freq_range": (
            np.min(region_aa_sub_freq["cdr"]),
            np.max(region_aa_sub_freq["cdr"]),
        ),
    }

    return model_performance


def locate_child_substitutions(parent_aa, child_aa):
    """
    Return the location of the amino acid substitutions for a given parent-child pair.

    Parameters:
    parent_aa (str): Amino acid sequence of parent.
    child_aa (str): Amino acid sequence of child.

    Returns:
    child_sub_sites (np.array): Location of substitutions in parent-child pair.

    """
    child_sub_sites = [i for i in range(len(parent_aa)) if parent_aa[i] != child_aa[i]]

    child_sub_sites = np.array(child_sub_sites)

    return child_sub_sites


def identify_child_substitutions(parent_aa, child_aa):
    """
    Return the identity of the amino acid substitutions for a given parent-child pair.

    Parameters:
    parent_aa (str): Amino acid sequence of parent.
    child_aa (str): Amino acid sequence of child.

    Returns:
    child_aa_subs (np.array): Amino acid substitutions in parent-child pair.

    """
    child_aa_subs = [
        child_aa[i] for i in range(len(parent_aa)) if parent_aa[i] != child_aa[i]
    ]

    child_aa_subs = np.array(child_aa_subs)

    return child_aa_subs


def calculate_site_substitution_probabilities(aaprobs, parent_aa):
    """
    Calculate the probability of substitution at each site for a parent sequence.

    Parameters:
    aaprobs (np.ndarray): A 2D array containing the normalized probabilities of the amino acids by site for a parent sequence.
    parent_aa (str): Amino acid sequence of parent.

    Returns:
    site_sub_probs (np.array): 1D array containing probability of substitution at each site for a parent sequence.

    """
    site_sub_probs = []

    for i in range(len(parent_aa)):
        # assign 0 probability of substitution to sites outside region of interest in CDR and FWK sequences
        if parent_aa[i] == "-":
            site_sub_probs.append(0.0)
        else:
            sub_prob = np.sum(
                aaprobs[i, :][
                    [
                        AA_STR_SORTED.index(aa)
                        for aa in AA_STR_SORTED
                        if aa != parent_aa[i]
                    ]
                ]
            )
            site_sub_probs.append(sub_prob)

    site_sub_probs = np.array(site_sub_probs)

    assert site_sub_probs.size == len(
        parent_aa
    ), "The number of substitution probabilities does not match the number of amino acid sites."

    return site_sub_probs


def highest_ranked_substitution(matrix_i, parent_aa, i):
    """
    Return the highest ranked substitution for site i in a given parent-child pair.

    Parameters:
    matrix_i (np.array): aaprob matrix for parent-child pair at aa site i.
    parent_aa (str): Parent amino acid sequence.
    i (int): Index of amino acid site substituted.

    Returns:
    pred_aa_sub (str): Predicted amino acid substitution (most likely non-parent aa).

    """
    prob_sorted_aa_indices = matrix_i.argsort()[::-1]

    pred_aa_ranked = "".join((np.array(list(AA_STR_SORTED))[prob_sorted_aa_indices]))

    # skip most likely aa if it is the parent aa (enforce substitution)
    if pred_aa_ranked[0] == parent_aa[i]:
        pred_aa_sub = pred_aa_ranked[1]
    elif pred_aa_ranked[0] != parent_aa[i]:
        pred_aa_sub = pred_aa_ranked[0]

    return pred_aa_sub


def locate_top_k_substitutions(site_sub_probs, k_sub):
    """
    Return the top k substitutions predicted for a parent-child pair given precalculated site substitution probabilities.

    Parameters:
    site_sub_probs (np.array): Probability of substition at each site for a parent sequence.
    k_sub (int): Number of substitutions observed in PCP.

    Returns:
    pred_sub_sites (np.array): Location of top-k predicted substitutions by model (unordered).

    """
    if k_sub == 0:
        return []

    # np.argpartition returns indices of top k elements in unsorted order
    pred_sub_sites = np.argpartition(site_sub_probs, -k_sub)[-k_sub:]

    assert (
        pred_sub_sites.size == k_sub
    ), "The number of predicted substitution sites does not match the number of actual substitution sites."

    return pred_sub_sites


def calculate_sub_accuracy(pcp_sub_aa_ids, model_sub_aa_ids, k_subs):
    """
    Calculate substitution accuracy for all PCPs in one data set/HDF5 file.
    Returns substitution accuracy score for use in evaluate() and output files.

    Parameters:
    pcp_aa_sub_ids (list of np.array): Amino acid substitutions in each PCP.
    model_sub_aa_ids (list of np.array): Amino acid substitutions predicted by model at substituted sites in each PCP.
    k_subs (list): Number of substitutions observed in each PCP.

    Returns:
    sub_accuracy (float): Calculated substitution accuracy for data set of PCPs.

    """
    num_sub_correct = [
        np.sum(pcp_sub_aa_ids[i] == model_sub_aa_ids[i])
        for i in range(len(model_sub_aa_ids))
    ]

    sub_accuracy = sum(num_sub_correct) / sum(k_subs)

    return sub_accuracy


def calculate_r_precision(pcp_sub_locations, top_k_sub_locations, k_subs):
    """
    Calculate r-precision for all PCPs in one data set/HDF5 file.
    Returns r-precision score for use in evaluate() and output files.

    Parameters:
    pcp_sub_locations (list of np.array): Location of substitutions in parent-child pairs.
    top_k_sub_locations (list of np.array): Location of top-k predicted substitutions by model (unordered for each PCP).
    k_subs (list): Number of substitutions observed in each PCP.

    Returns:
    r_precision (float): Calculated r-precision for data set of PCPs.

    """
    correct_site_predictions = [
        np.intersect1d(pcp_sub_location, top_k_sub_location)
        for pcp_sub_location, top_k_sub_location in zip(
            pcp_sub_locations, top_k_sub_locations
        )
    ]

    k_subs_correct = [
        len(correct_site_prediction)
        for correct_site_prediction in correct_site_predictions
    ]

    pcp_r_precision = [
        k_correct / k_total
        for k_correct, k_total in zip(k_subs_correct, k_subs)
        if k_total > 0
    ]

    r_precision = sum(pcp_r_precision) / len(pcp_r_precision)

    return r_precision


def calculate_cross_entropy_loss(pcp_sub_locations, site_sub_probs):
    """
    Calculate cross entropy loss for all PCPs in one data set/HDF5 file.

    Parameters:
    pcp_sub_locations (list of np.array): Location of substitutions in parent-child pairs.
    site_sub_probs (list of np.array): Probability of substition at each site for parent sequences.

    Returns:
    cross_entropy_loss (float): Calculated cross entropy loss for data set of PCPs.

    """
    log_probs_substitution = []
    for i in range(len(site_sub_probs)):
        if pcp_sub_locations[i].size > 0:
            assert any(
                pcp_sub_locations[i][j] < len(site_sub_probs[i])
                for j in range(len(pcp_sub_locations[i]))
            ), "The location of a substitution is greater than the number of sites in the parent sequence."
        for idx, p_i in np.ndenumerate(site_sub_probs[i]):
            if idx in pcp_sub_locations[i]:
                log_probs_substitution.append(np.log(p_i if p_i != 0 else SMALL_PROB))
            else:
                log_probs_substitution.append(
                    np.log(1 - p_i) if p_i != 1 else np.log(1 - SMALL_PROB)
                )

    cross_entropy_loss = (
        -1 / len(log_probs_substitution) * np.sum(log_probs_substitution)
    )

    return cross_entropy_loss


def calculate_aa_substitution_frequencies_by_region(parent_aa, child_aa):
    """
    Calculate the fraction of sites that differ between the parent and child FWK or CDR sequences.

    Parameters:
    parent_aa (str): Amino acid sequence of parent. FWK sequences will have CDR sites masked with '-' and vice versa.
    child_aa (str): Amino acid sequence of child.

    Returns:
    aa_sub_frequency (float): Fraction of sites that differ between the parent and child FWK or CDR sequences.

    """
    parent = parent_aa.replace("-", "")
    child = child_aa.replace("-", "")

    assert len(parent) == len(
        child
    ), "Parent and child FWK/CDR sequences must be the same length."

    aa_sub_frequency = sum(1 for p, c in zip(parent, child) if p != c) / len(parent)

    return aa_sub_frequency


def get_site_mutabilities_df(
    aaprob_path,
    anarci_path=None,
    collapse=True,
    verbose=False,
):
    """
    Computes the amino acid site mutability probabilities
    for every site of every parent in a dataset.
    Returns a dataframe that annotates for each site
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    whether a mutation is observed in the child sequence,
    and whether the site is in a CDR.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.
    anarci_path (str): path to ANARCI output for sequence numbering.
                       If None, the (0-based) index along the sequence string is used.
    collapse (bool): whether sites with lettered suffix should be consolidated into one site
                     (e.g. 111A, 111B, etc. will counted as site 111).
                     Ignored if anarci_path is None.
    verbose (bool): whether to print (sample ID, family ID) info when ANARCI output has sequence length mismatch.
                    Ignored if anarci_path is None.

    Returns:
    output_df (pd.DataFrame): dataframe with columns pcp_index, site, prob, mutation, is_cdr.
    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    numbering_dict = get_numbering_dict(anarci_path, pcp_df, collapse, verbose)

    pcp_index_col = []
    sites_col = []
    site_sub_probs = []
    site_sub_flags = []
    is_cdr_col = []
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            parent = parent_aa_seqs[index]
            child = child_aa_seqs[index]

            if anarci_path is None:
                sites_col.append(np.arange(len(parent)))
            else:
                nbkey = tuple(pcp_df.loc[pcp_index][["sample_id", "family"]])
                if nbkey in numbering_dict:
                    sites_col.append(numbering_dict[nbkey])
                else:
                    continue

            pcp_index_col.append([pcp_index] * len(parent))
            site_sub_probs.append(
                calculate_site_substitution_probabilities(matrix, parent)
            )
            site_sub_flags.append([p != c for p, c in zip(parent, child)])

            cdr1 = (
                pcp_df.loc[pcp_index]["cdr1_codon_start"] // 3,
                pcp_df.loc[pcp_index]["cdr1_codon_end"] // 3,
            )
            cdr2 = (
                pcp_df.loc[pcp_index]["cdr2_codon_start"] // 3,
                pcp_df.loc[pcp_index]["cdr2_codon_end"] // 3,
            )
            cdr3 = (
                pcp_df.loc[pcp_index]["cdr3_codon_start"] // 3,
                pcp_df.loc[pcp_index]["cdr3_codon_end"] // 3,
            )
            is_cdr_col.append(
                [
                    (
                        True
                        if (i >= cdr1[0] and i <= cdr1[1])
                        or (i >= cdr2[0] and i <= cdr2[1])
                        or (i >= cdr3[0] and i <= cdr3[1])
                        else False
                    )
                    for i in range(len(parent))
                ]
            )

    output_df = pd.DataFrame(
        columns=["pcp_index", "site", "prob", "mutation", "is_cdr"]
    )
    output_df["pcp_index"] = np.concatenate(pcp_index_col)
    output_df["site"] = np.concatenate(sites_col)
    output_df["prob"] = np.concatenate(site_sub_probs)
    output_df["mutation"] = np.concatenate(site_sub_flags)
    output_df["is_cdr"] = np.concatenate(is_cdr_col)

    return output_df


def plot_observed_vs_expected(
    df,
    counts_ax,
    oe_ax,
    diff_ax,
    logprobs=True,
    binning=None,
    counts_color="#B3CDE3",
    pcurve_color="#D95F02",
    model_color="#0072B2",
    model_name="Expected",
    logy=False,
    normalize=False,
):
    """
    Draws a figure with up to 3 panels showing:
    counts of sites in bins of amino acid substitution probability,
    observed vs expected number of mutations in bins of amino acid substitution probability,
    and per bin differences between observed and expected.
    The expected number of mutations is computed as the total probability
    of the sites that fall in that mutability bin.
    The input dataframe requires two columns: 'prob'
    (site mutability -- may be at level of nucleotide, or codon, or amino acid, etc.)
    and 'mutation' (True/False whether the site has an observed mutation or not).
    Each dataframe row corresponds to a site in a specific sequence.
    Thus, the total number of rows is the total number of sites from
    all sequences in the dataset.

    Parameters:
    df (pd.DataFrame): dataframe of site mutabilities.
    counts_ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    oe_ax (fig.ax): figure axis for plotting observed vs expected number of mutations. If None, plot is not drawn.
    diff_ax (fig.ax): figure axis for ploting observed vs expected differences. If None, plot is not drawn.
    logprobs (bool): whether to plot log-probabilities (True) or plot probabilities (False).
    binning (list): list of bin boundaries (i.e. n+1 boundaries for n bins). If None, a default binning is used.
    counts_color (str): color for the counts of sites plot.
    pcurve_color (str): color for the probability curve in the counts of sites plot.
    model_color (str): color for the plot of expected number of mutations.
    model_name (str): legend label for the plot of expected number of mutations.
    logy (bool): whether to show y-axis in log-scale.
    normalize (bool): whether to scale the area of the expected mutations distribution to match the observed mutations distribution.

    Returns:
    A dictionary with results labeled:
    overlap (float): area of overlap between observed and expected, divided by the average of the two areas.
    residual (float): square root of the sum of squared bin-by-bin differences between observed and expected, divided by total expected.
    counts_twinx_ax (fig.ax): handle to the probability y-axis (right side) of the counts plot, if drawn.

    """
    model_probs = df["prob"].to_numpy()

    # set default binning if None specified
    if binning is None:
        if logprobs:
            min_logprob = 1.05 * np.log10(model_probs).min()
            binning = np.linspace(min_logprob, 0, 101)
        else:
            max_prob = min(1, 1.05 * model_probs.max())
            binning = np.linspace(0, max_prob, 101)

    # compute expectation
    bin_index_col = []
    for p in model_probs:
        if logprobs:
            index = bisect.bisect(binning, np.log10(p)) - 1
        else:
            index = bisect.bisect(binning, p) - 1
        bin_index_col.append(index)
    df["bin_index"] = bin_index_col

    expected = []
    exp_err = []
    for i in range(len(binning) - 1):
        binprobs = df[df["bin_index"] == i]["prob"].to_numpy()
        expected.append(np.sum(binprobs))
        exp_err.append(np.sqrt(np.sum(binprobs * (1 - binprobs))))
    expected = np.array(expected)
    exp_err = np.array(exp_err)

    # count observed mutations
    if logprobs:
        obs_probs = np.log10(df[df["mutation"] > 0]["prob"].to_numpy())
        xlabel = "$\log_{10}$(amino acid substitution probability)"
    else:
        obs_probs = df[df["mutation"] > 0]["prob"].to_numpy()
        xlabel = "amino acid substitution probability"
    observed = np.histogram(obs_probs, binning)[0]

    # normalize total expected to equal total observed
    if normalize == True:
        fnorm = np.sum(observed) / np.sum(expected)
        expected = fnorm * expected
        exp_err = fnorm * exp_err

    # compute overlap metric
    intersect = np.sum(np.minimum(observed, expected))
    denom = 0.5 * (np.sum(observed) + np.sum(expected))
    overlap = intersect / denom

    # compute residual metric
    diff = observed - expected
    residual = np.sqrt(np.sum(diff * diff)) / np.sum(expected)

    # midpoints of each bin
    xvals = [0.5 * (binning[i] + binning[i + 1]) for i in range(len(binning) - 1)]

    # bin widths
    binw = [(binning[i + 1] - binning[i]) for i in range(len(binning) - 1)]

    # plot site counts
    counts_twinx_ax = None
    if counts_ax is not None:
        if logprobs:
            hist_data = np.log10(model_probs)
        else:
            hist_data = model_probs
        counts_ax.hist(hist_data, bins=binning, color=counts_color)
        if (oe_ax is None) and (diff_ax is None):
            counts_ax.tick_params(axis="x", labelsize=16)
            counts_ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
        counts_ax.tick_params(axis="y", labelsize=16)
        counts_ax.set_ylabel("number of sites", fontsize=20, labelpad=10)
        counts_ax.grid()

        if logy:
            counts_ax.set_yscale("log")

        if logprobs:
            yvals = np.power(10, xvals)
        else:
            yvals = xvals

        counts_twinx_ax = counts_ax.twinx()
        counts_twinx_ax.plot(xvals, yvals, color=pcurve_color)
        counts_twinx_ax.tick_params(axis="y", labelcolor=pcurve_color, labelsize=16)
        counts_twinx_ax.set_ylabel("probability", fontsize=20, labelpad=10)
        counts_twinx_ax.set_ylim(0, 1)

    # plot observed vs expected number of mutations
    if oe_ax is not None:
        oe_ax.bar(
            xvals,
            expected,
            width=binw,
            facecolor="white",
            edgecolor=model_color,
            label=model_name,
        )
        oe_ax.plot(
            xvals,
            observed,
            marker="o",
            markersize=4,
            linewidth=0,
            color="#000000",
            label="Observed",
        )
        if diff_ax is None:
            oe_ax.tick_params(axis="x", labelsize=16)
            oe_ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
        oe_ax.tick_params(axis="y", labelsize=16)
        oe_ax.set_ylabel("number of mutations", fontsize=20, labelpad=10)

        # For some reason, regardless of draw order, legend labels are always ordered:
        #   Observed
        #   Model
        # Force reverse the order.
        leg_handles, leg_labels = oe_ax.get_legend_handles_labels()
        oe_ax.legend(leg_handles[::-1], leg_labels[::-1], fontsize=15)

        if logy:
            oe_ax.set_yscale("log")

        boxes0 = [
            Rectangle(
                (binning[ibin], expected[ibin] - exp_err[ibin]),
                binning[ibin + 1] - binning[ibin],
                2 * exp_err[ibin],
            )
            for ibin in range(len(exp_err))
        ]
        pc0 = PatchCollection(
            boxes0, facecolor="none", edgecolor="grey", linewidth=0, hatch="//////"
        )
        oe_ax.add_collection(pc0)

    # plot observed vs expected difference
    if diff_ax is not None:
        diff_ax.plot(
            xvals,
            [yo - ye for yo, ye in zip(observed, expected)],
            marker="o",
            markersize=4,
            linewidth=0,
            color="#000000",
        )
        diff_ax.axhline(y=0, color="k", linestyle="--")
        diff_ax.tick_params(axis="x", labelsize=16)
        diff_ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
        diff_ax.tick_params(axis="y", labelsize=16)
        diff_ax.set_ylabel("Obs - Exp", fontsize=20, labelpad=10)

        boxes1 = [
            Rectangle(
                (binning[ibin], -exp_err[ibin]),
                binning[ibin + 1] - binning[ibin],
                2 * exp_err[ibin],
            )
            for ibin in range(len(exp_err))
        ]
        pc1 = PatchCollection(
            boxes1, facecolor="none", edgecolor="grey", linewidth=0, hatch="//////"
        )
        diff_ax.add_collection(pc1)

    return {
        "overlap": overlap,
        "residual": residual,
        "counts_twinx_ax": counts_twinx_ax,
    }


def plot_sites_observed_vs_expected(
    df,
    ax,
    fwk_color="#0072B2",
    cdr_color="#E69F00",
    logy=False,
):
    """
    Draws a figure of observed amino acid substitutions and the top-k predictions across PCPs in a dataset.
    The input dataframe requires three columns:
    'site' (site position -- may be at the level of nucleotide, or codon, or amino acid, etc.)
    'prob' (site mutability)
    'mutation' (True/False whether the site has an observed mutation or not)
    'is_cdr' (True/False whether the site is in a CDR)
    Each dataframe row corresponds to a site in a specific sequence.
    Only sites that are True in either 'obs' or 'pred' columns are involved in the plotting and calculation.
    Hence, sites in a PCP that neither have an observed substitution nor are predicted in the top-k
    can be excluded from the dataframe.

    Parameters:
    df (pd.DataFrame): dataframe of observed and predicted sites of substitution.
    ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    fwk_color (str): color for the FWK sites.
    cdr_color (str): color for the CDR sites.
    logy (bool): whether to show y-axis in log-scale.

    Returns:
    A dictionary with results labeled:
    overlap (float): area of overlap between observed and expected, divided by the average of the two areas.
    residual (float): square root of the sum of squared bin-by-bin differences between observed and expected, divided by total expected.

    """
    if df.dtypes["site"] == "object":
        nblist = df["site"].drop_duplicates().to_list()
        xvals = sorted(
            nblist,
            key=lambda x: (
                int(x) if x.isnumeric() else int(x[:-1]),
                "" if x.isnumeric() else x[-1],
            ),
        )
        ixvals = np.arange(len(xvals))
    else:
        xvals = np.arange(np.min(df["site"]), np.max(df["site"]) + 1)
        ixvals = xvals

    expected = []
    exp_err = []
    observed = []
    fwk_expected = []
    for site in xvals:
        site_df = df[df["site"] == site]
        site_probs = site_df["prob"].to_numpy()
        expected.append(np.sum(site_probs))
        exp_err.append(np.sqrt(np.sum(site_probs * (1 - site_probs))))
        observed.append(df[(df["mutation"] == 1) & (df["site"] == site)].shape[0])
        site_fwk_probs = site_df[site_df["is_cdr"] == False]["prob"].to_numpy()
        fwk_expected.append(np.sum(site_fwk_probs))

    expected = np.array(expected)
    exp_err = np.array(exp_err)
    observed = np.array(observed)

    # compute overlap metric
    intersect = np.sum(np.minimum(observed, expected))
    denom = 0.5 * (np.sum(observed) + np.sum(expected))
    overlap = intersect / denom

    # compute residual metric
    diff = observed - expected
    residual = np.sqrt(np.sum(diff * diff)) / np.sum(expected)

    if ax is not None:
        ax.bar(
            xvals,
            expected,
            width=1,
            facecolor="white",
            edgecolor=cdr_color,
            label="Expected (CDR)",
        )

        ax.bar(
            xvals,
            fwk_expected,
            width=1,
            facecolor="white",
            edgecolor=fwk_color,
            label="Expected (FWK)",
        )

        ax.plot(
            xvals,
            observed,
            marker="o",
            markersize=4,
            linewidth=0,
            color="#000000",
            label="Observed",
        )

        if df.dtypes["site"] == "object":
            ax.tick_params(axis="x", labelsize=7, labelrotation=90)
        else:
            ax.tick_params(axis="x", labelsize=16)
        ax.set_xlabel("amino acid sequence position", fontsize=20, labelpad=10)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylabel("number of substitutions", fontsize=20, labelpad=10)

        if logy:
            ax.set_yscale("log")

        ax.legend(fontsize=15)

        boxes0 = [
            Rectangle(
                (ixvals[i] - 0.5, expected[i] - exp_err[i]),
                1,
                2 * exp_err[i],
            )
            for i in range(len(expected))
        ]
        pc0 = PatchCollection(
            boxes0, facecolor="none", edgecolor="grey", linewidth=0, hatch="//////"
        )
        ax.add_collection(pc0)

    return {
        "overlap": overlap,
        "residual": residual,
    }


def get_site_substitutions_df(
    aaprob_path,
    anarci_path=None,
    collapse=True,
    verbose=False,
):
    """
    Determines the sites of observed and predicted substitutions of every PCP in a dataset.
    Predicted substitutions are the sites in the top-k of mutability,
    where k is the number of observed substition in the PCP.
    Returns a dataframe that annotates for each site of observed and/or predicted substitution:
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    whether the site has an observed substutition,
    and whether the site is predicted to have a substitution.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.
    anarci_path (str): path to ANARCI output for sequence numbering.
                       If None, the (0-based) index along the sequence string is used.
    collapse (bool): whether sites with lettered suffix should be consolidated into one site
                     (e.g. 111A, 111B, etc. will counted as site 111).
                     Ignored if anarci_path is None.
    verbose (bool): whether to print (sample ID, family ID) info when ANARCI output has sequence length mismatch.
                    Ignored if anarci_path is None.

    Returns:
    output_df (pd.DataFrame): dataframe with columns pcp_index, site, obs, pred.
    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    pcp_sub_locations = [
        locate_child_substitutions(parent, child)
        for parent, child in zip(parent_aa_seqs, child_aa_seqs)
    ]

    # k represents the number of substitutions observed in each PCP, top k substitutions will be evaluated for r-precision
    k_subs = [len(pcp_sub_location) for pcp_sub_location in pcp_sub_locations]

    pcp_indices = []
    site_sub_probs = []
    pcp_sample_family_dict = {}
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            pcp_indices.append(pcp_index)

            site_sub_probs.append(
                calculate_site_substitution_probabilities(matrix, parent_aa_seqs[index])
            )

            pcp_sample_family_dict[pcp_index] = tuple(
                pcp_df.loc[pcp_index][["sample_id", "family"]]
            )

    top_k_sub_locations = [
        locate_top_k_substitutions(site_sub_prob, k_sub)
        for site_sub_prob, k_sub in zip(site_sub_probs, k_subs)
    ]

    numbering_dict = get_numbering_dict(anarci_path, pcp_df, collapse, verbose)

    pcp_index_col = []
    site_col = []
    obs_col = []
    pred_col = []
    for i in range(len(pcp_indices)):
        obs_pred_sites = np.union1d(top_k_sub_locations[i], pcp_sub_locations[i])

        nbkey = pcp_sample_family_dict[pcp_indices[i]]
        if (anarci_path is not None) and (nbkey not in numbering_dict):
            continue

        for site in obs_pred_sites:
            pcp_index_col.append(pcp_indices[i])
            if anarci_path is None:
                site_col.append(site)
            else:
                site_col.append(numbering_dict[nbkey][site])
            obs_col.append(site in pcp_sub_locations[i])
            pred_col.append(site in top_k_sub_locations[i])

    output_df = pd.DataFrame(columns=["pcp_index", "site", "obs", "pred"])
    output_df["pcp_index"] = pcp_index_col
    output_df["site"] = site_col
    output_df["obs"] = obs_col
    output_df["pred"] = pred_col

    return output_df


def plot_sites_observed_vs_top_k_predictions(
    df,
    ax,
    correct_color="#009E73",
    correct_label="Correct",
    incorrect_color="#D95F02",
    incorrect_label="Incorrect",
    logy=False,
):
    """
    Draws a figure of observed amino acid substitutions and the top-k predictions across PCPs in a dataset.
    The input dataframe requires three columns:
    'site' (site position -- may be at the level of nucleotide, or codon, or amino acid, etc.)
    'obs' (True/False whether the site has an observed substitution)
    'pred' (True/False whether the site is in the top-k predicted substitutions)
    Each dataframe row corresponds to a site in a specific sequence.
    Only sites that are True in either 'obs' or 'pred' columns are involved in the plotting and calculation.
    Hence, sites in a PCP that neither have an observed substitution nor are predicted in the top-k
    can be excluded from the dataframe.

    Parameters:
    df (pd.DataFrame): dataframe of observed and predicted sites of substitution.
    ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    correct_color (str): color for the plot of correct predictions.
    correct_label (str): legend label for the plot of correct predictions.
    incorrect_color (str): color for the plot of incorrect predictions.
    incorrect_label (str): legend label for the plot of incorrect predictions.
    logy (bool): whether to show y-axis in log-scale.

    Returns:
    A dictionary with results labeled:
    overlap (float): area of overlap between observed and expected, divided by the average of the two areas.
    residual (float): square root of the sum of squared bin-by-bin differences between observed and expected, divided by total expected.
    r-precision (fig.ax): R-precision of the dataset.

    """
    if df.dtypes["site"] == "object":
        nblist = df["site"].drop_duplicates().to_list()
        xvals = sorted(
            nblist,
            key=lambda x: (
                int(x) if x.isnumeric() else int(x[:-1]),
                "" if x.isnumeric() else x[-1],
            ),
        )
    else:
        xvals = np.arange(np.min(df["site"]), np.max(df["site"]) + 1)

    predicted = []
    observed = []
    correct = []
    for site in xvals:
        site_df = df[df["site"] == site]
        npred = site_df[site_df["pred"] == True].shape[0]
        predicted.append(npred)
        nobs = site_df[site_df["obs"] == True].shape[0]
        observed.append(nobs)
        ncorr = site_df[(site_df["pred"] == True) & (site_df["obs"] == True)].shape[0]
        correct.append(ncorr)
    predicted = np.array(predicted)
    observed = np.array(observed)
    correct = np.array(correct)

    # compute overlap metric
    intersect = np.sum(np.minimum(observed, predicted))
    denom = 0.5 * (np.sum(observed) + np.sum(predicted))
    overlap = intersect / denom

    # compute residual metric
    diff = observed - predicted
    residual = np.sqrt(np.sum(diff * diff)) / np.sum(predicted)

    # compute R-precision
    tmpdf = (
        df[df["obs"] == True][["pcp_index", "obs", "pred"]].groupby("pcp_index").sum()
    )
    pcp_rprec = tmpdf["pred"].to_numpy() / tmpdf["obs"].to_numpy()
    rprec = sum(pcp_rprec) / len(pcp_rprec)

    if ax is not None:
        ax.bar(
            xvals,
            predicted,
            width=1,
            facecolor="white",
            edgecolor=incorrect_color,
            label=incorrect_label,
        )

        ax.bar(
            xvals,
            correct,
            width=1,
            color=correct_color,
            edgecolor=correct_color,
            label=correct_label,
        )

        ax.plot(
            xvals,
            observed,
            marker="o",
            markersize=4,
            linewidth=0,
            color="#000000",
            label="Observed",
        )

        if df.dtypes["site"] == "object":
            ax.tick_params(axis="x", labelsize=7, labelrotation=90)
        else:
            ax.tick_params(axis="x", labelsize=16)
        ax.set_xlabel("amino acid sequence position", fontsize=20, labelpad=10)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylabel("number of substitutions", fontsize=20, labelpad=10)

        if logy:
            ax.set_yscale("log")

        ax.legend(fontsize=15)

    return {
        "overlap": overlap,
        "residual": residual,
        "r-precision": rprec,
    }


def get_numbering_dict(anarci_path, pcp_df=None, collapse=True, verbose=False):
    """
    Process the ANARCI output to make site numbering lists for each clonal family.

    Parameters:
    anarci_path (str): path to ANARCI output for sequence numbering.
    pcp_df (pd.Dataframe): PCP file to filter for relevant clonal families and check ANARCI sequence lengths.
    collapse (bool): whether sites with lettered suffix should be consolidated into one site
                     (e.g. 111A, 111B, etc. will counted as site 111).
                     Ignored if anarci_path is None.
    verbose (bool): whether to print (sample ID, family ID) info when ANARCI output has sequence length mismatch.

    Returns:
    A dictionary with keys as 2-tuples of (sample_id, family), and with values as lists of numberings for each site in the clonal family.
    If collapse is True, the numberings are lists of integers, otherwise the numberings are lists of strings.
    """
    numbering_dict = {}

    if anarci_path is not None:
        anarci_df = pd.read_csv(anarci_path)

        # assumes numbering starts at column 13 in ANARCI output
        numbering_cols = anarci_df.columns[13:]

        for i, row in anarci_df.iterrows():
            # assumes clonal family ID has format "{sample_id}|{family}|{seq_name}"
            [sample_id, family, seq_name] = row["Id"].split("|")
            seqlist = [row[col] for col in numbering_cols]
            numbering = [nn for nn, aa in zip(numbering_cols, seqlist) if aa != "-"]

            if pcp_df is not None:
                # Check if clonal family is in PCP file, and that ANARCI preserved sequence length.
                # If not, exclude clonal family from output.
                test_df = pcp_df[
                    (pcp_df["sample_id"] == sample_id)
                    & (pcp_df["family"] == int(family))
                ]
                if test_df.shape[0] == 0:
                    continue
                else:
                    test_seq = translate_sequence(test_df.head(1)["parent"].item())
                    if len(test_seq) != len(numbering):
                        if verbose == True:
                            print("ANARCI seq length mismatch!", sample_id, family)
                        continue

            if collapse:
                for j in range(len(numbering)):
                    nn = numbering[j]
                    if nn[-1] not in "0123456789":
                        numbering[j] = int(nn[:-1])
                    else:
                        numbering[j] = int(nn)

            numbering_dict[(sample_id, int(family))] = numbering

    return numbering_dict
