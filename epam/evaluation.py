"""Code for evaluating model performance."""

import h5py
import pandas as pd
import numpy as np
import bisect
from epam.sequences import AA_STR_SORTED
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.sequences import translate_sequences, pcp_criteria_check
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
    Returns evaluation metrics for a single aaprob matrix (generated from one model on one data set).

    Parameters:
    aaprob_path (str): Path to aaprob matrix for parent-child pairs.

    Returns:
    model_performance (dict): Dictionary of model performance metrics for a single aaprob matrix.

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

    pcp_sub_aa_ids = [
        identify_child_substitutions(parent, child)
        for parent, child in zip(parent_aa_seqs, child_aa_seqs)
    ]

    # k represents the number of substitutions observed in each PCP, top k substitutions will be evaluated for r-precision
    k_subs = [len(pcp_sub_location) for pcp_sub_location in pcp_sub_locations]

    site_sub_probs = []
    model_sub_aa_ids = []

    with h5py.File(aaprob_path, "r") as matfile:
        model_name = matfile.attrs["model_name"]
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            site_sub_probs.append(
                calculate_site_substitution_probabilities(matrix, parent_aa_seqs[index])
            )

            pred_aa_sub = [
                highest_ranked_substitution(matrix[j, :], parent_aa_seqs[index], j)
                for j in range(len(parent_aa_seqs[index]))
                if parent_aa_seqs[index][j] != child_aa_seqs[index][j]
            ]

            model_sub_aa_ids.append(pred_aa_sub)

    top_k_sub_locations = [
        locate_top_k_substitutions(site_sub_prob, k_sub)
        for site_sub_prob, k_sub in zip(site_sub_probs, k_subs)
    ]

    sub_acc = calculate_sub_accuracy(pcp_sub_aa_ids, model_sub_aa_ids, k_subs)
    r_prec = calculate_r_precision(pcp_sub_locations, top_k_sub_locations, k_subs)
    cross_ent = calculate_cross_entropy_loss(pcp_sub_locations, site_sub_probs)

    model_performance = {
        "data_set": pcp_path,
        "pcp_count": len(pcp_df),
        "model": model_name,
        "sub_accuracy": sub_acc,
        "r_precision": r_prec,
        "cross_entropy": cross_ent,
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
    site_sub_probs = [
        np.sum(
            aaprobs[i, :][
                [AA_STR_SORTED.index(aa) for aa in AA_STR_SORTED if aa != parent_aa[i]]
            ]
        )
        for i in range(len(parent_aa))
    ]

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
                log_probs_substitution.append(np.log(p_i))
            else:
                log_probs_substitution.append(np.log(1 - p_i))

    cross_entropy_loss = (
        -1 / len(log_probs_substitution) * np.sum(log_probs_substitution)
    )

    return cross_entropy_loss


def get_site_mutabilities_df(aaprob_path):
    """
    Computes the amino acid site mutability probabilities
    for every site of every parent in a dataset.
    Returns a dataframe that annotates for each site 
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    and whether a mutation is observed in the child sequence.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.

    Returns:
    output_df (pd.DataFrame): dataframe with columns pcp_index, site, prob, mutation.
    """
    pcp_path = pcp_path_of_aaprob_path(aaprob_path)
    pcp_df = load_and_filter_pcp_df(pcp_path)
    nt_seqs = list(zip(pcp_df["parent"], pcp_df["child"]))
    aa_seqs = [tuple(translate_sequences(pcp_pair)) for pcp_pair in nt_seqs]
    parent_aa_seqs, child_aa_seqs = zip(*aa_seqs)

    pcp_index_col = []
    sites_col = []
    site_sub_probs = []
    site_sub_flags = []
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            parent = parent_aa_seqs[index]
            child = child_aa_seqs[index]

            pcp_index_col.append([pcp_index] * len(parent))
            sites_col.append(np.arange(len(parent)))
            site_sub_probs.append(
                calculate_site_substitution_probabilities(matrix, parent)
            )
            site_sub_flags.append([p != c for p, c in zip(parent, child)])

    output_df = pd.DataFrame(columns=["pcp_index", "site", "prob", "mutation"])
    output_df["pcp_index"] = np.concatenate(pcp_index_col)
    output_df["site"] = np.concatenate(sites_col)
    output_df["prob"] = np.concatenate(site_sub_probs)
    output_df["mutation"] = np.concatenate(site_sub_flags)

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
    counts of sites in bins of mutability probability,
    observed vs expected number of mutations in bins of mutability probability,
    and per bin differences between observed and expected.
    The expected number of mutations is computed as the total probability
    of the sites that fall in that mutability bin.
    The input dataframe requires two columns: 'prob'
    (site mutability -- may be at level of nucleotide, or codon, or amino acid, etc.)
    and 'mutation' (1 or 0 if the site has an observed mutation or not).
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
        xlabel = "$\log_{10}$(mutability probability)"
    else:
        obs_probs = df[df["mutation"] > 0]["prob"].to_numpy()
        xlabel = "mutability probability"
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
