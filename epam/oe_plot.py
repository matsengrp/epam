"""Code for generating observed vs expected plots"""

import h5py
import numpy as np
import pandas as pd
import bisect
import matplotlib.pyplot as plt
from epam.utils import pcp_path_of_aaprob_path, load_and_filter_pcp_df
from epam.evaluation import (
    locate_child_substitutions,
    calculate_site_substitution_probabilities,
    locate_top_k_substitutions,
)
from netam.sequences import (
    translate_sequences,
    translate_sequence,
)
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def annotate_sites_df(
    df,
    pcp_df,
    numbering_dict=None,
):
    """
    Add annotations to a per-site DataFrame to indicate position of each site and whether each site is in a CDR.
    The input DataFrame describes a site in each row is expected to have the 'pcp_index' column,
    indicating the index of the PCP the site belongs to.

    Parameters:
    df (pd.DataFrame): site mutabilities DataFrame.
    pcp_df (pd.DataFrame): PCP file of the dataset.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with additional columns 'site' and 'is_cdr'.
                              Note that if numbering_dict is provided and there are clonal families with missing
                              ANARCI numberings, the associated sites (rows) will be excluded.
    """
    sites_col = []
    is_cdr_col = []

    pcp_groups = df.groupby("pcp_index")
    for pcp_index in df["pcp_index"].drop_duplicates():
        pcp_row = pcp_df.loc[pcp_index]

        group_df = pcp_groups.get_group(pcp_index)
        nsites = group_df.shape[0]
        assert (
            nsites == len(pcp_row["parent"]) // 3
        ), f"number of sites ({nsites}) does not match sequence length ({len(pcp_row['parent']) // 3})"

        if numbering_dict is None:
            sites_col.append(np.arange(nsites))
        else:
            nbkey = tuple(pcp_row[["sample_id", "family"]])
            if nbkey in numbering_dict:
                sites_col.append(numbering_dict[nbkey])
            else:
                # Assign sites as "None", marking them for exclusion from output.
                sites_col.append(["None"] * nsites)

        cdr1 = (
            pcp_row["cdr1_codon_start"] // 3,
            pcp_row["cdr1_codon_end"] // 3,
        )
        cdr2 = (
            pcp_row["cdr2_codon_start"] // 3,
            pcp_row["cdr2_codon_end"] // 3,
        )
        cdr3 = (
            pcp_row["cdr3_codon_start"] // 3,
            pcp_row["cdr3_codon_end"] // 3,
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
                for i in range(nsites)
            ]
        )

    df["site"] = np.concatenate(sites_col)
    df["is_cdr"] = np.concatenate(is_cdr_col)
    if numbering_dict is None:
        return df
    else:
        return df[df["site"] != "None"]


def get_site_mutabilities_df(
    aaprob_path,
    numbering_dict=None,
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
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with columns pcp_index, site, prob, mutation, is_cdr.
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
    is_cdr_col = []
    with h5py.File(aaprob_path, "r") as matfile:
        for index in range(len(parent_aa_seqs)):
            pcp_index = pcp_df.index[index]
            pcp_row = pcp_df.loc[pcp_index]
            grp = matfile[
                "matrix" + str(pcp_index)
            ]  # assumes "matrix0" naming convention and that matrix names and pcp indices match
            matrix = grp["data"]

            parent = parent_aa_seqs[index]
            child = child_aa_seqs[index]

            if numbering_dict is None:
                sites_col.append(np.arange(len(parent)))
            else:
                nbkey = tuple(pcp_row[["sample_id", "family"]])
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
                pcp_row["cdr1_codon_start"] // 3,
                pcp_row["cdr1_codon_end"] // 3,
            )
            cdr2 = (
                pcp_row["cdr2_codon_start"] // 3,
                pcp_row["cdr2_codon_end"] // 3,
            )
            cdr3 = (
                pcp_row["cdr3_codon_start"] // 3,
                pcp_row["cdr3_codon_end"] // 3,
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
    numbering_dict=None,
    fwk_color="#0072B2",
    cdr_color="#E69F00",
    logy=False,
):
    """
    Draws a figure of observed vs expected number of mutations at each site position across PCPs in a dataset.
    The input dataframe requires four columns:
    'site' (site position -- may be at the level of nucleotide, or codon, or amino acid, etc.)
    'prob' (site mutability)
    'mutation' (True/False whether the site has an observed mutation or not)
    'is_cdr' (True/False whether the site is in a CDR)
    Each dataframe row corresponds to a site in a specific sequence.

    Parameters:
    df (pd.DataFrame): dataframe of observed and predicted sites of substitution.
    ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.
    fwk_color (str): color for the FWK sites.
    cdr_color (str): color for the CDR sites.
    logy (bool): whether to show y-axis in log-scale.

    Returns:
    A dictionary with results labeled:
    overlap (float): area of overlap between observed and expected, divided by the average of the two areas.
    residual (float): square root of the sum of squared bin-by-bin differences between observed and expected, divided by total expected.

    """
    if numbering_dict is None:
        xvals = np.arange(np.min(df["site"]), np.max(df["site"]) + 1)
    else:
        xvals = numbering_dict[("reference", 0)]
    ixvals = np.arange(len(xvals))

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
        ax.margins(x=0.01)

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


def get_subs_and_preds_from_mutabilities_df(df, pcp_df):
    """
    Determines the sites of observed and predicted substitutions of every PCP in a dataset,
    from a site mutabilities DataFrame, which has columns
    'pcp_index' (index of the PCP that the site belongs to),
    'prob' (the mutability probability of the site),
    'mutation' (whether the site has an observed mutation).
    Predicted substitutions are the sites in the top-k of mutability,
    where k is the number of observed substition in the PCP.

    Parameters:
    df (pd.DataFrame): site mutabilities DataFrame.
    pcp_df (pd.DataFrame): PCP file of the dataset.

    Returns tuple with:
    pcp_indices (list): indices to the reference PCP file.
    pcp_sub_locations (list): per-PCP lists of substitution locations (positions along the sequence string).
    top_k_sub_locations (list): per-PCP lists of top-k mutability locations (positions along the sequence string).
    pcp_sample_family_dict (dict): mapping PCP index to (sample_id, family) 2-tuple.
    """
    pcp_indices = list(df["pcp_index"].drop_duplicates())
    pcp_sub_locations = []
    top_k_sub_locations = []
    pcp_sample_family_dict = {}

    for pcp_index in pcp_indices:
        probs = list(df[df["pcp_index"] == pcp_index]["prob"])
        mutations = list(df[df["pcp_index"] == pcp_index]["mutation"])
        pcp_sub_locations.append(
            list(i for i in range(len(mutations)) if mutations[i] == True)
        )
        top_k_sub_locations.append(locate_top_k_substitutions(probs, sum(mutations)))
        pcp_sample_family_dict[pcp_index] = tuple(
            pcp_df.loc[pcp_index][["sample_id", "family"]]
        )

    return (pcp_indices, pcp_sub_locations, top_k_sub_locations, pcp_sample_family_dict)


def get_subs_and_preds_from_aaprob(aaprob_path):
    """
    Determines the sites of observed and predicted substitutions of every PCP in a dataset,
    from the aaprob file of matrices.
    Predicted substitutions are the sites in the top-k of mutability,
    where k is the number of observed substition in the PCP.

    Parameters:
    aaprob_path (str): path to aaprob matrix for parent-child pairs.

    Returns tuple with:
    pcp_indices (list): indices to the reference PCP file.
    pcp_sub_locations (list): per-PCP lists of substitution locations (positions along the sequence string).
    top_k_sub_locations (list): per-PCP lists of top-k mutability locations (positions along the sequence string).
    pcp_sample_family_dict (dict): mapping PCP index to (sample_id, family) 2-tuple.
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

    return (pcp_indices, pcp_sub_locations, top_k_sub_locations, pcp_sample_family_dict)


def get_site_substitutions_df(
    subs_and_preds_tuple,
    numbering_dict=None,
):
    """
    Returns a dataframe that annotates for each site of observed and/or predicted substitution:
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    whether the site has an observed substutition,
    and whether the site is predicted to have a substitution.

    Parameters:
    subs_and_preds_tuple (tuple): 4-tuple of
                                  pcp_indices - list of indices to the reference PCP file,
                                  pcp_sub_locations - list of per-PCP lists of substitution locations,
                                  top_k_sub_locations - list of per-PCP lists of top-k mutability locations,
                                  pcp_sample_family_dict - dictionary mapping PCP index to (sample_id, family) 2-tuple.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with columns 'pcp_index', 'site', 'obs', 'pred'.
    """
    pcp_indices = subs_and_preds_tuple[0]
    pcp_sub_locations = subs_and_preds_tuple[1]
    top_k_sub_locations = subs_and_preds_tuple[2]
    pcp_sample_family_dict = subs_and_preds_tuple[3]

    pcp_index_col = []
    site_col = []
    obs_col = []
    pred_col = []
    for i in range(len(pcp_indices)):
        obs_pred_sites = np.union1d(top_k_sub_locations[i], pcp_sub_locations[i])

        nbkey = pcp_sample_family_dict[pcp_indices[i]]
        if (numbering_dict is not None) and (nbkey not in numbering_dict):
            continue

        for site in obs_pred_sites:
            pcp_index_col.append(pcp_indices[i])
            if numbering_dict is None:
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
    numbering_dict=None,
    correct_color="#009E73",
    correct_label="Correct",
    incorrect_color="#009E73",
    incorrect_label="Incorrect",
    logy=False,
):
    """
    Draws a figure of observed mutations and the top-k predictions across PCPs in a dataset.
    The input dataframe requires three columns:
    'site' (site position -- may be at the level of nucleotide, or codon, or amino acid, etc.)
    'obs' (True/False whether the site has an observed substitution)
    'pred' (True/False whether the site is in the top-k predicted substitutions)
    Each dataframe row corresponds to a site in a specific sequence.
    Only sites that are True in either 'obs' or 'pred' columns are involved in the plotting and calculation.
    Hence, sites in a PCP that have neither an observed substitution nor are predicted in the top-k
    can be excluded from the dataframe.

    Parameters:
    df (pd.DataFrame): dataframe of observed and predicted sites of substitution.
    ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.
    correct_color (str): color for the plot of correct predictions.
    correct_label (str): legend label for the plot of correct predictions.
    incorrect_color (str): color for the plot of incorrect predictions.
    incorrect_label (str): legend label for the plot of incorrect predictions.
    logy (bool): whether to show y-axis in log-scale.

    Returns:
    A dictionary with results labeled:
    overlap (float): area of overlap between observed and expected, divided by the average of the two areas.
    residual (float): square root of the sum of squared bin-by-bin differences between observed and expected, divided by total expected.
    r-precision (float): R-precision of the dataset.

    """
    if numbering_dict is None:
        xvals = np.arange(np.min(df["site"]), np.max(df["site"]) + 1)
    else:
        xvals = numbering_dict[("reference", 0)]

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
        ax.margins(x=0.01)

        if logy:
            ax.set_yscale("log")

        ax.legend(fontsize=15)

    return {
        "overlap": overlap,
        "residual": residual,
        "r-precision": rprec,
    }


def get_numbering_dict(anarci_path, pcp_df=None, verbose=False, checks="imgt"):
    """
    Process ANARCI output to make site numbering lists for each clonal family.

    Parameters:
    anarci_path (str): path to ANARCI output for sequence numbering.
    pcp_df (pd.Dataframe): PCP file to filter for relevant clonal families and check ANARCI sequence lengths.
    verbose (bool): whether to print (sample ID, family ID) info when ANARCI output has sequence length mismatch.
    checks (str): perform checks and updates for a specified numbering scheme.
                  Currently, 'imgt' is the only input that has an effect.

    Returns:
    A dictionary with keys as 2-tuples of (sample_id, family), and with values as lists of numberings for each site in the clonal family.
    Note that the numberings are lists of strings.
    The dictionary also has an entry with key ('reference', 0) and value the list of all site numberings,
    to be used for setting x-axis tick labels when plotting.
    """
    if anarci_path is None:
        return None

    numbering_dict = {}

    anarci_df = pd.read_csv(anarci_path)

    # assumes numbering starts at column 13 in ANARCI output
    numbering_cols = list(anarci_df.columns[13:])
    numbering_used = [False] * len(numbering_cols)

    for i, row in anarci_df.iterrows():
        # assumes clonal family ID has format "{sample_id}|{family}|{seq_name}"
        [sample_id, family, seq_name] = row["Id"].split("|")
        seqlist = [row[col] for col in numbering_cols]
        numbering = [nn for nn, aa in zip(numbering_cols, seqlist) if aa != "-"]

        if checks == "imgt":
            # For IMGT, numbered insertions can only be 111.* or 112.*.
            # Other numbered insertions come from ANARCI and the clonal family will be excluded
            exclude = False
            for nn in numbering:
                if "." in nn and nn[:3] != "111" and nn[:3] != "112":
                    print("Invalid IMGT insertion", sample_id, family, nn)
                    exclude = True
                    break
            if exclude == True:
                continue

        if pcp_df is not None:
            # Check if clonal family is in PCP file, and that ANARCI preserved sequence length.
            # If not, exclude clonal family from output.
            test_df = pcp_df[
                (pcp_df["sample_id"] == sample_id) & (pcp_df["family"] == int(family))
            ]
            if test_df.shape[0] == 0:
                continue
            else:
                pcp_row = test_df.head(1)
                test_seq = translate_sequence(pcp_row["parent"].item())
                if len(test_seq) != len(numbering):
                    if verbose == True:
                        print("ANARCI seq length mismatch!", sample_id, family)
                    continue

                if checks == "imgt":
                    # Check CDR annotation in PCP file is consistent with IMGT numbering.
                    # If not, exclude the clonal family.
                    cdr1 = (
                        pcp_row["cdr1_codon_start"].item() // 3,
                        pcp_row["cdr1_codon_end"].item() // 3,
                    )
                    cdr2 = (
                        pcp_row["cdr2_codon_start"].item() // 3,
                        pcp_row["cdr2_codon_end"].item() // 3,
                    )
                    cdr3 = (
                        pcp_row["cdr3_codon_start"].item() // 3,
                        pcp_row["cdr3_codon_end"].item() // 3,
                    )

                    cdr_anno = [
                        (
                            True
                            if (i >= cdr1[0] and i <= cdr1[1])
                            or (i >= cdr2[0] and i <= cdr2[1])
                            or (i >= cdr3[0] and i <= cdr3[1])
                            else False
                        )
                        for i in range(len(test_seq))
                    ]

                    exclude = False
                    for nn, is_cdr in zip(numbering, cdr_anno):
                        if is_imgt_cdr(nn) != is_cdr:
                            print(
                                "IMGT mismatch with CDR annotation!", sample_id, family
                            )
                            exclude = True
                            break
                    if exclude == True:
                        continue

        numbering_dict[(sample_id, int(family))] = numbering

        # keep track of which site numbers are used
        for nn in numbering:
            numbering_used[numbering_cols.index(nn)] = True

    # make a numbering reference of site numbers that are used
    numbering_dict[("reference", 0)] = [
        nn for nn, used in zip(numbering_cols, numbering_used) if used == True
    ]

    return numbering_dict


def is_imgt_cdr(site):
    """
    Determines whether an amino acid site is in a CDR according to IMGT numbering.

    Parameters:
    site (str): IMGT number of an amino acid site.

    Returns:
    True or False whether the site is in a CDR.
    """
    IMGT_CDR1 = (27, 38)
    IMGT_CDR2 = (56, 65)
    IMGT_CDR3 = (105, 117)

    # Note: IMGT uses decimals for insertions (e.g. '111.3')
    if "." in site:
        sitei = int(site.split(".")[0])
    else:
        sitei = int(site)

    return (
        (sitei >= IMGT_CDR1[0] and sitei <= IMGT_CDR1[1])
        or (sitei >= IMGT_CDR2[0] and sitei <= IMGT_CDR2[1])
        or (sitei >= IMGT_CDR3[0] and sitei <= IMGT_CDR3[1])
    )
