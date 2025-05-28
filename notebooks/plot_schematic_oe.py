import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
from epam.utils import load_and_filter_pcp_df
from netam.oe_plot import (
    plot_sites_observed_vs_expected,
)
from epam.df_for_plots import (
    get_numbering_dict,
    get_site_mutabilities_df,
)

# Okabe-Ito colors
oi_black         = '#000000'
oi_orange        = '#E69F00'
oi_skyblue       = '#56B4E9'
oi_bluishgreen   = '#009E73'
oi_yellow        = '#F0E442'
oi_blue          = '#0072B2'
oi_vermillion    = '#D55E00'
oi_reddishpurple = '#CC79A7'

aaprobs_flairr_dir = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v2/ford-flairr-seq-prod_pcp_2024-07-26_MASKED_NI_noN_no-naive"
anarci_flairr = "/fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/ford-flairr-seq-prod_imgt.csv"
pcp_path = "/home/mjohnso4/epam/pcp_inputs/ford-flairr-seq-prod_pcp_2024-07-26_MASKED_NI_noN_no-naive.csv"
schematic_output_dir = "/home/mjohnso4/epam/output"

major_text_size = 30 #20
minor_text_size = 20 #15

pcp_df = load_and_filter_pcp_df(pcp_path)
numbering, excluded = get_numbering_dict(anarci_flairr, pcp_df, True, "imgt")

def plot_schematic():
    aaprob = f"{aaprobs_flairr_dir}/ThriftyProdHumV0.2-59/combined_aaprob.hdf5"

    fig, ax = plt.subplots(figsize=[10,6]) #15,5
    fig.patch.set_facecolor('white')

    site_sub_probs_df = get_site_mutabilities_df(aaprob, numbering)
    site_numbers = [str(i) for i in range(1, 41)] # 71
    zoomed_df = site_sub_probs_df[site_sub_probs_df['site'].isin(site_numbers)]
    results = plot_sites_observed_vs_expected(zoomed_df, ax, numbering)

    # ax.text(
    #     0.975, 0.975,
    #     f'overlap={results["overlap"]:.3g}\nresidual={results["residual"]:.3g}',
    #     verticalalignment='top', 
    #     horizontalalignment='right', 
    #     transform=ax.transAxes,
    #     fontsize=15
    # )

    ax.set_xlim(-.5, 36.5) # 51 - 46.5, 41 - 36.5, 71 - 66.5
    ax.set_xlabel("IMGT position", fontsize=major_text_size, labelpad=10)
    ax.tick_params(axis='x', labelsize=minor_text_size)
    ax.set_ylabel("number of substitutions", fontsize=major_text_size, labelpad=10)
    ax.legend(loc='upper left', fontsize=minor_text_size)
    plt.tight_layout()

    outfname = f"{schematic_output_dir}/schematic_zoom_oe.pdf"
    print(outfname, 'created!')
    plt.savefig(outfname)
    plt.close()

plot_schematic()
