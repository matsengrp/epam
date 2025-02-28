import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
from epam.utils import load_and_filter_pcp_df
from epam.oe_plot import (
    get_numbering_dict,
    get_site_mutabilities_df, 
    plot_sites_observed_vs_expected,
    get_subs_and_preds_from_aaprob,
    get_site_substitutions_df, 
    plot_sites_observed_vs_top_k_predictions,
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

ensemble_output_dir = "/home/mjohnso4/epam/output/v1_ensemble/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive"
anarci_flairr = "/fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v1/anarci/ford-flairr-seq-prod_anarci-seqs_imgt_H_patch.csv"
plot_output_dir = ensemble_output_dir
pcp_path = "/home/mjohnso4/epam/pcp_inputs/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv"

pcp_df = load_and_filter_pcp_df(pcp_path)
numbering, excluded = get_numbering_dict(anarci_flairr, pcp_df, True, "imgt")

# def plot_esm_scaling():
#     aaprob_no_scale = "/home/mjohnso4/epam/output/v1/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/set1/ESM1v_mask/combined_aaprob.hdf5"
#     aaprob_scale = f"{ensemble_output_dir}/esm1/ESM1v_mask/combined_aaprob.hdf5"

#     title = "ESM1v standalone - FLAIRR"

#     fig = plt.figure(figsize=[15,8])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(2, height_ratios=[4,4])
#     axs = gs.subplots(sharex=True, sharey=False)

#     old_site_sub_probs_df = get_site_mutabilities_df(aaprob_no_scale, numbering)
#     results = plot_sites_observed_vs_expected(old_site_sub_probs_df, axs[0], numbering)
#     axs[0].text(
#         0.02, 0.8,
#         f'overlap={results["overlap"]:.3g}\nresidual={results["residual"]:.3g}',
#         verticalalignment ='top', 
#         horizontalalignment ='left', 
#         transform = axs[0].transAxes,
#         fontsize=15
#     )
#     axs[0].axes.get_xaxis().get_label().set_visible(False)
#     axs[0].axes.get_yaxis().get_label().set_visible(False)
#     # axs[0].legend(loc='upper left', fontsize=12)

#     axs[0].set_title("No scaling", fontsize =18)

#     new_site_sub_probs_df = get_site_mutabilities_df(aaprob_scale, numbering)
#     new_results = plot_sites_observed_vs_expected(new_site_sub_probs_df, axs[1], numbering)
#     axs[1].text(
#         0.02, 0.8,
#         f'overlap={new_results["overlap"]:.3g}\nresidual={new_results["residual"]:.3g}',
#         verticalalignment ='top', 
#         horizontalalignment ='left', 
#         transform = axs[1].transAxes,
#         fontsize=15
#     )
#     axs[1].set_xlabel("IMGT position", fontsize=20, labelpad=10)
#     axs[1].axes.get_yaxis().get_label().set_visible(False)

#     axs[1].set_title("With scaling", fontsize =18)

#     # Suppress individual legends in both panels
#     axs[0].legend().set_visible(False)
#     axs[1].legend().set_visible(False)

#     fig.suptitle(title, fontsize=20)
#     fig.supylabel('number of substitutions', fontsize=20)
#     plt.tight_layout()

#     outfname = f"{plot_output_dir}/scaling_update_oe.png"
#     print(outfname,'created!')
#     plt.savefig(outfname)
#     plt.close()

# # plot_esm_scaling()

def plot_esm_ensemble(model = "ESM-1v"):
    aaprob_esm1 = f"{ensemble_output_dir}/esm1/{model}/combined_aaprob.hdf5"
    aaprob_esm2 = f"{ensemble_output_dir}/esm2/{model}/combined_aaprob.hdf5"
    aaprob_esm3 = f"{ensemble_output_dir}/esm3/{model}/combined_aaprob.hdf5"
    aaprob_esm4 = f"{ensemble_output_dir}/esm4/{model}/combined_aaprob.hdf5"
    aaprob_esm5 = f"{ensemble_output_dir}/esm5/{model}/combined_aaprob.hdf5"
    aaprob_ensemble = f"{ensemble_output_dir}/ensemble_set/{model}/combined_aaprob.hdf5"

    title = f"{model} - FLAIRR"

    fig = plt.figure(figsize=[15,12])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(6, height_ratios=[2,2,2,2,2,2])
    axs = gs.subplots(sharex=True, sharey=False)

    site_sub_probs_df_1 = get_site_mutabilities_df(aaprob_esm1, numbering)
    results_1 = plot_sites_observed_vs_expected(site_sub_probs_df_1, axs[0], numbering)
    axs[0].text(
        0.02, 0.9,
        f'overlap={results_1["overlap"]:.3g}\nresidual={results_1["residual"]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = axs[0].transAxes,
        fontsize=15
    )
    axs[0].axes.get_xaxis().get_label().set_visible(False)
    axs[0].axes.get_yaxis().get_label().set_visible(False)
    axs[0].set_title("1", fontsize =18)

    site_sub_probs_df_2 = get_site_mutabilities_df(aaprob_esm2, numbering)
    results_2 = plot_sites_observed_vs_expected(site_sub_probs_df_2, axs[1], numbering)
    axs[1].text(
        0.02, 0.9,
        f'overlap={results_2["overlap"]:.3g}\nresidual={results_2["residual"]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = axs[1].transAxes,
        fontsize=15
    )
    axs[1].axes.get_xaxis().get_label().set_visible(False)
    axs[1].axes.get_yaxis().get_label().set_visible(False)
    axs[1].set_title("2", fontsize =18)

    site_sub_probs_df_3 = get_site_mutabilities_df(aaprob_esm3, numbering)
    results_3 = plot_sites_observed_vs_expected(site_sub_probs_df_3, axs[2], numbering)
    axs[2].text(
        0.02, 0.9,
        f'overlap={results_3["overlap"]:.3g}\nresidual={results_3["residual"]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = axs[2].transAxes,
        fontsize=15
    )
    axs[2].axes.get_xaxis().get_label().set_visible(False)
    axs[2].axes.get_yaxis().get_label().set_visible(False)
    axs[2].set_title("3", fontsize =18)

    site_sub_probs_df_4 = get_site_mutabilities_df(aaprob_esm4, numbering)
    results_4 = plot_sites_observed_vs_expected(site_sub_probs_df_4, axs[3], numbering)
    axs[3].text(
        0.02, 0.9,
        f'overlap={results_4["overlap"]:.3g}\nresidual={results_4["residual"]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = axs[3].transAxes,
        fontsize=15
    )
    axs[3].axes.get_xaxis().get_label().set_visible(False)
    axs[3].axes.get_yaxis().get_label().set_visible(False)
    axs[3].set_title("4", fontsize =18)

    site_sub_probs_df_5 = get_site_mutabilities_df(aaprob_esm5, numbering)
    results_5 = plot_sites_observed_vs_expected(site_sub_probs_df_5, axs[4], numbering)
    axs[4].text(
        0.02, 0.9,
        f'overlap={results_5["overlap"]:.3g}\nresidual={results_5["residual"]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = axs[4].transAxes,
        fontsize=15
    )
    axs[4].axes.get_xaxis().get_label().set_visible(False)
    axs[4].axes.get_yaxis().get_label().set_visible(False)
    axs[4].set_title("5", fontsize =18)

    site_sub_probs_df_all = get_site_mutabilities_df(aaprob_ensemble, numbering)
    results_all = plot_sites_observed_vs_expected(site_sub_probs_df_all, axs[5], numbering)
    axs[5].text(
        0.02, 0.9,
        f'overlap={results_all["overlap"]:.3g}\nresidual={results_all["residual"]:.3g}',
        verticalalignment ='top', 
        horizontalalignment ='left', 
        transform = axs[5].transAxes,
        fontsize=15
    )
    axs[5].axes.get_xaxis().get_label().set_visible(False)
    axs[5].axes.get_yaxis().get_label().set_visible(False)
    axs[5].set_title("Ensemble", fontsize =18)

    axs[0].legend().set_visible(False)
    axs[1].legend().set_visible(False)
    axs[2].legend().set_visible(False)
    axs[3].legend().set_visible(False)
    axs[4].legend().set_visible(False)
    axs[5].legend().set_visible(False)


    fig.suptitle(title, fontsize=20)
    fig.supylabel('number of substitutions', fontsize=20)
    plt.tight_layout()

    outfname = f"{plot_output_dir}/{model}_ensemble_oe.png"
    print(outfname,'created!')
    plt.savefig(outfname)
    plt.close()


# plot_esm_ensemble("ESM1v_mask")
# # plot_esm_ensemble("NetamESM_mask")
# # plot_esm_ensemble("S5FESM_mask")



full_results_df = pd.read_csv(f"{ensemble_output_dir}/combined_performance.csv")

results_df = full_results_df[full_results_df['model'] != 'ESM1v_no-scale']

results_df[['epam_model', 'esm_version']] = results_df['model'].str.split('_', expand=True)

version_colors = {'ensemble': oi_black, '1': oi_orange, '2': oi_skyblue, '3': oi_bluishgreen, '4': oi_yellow, '5': oi_blue}#, 'no-scale': oi_vermillion}

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('white')
# gs = fig.add_gridspec(2, 3)
gs = fig.add_gridspec(1, 3)
axs = gs.subplots(sharex=False, sharey=True)

axs[0].scatter(results_df['sub_accuracy'], results_df['epam_model'], c=results_df['esm_version'].map(version_colors), alpha=1, s=60)
# axs[0].barh(results_df['sub_accuracy'], results_df['epam_model'], color=results_df['esm_version'].map(version_colors), alpha=1)
axs[0].set_title("Substitution accuracy")

axs[1].scatter(results_df['cross_entropy'], results_df['epam_model'], c=results_df['esm_version'].map(version_colors), alpha=1, s=60)
# axs[1].barh(results_df['cross_entropy'], results_df['epam_model'], color=results_df['esm_version'].map(version_colors), alpha=1)
axs[1].set_title("Cross-entropy loss")

axs[2].scatter(results_df['r_precision'], results_df['epam_model'], c=results_df['esm_version'].map(version_colors), alpha=1, s=60)
# axs[2].barh(results_df['r_precision'], results_df['epam_model'], color=results_df['esm_version'].map(version_colors), alpha=1)
axs[2].set_title("R-precision")

# axs[1,0].scatter(results_df['overlap'], results_df['epam_model'], c=results_df['esm_version'].map(version_colors), alpha=0.6, s=60)
# axs[1,0].set_title("Overlap")

# axs[1,1].scatter(results_df['residual'], results_df['epam_model'], c=results_df['esm_version'].map(version_colors), alpha=0.6, s=60)
# axs[1,1].set_title("Residual")

legend_patches = [mpatches.Patch(color=color, label=label) for label, color in version_colors.items()]

axs[2].legend(handles=legend_patches, title='ESM version', loc='center', bbox_to_anchor=(0.75, 0.225))
# axs[2].axis('off')

plt.tight_layout()
plt.savefig("/home/mjohnso4/epam/output/v1_ensemble/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/ensemble_performance.png")
plt.savefig("/home/mjohnso4/epam/output/v1_ensemble/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/ensemble_performance.pdf")
plt.show()
# plt.close()

# fig = plt.figure(figsize=(10,5))
# fig.patch.set_facecolor('white')
# # gs = fig.add_gridspec(2, 3)
# gs = fig.add_gridspec(1, 3)
# axs = gs.subplots(sharex=False, sharey=True)

# for ax, x_col, title in zip(axs, ['cdr_sub_accuracy', 'cdr_cross_entropy', 'cdr_r_precision'], 
#                             ["Substitution accuracy (CDR)", "Cross-entropy loss (CDR)", "R-precision (CDR)"]):
    
#     # Scatter for non-ensemble points
#     ax.scatter(
#         results_df[x_col][results_df['esm_version'] != 'ensemble'],
#         results_df['epam_model'][results_df['esm_version'] != 'ensemble'],
#         c=results_df['esm_version'][results_df['esm_version'] != 'ensemble'].map(version_colors),
#         alpha=1, s=60
#     )

#     # Scatter for ensemble points with special settings
#     ax.scatter(
#         results_df[x_col][results_df['esm_version'] == 'ensemble'],
#         results_df['epam_model'][results_df['esm_version'] == 'ensemble'],
#         c='black',
#         alpha=1, s=120,  # Larger size for 'ensemble'
#         # edgecolor=version_colors['ensemble'], facecolors='none',  # Hollow marker for 'ensemble'
#         marker='x'  # Circle marker for 'ensemble'
#     )

#     ax.set_title(title)

# # The rest of your code, like the legend and layout
# legend_patches = [mpatches.Patch(color=color, label=label) for label, color in version_colors.items()]
# axs[2].legend(handles=legend_patches, title='ESM version', loc='center', bbox_to_anchor=(0.75, 0.225))

# plt.tight_layout()
# plt.savefig("/home/mjohnso4/epam/output/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/ensemble_cdr_performance.png")
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(10,5))
# fig.patch.set_facecolor('white')
# # gs = fig.add_gridspec(2, 3)
# gs = fig.add_gridspec(1, 3)
# axs = gs.subplots(sharex=False, sharey=True)

# for ax, x_col, title in zip(axs, ['fwk_sub_accuracy', 'fwk_cross_entropy', 'fwk_r_precision'], 
#                             ["Substitution accuracy (FWK)", "Cross-entropy loss (FWK)", "R-precision (FWK)"]):
    
#     # Scatter for non-ensemble points
#     ax.scatter(
#         results_df[x_col][results_df['esm_version'] != 'ensemble'],
#         results_df['epam_model'][results_df['esm_version'] != 'ensemble'],
#         c=results_df['esm_version'][results_df['esm_version'] != 'ensemble'].map(version_colors),
#         alpha=1, s=60
#     )

#     # Scatter for ensemble points with special settings
#     ax.scatter(
#         results_df[x_col][results_df['esm_version'] == 'ensemble'],
#         results_df['epam_model'][results_df['esm_version'] == 'ensemble'],
#         c='black',
#         alpha=1, s=120,  # Larger size for 'ensemble'
#         # edgecolor=version_colors['ensemble'], facecolors='none',  # Hollow marker for 'ensemble'
#         marker='x'  # Circle marker for 'ensemble'
#     )

#     ax.set_title(title)

# # The rest of your code, like the legend and layout
# legend_patches = [mpatches.Patch(color=color, label=label) for label, color in version_colors.items()]
# axs[2].legend(handles=legend_patches, title='ESM version', loc='center', bbox_to_anchor=(0.75, 0.225))

# plt.tight_layout()
# plt.savefig("/home/mjohnso4/epam/output/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/ensemble_fwk_performance.png")
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(10,5))
# fig.patch.set_facecolor('white')
# # gs = fig.add_gridspec(2, 3)
# gs = fig.add_gridspec(1, 3)
# axs = gs.subplots(sharex=False, sharey=True)

# for ax, x_col, title in zip(axs, ['sub_accuracy', 'cross_entropy', 'r_precision'], 
#                             ["Substitution accuracy", "Cross-entropy loss", "R-precision"]):
    
#     # Scatter for non-ensemble points
#     ax.scatter(
#         results_df[x_col][results_df['esm_version'] != 'ensemble'],
#         results_df['epam_model'][results_df['esm_version'] != 'ensemble'],
#         c=results_df['esm_version'][results_df['esm_version'] != 'ensemble'].map(version_colors),
#         alpha=1, s=60
#     )

#     # Scatter for ensemble points with special settings
#     ax.scatter(
#         results_df[x_col][results_df['esm_version'] == 'ensemble'],
#         results_df['epam_model'][results_df['esm_version'] == 'ensemble'],
#         c='black',
#         alpha=1, s=120,  # Larger size for 'ensemble'
#         # edgecolor=version_colors['ensemble'], facecolors='none',  # Hollow marker for 'ensemble'
#         marker='x'  # Circle marker for 'ensemble'
#     )

#     ax.set_title(title)

# # The rest of your code, like the legend and layout
# legend_patches = [mpatches.Patch(color=color, label=label) for label, color in version_colors.items()]
# axs[2].legend(handles=legend_patches, title='ESM version', loc='center', bbox_to_anchor=(0.75, 0.225))

# plt.tight_layout()
# plt.savefig("/home/mjohnso4/epam/output/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/ensemble_performance_alt.png")
# plt.show()
# plt.close()