import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from epam.models import AbLang2, CachedESM1v
from epam.evaluation import evaluate_dataset
from epam.oe_plot import (
    get_numbering_dict,
    annotate_sites_df,
    get_site_mutabilities_df, 
    plot_sites_observed_vs_expected,
    get_subs_and_preds_from_mutabilities_df, 
    get_subs_and_preds_from_aaprob,
    get_site_substitutions_df, 
    plot_sites_observed_vs_top_k_predictions,
)

example_pcp = "/home/mjohnso4/epam/pcp_inputs/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv"
anarci_path = '/fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v1/anarci/rodriguez-airr-seq-race-prod_anarci-seqs_imgt_H_patch.csv'
ab_sigmoid_aaprobs = "/home/mjohnso4/epam/output/ablang2_sigmoid_aaprobs.hdf5"
ab_ratio_aaprobs = "/home/mjohnso4/epam/output/ablang2_ratio_aaprobs.hdf5"
ab_wt_aaprobs = "//fh/fast/matsen_e/shared/bcr-mut-sel/epam/output/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/AbLang2_wt/combined_aaprob.hdf5"
example_pcp_batch1 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_1.csv"
example_pcp_batch2 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_2.csv"
example_pcp_batch3 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_3.csv"
example_pcp_batch4 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_4.csv"
example_pcp_batch5 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_5.csv"
example_pcp_batch6 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_6.csv"
example_esm_batch1 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_1.hdf5" 
example_esm_batch2 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_2.hdf5" 
example_esm_batch3 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_3.hdf5" 
example_esm_batch4 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_4.hdf5" 
example_esm_batch5 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_5.hdf5" 
example_esm_batch6 = "/fh/fast/matsen_e/shared/bcr-mut-sel/epam/pcp_batched_inputs/v1/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_6.hdf5" 
esm_sigmoid_aaprobs_batch1 = "/home/mjohnso4/epam/output/esm1v_sigmoid_aaprobs_1.hdf5"
esm_ratio_aaprobs_batch1 = "/home/mjohnso4/epam/output/esm1v_ratio_aaprobs_1.hdf5"
esm_sigmoid_aaprobs_batch2 = "/home/mjohnso4/epam/output/esm1v_sigmoid_aaprobs_2.hdf5"
esm_ratio_aaprobs_batch2 = "/home/mjohnso4/epam/output/esm1v_ratio_aaprobs_2.hdf5"
esm_sigmoid_aaprobs_batch3 = "/home/mjohnso4/epam/output/esm1v_sigmoid_aaprobs_3.hdf5"
esm_ratio_aaprobs_batch3 = "/home/mjohnso4/epam/output/esm1v_ratio_aaprobs_3.hdf5"
esm_sigmoid_aaprobs_batch4 = "/home/mjohnso4/epam/output/esm1v_sigmoid_aaprobs_4.hdf5"
esm_ratio_aaprobs_batch4 = "/home/mjohnso4/epam/output/esm1v_ratio_aaprobs_4.hdf5"
esm_sigmoid_aaprobs_batch5 = "/home/mjohnso4/epam/output/esm1v_sigmoid_aaprobs_5.hdf5"
esm_ratio_aaprobs_batch5 = "/home/mjohnso4/epam/output/esm1v_ratio_aaprobs_5.hdf5"
esm_sigmoid_aaprobs_batch6 = "/home/mjohnso4/epam/output/esm1v_sigmoid_aaprobs_6.hdf5"
esm_ratio_aaprobs_batch6 = "/home/mjohnso4/epam/output/esm1v_ratio_aaprobs_6.hdf5"
esm_sigmoid_aaprobs_full = "/home/mjohnso4/epam/output/esm1v_sigmoid_aaprobs_full.hdf5"
esm_ratio_aaprobs_full = "/home/mjohnso4/epam/output/esm1v_ratio_aaprobs_full.hdf5"


# ablang2_sigmoid = AbLang2(masking=True, skip_sigmoid=False)
# ablang2_sigmoid.write_aaprobs(example_pcp, ab_sigmoid_aaprobs)

# ablang2_ratio = AbLang2(masking=True, skip_sigmoid=True)
# ablang2_ratio.write_aaprobs(example_pcp, ab_ratio_aaprobs)

# ab_sigmoid_results = evaluate_dataset(ab_sigmoid_aaprobs)
# print("Sigmoid results:")
# print(ab_sigmoid_results)

# ab_ratio_results = evaluate_dataset(ab_ratio_aaprobs)
# print("Ratio results:")
# print(ab_ratio_results)

# ab_wt_results = evaluate_dataset(ab_wt_aaprobs)
# print("WT results:")
# print(ab_wt_results)

# AbLang2 results:
# Sigmoid results:
# {'data_set': '/home/mjohnso4/epam/pcp_inputs/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv', 'pcp_count': 19550, 'model': 'AbLang2', 'sub_accuracy': 0.3490095564090176, 'r_precision': 0.0998870301725035, 'cross_entropy': 0.17082044914214392, 'fwk_sub_accuracy': 0.4114503494733289, 'fwk_r_precision': 0.13156431469819652, 'fwk_cross_entropy': 0.10661645353428281, 'cdr_sub_accuracy': 0.26196824152127035, 'cdr_r_precision': 0.12638061169617137, 'cdr_cross_entropy': 0.06420399560786122, 'avg_k_subs': 6.246393861892583, 'avg_aa_sub_freq': 0.05149306086254904, 'aa_sub_freq_range': (0.006993006993006993, 0.38461538461538464), 'fwk_avg_k_subs': 3.6371867007672636, 'fwk_avg_aa_sub_freq': 0.03996908462381608, 'fwk_aa_sub_freq_range': (0.0, 0.4175824175824176), 'cdr_avg_k_subs': 2.60920716112532, 'cdr_avg_aa_sub_freq': 0.08729009401013635, 'cdr_aa_sub_freq_range': (0.0, 0.7058823529411765)}
# Ratio results:
# {'data_set': '/home/mjohnso4/epam/pcp_inputs/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv', 'pcp_count': 19550, 'model': 'AbLang2', 'sub_accuracy': 0.3490095564090176, 'r_precision': 0.10149842906108313, 'cross_entropy': 0.1737176070581584, 'fwk_sub_accuracy': 0.4114503494733289, 'fwk_r_precision': 0.1278099978379291, 'fwk_cross_entropy': 0.10739932030623855, 'cdr_sub_accuracy': 0.26196824152127035, 'cdr_r_precision': 0.13187957362100092, 'cdr_cross_entropy': 0.06631828675191989, 'avg_k_subs': 6.246393861892583, 'avg_aa_sub_freq': 0.05149306086254904, 'aa_sub_freq_range': (0.006993006993006993, 0.38461538461538464), 'fwk_avg_k_subs': 3.6371867007672636, 'fwk_avg_aa_sub_freq': 0.03996908462381608, 'fwk_aa_sub_freq_range': (0.0, 0.4175824175824176), 'cdr_avg_k_subs': 2.60920716112532, 'cdr_avg_aa_sub_freq': 0.08729009401013635, 'cdr_aa_sub_freq_range': (0.0, 0.7058823529411765)}
# WT results:
# {'data_set': 'pcp_inputs/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv', 'pcp_count': 19550, 'model': 'AbLang2_wt', 'sub_accuracy': 0.34842814677727096, 'r_precision': 0.09949780465770837, 'cross_entropy': 0.18204319049140008, 'fwk_sub_accuracy': 0.41123939977779966, 'fwk_r_precision': 0.11890749645078344, 'fwk_cross_entropy': 0.11197480453889497, 'cdr_sub_accuracy': 0.2608704175651833, 'cdr_r_precision': 0.13032359322207943, 'cdr_cross_entropy': 0.07006838595250509, 'avg_k_subs': 6.246393861892583, 'avg_aa_sub_freq': 0.05149306086254904, 'aa_sub_freq_range': (0.006993006993006993, 0.38461538461538464), 'fwk_avg_k_subs': 3.6371867007672636, 'fwk_avg_aa_sub_freq': 0.03996908462381608, 'fwk_aa_sub_freq_range': (0.0, 0.4175824175824176), 'cdr_avg_k_subs': 2.60920716112532, 'cdr_avg_aa_sub_freq': 0.08729009401013635, 'cdr_aa_sub_freq_range': (0.0, 0.7058823529411765)}

# esm_sigmoid1 = CachedESM1v(sf_rescale="sigmoid-normalize")
# esm_sigmoid1.preload_esm_data(example_esm_batch1)
# esm_sigmoid1.write_aaprobs(example_pcp_batch1, esm_sigmoid_aaprobs_batch1)

# esm_sigmoid2 = CachedESM1v(sf_rescale="sigmoid-normalize")
# esm_sigmoid2.preload_esm_data(example_esm_batch2)
# esm_sigmoid2.write_aaprobs(example_pcp_batch2, esm_sigmoid_aaprobs_batch2)

# esm_sigmoid3 = CachedESM1v(sf_rescale="sigmoid-normalize")
# esm_sigmoid3.preload_esm_data(example_esm_batch3)
# esm_sigmoid3.write_aaprobs(example_pcp_batch3, esm_sigmoid_aaprobs_batch3)

# esm_sigmoid4 = CachedESM1v(sf_rescale="sigmoid-normalize")
# esm_sigmoid4.preload_esm_data(example_esm_batch4)
# esm_sigmoid4.write_aaprobs(example_pcp_batch4, esm_sigmoid_aaprobs_batch4)

# esm_sigmoid5 = CachedESM1v(sf_rescale="sigmoid-normalize")
# esm_sigmoid5.preload_esm_data(example_esm_batch5)
# esm_sigmoid5.write_aaprobs(example_pcp_batch5, esm_sigmoid_aaprobs_batch5)

# esm_sigmoid6 = CachedESM1v(sf_rescale="sigmoid-normalize")
# esm_sigmoid6.preload_esm_data(example_esm_batch6)
# esm_sigmoid6.write_aaprobs(example_pcp_batch6, esm_sigmoid_aaprobs_batch6)

# esm_ratio1 = CachedESM1v(sf_rescale="ratio-normalize")
# esm_ratio1.preload_esm_data(example_esm_batch1)
# esm_ratio1.write_aaprobs(example_pcp_batch1, esm_ratio_aaprobs_batch1)

# esm_ratio2 = CachedESM1v(sf_rescale="ratio-normalize")
# esm_ratio2.preload_esm_data(example_esm_batch2)
# esm_ratio2.write_aaprobs(example_pcp_batch2, esm_ratio_aaprobs_batch2)

# esm_ratio3 = CachedESM1v(sf_rescale="ratio-normalize")
# esm_ratio3.preload_esm_data(example_esm_batch3)
# esm_ratio3.write_aaprobs(example_pcp_batch3, esm_ratio_aaprobs_batch3)

# esm_ratio4 = CachedESM1v(sf_rescale="ratio-normalize")
# esm_ratio4.preload_esm_data(example_esm_batch4)
# esm_ratio4.write_aaprobs(example_pcp_batch4, esm_ratio_aaprobs_batch4)

# esm_ratio5 = CachedESM1v(sf_rescale="ratio-normalize")
# esm_ratio5.preload_esm_data(example_esm_batch5)
# esm_ratio5.write_aaprobs(example_pcp_batch5, esm_ratio_aaprobs_batch5)

# esm_ratio6 = CachedESM1v(sf_rescale="ratio-normalize")
# esm_ratio6.preload_esm_data(example_esm_batch6)
# esm_ratio6.write_aaprobs(example_pcp_batch6, esm_ratio_aaprobs_batch6)

# esm_sigmoid_results = evaluate_dataset(example_pcp, esm_sigmoid_aaprobs_full)
# print("Sigmoid results:")
# print(esm_sigmoid_results)

# esm_ratio_results = evaluate_dataset(example_pcp, esm_ratio_aaprobs_full)
# print("Ratio results:")
# print(esm_ratio_results)

# ESM1v results:
# Sigmoid results:
# {'data_set': '/home/mjohnso4/epam/pcp_inputs/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv', 'pcp_count': 19550, 'model': 'CachedESM1v', 'sub_accuracy': 0.25080046185215815, 'r_precision': 0.09470542265225076, 'cross_entropy': 0.7139417864445279, 'fwk_sub_accuracy': 0.30447072721391705, 'fwk_r_precision': 0.09984408328490127, 'fwk_cross_entropy': 0.3289032853714518, 'cdr_sub_accuracy': 0.17598510096059597, 'cdr_r_precision': 0.1277572019543434, 'cdr_cross_entropy': 0.3850385008924878, 'avg_k_subs': 6.246393861892583, 'avg_aa_sub_freq': 0.05149306086254904, 'aa_sub_freq_range': (0.006993006993006993, 0.38461538461538464), 'fwk_avg_k_subs': 3.6371867007672636, 'fwk_avg_aa_sub_freq': 0.03996908462381608, 'fwk_aa_sub_freq_range': (0.0, 0.4175824175824176), 'cdr_avg_k_subs': 2.60920716112532, 'cdr_avg_aa_sub_freq': 0.08729009401013635, 'cdr_aa_sub_freq_range': (0.0, 0.7058823529411765)}
# Ratio results:
# {'data_set': '/home/mjohnso4/epam/pcp_inputs/rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv', 'pcp_count': 19550, 'model': 'CachedESM1v', 'sub_accuracy': 0.25080046185215815, 'r_precision': 0.09681557415732235, 'cross_entropy': 0.8483799413630475, 'fwk_sub_accuracy': 0.30447072721391705, 'fwk_r_precision': 0.09725003489169114, 'fwk_cross_entropy': 0.4085531613963162, 'cdr_sub_accuracy': 0.17598510096059597, 'cdr_r_precision': 0.13525878663381496, 'cdr_cross_entropy': 0.4398267797636194, 'avg_k_subs': 6.246393861892583, 'avg_aa_sub_freq': 0.05149306086254904, 'aa_sub_freq_range': (0.006993006993006993, 0.38461538461538464), 'fwk_avg_k_subs': 3.6371867007672636, 'fwk_avg_aa_sub_freq': 0.03996908462381608, 'fwk_aa_sub_freq_range': (0.0, 0.4175824175824176), 'cdr_avg_k_subs': 2.60920716112532, 'cdr_avg_aa_sub_freq': 0.08729009401013635, 'cdr_aa_sub_freq_range': (0.0, 0.7058823529411765)}


# # oe plots
# pcp_df = pd.read_csv(example_pcp, index_col=0)
# numbering, excluded = get_numbering_dict(anarci_path, pcp_df, True, "imgt")

# def plot_sites_oe_for_aaprob(model_aaprob, title, fig_name):
#     model_probs_df = get_site_mutabilities_df(model_aaprob, numbering)
#     fig = plt.figure(figsize=[15,5])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(1, height_ratios=[4])
#     axs = gs.subplots(sharex=True, sharey=False)

#     results = plot_sites_observed_vs_expected(model_probs_df, axs, numbering)
#     axs.text(
#             0.02, 0.95,
#             f'overlap={results["overlap"]:.3g}\nresidual={results["residual"]:.3g}',
#             verticalalignment ='top', 
#             horizontalalignment ='left',
#             transform = axs.transAxes,
#             # fontsize=15
#         )
#     axs.legend(loc='upper right') #, fontsize=15)
#     axs.axes.get_xaxis().get_label().set_visible(False)
#     axs.axes.get_yaxis().get_label().set_visible(False)
#     axs.set_title(title) #, fontsize=15)
#     plt.tight_layout()
#     plt.savefig(f"output/{fig_name}.png")
#     plt.show()
#     plt.close()

# plot_sites_oe_for_aaprob(ab_sigmoid_aaprobs, "AbLang2_mask with sigmoid transformation", "ablang2_sigmoid_sites_oe")
# plot_sites_oe_for_aaprob(ab_ratio_aaprobs, "AbLang2_mask without sigmoid transformation", "ablang2_ratio_sites_oe")
# plot_sites_oe_for_aaprob(ab_wt_aaprobs, "AbLang2_wt", "ablang2_wt_sites_oe")
# plot_sites_oe_for_aaprob(esm_sigmoid_aaprobs_full, "ESM1v_mask with sigmoid transformation", "esm1v_sigmoid_sites_oe")
# plot_sites_oe_for_aaprob(esm_ratio_aaprobs_full, "ESM1v_mask without sigmoid transformation", "esm1v_ratio_sites_oe")
