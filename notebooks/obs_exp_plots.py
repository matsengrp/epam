import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
from epam.evaluation import get_site_mutabilities_df, plot_observed_vs_expected
from epam.utils import load_and_filter_pcp_df
from netam.sequences import aa_mutation_frequency
from epam.utils import SMALL_PROB

# Okabe-Ito colors
oi_black         = '#000000'
oi_orange        = '#E69F00'
oi_skyblue       = '#56B4E9'
oi_bluishgreen   = '#009E73'
oi_yellow        = '#F0E442'
oi_blue          = '#0072B2'
oi_vermillion    = '#D55E00'
oi_reddishpurple = '#CC79A7'

os.chdir("/home/mjohnso4/epam/")
datasets = [
    "flairr", 
    "rodriguez", 
    "tang", 
    "wyatt"
]
full_ds_names = [
    "ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive", 
    "rodriguez-airr-seq-race-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive",
    "tang-deepshm-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive_rearranged",
    "wyatt-10x-1p5m_pcp_2024-04-01_NI_noN_no-naive"
]
models = [
    "ablang1", 
    # "esm",
    # "shmple", 
    # "shmple-prod", 
    # "shmple-esm",
    "ablang2-wt",
    "ablang2-mask"
]
full_model_names = [
    "set1/AbLang1",
    # "set1/ESM1v_mask",
    # "set2/SHMple_default",
    # "set2/SHMple_productive",
    # "set3/SHMpleESM_mask",
    "set1/AbLang2_wt",
    "set1/AbLang2_mask"
]
model_plot_titles = [
    "AbLang1",
    # "ESM1v masked",
    # "SHMple",
    # "SHMple productive",
    # "SHMpleESM",
    "AbLang2 wt",
    "AbLang2 masked"
]

def make_log_plots(model_df, model_name, set_figsize=[10,12], saving=None): #10,8 for two plots
    fig = plt.figure(figsize=set_figsize)
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, height_ratios=[4,5,3])
    axs = gs.subplots(sharex=True, sharey=False)
    metrics = plot_observed_vs_expected(model_df, axs[0], axs[1], axs[2])
    fig.suptitle(f'{model_name}\noverlap={metrics["overlap"]:.3g}, residual={metrics["residual"]:.3g}', fontsize=20)
    plt.tight_layout()
    if saving:
        plt.savefig(f"output/plots/{saving}-log.png")
    plt.show()
    plt.close()
    return metrics

def make_log_norm_plots(model_df, model_name, set_figsize=[10,12], saving=None): #10,8 for two plots
    fig = plt.figure(figsize=set_figsize)
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, height_ratios=[4,5,3])
    axs = gs.subplots(sharex=True, sharey=False)
    metrics = plot_observed_vs_expected(model_df, axs[0], axs[1], axs[2], normalize=True)
    fig.suptitle(f'{model_name}\noverlap={metrics["overlap"]:.3g}, residual={metrics["residual"]:.3g}', fontsize=20)
    plt.tight_layout()
    if saving:
        plt.savefig(f"output/plots/{saving}-log-norm.png")
    plt.show()
    plt.close()
    return metrics

# Plots for all models in all data sets
def get_site_mut_for_all_models_and_datasets():
    for i in range(len(datasets)):
        for j in range(len(models)):
            path_to_aaprobs = f"output/{full_ds_names[i]}/{full_model_names[j]}/combined_aaprob.hdf5"
            print(f"Creating dataframe for {datasets[i]} {models[j]}: {path_to_aaprobs}")
            probs_df = get_site_mutabilities_df(path_to_aaprobs)
            probs_df.to_csv(f"output/plots/mutabilities/{datasets[i]}_{models[j]}_site_mutabilities.csv")

def plot_all_models_and_datasets():
    for i in range(len(datasets)):
        for j in range(len(models)):
            path_to_mut_csv = f"output/plots/mutabilities/{datasets[i]}_{models[j]}_site_mutabilities.csv"
            print(f"Plotting {datasets[i]} {models[j]}: {path_to_mut_csv}")
            probs_df = pd.read_csv(path_to_mut_csv)
            if 0 in probs_df['prob'].values:
                print(f"Replacing {(probs_df['prob'] == 0).sum()} instances of prob=0 with {SMALL_PROB}")
                probs_df.replace(0, SMALL_PROB, inplace=True)
            if 1 in probs_df['prob'].values:
                print(f"Replacing {(probs_df['prob'] == 1).sum()} instances of prob=1 with {1-SMALL_PROB}")
                probs_df.replace(1, 1-SMALL_PROB, inplace=True)
            make_log_plots(probs_df, model_plot_titles[j], saving=f"{datasets[i]}_{models[j]}")
            make_log_norm_plots(probs_df, model_plot_titles[j], saving=f"{datasets[i]}_{models[j]}")

# get_site_mut_for_all_models_and_datasets()
# plot_all_models_and_datasets()


# Plots for AbLang1+2+optimization investigations
            
def get_flairr_w_bl_opt_df(bl_filename):
    path_to_flairr_pcp = f"pcp_inputs/ford-flairr-seq-prod_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv"
    path_to_bl_info = f"output/{full_ds_names[0]}/all_the_ablangs/{bl_filename}.csv"

    flairr_df = load_and_filter_pcp_df(path_to_flairr_pcp)

    ablang_bl_df = pd.read_csv(path_to_bl_info, index_col='pcp_index')
    ablang_bl_df.rename(columns={'parent': 'parent_aa', 'child': 'child_aa'}, inplace=True)

    flairr_bl_df = flairr_df.merge(ablang_bl_df, left_index=True, right_index=True, how="inner")
    flairr_bl_df["aa_sub_freq"] = flairr_bl_df.apply(lambda row: aa_mutation_frequency(row["parent_aa"], row["child_aa"]), axis=1)
    flairr_bl_df["opt_scale_factor"] = np.exp(-flairr_bl_df["opt_branch_length"])

    return flairr_bl_df


def plot_ablang1_bl_opt_results():
    path_to_std_ablang_aaprobs = f"output/{full_ds_names[0]}/all_the_ablangs/ablang1-no_scale-aaprob.hdf5"

    std_ablang_df = get_site_mutabilities_df(path_to_std_ablang_aaprobs)
    make_log_plots(std_ablang_df, f"AbLang1 no scaling ({std_ablang_df['pcp_index'].nunique()} PCPs)", saving=f"{datasets[0]}_{models[0]}1_no_scale")
    make_log_norm_plots(std_ablang_df, f"AbLang1 no scaling ({std_ablang_df['pcp_index'].nunique()} PCPs)", saving=f"{datasets[0]}_{models[0]}1_no_scale")

    for opt_version in ["v1", "v2"]:
        opt_ablang_df = get_site_mutabilities_df(f"output/{full_ds_names[0]}/all_the_ablangs/ablang1-opt-{opt_version}-aaprob.hdf5")
        if opt_version == "v1":
            flairr_bl_df = get_flairr_w_bl_opt_df("AbLang1-opt_v1_branch_opt_fails_1716484214")
        else:
            flairr_bl_df = get_flairr_w_bl_opt_df("AbLang1-opt_v2_branch_opt_fails_1716484574")
        
        pcp_converge = flairr_bl_df[flairr_bl_df["fail_to_converge"] == False].index.tolist()
        pcp_no_converge = flairr_bl_df[flairr_bl_df["fail_to_converge"] == True].index.tolist()

        pcp_has_subs = flairr_bl_df[flairr_bl_df["aa_sub_freq"] > 0].index.tolist()
        pcp_no_subs = flairr_bl_df[flairr_bl_df["aa_sub_freq"] == 0].index.tolist()

        opt_converged_site_df = opt_ablang_df[opt_ablang_df['pcp_index'].isin(pcp_converge)].copy()
        opt_no_converged_site_df = opt_ablang_df[opt_ablang_df['pcp_index'].isin(pcp_no_converge)].copy()

        opt_pcp_subs_df = opt_ablang_df[opt_ablang_df['pcp_index'].isin(pcp_has_subs)].copy()   
        opt_pcp_no_subs_df = opt_ablang_df[opt_ablang_df['pcp_index'].isin(pcp_no_subs)].copy()
        
        make_log_plots(opt_ablang_df, f"AbLang1 w/ {opt_version} ({opt_ablang_df['pcp_index'].nunique()} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}")
        make_log_plots(opt_converged_site_df, f"AbLang1 w/ {opt_version} converged ({len(pcp_converge)} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}_converged")
        make_log_plots(opt_no_converged_site_df, f"AbLang1 w/ {opt_version} did not converge ({len(pcp_no_converge)} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}_no_converge")
        make_log_plots(opt_pcp_subs_df, f"AbLang1 w/ {opt_version} PCPs with AA substitutions ({len(pcp_has_subs)} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}_pcp_subs")
        make_log_plots(opt_pcp_no_subs_df, f"AbLang1 w/ {opt_version} PCPs with no AA substitutions ({len(pcp_no_subs)} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}_pcp_no_subs")

        make_log_norm_plots(opt_ablang_df, f"AbLang1 w/ {opt_version} ({opt_ablang_df['pcp_index'].nunique()} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}")
        make_log_norm_plots(opt_converged_site_df, f"AbLang1 w/ {opt_version} converged ({len(pcp_converge)} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}_converged")
        make_log_norm_plots(opt_no_converged_site_df, f"AbLang1 w/ {opt_version} did not converge ({len(pcp_no_converge)} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}_no_converge")
        make_log_norm_plots(opt_pcp_subs_df, f"AbLang1 w/ {opt_version} PCPs with AA substitutions ({len(pcp_has_subs)} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}_pcp_subs")
        make_log_norm_plots(opt_pcp_no_subs_df, f"AbLang1 w/ {opt_version} PCPs with no AA substitutions ({len(pcp_no_subs)} PCPs)", saving=f"{datasets[0]}_{models[0]}1_{opt_version}_pcp_no_subs")

# plot_ablang1_bl_opt_results()
        
def plot_ablang2_bl_opt_results():
    for score_stg in {'wt', 'mask'}:
        for opt_stg in {'opt', 'no-opt'}:
            path_to_aaprobs = f"output/{full_ds_names[0]}/all_the_ablangs/ablang2-{score_stg}-{opt_stg}-aaprob.hdf5"
            ablang2_df = get_site_mutabilities_df(path_to_aaprobs)
            if opt_stg == 'no-opt':
                # make_log_plots(ablang2_df, f"AbLang2 {score_stg} w/o optimization ({ablang2_df['pcp_index'].nunique()} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}")
                # make_log_norm_plots(ablang2_df, f"AbLang2 {score_stg} w/o optimization ({ablang2_df['pcp_index'].nunique()} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}")
                pass
            else:
                # make_log_plots(ablang2_df, f"AbLang2 {score_stg} w/ optimization ({ablang2_df['pcp_index'].nunique()} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}")
                # make_log_norm_plots(ablang2_df, f"AbLang2 {score_stg} w/ optimization ({ablang2_df['pcp_index'].nunique()} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}")
                if score_stg == 'wt':
                    flairr_bl_df = get_flairr_w_bl_opt_df("AbLang2-wt-opt_branch_opt_fails_1716485980")
                elif score_stg == 'mask':
                    flairr_bl_df = get_flairr_w_bl_opt_df("AbLang2-mask-opt_branch_opt_fails_1716486679")
                else:
                    raise ValueError("Invalid score stage")
                
                # pcp_converge = flairr_bl_df[flairr_bl_df["fail_to_converge"] == False].index.tolist()
                # pcp_no_converge = flairr_bl_df[flairr_bl_df["fail_to_converge"] == True].index.tolist()
                # opt_converged_site_df = ablang2_df[ablang2_df['pcp_index'].isin(pcp_converge)].copy()
                # opt_no_converged_site_df = ablang2_df[ablang2_df['pcp_index'].isin(pcp_no_converge)].copy()
                # make_log_plots(opt_converged_site_df, f"AbLang2 {score_stg} w/ optimization converged ({len(pcp_converge)} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}_converged")
                # make_log_plots(opt_no_converged_site_df, f"AbLang2 {score_stg} w/ optimization did not converge ({len(pcp_no_converge)} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}_no_converge")
                # make_log_norm_plots(opt_converged_site_df, f"AbLang2 {score_stg} w/ optimization converged ({len(pcp_converge)} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}_converged")
                # make_log_norm_plots(opt_no_converged_site_df, f"AbLang2 {score_stg} w/ optimization did not converge ({len(pcp_no_converge)} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}_no_converge")

                pcp_has_subs = flairr_bl_df[flairr_bl_df["aa_sub_freq"] > 0].index.tolist()
                pcp_no_subs = flairr_bl_df[flairr_bl_df["aa_sub_freq"] == 0].index.tolist()
                opt_pcp_subs_df = ablang2_df[ablang2_df['pcp_index'].isin(pcp_has_subs)].copy()   
                opt_pcp_no_subs_df = ablang2_df[ablang2_df['pcp_index'].isin(pcp_no_subs)].copy()
                make_log_plots(opt_pcp_subs_df, f"AbLang2 {score_stg} w/ optimization with AA substitutions ({len(pcp_has_subs)} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}_pcp_subs")
                make_log_plots(opt_pcp_no_subs_df, f"AbLang2 {score_stg} w/ optimization with no AA substitutions ({len(pcp_no_subs)} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}_pcp_no_subs")
                make_log_norm_plots(opt_pcp_subs_df, f"AbLang2 {score_stg} w/ optimization with AA substitutions ({len(pcp_has_subs)} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}_pcp_subs")
                make_log_norm_plots(opt_pcp_no_subs_df, f"AbLang2 {score_stg} w/ optimization with no AA substitutions ({len(pcp_no_subs)} PCPs)", saving=f"{datasets[0]}_{models[0]}2_{score_stg}_{opt_stg}_pcp_no_subs")

# plot_ablang2_bl_opt_results()

# Plots for AA sub frequencies in pcps
def plot_flairr_pcp_aa_sub_freqs():
    flairr_bl_df = get_flairr_w_bl_opt_df()

    pcp_fail_to_converge = flairr_bl_df[flairr_bl_df["fail_to_converge"] == True]
    pcp_converges = flairr_bl_df[flairr_bl_df["fail_to_converge"] == False]
    
    fig = plt.figure(figsize=(10,8))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, height_ratios=[4,4])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].hist(pcp_converges["opt_branch_length"], bins=20, color=oi_reddishpurple, alpha=0.5, label="Converged")
    if len(pcp_fail_to_converge) > 0:
        axs[0].hist(pcp_fail_to_converge["opt_branch_length"], bins=20, color=oi_black, alpha=0.5, label="Failed to converge")
    axs[0].set_ylabel("Count", fontsize=16)
    axs[0].set_title(f"FLAIRR pcp = {flairr_bl_df.shape[0]}", fontsize=20)
    axs[0].legend(loc="upper right")

    axs[1].scatter(pcp_converges["opt_branch_length"], pcp_converges["aa_sub_freq"], color=oi_reddishpurple, alpha=0.2, label="Converged")
    if len(pcp_fail_to_converge) > 0:
        axs[1].scatter(pcp_fail_to_converge["opt_branch_length"], pcp_fail_to_converge["aa_sub_freq"], color=oi_black, alpha=0.2, label="Failed to converge")
    axs[1].set_xlabel("Optimized $t$", fontsize=16)
    axs[1].set_ylabel("AA mutation frequency", fontsize=16)
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("output/plots/flairr_pcp_aa_sub_freqs.png")
    plt.show()

# plot_flairr_pcp_aa_sub_freqs()

# Compare performance metrics across data sets and models
def compare_performance_metrics():
    flairr_df = pd.read_csv(f"output/{full_ds_names[0]}/combined_performance.csv")
    flairr_df['data'] = 'flairr'
    rodriguez_df = pd.read_csv(f"output/{full_ds_names[1]}/combined_performance.csv")
    rodriguez_df['data'] = 'race'
    tang_df = pd.read_csv(f"output/{full_ds_names[2]}/combined_performance.csv")
    tang_df['data'] = 'tang'
    tang_df.replace(np.inf, 0.18377580373657282, inplace=True)
    wyatt_df = pd.read_csv(f"output/{full_ds_names[3]}/combined_performance.csv")
    wyatt_df['data'] = 'wyatt'

    data = [
        {'data': 'flairr', 'model': 'AbLang_heavy', 'overlap': 0.5632508414022324, 'residual': 0.15963287194438633},
        {'data': 'flairr', 'model': 'ESM1v_mask', 'overlap': 0.23830103973985003, 'residual': 0.34077190806773844},
        {'data': 'flairr', 'model': 'SHMple_default', 'overlap': 0.9361792375174238, 'residual': 0.023987510753383005},
        {'data': 'flairr', 'model': 'SHMple_productive', 'overlap': 0.9496930788064567, 'residual': 0.020717801893052654},
        {'data': 'flairr', 'model': 'SHMpleESM_mask', 'overlap': 0.308427162681062, 'residual': 0.8130260040447933},
        {'data': 'race', 'model': 'AbLang_heavy', 'overlap': 0.5855264488739582, 'residual': 0.15338151242914316},
        {'data': 'race', 'model': 'ESM1v_mask', 'overlap': 0.23778939182595193, 'residual': 0.34701348309761776},
        {'data': 'race', 'model': 'SHMple_default', 'overlap': 0.9535692302879906, 'residual': 0.017277555204613406},
        {'data': 'race', 'model': 'SHMple_productive', 'overlap': 0.9542831877292881, 'residual': 0.019024448182531132},
        {'data': 'race', 'model': 'SHMpleESM_mask', 'overlap': 0.3107049443023042, 'residual': 0.8277329824745648},
        {'data': 'tang', 'model': 'AbLang_heavy', 'overlap': 0.5531928786886899, 'residual': 0.16206051117975095},
        {'data': 'tang', 'model': 'ESM1v_mask', 'overlap': 0.1989942991425714, 'residual': 0.34545743339506413},
        {'data': 'tang', 'model': 'SHMple_default', 'overlap': 0.9389008478745828, 'residual': 0.022953614147611375},
        {'data': 'tang', 'model': 'SHMple_productive', 'overlap': 0.9398100842955867, 'residual': 0.0232494854857495},
        {'data': 'tang', 'model': 'SHMpleESM_mask', 'overlap': 0.2937661702763633, 'residual': 0.8791778179472269},
        {'data': 'wyatt', 'model': 'AbLang_heavy', 'overlap': 0.5366759614838402, 'residual': 0.17110300024189803},
        {'data': 'wyatt', 'model': 'ESM1v_mask', 'overlap': 0.24232196582625856, 'residual': 0.33434188453720953},
        {'data': 'wyatt', 'model': 'SHMple_default', 'overlap': 0.9500925648903427, 'residual': 0.022259980595459184},
        {'data': 'wyatt', 'model': 'SHMple_productive', 'overlap': 0.9448893960377275, 'residual': 0.02420027794589979},
        {'data': 'wyatt', 'model': 'SHMpleESM_mask', 'overlap': 0.29054152216586426, 'residual': 0.9268475273740655}
    ]

    obs_exp_metrics_df = pd.DataFrame(data)

    full_perf_df = pd.concat([flairr_df, rodriguez_df, tang_df, wyatt_df], ignore_index=True)

    all_perf_metrics_df = full_perf_df.merge(obs_exp_metrics_df, on=['data', 'model'], how='inner')

    fig = plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 3)
    axs = gs.subplots(sharex=False, sharey=True)

    data_colors = {'flairr': oi_black, 'race': oi_reddishpurple, 'tang': oi_skyblue, 'wyatt': oi_bluishgreen}

    axs[0,0].scatter(all_perf_metrics_df['sub_accuracy'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[0,0].set_title("Substitution accuracy")

    axs[0,1].scatter(all_perf_metrics_df['cross_entropy'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[0,1].set_title("Cross-entropy loss")

    axs[0,2].scatter(all_perf_metrics_df['r_precision'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[0,2].set_title("R-precision")

    axs[1,0].scatter(all_perf_metrics_df['overlap'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[1,0].set_title("Overlap")

    axs[1,1].scatter(all_perf_metrics_df['residual'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[1,1].set_title("Residual")

    # fig.delaxes(axs[1,2])

    # Create legend patches
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in data_colors.items()]

    axs[1,2].legend(handles=legend_patches, title='dataset', loc='center', bbox_to_anchor=(0.5, 0.5))
    axs[1,2].axis('off')

    plt.tight_layout()
    plt.savefig("output/plots/overall_performance.png")
    plt.show()
    plt.close()

    # print(tang_df[['sub_accuracy','r_precision','cross_entropy']])
    # print(flairr_df[['sub_accuracy','r_precision','cross_entropy']])
    # print(rodriguez_df[['sub_accuracy','r_precision','cross_entropy']])
    # print(wyatt_df[['sub_accuracy','r_precision','cross_entropy']])
    
# compare_performance_metrics()
    
def plot_ablang1_opt_v_std():
    path_to_comparison_eval = f"output/{full_ds_names[0]}/ablang1-comp-aaprob-eval.csv"
    comp_eval_df = pd.read_csv(path_to_comparison_eval)

    fig = plt.figure(figsize=(7,4))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 3)
    axs = gs.subplots(sharex=False, sharey=True)

    axs[0].barh(comp_eval_df['model'], comp_eval_df['cross_entropy'], color=oi_black, alpha=0.6)
    axs[0].set_title("Cross-entropy loss")

    axs[1].barh(comp_eval_df['model'], comp_eval_df['r_precision'], color=oi_black, alpha=0.6)
    axs[1].set_title("R-precision")

    axs[2].barh(comp_eval_df['model'], comp_eval_df['sub_accuracy'], color=oi_black, alpha=0.6)
    axs[2].set_title("Substitution accuracy")

    plt.tight_layout()
    plt.savefig("output/plots/ablang1_opt_v_std_performance.png")
    plt.show()
    plt.close()

# plot_ablang1_opt_v_std()
    
def plot_ablangs_v_esm():
    path_to_comparison_eval = f"output/{full_ds_names[0]}/ablang1-2-comp-aaprob-eval.csv"
    comp_eval_df = pd.read_csv(path_to_comparison_eval)

    fig = plt.figure(figsize=(7,4))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 3)
    axs = gs.subplots(sharex=False, sharey=True)

    axs[0].barh(comp_eval_df['model'], comp_eval_df['cross_entropy'], color=oi_black, alpha=0.6)
    axs[0].set_title("Cross-entropy loss")

    axs[1].barh(comp_eval_df['model'], comp_eval_df['r_precision'], color=oi_black, alpha=0.6)
    axs[1].set_title("R-precision")

    axs[2].barh(comp_eval_df['model'], comp_eval_df['sub_accuracy'], color=oi_black, alpha=0.6)
    axs[2].set_title("Substitution accuracy")

    plt.tight_layout()
    plt.savefig("output/plots/ablang1_2_esm_performance.png")
    plt.show()
    plt.close()

# plot_ablangs_v_esm()


# Plot updated SHMpleESM FLAIRR results
def get_abstract_flairr():
    i = 0
    path_to_aaprobs = f"output/{full_ds_names[i]}/abstract_SHMpleESM_mask.hdf5"
    print(f"Creating dataframe for {datasets[i]} abstract SHMpleESM: {path_to_aaprobs}")
    probs_df = get_site_mutabilities_df(path_to_aaprobs)
    print(f"Plotting")
    if 0 in probs_df['prob'].values:
        print(f"Replacing {(probs_df['prob'] == 0).sum()} instances of prob=0 with {SMALL_PROB}")
        probs_df.replace(0, SMALL_PROB, inplace=True)
    if 1 in probs_df['prob'].values:
        print(f"Replacing {(probs_df['prob'] == 1).sum()} instances of prob=1 with {1-SMALL_PROB}")
        probs_df.replace(1, 1-SMALL_PROB, inplace=True)
    make_log_plots(probs_df, model_plot_titles[4], saving=f"{datasets[i]}_abstractSHMpleESM")
    make_log_norm_plots(probs_df, model_plot_titles[4], saving=f"{datasets[i]}_abstractSHMpleESM")

# get_abstract_flairr()
    

# Compare AbLang performance metrics across data sets
def compare_ablang_perf():
    flairr_df = pd.read_csv(f"output/{full_ds_names[0]}/combined_performance.csv")
    flairr_df['data'] = 'flairr'
    flairr_shm_prod = flairr_df[flairr_df['model'] == 'SHMple_productive']
    flairr_ab_df = pd.read_csv(f"output/{full_ds_names[0]}/combined_performance_ablangs.csv")
    flairr_ab_df['data'] = 'flairr'
    rodriguez_df = pd.read_csv(f"output/{full_ds_names[1]}/combined_performance.csv")
    rodriguez_df['data'] = 'race'
    rodriguez_shm_prod = rodriguez_df[rodriguez_df['model'] == 'SHMple_productive']
    rodriguez_ab_df = pd.read_csv(f"output/{full_ds_names[1]}/combined_performance_ablangs.csv")
    rodriguez_ab_df['data'] = 'race'
    tang_df = pd.read_csv(f"output/{full_ds_names[2]}/combined_performance.csv")
    tang_df['data'] = 'tang'
    tang_shm_prod = tang_df[tang_df['model'] == 'SHMple_productive']
    tang_ab_df = pd.read_csv(f"output/{full_ds_names[2]}/combined_performance_ablangs.csv")
    tang_ab_df['data'] = 'tang'
    wyatt_df = pd.read_csv(f"output/{full_ds_names[3]}/combined_performance.csv")
    wyatt_df['data'] = 'wyatt'
    wyatt_shm_prod = wyatt_df[wyatt_df['model'] == 'SHMple_productive']
    wyatt_ab_df = pd.read_csv(f"output/{full_ds_names[3]}/combined_performance_ablangs.csv")
    wyatt_ab_df['data'] = 'wyatt'

    data = [
        {'data': 'flairr', 'model': 'AbLang1', 'overlap': 0.565, 'residual': 0.159},
        {'data': 'flairr', 'model': 'AbLang2_wt', 'overlap': 0.772, 'residual': 0.098},
        {'data': 'flairr', 'model': 'AbLang2_mask', 'overlap': 0.927, 'residual': 0.0328},
        {'data': 'flairr', 'model': 'SHMple_productive', 'overlap': 0.9497, 'residual': 0.02072},
        {'data': 'race', 'model': 'AbLang1', 'overlap': 0.587, 'residual': 0.153},
        {'data': 'race', 'model': 'AbLang2_wt', 'overlap': 0.789, 'residual': 0.0939},
        {'data': 'race', 'model': 'AbLang2_mask', 'overlap': 0.927, 'residual': 0.0344},
        {'data': 'race', 'model': 'SHMple_productive', 'overlap': 0.954, 'residual': 0.01902},
        {'data': 'tang', 'model': 'AbLang1', 'overlap': 0.554, 'residual': 0.161},
        {'data': 'tang', 'model': 'AbLang2_wt', 'overlap': 0.796, 'residual': 0.0873},
        {'data': 'tang', 'model': 'AbLang2_mask', 'overlap': 0.922, 'residual': 0.036},
        {'data': 'tang', 'model': 'SHMple_productive', 'overlap': 0.9398, 'residual': 0.02325},
        {'data': 'wyatt', 'model': 'AbLang1', 'overlap': 0.537, 'residual': 0.17},
        {'data': 'wyatt', 'model': 'AbLang2_wt', 'overlap': 0.772, 'residual': 0.098},
        {'data': 'wyatt', 'model': 'AbLang2_mask', 'overlap': 0.807, 'residual': 0.0862},
        {'data': 'wyatt', 'model': 'SHMple_productive', 'overlap': 0.903, 'residual': 0.044}
    ]

    obs_exp_metrics_df = pd.DataFrame(data)

    full_perf_df = pd.concat(
        [
            flairr_shm_prod, rodriguez_shm_prod, tang_shm_prod, wyatt_shm_prod,
            flairr_ab_df, rodriguez_ab_df, tang_ab_df, wyatt_ab_df
        ], 
        ignore_index=True
    )

    all_perf_metrics_df = full_perf_df.merge(obs_exp_metrics_df, on=['data', 'model'], how='inner')

    fig = plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 3)
    axs = gs.subplots(sharex=False, sharey=True)

    data_colors = {'flairr': oi_black, 'race': oi_reddishpurple, 'tang': oi_skyblue, 'wyatt': oi_bluishgreen}

    axs[0,0].scatter(all_perf_metrics_df['sub_accuracy'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[0,0].set_title("Substitution accuracy")

    axs[0,1].scatter(all_perf_metrics_df['cross_entropy'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[0,1].set_title("Cross-entropy loss")

    axs[0,2].scatter(all_perf_metrics_df['r_precision'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[0,2].set_title("R-precision")

    axs[1,0].scatter(all_perf_metrics_df['overlap'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[1,0].set_title("Overlap")

    axs[1,1].scatter(all_perf_metrics_df['residual'], all_perf_metrics_df['model'], c=all_perf_metrics_df['data'].map(data_colors), alpha=0.6, s=60)
    axs[1,1].set_title("Residual")

    # fig.delaxes(axs[1,2])

    # Create legend patches
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in data_colors.items()]

    axs[1,2].legend(handles=legend_patches, title='dataset', loc='center', bbox_to_anchor=(0.5, 0.5))
    axs[1,2].axis('off')

    plt.tight_layout()
    plt.savefig("output/plots/all_ablang_performance.png")
    plt.show()
    plt.close()

compare_ablang_perf()