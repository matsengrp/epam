# epam: evaluating predictions of affinity maturation

## Developer install

The environment can be set up with [Mamba](https://github.com/mamba-org/mamba). Create the environment and install relevant packages as follows:

    mamba create -n epam python=3.11
    mamba activate epam
    mamba install datrie
    make install

The `netam` package needs to be installed within the epam conda environment (though outside the epam file directory). Separately clone the repository (https://github.com/matsengrp/netam), checkout the relevant version of the repo `git checkout 22c8873`, and run `make install`.

Fetch data for the `thrifty-models` submodule:

    git submodule init
    git submodule update

The "Thrifty-SHM" model in [Johnson et al. (2025)](https://doi.org/10.1101/2025.06.16.659977) is `thrifty-models/models/ThriftyHumV0.2-59`.

To get the model files for "Thrifty-prod", "S5F", and "ReplaySHM + DMS" in [Johnson et al. (2025)](https://doi.org/10.1101/2025.06.16.659977):
- Go to [https://doi.org/10.5281/zenodo.17353498](https://zenodo.org/records/17353498)
- Download `models_setup.tar.gz` to the root directory of the epam repository
- Extract the files: `tar -xzvf models_setup.tar.gz`

|model | files|
|---|---|
| Thrifty-prod | `thrifty-models/models/cnn_ind_lrg-v1wyatt-simple-0.*` |
| S5F | mutability rates: `data/S5F/hh_s5f_muts.csv`</br>substitution rates: `data/S5F/hh_s5f_subs.csv` |
| ReplaySHM + DMS | heavy chain SHM: `data/gcreplay/chigy_hc_mutation_rates_nt.csv`</br>light chain SHM: `data/gcreplay/chigy_lc_mutation_rates_nt.csv`</br>DMS: `data/gcreplay/final_variant_scores.csv` |

Scripts to reproduce figures in [Johnson et al. (2025)](https://doi.org/10.1101/2025.06.16.659977) are found in `notebooks/`.

## Terminology

* An "aaprob" matrix is a matrix that expresses the probability of various child amino acid substitutions for each site, laid out with sites on the rows and amino acids on the columns.

## Use

Running it to get matrices:

    epam aaprob NetamSHM '{"model_path_prefix": "/fh/fast/matsen_e/shared/bcr-mut-sel/netam-shm/trained_models/cnn_ind_med-shmoof_small-full-0"}' _ignore/wyatt-10x-1p5m_pcp_2023-09-11.first100.csv _ignore/output.hdf5

## Parent-child pairs of B cell receptor sequences

Dataframes of parent-child pairs analyzed in [Johnson et al. (2025)](https://doi.org/10.1101/2025.06.16.659977) can be obtained by downloading `pcps.tar.gz` from [https://doi.org/10.5281/zenodo.17353498](https://zenodo.org/records/17353498).