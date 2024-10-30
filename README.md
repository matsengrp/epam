# epam: evaluating predictions of affinity maturation

## Developer install

    conda create -n epam python=3.9
    conda activate epam
    make install
Note: Depending on OS, a conda-forge or mamba install of [datrie](https://github.com/conda-forge/datrie-feedstock) may be required before running `make install`.

The `netam` package needs to be installed within the epam conda environment (though outside the epam file directory). Separately clone the repository (https://github.com/matsengrp/netam) and run `make install`.


Getting GCReplay-DMS data:

    mkdir data/gcreplay
    scp <username>@quokka:/fh/fast/matsen_e/ksung2/epam-gcreplay/final_variant_scores.csv data/gcreplay/
    scp <username>@quokka:/fh/fast/matsen_e/ksung2/epam-gcreplay/chigy_?c_mutation_rates_nt.csv data/gcreplay/

Getting S5F data:

    scp -r <username>@quokka:/fh/fast/matsen_e/ksung2/shmple-gpu/shmple/model_weights/S5F data/


## Terminology

* An "aaprob" matrix is a matrix that expresses the probability of various child amino acid substitutions for each site, laid out with sites on the rows and amino acids on the columns.

## Use

Running it to get matrices:

    epam aaprob NetamSHM '{"model_path_prefix": "/fh/fast/matsen_e/shared/bcr-mut-sel/netam-shm/trained_models/cnn_ind_med-shmoof_small-full-0"}' _ignore/wyatt-10x-1p5m_pcp_2023-09-11.first100.csv _ignore/output.hdf5
