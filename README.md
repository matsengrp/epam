# epam: evaluating predictions of affinity maturation

## Developer install

    conda create -n epam python=3.9
    conda activate epam
    make install
Note: Depending on OS, a conda-forge or mamba install of [datrie](https://github.com/conda-forge/datrie-feedstock) may be required before running `make install`.

To install `netam` in the epam conda env run the following from within the netam repo (separately cloned):

    pip install -e .

Installing on GPU:

    conda create -n epam-gpu python=3.9
    conda activate epam-gpu
    conda install datrie
    make install-cuda

Getting SHMple weights:

    mkdir data/shmple_weights
    scp -r <username>@quokka:/fh/fast/matsen_e/ksung2/shmple-gpu/weights/my_shmoof data/shmple_weights/
    scp -r <username>@quokka:/fh/fast/matsen_e/ksung2/shmple-gpu/weights/prod_shmple data/shmple_weights/

Getting GCReplay-DMS data:

    mkdir data/gcreplay
    scp <username>@quokka:/fh/fast/matsen_e/ksung2/epam-gcreplay/final_variant_scores.csv data/gcreplay/
    scp <username>@quokka:/fh/fast/matsen_e/ksung2/epam-gcreplay/chigy_?c_mutation_rates_nt.csv data/gcreplay/

## Terminology

* An "aaprob" matrix is a matrix that expresses the probability of various child amino acid substitutions for each site, laid out with sites on the rows and amino acids on the columns.

## Use

Running it to get matrices:

    epam aaprob SHMple '{"weights_directory":"data/shmple_weights/my_shmoof"}' _ignore/wyatt-10x-1p5m_pcp_2023-09-11.first100.csv _ignore/output.hdf5
