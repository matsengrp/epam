# epam
Evaluating predictions of affinity maturation

## Developer install

```
conda create -n epam python=3.9
conda activate epam
make install
```

Clone the SHMple repo to any location of your choice and install:
```
git clone https://github.com/matsengrp/shmple
git checkout d8902851bfdde428d8f03ca62dfe90e57c9a6e14
cd shmple
make install
```
(you can do `make install-cuda` instead, if GPU is available)

```
mkdir data/shmple_weights
scp -r <username>@quokka:/fh/fast/matsen_e/ksung2/shmple-gpu$ ls /fh/fast/matsen_e/ksung2/shmple-gpu/weights/my_shmoof data/shmple_weights/
```
