#!/bin/bash

source /home/mjohnso4/miniforge3/etc/profile.d/conda.sh
conda activate anarci 

# Run ANARCI on epam PCP files
# IMGT numbering
# Heavy chains
ANARCI -i /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/ford-flairr-seq-prod_anarci-seqs.fasta -o /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/ford-flairr-seq-prod_anarci-seqs_imgt -s imgt -r heavy --use_species human --csv -p 5 --assign_germline
ANARCI -i /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/rodriguez-airr-seq-race-prod_anarci-seqs.fasta -o /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/rodriguez-airr-seq-race-prod_anarci-seqs_imgt -s imgt -r heavy --use_species human --csv -p 5 --assign_germline
ANARCI -i /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/tang-deepshm-prod_anarci-seqs.fasta -o /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/tang-deepshm-prod_anarci-seqs_imgt -s imgt -r heavy --use_species human --csv -p 5 --assign_germline
ANARCI -i /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/wyatt-10x-1p5m_paired-igh_fs-all_anarci-seqs.fasta -o /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/wyatt-10x-1p5m_paired-igh_fs-all_anarci-seqs_imgt -s imgt -r heavy --use_species human --csv -p 5 --assign_germline

# Light chains
ANARCI -i /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/wyatt-10x-1p5m_paired-igk_fs-all_anarci-seqs.fasta -o /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/wyatt-10x-1p5m_paired-igk_fs-all_anarci-seqs_imgt -s imgt -r light --use_species human --csv -p 5 --assign_germline
ANARCI -i /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/wyatt-10x-1p5m_paired-igl_fs-all_anarci-seqs.fasta -o /fh/fast/matsen_e/shared/bcr-mut-sel/pcps/v2/anarci/wyatt-10x-1p5m_paired-igl_fs-all_anarci-seqs_imgt -s imgt -r light --use_species human --csv -p 5 --assign_germline