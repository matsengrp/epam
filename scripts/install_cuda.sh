#!/usr/bin/env bash

# Ensure conda is installed
command -v conda >/dev/null || { echo "Command 'conda' not found. Install Anaconda to install with CUDA support."; exit 1; }

# Install nvidia cuda drivers
#	from https://www.tensorflow.org/install/pip
conda install -y -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163

# Set up system to use these drivers
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Install NVCC
conda install -y -c nvidia cuda-nvcc=11.3.58

# Configure the XLA cuda directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Copy libdevice file to the required path
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/