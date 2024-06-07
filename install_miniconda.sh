#!/bin/bash

# Download and install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# Initialize conda
source ~/.bashrc
conda init

# Install and initialize mamba 
conda install -y mamba -c conda-forge
mamba init
source ~/.bashrc
