#!/bin/bash

if [ $# -eq 0 ]; then
  conda_home="$HOME"/miniconda3
else
  conda_home=$1
fi


echo "export CONDA_HOME=""$conda_home" >> "$HOME"/.bashrc
echo "" >> "$HOME"/.bashrc

# Download and install miniconda
mkdir -p "$conda_home"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$conda_home"/miniconda.sh
bash "$conda_home"/miniconda.sh -b -u -p "$conda_home"
rm -rf "$conda_home"/miniconda.sh
source ~/.bashrc

# Initialize conda
"$conda_home"/bin/conda init bash
source ~/.bashrc

# Install and initialize mamba 
conda install -y mamba -c conda-forge
mamba shell init --shell bash --root-prefix="$CONDA_PREFIX"
source ~/.bashrc
