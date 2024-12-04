#!/bin/bash

options=$(getopt -o n:,t:,c:,p: -l name:,type:,conda:,llm:,python: -- "$@")
script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
cuda_version=$(nvidia-smi | grep 'CUDA Version' | awk -F" " '{print $9}' | awk -F"." '{print $1}')
cuda_minor=$(nvidia-smi | grep 'CUDA Version' | awk -F" " '{print $9}' | awk -F"." '{print $2}')

eval set -- "$options"

while [ $# -gt 0 ]
do
  case $1 in
    -n|--name) env_name=${2} ; shift ;;
    -t|--type) env_type=${2} ; shift ;;
    -c|--conda) conda_home=${2} ; shift ;;
    -p|--python) python_ver=${2} ; shift ;;
    (--) shift; break ;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1 ;;
    (*) break ;;
    esac
    shift
done

if [ -z "$env_name" ]; then
  env_name="deep_rl_env"
fi

if [ -z "$env_type" ]; then
  env_type="pip"
fi

if [ -z "$conda_home" ]; then
  conda_home="$CONDA_HOME"
fi

if [ -z "$python_ver" ]; then
  python_ver='3.11'
fi

cuda_version=$(nvidia-smi | grep 'CUDA Version' | awk -F" " '{print $9}' | awk -F"." '{print $1}')
cuda_minor=$(nvidia-smi | grep 'CUDA Version' | awk -F" " '{print $9}' | awk -F"." '{print $2}')

if [ "$env_type" = "conda" ]; then

  if ! command -v mamba &> /dev/null; then
    conda install -y conda-forge::mamba
    mamba init
    source "$HOME"/.bashrc
  else
    if [ -z "$CONDA_SHLVL" ]; then
        mamba init
        source "$HOME"/.bashrc
    fi
  fi

  mamba update -y -n base conda
  mamba create -y -n "$env_name" python="$python_ver"
  source "$HOME"/.bashrc
  source "$conda_home"/etc/profile.d/conda.sh
  mamba activate "$env_name"

  mamba install -y -c conda-forge numpy scipy matplotlib pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv
  if [ "$cuda_version" = 11 ] && [ "$cuda_minor" -ge 8 ]; then
    echo "Installing PyTorch for Cuda $cuda_version.$cuda_minor"
    mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  elif [ "$cuda_version" = 12 ] && [ "$cuda_minor" -ge 4 ]; then
    echo "Installing PyTorch for Cuda $cuda_version.$cuda_minor"
#    python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
    mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
  elif [ "$cuda_version" = 12 ] && [ "$cuda_minor" -ge 0 ] && [ "$cuda_minor" -lt 4 ]; then
    echo "Installing PyTorch for Cuda $cuda_version.$cuda_minor"
    mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  else
    echo "Cuda version $cuda_version and minor version $cuda_minor not supported by PyTorch. Installing only CPU support."
    mamba install -y pytorch torchvision torchaudio cpuonly -c pytorch
  fi
  mamba install -y -c conda-forge stable-baselines3 tensorboard wandb gymnasium pygame
  mamba install -y -c huggingface -c conda-forge transformers
  mamba install -y -c huggingface -c conda-forge datasets
  mamba install -y -c conda-forge evaluate accelerate
  python3 -m pip install sentencepiece # vllm

  mamba deactivate

  {
    echo "alias activateLLM=\"conda activate ""$env_name""\""
    echo "alias llms=\"conda activate ""$env_name""; cd ""$script_path""\""
  } >> ~/.bash_aliases

  env_home="$HOME"/miniconda3/envs/"$env_name"

  mkdir -p "$env_home"/etc/conda/activate.d/
  mkdir -p "$env_home"/etc/conda/deactivate.d/
  touch "$env_home"/etc/conda/activate.d/env_vars.sh
  touch "$env_home"/etc/conda/deactivate.d/env_vars.sh
  {
    echo "#!/bin/sh"
    echo "EXTRA_PATH=""$script_path""/legibility/src:""$script_path""/llms/src"
    echo "OLD_PYTHONPATH=\$PYTHONPATH"
    echo "PYTHONPATH=\$EXTRA_PATH:\$PYTHONPATH"
    echo "OLD_PATH=\$PATH"
    echo "OLD_LD_LIBRARY=\$LD_LIBRARY_PATH"
    echo "PATH=/usr/local/cuda/bin:\$PATH"
    echo "LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "export PATH"
    echo "export LD_LIBRARY_PATH"
    echo "export PYTHONPATH"
    echo "export OLD_PYTHONPATH"
    echo "export OLD_PATH"
    echo "export OLD_LD_LIBRARY"
  } >> "$env_home"/etc/conda/activate.d/env_vars.sh

  {
    echo "#!/bin/sh"
    echo "PYTHONPATH=\$OLD_PYTHONPATH"
    echo "LD_LIBRARY_PATH=\$OLD_LD_LIBRARY"
    echo "PATH=\$OLD_PATH"
    echo "unset OLD_PYTHONPATH"
    echo "unset OLD_PATH"
    echo "unset OLD_LD_LIBRARY"
  } >> "$env_home"/etc/conda/deactivate.d/env_vars.sh

else

  mkdir -p ~/python_envs
  python3 -m venv "$HOME"/python_envs/"$env_name"
  source "$HOME/python_envs/$env_name/bin/activate"

  python3 -m pip install --upgrade pip
  python3 -m pip install numpy scipy matplotlib ipython jupyter pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv-python gym pyglet
  python3 -m pip install stable-baselines3 tensorboard wandb gymnasium pygame
  if [ "$cuda_version" = 11 ] && [ "$cuda_minor" -ge 8 ]; then
    echo "Installing PyTorch for Cuda $cuda_version.$cuda_minor"
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  elif [ "$cuda_version" = 12 ] && [ "$cuda_minor" -ge 4 ]; then
    echo "Installing PyTorch for Cuda $cuda_version.$cuda_minor"
    python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
  elif [ "$cuda_version" = 12 ] && [ "$cuda_minor" -ge 0 ] && [ "$cuda_minor" -lt 4 ]; then
    echo "Installing PyTorch for Cuda $cuda_version.$cuda_minor"
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    echo "Cuda version $cuda_version and minor version $cuda_minor not supported by PyTorch. Installing only CPU support."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
  python3 -m pip install transformers datasets evaluate accelerate # vllm

  deactivate

  {
    echo "alias activateLLM=\"source \"\$HOME\"/python_envs/""$env_name""/bin/activate\""
    echo "alias llms=\"source \"\$HOME\"/python_envs/""$env_name""/bin/activate; cd ""$script_path""\""
  } >> ~/.bash_aliases

  {
    echo "EXTRA_PATH=""$script_path""/src"
    echo "PYTHONPATH=\$EXTRA_PATH:\$PYTHONPATH"
    echo "PATH=/usr/local/cuda/bin:\$PATH"
    echo "LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "export PYTHONPATH"
    echo "export PATH"
    echo "export LD_LIBRARY_PATH"
  } >> "$HOME"/python_envs/"$env_name"/bin/activate

fi

source "$HOME"/.bashrc
