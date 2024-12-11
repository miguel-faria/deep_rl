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
    --llm) use_llms=${2} ; shift ;;
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

if [ -z "$use_llms" ]; then
  use_llms=0
fi

if [ -z "$python_ver" ]; then
  python_ver='3.11'
fi

if [ "$cuda_version" = 12 ]; then
  if [ "$cuda_minor" = 0 ]; then
    jax_version="jax[cuda""$cuda_version""_pip]==0.4.20"
  elif [ "$cuda_minor" = 1 ]; then
    jax_version="jax[cuda""$cuda_version""_pip]==0.4.26"
  else
    jax_version="jax[cuda""$cuda_version""_pip]"
  fi
else
  jax_version="jax[cuda""$cuda_version""_pip]==0.4.10"
fi

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

  if [ "$cuda_version" = 11 ]; then
    mamba install -y -c conda-forge numpy==1.24.3 ml_dtypes==0.1.0 scipy matplotlib pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv
    mamba install -y -c conda-forge stable-baselines3 tensorboard wandb gymnasium pygame
    mamba install -y -c conda-forge optax==0.1.5 flax==0.6.1
  else
    mamba install -y -c conda-forge numpy scipy matplotlib pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv
    mamba install -y -c conda-forge stable-baselines3 tensorboard wandb gymnasium pygame
    mamba install -y -c conda-forge optax flax
  fi

  python3 -m pip install --upgrade "$jax_version" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  python3 -m pip install gym pyglet==1.5.29
  # pip3 install --upgrade nvidia-cudnn-cu11 nvidia-cufft-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvcc-cu11 nvidia-cuda-runtime-cu11

  if [ "$use_llms" -eq 1 ]; then
      mamba install -y -c huggingface -c conda-forge transformers datasets
      mamba install -y -c conda-forge evaluate accelerate
      python3 -m pip install sentencepiece vllm
  fi

  mamba deactivate

  {
    echo "alias activateDRL=\"conda activate ""$env_name""\""
    echo "alias deepRL=\"conda activate ""$env_name""; cd ""$script_path""\""
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
  if [ "$cuda_version" = 11 ]; then
    python3 -m pip install numpy==1.26.4 ml_dtypes==0.1.0 scipy matplotlib ipython jupyter pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv-python gym pyglet==1.5.29
    python3 -m pip install --upgrade "jax[cuda""$cuda_version""_pip]==0.4.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html optax==0.1.7 flax==0.7.0 chex==0.1.82 ml-dtypes==0.2.0
    python3 -m pip install stable-baselines3 tensorboard wandb gymnasium pygame
    python3 -m pip install --upgrade nvidia-cudnn-cu11==8.9.2.26 nvidia-cufft-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvcc-cu11 nvidia-cuda-runtime-cu11
  else
    python3 -m pip install numpy scipy matplotlib ipython jupyter pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv-python gym pyglet==1.5.29
    python3 -m pip install --upgrade "jax[cuda""$cuda_version""_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    python3 -m pip install optax flax
    python3 -m pip install stable-baselines3 tensorboard wandb gymnasium pygame
  fi
#  python3 -m pip install --upgrade nvidia-cudnn-cu11 nvidia-cufft-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvcc-cu11 nvidia-cuda-runtime-cu11

  if [ "$use_llms" -eq 1 ]; then
      python3 -m pip install transformers datasets evaluate accelerate vllm
  fi

  deactivate

  {
    echo "alias activateDRL=\"source \"\$HOME\"/python_envs/""$env_name""/bin/activate\""
    echo "alias deepRL=\"source \"\$HOME\"/python_envs/""$env_name""/bin/activate; cd ""$script_path""\""
  } >> ~/.bash_aliases

  {
    echo "EXTRA_PATH=""$script_path""/legibility/src"
    echo "PYTHONPATH=\$EXTRA_PATH:\$PYTHONPATH"
    echo "PATH=/usr/local/cuda/bin:\$PATH"
    echo "LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "export PYTHONPATH"
    echo "export PATH"
    echo "export LD_LIBRARY_PATH"
  } >> "$HOME"/python_envs/"$env_name"/bin/activate

fi

source "$HOME"/.bashrc
