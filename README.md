# Setup 



Clone the Repo

```
git clone git@github.com:sainavaneet/OpenARM-VLA.git
```

make sure you have the docker setup available with the GPU access


# update submodules

```
cd OpenARM-VLA
git submodule update --init --recursive

```

# Local Setup

python -m pip install -e openarm_isaac_lab/source/openarm


Make sure you Have nvcc available you can skip this if you have `nvcc` is already installed 

```sudo mkdir -p /etc/apt/keyrings \
  && curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/3bf863cc.pub" \
    | gpg --dearmor | sudo tee /etc/apt/keyrings/cuda-archive-keyring.gpg > /dev/null \
  && echo "deb [signed-by=/etc/apt/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/ /" \
    | sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null \
  && sudo apt-get update \
  && sudo apt-get install -y --no-install-recommends cuda-toolkit-12-8
```


install the mamba ssm
python -m pip install --no-cache-dir --force-reinstall --no-deps mamba-ssm --no-build-isolation


# Docker setup 

```bash 
```