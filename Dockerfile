# syntax=docker/dockerfile:1.5
FROM nvcr.io/nvidia/isaac-lab:2.3.0
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE="/workspace"
ENV OPENARM_ROOT="/workspace/OpenARM-VLA"
ENV OPENARM_DATASET_ROOT="/workspace/OpenARM-VLA/datasets"
ENV DATASET_ROOT="/workspace/OpenARM-VLA/datasets"



RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        tzdata \
        vim \
        libgl1-mesa-dev \
        libglib2.0-0 \
        libsm6 libxrender1 \
        libxext6 \
        ffmpeg \
        libegl1-mesa-dev \
        libglew-dev \
        libosmesa6-dev \
        libglfw3 \
        mesa-utils \
        cmake \
        build-essential \
        libx11-dev \
        libxext-dev \
        patchelf \
        gnupg \
        lsb-release

# Install CUDA toolkit (nvcc) for building mamba-ssm
RUN mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /etc/apt/keyrings/cuda-archive-keyring.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/ /" \
    > /etc/apt/sources.list.d/cuda.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-toolkit-12-8 \
 && rm -rf /var/lib/apt/lists/*


ENV TZ Asia/Tokyo
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV WANDB_MODE=online
ENV WANDB_DISABLED=false



ADD https://astral.sh/uv/install.sh /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
RUN sh /uv-installer.sh && rm /uv-installer.sh


WORKDIR ${OPENARM_ROOT}

COPY . ${OPENARM_ROOT}/

WORKDIR ${OPENARM_ROOT}
ENV UV_PYTHON=/workspace/isaaclab/_isaac_sim/kit/python/bin/python3
ENV PYTHONPATH="${OPENARM_ROOT}:${PYTHONPATH}"
RUN --mount=type=cache,target=/root/.cache/pip \
     ${UV_PYTHON} -m pip install mamba-ssm && \
     uv pip install -e "${OPENARM_ROOT}"

WORKDIR ${OPENARM_ROOT}/openarm_isaac_lab
RUN uv pip install toml \
 && uv pip install -e "${OPENARM_ROOT}/openarm_isaac_lab/source/openarm"


WORKDIR ${WORKSPACE}
