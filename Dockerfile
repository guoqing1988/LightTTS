ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04
ARG PYTHON_VERSION=3.10
ARG MAMBA_VERSION=24.7.1-0
ARG TARGETPLATFORM
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

WORKDIR /root

RUN chmod 777 -R /tmp && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    g++ \
    make \
    git \
    ffmpeg \
    unzip && \
    rm -rf /var/lib/apt/lists/*

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
    *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -o ~/mambaforge.sh "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh" && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

    RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

COPY ./requirements.txt /lighttts/requirements.txt
RUN pip install -U pip
RUN pip install -r /lighttts/requirements.txt --no-cache-dir

COPY . /lighttts
WORKDIR /lighttts
RUN cd pretrained_models/CosyVoice-ttsfrd/ && \
    unzip resource.zip -d . && \
    pip install ttsfrd_dependency-0.1-py3-none-any.whl && \
    pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
