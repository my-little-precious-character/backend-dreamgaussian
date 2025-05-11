# Dockerfile for DreamGaussian with GPU support

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic dependencies
RUN apt-get update && apt-get install -y \
    git python3 python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 \
    build-essential cmake curl unzip libegl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python packages
RUN pip3 install torch-ema einops tensorboardX plyfile dearpygui huggingface_hub \
    diffusers accelerate transformers xatlas trimesh \
    PyMCubes pymeshlab rembg[gpu,cli] omegaconf ninja torchvision scikit-learn uv matplotlib

# Clone DreamGaussian
RUN mkdir -p /app && \
    git clone https://github.com/dreamgaussian/dreamgaussian.git /app/dreamgaussian
WORKDIR /app/dreamgaussian

ENV TORCH_CUDA_ARCH_LIST="8.9"

# diff-gaussian-rasterization 설치 (wheels)
RUN git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
RUN pip install ./diff-gaussian-rasterization
RUN pip install ./simple-knn

# for text_mv.yaml (mvDream)
Run pip install git+https://github.com/bytedance/MVDream.git@main

# Pre-built rasterizer wheels
# RUN pip3 install \  
#     https://github.com/camenduru/diff-gaussian-rasterization/releases/download/v1.0/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.1.whl \
#     https://github.com/camenduru/diff-gaussian-rasterization/releases/download/v1.0/simple_knn-0.0.0-cp310-cp310-linux_x86_64.1.whl

# nvdiffrast
RUN pip3 install git+https://github.com/NVlabs/nvdiffrast

# kiuikit
RUN pip3 install git+https://github.com/ashawkey/kiuikit

# Create a data directory
RUN mkdir -p /app/dreamgaussian/data
RUN mkdir -p /app/backend-dreamgaussian

COPY . /app/backend-dreamgaussian/
WORKDIR /app/backend-dreamgaussian

# mvdream 파일 덮어쓰기
RUN cp -f ./mvdream_utils.py /app/dreamgaussian/guidance/mvdream_utils.py

# install dependencies
RUN uv pip install -r pyproject.toml --system

# use port 8000
EXPOSE 8000


# Default entrypoint (change this to actual DreamGaussian script as needed)
CMD ["bash"]
