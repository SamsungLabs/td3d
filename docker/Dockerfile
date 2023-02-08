ARG PYTORCH="1.9.1"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
RUN pip install mmdet==2.24.1
RUN pip install mmsegmentation==0.24.1

# Install albumentation
RUN pip install albumentations==1.3.0

# Install MMDetection3D
RUN git clone https://github.com/samsunglabs/td3d.git /mmdetection3d
WORKDIR /mmdetection3d
RUN pip install --no-cache-dir -e .

# Install MinkowskiEngine
RUN apt-get install -y libopenblas-dev
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine@405b39cb7e68c2ec1b4ac8c4b4ca9fa6fcfcb2fc -v --no-deps \
    --install-option="--blas=openblas" \
    --install-option="--force_cuda"
