# Environment for TransMorph-Affine and TransMorph (deformable) on the OrCaScore dataset.
# The code is NOT copied into the image: the repository is mounted at run time (see run.sh),
# so editing the code does not require a rebuild.
FROM python:3.11-slim

# DEBIAN_FRONTEND: apt never stops to ask questions during the build
# PYTHONUNBUFFERED: prints show up immediately in `docker logs` and in the log file
# PIP_NO_CACHE_DIR: no pip download cache kept in the image (smaller image)
# TORCH_HOME: fixed place for the LPIPS/AlexNet weights downloaded below
# MPLCONFIGDIR: writable matplotlib cache, since the container runs as a non-root user
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch_cache \
    MPLCONFIGDIR=/tmp/matplotlib

# git: the experiment.json records the commit of the code that was run
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        nano \
        procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
 && pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# download the LPIPS (AlexNet) weights at build time, so inference needs no internet;
# readable by any user, since containers run with --user (see run.sh)
RUN python -c "import lpips; lpips.LPIPS(net='alex', verbose=False)" \
 && chmod -R a+rX /opt/torch_cache

WORKDIR /workspace
CMD ["bash"]
