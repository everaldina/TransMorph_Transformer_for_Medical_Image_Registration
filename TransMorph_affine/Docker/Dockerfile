# Install any python packages you need
FROM python:3.11-slim

# Evita interações durante instalação de pacotes
ENV DEBIAN_FRONTEND=noninteractive

# Atualiza pacotes do sistema e instala dependências do PyTorch e utilitários
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    ffmpeg \
    libsm6 \
    libxext6 \
    nano \
    mc \
    dos2unix \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Diretório da aplicação
WORKDIR /repositorio

RUN pip install --upgrade pip \
 && pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["bash"]