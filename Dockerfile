# Use Ubuntu as base image
FROM ubuntu:20.04

# Avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    libprotobuf-dev \
    protobuf-compiler \
    libprotoc-dev \
    sentencepiece \
    libsentencepiece-dev \
    gdb \
    file \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow
RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.14.1.tar.gz && \
    tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.14.1.tar.gz && \
    rm libtensorflow-cpu-linux-x86_64-2.14.1.tar.gz && \
    ldconfig

# Set working directory
WORKDIR /app

# Copy your project files
COPY . .

# Create build directory and build project
RUN rm -rf build && mkdir -p build && \
    cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_SENTENCEPIECE=OFF \
        -DTENSORFLOW_ROOT=/usr/local && \
    make -j$(nproc) VERBOSE=1

# Set environment variable for library path
ENV LD_LIBRARY_PATH=/usr/local/lib

# Create a directory for models
RUN mkdir -p /app/models

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$#" -ne 2 ]; then\n\
    echo "Usage: $0 <tf_model_path> <spiece_model_path>"\n\
    exit 1\n\
fi\n\
\n\
echo "Running with paths:"\n\
echo "TF Model: $1"\n\
echo "SPiece Model: $2"\n\
\n\
echo "Checking file existence:"\n\
ls -l "$1"\n\
ls -l "$2"\n\
\n\
cd /app/build\n\
gdb --args ./t5_inference "$1" "$2"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]