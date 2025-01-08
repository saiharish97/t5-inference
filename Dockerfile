# Use an older Ubuntu as base image
FROM --platform=linux/amd64 ubuntu:20.04

# Avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
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
    file \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow and verify its architecture
RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz && \
    tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz && \
    rm libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz && \
    ldconfig && \
    file /usr/local/lib/libtensorflow.so

# Set working directory
WORKDIR /app

# Copy your project files
COPY . .

# Create build directory and build project with verbose output
RUN mkdir -p build && \
    cd build && \
    echo "Building for architecture:" && arch && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DBUILD_SENTENCEPIECE=ON \
        -DTENSORFLOW_ROOT=/usr/local \
        -DCMAKE_VERBOSE_MAKEFILE=ON && \
    make VERBOSE=1

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV TF_CPP_MIN_LOG_LEVEL=0
ENV TF_CPP_MIN_VLOG_LEVEL=3
ENV TF_CPP_MAX_VLOG_LEVEL=3
# Disable TensorFlow optimizations
ENV TF_DISABLE_MKL=1
ENV TF_DISABLE_JEMALLOC=1
ENV TF_DISABLE_AVX=1
ENV TF_DISABLE_AVX2=1

# Create a directory for models
RUN mkdir -p /app/models

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$#" -ne 2 ]; then\n\
    echo "Usage: $0 <tf_model_path> <spiece_model_path>"\n\
    exit 1\n\
fi\n\
echo "Running on architecture: $(arch)"\n\
file /app/build/t5_inference\n\
file /usr/local/lib/libtensorflow.so\n\
ldd /app/build/t5_inference\n\
cd /app/build\n\
./t5_inference "$1" "$2"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]