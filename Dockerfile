# Use Ubuntu base image
FROM ubuntu:20.04

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

# Create build directory and build project with debug info
RUN rm -rf build && mkdir -p build && \
    cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_SENTENCEPIECE=ON \
        -DTENSORFLOW_ROOT=/usr/local \
        -DCMAKE_CXX_FLAGS="-g -O0" && \
    make -j$(nproc) VERBOSE=1

# Set environment variable for library path and debugging
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV TF_CPP_MIN_LOG_LEVEL=0
ENV TF_CPP_MIN_VLOG_LEVEL=3

# Create a directory for models
RUN mkdir -p /app/models

# Create an entrypoint script with debugging
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
echo "Running under GDB for backtrace:"\n\
cd /app/build\n\
gdb -batch -ex "run" -ex "bt" --args ./t5_inference "$1" "$2"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]