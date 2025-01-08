# Use Ubuntu as base image with explicit platform
FROM --platform=linux/amd64 ubuntu:22.04

# Avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    pkg-config \
    python3 \
    python3-pip \
    ninja-build \
    gdb \
    file \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Abseil-cpp (required for Protobuf)
RUN cd /tmp && \
    wget --no-check-certificate https://github.com/abseil/abseil-cpp/releases/download/20240722.0/abseil-cpp-20240722.0.tar.gz && \
    tar xf abseil-cpp-20240722.0.tar.gz && \
    cd abseil-cpp-20240722.0 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr \
          -DCMAKE_BUILD_TYPE=Release \
          -DABSL_PROPAGATE_CXX_STD=ON \
          -DBUILD_SHARED_LIBS=ON \
          -G Ninja .. && \
    ninja && \
    ninja install && \
    cd ../.. && \
    rm -rf abseil-cpp-20240722.0*

# Install Protobuf with correct configuration
RUN cd /tmp && \
    wget --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v29.2/protobuf-29.2.tar.gz && \
    tar xf protobuf-29.2.tar.gz && \
    cd protobuf-29.2 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_SKIP_INSTALL_RPATH=ON \
          -Dprotobuf_BUILD_TESTS=OFF \
          -Dprotobuf_ABSL_PROVIDER=package \
          -Dprotobuf_BUILD_LIBUPB=OFF \
          -Dprotobuf_BUILD_SHARED_LIBS=ON \
          -G Ninja .. && \
    ninja && \
    ninja install && \
    ldconfig && \
    cd ../.. && \
    rm -rf protobuf-29.2*

# Install SentencePiece
RUN git clone https://github.com/google/sentencepiece.git && \
    cd sentencepiece && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig -v && \
    cd ../.. && \
    rm -rf sentencepiece

# Install TensorFlow
RUN wget --no-check-certificate \
    https://storage.googleapis.com/tensorflow/versions/2.18.0/libtensorflow-cpu-linux-x86_64.tar.gz && \
    tar -zxf libtensorflow-cpu-linux-x86_64.tar.gz -C /usr/local && \
    rm libtensorflow-cpu-linux-x86_64.tar.gz && \
    ldconfig /usr/local/lib

# Set environment variables for libraries
ENV LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Build project
RUN mkdir build && \
    cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DTENSORFLOW_ROOT=/usr/local && \
    make -j$(nproc)

# Create models directory
RUN mkdir -p /app/models

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$#" -ne 2 ]; then\n\
    echo "Usage: $0 <tf_model_path> <spiece_model_path>"\n\
    exit 1\n\
fi\n\
\n\
cd /app/build\n\
./t5_inference "$1" "$2"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]