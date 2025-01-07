# T5 Inference

A C++ implementation for T5 model inference using TensorFlow C API and SentencePiece tokenizer.

## Prerequisites

- CMake (>= 3.14)
- C++17 compatible compiler
- TensorFlow C API
- SentencePiece

### Installation

#### macOS
```bash
# Install TensorFlow C API
brew install libtensorflow

# Install CMake if not already installed
brew install cmake
```

#### Linux
```bash
# Install TensorFlow C API
# Download and extract the TensorFlow C API from tensorflow.org
# Set TENSORFLOW_ROOT environment variable to the extracted location

# Install CMake
sudo apt-get install cmake build-essential
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

1. Download the T5 model and tokenizer:
```bash
# Instructions for downloading model files
```

2. Run the program:
```bash
./t5_inference <saved_model_path> <spm_model_path>
```

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── model.h
│   └── tokenizer.h
├── src/
│   ├── main.cpp
│   ├── model.cpp
│   └── tokenizer.cpp
└── README.md
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.