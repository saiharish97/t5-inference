cmake_minimum_required(VERSION 3.14)
project(t5_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find TensorFlow
if(DEFINED ENV{TENSORFLOW_ROOT})
    set(TF_ROOT $ENV{TENSORFLOW_ROOT})
else()
    if(APPLE)
        set(TF_ROOT "/opt/homebrew/Cellar/libtensorflow/2.18.0")
    else()
        set(TF_ROOT "/usr/local/tensorflow")
    endif()
endif()

# Find TensorFlow headers and library
find_path(TF_INCLUDE_DIR
    NAMES tensorflow/c/c_api.h
    PATHS
        ${TF_ROOT}/include
        /usr/include
        /usr/local/include
    REQUIRED
)

if(APPLE)
    find_library(TF_LIB
        NAMES tensorflow libtensorflow.2.18.0.dylib
        PATHS ${TF_ROOT}/lib
        REQUIRED
    )
else()
    find_library(TF_LIB
        NAMES tensorflow libtensorflow.so
        PATHS 
            ${TF_ROOT}/lib
            /usr/lib
            /usr/local/lib
        REQUIRED
    )
endif()

message(STATUS "Found TensorFlow include dir: ${TF_INCLUDE_DIR}")
message(STATUS "Found TensorFlow library: ${TF_LIB}")

# Find SentencePiece
if(APPLE)
    # For brew-installed SentencePiece
    find_path(SENTENCEPIECE_INCLUDE_DIRS
        NAMES sentencepiece_processor.h
        PATHS
            /opt/homebrew/include
            /usr/local/include
        REQUIRED
    )
    
    find_library(SENTENCEPIECE_LIBRARIES
        NAMES sentencepiece
        PATHS
            /opt/homebrew/lib
            /usr/local/lib
        REQUIRED
    )
else()
    # For apt-installed SentencePiece
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
endif()

message(STATUS "Found SentencePiece include dirs: ${SENTENCEPIECE_INCLUDE_DIRS}")
message(STATUS "Found SentencePiece libraries: ${SENTENCEPIECE_LIBRARIES}")

# Add executable
add_executable(t5_inference 
    src/main.cpp
    src/tokenizer.cpp
    src/model.cpp
)

# Include directories
target_include_directories(t5_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TF_INCLUDE_DIR}
        ${SENTENCEPIECE_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(t5_inference
    PUBLIC
        ${TF_LIB}
        ${SENTENCEPIECE_LIBRARIES}
)

# Platform-specific settings
if(APPLE)
    target_compile_options(t5_inference PRIVATE -Wall -Wextra)
    set_target_properties(t5_inference PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path;@loader_path/../lib;${TF_ROOT}/lib"
    )
else()
    target_compile_options(t5_inference PRIVATE -Wall -Wextra -Wl,--no-as-needed)
    set_target_properties(t5_inference PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${TF_ROOT}/lib"
    )
endif()