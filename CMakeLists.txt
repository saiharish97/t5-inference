cmake_minimum_required(VERSION 3.14)
project(t5_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_SENTENCEPIECE "Build SentencePiece from source" ON)
option(USE_NNSTREAMER "Enable NNStreamer support" OFF)

# Find TensorFlow
if(DEFINED ENV{TENSORFLOW_ROOT})
    set(TF_ROOT $ENV{TENSORFLOW_ROOT})
else()
    if(APPLE)
        set(TF_ROOT "/opt/homebrew/Cellar/libtensorflow/2.18.0")
    else()
        set(TF_ROOT "/usr/local")
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

# SentencePiece setup
if(BUILD_SENTENCEPIECE)
    include(FetchContent)
    FetchContent_Declare(
        sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG v0.1.97
    )
    
    # Configure SentencePiece build options
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build shared library" FORCE)
    set(SPM_ENABLE_SAMPLES OFF CACHE BOOL "Build sample programs" FORCE)
    set(SPM_ENABLE_TESTS OFF CACHE BOOL "Build test programs" FORCE)
    
    FetchContent_MakeAvailable(sentencepiece)
    FetchContent_GetProperties(sentencepiece)
    set(SENTENCEPIECE_INCLUDE_DIR ${sentencepiece_SOURCE_DIR}/src)
    set(SENTENCEPIECE_LIB sentencepiece)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
    set(SENTENCEPIECE_INCLUDE_DIR ${SENTENCEPIECE_INCLUDE_DIRS})
    set(SENTENCEPIECE_LIB ${SENTENCEPIECE_LIBRARIES})
endif()

message(STATUS "Found SentencePiece include dir: ${SENTENCEPIECE_INCLUDE_DIR}")

# NNStreamer setup (Linux only)
if(USE_NNSTREAMER AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NNSTREAMER REQUIRED nnstreamer)
    message(STATUS "Found NNStreamer: ${NNSTREAMER_LIBRARIES}")
endif()

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
        ${SENTENCEPIECE_INCLUDE_DIR}
)

# Base libraries
set(LIBS_TO_LINK
    ${TF_LIB}
    ${SENTENCEPIECE_LIB}
)

# Add NNStreamer if enabled
if(USE_NNSTREAMER AND NOT APPLE)
    target_include_directories(t5_inference PUBLIC ${NNSTREAMER_INCLUDE_DIRS})
    list(APPEND LIBS_TO_LINK ${NNSTREAMER_LIBRARIES})
    target_compile_definitions(t5_inference PRIVATE USE_NNSTREAMER)
endif()

# Link all libraries
target_link_libraries(t5_inference PUBLIC ${LIBS_TO_LINK})

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