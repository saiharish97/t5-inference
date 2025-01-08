cmake_minimum_required(VERSION 3.14)
project(t5_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_SENTENCEPIECE "Build SentencePiece from source" ON)
option(USE_NNSTREAMER "Enable NNStreamer support" OFF)
option(USE_SYSTEM_PROTOBUF "Use system protobuf instead of building from source" ${APPLE})

# Handle Protobuf
if(NOT USE_SYSTEM_PROTOBUF)
    include(ExternalProject)
    set(PROTOBUF_PREFIX ${CMAKE_BINARY_DIR}/protobuf)
    set(PROTOBUF_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/protobuf/install)
    
    ExternalProject_Add(protobuf_external
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG v3.19.4
        PREFIX ${PROTOBUF_PREFIX}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_PREFIX}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_EXAMPLES=OFF
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        BUILD_BYPRODUCTS
            ${PROTOBUF_INSTALL_PREFIX}/lib/libprotobuf${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${PROTOBUF_INSTALL_PREFIX}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    
    # Delay protobuf path setup until after it's built
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INSTALL_PREFIX}/include)
    
    # Create imported target for protobuf
    add_library(protobuf::libprotobuf STATIC IMPORTED)
    file(MAKE_DIRECTORY ${Protobuf_INCLUDE_DIRS}) # Prevent include directory warnings
    
    # Set protobuf imported target properties
    if(APPLE)
        set_target_properties(protobuf::libprotobuf PROPERTIES
            IMPORTED_LOCATION ${PROTOBUF_INSTALL_PREFIX}/lib/libprotobuf${CMAKE_SHARED_LIBRARY_SUFFIX}
            INTERFACE_INCLUDE_DIRECTORIES ${Protobuf_INCLUDE_DIRS}
        )
    else()
        set_target_properties(protobuf::libprotobuf PROPERTIES
            IMPORTED_LOCATION ${PROTOBUF_INSTALL_PREFIX}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}
            INTERFACE_INCLUDE_DIRECTORIES ${Protobuf_INCLUDE_DIRS}
        )
    endif()
    
    set(Protobuf_LIBRARIES protobuf::libprotobuf)
else()
    find_package(Protobuf REQUIRED)
endif()

message(STATUS "Using Protobuf include: ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Using Protobuf libs: ${Protobuf_LIBRARIES}")

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
    
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build shared library" FORCE)
    set(SPM_ENABLE_SAMPLES OFF CACHE BOOL "Build sample programs" FORCE)
    set(SPM_ENABLE_TESTS OFF CACHE BOOL "Build test programs" FORCE)
    
    # Point SentencePiece to our local protobuf if we're building it
    if(NOT USE_SYSTEM_PROTOBUF)
        set(CMAKE_PREFIX_PATH ${PROTOBUF_INSTALL_PREFIX} ${CMAKE_PREFIX_PATH})
    endif()
    
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

if(NOT USE_SYSTEM_PROTOBUF)
    add_dependencies(t5_inference protobuf_external)
    add_dependencies(protobuf::libprotobuf protobuf_external)
endif()

# Include directories
target_include_directories(t5_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TF_INCLUDE_DIR}
        ${SENTENCEPIECE_INCLUDE_DIR}
        ${Protobuf_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(t5_inference
    PUBLIC
        ${TF_LIB}
        ${SENTENCEPIECE_LIB}
        ${Protobuf_LIBRARIES}
)

# Add NNStreamer if enabled
if(USE_NNSTREAMER AND NOT APPLE)
    target_include_directories(t5_inference PUBLIC ${NNSTREAMER_INCLUDE_DIRS})
    target_link_libraries(t5_inference PUBLIC ${NNSTREAMER_LIBRARIES})
    target_compile_definitions(t5_inference PRIVATE USE_NNSTREAMER)
endif()

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