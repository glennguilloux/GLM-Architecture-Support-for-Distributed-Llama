# CMakeLists.txt for GLM Architecture Support
# Extended from distributed-llama to support GLM-4 and INTELLECT-3

cmake_minimum_required(VERSION 3.16)
project(GLMDistributedLlama VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use system default compilers (can be overridden if needed)
# set(CMAKE_C_COMPILER /home/glenn/gcc11-install/usr/local/bin/gcc-11)
# set(CMAKE_CXX_COMPILER /home/glenn/gcc11-install/usr/local/bin/g++-11)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Check for SIMD support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Check for AVX2 support
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    endif()
endif()

# Source files
set(LLAMA_SOURCES
    src/llama.cpp
    src/app.cpp
    src/llm.cpp
    src/llm-cli.cpp
    src/llm-api.cpp
    src/tokenizer.cpp
    src/tokenizer-cli.cpp
    src/tokenizer-lib.cpp
    src/llamafile-sgemm.cpp
    src/nn-core.cpp
    src/nn-cpu.cpp
    src/nn-cpu-ops.cpp
    src/nn-executor.cpp
    src/nn-network.cpp
    src/nn-quants.cpp
    src/nn-quantize.cpp
    src/llama-cli.cpp
    src/llama-server.cpp
)

# GLM-4 specific sources
set(GLM_SOURCES
    src/glm/glm-4.cpp
    src/glm/glm-tokenizer.cpp
    src/glm/glm-quantize.cpp
    src/glm/glm-attention.cpp
    src/glm/glm-layer-norm.cpp
    src/glm/glm-rope.cpp
)

# INTELLECT-3 MoE sources
set(INTELLECT3_SOURCES
    src/glm/intellect-3.cpp
    src/glm/intellect-router.cpp
    src/glm/expert-network.cpp
    src/glm/moe-manager.cpp
    src/glm/distributed-moe.cpp
    src/glm/expert-cache.cpp
)

# Combine all sources
set(ALL_SOURCES ${LLAMA_SOURCES} ${GLM_SOURCES} ${INTELLECT3_SOURCES})

# Library target
add_library(glm-dllama ${ALL_SOURCES})
target_link_libraries(glm-dllama PRIVATE Threads::Threads)

# Executable targets

# 1. Main dllama executable
add_executable(dllama src/main.cpp)
target_link_libraries(dllama PRIVATE glm-dllama)

# 2. GLM-specific launcher
add_executable(glm-launcher src/glm/glm-launcher.cpp)
target_link_libraries(glm-launcher PRIVATE glm-dllama)

# 3. INTELLECT-3 worker
add_executable(intellect-worker src/glm/intellect-worker.cpp)
target_link_libraries(intellect-worker PRIVATE glm-dllama)

# 4. Benchmark tool
add_executable(glm-benchmark src/glm/glm-benchmark.cpp)
target_link_libraries(glm-benchmark PRIVATE glm-dllama)

# Include directories
target_include_directories(glm-dllama PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/glm
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(glm-dllama PRIVATE -funroll-loops)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(glm-dllama PRIVATE -ffast-math)
endif()

# Platform-specific settings
if(APPLE)
    target_link_libraries(glm-dllama PRIVATE "-framework Foundation" "-framework Metal" "-framework MetalKit")
endif()

if(UNIX AND NOT APPLE)
    find_package(OpenMP REQUIRED)
    target_link_libraries(glm-dllama PRIVATE OpenMP::OpenMP_CXX)
endif()

# GLM-specific compiler definitions
target_compile_definitions(glm-dllama PRIVATE
    GLM_ARCHITECTURE_SUPPORT=1
    INTELLECT3_MOE_SUPPORT=1
    CONSUMER_HARDWARE_OPTIMIZATION=1
)

# Install targets
install(TARGETS dllama glm-launcher intellect-worker glm-benchmark
    RUNTIME DESTINATION bin
)

install(TARGETS glm-dllama
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/glm-distributed-llama
)

# Packaging
set(CPACK_PACKAGE_NAME "glm-distributed-llama")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GLM Architecture Support for Distributed Llama")
set(CPACK_PACKAGE_VENDOR "Glenn Guilloux")
set(CPACK_GENERATOR "TGZ;ZIP")

# Include CPack
include(CPack)

# Custom targets

# 1. Build all targets
add_custom_target(build-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    DEPENDS dllama glm-launcher intellect-worker glm-benchmark
    COMMENT "Building all GLM Architecture Support targets"
)

# 2. Run tests (renamed to avoid CTest conflict)
add_custom_target(run-tests
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run-tests.sh
    DEPENDS dllama
    COMMENT "Running test suite for GLM Architecture Support"
)

# 3. Benchmark
add_custom_target(benchmark
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/glm-benchmark
    DEPENDS glm-benchmark
)

# 4. Install dependencies
add_custom_target(install-deps
    COMMAND apt-get update && apt-get install -y build-essential cmake git python3
    COMMAND pip3 install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
)

# Configuration file (commented out until template is created)
# configure_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/glm-config.h.in
#     ${CMAKE_CURRENT_BINARY_DIR}/glm-config.h
# )

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif()

# Set CUDA host compiler to GCC 11 (CUDA 12.8 compatible)
set(CMAKE_CUDA_HOST_COMPILER /home/glenn/gcc11-install/usr/local/bin/gcc-11)

# GPU support detection with fallback to CPU-only build
# Temporarily disable CUDA due to GCC 11 header issues
# TODO: Fix GCC 11 installation for proper CUDA support
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND AND FALSE)  # Disabled until GCC 11 headers are fixed
    # Enable CUDA support with modern package finding
    enable_language(CUDA)
    
    # Set CUDA standard and compatibility flags
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # CUDA compiler flags for compatibility
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
    
    target_compile_definitions(glm-dllama PRIVATE CUDA_SUPPORT=1)
    
    # CUDA-specific sources for GLM acceleration
    set(CUDA_SOURCES
        src/gpu/glm-gpu-kernels.cu
        src/gpu/cuda-attention.cu
        src/gpu/cuda-moe.cu
        src/gpu/cuda-quantize.cu
        src/gpu/cuda-rope.cu
    )
    
    target_sources(glm-dllama PRIVATE ${CUDA_SOURCES})
    target_include_directories(glm-dllama PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(glm-dllama PRIVATE ${CUDAToolkit_LIBRARIES})
    
    message(STATUS "✅ CUDA Support: ENABLED")
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit Path: ${CUDAToolkit_ROOT}")
    message(STATUS "CUDA Architectures: ${CUDAToolkit_TARGET_ARCHITECTURES}")
else()
    # Build CPU-only version (will be optimized later)
    target_compile_definitions(glm-dllama PRIVATE CPU_ONLY=1)
    message(STATUS "🔄 CUDA Support: DISABLED (Building CPU-optimized version)")
    message(STATUS "CUDA will be enabled once GCC 11 headers are fixed")
    message(STATUS "CPU-optimized GLM implementation will still work!")
endif()

# Debug information
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(glm-dllama PRIVATE DEBUG_MODE=1)
    add_definitions(-DDEBUG)
endif()

# Profile information
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_definitions(glm-dllama PRIVATE PROFILING=1)
endif()
