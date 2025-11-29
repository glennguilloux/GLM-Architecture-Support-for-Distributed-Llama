#!/bin/bash

# Test build script for GLM Architecture Support
# This script will clean, configure, and build the project

echo "🚀 GLM Architecture Support - Build Test Script"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ Error: CMakeLists.txt not found. Run this script from the project root."
    exit 1
fi

# Check GCC 11 installation
echo "🔍 Checking GCC 11 installation..."
if [ -f "/home/glenn/gcc11-install/usr/local/bin/gcc-11" ]; then
    echo "✅ GCC 11 found at: /home/glenn/gcc11-install/usr/local/bin/gcc-11"
    /home/glenn/gcc11-install/usr/local/bin/gcc-11 --version
else
    echo "❌ GCC 11 not found at expected location."
    exit 1
fi

# Check CUDA installation
echo "🔍 Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✅ NVCC found:"
    nvcc --version
else
    echo "❌ NVCC not found. Please install CUDA toolkit."
    exit 1
fi

# Clean previous build
echo "🧹 Cleaning previous build..."
if [ -d "build" ]; then
    rm -rf build/
    echo "✅ Removed build/ directory"
fi

# Create and enter build directory
echo "🏗️ Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "⚙️ Configuring with CMake..."
cmake ..

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed!"
    exit 1
fi

echo "✅ CMake configuration successful!"

# Build the project
echo "🔨 Building project..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"

# Show available executables
echo "📦 Available executables:"
ls -la dllama glm-launcher intellect-worker glm-benchmark 2>/dev/null || echo "Build may not have completed all targets"

echo ""
echo "🎉 BUILD SUCCESSFUL!"
echo "Your CPU-optimized GLM Architecture Support project is ready!"
echo ""
echo "Next steps:"
echo "1. Test with: ./dllama --help"
echo "2. Run GLM demo: python ../launch-glm.py list"
echo "3. Submit sponsorship applications"
echo "4. Deploy to GitHub repository"
echo ""
echo "🚀 Ready for deployment to GitHub!"
