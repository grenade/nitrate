#!/bin/bash

# Nitrate miner launcher script with CUDA support
# Builds with multi-architecture fatbin for broad GPU compatibility

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Optional: Override target architectures if needed
# By default, builds a fatbin with code for all supported GPUs from Turing to Blackwell
# Examples:
#   export CUDA_ARCH=sm_86                    # Build only for RTX 3060
#   export CUDA_ARCH=sm_89                    # Build only for RTX 4090
#   export CUDA_ARCH=sm_86,sm_89,sm_120       # Build for specific GPUs
#   export CUDA_ARCH=compute_120              # Build PTX only for JIT compilation
# Default includes: sm_75,sm_80,sm_86,sm_87,sm_89,sm_90,sm_120

# Configuration file (default to quadbrat.toml if not specified)
CONFIG_FILE="${1:-quadbrat.toml}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found"
    echo "Usage: $0 [config-file]"
    exit 1
fi

echo "========================================="
echo "Nitrate GPU Miner Launcher"
echo "========================================="
echo "Configuration: $CONFIG_FILE"
echo "CUDA Home: $CUDA_HOME"

# Detect GPU and show info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Detected GPU(s):"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader | while IFS=, read -r idx name cap mem; do
        cap_trimmed=$(echo $cap | xargs)
        sm_ver=$(echo $cap_trimmed | tr -d '.')
        echo "  GPU $idx: $name (sm_$sm_ver, $mem)"
    done
fi

# Show CUDA architecture mode
echo ""
if [ -n "$CUDA_ARCH" ]; then
    echo "Build Mode: Custom architectures - $CUDA_ARCH"
else
    echo "Build Mode: Multi-architecture fatbin"
    echo "  Includes optimized code for:"
    echo "    - Turing (RTX 2000/GTX 1600) - sm_75"
    echo "    - Ampere (RTX 3000) - sm_80/86/87"
    echo "    - Ada Lovelace (RTX 4000) - sm_89"
    echo "    - Hopper (H100/H200) - sm_90"
    echo "    - Blackwell (RTX 5090) - sm_120"
    echo "  The CUDA runtime will automatically select the best code for your GPU"
fi

echo ""

# Build and run the miner
# First ensure it's built with the correct architecture
echo "Building with CUDA support..."
cargo build -p nitrate-cli --features gpu-cuda --release

if [ $? -ne 0 ]; then
    echo "Build failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "Starting miner..."
echo "Press Ctrl-C to stop"
echo "========================================="
echo ""

# Run the miner
cargo run -p nitrate-cli --features gpu-cuda --release -- --config "$CONFIG_FILE"
