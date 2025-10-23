#!/bin/bash

# Nitrate GPU Miner - Optimized Run Script
# This script builds and runs the miner with performance optimizations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Nitrate GPU Miner - Optimized Runner ===${NC}"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Warning: nvidia-smi not found. CUDA may not be available.${NC}"
    echo "Continuing with build anyway..."
else
    echo -e "${GREEN}CUDA GPUs detected:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while IFS=',' read -r idx name mem; do
        echo "  GPU $idx: $name (${mem} MB)"
    done
    echo ""
fi

# Parse command line arguments
CONFIG_FILE="miner.toml"
BUILD_MODE="release"
CLEAN_BUILD=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --tuned)
            CONFIG_FILE="miner.tuned.toml"
            shift
            ;;
        --debug)
            BUILD_MODE="debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config FILE    Use specified config file (default: miner.toml)"
            echo "  --tuned         Use pre-tuned config for RTX 5090 (miner.tuned.toml)"
            echo "  --debug         Build in debug mode (slower but with debug symbols)"
            echo "  --clean         Clean build from scratch"
            echo "  --skip-build    Skip building, just run existing binary"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Build and run with default config"
            echo "  $0 --tuned           # Run with RTX 5090 optimized settings"
            echo "  $0 --config my.toml  # Use custom config file"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file '$CONFIG_FILE' not found!${NC}"
    echo "Please create a config file or use --config to specify one."
    exit 1
fi

echo -e "${YELLOW}Using config: $CONFIG_FILE${NC}"

# Build the project if not skipping
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo -e "${YELLOW}Building Nitrate with CUDA support...${NC}"

    if [ "$CLEAN_BUILD" = true ]; then
        echo "Cleaning previous build..."
        cargo clean
    fi

    # Set CUDA architecture for RTX 50/40/30 series
    # sm_89 = Ada Lovelace (RTX 40)
    # sm_86 = Ampere (RTX 30)
    # sm_90 = Blackwell (RTX 50) - might need adjustment when available
    export CUDA_ARCH="sm_86"

    # Build with CUDA feature
    if [ "$BUILD_MODE" = "release" ]; then
        echo "Building in release mode (optimized)..."
        cargo build --release -p nitrate-cli --features gpu-cuda
        BINARY="./target/release/nitrate-cli"
    else
        echo "Building in debug mode..."
        cargo build -p nitrate-cli --features gpu-cuda
        BINARY="./target/debug/nitrate-cli"
    fi

    echo -e "${GREEN}Build complete!${NC}"
else
    # Determine binary path based on build mode
    if [ "$BUILD_MODE" = "release" ]; then
        BINARY="./target/release/nitrate-cli"
    else
        BINARY="./target/debug/nitrate-cli"
    fi

    if [ ! -f "$BINARY" ]; then
        echo -e "${RED}Error: Binary not found at $BINARY${NC}"
        echo "Please build first or remove --skip-build flag"
        exit 1
    fi
fi

# Check if metrics port is already in use
METRICS_PORT=$(grep telemetry_addr "$CONFIG_FILE" | cut -d'"' -f2 | cut -d':' -f2)
if [ ! -z "$METRICS_PORT" ]; then
    if lsof -Pi :$METRICS_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $METRICS_PORT is already in use. Metrics may not be available.${NC}"
    fi
fi

# Run the miner
echo ""
echo -e "${GREEN}Starting Nitrate GPU Miner...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""
echo "Metrics available at: http://localhost:${METRICS_PORT:-9100}/metrics"
echo "----------------------------------------"
echo ""

# Set environment for better CUDA performance
export CUDA_LAUNCH_BLOCKING=0  # Async kernel launches
export CUDA_CACHE_DISABLE=0     # Enable CUDA cache

# Run with info logging by default
RUST_LOG=${RUST_LOG:-nitrate=info,nitrate_core=info,nitrate_gpu_cuda=info}
export RUST_LOG

exec $BINARY --config "$CONFIG_FILE"
