#!/bin/bash

# train_models_linux.sh - Linux-specific training script with ML runtime initialization
# This script ensures all required ML runtime files are created before training begins

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting ML Training Pipeline for Linux${NC}"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if required directories exist
if [ ! -d "src" ] || [ ! -d "scripts" ]; then
    echo -e "${RED}Error: This script must be run from the project root directory${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Initialize ML runtime environment
echo -e "${YELLOW}Initializing ML runtime environment...${NC}"
if python3 -c "from startup_init import initialize_runtime; exit(0 if initialize_runtime(verbose=True) else 1)"; then
    echo -e "${GREEN}✓ ML runtime initialization completed successfully${NC}"
else
    echo -e "${RED}✗ ML runtime initialization failed${NC}"
    exit 1
fi

# Check if configuration file exists
CONFIG_FILE="src/config/config_training.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Run the enhanced trainer
echo -e "${YELLOW}Starting enhanced trainer...${NC}"
if python3 scripts/enhanced_trainer.py --config "$CONFIG_FILE" "$@"; then
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi

echo -e "${GREEN}Training pipeline completed successfully!${NC}"