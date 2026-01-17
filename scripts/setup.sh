#!/bin/bash
#
# Setup script for AI Governance Starter Kit
#
# This script sets up the development environment and validates the installation.

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   AI Governance Starter Kit - Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Check Python version
echo -e "${BLUE}[1/6]${NC} Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION} detected"
else
    echo -e "${RED}✗${NC} Python 3.9+ required. Found: ${PYTHON_VERSION}"
    exit 1
fi

# Check if pip is available
echo -e "\n${BLUE}[2/6]${NC} Checking pip..."
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} pip3 is available"
else
    echo -e "${RED}✗${NC} pip3 not found. Please install pip."
    exit 1
fi

# Install runtime dependencies
echo -e "\n${BLUE}[3/6]${NC} Installing runtime dependencies..."
pip3 install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Runtime dependencies installed"

# Install development dependencies
echo -e "\n${BLUE}[4/6]${NC} Installing development dependencies..."
pip3 install -q -r requirements-dev.txt
echo -e "${GREEN}✓${NC} Development dependencies installed"

# Set up pre-commit hooks
echo -e "\n${BLUE}[5/6]${NC} Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo -e "${GREEN}✓${NC} Pre-commit hooks installed"
else
    echo -e "${YELLOW}⚠${NC} pre-commit not found, skipping hook installation"
fi

# Validate registry
echo -e "\n${BLUE}[6/6]${NC} Validating model registry..."
if python3 scripts/validate_registry.py; then
    echo -e "${GREEN}✓${NC} Model registry validation passed"
else
    echo -e "${YELLOW}⚠${NC} Model registry validation failed (you can fix this later)"
fi

# Run tests to verify setup
echo -e "\n${BLUE}Running tests to verify setup...${NC}"
if pytest tests/test_registry_validation.py -v --tb=short; then
    echo -e "${GREEN}✓${NC} Tests passed"
else
    echo -e "${YELLOW}⚠${NC} Some tests failed (you can fix this later)"
fi

# Print success message
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✓ Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  1. Review the README.md for an overview"
echo "  2. Check docs/templates/ for documentation templates"
echo "  3. Review models/examples/ for model examples"
echo "  4. Run 'make help' to see available commands"
echo ""
echo "Quick commands:"
echo "  make validate     - Validate model registry"
echo "  make test         - Run all tests"
echo "  make lint         - Check code quality"
echo "  make train-examples - Train example models"
echo ""
echo "For more help, see README.md or run 'make help'"
echo ""
