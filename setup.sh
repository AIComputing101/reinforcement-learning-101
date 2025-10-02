#!/usr/bin/env bash
# Smart setup script for reinforcement-learning-101
# Auto-detects GPU and sets up appropriate environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() { echo -e "${BLUE}ℹ ${NC}$1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Detect GPU type
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "cuda"
            return
        fi
    fi

    if command -v rocminfo &> /dev/null; then
        if rocminfo &> /dev/null; then
            echo "rocm"
            return
        fi
    fi

    if [ -d "/dev/dri" ] && [ -e "/dev/kfd" ]; then
        echo "rocm"
        return
    fi

    echo "cpu"
}

# Setup native Python environment
setup_native() {
    local backend=$1
    print_info "Setting up native Python environment (${backend})..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.11 or later."
        exit 1
    fi

    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_info "Detected Python ${python_version}"

    # Create virtual environment
    if [ ! -d ".venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv .venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    # Install base requirements
    print_info "Installing base requirements..."
    pip install -r requirements/requirements-base.txt

    # Install PyTorch based on backend
    case $backend in
        cuda)
            print_info "Installing PyTorch with CUDA support..."
            pip install -r requirements/requirements-torch-cuda.txt
            ;;
        rocm)
            print_info "Installing PyTorch with ROCm support..."
            pip install -r requirements/requirements-torch-rocm.txt
            ;;
        cpu)
            print_info "Installing PyTorch (CPU-only)..."
            pip install -r requirements/requirements-torch-cpu.txt
            ;;
    esac

    print_success "Native environment setup complete!"
    print_info "Activate with: source .venv/bin/activate"
}

# Setup Docker environment
setup_docker() {
    local backend=$1
    print_info "Setting up Docker environment (${backend})..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi

    # Check docker-compose
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose v2."
        exit 1
    fi

    # Build and run container
    print_info "Building Docker image..."
    bash docker/run.sh "$backend"

    print_success "Docker environment setup complete!"
}

# Main script
main() {
    echo ""
    print_info "=== Reinforcement Learning 101 - Environment Setup ==="
    echo ""

    # Parse arguments
    MODE=${1:-auto}
    BACKEND=${2:-auto}

    # Auto-detect GPU if needed
    if [ "$BACKEND" = "auto" ]; then
        BACKEND=$(detect_gpu)
        print_info "Auto-detected backend: ${BACKEND}"
    fi

    # Validate backend
    if [[ ! "$BACKEND" =~ ^(cpu|cuda|rocm)$ ]]; then
        print_error "Invalid backend: $BACKEND (must be cpu, cuda, or rocm)"
        exit 1
    fi

    # Display GPU info
    case $BACKEND in
        cuda)
            if command -v nvidia-smi &> /dev/null; then
                print_info "NVIDIA GPU detected:"
                nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1
            fi
            ;;
        rocm)
            print_info "AMD GPU detected (ROCm)"
            ;;
        cpu)
            print_info "No GPU detected, using CPU-only mode"
            ;;
    esac
    echo ""

    # Setup based on mode
    case $MODE in
        native)
            setup_native "$BACKEND"
            ;;
        docker)
            setup_docker "$BACKEND"
            ;;
        auto)
            print_info "Choose setup mode:"
            echo "  1) Native Python environment (recommended for development)"
            echo "  2) Docker container (recommended for reproducibility)"
            echo ""
            read -p "Select mode [1-2]: " choice
            case $choice in
                1) setup_native "$BACKEND" ;;
                2) setup_docker "$BACKEND" ;;
                *) print_error "Invalid choice"; exit 1 ;;
            esac
            ;;
        *)
            print_error "Invalid mode: $MODE"
            echo ""
            echo "Usage: $0 [native|docker|auto] [cpu|cuda|rocm|auto]"
            echo ""
            echo "Examples:"
            echo "  $0                    # Auto-detect GPU and choose mode interactively"
            echo "  $0 native             # Setup native environment with auto-detected GPU"
            echo "  $0 docker cuda        # Setup Docker with CUDA support"
            echo "  $0 native cpu         # Setup native environment (CPU-only)"
            exit 1
            ;;
    esac

    echo ""
    print_success "Setup complete! 🚀"
    echo ""
    print_info "Quick start:"
    if [ "$MODE" = "native" ] || [ "$MODE" = "auto" ]; then
        echo "  source .venv/bin/activate"
    fi
    echo "  python modules/module_01_intro/examples/bandit_epsilon_greedy.py"
    echo ""
}

main "$@"
