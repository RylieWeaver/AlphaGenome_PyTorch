#!/usr/bin/env bash
set -euo pipefail  # Exit immediately on error

# Usage check
echo "Usage: $0 <project_name> <env_name> <directory> [python_version]"
echo "Example: $0 AlphaGenome ag-env /path/to/directory 3.12"

# Parse arguments
PROJECT_NAME=${1:-AlphaGenome}
ENV_NAME=${2:-ag-env}
TARGET_DIR=${3:-$(pwd)}  # default to current directory
TARGET_DIR=$(eval echo "$TARGET_DIR")
PYTHON_VERSION=${4:-3.12}  # default to 3.12 if not provided

# Travel to target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Install uv if missing
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv already installed."
fi

# Initialize project
echo "Initializing project: $PROJECT_NAME"
uv init "$PROJECT_NAME" --bare --python "$PYTHON_VERSION"
cd "$PROJECT_NAME"

# Create venv
echo "Creating virtual environment: $ENV_NAME"
uv venv "$ENV_NAME" --python "$PYTHON_VERSION" --native-tls

# Activate venv
echo "Activating environment..."
source "$ENV_NAME/bin/activate"

# Install dependencies
echo "Installing dependencies..."
uv pip install torch numpy einops

echo
echo "Setup complete!"
echo "Activate your environment with:"
echo "source $TARGET_DIR/$PROJECT_NAME/$ENV_NAME/bin/activate"