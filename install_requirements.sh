#!/usr/bin/env bash
# install_requirements.sh
# Install all Python dependencies for sense_seek_neural_decoding.
# Usage: bash install_requirements.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Python dependencies from requirements.txt ..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Checking for AWS CLI (required for S3 download) ..."
if command -v aws &>/dev/null; then
    echo "  aws CLI found: $(aws --version)"
else
    echo "  aws CLI not found."
    OS="$(uname -s 2>/dev/null || echo 'Windows')"
    case "$OS" in
        Darwin)
            echo "  macOS detected.  Install via:"
            echo "    brew install awscli"
            echo "  or: pip install awscli"
            ;;
        Linux)
            echo "  Linux detected.  Install via:"
            echo "    sudo apt-get install awscli   (Debian/Ubuntu)"
            echo "    sudo yum install awscli        (RHEL/CentOS)"
            echo "  or: pip install awscli"
            ;;
        *)
            echo "  Windows detected.  Install via:"
            echo "    winget install Amazon.AWSCLI"
            echo "  or download the MSI installer from:"
            echo "    https://aws.amazon.com/cli/"
            echo "  or: pip install awscli"
            ;;
    esac
fi

echo ""
echo "Done. All Python packages installed."
