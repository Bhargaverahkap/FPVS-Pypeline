#!/usr/bin/env bash
# ===================================================================
# FPVS-Pypeline setup for macOS (also works on Linux).
# Creates a virtual environment and installs all required libraries.
# Usage:
#     chmod +x setup_macos.sh
#     ./setup_macos.sh
# ===================================================================

set -euo pipefail

cd "$(dirname "$0")"

echo
echo "=== Checking for Python ==="
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "Python 3 not found."
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew >/dev/null 2>&1; then
            echo "Installing Python with Homebrew..."
            brew install python
            PY=python3
        else
            echo "Install Homebrew first:"
            echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            echo "then run: brew install python"
            exit 1
        fi
    else
        echo "Install Python 3.10+ with your package manager, e.g.:"
        echo "    sudo apt install python3 python3-venv python3-pip"
        exit 1
    fi
fi

"$PY" --version

echo
echo "=== Checking for Tk (needed by the desktop app) ==="
if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
    echo "Python has no Tk support, installing it..."
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install python-tk || true
        else
            echo "Install Homebrew, then run: brew install python-tk"
            exit 1
        fi
    else
        if command -v apt >/dev/null 2>&1; then
            sudo apt install -y python3-tk || true
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y python3-tkinter || true
        else
            echo "Install the Tk package for your distribution, then run this again."
            exit 1
        fi
    fi
    if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
        echo "Tk is still missing. The pipeline modules will work, the desktop app will not."
        exit 1
    fi
fi
echo "Tk found."

echo
echo "=== Creating virtual environment in .venv ==="
if [ ! -d ".venv" ]; then
    "$PY" -m venv .venv
else
    echo ".venv already exists, reusing it."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo
echo "=== Installing libraries ==="
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "=== Registering Jupyter kernel ==="
python -m ipykernel install --user --name fpvs --display-name "Python (FPVS)"

echo
echo "=== Done ==="
echo "Activate the environment in a new terminal with:"
echo "    source .venv/bin/activate"
echo "Then start Jupyter with:"
echo "    jupyter lab"
echo
