#!/usr/bin/env bash
set -e

echo "=== Checking for Poetry ==="
if command -v poetry &> /dev/null; then
    echo "Poetry already installed: $(poetry --version)"
else
    echo "Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "=== Ensuring Poetry is on PATH ==="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v poetry &> /dev/null; then
    echo "Poetry install failed or not on PATH"
    exit 1
fi

echo ""
echo "=== Configuring Poetry to create .venv inside project ==="
poetry config virtualenvs.in-project true

echo ""
echo "=== Installing project dependencies ==="
poetry install --no-interaction

echo ""
echo "=== Activating virtual environment ==="
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated .venv"
else
    echo "No .venv found. Poetry should have created it. Exiting."
    exit 1
fi

echo ""
echo "=== Installing ThreedFront ==="
if [ -d "../ThreedFront" ]; then
    pip install -e ../ThreedFront
    echo "ThreedFront installed."
else
    echo "Warning: ../ThreedFront directory not found. Skipping."
fi

echo ""
echo "✅ Setup complete. Run your project with:"
echo "source .venv/bin/activate"
