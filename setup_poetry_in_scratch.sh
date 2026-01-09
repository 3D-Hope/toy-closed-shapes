#!/bin/bash
# Script to manually install Poetry to /scratch directory
# Run this once on your SLURM cluster to set up Poetry

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "Poetry Installation Script for /scratch"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
POETRY_HOME="/scratch/pramish_paudel/tools/poetry"
SCRATCH_BIN="/scratch/pramish_paudel/tools/bin"

echo "Installation directories:"
echo "  Poetry home: $POETRY_HOME"
echo "  Binary directory: $SCRATCH_BIN"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p "$POETRY_HOME"
mkdir -p "$SCRATCH_BIN"
echo "✅ Directories created"
echo ""

# Check if Poetry is already installed
if [ -f "$POETRY_HOME/bin/poetry" ]; then
    echo "⚠️  Poetry already exists at $POETRY_HOME/bin/poetry"
    read -p "Do you want to reinstall? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping installation"
        exit 0
    fi
    echo "Removing existing Poetry installation..."
    rm -rf "$POETRY_HOME"
    mkdir -p "$POETRY_HOME"
fi

# Method 1: Official Poetry Installer (Recommended)
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Method 1: Installing Poetry using official installer..."
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

curl -sSL https://install.python-poetry.org | POETRY_HOME="$POETRY_HOME" python3 -

if [ -f "$POETRY_HOME/bin/poetry" ]; then
    echo "✅ Poetry successfully installed!"
    echo ""
    echo "Poetry executable: $POETRY_HOME/bin/poetry"
    echo "Version: $($POETRY_HOME/bin/poetry --version)"
    echo ""
    
    # Create symlink in bin directory for easy access
    echo "Creating symlink in $SCRATCH_BIN..."
    ln -sf "$POETRY_HOME/bin/poetry" "$SCRATCH_BIN/poetry"
    echo "✅ Symlink created: $SCRATCH_BIN/poetry"
    echo ""
    
    # Test Poetry
    echo "Testing Poetry installation..."
    "$POETRY_HOME/bin/poetry" --version
    echo "✅ Poetry is working!"
    echo ""
    
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Installation Complete!"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "To use Poetry, add this to your PATH:"
    echo "  export PATH=\"$POETRY_HOME/bin:\$PATH\""
    echo ""
    echo "Or use the direct path:"
    echo "  $POETRY_HOME/bin/poetry"
    echo ""
    echo "Or use the symlink:"
    echo "  $SCRATCH_BIN/poetry"
    echo ""
    
else
    echo "❌ Installation failed!"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Trying Method 2: Installing via pip..."
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Method 2: Install via pip to custom location
    POETRY_LIB_DIR="/scratch/pramish_paudel/tools/poetry-libs"
    mkdir -p "$POETRY_LIB_DIR"
    
    echo "Installing Poetry to $POETRY_LIB_DIR..."
    pip3 install --target="$POETRY_LIB_DIR" poetry
    
    # Create wrapper script
    cat > "$SCRATCH_BIN/poetry" << 'EOF'
#!/bin/bash
POETRY_LIB_DIR="/scratch/pramish_paudel/tools/poetry-libs"
export PYTHONPATH="$POETRY_LIB_DIR:$PYTHONPATH"
python3 -m poetry "$@"
EOF
    
    chmod +x "$SCRATCH_BIN/poetry"
    
    echo "✅ Poetry installed via pip"
    echo ""
    echo "Testing Poetry installation..."
    "$SCRATCH_BIN/poetry" --version
    echo "✅ Poetry is working!"
    echo ""
    
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Installation Complete (pip method)!"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "To use Poetry:"
    echo "  $SCRATCH_BIN/poetry"
    echo ""
fi

# Create a test project to verify Poetry works
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Verification Test"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

TEST_DIR="/tmp/poetry-test-$$"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "Creating test project..."
if [ -f "$POETRY_HOME/bin/poetry" ]; then
    "$POETRY_HOME/bin/poetry" new test-project --quiet
    POETRY_CMD="$POETRY_HOME/bin/poetry"
else
    "$SCRATCH_BIN/poetry" new test-project --quiet
    POETRY_CMD="$SCRATCH_BIN/poetry"
fi

cd test-project

echo "Configuring Poetry..."
$POETRY_CMD config virtualenvs.in-project true

echo "Installing test dependencies..."
$POETRY_CMD add --quiet requests

if [ -d ".venv" ]; then
    echo "✅ Virtual environment created successfully!"
    echo "✅ All tests passed!"
else
    echo "⚠️  Virtual environment not created, but Poetry is functional"
fi

# Cleanup
cd /
rm -rf "$TEST_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Setup Complete!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Poetry installed to: $POETRY_HOME"
echo "  ✅ Binary available at: $SCRATCH_BIN/poetry"
echo ""
echo "Next steps:"
echo "  1. Add to your PATH: export PATH=\"$POETRY_HOME/bin:\$PATH\""
echo "  2. Or use the full path in scripts: $POETRY_HOME/bin/poetry"
echo "  3. The SLURM training script will automatically use this installation"
echo ""
