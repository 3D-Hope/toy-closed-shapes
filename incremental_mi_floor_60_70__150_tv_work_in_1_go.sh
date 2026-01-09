#!/bin/bash
#SBATCH --job-name=incremental_mi_floor_60_70__150_tv
#SBATCH --nodelist=sof1-h200-4
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Better error reporting including line number
trap 'ERR_CODE=$?; echo "❌ Error on line ${LINENO:-?}. Exit code: $ERR_CODE" >&2; exit $ERR_CODE' ERR
trap 'echo "🛑 Job interrupted"; exit 130' INT

# -------------------------
# Basic setup / logging
# -------------------------
mkdir -p logs
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

echo "System information:"
free -h || true
df -h /scratch/pramish_paudel || df -h . || true
echo ""

export WANDB_ENTITY="078bct021-ashok-d"

# -------------------------
# Stage 1 & 2: copy caches/dataset (keeps your original behavior)
# -------------------------
echo "STAGE 1: Copy/extract caches and dataset..."

# Helper to copy and unzip if missing
_copy_and_unzip_if_missing() {
    local src_zip=$1
    local dst_dir=$2
    local dst_base=$(dirname "$dst_dir")
    local zip_name=$(basename "$src_zip")
    if [ -d "$dst_dir" ]; then
        echo "✅ $dst_dir already exists"
        return 0
    fi
    echo "Copying $zip_name to $dst_base..."
    rsync -aHzv --progress "$src_zip" "$dst_base/" || {
        echo "❌ Failed to copy $src_zip"; return 1
    }
    echo "Extracting $zip_name..."
    unzip -o "$dst_base/$zip_name" -d "$dst_base" || {
        echo "❌ Failed to extract $dst_base/$zip_name"; return 1
    }
    rm -f "$dst_base/$zip_name"
    echo "✅ $dst_dir copied & extracted"
    return 0
}

# Use the helper for your caches/dataset
rm -rf /scratch/pramish_paudel/bedroom_sdf_cache || true
_copy_and_unzip_if_missing /home/pramish_paudel/3dhope_data/bedroom_sdf_cache.zip /scratch/pramish_paudel/bedroom_sdf_cache || { echo "Failed stage: bedroom_sdf_cache"; exit 1; }
ls -la /scratch/pramish_paudel/bedroom_sdf_cache || true

rm -rf /scratch/pramish_paudel/bedroom_accessibility_cache || true
_copy_and_unzip_if_missing /home/pramish_paudel/3dhope_data/bedroom_accessibility_cache.zip /scratch/pramish_paudel/bedroom_accessibility_cache || { echo "Failed stage: bedroom_accessibility_cache"; exit 1; }
ls -la /scratch/pramish_paudel/bedroom_accessibility_cache || true

echo "STAGE 2: Checking bedroom dataset..."
if [ ! -d "/scratch/pramish_paudel/bedroom" ]; then
    _copy_and_unzip_if_missing /home/pramish_paudel/3dhope_data/bedroom.zip /scratch/pramish_paudel/bedroom || { echo "❌ Failed to copy/extract bedroom dataset"; exit 1; }
else
    echo "✅ bedroom dataset already exists in scratch"
fi
echo ""

# -------------------------
# Stage 3: Miniforge/Conda setup
# -------------------------
echo "STAGE 3: Setting up Miniforge (if missing)..."
CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniforge to $CONDA_DIR..."
    mkdir -p /scratch/pramish_paudel/tools/
    cd /scratch/pramish_paudel/tools/
    # Generic installer filename; Miniforge release provides platform-specific installers named like Miniforge3-*.sh
    MINIFORGE_SH="miniforge_installer.sh"
    # try download generic latest release asset pattern from github (best-effort)
    wget -q --show-progress "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O "$MINIFORGE_SH" || {
        echo "❌ Failed to download Miniforge installer"; exit 1
    }
    bash "$MINIFORGE_SH" -b -p "$CONDA_DIR" || { echo "❌ Failed to install Miniforge"; exit 1; }
    rm -f "$MINIFORGE_SH"
    echo "✅ Miniforge installed at $CONDA_DIR"
else
    echo "✅ Miniforge already exists at $CONDA_DIR"
fi

# Source conda hooks reliably
echo "Sourcing conda..."
# shellcheck source=/dev/null
if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    echo "❌ Expected conda.sh not found at $CONDA_DIR/etc/profile.d/conda.sh"; exit 1
fi
# Ensure conda command available in this shell
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)" || true

echo ""

# -------------------------
# Stage 4: Create and activate conda env (explicit python version)
# -------------------------
echo "STAGE 4: Creating/activating conda env '3dhope_rl' with python=3.10..."
CONDA_ENV_NAME="3dhope_rl"
DESIRED_PY="3.10"

# Create if missing
if ! "$CONDA_DIR/bin/conda" env list | awk '{print $1}' | grep -xq "$CONDA_ENV_NAME"; then
    echo "Creating conda environment: $CONDA_ENV_NAME (python=$DESIRED_PY)"
    "$CONDA_DIR/bin/conda" create -n "$CONDA_ENV_NAME" python="$DESIRED_PY" -y || { echo "❌ Failed to create conda env"; exit 1; }
else
    echo "✅ Conda env $CONDA_ENV_NAME already present"
fi

# Activate environment
echo "Activating conda environment: $CONDA_ENV_NAME"
# Prefer conda activate (requires conda.sh sourced above)
conda activate "$CONDA_ENV_NAME" || { echo "❌ Failed to activate conda env"; exit 1; }

# Ensure conda python is first and export PATH accordingly
export PATH="${CONDA_PREFIX:-$CONDA_DIR/envs/$CONDA_ENV_NAME}/bin:$PATH"
hash -r || true

echo "Environment verification:"
echo "  CONDA_PREFIX: ${CONDA_PREFIX:-N/A}"
echo "  Active conda environment: ${CONDA_DEFAULT_ENV:-N/A}"
echo "  Python path: $(which python)"
echo "  Python version: $(python --version 2>&1)"
echo "  Pip path: $(which pip)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 5: Poetry Installation and Configuration
# ═══════════════════════════════════════════════════════════════════════════════════
echo "STAGE 5: Poetry Setup"
echo "───────────────────────────────────────────────────────────────────────────────"

cd ~/codes/3dhope_rl/ || {
    echo "❌ Failed to change to project directory ~/codes/3dhope_rl/"; exit 1
}
echo "Current directory: $(pwd)"
echo ""

# Set scratch poetry location and preferred binary path
POETRY_HOME="/scratch/pramish_paudel/tools/poetry"
POETRY_BIN="$POETRY_HOME/bin/poetry"

# TODO: 
rm -rf /scratch/pramish_paudel/tools/poetry

# Remove ~/.local/bin from PATH to avoid accidentally using a global poetry
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -x "$HOME/.local/bin" | tr '\n' ':' | sed 's/:$//')
# Install poetry into scratch if missing
if [ -x "$POETRY_BIN" ]; then
    echo "✅ Found Poetry at: $POETRY_BIN"
else
    echo "📦 Installing Poetry to $POETRY_HOME..."
    mkdir -p "$POETRY_HOME"
    # Use the official installer via python inside the conda env (this ensures correct python used)
    curl -sSL https://install.python-poetry.org | POETRY_HOME="$POETRY_HOME" python - || {
        echo "❌ Failed to install Poetry to $POETRY_HOME"; exit 1
    }
    if [ ! -x "$POETRY_BIN" ]; then
        echo "❌ Poetry installation failed - $POETRY_BIN not found or not executable"; exit 1
    fi
    echo "✅ Poetry installed to $POETRY_HOME"
fi

# Put scratch poetry first in PATH to ensure it's used
export PATH="$POETRY_HOME/bin:$PATH"
hash -r || true

POETRY_PATH="$(command -v poetry || true)"
echo ""
echo "Poetry Information:"
echo "  Expected: $POETRY_BIN"
echo "  Actual:   ${POETRY_PATH:-N/A}"
if [ -n "$POETRY_PATH" ]; then
    echo "  Version:  $(poetry --version 2>/dev/null || echo 'unknown')"
fi

# Ensure we are using the exact poetry we installed
if [ "$POETRY_PATH" != "$POETRY_BIN" ]; then
    echo "❌ ERROR: Wrong Poetry binary is being used! Expected $POETRY_BIN but got ${POETY_PATH:-$POETRY_PATH}"
    echo "   To avoid this, ensure no other poetry is earlier in PATH."
    # fail fast to avoid installing with wrong poetry
    exit 1
fi
echo "  ✅ Confirmed: Using scratch Poetry"
echo ""


# Run `poetry install` but ensure we run poetry from the exact path and use conda python
POETRY_CMD="$POETRY_BIN"
POETRY_INSTALL_LOG="/tmp/poetry_install.log"

# TODO: 
$POETRY_CMD config virtualenvs.in-project true
echo "Running: $POETRY_CMD install --no-interaction --no-ansi"
if "$POETRY_CMD" install --no-interaction --no-ansi 2>&1 | tee "$POETRY_INSTALL_LOG"; then
    echo "✅ Poetry install succeeded"
else
    echo "⚠️ Poetry install failed — showing last 60 lines of log:"
    tail -n 60 "$POETRY_INSTALL_LOG" || true
    echo "Falling back to pip install -e . using current python ($CONDA_PREFIX/bin/python)..."
    # Always use the conda python
    "$CONDA_PREFIX/bin/pip" install -e . || { echo "❌ Fallback pip install failed"; exit 1; }
fi

# -------------------------
# Stage 8: Activate project's .venv (if created) OR keep conda env
# -------------------------
if [ -d ".venv" ]; then
    echo "Activating project .venv..."
    # shellcheck disable=SC1091
    source .venv/bin/activate || { echo "❌ Failed to activate .venv"; exit 1; }
    echo "✅ Using Poetry .venv: $(which python)"
else
    echo "⚠️ No .venv found — continuing using conda env: ${CONDA_DEFAULT_ENV:-N/A}"
fi

# quick verify that hydra (or other crucial libs) are importable
echo "Verifying important packages (hydra, omegaconf)..."
python - <<'PYTEST' || { echo "❌ Required python imports failed"; exit 1; }
try:
    import importlib, sys
    modnames = ["hydra", "omegaconf"]
    missing = []
    for m in modnames:
        try:
            importlib.import_module(m)
        except Exception as e:
            missing.append((m,str(e)))
    if missing:
        print("MISSING:", missing)
        sys.exit(2)
    else:
        print("All checks passed:", [importlib.import_module(m).__name__ for m in modnames])
except Exception as e:
    print("Import-time error:", str(e))
    raise
PYTEST

echo ""

# -------------------------
# Stage 9: GPU check
# -------------------------
echo "STAGE 9: GPU check (nvidia-smi):"
nvidia-smi || echo "⚠️ nvidia-smi failed or not present on this node"

# -------------------------
# Stage 10: Run training
# -------------------------
echo "✅ All dependencies installed and configured"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "STAGE 10: Starting RL training..."
echo "Training started at: $(date)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
# Install ThreedFront package
echo "Installing ThreedFront package..."
pip install -e ../ThreedFront || echo "⚠️  ThreedFront install failed"


export PYTHONUNBUFFERED=1
export DISPLAY=:0

# Use the active conda python to launch to avoid any confusion
# TODO: 
    # checkpoint_version=20 \
PYTHONPATH=. python -u  main.py +name=incremental_mi_floor_60_70__150_tv \
    load=pcnfeqr0 \
    checkpoint_version=20 \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data=/scratch/pramish_paudel/ \
    dataset.data.path_to_dataset_files=/home/pramish_paudel/codes/ThreedFront/dataset_files \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=300000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=100.0 \
    experiment.reset_lr_scheduler=true \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    experiment.training.checkpointing.every_n_train_steps=2000 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=True \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.classifier_free_guidance.use_floor=true \
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    dataset.sdf_cache_dir=/scratch/pramish_paudel/bedroom_sdf_cache/ \
    dataset.accessibility_cache_dir=/scratch/pramish_paudel/bedroom_accessibility_cache/ \
    algorithm.custom.num_classes=22 \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.obj_vec_len=30 \
    algorithm.custom.obj_diff_vec_len=30 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.dataset_directory=bedroom \
    dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
    dataset.data.room_type=bedroom \
    algorithm.custom.old=False \
    algorithm.ddpo.dynamic_constraint_rewards.reward_base_dir=/home/pramish_paudel/codes/3dhope_rl/dynamic_constraint_rewards \
    algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with tv stand and desk and chair for working." \
    algorithm.ddpo.dynamic_constraint_rewards.agentic=True \
    algorithm.ddpo.dynamic_constraint_rewards.universal_weight=0.0 \
    algorithm.ddpo.batch_size=192 \
    experiment.training.batch_size=192 \
    experiment.validation.batch_size=192 \
    experiment.test.batch_size=192 \
    algorithm.ddpo.incremental_training=true \
    algorithm.ddpo.training_steps_start=0 \
    algorithm.ddpo.joint_training=False \
    algorithm.ddpo.increments='[60,70,80,90,100,110,120,130,140,150]' \
    algorithm.ddpo.increment_type="constant" \
    algorithm.ddpo.increment_linear_slope=40 \
    algorithm.ddpo.training_iter_per_increment=6000 \

# -------------------------
# Final status
# -------------------------
EXIT_CODE=$?
echo ""
echo "════════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully at: $(date)"
    exit 0
else
    echo "❌ Training failed with exit code $EXIT_CODE at: $(date)"
    exit $EXIT_CODE
fi
echo "════════════════════════════════════════════════════════════════════════"
