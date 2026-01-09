# run in ajad pc
# Define your SSH destination
# ssh_machine="s_01k9btwd5h51p4e1qsrv1anxcg@ssh.lightning.ai"
# remote_dir="/teamspace/studios/this_studio"

# # # Upload all three files via scp
# scp -v \
#   /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_accessibility_cache.zip \
#   /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache.zip \
#   "$ssh_machine:$remote_dir/"

# scp -v \
#   /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom.zip \
#   "$ssh_machine:$remote_dir/"

# run in lightning pc

#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# EDIT THESE
# Path where you want the 3dhope_rl repo cloned (absolute or relative)
3DHOPE_DIR="${HOME}/codes/3dhope_rl"    # e.g. /home/you/codes/3dhope_rl
THREEDFRONT_DIR="${HOME}/codes/ThreedFront" # e.g. /home/you/codes/ThreedFront

# Optional: WANDB API key (export this before running, or set here)
# export WANDB_API_KEY="your_wandb_api_key_here"
# -------------------------

# helper
ensure_dir() {
  mkdir -p "$1"
}

echo "Creating workspace directories..."
ensure_dir "$(dirname "$3DHOPE_DIR")"
ensure_dir "$(dirname "$THREEDFRONT_DIR")"

# clone repos if not present
if [[ ! -d "$3DHOPE_DIR" ]]; then
  git clone https://github.com/3D-Hope/3dhope_rl.git "$3DHOPE_DIR"
else
  echo "3dhope_rl already exists at $3DHOPE_DIR"
fi

if [[ ! -d "$THREEDFRONT_DIR" ]]; then
  git clone https://github.com/3D-Hope/ThreedFront.git "$THREEDFRONT_DIR"
else
  echo "ThreedFront already exists at $THREEDFRONT_DIR"
fi

# Install poetry if missing
if ! command -v poetry >/dev/null 2>&1; then
  echo "Installing poetry..."
  curl -sSL https://install.python-poetry.org | python3 -
  # ensure poetry is on PATH for this shell (may need to adjust depending on install location)
  export PATH="$HOME/.local/bin:$PATH"
fi

# Install python deps via poetry in the 3dhope_rl repo
pushd "$3DHOPE_DIR" >/dev/null
echo "Installing python deps with poetry..."
poetry install --no-interaction 2>&1 | tee /tmp/poetry_install.log || { echo "poetry install failed"; exit 6; }
popd >/dev/null

# Install ThreedFront in editable mode (use poetry-run pip to ensure same python)
pushd "$THREEDFRONT_DIR" >/dev/null
# Use pip from the poetry environment of 3dhope_rl to install dependency in the same interpreter
# If you want to use the system python -> pip install -e .
if command -v poetry >/dev/null 2>&1; then
  echo "Installing ThreedFront (editable) using poetry-run pip from 3dhope_rl environment..."
  pushd "$3DHOPE_DIR" >/dev/null
  POETRY_PY=$(poetry env info -p 2>/dev/null || true)
  if [[ -n "$POETRY_PY" ]]; then
    # use this poetry environment's python/pip
    "$POETRY_PY/bin/pip" install -e "$THREEDFRONT_DIR" || echo "⚠️ pip install -e ThreedFront failed"
  else
    # fallback: use system pip
    pip install -e "$THREEDFRONT_DIR" || echo "⚠️ pip install -e ThreedFront failed"
  fi
  popd >/dev/null
else
  pip install -e "$THREEDFRONT_DIR" || echo "⚠️ ThreedFront install failed"
fi
popd >/dev/null

# Install gdown if missing
if ! python3 -c "import gdown" >/dev/null 2>&1; then
  echo "Installing gdown..."
  python3 -m pip install --user gdown
  export PATH="$HOME/.local/bin:$PATH"
fi

# Download zips via gdown if not already uploaded via scp
cd "$3DHOPE_DIR"

# Where the script expects the zip files: either UPLOADED_DIR (from scp) or download into current dir
if [[ -d "$UPLOADED_DIR" && -f "$UPLOADED_DIR/bedroom_accessibility_cache.zip" ]]; then
  echo "Using uploaded zip files from $UPLOADED_DIR"
  cp "$UPLOADED_DIR"/bedroom_accessibility_cache.zip "$3DHOPE_DIR"/ || true
  cp "$UPLOADED_DIR"/bedroom_sdf_cache.zip "$3DHOPE_DIR"/ || true
  cp "$UPLOADED_DIR"/bedroom.zip "$3DHOPE_DIR"/ || true
else
  # fallback: attempt to gdown
  echo "Attempting to download files via gdown (fallback)."
  gdown --fuzzy "https://drive.google.com/file/d/1juiwq83aHO_85BxfPT5KPU7eqYOWgjqL/view?usp=drive_link" || echo "gdown failed for accessibility cache"
  gdown --fuzzy "https://drive.google.com/file/d/1O0WqaoyYZNIf3aO5JqEwkBBNP2y4sdOe/view?usp=drive_link" || echo "gdown failed for sdf cache"
  gdown --fuzzy "https://drive.google.com/file/d/1XFcaS1oBn-32VWMk24BOYYAZIG4K9qZN/view?usp=drive_link" || echo "gdown failed for bedroom dataset"
fi

# Unzip into the right target dirs
echo "Unzipping uploaded files..."
unzip -oq bedroom_accessibility_cache.zip -d "$3DHOPE_DIR/bedroom_accessibility_cache/" || echo "unzip bedroom_accessibility_cache failed/was missing"
unzip -oq bedroom_sdf_cache.zip -d "$3DHOPE_DIR/bedroom_sdf_cache/" || echo "unzip bedroom_sdf_cache failed/was missing"
unzip -oq bedroom.zip -d "$3DHOPE_DIR/bedroom/" || echo "unzip bedroom.zip failed/was missing"
echo "Unzipping done."

# Wandb login (optional)
if [[ -n "${WANDB_API_KEY-}" ]]; then
  echo "Logging into wandb..."
  if ! command -v wandb >/dev/null 2>&1; then
    python3 -m pip install --user wandb
    export PATH="$HOME/.local/bin:$PATH"
  fi
  wandb login --relogin "$WANDB_API_KEY" || echo "⚠️ wandb login failed, continuing..."
else
  echo "WANDB_API_KEY not set — skipping wandb login."
fi

# Environment variables recommended
export PYTHONUNBUFFERED=1
export DISPLAY=:0

# Run training using poetry-run to avoid activation issues
echo "Starting training at: $(date)"

pushd "$3DHOPE_DIR" >/dev/null

# Replace load=rrudae6n etc with your desired config. Use poetry run to run inside the installed env.
poetry run python -u main.py +name=vaastu2 \
    load=cmdpm5nv \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data="$3DHOPE_DIR/" \
    dataset.data.path_to_dataset_files="$THREEDFRONT_DIR/dataset_files" \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    experiment.training.max_steps=1020000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=50.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=128 \
    experiment.training.checkpointing.every_n_train_steps=500 \
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
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    experiment.training.precision=bf16-mixed \
    experiment.validation.precision=bf16-mixed \
    experiment.test.precision=bf16-mixed \
    experiment.matmul_precision=medium \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.ddpo.dynamic_constraint_rewards.stats_path="$3DHOPE_DIR/dynamic_constraint_rewards/stats.json" \
    dataset.sdf_cache_dir="$3DHOPE_DIR/bedroom_sdf_cache/" \
    dataset.accessibility_cache_dir="$3DHOPE_DIR/bedroom_accessibility_cache/" \
    algorithm.custom.num_classes=22 \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.obj_vec_len=30 \
    algorithm.custom.obj_diff_vec_len=30 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.dataset_directory=bedroom \
    dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
    algorithm.custom.old=True \
    dataset.data.room_type=bedroom \
    algorithm.ddpo.dynamic_constraint_rewards.user_query="I want to follow Vaastu for bedroom layout. The beds headboard should face east."

popd >/dev/null

echo "Training script completed (exit status $? if returned)."
