#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print debug information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Scratch contents: $(ls /scratch/pramish_paudel)"
free -h
df -h
set -x

# if [ ! -f "/scratch/pramish_paudel/model.ckpt" ]; then
#     echo "Copying model checkpoint"
#     rsync -aHzv /home/pramish_paudel/3dhope_data/model.ckpt /scratch/pramish_paudel/
# else
#     echo "✅ Model checkpoint already exists in scratch."
# fi
if [ ! -d "/scratch/pramish_paudel/bedroom" ]; then
    echo "copying data "
    rsync -aHzv /home/pramish_paudel/3dhope_data/bedroom.zip /scratch/pramish_paudel/

    # Unzip dataset
    cd /scratch/pramish_paudel/
    unzip bedroom.zip
    rm bedroom.zip
else
    echo "✅ Bedroom dataset already exists in scratch."
fi

# 🔧 Setup Miniconda
CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"

if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda/Miniforge..."
    mkdir -p /scratch/pramish_paudel/tools/
    cd /scratch/pramish_paudel/tools/

    # Download Miniforge (better than Miniconda for conda-forge packages)
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O miniforge.sh

    # Install silently
    bash miniforge.sh -b -p $CONDA_DIR
    rm miniforge.sh
    echo "✅ Miniforge installed at $CONDA_DIR"
else
    echo "✅ Miniforge already exists at $CONDA_DIR"
fi

# Source conda
echo "Sourcing conda from: $CONDA_DIR/etc/profile.d/conda.sh"
source "$CONDA_DIR/etc/profile.d/conda.sh"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# 🐍 Create and setup environment
if ! conda env list | grep -q "3dhope_rl"; then
    echo "Creating conda environment: 3dhope_rl"
    conda create -n 3dhope_rl python=3.10 -y

    echo "Activating conda environment: 3dhope_rl"
    conda activate 3dhope_rl


    echo "✅ Environment setup complete!"
else
    echo "✅ Environment '3dhope_rl' already exists"
    conda activate 3dhope_rl
fi

# Verify setup
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"


# Check GPU
echo "GPU information:"
nvidia-smi

# 📂 Move to your training script directory
cd ~/codes/3dhope_rl/

# Verify we're in the right directory
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la

# Install project dependencies
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
wandb login
pip install -e ../ThreedFront
export PYTHONUNBUFFERED=1
export DISPLAY=:0


# 🚀 Run training
echo "Starting training at: $(date)"
export PYTHONUNBUFFERED=1
PYTHONPATH=. python -u main.py +name=flux_transformer_floor_cond \
    resume=eviaimru \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data=/scratch/pramish_paudel/ \
    dataset.data.path_to_dataset_files=/home/pramish_paudel/codes/ThreedFront/dataset_files \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_flux_transformer \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.ema.use=True \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    experiment.training.batch_size=128 \
    experiment.validation.batch_size=128 \
    experiment.test.batch_size=128 \
    experiment.training.optim.accumulate_grad_batches=2 \
    experiment.training.precision=bf16-mixed \
    experiment.validation.precision=bf16-mixed \
    experiment.test.precision=bf16-mixed \
    experiment.matmul_precision=medium



# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully at: $(date)"
else
    echo "❌ Training failed at: $(date)"
fi

echo "Job completed at: $(date)"
