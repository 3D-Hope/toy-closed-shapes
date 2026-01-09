#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --time=1-06:00:00 
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

# Set environment variables for headless rendering
export PYTHONUNBUFFERED=1
export DISPLAY=:0

# Install Xvfb for virtual display (needed for rendering during validation)
echo "Installing Xvfb for headless rendering..."
conda install -y -c conda-forge xorg-libxfb xvfb-run xorg-libxrender xorg-libxext
echo "✅ Xvfb installed"


# 🚀 Run training
echo "Starting training at: $(date)"

# Create logs directory for individual training runs
mkdir -p training_logs

echo "="*80
echo "STARTING SIMULTANEOUS TRAINING RUNS"
echo "="*80

# Run DiffuScene baseline in background
echo "[1/2] Starting DiffuScene baseline training..."
PYTHONPATH=. python -u main.py +name=diffuscene_baseline \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data=/scratch/pramish_paudel/ \
    dataset.data.path_to_dataset_files=/home/pramish_paudel/codes/ThreedFront/dataset_files \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_diffuscene \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.loss=true \
    experiment.training.max_steps=1e6 \
    resume=jfgw3io6 \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    > training_logs/diffuscene_baseline.log 2>&1 &

DIFFUSCENE_PID=$!
echo "✓ DiffuScene baseline started with PID: $DIFFUSCENE_PID"

# Small delay to avoid simultaneous initialization issues
sleep 120

# Run MiDiffusion baseline in background
echo "[2/2] Starting MiDiffusion baseline training..."
PYTHONPATH=. python -u main.py +name=continuous_midiffusion_baseline \
    resume=pfksynuz \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data=/scratch/pramish_paudel/ \
    dataset.data.path_to_dataset_files=/home/pramish_paudel/codes/ThreedFront/dataset_files \
    dataset._name=custom_scene \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.trainer=ddpm \
    experiment.find_unused_parameters=True \
    algorithm.classifier_free_guidance.use=False \
    algorithm.classifier_free_guidance.weight=0 \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    > training_logs/midiffusion_baseline.log 2>&1 &

MIDIFFUSION_PID=$!
echo "✓ MiDiffusion baseline started with PID: $MIDIFFUSION_PID"

echo ""
echo "="*80
echo "BOTH TRAINING RUNS STARTED"
echo "="*80
echo "DiffuScene PID    : $DIFFUSCENE_PID"
echo "MiDiffusion PID   : $MIDIFFUSION_PID"
echo "Logs directory    : training_logs/"
echo ""
echo "Monitor progress with:"
echo "  tail -f training_logs/diffuscene_baseline.log"
echo "  tail -f training_logs/midiffusion_baseline.log"
echo "="*80
echo ""

# Monitor GPU usage
echo "Initial GPU status:"
nvidia-smi

# Wait for both processes to complete
echo ""
echo "Waiting for training processes to complete..."
echo "(This will run until both finish or job time limit is reached)"
echo ""

# Function to check if process is still running
check_process() {
    if ps -p $1 > /dev/null 2>&1; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Monitor both processes
while true; do
    DIFFUSCENE_RUNNING=false
    MIDIFFUSION_RUNNING=false
    
    if check_process $DIFFUSCENE_PID; then
        DIFFUSCENE_RUNNING=true
    fi
    
    if check_process $MIDIFFUSION_PID; then
        MIDIFFUSION_RUNNING=true
    fi
    
    # Check if both are done
    if ! $DIFFUSCENE_RUNNING && ! $MIDIFFUSION_RUNNING; then
        echo ""
        echo "Both training processes have completed!"
        break
    fi
    
    # Print status every 5 minutes
    sleep 300
    echo "[$(date)] Status - DiffuScene: $(if $DIFFUSCENE_RUNNING; then echo 'RUNNING'; else echo 'STOPPED'; fi), MiDiffusion: $(if $MIDIFFUSION_RUNNING; then echo 'RUNNING'; else echo 'STOPPED'; fi)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1
done

# Wait for any remaining background processes
wait

# Check exit status of both
DIFFUSCENE_EXIT=0
MIDIFFUSION_EXIT=0

wait $DIFFUSCENE_PID 2>/dev/null
DIFFUSCENE_EXIT=$?

wait $MIDIFFUSION_PID 2>/dev/null
MIDIFFUSION_EXIT=$?

echo ""
echo "="*80
echo "TRAINING RESULTS"
echo "="*80

echo ""
echo "="*80
echo "TRAINING RESULTS"
echo "="*80
echo "DiffuScene exit code    : $DIFFUSCENE_EXIT $(if [ $DIFFUSCENE_EXIT -eq 0 ]; then echo '✅'; else echo '❌'; fi)"
echo "MiDiffusion exit code   : $MIDIFFUSION_EXIT $(if [ $MIDIFFUSION_EXIT -eq 0 ]; then echo '✅'; else echo '❌'; fi)"
echo "="*80
echo ""

# Overall exit status (fail if either failed)
if [ $DIFFUSCENE_EXIT -eq 0 ] && [ $MIDIFFUSION_EXIT -eq 0 ]; then
    echo "✅ Both training runs completed successfully at: $(date)"
    EXIT_STATUS=0
else
    echo "❌ One or more training runs failed at: $(date)"
    echo "   Check logs in training_logs/ for details"
    EXIT_STATUS=1
fi

echo "Job completed at: $(date)"
exit $EXIT_STATUS
