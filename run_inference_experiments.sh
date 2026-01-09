#!/bin/bash

# Script to run inference experiments with different configurations
# Experiments for runs: bgdrozky (Flux Transformer), juy0jvto (Flux Transformer)
# Testing: DDPM/DDIM schedulers with EMA=True only

set -e  # Exit on error

# Create logs directory if it doesn't exist
LOGS_DIR="logs/inference_experiments"
mkdir -p "$LOGS_DIR"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOGS_DIR/experiment_batch_${TIMESTAMP}.log"
RESULTS_CSV="$LOGS_DIR/experiment_results_${TIMESTAMP}.csv"

# Array to store experiment results
declare -a EXPERIMENT_RESULTS

# Function to log message to both console and file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

echo "Logging all output to: $LOG_FILE"
echo ""

# Color codes for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print experiment header
print_experiment_header() {
    local run_id=$1
    local scheduler=$2
    local exp_num=$3
    local model_name=$4
    
    log ""
    log "################################################################################"
    log "################################################################################"
    log "###"
    log "###  EXPERIMENT $exp_num"
    log "###"
    log "###  Model: $model_name"
    log "###  Run ID: $run_id"
    log "###  Scheduler: $scheduler"
    log "###  EMA: True (always)"
    log "###"
    log "################################################################################"
    log "################################################################################"
    log ""
}

# Function to run experiment
run_experiment() {
    local run_id=$1
    local scheduler=$2
    local exp_num=$3
    local algorithm=$4
    local model_name=$5
    local use_floor=${6:-False}  # Optional parameter, default False
    local is_rl=${7:-False}  # Optional parameter to indicate RL models
    
    # Set num_timesteps based on scheduler
    local num_timesteps
    if [ "$scheduler" = "ddpm" ]; then
        num_timesteps=1000
    else
        num_timesteps=150
    fi
    
    # Set trainer based on whether it's RL model
    local trainer
    if [ "$is_rl" = "True" ]; then
        trainer="rl_score"
    else
        trainer="ddpm"
    fi
    
    # Always use EMA=True
    local ema="True"
    
    print_experiment_header "$run_id" "$scheduler" "$exp_num" "$model_name"
    
    log "Configuration:"
    log "  - Model: $model_name"
    log "  - load: $run_id"
    log "  - algorithm: $algorithm"
    log "  - algorithm.trainer: $trainer"
    log "  - algorithm.noise_schedule.scheduler: $scheduler"
    log "  - algorithm.ema.use: $ema (always True)"
    log "  - algorithm.classifier_free_guidance.use_floor: $use_floor"
    log "  - num_scenes: 256"
    log "  - algorithm.noise_schedule.ddim.num_inference_timesteps: $num_timesteps"
    log ""
    log "Starting experiment at: $(date)"
    log ""
    
    # Run the experiment and redirect output to both console and log file
    PYTHONPATH=. python scripts/custom_sample_and_render.py \
        dataset=custom_scene \
        dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
        dataset.max_num_objects_per_scene=12 \
        +num_scenes=1000 \
        algorithm=$algorithm \
        experiment.find_unused_parameters=True \
        algorithm.classifier_free_guidance.use=False \
        algorithm.classifier_free_guidance.weight=0 \
        algorithm.num_additional_tokens_for_sampling=0 \
        algorithm.custom.loss=true \
        algorithm.noise_schedule.ddim.num_inference_timesteps=$num_timesteps \
        algorithm.trainer=$trainer \
        load=$run_id \
        algorithm.noise_schedule.scheduler=$scheduler \
        algorithm.ema.use=True \
        experiment.test.batch_size=196 \
        algorithm.classifier_free_guidance.use_floor=$use_floor 2>&1 | tee -a "$LOG_FILE"
    
    # Find the generated pkl file (most recent sampled_scenes_results.pkl)
    local output_dir=$(find outputs -type f -name "sampled_scenes_results.pkl" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- | xargs dirname)
    local pkl_file="$output_dir/sampled_scenes_results.pkl"
    
    log ""
    log "Experiment completed at: $(date)"
    
    # Store result information
    if [ -f "$pkl_file" ]; then
        log "✅ Generated results saved to: $pkl_file"
        # Store in results array: exp_num,model_name,run_id,scheduler,ema,num_timesteps,pkl_file
        EXPERIMENT_RESULTS+=("$exp_num,$model_name,$run_id,$scheduler,$ema,$num_timesteps,$pkl_file")
    else
        log "⚠️  Warning: Could not find generated pkl file for this experiment"
        EXPERIMENT_RESULTS+=("$exp_num,$model_name,$run_id,$scheduler,$ema,$num_timesteps,NOT_FOUND")
    fi
    
    log ""
    log "################################################################################"
    log "###  END OF EXPERIMENT $exp_num"
    log "################################################################################"
    log ""
    log ""
}

# Main execution
log ""
log "================================================================================"
log "                    INFERENCE EXPERIMENTS BATCH"
log "================================================================================"
log ""
log "Total experiments to run: 4"
log ""
log "Run IDs:"
log "  1. xn1h20rz (Flux Transformer RL, 2-stage)"
log "  2. f6vipupt (Flux Transformer RL, 1-stage)"
log ""
log "Schedulers: DDPM (1000 timesteps), DDIM (150 timesteps)"
log "EMA settings: True (always)"
log ""
log "Starting batch at: $(date)"
log "================================================================================"
log ""

exp_counter=1

# Experiments for run xn1h20rz (Flux Transformer RL, 2-stage)
log ""
log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
log ">>>"
log ">>>  STARTING EXPERIMENTS FOR RUN: xn1h20rz (Flux Transformer RL, 2-stage)"
log ">>>"
log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
log ""

run_experiment "xn1h20rz" "ddpm" "$exp_counter" "scene_diffuser_flux_transformer" "Flux Transformer RL (2-stage)" "False" "True"
((exp_counter++))

run_experiment "xn1h20rz" "ddim" "$exp_counter" "scene_diffuser_flux_transformer" "Flux Transformer RL (2-stage)" "False" "True"
((exp_counter++))

# Experiments for run f6vipupt (Flux Transformer RL, 1-stage)
log ""
log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
log ">>>"
log ">>>  STARTING EXPERIMENTS FOR RUN: f6vipupt (Flux Transformer RL, 1-stage)"
log ">>>"
log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
log ""

run_experiment "f6vipupt" "ddpm" "$exp_counter" "scene_diffuser_flux_transformer" "Flux Transformer RL (1-stage)" "False" "True"
((exp_counter++))

run_experiment "f6vipupt" "ddim" "$exp_counter" "scene_diffuser_flux_transformer" "Flux Transformer RL (1-stage)" "False" "True"
((exp_counter++))

# # Experiments for run jfgw3io6 (DiffuScene) - COMMENTED OUT
# log ""
# log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
# log ">>>"
# log ">>>  STARTING EXPERIMENTS FOR RUN: jfgw3io6 (DiffuScene)"
# log ">>>"
# log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
# log ""
# 
# run_experiment "jfgw3io6" "ddpm" "$exp_counter" "scene_diffuser_diffuscene" "DiffuScene"
# ((exp_counter++))
# 
# run_experiment "jfgw3io6" "ddim" "$exp_counter" "scene_diffuser_diffuscene" "DiffuScene"
# ((exp_counter++))

# # Experiments for run pfksynuz (Continuous MI) - COMMENTED OUT
# log ""
# log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
# log ">>>"
# log ">>>  STARTING EXPERIMENTS FOR RUN: pfksynuz (Continuous MI)"
# log ">>>"
# log ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
# log ""
# 
# run_experiment "pfksynuz" "ddpm" "$exp_counter" "scene_diffuser_midiffusion" "Continuous MI"
# ((exp_counter++))
# 
# run_experiment "pfksynuz" "ddim" "$exp_counter" "scene_diffuser_midiffusion" "Continuous MI"
# ((exp_counter++))

# Summary
log ""
log "================================================================================"
log "                    ALL EXPERIMENTS COMPLETED"
log "================================================================================"
log ""
log "Total experiments run: 4"
log "Completed at: $(date)"
log ""
log "Summary of experiments:"
log ""
log "Run: xn1h20rz (Flux Transformer RL, 2-stage)"
log "  1. DDPM (1000) + EMA=True"
log "  2. DDIM (150) + EMA=True"
log ""
log "Run: f6vipupt (Flux Transformer RL, 1-stage)"
log "  3. DDPM (1000) + EMA=True"
log "  4. DDIM (150) + EMA=True"
log ""
log "================================================================================"
log ""

# Generate CSV file with results
log "Generating results CSV file..."
log ""

# CSV Header
echo "experiment_num,model_name,run_id,scheduler,ema,num_timesteps,pkl_file_path" > "$RESULTS_CSV"

# Add all experiment results
for result in "${EXPERIMENT_RESULTS[@]}"; do
    echo "$result" >> "$RESULTS_CSV"
done

log "✅ Results CSV saved to: $RESULTS_CSV"
log ""

# Print the results table to log
log "================================================================================"
log "                    EXPERIMENT RESULTS TABLE"
log "================================================================================"
log ""
log "$(printf '%-5s %-20s %-12s %-10s %-6s %-15s %-s\n' 'Exp#' 'Model' 'Run ID' 'Scheduler' 'EMA' 'Timesteps' 'PKL File')"
log "$(printf '%-5s %-20s %-12s %-10s %-6s %-15s %-s\n' '----' '--------------------' '------------' '----------' '------' '---------------' '--------')"

for result in "${EXPERIMENT_RESULTS[@]}"; do
    IFS=',' read -r exp_num model_name run_id scheduler ema num_timesteps pkl_file <<< "$result"
    # Truncate pkl path for display
    pkl_display=$(echo "$pkl_file" | sed 's|outputs/||')
    log "$(printf '%-5s %-20s %-12s %-10s %-6s %-15s %s\n' "$exp_num" "$model_name" "$run_id" "$scheduler" "$ema" "$num_timesteps" "$pkl_display")"
done

log ""
log "================================================================================"
log ""
log "Full paths in CSV file: $RESULTS_CSV"
log "Log file saved to: $LOG_FILE"
log ""
log "To view the CSV:"
log "  cat $RESULTS_CSV"
log ""
log "To view results in column format:"
log "  column -t -s',' $RESULTS_CSV"
log ""
log "================================================================================"
log ""