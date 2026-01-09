#!/bin/bash

# Batch evaluation script for multiple pickle files - PARALLEL VERSION
# Each pickle file is processed (render + evaluate) in parallel
cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront

# Define the base directory
BASE_DIR="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/"

# Array of pickle files to evaluate with floor conditioning info
# Format: "pkl_file|use_floor"
PKL_FILES_WITH_FLAGS=(
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-27/06-10-10/sampled_scenes_results.pkl|with_floor" # scaled ver 16 6sq3fgpv
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-27/06-17-28/sampled_scenes_results.pkl|with_floor" # scaled ver 6 
    "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-27/08-07-45/sampled_scenes_results.pkl|with_floor" # scaled ver 20 6sq3fgpv
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-24/16-04-38/sampled_scenes_results.pkl|with_floor"
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-15/05-16-46/sampled_scenes_results.pkl|with_floor" # ver 10 universal + dynamic rl
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-13/04-33-12/sampled_scenes_results.pkl|with_floor" # mi floor only dynamic rl, left side
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-13/04-39-51/sampled_scenes_results.pkl|with_floor" # mi floor universal + dynamic rl, right side


    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-14/08-52-34/sampled_scenes_results.pkl|with_floor" #baseline with loss corrected
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/new_Mi/MiDiffusion/output/predicted_results/results.pkl|with_floor" # mi floor MiDiffusion officaial ckpt
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-04/20-44-55/sampled_scenes_results.pkl|with_floor" # # mi floor living no obj
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/04-11-45/sampled_scenes_results.pkl|with_floor" # mi floor obj living
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/04-01-54/sampled_scenes_results.pkl|with_floor" # mi floor no obj living
    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/04-08-05/sampled_scenes_results.pkl|no_floor"  # mi no floor obj living(still training 80k epochs)


    # "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-27/13-04-43/sampled_scenes_results.pkl|with_floor" # Newest rl with floor
    # "$BASE_DIR/outputs/2025-10-22/11-12-45/sampled_scenes_results.pkl|no_floor"  # 45
    # "$BASE_DIR/outputs/2025-10-22/11-20-15/sampled_scenes_results.pkl|no_floor"   #50
    # "$BASE_DIR/outputs/2025-10-22/11-25-41/sampled_scenes_results.pkl|no_floor"   #55
    # "$BASE_DIR/outputs/2025-10-22/11-25-09/sampled_scenes_results.pkl|no_floor"   #35
    # "$BASE_DIR/outputs/2025-10-22/11-35-42/sampled_scenes_results.pkl|no_floor"   #40
    # "$BASE_DIR/outputs/2025-10-22/04-42-33/sampled_scenes_results.pkl|no_floor"   # 3. 1 stage
    # "$BASE_DIR/outputs/2025-10-22/04-45-30/sampled_scenes_results.pkl|no_floor"   # 4. 1 stage
    # "$BASE_DIR/outputs/2025-10-18/08-37-12/sampled_scenes_results.pkl|with_floor" # 5. Continuous MiDiffusion Floor, rrudae6n, ddpm, True, 1000
    # "$BASE_DIR/outputs/2025-10-18/08-41-48/sampled_scenes_results.pkl|with_floor" # 6. Continuous MiDiffusion Floor, rrudae6n, ddim, True, 150
)

# Get total number of files
TOTAL=${#PKL_FILES_WITH_FLAGS[@]}

echo "==============================================="
echo "Parallel Batch Evaluation Script"
echo "Total files to process: $TOTAL"
echo "==============================================="

# Set concurrency level: Number of pickle files to process in parallel
# Each job will: render its PKL → run all 6 metrics on that PKL
# Default: 4 parallel jobs (one per pickle file)
CONCURRENCY=${CONCURRENCY:-4}
echo "Concurrency level: $CONCURRENCY parallel jobs"
echo "Each job processes ONE complete pickle file (render + all metrics)"

# Log file for summary
LOG_FILE="batch_eval_$(date +%Y%m%d_%H%M%S).log"
echo "Batch Evaluation Log - $(date)" > "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "Concurrency: $CONCURRENCY parallel jobs" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Activate conda environment
echo ""
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate midiffusion

# ========================================
# PARALLEL PROCESSING with Semaphore
# ========================================

# Create a FIFO semaphore to limit concurrency
FIFO="/tmp/batch_eval_fifo.$$"
mkfifo "$FIFO"
exec 3<> "$FIFO"
rm "$FIFO"

# Seed tokens for concurrency control
for i in $(seq 1 $CONCURRENCY); do
    echo >&3
done

# Process function for each pickle file
process_entry() {
    local PKL_FILE="$1"
    local FLOOR_FLAG="$2"
    local IDX="$3"
    
    # Create a unique log for this job
    local JOB_LOG="${LOG_FILE%.log}_job${IDX}.log"

    echo "" | tee -a "$LOG_FILE" "$JOB_LOG"
    echo "-----------------------------------------------" | tee -a "$LOG_FILE" "$JOB_LOG"
    echo "[Job ${IDX}] START processing: $(basename $(dirname "$PKL_FILE")) [${FLOOR_FLAG}]" | tee -a "$LOG_FILE" "$JOB_LOG"
    echo "[Job ${IDX}] Started at: $(date)" | tee -a "$LOG_FILE" "$JOB_LOG"

    if [ ! -f "$PKL_FILE" ]; then
        echo "[Job ${IDX}] SKIPPED (not found): $PKL_FILE" | tee -a "$LOG_FILE" "$JOB_LOG"
        echo >&3  # Release semaphore token
        return
    fi

    # # Remove PNG files in the directory containing the PKL file (if any)
    # # local PKL_DIR
    # PKL_DIR=$(dirname "$PKL_FILE")
    # if compgen -G "$PKL_DIR/*.png" > /dev/null; then
    #     echo "[Job ${IDX}] Removing PNG files in $PKL_DIR" | tee -a "$LOG_FILE" "$JOB_LOG"
    #     rm -f "$PKL_DIR"/*.png
    # fi

    # # -------- Render Phase --------
    # START_TIME_RENDER=$(date +%s)
    # # if [ "$FLOOR_FLAG" = "with_floor" ]; then
    # #     RENDER_CMD=(python scripts/render_results.py "$PKL_FILE" --no_texture --retrieve_by_size)
    # # else
    # #     RENDER_CMD=(python scripts/render_results.py "$PKL_FILE" --no_texture --without_floor)
    # # fi
    # RENDER_CMD=(python scripts/render_results.py "$PKL_FILE" --no_texture --retrieve_by_size)
    # echo "[Job ${IDX}] Running render: ${RENDER_CMD[*]}" | tee -a "$LOG_FILE" "$JOB_LOG"
    # if "${RENDER_CMD[@]}" >> "$JOB_LOG" 2>&1; then
    #     END_TIME_RENDER=$(date +%s)
    #     DURATION_RENDER=$((END_TIME_RENDER - START_TIME_RENDER))
    #     echo "[Job ${IDX}] ✅ Rendered in ${DURATION_RENDER}s [${FLOOR_FLAG}]" | tee -a "$LOG_FILE" "$JOB_LOG"
    # else
    #     echo "[Job ${IDX}] ❌ Rendering FAILED for $PKL_FILE" | tee -a "$LOG_FILE" "$JOB_LOG"
    #     echo >&3  # Release semaphore token
    #     return
    # fi

    # -------- Evaluation Phase --------
    START_TIME_EVAL=$(date +%s)
    EVAL_SUCCESS=true
    if [ "$FLOOR_FLAG" = "with_floor" ]; then
        FLOOR_EVAL_FLAG=""
    else
        FLOOR_EVAL_FLAG="--no_floor"
    fi

    echo "[Job ${IDX}] [1/7] Computing FID scores... [${FLOOR_FLAG}]" | tee -a "$LOG_FILE" "$JOB_LOG"
    if ! python scripts/compute_fid_scores.py "$PKL_FILE" \
        --output_directory ./fid_tmps \
        --no_texture \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ >> "$JOB_LOG" 2>&1; then
        EVAL_SUCCESS=false
    fi

    echo "[Job ${IDX}] [2/7] Computing KID scores... [${FLOOR_FLAG}]" | tee -a "$LOG_FILE" "$JOB_LOG"
    if ! python scripts/compute_fid_scores.py "$PKL_FILE" \
        --compute_kid \
        --output_directory ./fid_tmps \
        --no_texture \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ >> "$JOB_LOG" 2>&1; then
        EVAL_SUCCESS=false
    fi

    echo "[Job ${IDX}] [3/7] Running bbox analysis..." | tee -a "$LOG_FILE" "$JOB_LOG"
    if ! python scripts/bbox_analysis.py "$PKL_FILE" >> "$JOB_LOG" 2>&1; then
        EVAL_SUCCESS=false
    fi

    echo "[Job ${IDX}] [4/7] Computing KL divergence..." | tee -a "$LOG_FILE" "$JOB_LOG"
    if ! python scripts/evaluate_kl_divergence_object_category.py "$PKL_FILE" \
        --output_directory ./kl_tmps >> "$JOB_LOG" 2>&1; then
        EVAL_SUCCESS=false
    fi

    echo "[Job ${IDX}] [5/7] Calculating object count..." | tee -a "$LOG_FILE" "$JOB_LOG"
    if ! python scripts/calculate_num_obj.py "$PKL_FILE" >> "$JOB_LOG" 2>&1; then
        EVAL_SUCCESS=false
    fi

    

    echo "[Job ${IDX}] [7/7] Running classifier... [${FLOOR_FLAG}]" | tee -a "$LOG_FILE" "$JOB_LOG"
    if ! python scripts/synthetic_vs_real_classifier.py "$PKL_FILE" \
        --output_directory ./classifier_tmps \
        --no_texture \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ >> "$JOB_LOG" 2>&1; then
        EVAL_SUCCESS=false
    fi

    # Physcene Metrics
    echo "[Job ${IDX}] [6/7] Running physcene metrics... [${FLOOR_FLAG}]" | tee -a "$LOG_FILE" "$JOB_LOG"
    if ! python ../steerable-scene-generation/scripts/physcene_metrics.py "$PKL_FILE" >> "$JOB_LOG" 2>&1; then
        EVAL_SUCCESS=false
    fi

    END_TIME_EVAL=$(date +%s)
    DURATION_EVAL=$((END_TIME_EVAL - START_TIME_EVAL))
    TOTAL_DURATION=$((END_TIME_EVAL - START_TIME_RENDER))

    if [ "$EVAL_SUCCESS" = true ]; then
        echo "[Job ${IDX}] ✅ SUCCESS: All evaluations completed" | tee -a "$LOG_FILE" "$JOB_LOG"
        echo "[Job ${IDX}] Render time: ${DURATION_RENDER}s, Eval time: ${DURATION_EVAL}s, Total: ${TOTAL_DURATION}s" | tee -a "$LOG_FILE" "$JOB_LOG"
        echo "[Job ${IDX}] File: $(basename $(dirname $PKL_FILE)) [${FLOOR_FLAG}]" | tee -a "$LOG_FILE" "$JOB_LOG"
        echo "SUCCESS: Job ${IDX} - $(basename $(dirname $PKL_FILE)) [${FLOOR_FLAG}] - Total: ${TOTAL_DURATION}s" >> "$LOG_FILE"
    else
        echo "[Job ${IDX}] ❌ FAILED: Some evaluations failed" | tee -a "$LOG_FILE" "$JOB_LOG"
        echo "[Job ${IDX}] File: $(basename $(dirname $PKL_FILE)) [${FLOOR_FLAG}]" | tee -a "$LOG_FILE" "$JOB_LOG"
        echo "FAILED: Job ${IDX} - $(basename $(dirname $PKL_FILE)) [${FLOOR_FLAG}] - Duration: ${TOTAL_DURATION}s" >> "$LOG_FILE"
    fi

    echo "[Job ${IDX}] Finished at: $(date)" | tee -a "$LOG_FILE" "$JOB_LOG"
    echo "[Job ${IDX}] Detailed logs in: $JOB_LOG" | tee -a "$LOG_FILE"
    
    echo >&3  # Release semaphore token
}

# ========================================
# Launch all jobs in parallel
# ========================================
echo ""
echo "Launching parallel jobs..."
SCRIPT_START=$(date +%s)

IDX=1
for ENTRY in "${PKL_FILES_WITH_FLAGS[@]}"; do
    IFS='|' read -r PKL_FILE FLOOR_FLAG <<< "$ENTRY"
    
    # Acquire token (blocks if concurrency limit reached)
    read -u 3
    
    # Launch job in background
    process_entry "$PKL_FILE" "$FLOOR_FLAG" "$IDX" &
    
    echo "Launched Job ${IDX}: $(basename $(dirname "$PKL_FILE")) [${FLOOR_FLAG}]"
    ((IDX++))
done
w
echo ""
echo "All jobs launched. Waiting for completion..."

# Wait for all background jobs to finish
wait

# Close semaphore fd
exec 3>&-

SCRIPT_END=$(date +%s)
SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))

echo ""
echo "==============================================="
echo "Batch Evaluation Complete!"
echo "==============================================="
echo "Processed: $TOTAL files in parallel"
echo "Total execution time: ${SCRIPT_DURATION}s ($((SCRIPT_DURATION / 60))m $((SCRIPT_DURATION % 60))s)"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Summary:"
echo "--------"
grep -E "SUCCESS|FAILED|SKIPPED" "$LOG_FILE" | grep "Job"
echo ""
echo "Individual job logs:"
ls -1 ${LOG_FILE%.log}_job*.log 2>/dev/null
echo ""