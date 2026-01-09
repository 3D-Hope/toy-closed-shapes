#!/bin/bash

python scripts/sample_polygon_input.py

python scripts/generate_floor_plan_from_polygon.py tmp/polygon_world.npy --output tmp/floor_plan_world.npz

python ../ThreedFront/scripts/preprocess_floorplan_custom.py tmp/polygon_world.npy --output_fpbpn tmp/polygon_world_fpbpn.npy

# Run sampling and capture the output
python scripts/sampling_for_app.py +num_scenes=5 \
    load=gtjphzpb \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion\
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.classifier_free_guidance.use_floor=true \
    algorithm.custom.old=False \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    wandb.mode=disabled


# Extract the output directory from stdout
OUTPUT_DIR=$(cat tmp/output_path.txt | grep "OUTPUT_DIR=" | cut -d'=' -f2)

# Print captured value for verification
echo "Captured output directory: $OUTPUT_DIR"

# Build the results file path
RESULTS_PKL="$OUTPUT_DIR/sampled_scenes_results.pkl"

# Verify the file exists
if [ ! -f "$RESULTS_PKL" ]; then
    echo "ERROR: Results file not found at: $RESULTS_PKL"
    echo "Searching for latest results file..."
    # Find most recent results file as fallback
    RESULTS_PKL=$(find outputs -name "sampled_scenes_results.pkl" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
    echo "Using latest file: $RESULTS_PKL"
fi

# Now render the output wherever sampling saved it
if [ -f "$RESULTS_PKL" ]; then
    echo "Rendering scenes from: $RESULTS_PKL"
    python ../ThreedFront/scripts/render_results_3d_custom_floor.py \
      "$RESULTS_PKL" \
      --floor_plan_npy /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp/floor_plan_world.npz \
      --retrieve_by_size
else
    echo "ERROR: Could not find results file to render"
    exit 1
fi