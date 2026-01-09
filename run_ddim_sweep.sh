#!/bin/bash

LOGFILE=$1   # log file name passed from command line

# TIMESTEPS=(10 20 30 40 50 60 70 80 90 100 150)
TIMESTEPS=(10 20)

for T in "${TIMESTEPS[@]}"; do
    echo "============================================" | tee -a "$LOGFILE"
    echo "Running DDIM with num_inference_timesteps=$T" | tee -a "$LOGFILE"
    echo "============================================" | tee -a "$LOGFILE"

    PYTHONPATH=. python dynamic_constraint_rewards/compute_success_rates.py \
        +num_scenes=1000 \
        load=h9nztqlg \
        dataset=custom_scene \
        algorithm=scene_diffuser_midiffusion \
        dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
        dataset.max_num_objects_per_scene=12 \
        experiment.test.batch_size=256 \
        algorithm.trainer=ddpm \
        algorithm.noise_schedule.scheduler=ddim \
        algorithm.noise_schedule.ddim.num_inference_timesteps=$T \
        experiment.find_unused_parameters=True \
        algorithm.classifier_free_guidance.use=False \
        algorithm.classifier_free_guidance.use_floor=True \
        algorithm.classifier_free_guidance.weight=0 \
        algorithm.custom.loss=true \
        algorithm.ema.use=True \
        dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm_no_prm \
        algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with tv stand and desk and chair for working." \
        algorithm.ddpo.dynamic_constraint_rewards.use=True \
        wandb.mode=disabled \
        algorithm.custom.old=False \
        >> "$LOGFILE" 2>&1

done
