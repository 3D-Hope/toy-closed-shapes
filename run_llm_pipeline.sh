source .venv/bin/activate
python dynamic_constraint_rewards/run_llm_pipeline.py dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.data.room_type=bedroom dataset.model_path_vec_len=30 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData dataset.data.dataset_directory=bedroom dataset.data.annotation_file=bedroom_threed_front_splits_original.csv dataset.max_num_objects_per_scene=12 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=30 algorithm.custom.obj_diff_vec_len=30 algorithm.custom.num_classes=22 dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm algorithm.validation.num_samples_to_render=0 algorithm.validation.num_samples_to_visualize=0 algorithm.validation.num_directives_to_generate=0 algorithm.test.num_samples_to_render=0 algorithm.test.num_samples_to_visualize=0 algorithm.test.num_directives_to_generate=0 algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 algorithm.custom.old=True algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with a bed at the center of the room ie 0 0 0 position" algorithm.ddpo.dynamic_constraint_rewards.use=True

# python dynamic_constraint_rewards/get_reward_stats.py load=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/model.ckpt dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
#     dataset.data.room_type=bedroom \
#     dataset.model_path_vec_len=30 \
#     dataset.data.dataset_directory=bedroom \
#     dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm.custom.objfeat_dim=0 \
#     algorithm.custom.obj_vec_len=30 \
#     algorithm.custom.obj_diff_vec_len=30 \
#     algorithm.custom.num_classes=22 \
#     dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     dataset.sdf_cache_dir=./bedroom_sdf_cache \
#     dataset.accessibility_cache_dir=./bedroom_accessibility_cache \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion \
#     algorithm.trainer=rl_score \
#     algorithm.noise_schedule.scheduler=ddim \
#     experiment.training.max_steps=1020000 \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     algorithm.ddpo.ddpm_reg_weight=50.0 \
#     experiment.reset_lr_scheduler=True \
#     experiment.training.lr=1e-6 \
#     experiment.lr_scheduler.num_warmup_steps=250 \
#     algorithm.ddpo.batch_size=128 \
#     experiment.training.checkpointing.every_n_train_steps=500 \
#     algorithm.num_additional_tokens_for_sampling=0 \
#     algorithm.ddpo.n_timesteps_to_sample=100 \
#     experiment.find_unused_parameters=True \
#     dataset.sdf_cache_dir=./bedroom_sdf_cache/ \
#     dataset.accessibility_cache_dir=./bedroom_accessibility_cache/ \
#     algorithm.custom.num_classes=22 \
#     algorithm.custom.objfeat_dim=0 \
#     algorithm.custom.obj_vec_len=30 \
#     algorithm.custom.obj_diff_vec_len=30 \
#     algorithm.custom.old=True \
#     algorithm.ddpo.dynamic_constraint_rewards.use=True \
#     algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with bed exactly size 2*2m."
    
    # algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom where all objects are close together to leave space for homeworkout" 



# Tested prompts
# algorithm.ddpo.dynamic_constraint_rewards.user_query="I want to follow Vaastu for bedroom layout. The beds headboard should face east." 
# algorithm.ddpo.dynamic_constraint_rewards.user_query="A kids bedroom where table top not reachable by a kid of 2 years old."
# algorithm.ddpo.dynamic_constraint_rewards.user_query="Tv directly in front of the bed."

# algorithm.ddpo.dynamic_constraint_rewards.user_query="A kids bedroom for 2 years old kid."