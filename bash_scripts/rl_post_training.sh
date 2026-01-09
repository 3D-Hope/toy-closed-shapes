source .venv/bin/activate

PYTHONPATH=. python main.py +name=first_rl \
load=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/checkpoints/078bct021-ashok-d/3dhope_rl/bgdrozky/model.ckpt \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_flux_transformer \
algorithm.classifier_free_guidance.use=False \
algorithm.ema.use=False \
algorithm.trainer=rl_score \
algorithm.ddpo.use_iou_reward=False \
algorithm.ddpo.use_has_sofa_reward=True \
algorithm.ddpo.use_object_number_reward=False \
algorithm.noise_schedule.scheduler=ddim \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
experiment.training.max_steps=2e6 \
experiment.validation.limit_batch=1 \
experiment.validation.val_every_n_step=50 \
algorithm.ddpo.ddpm_reg_weight=200.0 \
experiment.reset_lr_scheduler=True \
experiment.training.lr=1e-6 \
experiment.lr_scheduler.num_warmup_steps=250 \
algorithm.ddpo.batch_size=4 \
experiment.training.checkpointing.every_n_train_steps=500 \
algorithm.num_additional_tokens_for_sampling=2 \
algorithm.ddpo.n_timesteps_to_sample=100 \
experiment.find_unused_parameters=True \
algorithm.custom.loss=true \
debug=True
#reduced from 32 algorithm.ddpo.batch_size=4 \
