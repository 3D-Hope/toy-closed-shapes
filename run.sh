source ../steerable-scene-generation/.venv/bin/activate

PYTHONPATH=. python scripts/custom_sample_and_render.py +num_scenes=1000 \
load=wd7mfc13 \
dataset=custom_scene \
dataset.processed_scene_data_path=../steerable-scene-generation/data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_midiffusion \
algorithm.trainer=ddpm \
algorithm.noise_schedule.scheduler=ddpm \
algorithm.noise_schedule.num_train_timesteps=1000 \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=false \
algorithm.classifier_free_guidance.weight=0 \
algorithm.custom.loss=true \
algorithm.custom.old=False \
algorithm.ema.use=false \
dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
algorithm.custom.old=False \
algorithm.model.n_layer=2 \
algorithm.model.n_embd=32 \
algorithm.model.dim_feedforward=128 \
wandb.mode=disabled \
experiment.test.batch_size=1000000 \

