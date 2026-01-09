source ../steerable-scene-generation/.venv/bin/activate

PYTHONPATH=. python main.py +name=mi_toy_test \
dataset=custom_scene \
dataset.processed_scene_data_path=../steerable-scene-generation/data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_midiffusion \
algorithm.trainer=ddpm \
algorithm.noise_schedule.scheduler=ddim \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=false \
algorithm.classifier_free_guidance.weight=0 \
algorithm.custom.loss=true \
algorithm.ema.use=True \
experiment.training.batch_size=2048 \
experiment.validation.batch_size=2048 \
experiment.test.batch_size=2048 \
# wandb.mode=disabled
