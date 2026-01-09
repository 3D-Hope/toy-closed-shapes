# TODO: changed to torch.cat([translations, sizes, angles, class_labels, objfeat], dim=-1), update this to run
# import logging
# import os
# import pickle
# import json
# from pathlib import Path

# import hydra
# import numpy as np
# import torch
# import yaml
# from omegaconf import DictConfig, OmegaConf
# from omegaconf.omegaconf import open_dict
# from threed_front.datasets import get_raw_dataset
# from threed_front.evaluation import ThreedFrontResults

# from steerable_scene_generation.datasets.custom_scene import get_dataset_raw_and_encoded
# from steerable_scene_generation.datasets.custom_scene.custom_scene_final import (
#     CustomDataset,
#     update_data_file_paths,
# )
# from steerable_scene_generation.experiments import build_experiment
# from steerable_scene_generation.utils.ckpt_utils import (
#     download_latest_or_best_checkpoint,
#     download_version_checkpoint,
#     is_run_id,
# )
# from steerable_scene_generation.utils.distributed_utils import is_rank_zero
# from steerable_scene_generation.utils.logging import filter_drake_vtk_warning
# from steerable_scene_generation.utils.omegaconf import register_resolvers
# from universal_constraint_rewards.commons import idx_to_labels

# idx_to_class = {
#     0: "armchair",
#     1: "bookshelf",
#     2: "cabinet",
#     3: "ceiling_lamp",
#     4: "chair",
#     5: "children_cabinet",
#     6: "coffee_table",
#     7: "desk",
#     8: "double_bed",
#     9: "dressing_chair",
#     10: "dressing_table",
#     11: "kids_bed",
#     12: "nightstand",
#     13: "pendant_lamp",
#     14: "shelf",
#     15: "single_bed",
#     16: "sofa",
#     17: "stool",
#     18: "table",
#     19: "tv_stand",
#     20: "wardrobe",
# }
# # Add logging filters.
# filter_drake_vtk_warning()

# # Disable tokenizer parallelism.
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ[
#     "HF_HOME"
# ] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface"
# os.environ[
#     "HF_DATASETS_CACHE"
# ] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface/datasets"


# @hydra.main(version_base=None, config_path="../configurations", config_name="config")
# def main(cfg: DictConfig) -> None:
#     if not is_rank_zero:
#         raise ValueError(
#             "This script must be run on the main process. Try export CUDA_VISIBLE_DEVICES=0."
#         )

#     # Set random seed
#     seed = 42
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)

#     # Resolve config
#     register_resolvers()
#     OmegaConf.resolve(cfg)
#     config = cfg.dataset

#     #     # Set predict mode.
#     cfg.algorithm.predict.do_sample = False
#     cfg.algorithm.predict.do_inference_time_search = False
#     cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False
#     cfg.algorithm.predict.do_rearrange = False
#     cfg.algorithm.predict.do_complete = False
#     cfg.algorithm.predict.do_inpainting = True
#     # Check if load path is provided
#     if "load" not in cfg or cfg.load is None:
#         raise ValueError("Please specify a checkpoint to load with 'load=...'")

#         # Get configuration values with defaults.
#     num_scenes = cfg.get("num_scenes", 1)
#     print(f"[DEBUG] Number of scenes to sample: {num_scenes}")
#     hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
#     cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)
#     print(f"[DEBUG] cfg_choice: {cfg_choice}")
#     with open_dict(cfg):
#         if cfg_choice["experiment"] is not None:
#             cfg.experiment._name = cfg_choice["experiment"]
#         if cfg_choice["dataset"] is not None:
#             cfg.dataset._name = cfg_choice["dataset"]
#         if cfg_choice["algorithm"] is not None:
#             cfg.algorithm._name = cfg_choice["algorithm"]

#     # Set up output directory
#     output_dir = Path(hydra_cfg.runtime.output_dir)
#     logging.info(f"Outputs will be saved to: {output_dir}")

#     if cfg.wandb.project is None:
#         cfg.wandb.project = str(Path(__file__).parent.parent.name)
#     # Load the checkpoint
#     load_id = cfg.load
#     name = f"custom_sampling_{load_id}"

#     import wandb

#     wandb.init(
#         name=name,
#         dir=str(output_dir),
#         config=OmegaConf.to_container(cfg),
#         project=cfg.wandb.project,
#         mode=cfg.wandb.mode,
#     )
#     if is_run_id(load_id):
#         run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
#         download_dir = output_dir / "checkpoints"
#         version = cfg["checkpoint_version"]
#         if version is not None and isinstance(version, int):
#             checkpoint_path = download_version_checkpoint(
#                 run_path=run_path, version=version, download_dir=download_dir
#             )
#         else:
#             checkpoint_path = download_latest_or_best_checkpoint(
#                 run_path=run_path,
#                 download_dir=download_dir,
#                 use_best=cfg.get("use_best", False),
#             )
#     else:
#         checkpoint_path = Path(load_id)

#     # Load dataset and experiment
#     raw_train_dataset = get_raw_dataset(
#         update_data_file_paths(config["data"], config),
#         # config["data"],
#         split=config["training"].get("splits", ["train", "val"]),
#         include_room_mask=config["network"].get("room_mask_condition", True),
#     )
#     raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
#         update_data_file_paths(config["data"], config),
#         split=config["validation"].get("splits", ["test"]),
#         max_length=config["max_num_objects_per_scene"],
#         include_room_mask=config["network"].get("room_mask_condition", True),
#     )
#     custom_dataset = CustomDataset(
#         cfg=cfg.dataset,
#         split=config["validation"].get("splits", ["test"]),
#         ckpt_path=str(checkpoint_path),
#     )

#     # Limit dataset to num_scenes samples
#     dataset_size = len(custom_dataset)
#     num_scenes_to_sample = num_scenes  # Always use requested num_scenes

#     # Create indices with resampling if needed
#     if num_scenes_to_sample <= dataset_size:
#         # Use first num_scenes samples without resampling
#         indices = list(range(num_scenes_to_sample))
#     else:
#         # Need to resample: repeat the dataset multiple times
#         print(
#             f"[INFO] Requested {num_scenes_to_sample} scenes but dataset only has {dataset_size} scenes."
#         )
#         print(
#             f"[INFO] Will resample with replacement to generate {num_scenes_to_sample} scenes."
#         )
#         indices = [i % dataset_size for i in range(num_scenes_to_sample)]

#     # Create subset of dataset with the indices (may include duplicates)
#     from torch.utils.data import Subset

#     limited_dataset = Subset(custom_dataset, indices)
#     sampled_dataset_indices = indices.copy()
#     batch_size = cfg.experiment.get("test", {}).get(
#         "batch_size", cfg.experiment.validation.batch_size
#     )
#     scenes = None
#     inpaint_masks = None
#     print(f"[DEBUG] Using batch size: {batch_size}")
#     dataloader = torch.utils.data.DataLoader(
#         limited_dataset,
#         batch_size=batch_size,
#         num_workers=4,
#         shuffle=False,
#         persistent_workers=False,
#         pin_memory=cfg.experiment.test.pin_memory,
#     )

#     idx_to_labels_room_type = idx_to_labels[cfg.dataset.data.room_type]
#     label_to_idx_room_type = {v: k for k, v in idx_to_labels_room_type.items()}
#     print(f"[DEBUG] label_to_idx_room_type: {label_to_idx_room_type}")
#     inpaint_masks_final = {}


#     # Build experiment and get diffuser
#     experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
#     print(f"[DEBUG] experiment: {cfg.algorithm.predict.inpaint_masks}")
#     if cfg.algorithm.predict.inpaint_masks is not None:
#         print(f"[DEBUG] Inpaint masks: {cfg.algorithm.predict.inpaint_masks}")
#         print(f"[DEBUG] Type of inpaint masks: {type(cfg.algorithm.predict.inpaint_masks)}")
#         if cfg.algorithm.predict.inpaint_masks is not None:
#             val = cfg.algorithm.predict.inpaint_masks
#             if isinstance(val, DictConfig):
#                 inpaint_masks = OmegaConf.to_container(val, resolve=True)  # -> plain dict
#             elif isinstance(val, str):
#                 inpaint_masks = yaml.safe_load(val)
#         inpaint_masks_final = inpaint_masks
#     else:
#         inpaint_path = cfg.algorithm.ddpo.dynamic_constraint_rewards.inpaint_path
#         with open(inpaint_path, "r") as f:
#             import json
#             inpaint_cfg = json.load(f)
#         inpaint_masks = inpaint_cfg["inpaint"]
#         print(f"[DEBUG] Inpaint masks: {inpaint_masks}")

#         dataset_stat_dir = os.path.join(cfg.dataset.data.path_to_processed_data, cfg.dataset.data.room_type, "dataset_stats.txt")
#         import random
#         with open(dataset_stat_dir, "r") as f:
#             import json
#             dataset_stats = json.load(f)
#         class_frequencies = dataset_stats["class_frequencies"]
#         for label_name, count in inpaint_masks.items():
#             labels = label_name.split(",")
#             if len(labels) == 1:
#                 label_name = labels[0]
#             else:
#                 weights = [class_frequencies[label_name] for label_name in labels]
#                 label_name = random.choices(labels, weights=weights, k=1)[0]
#             inpaint_masks_final[label_name] = count

#     print(f"[DEBUG] Inpaint masks final: {inpaint_masks_final}")

#     if inpaint_masks_final is None:
#         to_hardcode = None
#     else:
#         to_hardcode = {int(label_to_idx_room_type[k]): v for k, v in inpaint_masks_final.items()}
#     # SAUGAT
#     sampled_scene_batches = experiment.exec_task(
#         "inpaint",
#         dataloader=dataloader,
#         use_ema=cfg.algorithm.ema.use,
#         to_hardcode=to_hardcode,
#     )

#     # NOTE: fpbpn goes as a part of dataloader batch, scenes and inpaint_masks sent  for each indices of dataloader
#     # SAUGAT
#     # experiment.algo = experiment._build_algo(ckpt_path=checkpoint_path)
#     # diffuser = experiment.algo
#     # # print(f"[DEBUG] Diffuser: {diffuser}")
#     # # scene_vec_desc = diffuser.scene_vec_desc
#     # # print(f"[DEBUG] Scene vector descriptor: {scene_vec_desc}")
#     # B = num_scenes
#     # N = cfg.dataset.max_num_objects_per_scene
#     # V = cfg.dataset.model_path_vec_len
#     # single_bed_idx = None
#     # for idx, label in idx_to_class.items():
#     #     if label == "single_bed":
#     #         single_bed_idx = int(idx)
#     #         break
#     # if single_bed_idx is None:
#     #     raise ValueError("Single bed index not found")
#     # n_classes = len(idx_to_class) + 1 # +1 for the empty object
#     # # Each object's vector: [class_onehot][translation][size][angle]

#     # # 2. Prepare input scene
#     # scenes = torch.zeros((B, N, V), dtype=torch.float32)
#     # for b in range(B):
#     #     for obj_idx in [0]:
#     #         # Zero all, then set single_bed_idx for class_onehot part
#     #         scenes[b, obj_idx, :n_classes] = -1.0 #note we use -1 for not class index in ohe
#     #         scenes[b, obj_idx, single_bed_idx] = 1.0

#     # # 3. Build inpainting mask: all True (to sample), except disable mask for
#     # #     class labels for obj 0, 1. Everything else (incl. size) is generated.
#     # inpainting_masks = torch.ones((B, N, V), dtype=torch.bool)
#     # # for b in range(B):
#     # #     for obj_idx in [0]:
#     # #         inpainting_masks[b, obj_idx, :n_classes] = False  # Fix class label only
#     # dataset_size = len(custom_dataset)
#     # num_scenes_to_sample = num_scenes
#     # if num_scenes_to_sample <= dataset_size:
#     #     # Use first num_scenes samples without resampling
#     #     indices = list(range(num_scenes_to_sample))
#     # else:
#     #     # Need to resample: repeat the dataset multiple times
#     #     print(f"[INFO] Requested {num_scenes_to_sample} scenes but dataset only has {dataset_size} scenes.")
#     #     print(f"[INFO] Will resample with replacement to generate {num_scenes_to_sample} scenes.")
#     #     indices = [i % dataset_size for i in range(num_scenes_to_sample)]
#     # print(f"[Ashok] initial scene {scenes.shape}, inpainting mask {inpainting_masks.shape}")
#     # print(f"[Ashok] sampled indices {indices}")
#     # #TODO: MAKE THIS DO INFERENCE IN BATCHES
#     # # 4. Prepare data_batch and call inpaint
#     # fpbpn = [custom_dataset[i]["fpbpn"] for i in indices]
#     # fpbpn = torch.tensor(fpbpn, dtype=torch.float32)
#     # print(f"[DEBUG] scene shape: {scenes.shape}")
#     # data_batch = {
#     #     "id_indices": indices,
#     #     "scenes": scenes,
#     #     "inpainting_masks": inpainting_masks,
#     #     "fpbpn": fpbpn,
#     # }
#     # diffuser.put_model_in_eval_mode()
#     # inpainted_scenes = diffuser.inpaint_scenes(data_batch, use_ema=cfg.algorithm.ema.use)

#     # 5. Save in the same format as custom_sample_and_render.py
#     # raw_pkl_path = output_dir / "raw_sampled_scenes.pkl"
#     # with open(raw_pkl_path, "wb") as f:
#     #     pickle.dump(inpainted_scenes, f)

#     # sampled_scenes_np = inpainted_scenes.detach().cpu().numpy()
#     # print(f"[Ashok] sampled scene {sampled_scenes_np[0]}")

#     # SAUGAT
#     sampled_indices = sampled_dataset_indices
#     sampled_scenes = torch.cat(sampled_scene_batches, dim=0)
#     assert (
#         len(sampled_indices) == sampled_scenes.shape[0]
#     ), f"Mismatch: {len(sampled_indices)} indices vs {sampled_scenes.shape[0]} scenes"
#     with open(output_dir / "raw_sampled_scenes.pkl", "wb") as f:
#         pickle.dump(sampled_scenes, f)

#     if cfg.dataset.data.room_type == "livingroom":
#         n_classes = 25
#     else:
#         n_classes = 22
#     # SAUGAT

#     mask = ~torch.any(torch.isnan(sampled_scenes), dim=(1, 2))
#     # Filter scenes
#     sampled_scenes = sampled_scenes[mask]
#     # Filter corresponding dataset indices so they align with the kept scenes
#     mask_np = mask.detach().cpu().numpy().astype(bool)
#     sampled_indices = [idx for idx, keep in zip(sampled_indices, mask_np) if keep]

#     sampled_scenes_np = sampled_scenes.detach().cpu().numpy()

#     print(f"[DEBUG] sampled scenes np: {sampled_scenes_np[0]}")
#     bbox_params_list = []
#     for i in range(sampled_scenes_np.shape[0]):
#         class_labels, translations, sizes, angles = [], [], [], []
#         for j in range(sampled_scenes_np.shape[1]):
#             class_label_idx = np.argmax(sampled_scenes_np[i, j, :n_classes])
#             # if class_label_idx == 15:
#             #     print(f"[DEBUG] i {i} j {j} class label idx: {class_label_idx} SINGLE BED")
#             if class_label_idx != n_classes - 1:
#                 ohe = np.zeros(n_classes - 1)
#                 ohe[class_label_idx] = 1
#                 class_labels.append(ohe)
#                 translations.append(sampled_scenes_np[i, j, n_classes : n_classes + 3])
#                 sizes.append(sampled_scenes_np[i, j, n_classes + 3 : n_classes + 6])
#                 angles.append(sampled_scenes_np[i, j, n_classes + 6 : n_classes + 8])
#         bbox_params_list.append(
#             {
#                 "class_labels": np.array(class_labels)[None, :],
#                 "translations": np.array(translations)[None, :],
#                 "sizes": np.array(sizes)[None, :],
#                 "angles": np.array(angles)[None, :],
#             }
#         )

#     # print(f"[DEBUG] bbox params list: {bbox_params_list[0]}")

#     layout_list = []
#     successful_indices = list(range(len(bbox_params_list)))
#     for bbox_params_dict in bbox_params_list:
#         try:
#             boxes = encoded_dataset.post_process(bbox_params_dict)
#             bbox_params = {k: v[0] for k, v in boxes.items()}
#             layout_list.append(bbox_params)
#         except Exception as e:
#             print(f"[WARNING] Skipping scene due to post_process error: {e}")
#             continue

#     # print(f"[DEBUG] layout list: {layout_list[0]}")

#     threed_front_results = ThreedFrontResults(
#         raw_train_dataset, raw_dataset, config, successful_indices, layout_list
#     )
#     results_pkl_path = output_dir / "sampled_scenes_results.pkl"
#     with open(results_pkl_path, "wb") as f:
#         pickle.dump(threed_front_results, f)
#     print(f"Saved result to: {results_pkl_path}")


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     main()
