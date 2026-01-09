import os

from typing import Dict, Optional, Tuple

import torch

from diffusers import DDIMScheduler, DDPMScheduler
from tqdm import tqdm
import numpy as np

from dynamic_constraint_rewards.commons import import_dynamic_reward_functions

# from dynamic_constraint_rewards.scale_raw_rewards import RewardNormalizer
from steerable_scene_generation.datasets.scene.scene import SceneDataset
from universal_constraint_rewards.accessibility_reward import (
    AccessibilityCache,
    precompute_accessibility_cache,
)
from universal_constraint_rewards.commons import idx_to_labels, parse_and_descale_scenes
from universal_constraint_rewards.not_out_of_bound_reward import (
    SDFCache,
    precompute_sdf_cache,
)

from .ddpo_helpers import (
    composite_reward,
    ddim_step_with_logprob,
    ddpm_step_with_logprob,
    has_sofa_reward,
    iou_reward,
    non_penetration_reward,
    number_of_physically_feasible_objects_reward,
    object_number_reward,
    physcene_reward,
    prompt_following_reward,
    universal_reward,
)
from .scene_diffuser_base_continous import SceneDiffuserBaseContinous
from .trainer_ddpm import compute_ddpm_loss
import torch.nn.functional as F


class SceneDiffuserTrainerRL(SceneDiffuserBaseContinous):
    """
    Base class for classes that provide RL training logic.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

        # Variable for storing the reward computation cache.
        self.reward_cache = None
        self.cfg = cfg
        
        if (
            self.cfg.ddpo.dynamic_constraint_rewards.use
            or self.cfg.ddpo.use_universal_reward
            and not self.cfg.ddpo.universal_reward.use_physcene_reward
        ) and self.cfg.ddpo.dynamic_constraint_rewards.universal_weight > 0.0:
            print(f"this should not exist")
            user_query = self.cfg.ddpo.dynamic_constraint_rewards.user_query
            user_query = user_query.replace(" ", "_").replace(".", "")

            if (
                self.cfg.ddpo.dynamic_constraint_rewards.agentic
            ):  # during universal only training, this flag should be set
                stats_path = os.path.join(
                    self.cfg.ddpo.dynamic_constraint_rewards.reward_base_dir,
                    f"{user_query}_stats.json",
                )
            else:
                stats_path = os.path.join(
                    self.cfg.ddpo.dynamic_constraint_rewards.reward_base_dir,
                    f"{user_query}_stats_initial.json",
                )

            # self.reward_normalizer = RewardNormalizer(
            #     baseline_stats_path=stats_path
            # )
            self.reward_normalizer = None
            if not hasattr(self.cfg.dataset, "sdf_cache_dir") or not os.path.exists(
                self.cfg.dataset.sdf_cache_dir
            ):
                print(f"Precomputing SDF cache at {self.cfg.dataset.sdf_cache_dir}...")
                precompute_sdf_cache(
                    config=self.cfg,
                    num_workers=16,
                    sdf_cache_dir=self.cfg.dataset.sdf_cache_dir,
                )

            else:
                print(
                    f"SDF cache directory {self.cfg.dataset.sdf_cache_dir} already exists. Skipping SDF precomputation."
                )
            if not os.path.exists(self.cfg.dataset.accessibility_cache_dir):
                print(
                    f"Precomputing Accessibility cache at {self.cfg.dataset.accessibility_cache_dir}..."
                )
                precompute_accessibility_cache(
                    config=self.cfg,
                    num_workers=16,
                    accessibility_cache_dir=self.cfg.dataset.accessibility_cache_dir,
                )
            else:
                print(
                    f"Accessibility cache directory {self.cfg.dataset.accessibility_cache_dir} already exists. Skipping Accessibility precomputation."
                )
            self.train_sdf_cache = SDFCache(
                self.cfg.dataset.sdf_cache_dir, split="train_val"
            )
            self.val_sdf_cache = SDFCache(self.cfg.dataset.sdf_cache_dir, split="test")
            self.train_accessibility_cache = AccessibilityCache(
                self.cfg.dataset.accessibility_cache_dir, split="train_val"
            )
            self.val_accessibility_cache = AccessibilityCache(
                self.cfg.dataset.accessibility_cache_dir, split="test"
            )
        else:
            self.reward_normalizer = None
            self.train_sdf_cache = None
            self.val_sdf_cache = None
            self.train_accessibility_cache = None
            self.val_accessibility_cache = None

        if self.cfg.ddpo.dynamic_constraint_rewards.use:
            user_query = self.cfg.ddpo.dynamic_constraint_rewards.user_query

            if self.cfg.ddpo.dynamic_constraint_rewards.agentic:
                (
                    self.get_reward_functions,
                    self.test_reward_functions,
                ) = import_dynamic_reward_functions(
                    reward_code_dir=f"{user_query.replace(' ', '_').replace('.', '')}_dynamic_reward_functions_final"
                )
            else:
                (
                    self.get_reward_functions,
                    self.test_reward_functions,
                ) = import_dynamic_reward_functions(
                    reward_code_dir=f"{user_query.replace(' ', '_').replace('.', '')}_dynamic_reward_functions_initial"
                )

        else:
            self.get_reward_functions = None
            self.test_reward_functions = None
    
    def get_timesteps_for_inc_joint(self, n_timesteps):
        n_inference_steps = 150
        num_steps_total = 1000
        step_ratio = num_steps_total//n_inference_steps
        timestep_150 =  (np.arange(0, n_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

        # do linspace  to sample the equally distanced 100 indices 0 to 149 and use the timesteps at those indices for 100 timesteps to get timestep_100
        indices = np.linspace(0, len(timestep_150) - 1, n_timesteps).round().astype(np.int64)
        timesteps = timestep_150[indices]
        return torch.tensor(timesteps).to(self.device)

    def _generate_single_trajectory_group(
        self,
        batch_size: int,
        n_steps: int,
        cond_dict: dict | None,
        room_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate trajectories for a single group with fixed timestep count.
        Helper method for joint training.
        
        Args:
            batch_size (int): Number of samples in this group.
            n_steps (int): Number of denoising steps for this group.
            cond_dict (dict | None): Conditioning dictionary for this group.
            room_type (str): Type of room (bedroom/livingroom).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Trajectories (batch_size, n_steps+1, N, V)
                and log probabilities (batch_size, n_steps).
        """
        trajectory = []
        trajectory_log_props = []
        
        xt = self.sample_continuous_noise_prior(
                    batch_size,
                    2
                )
        trajectory.append(xt)
        timesteps = self.get_timesteps_for_inc_joint(n_steps).to(self.device)
        # print(f"[Ashok] Using ddim path for n_steps={n_steps}: {timesteps}")
        # Denoising loop
        if len(timesteps) > 100:
            timesteps_with_grads = set(
                torch.randperm(len(timesteps))[
                    :100
                ].tolist()
            )
        else:
            timesteps_with_grads = set(range(len(timesteps)))
        print(f" [Ashok] Sampling {len(timesteps)} timesteps with grads: {len(timesteps_with_grads)}")

        for t_idx, t in enumerate(timesteps):
            # Predict noise
            residual = self.predict_noise(xt, t, cond_dict=cond_dict)
            
            # Compute next step and log probability
            if isinstance(self.noise_scheduler, DDPMScheduler):
                # xt_next, log_prop = ddpm_step_with_logprob(
                #     scheduler=self.noise_scheduler,
                #     model_output=residual,
                #     timestep=t,
                #     sample=xt,
                # )
                raise Exception("DDPMScheduler not supported for joint training.")
            else:  # DDIMScheduler
                if t_idx not in timesteps_with_grads:
                    with torch.no_grad():
                        xt_next, log_prop = ddim_step_with_logprob(
                            scheduler=self.noise_scheduler,
                            model_output=residual,
                            timestep=t,
                            sample=xt,
                            eta=self.cfg.noise_schedule.ddim.eta,
                        )
                else:
                    xt_next, log_prop = ddim_step_with_logprob(
                        scheduler=self.noise_scheduler,
                        model_output=residual,
                        timestep=t,
                        sample=xt,
                        eta=self.cfg.noise_schedule.ddim.eta,
                    )
            
            xt = xt_next
            trajectory.append(xt)
            trajectory_log_props.append(log_prop)
        
        # Stack trajectories
        trajectories = torch.stack(trajectory, dim=1)  # (batch_size, n_steps+1, N, V)
        trajectories_log_props = torch.stack(trajectory_log_props, dim=1)  # (batch_size, n_steps)
        
        return trajectories, trajectories_log_props

    def generate_trajs_for_ddpo(
        self,
        last_n_timesteps_only: int = 0,
        n_timesteps_to_sample: int = 0,
        batch: Dict[str, torch.Tensor] | None = None,
        incremental_training: bool = False,
        joint_training: bool = False,
        joint_training_timesteps: list[int] | None = None,
        phase: str = "training",
    ) -> Tuple[torch.Tensor | list, torch.Tensor | None, dict | None]:
        """
        Generate denoising trajectories for DDPO.

        Args:
            last_n_timesteps_only (int): If not 0, only keep the last n timesteps.
            n_timesteps_to_sample (int): If not 0, uniformly sample this many timesteps
                for gradient computation. All other timesteps will use torch.no_grad().
            batch (Dict[str, torch.Tensor] | None): Training batch to sample
                conditioning from.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: A batch of denoising trajectories
            (shape: (B, N+1, V)), their log probabilities (shape: (B,)) and
            corresponding conditioning dictionary.
        """
        assert self.cfg.ddpo.batch_size > 1, "Need at least 2 samples for DDPO."
        assert not (
            last_n_timesteps_only != 0 and n_timesteps_to_sample != 0
        ), "Cannot specify both last_n_timesteps_only and n_timesteps_to_sample"

        self.put_model_in_eval_mode()
        room_type = getattr(self.cfg.dataset.data, "room_type", "bedroom")
        if not incremental_training and not joint_training:
            if isinstance(self.noise_scheduler, DDIMScheduler):
                self.noise_scheduler.set_timesteps(
                    self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
                )

            # Determine which timestep indices to compute gradients for.
            timesteps_with_grads = set()
            if n_timesteps_to_sample > 0:
                # Uniformly sample timestep indices.
                timesteps_with_grads = set(
                    range(len(self.noise_scheduler.timesteps))
                    if n_timesteps_to_sample >= len(self.noise_scheduler.timesteps)
                    else torch.randperm(len(self.noise_scheduler.timesteps))[
                        :n_timesteps_to_sample
                    ].tolist()
                )
            elif last_n_timesteps_only > 0:
                # Use the last n timesteps.
                num_timesteps = len(self.noise_scheduler.timesteps)
                last_n = min(last_n_timesteps_only, num_timesteps)
                timesteps_with_grads = set(range(num_timesteps - last_n, num_timesteps))
            else:
                # Use all timesteps.
                timesteps_with_grads = set(range(len(self.noise_scheduler.timesteps)))
        elif incremental_training:
            if isinstance(self.noise_scheduler, DDIMScheduler):
                self.noise_scheduler.set_timesteps(
                    n_timesteps_to_sample, device=self.device
                ) # both train and val will use custom timesteps
            else:
                raise NotImplementedError("Incremental training only implemented for DDIMScheduler.")
            timesteps_with_grads = set(range(len(self.noise_scheduler.timesteps)))
            
        elif joint_training:
            if isinstance(self.noise_scheduler, DDIMScheduler):
                # Joint training: generate separate trajectory groups for each timestep count
                # This avoids padding and makes loss computation cleaner
                trajectory_groups = []
                
                # Distribute batch across timestep counts
                samples_per_group = self.cfg.ddpo.batch_size // len(joint_training_timesteps)
                # remaining = self.cfg.ddpo.batch_size % len(joint_training_timesteps)
                
                start_idx = 0
                for group_idx, n_steps in enumerate(joint_training_timesteps):
                    # Handle uneven distribution (extra samples go to first groups)
                    # group_size = samples_per_group + (1 if group_idx < remaining else 0)
                    group_size = samples_per_group
                    
                    if group_size == 0:
                        continue
                    
                    end_idx = start_idx + group_size
                    
                    # Set scheduler for this group's timestep count
                    self.noise_scheduler.set_timesteps(n_steps, device=self.device)
                    # print(f"[Ashok] n_steps: {n_steps}, timesteps: {self.noise_scheduler.timesteps}")
                    # Sample subset of conditioning for this group
                    if batch is not None:
                        # Slice the batch to get the correct subset for this group
                        # if len(batch["scenes"]) <= end_idx:
                        #     group_batch = {k: v[start_idx:end_idx] for k, v in batch.items()}
                        # else:
                        #     group_batch = {k: v[start_idx:] for k, v in batch.items()}
                        group_cond_dict = self.dataset.sample_data_dict(
                            data=batch, num_items=group_size
                        )
                    else:
                        group_cond_dict = None
                    
                    # Generate trajectories for this group
                    group_trajectories, group_log_probs = self._generate_single_trajectory_group(
                        batch_size=group_size,
                        n_steps=n_steps,
                        cond_dict=group_cond_dict,
                        room_type=room_type,
                    )
                    
                    # Store group with metadata
                    trajectory_groups.append({
                        'trajectories': group_trajectories,  # (group_size, n_steps+1, N, V)
                        'log_probs': group_log_probs,       # (group_size, n_steps)
                        'n_steps': n_steps,
                        'cond_dict': group_cond_dict,
                    })
                    
                    start_idx = end_idx
                
                # Return list of groups instead of single tensors
                # The calling code will handle processing each group separately
                return trajectory_groups, None, None
            
            else:
                raise NotImplementedError("Joint training only implemented for DDIMScheduler.")
        trajectory = []
        trajectory_log_props = []

        # Sample random noise.
        num_objects_per_scene = (
            self.cfg.max_num_objects_per_scene
            + self.cfg.num_additional_tokens_for_sampling
        )
        if room_type == "livingroom":
            xt = self.sample_continuous_noise_prior(
                (
                    self.cfg.ddpo.batch_size,
                    num_objects_per_scene,
                    self.cfg.custom.num_classes
                    + self.cfg.custom.translation_dim
                    + self.cfg.custom.size_dim
                    + self.cfg.custom.angle_dim
                    + self.cfg.custom.objfeat_dim,
                )
            ).to(
                self.device
            )  # Shape (B, N, V)
        elif room_type == "bedroom":
            xt = self.sample_continuous_noise_prior(
                (
                    self.cfg.ddpo.batch_size,
                    num_objects_per_scene,
                    self.scene_vec_desc.get_object_vec_len(),
                )
            ).to(
                self.device
            )  # Shape (B, N, V)
        else:
            raise ValueError(f"Unknown room type: {room_type}")
        trajectory.append(xt)
        # print(
            # f"[Ashok] num data {len(batch['idx'])}, batch size {self.cfg.ddpo.batch_size}"
        # )
        # Create conditioning dictionary from batch if available.
        cond_dict = None
        if batch is not None:
            cond_dict = self.dataset.sample_data_dict(
                data=batch, num_items=self.cfg.ddpo.batch_size
            )
        # Optional: RL inpainting using cfg.algorithm.predict.inpaint_masks
        # use_inpaint = bool(getattr(self.cfg.ddpo, "use_inpaint", False))
        # if use_inpaint:
        # Build label->idx map from room type labels
        # print(f"[Ashok] room_type: {room_type}")

        # label_map = idx_to_labels.get(room_type, idx_to_labels["bedroom"])  # fall back to bedroom
        # label_to_idx = {v: k for k, v in label_map.items()}

        # Determine class dimension
        # if hasattr(self.cfg, "custom") and hasattr(self.cfg.custom, "num_classes"):
        #     num_classes = int(self.cfg.custom.num_classes)
        # else:
        #     num_classes = len(label_map) + 1  # +1 for empty

        # user_query = self.cfg.ddpo.dynamic_constraint_rewards.user_query
        # user_query = user_query.replace(' ', '_').replace('.', '')
        # inpaint_path = os.path.join(self.cfg.ddpo.dynamic_constraint_rewards.reward_base_dir, f"{user_query}_responses_tmp/llm_response_3.json")
        # # print(f"[Ashok] inpaint_path: {inpaint_path}")
        # with open(inpaint_path, "r") as f:
        #     import json
        #     inpaint_cfg = json.load(f)
        # # print(f"[Ashok] inpaint_cfg: {inpaint_cfg}")
        # inpaint_cfg = inpaint_cfg["inpaint"]

        # Read inpaint config dict: {label_name: count}
        # inpaint_cfg = getattr(self.cfg.predict, "inpaint_masks", None)

        # if inpaint_cfg is None:
        #     raise ValueError(
        #         "cfg.ddpo.use_inpaint=True but cfg.algorithm.predict.inpaint_masks is not provided."
        #     )

        # Handle DictConfig or string representations
        # from omegaconf import DictConfig, OmegaConf
        # if isinstance(inpaint_cfg, str):
        #     # Parse string as YAML/dict
        #     import yaml
        #     inpaint_cfg = yaml.safe_load(inpaint_cfg)
        # elif isinstance(inpaint_cfg, DictConfig):
        #     inpaint_cfg = OmegaConf.to_container(inpaint_cfg, resolve=True)

        # print(f"Using inpainting with config: {inpaint_cfg}")
        # Initialize mask and originals
        # inpainting_masks = torch.ones_like(xt, dtype=torch.bool, device=self.device)  # (B,N,V)
        # original_scenes = torch.zeros_like(xt, device=self.device)  # (B,N,V)

        # hardcoded_count = 0
        # dataset_stat_dir = os.path.join(self.cfg.dataset.data.path_to_processed_data, self.cfg.dataset.data.room_type, "dataset_stats.txt")

        # with open(dataset_stat_dir, "r") as f:
        #     import json
        #     dataset_stats = json.load(f)
        # class_frequencies = dataset_stats["class_frequencies"]

        # # DictConfig and dict both support .items()
        # for label_name, count in inpaint_cfg.items():
        #     labels = label_name.split(",")
        #     if len(labels) == 1:
        #         label_name = labels[0]
        #     else:
        #         weights = [class_frequencies[label_name] for label_name in labels]
        #         label_name = random.choices(labels, weights=weights, k=1)[0]
        #     class_idx = int(label_to_idx[str(label_name)])
        #     count = int(count)
        #     end = hardcoded_count + count
        #     if end > xt.shape[1]:
        #         end = xt.shape[1]
        #     if hardcoded_count >= end:
        #         continue
        #     # Freeze class slots for these objects
        #     inpainting_masks[:, hardcoded_count:end, :num_classes] = False
        #     # Set original class one-hot: default -1, with target class 1
        #     original_scenes[:, hardcoded_count:end, :num_classes] = -1.0
        #     original_scenes[:, hardcoded_count:end, class_idx] = 1.0
        #     hardcoded_count = end

        # # Apply mask to initial noise
        # xt = torch.where(inpainting_masks, xt, original_scenes)
        # trajectory[0] = xt
        # print(f"Inpainting masks applied for {hardcoded_count} objects per scene.")
        # print(f"[Ashok] Generating trajectories timesteps {self.noise_scheduler.timesteps}")
        # print(f"[Ashok] timesteps_with_grads: {timesteps_with_grads}")
        # import sys; sys.exit()
        timesteps = self.noise_scheduler.timesteps
        if incremental_training:
            timesteps = self.get_timesteps_for_inc_joint(n_timesteps_to_sample).to(self.device)
            if len(timesteps) > 100:
                timesteps_with_grads = set(
                    torch.randperm(len(timesteps))[
                        :100
                    ].tolist()
                )
                # print(f"[Ashok] Incremental training: limiting to 100 timesteps with grads: {timesteps_with_grads}")

                
            # print(f"[Ashok] Incremental training with {n_timesteps_to_sample} timesteps: {timesteps}")
            print(f"[Ashok] Incremental training with {n_timesteps_to_sample}  require grad {len(timesteps_with_grads)} timesteps.")
        # if phase != "training":
        # print(f"[Ashok] {phase} phase, timesteps: {timesteps}")
        for t_idx, t in enumerate(
            tqdm(
                timesteps,
                desc="  Sampling scenes (Traj generation)",
                leave=False,
                position=1,
            )
        ):
            # Predict the noise for the current timestep.
            if t_idx not in timesteps_with_grads:
                # Don't compute gradients.
                with torch.no_grad():
                    residual = self.predict_noise(xt, t, cond_dict=cond_dict)
            else:
                residual = self.predict_noise(
                    xt, t, cond_dict=cond_dict
                )  # Shape (B, N, V)

            # Compute the updated sample and log probability.
            if isinstance(self.noise_scheduler, DDPMScheduler):
                xt_next, log_prop = ddpm_step_with_logprob(
                    scheduler=self.noise_scheduler,
                    model_output=residual,
                    timestep=t,
                    sample=xt,
                    # mask=inpainting_masks if use_inpaint else None,
                )
            else:  # DDIMScheduler
                xt_next, log_prop = ddim_step_with_logprob(
                    scheduler=self.noise_scheduler,
                    model_output=residual,
                    timestep=t,
                    sample=xt,
                    eta=self.cfg.noise_schedule.ddim.eta,
                    # mask=inpainting_masks if use_inpaint else None,
                )
            # if torch.isnan(xt_next).any():
            #     print(f"[Ashok] NaNs detected in xt_next at timestep {t},")
            #     xt_next = torch.nan_to_num(xt_next)
            # If inpainting, keep unmasked values fixed to originals
            # if use_inpaint:
            #     xt = torch.where(inpainting_masks, xt_next, original_scenes)
            # else:
            #     xt = xt_next
            xt = xt_next

            trajectory.append(xt)
            trajectory_log_props.append(log_prop)

        # Stack so that batch dimension is first.
        trajectories = torch.stack(trajectory, dim=1)  # Shape (B, T+1, N, V)
        trajectories_log_props = torch.stack(
            trajectory_log_props, dim=1
        )  # Shape (B, T)

        if last_n_timesteps_only != 0:
            trajectories = torch.cat(
                (trajectories[:, :1], trajectories[:, -last_n_timesteps_only:]), dim=1
            )
            trajectories_log_props = trajectories_log_props[:, -last_n_timesteps_only:]

        return trajectories, trajectories_log_props, cond_dict

    def compute_rewards_from_trajs(
        self,
        trajectories: torch.Tensor,
        cond_dict: dict | None = None,
        are_trajectories_normalized: bool = True,
    ) -> torch.Tensor:
        """
        Compute rewards from denoising trajectories.

        Args:
            trajectories (torch.Tensor): Denoising trajectories of shape (B, T, N, V).
            cond_dict (dict | None): Conditioning dictionary that was used to generate
                the trajectories.
            are_trajectories_normalized (bool): Whether the trajectories are normalized.

        Returns:
            torch.Tensor: Rewards of shape (B,)
        """
        if (
            sum(
                [
                    self.cfg.ddpo.use_non_penetration_reward,
                    self.cfg.ddpo.use_object_number_reward,
                    self.cfg.ddpo.use_prompt_following_reward,
                    self.cfg.ddpo.use_physical_feasible_objects_reward,
                    self.cfg.ddpo.dynamic_constraint_rewards.use,
                ]
            )
            > 1
        ):
            raise ValueError("Only one reward function is supported at a time.")

        # Only compute rewards for the last timestep.
        x0 = trajectories[:, -1]  # Shape (B, T, N, V) -> Shape (B, N, V)

        if are_trajectories_normalized:
            # Apply inverse normalization.
            x0 = self.dataset.inverse_normalize_scenes(x0)  # Shape (B, N, V)

        if self.cfg.ddpo.use_non_penetration_reward:
            rewards, self.reward_cache = non_penetration_reward(
                scenes=x0,
                scene_vec_desc=self.scene_vec_desc,
                num_workers=self.cfg.ddpo.num_reward_workers,
                cache=self.reward_cache,
                return_updated_cache=True,
            )
        elif self.cfg.ddpo.use_custom_non_penetration_reward:
            rewards, self.reward_cache = non_penetration_reward(
                scenes=x0,
                num_classes=self.cfg.custom.num_classes,
                num_workers=self.cfg.ddpo.num_reward_workers,
            )

        elif self.cfg.ddpo.use_object_number_reward:
            if self.cfg.custom.use:
                # Custom dataset with 30-dimensional object vectors and first 22 are
                # class labels.
                assert (
                    x0.shape[-1] == self.cfg.custom.num_classes + 3 + 3 + 2
                ), f"Expected object vectors to have {self.cfg.custom.num_classes + 3 + 3 + 2} dimensions at x[-1], got {x0.shape}."
                rewards = object_number_reward(
                    scenes=x0, scene_vec_desc=self.scene_vec_desc, cfg=self.cfg
                )
            else:
                rewards = object_number_reward(
                    scenes=x0, scene_vec_desc=self.scene_vec_desc
                )

        elif self.cfg.ddpo.use_iou_reward:
            print("Using IoU reward")
            # Use IoU as reward - less overlap between objects is better
            rewards = iou_reward(scenes=x0, scene_diffuser=self, cfg=self.cfg)

        elif self.cfg.ddpo.use_has_sofa_reward:
            # print("Using IoU reward")
            # Use IoU as reward - less overlap between objects is better
            rewards = has_sofa_reward(
                scenes=x0, scene_vec_desc=self.scene_vec_desc, cfg=self.cfg
            )

        elif self.cfg.ddpo.use_universal_reward:
            print("Using universal rewards")

            is_val = len(self.dataset) <= 200
            print(f"[Ashok] is_val: {is_val}")
            # Get room type from config
            # room_type = "bedroom"  # default
            room_type = self.cfg.dataset.data.room_type  # default
            if hasattr(self.cfg.ddpo, "universal_reward"):
                room_type = self.cfg.ddpo.universal_reward.get("room_type", "bedroom")

            # Get importance weights from config if available
            importance_weights = None
            if hasattr(self.cfg.ddpo, "universal_reward") and hasattr(
                self.cfg.ddpo.universal_reward, "importance_weights"
            ):
                importance_weights = dict(
                    self.cfg.ddpo.universal_reward.importance_weights
                )

            # Get number of classes from config
            num_classes = (
                self.cfg.custom.num_classes if hasattr(self.cfg, "custom") else 22
            )

            # Parse and descale scenes
            parsed_scenes = parse_and_descale_scenes(
                x0, num_classes=num_classes, room_type=room_type
            )

            if self.cfg.ddpo.universal_reward.use_physcene_reward:
                # TODO: Get floor plan args from dataset, how do we get the dataset here? we need get_floor_plan_args from custom dataset here to get the values for each scene in the batch
                indices = cond_dict["idx"]
                # Efficiently gather floor plan args for all indices in the batch
                floor_plan_args_list = [
                    self.dataset.get_floor_plan_args(idx) for idx in indices
                ]
                # Stack each key across the batch for tensor conversion
                floor_plan_args = {
                    key: [args[key] for args in floor_plan_args_list]
                    for key in [
                        "floor_plan_centroid",
                        "floor_plan_vertices",
                        "floor_plan_faces",
                        "room_outer_box",
                    ]
                }
                # Compute physcene reward
                rewards, reward_components = physcene_reward(
                    parsed_scenes=parsed_scenes,
                    reward_normalizer=self.reward_normalizer,
                    scene_vec_desc=self.scene_vec_desc,
                    cfg=self.cfg,
                    room_type=room_type,
                    importance_weights=importance_weights,
                    floor_plan_args=floor_plan_args,
                )

                # Log individual components for analysis using log_dict for proper step tracking
                if hasattr(self, "log_dict") and reward_components:
                    reward_metrics = {
                        f"reward_components/{name}_mean": values.mean()
                        for name, values in reward_components.items()
                    }
                    self.log_dict(
                        reward_metrics,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                    )
            else:
                # Compute universal reward with all physics constraints
                rewards, reward_components = universal_reward(
                    parsed_scenes=parsed_scenes,
                    reward_normalizer=self.reward_normalizer,
                    scene_vec_desc=self.scene_vec_desc,
                    cfg=self.cfg,
                    room_type=room_type,
                    # importance_weights=importance_weights,
                    floor_polygons=[
                        self.dataset.get_floor_polygon_points(idx)
                        for idx in cond_dict["idx"]
                    ],
                    indices=cond_dict["idx"],
                    is_val=is_val,
                    sdf_cache_dir=self.cfg.dataset.sdf_cache_dir,
                    sdf_cache=self.train_sdf_cache
                    if not is_val
                    else self.val_sdf_cache,
                    accessibility_cache=self.train_accessibility_cache
                    if not is_val
                    else self.val_accessibility_cache,
                )

                # Log individual components for analysis using log_dict for proper step tracking
                if hasattr(self, "log_dict") and reward_components:
                    reward_metrics = {
                        f"reward_components/{name}_mean": values.mean()
                        for name, values in reward_components.items()
                    }
                    self.log_dict(
                        reward_metrics,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                    )

        elif self.cfg.ddpo.dynamic_constraint_rewards.use:
            # print("Using dynamic constraint rewards and universal rewards")
            # Get room type from config
            room_type = self.cfg.dataset.data.room_type  # default

            if hasattr(self.cfg.ddpo, "dynamic_constraint_rewards"):
                room_type = self.cfg.ddpo.dynamic_constraint_rewards.get(
                    "room_type", self.cfg.dataset.data.room_type
                )

            indices = cond_dict["idx"]
            floor_plan_args_list = [
                self.dataset.get_floor_plan_args(idx) for idx in indices
            ]
            floor_plan_args = {
                key: [args[key] for args in floor_plan_args_list]
                for key in [
                    "floor_plan_centroid",
                    "floor_plan_vertices",
                    "floor_plan_faces",
                    "room_outer_box",
                ]
            }

            # Compute composite reward with all physics constraints
            is_val = len(self.dataset) <= 200
            rewards, reward_components = composite_reward(
                scenes=x0,
                scene_vec_desc=self.scene_vec_desc,
                cfg=self.cfg,
                room_type=room_type,
                reward_normalizer=self.reward_normalizer,
                get_reward_functions=self.get_reward_functions,
                floor_polygons=[
                    self.dataset.get_floor_polygon_points(idx)
                    for idx in cond_dict["idx"]
                ],
                indices=indices,
                is_val=is_val,
                sdf_cache_dir=self.cfg.dataset.sdf_cache_dir,
                sdf_cache=self.train_sdf_cache if not is_val else self.val_sdf_cache,
                accessibility_cache=self.train_accessibility_cache
                if not is_val
                else self.val_accessibility_cache,
                floor_plan_args=floor_plan_args,
            )

            # Log individual components for analysis using log_dict for proper step tracking
            if hasattr(self, "log_dict") and reward_components:
                reward_metrics = {
                    f"reward_components/{name}_mean": values.mean()
                    for name, values in reward_components.items()
                }
                # print("Logging reward components...", reward_metrics.keys())
                self.log_dict(
                    reward_metrics,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

        elif self.cfg.ddpo.use_prompt_following_reward:
            prompts = cond_dict["language_annotation"]
            rewards = prompt_following_reward(
                scenes=x0, prompts=prompts, scene_vec_desc=self.scene_vec_desc
            )
        elif self.cfg.ddpo.use_physical_feasible_objects_reward:
            rewards = number_of_physically_feasible_objects_reward(
                scenes=x0,
                scene_vec_desc=self.scene_vec_desc,
                cfg=self.cfg.ddpo.physical_feasibility,
                num_workers=self.cfg.ddpo.num_reward_workers,
            )
        else:
            raise ValueError("Need to select one reward function.")

        return rewards

    def compute_advantages(
        self, rewards: torch.Tensor, phase: str = "training"
    ) -> torch.Tensor:
        """
        Compute advantages from rewards. The advantages are normalized rewards.

        When using multiple GPU workers, this method synchronizes reward statistics
        across all workers to ensure consistent advantage scaling.

        Args:
            rewards (torch.Tensor): Rewards of shape (B,).
            phase (str): Phase of training. Used for logging.

        Returns:
            torch.Tensor: Advantages of shape (B,).
        """
        # Small epsilon to prevent division by zero.
        eps = 1e-12

        # Synchronize statistics across all workers if using distributed training.
        if self.trainer.world_size > 1:
            # Compute local statistics.
            local_reward_mean = rewards.mean()
            local_reward_squared_mean = (rewards**2).mean()

            # Gather statistics from all workers.
            gathered_means = self.all_gather(local_reward_mean)
            gathered_squared_means = self.all_gather(local_reward_squared_mean)

            # Compute global statistics.
            reward_mean = gathered_means.mean()
            # Need to aggregate according to Var = E[(X - μ)²] = E[X²] - μ².
            global_reward_squared_mean = gathered_squared_means.mean()
            global_reward_var = global_reward_squared_mean - reward_mean**2
            reward_std = torch.sqrt(torch.clamp(global_reward_var, min=eps))
        else:
            # Use local statistics for single worker.
            reward_mean = rewards.mean()
            reward_std = rewards.std()

        # Compute advantages using synchronized statistics.
        advantages = (rewards - reward_mean) / (reward_std + eps)  # Shape (B,)

        self.log_dict(
            {
                f"{phase}/mean_reward": reward_mean.item(),
                f"{phase}/std_reward": reward_std.item(),
            },
            sync_dist=True,
            batch_size=self.cfg.ddpo.batch_size,
        )

        # Clip the advantages
        advantages = torch.clamp(
            advantages,
            min=-self.cfg.ddpo.advantage_max,
            max=self.cfg.ddpo.advantage_max,
        )

        return advantages

    def compute_ddpm_loss(self, batch: Dict[str, torch.Tensor], num_train_timesteps: int| None=None) -> torch.Tensor:
        """
        DDPM forward pass.
        This is a replication of the DDPM forward pass in the `trainer_ddpm.py` file
        to allow the RL trainers to stay separate from the DDPM trainer.
        """
        scenes = batch["scenes"]

        # Sample noise to add to the scenes.
        noise = torch.randn(scenes.shape).to(self.device)  # Shape (B, N, V)

        # Sample a timestep for each scene.
        timesteps = (
            torch.randint(
                0,
                num_train_timesteps if num_train_timesteps is not None else self.noise_scheduler.config.num_train_timesteps,
                (scenes.shape[0],),
            )
            .long()
            .to(self.device)
        )

        # Add noise to the scenes.
        noisy_scenes = self.noise_scheduler.add_noise(
            scenes, noise, timesteps
        )  # Shape (B, N, V)

        predicted_noise = self.predict_noise(
            noisy_scenes=noisy_scenes, timesteps=timesteps, cond_dict=batch
        )  # Shape (B, N, V)

        # Compute loss.
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def sample_scenes(
        self,
        num_samples: int,
        is_test: bool = False,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Sample scenes from the model. The returned scenes are unormalized.

        Args:
            num_samples (int): The number of scenes to sample.
            is_test (bool): Whether its the testing phase.
            batch_size (Optional[int]): The batch size to use for sampling scenes. If
                None, sample all scenes at once.
            use_ema (bool): Whether to use the EMA model.
            data_batch (Dict[str, torch.Tensor] | None): The optional data batch. Some
                sampling methods might use parts of this as conditioning for sampling.

        Returns:
            torch.Tensor: The unnormalized scenes of shape (num_samples, N, V).
        """
        # oldtODO: allow inpainting mask here for rl inpaint sampling
        cond_dict = None
        if data_batch is not None:
            cond_dict = self.dataset.sample_data_dict(
                data=data_batch, num_items=num_samples
            )
        sampled_scenes = super().sample_scenes(
            num_samples=num_samples,
            is_test=is_test,
            batch_size=batch_size,
            use_ema=use_ema,
            data_batch=cond_dict,
        )

        # Compute rewards for the sampled scenes.
        with torch.no_grad():
            # Predict doesn't support logging.
            if not self.trainer.state.stage == "predict":
                rewards = self.compute_rewards_from_trajs(
                    sampled_scenes.unsqueeze(1),  # Shape (B, 1, N, V)
                    cond_dict=cond_dict,
                    are_trajectories_normalized=False,
                )
                self.log("sampled_scenes/reward", rewards.mean().item())
                print(f"[Ashok] Sampled scenes mean reward: {rewards.mean().item()}")

        return sampled_scenes
