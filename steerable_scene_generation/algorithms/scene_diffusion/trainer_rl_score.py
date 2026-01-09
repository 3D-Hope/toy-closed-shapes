from typing import Dict

import torch
import numpy as np
from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .trainer_rl import SceneDiffuserTrainerRL


class SceneDiffuserTrainerScore(SceneDiffuserTrainerRL):
    """
    Class that provides REINFORCE (score function gradient estimator) training logic.
    This corresponds to DPPO_{SF} (https://arxiv.org/abs/2305.13301).
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)
        self.incremental_training = self.cfg.ddpo.incremental_training
        self.joint_training = self.cfg.ddpo.joint_training
        if self.incremental_training and self.joint_training:
            raise ValueError(
                "Cannot have both incremental_training and joint_training set to True."
            )
        if self.incremental_training:
            self.training_steps = self.cfg.ddpo.training_steps_start
            self.increments = list(self.cfg.ddpo.increments)
            # self.training_steps_per_increment = [6000, 5500, 5100, 4800, 4600, 4300, 4100, 3900, 3700, 3600]
            if self.cfg.ddpo.increment_type == 'constant':
                self.training_steps_per_increment = [self.cfg.ddpo.training_iter_per_increment for _ in range(len(self.increments))]
            elif self.cfg.ddpo.increment_type == 'linear':
                self.training_steps_per_increment = [self.cfg.ddpo.increment_linear_slope * i for i in self.increments]
                print(f"[Ashok] Linear increment type with slope {self.cfg.ddpo.increment_linear_slope},increments {self.increments} training steps per increment: {self.training_steps_per_increment}")
            # self.training_steps_per_increment = [1 for _ in range(10)]  # For testing
            self.cum_sum_steps = np.cumsum(self.training_steps_per_increment).tolist()
            self.min_denoising_steps = min(self.increments)
            self.max_denoising_steps = max(self.increments)
            self.num_increments = len(self.increments)
        # self.joint_training_timesteps = [10, 25, 40, 65, 80, 95, 110, 125, 150] if self.joint_training else None
        self.joint_training_timesteps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] if self.joint_training else None

    # def get_incremental_timesteps(self, k):
    #     L = list(range(0, 895, 6))
    #     n = len(L)
    #     sample_sizes = list(range(10, 151, 10))
    #     k = sample_sizes[k]    
    #     indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    #     return [L[i] for i in indices]


    def _merge_cond_dicts(self, trajectory_groups):
        """Merge conditioning dicts from all groups into one batch."""
        if not trajectory_groups or trajectory_groups[0]['cond_dict'] is None:
            return None
            
        merged = {}
        for key in trajectory_groups[0]['cond_dict'].keys():
            values = [group['cond_dict'][key] for group in trajectory_groups]
            if isinstance(values[0], torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                merged[key] = sum(values, [])  # Flatten lists
            else:
                merged[key] = values  # Keep as list
        return merged

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str = "training",
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        DDPO-like (https://arxiv.org/abs/2305.13301) forward pass. Training with RL.
        Returns the loss.
        """

        if self.incremental_training:
            # k = min(
            #     (self.training_steps // self.training_steps_per_increment),
            #     self.num_increments - 1,
            # )
            #  n_timesteps_to_sample = list(range(10, 151, 150//self.num_increments))[k]
            
            if self.training_steps >= self.cum_sum_steps[-1]:
                n_timesteps_to_sample = 150
            else:
                which_increment = np.searchsorted(self.cum_sum_steps, self.training_steps)
                # n_timesteps_to_sample = list(range(self.min_denoising_steps, self.max_denoising_steps + 1, self.max_denoising_steps//self.num_increments))[which_increment]
                n_timesteps_to_sample = self.increments[int(which_increment)]
            
            
        else:
            n_timesteps_to_sample = self.cfg.ddpo.n_timesteps_to_sample
        # Get diffusion trajectories.
        result = self.generate_trajs_for_ddpo(
            last_n_timesteps_only=self.cfg.ddpo.last_n_timesteps_only,
            n_timesteps_to_sample=n_timesteps_to_sample,
            batch=batch,
            incremental_training=self.incremental_training,
            joint_training=self.joint_training,
            joint_training_timesteps=self.joint_training_timesteps,
            phase=phase,
        )
        
        # Handle different return types based on training mode
        if self.joint_training:
            # Result is a list of trajectory groups
            trajectory_groups = result[0]
            
            # Process each group separately
            all_rewards = []
            all_log_prob_sums = []
            
            for group in trajectory_groups:
                # Remove initial noisy scene
                trajectories = group['trajectories'][:, 1:]  # (B_group, T, N, V)
                log_probs = group['log_probs']  # (B_group, T)
                cond_dict = group['cond_dict']
                
                # Compute rewards (uses only last timestep)
                rewards = self.compute_rewards_from_trajs(
                    trajectories=trajectories, cond_dict=cond_dict
                )  # (B_group,)
                
                # Sum log probs across timesteps
                log_prob_sums = torch.sum(log_probs, dim=1)  # (B_group,)
                
                all_rewards.append(rewards)
                all_log_prob_sums.append(log_prob_sums)
                
                # print(f"[Ashok] Group with {group['n_steps']} steps: {trajectories.shape[0]} samples, "
                #       f"log_prob_sum range: [{log_prob_sums.min().item():.3f}, {log_prob_sums.max().item():.3f}]")
            
            # Concatenate all groups
            rewards = torch.cat(all_rewards, dim=0)  # (B,)
            log_prob_sums = torch.cat(all_log_prob_sums, dim=0)  # (B,)
            
            # Compute advantages across full batch
            advantages = self.compute_advantages(rewards, phase=phase)  # (B,)
            # print(f" rewards {rewards}, advantages {advantages}, log_prob_sums {log_prob_sums}  ")
            # REINFORCE loss
            loss = -torch.mean(log_prob_sums * advantages)
            print(f"[Ashok] Joint training - total samples: {rewards.shape[0]}, reinforce loss: {loss.item()}")
            
            # DDPM regularization (merge all cond_dicts)
            if self.cfg.ddpo.ddpm_reg_weight > 0.0:
                merged_cond_dict = self._merge_cond_dicts(trajectory_groups)
                ddpm_loss = self.compute_ddpm_loss(merged_cond_dict)
                loss += ddpm_loss * self.cfg.ddpo.ddpm_reg_weight
                print(f"[Ashok] reg ddpm loss values: {ddpm_loss.item()*self.cfg.ddpo.ddpm_reg_weight}")
        
        else:
            # Standard/incremental training - single tensor return
            trajectories, trajectories_log_props, cond_dict = result
            
            if self.incremental_training and phase == "training":
                self.training_steps += 1
                print(f"[Ashok] Incremental training {self.training_steps} timesteps.")
                # if self.training_steps % self.training_steps_per_increment == 0:
                #     print(
                #         f"[Ashok] Incremental training: Moving to next increment after {self.training_steps} steps. current steps: {n_timesteps_to_sample}"
                #     )
            
            # Remove initial noisy scene.
            trajectories = trajectories[
                :, 1:
            ]  # Shape (B, T, N, V) T=timesteps per sample eg, 150

            # Compute rewards.
            rewards = self.compute_rewards_from_trajs(
                trajectories=trajectories, cond_dict=cond_dict
            )  # Shape (B,)

            # Compute advantages.
            advantages = self.compute_advantages(rewards, phase=phase)  # Shape (B,)

            # REINFORCE loss.
            # print(f"[Ashok] trajectories_log_props: {trajectories_log_props.shape}, self.training_steps: {self.training_steps if self.incremental_training else 'N/A'}") # (B, T)
            loss = -torch.mean(torch.sum(trajectories_log_props, dim=1) * advantages)
            print(f"[Ashok] reinforce loss values: {loss.item()}")
            # DDPM loss for regularization.
            if self.cfg.ddpo.ddpm_reg_weight > 0.0:
                # ddpm_loss = self.compute_ddpm_loss(batch)
                ddpm_loss = self.compute_ddpm_loss(cond_dict)
                # ddpm_loss = self.compute_ddpm_loss(cond_dict, num_train_timesteps=n_timesteps_to_sample) #TODO: try this for incremental
                loss += ddpm_loss * self.cfg.ddpo.ddpm_reg_weight
                print(
                    f"[Ashok] reg ddpm loss values: {ddpm_loss.item()*self.cfg.ddpo.ddpm_reg_weight}"
                )

        return loss
