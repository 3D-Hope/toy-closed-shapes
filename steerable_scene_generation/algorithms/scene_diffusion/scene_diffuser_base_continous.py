import logging

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch

from diffusers import DDIMScheduler
from tqdm import tqdm

from steerable_scene_generation.datasets.scene.scene import SceneDataset
from steerable_scene_generation.utils.caching import conditional_cache

from .scene_diffuser_base import SceneDiffuserBase

logger = logging.getLogger(__name__)


class SceneDiffuserBaseContinous(SceneDiffuserBase, ABC):
    """
    Abstract base class for continous diffusion on scene vectors. This builds on top of
    `SceneDiffuserBase` to add continous diffusion specific functionality.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

    @abstractmethod
    def predict_noise(
        self,
        noisy_scenes: torch.Tensor,
        timesteps: Union[torch.IntTensor, int],
        cond_dict: Dict[str, Any] = None,
        use_ema: bool = False,
    ) -> torch.Tensor:
        """Predict the noise for a batch of noisy scenes.

        Args:
            noisy_scenes (torch.Tensor): Input of shape (B, N, V) where N are the
                number of objects and V is the object feature vector length.
            timesteps (Union[torch.IntTensor, int]): The diffusion step to condition
                the denoising on.
            cond_dict: Dict[str, Any]: The dict containing the conditioning information.
            use_ema (bool): Whether to use the EMA model.

        Returns:
            torch.Tensor: Output of same shape as the input.
        """
        raise NotImplementedError

    def sample_scenes_without_guidance(
        self,
        num_samples: int,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Sample scenes from the model without guidance. The scenes are inverse
        normalized.
        """
        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Sample random noise.
        print()
        
        xt = self.sample_continuous_noise_prior(
            (
                num_samples,
                2
            )
        )  

        for t in tqdm(
            self.noise_scheduler.timesteps, desc="Sampling scenes", leave=False
        ):
            with torch.no_grad():
                # Keep timestep on CPU for indexing scheduler tensors
                t_expanded = t.expand(data_batch['scenes'].shape[0])
                residual = self.predict_noise(
                    xt, t_expanded, cond_dict=data_batch, use_ema=use_ema
                )

            # Compute the updated sample.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    residual, t, xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                # raise NotImplementedError("Only DDIM scheduler is implemented for now")
                scheduler_out = self.noise_scheduler.step(residual, t, xt)

            # Update the sample.
            xt = scheduler_out.prev_sample  # Shape (B, N, V)

            if self.cfg.visualization.visualize_intermediate_scenes:
                self.visualize_intermediate_scene(t, xt)


        return xt

    def sample_scenes_with_classifier_free_guidance(
        self, num_samples: int, cond_dict: Dict[str, Any] = None, use_ema: bool = False
    ) -> torch.Tensor:
        """
        Sample scenes from the model with classifier-free guidance.

        Args:
            num_samples (int): The number of samples to generate.
            cond_dict: Dict[str, Any]: The dict containing the conditioning information.
            use_ema (bool): Whether to use the EMA model.

        Returns:
            torch.Tensor: The generated samples of shape (B, N, V). The samples are
                unormalized.
        """
        raise NotImplementedError("sample with cfg not implemented yet")
        if cond_dict is not None:
            # Add the mask labels to the cond_dict.
            cond_dict = self.dataset.add_classifier_free_guidance_uncond_data(
                cond_dict.copy()
            )

        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Sample random noise.
        num_objects_per_scene = (
            self.cfg.max_num_objects_per_scene
            + self.cfg.num_additional_tokens_for_sampling
        )
        xt = self.sample_continuous_noise_prior(
            (
                num_samples,
                num_objects_per_scene,
                self.scene_vec_desc.get_object_vec_len(),
            )
        )  # Shape (B, N, V)

        for t in tqdm(
            self.noise_scheduler.timesteps,
            desc="Sampling scenes with classifier-free guidance",
            leave=False,
        ):
            with torch.no_grad():
                prediction = self.predict_noise(
                    xt.repeat(2, 1, 1), t, cond_dict=cond_dict, use_ema=use_ema
                )  # Shape (B*2, N, V)
                cond_pred = prediction[:num_samples]  # Shape (B, N, V)
                uncond_pred = prediction[num_samples:]  # Shape (B, N, V)

                # Residual has shape (B, N, V).
                weight = self.cfg.classifier_free_guidance.weight
                residual = (1 + weight) * cond_pred - weight * uncond_pred

            # Compute the updated sample.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    residual, t, xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(residual, t, xt)

            # Update the sample.
            xt = scheduler_out.prev_sample  # Shape (B, N, V)

        # Apply inverse normalization.
        xt = self.dataset.inverse_normalize_scenes(xt)  # Shape (B, N, V)

        return xt

    @torch.no_grad
    def sample_scenes_continous_or_discrete_only(
        self,
        num_samples: int,
        data_batch: Dict[str, torch.Tensor],
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        Sample scenes from the model where the continous or discrete part is kept from
        the scenes in `data_batch`.

        Args:
            num_samples (int): The number of samples to generate.
            data_batch (Dict[str, torch.Tensor]): The data batch that contains scenes of
                shape (M, N, V). Note that M must be greater or equal than
                `num_samples`.
            use_ema (bool): Whether to use the EMA model.

        Returns:
            torch.Tensor: The generated samples of shape (B, N, V). The samples are
                unormalized.
        """
        raise NotImplementedError("sample continous or discrete only not implemented yet")
        assert (
            self.cfg.continuous_discrete_only.continuous_only
            or self.cfg.continuous_discrete_only.discrete_only
        )
        scene_data_batch = data_batch["scenes"]
        assert len(scene_data_batch) >= num_samples
        scene_data_batch = scene_data_batch.to(self.device)

        if self.cfg.classifier_free_guidance.use:
            # Add the mask labels to the cond_dict.
            data_batch = self.dataset.add_classifier_free_guidance_uncond_data(
                data_batch.copy()
            )

        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Sample random noise for the continous or discrete part while taking the other
        # part from the data batch.
        if self.cfg.num_additional_tokens_for_sampling > 0:
            raise NotImplementedError(
                "Sampling from the continous or discrete part only is not implemented "
                "when there are additional tokens for sampling."
            )
        xt = self.sample_continuous_noise_prior(
            (
                num_samples,
                2
            )
        )  # Shape (B, N, V)
        if self.cfg.continuous_discrete_only.continuous_only:
            mask = torch.concatenate(
                [
                    torch.ones(
                        self.scene_vec_desc.translation_vec_len
                        + len(self.scene_vec_desc.rotation_parametrization)
                    ),
                    torch.zeros(self.scene_vec_desc.model_path_vec_len),
                ]
            ).to(self.device)
        elif self.cfg.continuous_discrete_only.discrete_only:
            mask = torch.concatenate(
                [
                    torch.zeros(
                        self.scene_vec_desc.translation_vec_len
                        + len(self.scene_vec_desc.rotation_parametrization)
                    ),
                    torch.ones(self.scene_vec_desc.model_path_vec_len),
                ]
            ).to(self.device)
        mask_expanded = (
            mask.unsqueeze(0).unsqueeze(0).expand(xt.shape)
        )  # Shape (B, N, V)
        xt = xt * mask_expanded + scene_data_batch[:num_samples] * (1 - mask_expanded)

        for t in tqdm(
            self.noise_scheduler.timesteps,
            desc="Sampling scenes (continuous or discrete part only)",
            leave=False,
        ):
            if self.cfg.classifier_free_guidance.use:
                prediction = self.predict_noise(
                    xt.repeat(2, 1, 1), t, cond_dict=data_batch, use_ema=use_ema
                )  # Shape (B*2, N, V)
                cond_pred = prediction[:num_samples]  # Shape (B, N, V)
                uncond_pred = prediction[num_samples:]  # Shape (B, N, V)

                # Residual has shape (B, N, V).
                weight = self.cfg.classifier_free_guidance.weight
                residual = (1 + weight) * cond_pred - weight * uncond_pred
            else:
                residual = self.predict_noise(xt, t, use_ema=use_ema)  # Shape (B, N, V)

            # Compute the updated sample.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    residual, t, xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(residual, t, xt)

            # Update the discrete/ continuous part of the sample.
            xt = scheduler_out.prev_sample * mask_expanded + xt * (1 - mask_expanded)

        # Apply inverse normalization.
        xt = self.dataset.inverse_normalize_scenes(xt)  # Shape (B, N, V)

        return xt

    @conditional_cache(argument_name="is_test")  # Only cache during test time.
    def sample_scenes(
        self,
        num_samples: int,
        is_test: bool = False,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = num_samples
        # print(f"[Ashok] sampling cfg {self.cfg.classifier_free_guidance.use}, scheduler {self.noise_scheduler}, eta {self.cfg.noise_schedule.ddim.eta}")

        # Determine the batches in which to sample the scenes.
        num_batches = num_samples // batch_size
        remainder = num_samples % batch_size
        batch_sizes = [batch_size] * num_batches
        if remainder > 0:
            batch_sizes.append(remainder)

        if (
            data_batch is not None
            and self.cfg.classifier_free_guidance.use
            and len(data_batch["text_cond"]["input_ids"]) != num_samples
        ):
            # Ensure the data batch size matches the number of samples.
            if len(data_batch["text_cond"]["input_ids"]) > num_samples:
                # Take first num_samples samples.
                data_batch = {k: v[:num_samples] for k, v in data_batch.items()}
            else:
                logger.warning(
                    f"Data batch size {len(data_batch['text_cond']['input_ids'])} is "
                    f"smaller than the number of samples {num_samples}. Sampling "
                    f"the first {len(data_batch['text_cond']['input_ids'])} samples."
                )
                data_batch = self.dataset.sample_data_dict(
                    data=data_batch, num_items=num_samples
                )

        # Optionally replace the data labels with the specified labels during testing.
        if (
            is_test
            and not self.cfg.classifier_free_guidance.sampling.use_data_labels
            and data_batch is not None
        ):
            txt_labels = self.cfg.classifier_free_guidance.sampling.labels
            data_batch = self.dataset.replace_cond_data(
                data=data_batch, txt_labels=txt_labels
            )

        # Sample scenes.
        sampled_scene_batches = []
        for num in tqdm(batch_sizes, desc="Sampling scene batches", leave=False):
            if (
                self.cfg.continuous_discrete_only.continuous_only
                or self.cfg.continuous_discrete_only.discrete_only
            ):
                scenes = self.sample_scenes_continous_or_discrete_only(
                    num, data_batch=data_batch, use_ema=use_ema
                )
            elif self.cfg.classifier_free_guidance.use:
                scenes = self.sample_scenes_with_classifier_free_guidance(
                    num, cond_dict=data_batch, use_ema=use_ema
                )
            else:
                print(f"[Ashok] sampling scene with out guidance")
                scenes = self.sample_scenes_without_guidance(
                    num, use_ema=use_ema, data_batch=data_batch
                )
            sampled_scene_batches.append(scenes)
        sampled_scenes = torch.cat(sampled_scene_batches, dim=0)

        if is_test:
            # Compute sampled scene metrics.
            self.log_sampled_scene_metrics(
                sampled_scenes, name="sampled_scenes/before_processing"
            )

        # Apply post-processing.
        sampled_scenes = self.apply_postprocessing(sampled_scenes).to(self.device)

        if is_test:
            # Compute sampled scene metrics.
            self.log_sampled_scene_metrics(
                sampled_scenes, name="sampled_scenes/after_processing"
            )

        return sampled_scenes

    def inpaint_scenes(
        self,
        data_batch: Dict[str, torch.Tensor],
        to_hardcode: Dict[str, int] = None,
        inpaint_masks=None,
        scenes=None,
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        to_hardcode: Dict[str, int] = None
        {obj_idx: count}
        """
        raise NotImplementedError("inpaint scenes not implemented yet")
        # Extract scenes and masks from the data batch.
        if to_hardcode is None:
            if scenes is None:
                print(f"[Ashok] inpainting using data batch scenes with toy 3 single beds")
                scenes = data_batch["scenes"]  # Shape (B, N, V)
                scenes[:, :3, :22] = -1.0  # (B, 2, 22)
                # For both objects in all scenes, set single_bed class (index 15) to 1.0
                scenes[:, :3, 15] = 1.0

            # print(f"[Ashok] inpainting mask {inpainting_masks[0]}")
            # print(f"[Ashok] scenes {scenes[0]}")

            if inpaint_masks is None:
                # Set these class probability slots for both objects to -1 as default (not marked class)

                # try:
                #     inpainting_masks = data_batch["inpainting_masks"]  # Shape (B, N, V)
                # except:
                # Fix inpainting mask and scene initialization for all scenes: set first two objects as single beds.
                inpainting_masks = torch.ones_like(scenes, dtype=torch.bool)
                # Set the first 22 dimensions (assumed class one-hot for 22 classes) for first two objects as not to inpaint (fixed)
                inpainting_masks[:, :3, :22] = False  # (B, 2, 22)

                
        if to_hardcode is not None:
            scenes = data_batch["scenes"]  # Shape (B, N, V)
            if self.cfg.custom.objfeat_dim == 0:
                scenes = scenes[:, :, :-32]

            inpainting_masks = torch.ones_like(scenes, dtype=torch.bool)
            hardcoded_count = 0
            for obj_idx, count in to_hardcode.items():
                inpainting_masks[:, hardcoded_count:hardcoded_count + count, :self.cfg.custom.num_classes] = False
                scenes[:, hardcoded_count:hardcoded_count + count, :self.cfg.custom.num_classes] = -1.0
                scenes[:, hardcoded_count:hardcoded_count + count, obj_idx] = 1.0
                hardcoded_count += count
                
                
        if not scenes.shape == inpainting_masks.shape:
            raise ValueError(
                "Scenes and inpainting masks must have the same shape. "
                f"Got {scenes.shape} and {inpainting_masks.shape}."
            )

        # Set timesteps for inference.
        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Initialize with random noise for masked regions.
        xt = self.sample_continuous_noise_prior(scenes.shape)  # Shape (B, N, V)
        
        print(f"[Ashok] xt noise shape {xt.shape}, scenes shape {scenes.shape}, inpainting mask shape {inpainting_masks.shape}")
        xt = torch.where(inpainting_masks, xt, scenes)  # Apply mask

        if self.cfg.classifier_free_guidance.use:
            # Add the mask labels to the data_batch.
            data_batch = self.dataset.add_classifier_free_guidance_uncond_data(
                data_batch.copy()
            )
        num_samples = scenes.shape[0]
        for t in tqdm(
            self.noise_scheduler.timesteps, desc="Inpainting scenes", leave=False
        ):
            with torch.no_grad():
                if self.cfg.classifier_free_guidance.use:
                    # Double the batch for classifier-free guidance.
                    noise_pred = self.predict_noise(
                        noisy_scenes=xt.repeat(2, 1, 1),
                        timesteps=t,
                        cond_dict=data_batch,
                        use_ema=use_ema,
                    )  # Shape (B*2, N, V)

                    noise_pred_cond = noise_pred[:num_samples]  # Shape (B, N, V)
                    noise_pred_uncond = noise_pred[num_samples:]  # Shape (B, N, V)

                    # Apply classifier-free guidance.
                    weight = self.cfg.classifier_free_guidance.weight
                    noise_pred = (
                        1 + weight
                    ) * noise_pred_cond - weight * noise_pred_uncond
                else:
                    noise_pred = self.predict_noise(
                        noisy_scenes=xt,
                        timesteps=t,
                        cond_dict=data_batch,
                        use_ema=use_ema,
                    )  # Shape (B, N, V)
            # print(f"[Ashok] inpainting cfg {self.cfg.classifier_free_guidance.use}, scheduler {self.noise_scheduler}, eta {self.cfg.noise_schedule.ddim.eta}")
            # Update the sample for masked regions.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    noise_pred, t, xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(noise_pred, t, xt)

            xt_next = scheduler_out.prev_sample  # Shape (B, N, V)

            # Only update masked regions, keep unmasked regions fixed.
            xt = torch.where(inpainting_masks, xt_next, scenes)

        # Apply inverse normalization.
        inpainted_scenes = self.dataset.inverse_normalize_scenes(xt)  # Shape (B, N, V)

        # Compute inpainted scene metrics.
        self.log_sampled_scene_metrics(
            inpainted_scenes, name="inpainted_scenes/before_processing"
        )

        # Apply post-processing.
        inpainted_scenes = self.apply_postprocessing(inpainted_scenes).to(self.device)

        # Compute inpainted scene metrics after processing.
        self.log_sampled_scene_metrics(
            inpainted_scenes, name="inpainted_scenes/after_processing"
        )

        return inpainted_scenes

    def _extract(self, a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def descale_to_origin(self, x, minimum, maximum):
        """
        x shape : BxNx3
        minimum, maximum shape: 3
        """
        x = (x + 1) / 2
        x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
        return x

    def bbox_iou_regularizer(
        self, recon, num_classes, t=None, iou_weight=0.1, using_as_reward=False
    ):
        """
        Calculate IoU-based regularization loss to penalize mesh/bbox collisions.

        Args:
            recon: Reconstructed scene tensor of shape (B, N, D)
                  where D includes class labels, positions, sizes, etc.
            t: Timesteps for the diffusion process, shape (B,)
            num_classes: Number of object classes
            iou_weight: Weight for the IoU loss component

        Returns:
            loss_iou: IoU-based regularization loss
        """
        batch_size = recon.shape[0]
        num_objects = recon.shape[1]
        device = recon.device

        # Define indices for your representation components
        class_indices = list(range(0, num_classes))
        pos_indices = list(
            range(len(class_indices), len(class_indices) + 3)
        )  # Next 3 dimensions for position
        size_indices = list(
            range(
                len(class_indices) + len(pos_indices),
                len(class_indices) + len(pos_indices) + 3,
            )
        )  # Next 3 dimensions for size

        # Extract positions and sizes
        positions = recon[:, :, pos_indices]  # Shape: [B, N, 3]
        sizes = recon[:, :, size_indices]  # Shape: [B, N, 3]
        classes = recon[:, :, class_indices]  # Shape: [B, N, num_classes]

        descale_trans = self.descale_to_origin(
            positions,
            torch.tensor([-2.7625005, 0.045, -2.75275], device=device),
            torch.tensor([2.7784417, 3.6248395, 2.8185427], device=device),
        )
        descale_sizes = self.descale_to_origin(
            sizes,
            torch.tensor([0.03998289, 0.02000002, 0.012772], device=device),
            torch.tensor([2.8682, 1.770065, 1.698315], device=device),
        )
        positions, sizes = descale_trans, descale_sizes
        # Determine if an object is empty (where last class has highest probability)
        # The last class (index num_classes-1) represents empty/no object
        empty_class_idx = num_classes - 1
        class_predictions = classes.argmax(dim=-1)  # Shape: [B, N]
        is_empty = class_predictions == empty_class_idx  # Shape: [B, N]
        valid_mask = (
            ~is_empty
        ).float()  # Shape: [B, N], 1 for valid objects, 0 for empty

        # Create mask for valid object pairs (only compare valid objects with each other)
        bbox_iou_mask = (
            valid_mask[:, :, None] * valid_mask[:, None, :]
        )  # Shape: [B, N, N]

        # Convert positions and sizes to axis-aligned bounding box format [x1, y1, z1, x2, y2, z2]
        # where (x1, y1, z1) is the minimum corner and (x2, y2, z2) is the maximum corner
        mins = positions - sizes  # Shape: [B, N, 3]
        maxs = positions + sizes  # Shape: [B, N, 3]

        # Concatenate to form [x1, y1, z1, x2, y2, z2]
        axis_aligned_bbox_corners = torch.cat([mins, maxs], dim=-1)  # Shape: [B, N, 6]
        assert (
            axis_aligned_bbox_corners.shape[-1] == 6
        ), f"Expected 6 dimensions for bounding box corners, got {axis_aligned_bbox_corners.shape[-1]}"

        # Calculate IoU between all pairs of bounding boxes
        bbox_iou = self.axis_aligned_bbox_overlaps_3d(
            axis_aligned_bbox_corners, axis_aligned_bbox_corners, mode="iou"
        )  # Shape: [B, N, N]

        # Zero out the diagonal (self-IoU is always 1)
        # diag_mask = 1.0 - torch.eye(num_objects, device=device)[None, :, :]  # Shape: [1, N, N]
        # bbox_iou = bbox_iou * diag_mask

        # Only consider IoU between valid objects
        bbox_iou_valid = bbox_iou * bbox_iou_mask  # Shape: [B, N, N]
        if using_as_reward:
            return bbox_iou_valid.sum(
                dim=list(range(1, len(bbox_iou_valid.shape)))
            )  # shape: [B,]
        # Calculate average IoU for valid objects
        # bbox_iou_valid_avg = bbox_iou_valid.sum(dim=list(range(1, len(bbox_iou_valid.shape)))) / (bbox_iou_mask.sum(dim=list(range(1, len(bbox_iou_valid.shape)))) + 1e-6)

        # Get the diffusion timestep-dependent weight
        w_iou = self._extract(
            self.noise_scheduler.alphas_cumprod.to(device), t, bbox_iou.shape
        )

        # Calculate final IoU loss with time-dependent weighting
        loss_iou = (w_iou * iou_weight * bbox_iou_valid).sum(
            dim=list(range(1, len(bbox_iou_valid.shape)))
        ) / (bbox_iou_mask.sum(dim=list(range(1, len(bbox_iou_valid.shape)))) + 1e-6)

        # Return the average IoU loss across the batch
        return loss_iou.mean()

    def axis_aligned_bbox_overlaps_3d(
        self, bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6
    ):
        """
        https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/iou_calculators/iou3d_calculator.py
        """
        """Calculate overlap between two set of axis aligned 3D bboxes. If
            ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
            of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
            bboxes1 and bboxes2.
            Args:
                bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
                    format or empty.
                bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
                    format or empty.
                    B indicates the batch dim, in shape (B1, B2, ..., Bn).
                    If ``is_aligned`` is ``True``, then m and n must be equal.
                mode (str): "iou" (intersection over union) or "giou" (generalized
                    intersection over union).
                is_aligned (bool, optional): If True, then m and n must be equal.
                    Defaults to False.
                eps (float, optional): A value added to the denominator for numerical
                    stability. Defaults to 1e-6.
            Returns:
                Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """

        assert mode in ["iou", "giou"], f"Unsupported mode {mode}"
        # Either the boxes are empty or the length of boxes's last dimension is 6
        assert bboxes1.size(-1) == 6 or bboxes1.size(0) == 0
        assert bboxes2.size(-1) == 6 or bboxes2.size(0) == 0

        # Batch dim must be the same
        # Batch dim: (B1, B2, ... Bn)
        assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
        batch_shape = bboxes1.shape[:-2]

        rows = bboxes1.size(-2)
        cols = bboxes2.size(-2)
        if is_aligned:
            assert rows == cols

        if rows * cols == 0:
            if is_aligned:
                return bboxes1.new(batch_shape + (rows,))
            else:
                return bboxes1.new(batch_shape + (rows, cols))

        area1 = (
            (bboxes1[..., 3] - bboxes1[..., 0])
            * (bboxes1[..., 4] - bboxes1[..., 1])
            * (bboxes1[..., 5] - bboxes1[..., 2])
        )
        area2 = (
            (bboxes2[..., 3] - bboxes2[..., 0])
            * (bboxes2[..., 4] - bboxes2[..., 1])
            * (bboxes2[..., 5] - bboxes2[..., 2])
        )

        if is_aligned:
            lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
            rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

            wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
            overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

            if mode in ["iou", "giou"]:
                union = area1 + area2 - overlap
            else:
                union = area1
            if mode == "giou":
                enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
                enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
        else:
            lt = torch.max(
                bboxes1[..., :, None, :3], bboxes2[..., None, :, :3]
            )  # [B, rows, cols, 3]
            rb = torch.min(
                bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:]
            )  # [B, rows, cols, 3]

            wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
            overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

            if mode in ["iou", "giou"]:
                union = area1[..., None] + area2[..., None, :] - overlap
            if mode == "giou":
                enclosed_lt = torch.min(
                    bboxes1[..., :, None, :3], bboxes2[..., None, :, :3]
                )
                enclosed_rb = torch.max(
                    bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:]
                )

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = overlap / union
        if mode in ["iou"]:
            return ious
        # calculate gious
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area
        return gious
