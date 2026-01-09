from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

from omegaconf import DictConfig

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .scene_diffuser_base_continous import SceneDiffuserBaseContinous


def compute_attribute_weighted_ddpm_loss(
    predicted_noise: torch.Tensor,
    noise: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg: DictConfig,
    log_fn: Callable | None = None,
) -> torch.Tensor:
    """
    Computes the attribute-weighted loss for DDPM training.

    Args:
        predicted_noise: The predicted noise tensor.
        noise: The target noise tensor.
        scene_vec_desc: The scene vector descriptor object.
        cfg: Configuration object containing loss weights.
        log_fn: Optional logging function that accepts a dictionary of metrics and a
            batch_size parameter. Expected signature:
            log_fn(metrics_dict: Dict[str, torch.Tensor], batch_size: int) -> None.

    Returns:
        The computed loss value.
    """
    batch_size = predicted_noise.shape[0]

    # Ensure that each object attribute is weighted equally, regardless of its
    # number of parameters.
    translation_loss = F.mse_loss(
        scene_vec_desc.get_translation_vec(predicted_noise),
        scene_vec_desc.get_translation_vec(noise),
    )
    rotation_loss = F.mse_loss(
        scene_vec_desc.get_rotation_vec(predicted_noise),
        scene_vec_desc.get_rotation_vec(noise),
    )
    model_path_loss = F.mse_loss(
        scene_vec_desc.get_model_path_vec(predicted_noise),
        scene_vec_desc.get_model_path_vec(noise),
    )
    loss = (
        cfg.loss.object_translation_attribute_weight * translation_loss
        + cfg.loss.object_rotation_attribute_weight * rotation_loss
        + cfg.loss.object_model_attribute_weight * model_path_loss
    )
    # Normalize the loss for the scaling not to affect the learning rate.
    loss /= (
        cfg.loss.object_translation_attribute_weight
        + cfg.loss.object_rotation_attribute_weight
        + cfg.loss.object_model_attribute_weight
    )

    if log_fn is not None:
        log_fn(
            {
                "training/translation_loss": translation_loss,
                "training/rotation_loss": rotation_loss,
                "training/model_path_loss": model_path_loss,
            },
            batch_size=batch_size,
        )

    return loss


def compute_ddpm_loss(
    predicted_noise: torch.Tensor,
    noise: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg: DictConfig,
    log_fn: Callable | None = None,
) -> torch.Tensor:
    """
    Computes the loss for DDPM training.

    Args:
        predicted_noise: The predicted noise tensor.
        noise: The target noise tensor.
        scene_vec_desc: The scene vector descriptor object.
        cfg: Configuration object containing loss settings.
        log_fn: Optional logging function that accepts a dictionary of metrics and a
            batch_size parameter.

    Returns:
        The computed loss value.
    """
    if cfg.custom.loss:
        if cfg.loss.use_separate_loss_per_object_attribute:
            num_classes = cfg.custom.num_classes
            pos_indices = list(
                range(
                    0, 3
                )  # oldtODO: USE cfg.algorithm.custom.num_classes and so on for these all
            )  # Next 3 dimensions for position
            size_indices = list(
                range(
                    len(pos_indices),
                    len(pos_indices) + 3,
                )
            )  # Next 3 dimensions for size
            rot_indices = list(
                range(
                    len(pos_indices) + len(size_indices),
                    len(pos_indices) + len(size_indices) + 2,
                )
            )  # Next 2 dimensions for rotation
            class_indices = list(
                range(
                    len(pos_indices) + len(size_indices) + len(rot_indices),
                    len(pos_indices) + len(size_indices) + len(rot_indices) + num_classes,
                )
            )
            pos_loss = F.mse_loss(
                predicted_noise[..., pos_indices],
                noise[..., pos_indices],
            )
            size_loss = F.mse_loss(
                predicted_noise[..., size_indices],
                noise[..., size_indices],
            )
            rot_loss = F.mse_loss(
                predicted_noise[..., rot_indices],
                noise[..., rot_indices],
            )
            class_loss = F.mse_loss(
                predicted_noise[..., class_indices],
                noise[..., class_indices],
            )
            loss = pos_loss + size_loss + rot_loss + class_loss

            if log_fn is not None:
                batch_size = predicted_noise.shape[0]
                log_fn(
                    {
                        "training/translation_loss": pos_loss.item(),
                        "training/rotation_loss": rot_loss.item(),
                        "training/class_loss": class_loss.item(),
                        "training/size_loss": size_loss.item(),
                    },
                    batch_size=batch_size,
                )
            return loss
        else:
            loss = F.mse_loss(predicted_noise, noise)
            return loss

    # if cfg.loss.use_separate_loss_per_object_attribute:
    #     return compute_attribute_weighted_ddpm_loss(
    #         predicted_noise=predicted_noise,
    #         noise=noise,
    #         scene_vec_desc=scene_vec_desc,
    #         cfg=cfg,
    #         log_fn=log_fn,
    #     )
    else:
        loss = F.mse_loss(predicted_noise, noise)
        return loss


class SceneDiffuserTrainerDDPM(SceneDiffuserBaseContinous):
    """
    Class that provides the DDPM training logic.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

    def custom_loss_function(
        self, predicted_noise: torch.Tensor, noise: torch.Tensor, timesteps=None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Custom loss function for your specific scene representation.

        Args:
            predicted_noise: Model predictions (B, N, D)
            noise: Ground truth noise (B, N, D)
            timesteps: Optional diffusion timesteps for IoU regularization

        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        total_loss = F.mse_loss(predicted_noise, noise)
        losses_dict = {}
        return total_loss, losses_dict


    def loss_function(
        self, predicted_noise: torch.Tensor, noise: torch.Tensor, timesteps=None
    ) -> torch.Tensor:
        """
        Compute the loss function for the DDPM model.

        Args:
            predicted_noise: The predicted noise tensor.
            noise: The target noise tensor.
            timesteps: Optional diffusion timesteps for time-dependent loss weighting.

        Returns:
            The computed loss value.
        """
        # Check if we're using a custom dataset and should use custom loss
        import numpy as np

        if hasattr(self.cfg, "custom") and self.cfg.custom.loss:
            # print(f"[Ashok] using custom loss")
            total_loss, loss_components = self.custom_loss_function(
                predicted_noise, noise, timesteps
            )

            # Log component losses
            for loss_name, loss_value in loss_components.items():
                self.log(f"train/{loss_name}", loss_value, prog_bar=True)

            return total_loss

        # Original loss calculation
        return compute_ddpm_loss(
            predicted_noise=predicted_noise,
            noise=noise,
            scene_vec_desc=self.scene_vec_desc,
            cfg=self.cfg,
            log_fn=self.log_dict,
        )

    def reset_continuous_or_discrete_part(
        self, new: torch.Tensor, old: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset the continuous or discrete part in `new` with `old`.

        Args:
            new (torch.Tensor): The new vector of shape (B, N, V).
            old (torch.Tensor): The old vector of shape (B, N, V).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of the `new` with the continuous
                or discrete part from `old` and the mask of shape (B, N, V) where ones
                correspond to the part to keep.
        """
        if self.cfg.continuous_discrete_only.discrete_only:
            # Only denoise the discrete part.
            mask = torch.concatenate(
                [
                    torch.zeros(
                        self.scene_vec_desc.translation_vec_len
                        + len(self.scene_vec_desc.rotation_parametrization)
                    ),
                    torch.ones(self.scene_vec_desc.model_path_vec_len),
                ]
            ).to(new.device)
        else:
            # Only denoise the continuous part.
            mask = torch.concatenate(
                [
                    torch.ones(
                        self.scene_vec_desc.translation_vec_len
                        + len(self.scene_vec_desc.rotation_parametrization)
                    ),
                    torch.zeros(self.scene_vec_desc.model_path_vec_len),
                ]
            ).to(new.device)

        # Expand the mask to match the shape of new.
        mask_expanded = (
            mask.unsqueeze(0).unsqueeze(0).expand(new.shape)
        )  # Shape (B, N, V)

        # Reset to old where the mask is zero.
        new_reset = new * mask_expanded + old * (1 - mask_expanded)
        return new_reset, mask_expanded

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str = "training",
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        DDPM forward pass. Normal diffusion training with maximum likelihood objective.
        Returns the loss.
        """
        scenes = batch["scenes"]

        noise = torch.randn(scenes.shape).to(self.device)  # Shape (B, N, V)

        # Sample a timestep for each scene.
        timesteps = (
            torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (scenes.shape[0],),
            )
            .long()
            .to(self.device)
        )

        # Add noise to the scenes.
        noisy_scenes = self.noise_scheduler.add_noise(
            scenes, noise, timesteps
        )  # Shape (B, N, V)
        # print(f"[Ashok] scene before adding noise {scenes.shape} shape of noisy scenes after add noise {noisy_scenes.shape} at trainer ddpm forward")


        predicted_noise = self.predict_noise(
            noisy_scenes=noisy_scenes,
            timesteps=timesteps,
            cond_dict=batch,
            use_ema=use_ema,
        )
        loss = F.mse_loss(predicted_noise, noise)
        return loss


# #--
# from typing import Callable, Dict, Tuple

# import torch
# import torch.nn.functional as F

# from omegaconf import DictConfig

# from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
# from steerable_scene_generation.datasets.scene.scene import SceneDataset

# from .scene_diffuser_base_continous import SceneDiffuserBaseContinous


# def compute_attribute_weighted_ddpm_loss(
#     predicted_noise: torch.Tensor,
#     noise: torch.Tensor,
#     scene_vec_desc: SceneVecDescription,
#     cfg: DictConfig,
#     log_fn: Callable | None = None,
# ) -> torch.Tensor:
#     """
#     Computes the attribute-weighted loss for DDPM training.

#     Args:
#         predicted_noise: The predicted noise tensor.
#         noise: The target noise tensor.
#         scene_vec_desc: The scene vector descriptor object.
#         cfg: Configuration object containing loss weights.
#         log_fn: Optional logging function that accepts a dictionary of metrics and a
#             batch_size parameter. Expected signature:
#             log_fn(metrics_dict: Dict[str, torch.Tensor], batch_size: int) -> None.

#     Returns:
#         The computed loss value.
#     """
#     batch_size = predicted_noise.shape[0]

#     # Ensure that each object attribute is weighted equally, regardless of its
#     # number of parameters.
#     translation_loss = F.mse_loss(
#         scene_vec_desc.get_translation_vec(predicted_noise),
#         scene_vec_desc.get_translation_vec(noise),
#     )
#     rotation_loss = F.mse_loss(
#         scene_vec_desc.get_rotation_vec(predicted_noise),
#         scene_vec_desc.get_rotation_vec(noise),
#     )
#     model_path_loss = F.mse_loss(
#         scene_vec_desc.get_model_path_vec(predicted_noise),
#         scene_vec_desc.get_model_path_vec(noise),
#     )
#     loss = (
#         cfg.loss.object_translation_attribute_weight * translation_loss
#         + cfg.loss.object_rotation_attribute_weight * rotation_loss
#         + cfg.loss.object_model_attribute_weight * model_path_loss
#     )
#     # Normalize the loss for the scaling not to affect the learning rate.
#     loss /= (
#         cfg.loss.object_translation_attribute_weight
#         + cfg.loss.object_rotation_attribute_weight
#         + cfg.loss.object_model_attribute_weight
#     )

#     if log_fn is not None:
#         log_fn(
#             {
#                 "training/translation_loss": translation_loss,
#                 "training/rotation_loss": rotation_loss,
#                 "training/model_path_loss": model_path_loss,
#             },
#             batch_size=batch_size,
#         )

#     return loss


# def compute_ddpm_loss(
#     predicted_noise: torch.Tensor,
#     noise: torch.Tensor,
#     scene_vec_desc: SceneVecDescription,
#     cfg: DictConfig,
#     log_fn: Callable | None = None,
# ) -> torch.Tensor:
#     """
#     Computes the loss for DDPM training.

#     Args:
#         predicted_noise: The predicted noise tensor.
#         noise: The target noise tensor.
#         scene_vec_desc: The scene vector descriptor object.
#         cfg: Configuration object containing loss settings.
#         log_fn: Optional logging function that accepts a dictionary of metrics and a
#             batch_size parameter.

#     Returns:
#         The computed loss value.
#     """
#     if cfg.loss.use_separate_loss_per_object_attribute:
#         return compute_attribute_weighted_ddpm_loss(
#             predicted_noise=predicted_noise,
#             noise=noise,
#             scene_vec_desc=scene_vec_desc,
#             cfg=cfg,
#             log_fn=log_fn,
#         )
#     else:
#         loss = F.mse_loss(predicted_noise, noise)
#         return loss


# class SceneDiffuserTrainerDDPM(SceneDiffuserBaseContinous):
#     """
#     Class that provides the DDPM training logic.
#     """

#     def __init__(self, cfg, dataset: SceneDataset):
#         """
#         cfg is a DictConfig object defined by
#         `configurations/algorithm/scene_diffuser_base_continous.yaml`.
#         """
#         super().__init__(cfg, dataset=dataset)

#     def custom_loss_function(
#         self, predicted_noise: torch.Tensor, noise: torch.Tensor, timesteps=None
#     ) -> Tuple[torch.Tensor, Dict[str, float]]:
#         """
#         Custom loss function for your specific scene representation.

#         Args:
#             predicted_noise: Model predictions (B, N, D)
#             noise: Ground truth noise (B, N, D)
#             timesteps: Optional diffusion timesteps for IoU regularization

#         Returns:
#             total_loss: Combined loss
#             loss_components: Dictionary of individual loss components
#         """
#         # Define indices for your representation components
#         num_classes = self.cfg.custom.num_classes
#         class_indices = list(range(0, num_classes))
#         pos_indices = list(
#             range(len(class_indices), len(class_indices) + 3)
#         )  # Next 3 dimensions for position
#         size_indices = list(
#             range(
#                 len(class_indices) + len(pos_indices),
#                 len(class_indices) + len(pos_indices) + 3,
#             )
#         )  # Next 3 dimensions for size
#         rot_indices = list(
#             range(
#                 len(class_indices) + len(pos_indices) + len(size_indices),
#                 len(class_indices) + len(pos_indices) + len(size_indices) + 2,
#             )
#         )  # Next 2 dimensions for rotation

#         # objfeat_indices = list(
#         #     range(
#         #         len(class_indices)
#         #         + len(pos_indices)
#         #         + len(size_indices)
#         #         + len(rot_indices),
#         #         len(class_indices)
#         #         + len(pos_indices)
#         #         + len(size_indices)
#         #         + len(rot_indices)
#         #         + 32,
#         #     )
#         # )  # All dimensions for object features

#         # Extract components from your representation using your custom indices
#         pred_pos = predicted_noise[..., pos_indices]
#         pred_size = predicted_noise[..., size_indices]
#         pred_rot = predicted_noise[..., rot_indices]
#         pred_class = predicted_noise[..., class_indices]
#         # pred_objfeat = predicted_noise[..., objfeat_indices]

#         target_pos = noise[..., pos_indices]
#         target_size = noise[..., size_indices]
#         target_rot = noise[..., rot_indices]
#         target_class = noise[..., class_indices]
#         # target_objfeat = noise[..., objfeat_indices]

#         # Calculate your custom losses
#         pos_loss = F.mse_loss(pred_pos, target_pos)
#         size_loss = F.mse_loss(pred_size, target_size)
#         rot_loss = F.mse_loss(pred_rot, target_rot)
#         class_loss = F.mse_loss(pred_class, target_class)
#         # objfeat_loss = F.mse_loss(pred_objfeat, target_objfeat)
#         # Weight the losses as needed (you can make these configurable)
#         pos_weight = 1.0
#         size_weight = 1.0
#         rot_weight = 1.0
#         class_weight = 1.0
#         # objfeat_weight = 1.0

#         total_loss = (
#             pos_weight * pos_loss
#             + size_weight * size_loss
#             + rot_weight * rot_loss
#             + class_weight * class_loss
#             # + objfeat_weight * objfeat_loss
#         )

#         return total_loss, {
#             "pos_loss": pos_loss.item(),
#             "size_loss": size_loss.item(),
#             "rot_loss": rot_loss.item(),
#             "class_loss": class_loss.item(),
#             # "objfeat_loss": objfeat_loss.item(),
#         }

#     def bbox_iou_regularizer(self, recon, t, num_classes):
#         pass

#         self.noise_scheduler.alphas_cumprod
#         def _extract(a, t, x_shape):
#             """
#             Extract some coefficients at specified timesteps,
#             then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
#             """
#             bs, = t.shape
#             assert x_shape[0] == bs
#             out = torch.gather(a, 0, t)
#             assert out.shape == torch.Size([bs])
#             return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

#         bounds sizes (array([0.03998289, 0.02000002, 0.012772  ], dtype=float32), array([2.8682  , 1.770065, 1.698315], dtype=float32)), translations (array([-2.7625005,  0.045    , -2.75275  ], dtype=float32), array([2.7784417, 3.6248395, 2.8185427], dtype=float32))
#         # regularizer to penalize mesh collision
#         # alpha_prod_t = scheduler.alphas_cumprod[t], gives wiou
#         # axis_aligned_bbox_corn = torch.cat([ descale_trans - descale_sizes, descale_trans + descale_sizes], dim=-1)
#         #             assert axis_aligned_bbox_corn.shape[-1] == 6 get world coord trans and unnormalized size, using postprocess of the threedfront

#         # only apply iou to non empty objects
#         # obj_recon = x_recon[:, :, self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim]
#                         # valid_mask = (obj_recon <=0).float().squeeze(2)
#         # #bbox_iou_mask = valid_mask[:, :, None] * valid_mask[:, None, :]
#         #                 bbox_iou = axis_aligned_bbox_overlaps_3d(axis_aligned_bbox_corn, axis_aligned_bbox_corn)
#         #                 bbox_iou_valid = bbox_iou * bbox_iou_mask
#         #                 bbox_iou_valid_avg = bbox_iou_valid.sum( dim=list(range(1, len(bbox_iou_valid.shape))) ) / ( bbox_iou_mask.sum( dim=list(range(1, len(bbox_iou_valid.shape))) ) + 1e-6)
#         #                 # get the iou loss weight w.r.t time
#         #                 w_iou = self._extract(self.alphas_cumprod.to(data_start.device), t, bbox_iou.shape)
#         #                 # loss_iou = (w_iou * 0.1 * bbox_iou).mean(dim=list(range(1, len(w_iou.shape))))
#         #                 loss_iou_valid_avg = (w_iou * 0.1 * bbox_iou_valid).sum( dim=list(range(1, len(bbox_iou_valid.shape))) ) / ( bbox_iou_mask.sum( dim=list(range(1, len(bbox_iou_valid.shape))) ) + 1e-6)
#         #             losses_weight += loss_iou_valid_avg


#     def axis_aligned_bbox_overlaps_3d(self, bboxes1,
#                                     bboxes2,
#                                     mode='iou',
#                                     is_aligned=False,
#                                     eps=1e-6):
#         '''
#         https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/iou_calculators/iou3d_calculator.py
#         '''
#         """Calculate overlap between two set of axis aligned 3D bboxes. If
#             ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
#             of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
#             bboxes1 and bboxes2.
#             Args:
#                 bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
#                     format or empty.
#                 bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
#                     format or empty.
#                     B indicates the batch dim, in shape (B1, B2, ..., Bn).
#                     If ``is_aligned`` is ``True``, then m and n must be equal.
#                 mode (str): "iou" (intersection over union) or "giou" (generalized
#                     intersection over union).
#                 is_aligned (bool, optional): If True, then m and n must be equal.
#                     Defaults to False.
#                 eps (float, optional): A value added to the denominator for numerical
#                     stability. Defaults to 1e-6.
#             Returns:
#                 Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
#         """

#         assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
#         # Either the boxes are empty or the length of boxes's last dimension is 6
#         assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
#         assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

#         # Batch dim must be the same
#         # Batch dim: (B1, B2, ... Bn)
#         assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
#         batch_shape = bboxes1.shape[:-2]

#         rows = bboxes1.size(-2)
#         cols = bboxes2.size(-2)
#         if is_aligned:
#             assert rows == cols

#         if rows * cols == 0:
#             if is_aligned:
#                 return bboxes1.new(batch_shape + (rows, ))
#             else:
#                 return bboxes1.new(batch_shape + (rows, cols))

#         area1 = (bboxes1[..., 3] -
#                 bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
#                     bboxes1[..., 5] - bboxes1[..., 2])
#         area2 = (bboxes2[..., 3] -
#                 bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
#                     bboxes2[..., 5] - bboxes2[..., 2])

#         if is_aligned:
#             lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
#             rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

#             wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
#             overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

#             if mode in ['iou', 'giou']:
#                 union = area1 + area2 - overlap
#             else:
#                 union = area1
#             if mode == 'giou':
#                 enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
#                 enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
#         else:
#             lt = torch.max(bboxes1[..., :, None, :3],
#                         bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
#             rb = torch.min(bboxes1[..., :, None, 3:],
#                         bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

#             wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
#             overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

#             if mode in ['iou', 'giou']:
#                 union = area1[..., None] + area2[..., None, :] - overlap
#             if mode == 'giou':
#                 enclosed_lt = torch.min(bboxes1[..., :, None, :3],
#                                         bboxes2[..., None, :, :3])
#                 enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
#                                         bboxes2[..., None, :, 3:])

#         eps = union.new_tensor([eps])
#         union = torch.max(union, eps)
#         ious = overlap / union
#         if mode in ['iou']:
#             return ious
#         # calculate gious
#         enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
#         enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
#         enclose_area = torch.max(enclose_area, eps)
#         gious = ious - (enclose_area - union) / enclose_area
#         return gious

#     def loss_function(
#         self, predicted_noise: torch.Tensor, noise: torch.Tensor, timesteps=None
#     ) -> torch.Tensor:
#         """
#         Compute the loss function for the DDPM model.

#         Args:
#             predicted_noise: The predicted noise tensor.
#             noise: The target noise tensor.
#             timesteps: Optional diffusion timesteps for time-dependent loss weighting.

#         Returns:
#             The computed loss value.
#         """
#         # Check if we're using a custom dataset and should use custom loss
#         import numpy as np

#         if hasattr(self.cfg, "custom") and self.cfg.custom.loss:
#             # print(f"[Ashok] using custom loss")
#             total_loss, loss_components = self.custom_loss_function(
#                 predicted_noise, noise, timesteps
#             )

#             # Log component losses
#             for loss_name, loss_value in loss_components.items():
#                 self.log(f"train/{loss_name}", loss_value, prog_bar=True)

#             return total_loss

#         # Original loss calculation
#         return compute_ddpm_loss(
#             predicted_noise=predicted_noise,
#             noise=noise,
#             scene_vec_desc=self.scene_vec_desc,
#             cfg=self.cfg,
#             log_fn=self.log_dict,
#         )

#     def reset_continuous_or_discrete_part(
#         self, new: torch.Tensor, old: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Reset the continuous or discrete part in `new` with `old`.

#         Args:
#             new (torch.Tensor): The new vector of shape (B, N, V).
#             old (torch.Tensor): The old vector of shape (B, N, V).

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]: A tuple of the `new` with the continuous
#                 or discrete part from `old` and the mask of shape (B, N, V) where ones
#                 correspond to the part to keep.
#         """
#         if self.cfg.continuous_discrete_only.discrete_only:
#             # Only denoise the discrete part.
#             mask = torch.concatenate(
#                 [
#                     torch.zeros(
#                         self.scene_vec_desc.translation_vec_len
#                         + len(self.scene_vec_desc.rotation_parametrization)
#                     ),
#                     torch.ones(self.scene_vec_desc.model_path_vec_len),
#                 ]
#             ).to(new.device)
#         else:
#             # Only denoise the continuous part.
#             mask = torch.concatenate(
#                 [
#                     torch.ones(
#                         self.scene_vec_desc.translation_vec_len
#                         + len(self.scene_vec_desc.rotation_parametrization)
#                     ),
#                     torch.zeros(self.scene_vec_desc.model_path_vec_len),
#                 ]
#             ).to(new.device)

#         # Expand the mask to match the shape of new.
#         mask_expanded = (
#             mask.unsqueeze(0).unsqueeze(0).expand(new.shape)
#         )  # Shape (B, N, V)

#         # Reset to old where the mask is zero.
#         new_reset = new * mask_expanded + old * (1 - mask_expanded)
#         return new_reset, mask_expanded

#     def forward(
#         self,
#         batch: Dict[str, torch.Tensor],
#         phase: str = "training",
#         use_ema: bool = False,
#     ) -> torch.Tensor:
#         """
#         DDPM forward pass. Normal diffusion training with maximum likelihood objective.
#         Returns the loss.
#         """
#         scenes = batch["scenes"]

#         # Sample noise to add to the scenes.
#         noise = torch.randn(scenes.shape).to(self.device)  # Shape (B, N, V)

#         # Sample a timestep for each scene.
#         timesteps = (
#             torch.randint(
#                 0,
#                 self.noise_scheduler.config.num_train_timesteps,
#                 (scenes.shape[0],),
#             )
#             .long()
#             .to(self.device)
#         )

#         # Add noise to the scenes.
#         noisy_scenes = self.noise_scheduler.add_noise(
#             scenes, noise, timesteps
#         )  # Shape (B, N, V)

#         if (
#             self.cfg.continuous_discrete_only.continuous_only
#             or self.cfg.continuous_discrete_only.discrete_only
#         ):
#             # Don't add noise to the continuous or discrete part.
#             noisy_scenes, mask = self.reset_continuous_or_discrete_part(
#                 new=noisy_scenes, old=scenes
#             )

#         predicted_noise = self.predict_noise(
#             noisy_scenes=noisy_scenes,
#             timesteps=timesteps,
#             cond_dict=batch,
#             use_ema=use_ema,
#         )  # Shape (B, N, V)

#         if (
#             self.cfg.continuous_discrete_only.continuous_only
#             or self.cfg.continuous_discrete_only.discrete_only
#         ):
#             predicted_noise *= mask
#             noise *= mask

#         # Compute loss.
#         # Check if we're using a custom dataset and should use custom loss
#         if (
#             hasattr(self.cfg, "dataset_name")
#             and self.cfg.dataset_name == "custom_scene"
#         ):
#             # Custom loss calculation for your representation
#             loss, loss_components = self.custom_loss_function(
#                 predicted_noise, noise, timesteps
#             )

#             # Log component losses
#             for loss_name, loss_value in loss_components.items():
#                 self.log(f"train/{loss_name}", loss_value, prog_bar=True)
#         else:
#             # Original loss calculation
#             loss = self.loss_function(predicted_noise, noise, timesteps)

#         return loss
