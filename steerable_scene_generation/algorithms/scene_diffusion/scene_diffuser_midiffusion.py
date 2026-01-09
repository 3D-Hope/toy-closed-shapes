from typing import Any, Dict, Type, Union

import torch

from steerable_scene_generation.algorithms.common.ema_model import EMAModel

from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .models import MIDiffusionContinuous
from .scene_diffuser_base_continous import SceneDiffuserBaseContinous


def create_scene_diffuser_midiffusion(
    trainer_class: Type[SceneDiffuserBaseContinous],
) -> Type[SceneDiffuserBaseContinous]:
    """
    Factory function to create a scene diffuser MIDiffusion class.
    https://arxiv.org/abs/2405.21066

    Args:
        trainer_class (Type[SceneDiffuserBaseContinous]): The base class for the scene
            diffuser. This class should be a subclass of SceneDiffuserBaseContinous and
            implement the forward method and any additional methods needed for the
            training method.

    Returns:
        SceneDiffuser: The scene diffuser class. This class implements any
            model-specific logic needed for the scene diffusion algorithm. See the
            docstring of the returned class for more details.
    """

    class SceneDiffuserMiDiffusion(trainer_class):
        """
        Scene diffusion on a set of un-ordered objects. The number of objects and types
        of objects are not fixed. The object vectors consist of [translation, rotation,
        model_vector]. All scenes have `max_num_objects_per_scene` objects.

        This implements the continuous baseline model from MiDiffusion:
        https://arxiv.org/abs/2405.21066
        """

        def __init__(self, cfg, dataset: SceneDataset):
            """
            cfg is a DictConfig object defined by
            `configurations/algorithm/scene_diffuser_midiffusion.yaml`.
            """
            super().__init__(cfg, dataset=dataset)

        def _build_model(self):
            """Create all pytorch models."""
            super()._build_model()

            # # Conditioning: Text OR Floor (mutually exclusive)
            # if self.cfg.classifier_free_guidance.use:
            #     self.txt_encoder, text_cond_dim = load_txt_encoder_from_config(
            #         self.cfg, component="encoder"
            #     )
            #     self.floor_encoder = None
            #     context_dim = text_cond_dim
            # elif self.cfg.classifier_free_guidance.use_floor:
            #     if self.cfg.model.n_layer == 8: last_dim = 64
            #     elif self.cfg.model.n_layer == 12: last_dim = 512
            #     else: raise NotImplementedError(f"Unsupported n_layer {self.cfg.model.n_layer} for floor encoder last dim")
            #     self.floor_encoder, floor_cond_dim = load_floor_encoder_from_config(last_dim=last_dim)
            #     self.txt_encoder = None
            #     context_dim = floor_cond_dim  # 64D from PointNet
            #     print(f"[Ashok] Using floor encoder with context dim: {context_dim}")
            # else:
            #     self.txt_encoder = None
            #     self.floor_encoder = None
            #     context_dim = 0
            # if self.cfg.custom.old and self.cfg.dataset.data.room_type == "bedroom":
            #     network_dim = {
            #         "objectness_dim": 0,  # Not used by our scene representation
            #         "class_dim": self.scene_vec_desc.get_model_path_vec_len(),
            #         "translation_dim": self.scene_vec_desc.get_translation_vec_len(),
            #         "size_dim": 0,  # Not used by our scene representation
            #         "angle_dim": self.scene_vec_desc.get_rotation_vec_len(),
            #         "objfeat_dim": 0,  # Not used by our scene representation
            #     }
            # # if self.cfg.custom.loss :
            # else:
            #     network_dim = {
            #         "objectness_dim": 0,  # Not used by our scene representation
            #         "class_dim": self.cfg.custom.num_classes,
            #         "translation_dim": self.cfg.custom.translation_dim,
            #         "size_dim": self.cfg.custom.size_dim,
            #         "angle_dim": self.cfg.custom.angle_dim,
            #         "objfeat_dim": self.cfg.custom.objfeat_dim,  # Not used by our scene representation
            #     }
            # network_dim = {
            #         "objectness_dim": 0,  # Not used by our scene representation
            #         "class_dim": self.scene_vec_desc.get_model_path_vec_len(),
            #         "translation_dim": self.scene_vec_desc.get_translation_vec_len(),
            #         "size_dim": 0,  # Not used by our scene representation
            #         "angle_dim": self.scene_vec_desc.get_rotation_vec_len(),
            #         "objfeat_dim": 0,  # Not used by our scene representation
            #     }
            network_dim = {
                "translation_dim": 2
            }
            context_dim = 0
            self.model = MIDiffusionContinuous(
                network_dim=network_dim,
                seperate_all=self.cfg.model.seperate_all,
                n_layer=self.cfg.model.n_layer,
                n_embd=self.cfg.model.n_embd,
                n_head=self.cfg.model.n_head,
                dim_feedforward=self.cfg.model.dim_feedforward,
                dropout=self.cfg.model.dropout,
                activate=self.cfg.model.activate,
                timestep_type=self.cfg.model.timestep_type,
                context_dim=context_dim,  # Text OR Floor context
                mlp_type=self.cfg.model.mlp_type,
            )

            if self.cfg.ema.use:
                self.ema = EMAModel(
                    model=self.model,
                    update_after_step=self.cfg.ema.update_after_step,
                    inv_gamma=self.cfg.ema.inv_gamma,
                    power=self.cfg.ema.power,
                    min_value=self.cfg.ema.min_value,
                    max_value=self.cfg.ema.max_value,
                )

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
                cond_dict: Dict[str, Any]: The dict containing the conditioning
                    information.
                use_ema (bool): Whether to use the EMA model.

            Returns:
                torch.Tensor: Output of same shape as the input.
            """
            # for key in cond_dict:
            # print(f"[Ashok] pred noise, {key}: {cond_dict[key].shape}")
            assert not (use_ema and not self.cfg.ema.use)
            model = self.ema.model if use_ema else self.model
            # print(
            #     f"[Ashok] in predict noise ema {use_ema}, cond dict {cond_dict.keys()}"
            # )
            # print(f"[Ashok] fpbpn.shape: {cond_dict['fpbpn'].shape}")
            # print(f"[Ashok] noisy_scenes.shape: {noisy_scenes.shape}")
            # import sys

            # sys.exit()

            # Process different timestep input formats.
            # if not torch.is_tensor(timesteps):
            #     # Preferably, timesteps should be a tensor to avoid device issues.
            #     timesteps = torch.tensor(
            #         [timesteps], dtype=torch.long, device=self.device
            #     )
            #     # Broadcast to batch dimension.
            #     timesteps = timesteps.expand(noisy_scenes.size(0))  # Shape (B,)
            # elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            #     timesteps = timesteps[None].to(self.device)
            #     # Broadcast to batch dimension.
            #     timesteps = timesteps.expand(noisy_scenes.size(0))  # Shape (B,)

            # for k, v in cond_dict.items():
            #     if torch.is_tensor(v):
            #         if torch.isnan(v).any():
            #             print(f"[WARNING] NaN detected in cond_dict['{k}']!")
            #             print(f"[DEBUG] cond_dict['{k}'] min: {v[~torch.isnan(v)].min()}")
            #             print(f"[DEBUG] cond_dict['{k}'] max: {v[~torch.isnan(v)].max()}")
            #             cond_dict[k] = torch.nan_to_num(v)
            # # Check for NaN in input and replace with zeros
            # if torch.isnan(noisy_scenes).any():
            #     print(f"[WARNING] NaN detected in noisy_scenes! Replacing with zeros.")
            #     noisy_scenes = torch.nan_to_num(noisy_scenes)

            # Context: Text OR Floor (mutually exclusive)
            context = None
            # if cond_dict is not None and self.txt_encoder is not None:
            #     # Text conditioning
            #     with torch.autocast(
            #         device_type=self.device.type,
            #         dtype=(
            #             torch.bfloat16 if self.device.type == "cuda" else torch.float32
            #         ),
            #     ):
            #         text_cond: torch.Tensor = self.txt_encoder(
            #             cond_dict["text_cond"]
            #         )  # Shape (B, max_length, C)
            #         # Average over the sequence length.
            #         text_cond = text_cond.mean(dim=1)  # Shape (B, C)
            #         # Convert to lightning dtype.
            #         text_cond = text_cond.to(noisy_scenes.dtype)

            #         # Expand context along num_objects dimension.
            #         context = text_cond.unsqueeze(1).expand(
            #             -1, noisy_scenes.size(1), -1
            #         )  # Shape (B, N, C)
            # elif cond_dict is not None and self.floor_encoder is not None:
            #     # Floor conditioning (same pattern as Flux)
            #     floor_cond = self.floor_encoder(
            #         cond_dict["fpbpn"].to(noisy_scenes.dtype)
            #     )  # Shape (B, 64)
            #     # print(f"[Ashok] Floor condition shape: {floor_cond.shape}")
            #     floor_cond = floor_cond.to(noisy_scenes.dtype)

            #     # Expand context along num_objects dimension (same as text)
            #     context = floor_cond.unsqueeze(1).expand(
            #         -1, noisy_scenes.size(1), -1
            #     )  # Shape (B, N, 64)
            # print(f"[Ashok] context at predict noise floor {context.shape}")
            # Predict the noise.
            # with torch.autograd.detect_anomaly():
            # print(f"[Ashok] noisy_scenes dtype: {noisy_scenes.dtype}, timesteps dtype: {timesteps.dtype}")
            # Convert to float32 to match model parameters
            noisy_scenes = noisy_scenes.float()
            predicted_noise = model(
                noisy_scenes, time=timesteps, context=context, context_cross=None
            )  # Shape (B, N, V)

            # if torch.isnan(predicted_noise).any():
            #     print(f"[WARNING] NaN detected in predicted_noise!")
            #     # print(f"[DEBUG] predicted_noise min: {predicted_noise[~torch.isnan(predicted_noise)].min()}")
            #     # print(f"[DEBUG] predicted_noise max: {predicted_noise[~torch.isnan(predicted_noise)].max()}")
            #     predicted_noise = torch.nan_to_num(predicted_noise)

            return predicted_noise

        def put_model_in_eval_mode(self) -> None:
            """Put the denoising model in evaluation mode."""
            self.model.eval()
            if self.cfg.ema.use:
                self.ema.eval()

        def on_train_batch_end(self, outputs, batch, batch_idx):
            if self.cfg.ema.use:
                self.ema.step(self.model)

    return SceneDiffuserMiDiffusion
