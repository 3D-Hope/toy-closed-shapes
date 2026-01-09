"""
Helpers for the DDPO algorithm (https://arxiv.org/abs/2305.13301).
"""
import os
import math
import multiprocessing

from functools import partial
from typing import List, Optional, Tuple, Union

import torch

from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from omegaconf import DictConfig
from pydrake.all import QueryObject, SignedDistancePair

from dynamic_constraint_rewards.commons import get_dynamic_reward
from dynamic_constraint_rewards.scale_raw_rewards import RewardNormalizer
from steerable_scene_generation.algorithms.common.dataclasses import (
    PlantSceneGraphCache,
    SceneVecDescription,
)
from steerable_scene_generation.utils.drake_utils import (
    create_plant_and_scene_graph_from_scene_with_cache,
)
from steerable_scene_generation.utils.prompt_following_metrics import (
    compute_prompt_following_metrics,
)
from universal_constraint_rewards.commons import (
    get_universal_reward,
    parse_and_descale_scenes,
)
from universal_constraint_rewards.physcene import (
    collision_constraint,
    room_layout_constraint,
    walkability_constraint,
)

from .inpainting_helpers import (
    generate_empty_object_inpainting_masks,
    generate_physical_feasibility_inpainting_masks,
)

num_cpus = multiprocessing.cpu_count()


def ddpm_step_with_logprob(
    scheduler: DDPMScheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    generator=None,
    prev_sample: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied and adapted from
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py
    to return the log probability of the previous sample. If the previous sample is not
    provided, it is computed from the model output as in the original implementation.
    The style matches the original implementation to facilitate comparison.

    Predict the sample from the previous timestep by reversing the SDE. This function
    propagates the diffusion process from the learned model outputs (most often the
    predicted noise).

    Args:
        scheduler (`diffusers.DDPMScheduler`):
            The scheduler object that contains the parameters of the diffusion process.
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        prev_sample (`torch.Tensor`, *optional*): The previous sample. If not provided,
            it is computed from the model output.
        mask (`torch.Tensor`, *optional*): A boolean mask of shape matching sample.
            If provided, only True elements contribute to the log probability.
            Shape: (B, N, V) where True means "editable/include in log prob".

    Returns:
        A tuple containing the (predicted) previous sample and the log probability of
        the previous sample.
    """
    assert isinstance(
        scheduler, DDPMScheduler
    ), "scheduler must be an instance of DDPMScheduler"

    t = timestep

    prev_t = scheduler.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in [
        "learned",
        "learned_range",
    ]:
        model_output, predicted_variance = torch.split(
            model_output, sample.shape[1], dim=1
        )
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one "
            "of `epsilon`, `sample` or `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * sample
    )

    # 6. Add noise
    variance_noise = randn_tensor(
        model_output.shape,
        generator=generator,
        device=model_output.device,
        dtype=model_output.dtype,
    )
    if scheduler.variance_type == "fixed_small_log":
        variance = scheduler._get_variance(t, predicted_variance=predicted_variance)
    elif scheduler.variance_type == "learned_range":
        variance = scheduler._get_variance(t, predicted_variance=predicted_variance)
        variance = torch.exp(0.5 * variance)
    else:
        variance = (
            scheduler._get_variance(t, predicted_variance=predicted_variance) ** 0.5
        )

    if prev_sample is None:
        # Don't add noise at t=0.
        prev_sample = (
            pred_prev_sample + variance * variance_noise if t > 0 else pred_prev_sample
        )

    # Log probability of prev_sample (Gaussian distribution).
    log_prob = (
        -((prev_sample.detach() - pred_prev_sample) ** 2) / (2 * (variance**2))
        - torch.log(variance)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    
    # Apply mask if provided: zero out log prob for masked (frozen) elements
    if mask is not None:
        log_prob = log_prob * mask.float()
    
    # Compute mean log probability over all but batch dimension. This is the combined
    # log probability as the individual elements of xt are independent.
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob


def ddim_step_with_logprob(
    scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 1.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied and adapted from
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
    to return the log probability of the previous sample. If the previous sample is not
    provided, it is computed from the model output as in the original implementation.
    The style matches the original implementation to facilitate comparison.

    Predict the sample at the previous timestep by reversing the SDE. Core function to
    propagate the diffusion process from the learned model outputs (most often the
    predicted noise).

    Args:
        scheduler (`diffusers.DDIMScheduler`): scheduler object that contains the
            parameters of the diffusion process.
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output`
            from the clipped predicted original sample. Necessary because predicted
            original sample is clipped to [-1, 1] when `self.config.clip_sample` is
            `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will
            have not effect.
        generator: random number generator.
        prev_sample (`torch.Tensor`, *optional*): The previous sample. If not provided,
            it is computed from the model output.
        mask (`torch.Tensor`, *optional*): A boolean mask of shape matching sample.
            If provided, only True elements contribute to the log probability.
            Shape: (B, N, V) where True means "editable/include in log prob".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the (predicted) previous
            sample and the log probability of the previous sample.
    """
    assert isinstance(
        scheduler, DDIMScheduler
    ), "scheduler must be an instance of DDIMScheduler"
    assert eta >= 0.0, "eta must be non-negative"
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' "
            "after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (
            beta_prod_t**0.5
        ) * sample
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of "
            "`epsilon`, `sample`, or `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from
    # https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from
    # https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either "
            "`generator` or `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # Log probability of prev_sample (Gaussian distribution).
    std_dev_t = torch.clip(std_dev_t, min=1e-6)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    
    # Apply mask if provided: zero out log prob for masked (frozen) elements
    if mask is not None:
        log_prob = log_prob * mask.float()
    
    # Compute mean log probability over all but batch dimension. This is the combined
    # log probability as the individual elements of xt are independent.
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob


def compute_non_penetration_reward(
    scene: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cache: Optional[PlantSceneGraphCache] = None,
    return_updated_cache: bool = True,
) -> Union[float, Tuple[float, PlantSceneGraphCache]]:
    """
    Get the non-penetration reward for a scene. The reward is the sum of the negative
    distances between the objects in the scene. If the scene is collision-free, the
    reward is 0.0 (the best possible reward).

    Args:
        scene (torch.Tensor): The unormalized scene to score. The scene is represented as
            a tensor of shape (num_objects, num_features).
        scene_vec_desc (SceneVecStructureDescription): The description of the scene
            vector structure.
        cache (Optional[PlantSceneGraphCache]): The PlantSceneGraphCache. If None or if
            the objects in the scene have changed, the plant and scene graph are
            recreated.
        return_updated_cache (bool): If True, the updated PlantSceneGraphCache is
            returned.

    Returns:
        The non-penetration reward for the scene. If return_updated_cache is True, the
        updated PlantSceneGraphCache is also returned.
    """
    # Create the diagram for the scene.
    cache, context, _ = create_plant_and_scene_graph_from_scene_with_cache(
        scene=scene, scene_vec_desc=scene_vec_desc, cache=cache
    )
    scene_graph = cache.scene_graph
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object: QueryObject = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )

    # Get all negative distances between the objects in the scene. These are the
    # penetration distances.
    signed_distance_pairs: List[
        SignedDistancePair
    ] = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.0)
    distances = [pair.distance for pair in signed_distance_pairs]

    # Compute the non-penetration reward.
    reward = sum(distances) if len(distances) > 0 else 0.0

    if return_updated_cache:
        return reward, cache
    return reward


def non_penetration_reward(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    num_workers: int = 1,
    cache: Optional[PlantSceneGraphCache] = None,
    return_updated_cache: bool = False,
) -> torch.Tensor:
    """
    Compute the non-penetration reward for a scene. The reward is the sum of the
    negative distances between the objects in the scene. If the scene is collision-free,
    the reward is 0.0 (the best possible reward).

    Args:
        scenes (torch.Tensor): The unormalized scenes to score. The scenes are
            represented as a tensor of shape (batch_size, num_objects, num_features).
        scene_vec_desc (SceneVecStructureDescription): The description of the scene
            vector structure.
        num_workers (int): The number of workers to use for parallel processing. Note
            that using multiple workers prevents the use of the cache and thus might be
            slower if all the scenes contain the same objects.
        cache (Optional[PlantSceneGraphCache]): The PlantSceneGraphCache. If None or if
            the objects in the scene have changed, the plant and scene graph are
            recreated.
        return_updated_cache (bool): If True, the updated PlantSceneGraphCache is
            returned.

    Returns:
        The non-penetration reward for the scenes. If return_updated_cache is True, the
        updated PlantSceneGraphCache is also returned.
    """
    device = scenes.device
    scenes = scenes.cpu().detach().numpy()

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            rewards = pool.map(
                partial(
                    compute_non_penetration_reward,
                    scene_vec_desc=scene_vec_desc,
                    cache=cache,
                    return_updated_cache=False,
                ),
                scenes,
            )
            rewards = torch.tensor(rewards, device=device)
    else:
        rewards = torch.zeros(scenes.shape[0], device=device)
        for i, scene in enumerate(scenes):
            rewards[i], cache = compute_non_penetration_reward(
                scene=scene, scene_vec_desc=scene_vec_desc, cache=cache
            )

    if return_updated_cache:
        return rewards, cache
    print("[Ashok] non-penetration rewards:", rewards)
    return rewards


def object_number_reward(
    scenes: torch.Tensor, scene_vec_desc: SceneVecDescription, cfg=None
) -> torch.Tensor:
    """
    Compute the object number reward for a scene. The reward is the number of objects
    in the scene.

    Args:
        scenes (torch.Tensor): The unormalized scenes to score of shape (B, N, V).
        scene_vec_desc (SceneVecStructureDescription): The description of the scene
            vector structure.

    Returns:
        The object number reward for the scenes of shape (B,).
    """
    rewards = torch.zeros(scenes.shape[0], device=scenes.device)  # Shape (B,)
    for i, scene in enumerate(scenes):
        # Count non-empty objects.
        if cfg is None or not cfg.custom.use:
            num_objects = sum(
                scene_vec_desc.get_model_path(obj) is not None for obj in scene
            )
        else:
            # Custom format with 30 dimensions and first 22 are class labels
            num_objects = (
                (scene[:, : cfg.custom.num_classes].argmax(dim=-1) != 21).sum().item()
            )
        rewards[i] = num_objects
    print("[Ashok] object no rewards:", rewards)
    return rewards


def iou_reward(scenes: torch.Tensor, scene_diffuser, cfg) -> torch.Tensor:
    """
    Compute the IoU reward for scenes. The reward is the negative of the average IoU
    between valid objects in each scene. This encourages scenes with less object overlap.

    Args:
        scenes (torch.Tensor): The unormalized scenes to score of shape (B, N, V).
        scene_diffuser: The scene diffuser model with IoU calculation function.
        cfg: Optional configuration object.

    Returns:
        The IoU reward for the scenes of shape (B,).
    """
    if scene_diffuser is None:
        raise ValueError("scene_diffuser must be provided for IoU reward calculation")

    # Convert list of scenes to batch tensor if needed
    if isinstance(scenes, list):
        scene_batch = torch.stack(scenes, dim=0)
    else:
        scene_batch = scenes

    iou_values = scene_diffuser.bbox_iou_regularizer(
        recon=scene_batch, num_classes=cfg.custom.num_classes, using_as_reward=True
    )
    # TODO: AVOID SELF IOU
    # Convert to list for compatibility if needed
    if isinstance(scenes, list):
        return iou_values.detach().cpu().tolist()
    # print("[Ashok] IoU values:", iou_values.shape)
    rewards = iou_values  # unnormalized raw iou values
    print("[Ashok] IoU rewards:", rewards)
    return rewards


def two_beds_reward(
    scenes: torch.Tensor, scene_vec_desc: SceneVecDescription, cfg=None
) -> torch.Tensor:
    """
    Reward = 1 if there are exactly 2 beds in the scene, else 0.
    """
    rewards = torch.zeros(scenes.shape[0], device=scenes.device)  # Shape (B,)
    for i, scene in enumerate(scenes):
        # Sum probabilities for bed objects
        beds_idx = [8, 15, 11]
        print(
            "[Ashok] scene[:, : cfg.custom.num_classes]:",
            scene[:, : cfg.custom.num_classes],
        )
        # Sum the probabilities of bed classes across all objects
        bed_probabilities = scene[
            :, beds_idx
        ].sum()  # TODO: need better reward. this naive approaach will simply lead to higher probs for bed classes not 100% but generally higher.
        rewards[i] = bed_probabilities.item()
    print("[Ashok] 2 beds rewards:", rewards)
    return rewards


def has_sofa_reward(
    scenes: torch.Tensor, scene_vec_desc: SceneVecDescription, cfg=None
) -> torch.Tensor:
    """
    Reward = 1 if there is a sofa in the scene, else 0.
    """
    rewards = torch.zeros(scenes.shape[0], device=scenes.device)  # Shape (B,)
    # print("[Ashok] scenes: ", scenes)
    for i, scene in enumerate(scenes):
        # print(f"class probs in reward {scene[:, : cfg.custom.num_classes]}")
        # Check if sofa class is present
        sofa_idx = 17
        has_sofa = (scene[:, sofa_idx] > 0).any().item()
        rewards[i] = float(has_sofa)
    print("[Ashok] has sofa rewards:", rewards)
    return rewards


def universal_reward(
    parsed_scenes: dict,
    scene_vec_desc: SceneVecDescription,
    indices,
    cfg=None,
    room_type: str = "bedroom",
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute composite reward using multiple physics-based constraints.

    This function uses the get_composite_reward from physical_constraint_rewards.commons
    which handles:
    - Gravity following (objects should rest on ground)
    - Non-penetration (no overlapping objects)
    - Must-have furniture (room-specific requirements)
    - Object count (realistic scene density)

    All rewards are normalized to [-1, 0] range, then weighted by importance.

    Args:
        parsed_scenes (dict): The parsed scene to score.
        scene_vec_desc (SceneVecDescription): The description of the scene vector structure.
        cfg (DictConfig, optional): Configuration object.
        room_type (str): Type of room for must-have furniture ('bedroom', 'living_room', etc.)
            If None, uses defaults from config or commons.py.

    Returns:
        tuple: (total_rewards, reward_components)
            - total_rewards: Tensor of shape (B,) with combined rewards
            - reward_components: Dict with individual reward values for logging
    """
    from universal_constraint_rewards.commons import get_universal_reward

    # Get number of classes from config
    num_classes = cfg.custom.num_classes if cfg and hasattr(cfg, "custom") else 22

    # Use importance weights from config if not provided
    task_cfg = cfg.ddpo.dynamic_constraint_rewards

    # Get task-specific settings
    # task_reward_type = task_cfg.get('task_reward_type', 'has_sofa')
    # task_weight = task_cfg.get('task_weight', 2.0)
    room_type = task_cfg.get("room_type", "bedroom")
    user_query = cfg.ddpo.dynamic_constraint_rewards.user_query
    user_query = user_query.replace(' ', '_').replace('.', '')
    weights_path = os.path.join(cfg.ddpo.dynamic_constraint_rewards.reward_base_dir, f"{user_query}_responses_tmp/llm_response_4.json")
    
    # Read json
    with open(weights_path, "r") as f:
        import json
        importance_weights = json.load(f)
    # print("[Ashok] Importance weights for universal reward:", importance_weights, "json ", weights_path)
    if task_cfg.get("room_type") == "bedroom":
        num_classes = 22
    elif task_cfg.get("room_type") == "livingroom":
        num_classes = 25
    else:
        raise ValueError(f"Unknown room type: {task_cfg.get('room_type')}")

    # Compute composite reward
    total_rewards, reward_components = get_universal_reward(
        parsed_scenes=parsed_scenes,
        num_classes=num_classes,
        importance_weights=importance_weights,
        room_type=room_type,
        indices=indices,
        **kwargs,
    )


    print(f"[Ashok] Total universal rewards: {total_rewards}")

    return total_rewards, reward_components


def physcene_reward(
    parsed_scene: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    floor_plan_args,
    cfg=None,
    room_type: str = "bedroom",
    weight_coll=2.0,
    weight_walk=100.0,
    weight_layout=300000.0,
    **kwargs,
) -> torch.Tensor:
    # Temporary code to test with fixed floor plan args
    # import os
    # import pickle
    # if not os.path.exists("/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/universal_constraint_rewards/floor_plan_args.pkl"):
    #     with open("/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/universal_constraint_rewards/floor_plan_args.pkl", "wb") as f:
    #         pickle.dump(floor_plan_args, f)

    # collision_loss = collision_constraint(parsed_scene, floor_plan_args=floor_plan_args)
    layout_loss = room_layout_constraint(parsed_scene, floor_plan_args=floor_plan_args)
    # walkability_loss = walkability_constraint(parsed_scene, floor_plan_args=floor_plan_args)

    # print(f"before scaling - Collision loss: {collision_loss}, Walkability loss: {walkability_loss}, Layout loss: {layout_loss}")
    # print(f"after scaling - Collision loss: {weight_coll*collision_loss}, Walkability loss: {weight_walk*walkability_loss}, Layout loss: {weight_layout*layout_loss}")

    # total_loss = weight_coll*collision_loss + weight_walk*walkability_loss + weight_layout*layout_loss
    total_loss = layout_loss
    rewards = -total_loss  # Negative loss as reward

    print(f"[Ashok] Physcene rewards: {rewards}")
    reward_components = {
        # 'collision_reward': -weight_coll*collision_loss,
        # 'layout_reward': -weight_layout*layout_loss,
        "layout_reward": -layout_loss,
        # 'walkability_reward': -weight_walk*walkability_loss,
    }
    return rewards, reward_components


def composite_reward(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg: DictConfig,
    room_type: str,
    reward_normalizer,
    get_reward_functions: dict,
    floor_polygons,
    indices,
    is_val,
    sdf_cache_dir,
    sdf_cache,
    accessibility_cache,
    floor_plan_args,
) -> tuple[torch.Tensor, dict]:
    """
    Compute composite reward (general scene quality) plus task-specific reward.

    This combines:
    - Composite reward: gravity + non-penetration + must-have + object count
    - Task-specific reward: has_sofa, two_beds, etc.

    Final reward = composite_reward + task_weight * task_reward

    Args:
        scenes (torch.Tensor): The scenes to score of shape (B, N, V).
        scene_vec_desc (SceneVecDescription): The description of the scene vector structure.
        cfg (DictConfig): Configuration object (required for task settings).
        reward_normalizer: Normalizer to scale rewards to [0, 1] range.
        get_reward_functions (dict): Dictionary of dynamic reward functions.
        room_type (str): Type of room for must-have furniture.

    Returns:
        tuple: (total_rewards, reward_components)
            - total_rewards: Tensor of shape (B,) with combined rewards
            - reward_components: Dict with individual reward values for logging
    """
    if cfg is None or not hasattr(cfg.ddpo, "dynamic_constraint_rewards"):
        raise ValueError(
            "cfg.ddpo.dynamic_constraint_rewards configuration is required"
        )

    task_cfg = cfg.ddpo.dynamic_constraint_rewards

    # Get task-specific settings
    # task_reward_type = task_cfg.get('task_reward_type', 'has_sofa')
    # task_weight = task_cfg.get('task_weight', 2.0)
    room_type = task_cfg.get("room_type", "bedroom")
    
    user_query = cfg.ddpo.dynamic_constraint_rewards.user_query
    user_query = user_query.replace(' ', '_').replace('.', '')
    
    weights_path = os.path.join(cfg.ddpo.dynamic_constraint_rewards.reward_base_dir, f"{user_query}_responses_tmp/llm_response_4.json")
    
    # Read json
    # with open(weights_path, "r") as f:
    #     import json
    #     importance_weights = json.load(f)
    importance_weights = None
    
    if task_cfg.get("room_type") == "bedroom":
        num_classes = 22
    elif task_cfg.get("room_type") == "livingroom":
        num_classes = 25
    else:
        raise ValueError(f"Unknown room type: {task_cfg.get('room_type')}")

    # Get number of classes from config
    # num_classes = cfg.custom.num_classes if cfg and hasattr(cfg, "custom") else 22

    parsed_scenes = parse_and_descale_scenes(scenes, num_classes=num_classes, room_type=room_type)
    # print(f"[Ashok] parsed scene {parsed_scenes}")
    # for key in parsed_scenes:
    #     print(f"[Ashok] datatype of {key} is {type(parsed_scenes[key])}")
    if not cfg.ddpo.dynamic_constraint_rewards.dynamic_only and cfg.ddpo.dynamic_constraint_rewards.universal_weight > 0:  # 1. Compute composite reward (general scene quality)
        universal_total, universal_components = get_universal_reward(
            parsed_scenes=parsed_scenes,
            reward_normalizer=reward_normalizer,
            num_classes=num_classes,
            importance_weights=importance_weights,
            room_type=room_type,
            floor_polygons=floor_polygons,
            indices=indices,
            is_val=is_val,
            sdf_cache=sdf_cache,
            floor_plan_args=floor_plan_args,
            accessibility_cache=accessibility_cache,
        )
    if cfg.ddpo.dynamic_constraint_rewards.dynamic_weight > 0:  # 2. Compute task-specific reward
        dynamic_total, dynamic_components = get_dynamic_reward(
            parsed_scenes=parsed_scenes,
            reward_normalizer=None, # Note: testing without normalizer for dynamic rewards
            get_reward_functions=get_reward_functions,
            num_classes=num_classes,
            dynamic_importance_weights=None, # Note: testing without importance weights for dynamic rewards
            config=cfg,
            floor_polygons=floor_polygons,
            indices=indices,
            is_val=is_val,
            sdf_cache_dir=sdf_cache_dir,
            sdf_cache=sdf_cache,
            accessibility_cache=accessibility_cache,
        )
    if not cfg.ddpo.dynamic_constraint_rewards.dynamic_only:
        if cfg.ddpo.dynamic_constraint_rewards.universal_weight <= 0:
            total_rewards = dynamic_total
            reward_components = dynamic_components.copy()
            
        elif cfg.ddpo.dynamic_constraint_rewards.dynamic_weight <= 0:
            total_rewards = universal_total
            reward_components = universal_components.copy()
        else:
            total_rewards = cfg.ddpo.dynamic_constraint_rewards.universal_weight * universal_total + cfg.ddpo.dynamic_constraint_rewards.dynamic_weight * dynamic_total
            reward_components = universal_components.copy()
            reward_components.update(dynamic_components)
    else:
        total_rewards = dynamic_total
        reward_components = dynamic_components
    # print(f"[Ashok] composite reward components: {reward_components.keys()}")
    return total_rewards, reward_components


def prompt_following_reward(
    scenes: torch.Tensor, prompts: list[str], scene_vec_desc: SceneVecDescription
) -> torch.Tensor:
    """
    Compute the prompt following reward for a set of scenes based on their prompts.

    This function calculates the fraction of prompts that are followed correctly
    by the corresponding scenes. It utilizes the `compute_prompt_following_metrics`
    function to derive the necessary metrics.

    Args:
        scenes (torch.Tensor): The unormalized scenes to evaluate, of shape (B, N, V).
        prompts (list[str]): A list of textual prompts describing the scenes.
        scene_vec_desc (SceneVecDescription): The scene vector description.

    Returns:
        torch.Tensor: The prompt following rewards of shape (B,), representing the
        fraction of correctly followed prompts.
    """
    if not len(scenes) == len(prompts):
        raise ValueError(
            "The number of scenes and prompts must be the same. "
            f"Got {len(scenes)} scenes and {len(prompts)} prompts."
        )

    prompt_following_metrics = compute_prompt_following_metrics(
        scene_vec_desc=scene_vec_desc, scenes=scenes, prompts=prompts, disable_tqdm=True
    )

    rewards = torch.tensor(
        prompt_following_metrics["per_prompt_following_fractions"],
        device=scenes.device,
    )  # Shape (B,)
    return rewards


def compute_physically_feasible_objects_reward(
    scene: torch.Tensor, scene_vec_desc: SceneVecDescription, cfg: DictConfig
) -> float:
    """
    Compute the number of physically feasible objects reward for a single scene.

    Args:
        scene (torch.Tensor): The unormalized scene to evaluate, of shape (N, V).
        scene_vec_desc (SceneVecDescription): The description of the scene
            vector structure.
        cfg (DictConfig): The configuration for the physical feasibility.

    Returns:
        float: The number of physically feasible objects reward for the scene.
    """
    # Add batch dimension for the single scene
    scene_batch = scene.unsqueeze(0)  # Shape (1, N, V)

    physical_mask, _, _ = generate_physical_feasibility_inpainting_masks(
        scenes=scene_batch,
        scene_vec_desc=scene_vec_desc,
        non_penetration_threshold=cfg.non_penetration_threshold,
        use_sim=cfg.use_sim,
        sim_duration=cfg.sim_duration,
        sim_time_step=cfg.sim_time_step,
        sim_translation_threshold=cfg.sim_translation_threshold,
        sim_rotation_threshold=cfg.sim_rotation_threshold,
        static_equilibrium_distance_threshold=cfg.static_equilibrium_distance_threshold,
    )  # Shape (1, N, V)

    empty_mask, _ = generate_empty_object_inpainting_masks(
        scenes=scene_batch, scene_vec_desc=scene_vec_desc
    )  # Shape (1, N, V)

    combined_mask = torch.logical_or(physical_mask, empty_mask)  # Shape (1, N, V)

    # Invert so that the mask represents the physically feasible objects
    combined_mask_inverted = torch.logical_not(combined_mask)  # Shape (1, N, V)

    # Convert to object-level for reward value consistency
    object_level_masks = combined_mask_inverted.any(dim=2)  # Shape (1, N)

    # The reward is the number of objects that are physically feasible
    reward = object_level_masks.sum().item()  # Scalar

    return reward


def number_of_physically_feasible_objects_reward(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg: DictConfig,
    num_workers: int = 1,
) -> torch.Tensor:
    """
    Compute the number of physically feasible objects reward for a scene. The reward
    is the number of objects that are physically feasible (non-penetration and
    static equilibrium).

    Args:
        scenes (torch.Tensor): The unormalized scenes to evaluate, of shape (B, N, V).
        scene_vec_desc (SceneVecDescription): The description of the scene
            vector structure.
        cfg (DictConfig): The configuration for the physical feasibility.
        num_workers (int): The number of workers to use for parallel processing.

    Returns:
        The number of physically feasible objects reward for the scenes of shape (B,).
    """
    device = scenes.device
    scenes_cpu = scenes.cpu().detach()

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            rewards = pool.map(
                partial(
                    compute_physically_feasible_objects_reward,
                    scene_vec_desc=scene_vec_desc,
                    cfg=cfg,
                ),
                scenes_cpu,
            )
            rewards = torch.tensor(rewards, device=device)
    else:
        rewards = torch.zeros(scenes.shape[0], device=device)
        for i, scene in enumerate(scenes_cpu):
            rewards[i] = compute_physically_feasible_objects_reward(
                scene=scene,
                scene_vec_desc=scene_vec_desc,
                cfg=cfg,
            )

    return rewards.float()
