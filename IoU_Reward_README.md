# 3D IoU Reward for RL Training

This feature adds a new reward function for RL training that penalizes object collisions in generated 3D scenes. The reward is based on the Intersection over Union (IoU) metric between axis-aligned bounding boxes of objects.

## Overview

The IoU reward helps to:

1. Reduce unrealistic object penetrations/collisions in generated scenes
2. Improve the physical plausibility of generated scenes
3. Guide the diffusion model to learn more physically valid object placements through RL

## How It Works

The reward function works by:

1. Extracting object positions, sizes, and class information from the scene representation
2. Identifying valid (non-empty) objects in the scene
3. Constructing axis-aligned bounding boxes for each valid object
4. Computing the IoU between all pairs of valid bounding boxes
5. Taking the negative of the average IoU (so less overlap = higher reward)
6. Using this value directly as the reward signal for RL

## How It Differs from IoU Regularization

While both the IoU regularization and IoU reward aim to reduce object collisions, they work differently:

- **IoU Regularization**: Applied during standard diffusion training as an additional loss term with timestep-dependent weighting
- **IoU Reward**: Used as a direct reward signal for RL training, without timestep-dependent weighting

## Configuration

To use IoU as a reward, add these parameters to your RL training command:

```bash
algorithm.ddpo.use_object_number_reward=False \
algorithm.ddpo.use_iou_reward=True \
```

## Implementation

The implementation includes:

1. A `calculate_scene_iou` function in `trainer_ddpm.py` that calculates the raw IoU values
2. An `iou_reward` function in `ddpo_helpers.py` that converts these IoU values into rewards
3. Updates to `trainer_rl.py` to handle the new reward type

## Usage Example

```bash
bash bash_scripts/run_rl_with_iou_reward.sh
```

This script sets up RL training with the IoU reward and optimizes memory usage to prevent OOM errors.

## Tips for Tuning

- Monitor the reward values during training to ensure they're providing a meaningful signal
- Adjust learning rates and PPO parameters based on the reward dynamics
- Consider combining with other rewards if needed (this would require custom reward function)

## Limitations

- The current implementation only handles axis-aligned bounding boxes
- IoU calculation may be computationally expensive with many objects
- May need to balance against other objectives (like having enough objects in the scene)