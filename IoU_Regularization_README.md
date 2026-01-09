# IoU Regularization for 3D Scene Generation

This feature adds a regularization term to the loss function that penalizes object collisions in generated 3D scenes. The regularization is based on the Intersection over Union (IoU) metric between axis-aligned bounding boxes of objects.

## Overview

The IoU regularization helps to:

1. Reduce unrealistic object penetrations/collisions in generated scenes
2. Improve the physical plausibility of generated scenes
3. Guide the diffusion model to learn more physically valid object placements

## How It Works

The regularization works by:

1. Extracting object positions, sizes, and class information from the scene representation
2. Identifying valid (non-empty) objects in the scene
3. Constructing axis-aligned bounding boxes for each valid object
4. Computing the IoU between all pairs of valid bounding boxes
5. Weighting the IoU loss based on the current diffusion timestep (using alphas_cumprod)
6. Adding the weighted IoU loss to the total loss function

## Configuration

To enable and configure IoU regularization, update the following parameters in your configuration file:

```yaml
loss:
  use_iou_regularization: true  # Set to true to enable IoU regularization
  iou_weight: 0.1  # Weight for the IoU loss component
```

You can add these parameters to your specific experiment configuration file or adjust them in `configurations/algorithm/scene_diffuser_base.yaml`.

## Parameters

- `use_iou_regularization` (bool): Whether to use IoU regularization. Default is `false`.
- `iou_weight` (float): Weight for the IoU loss component. Higher values enforce stronger penalties for object collisions. Default is `0.1`.

## Implementation Details

The IoU regularization is implemented in the `bbox_iou_regularizer` function in the `trainer_ddpm.py` file. The function:

1. Extracts object positions, sizes, and class information
2. Identifies valid objects based on class predictions
3. Creates axis-aligned bounding boxes for valid objects
4. Calculates IoU between all valid object pairs
5. Applies a timestep-dependent weight to the loss
6. Returns the weighted average IoU loss

The function uses the `axis_aligned_bbox_overlaps_3d` utility function to calculate IoU between 3D bounding boxes efficiently.

## Usage

To use IoU regularization in your training:

1. Enable it in your configuration:

```yaml
loss:
  use_iou_regularization: true
  iou_weight: 0.1
```

2. Run your training as usual:

```bash
# Your training command
```

3. Monitor the IoU loss in the training logs to see how it decreases over time, indicating fewer object collisions.

## Tips for Tuning

- Start with a small `iou_weight` (e.g., 0.05 or 0.1) and gradually increase if needed
- If generated scenes have too few objects or objects are placed too far apart, reduce the weight
- If objects still collide frequently, increase the weight
- The optimal weight depends on your specific dataset and other loss components

## Visualization

To visualize the effect of IoU regularization, you can:

1. Generate scenes with and without IoU regularization enabled
2. Compare the object placements and any collision/penetration issues
3. Look at the IoU loss values reported during training

## Limitations

- The current implementation only handles axis-aligned bounding boxes, which may not perfectly represent all object geometries
- Very dense scenes may still have some collisions due to the complexity of the optimization problem
- Extreme values of `iou_weight` may lead to unnatural object spacing or too few objects in the scene

## Future Extensions

Possible extensions to this feature include:

- Support for oriented bounding boxes for better collision detection
- Per-class IoU weights to handle different object types differently
- Gradual increase of the IoU weight during training for more stable convergence

## References

The IoU calculation algorithm is based on OpenMMLab's MMDetection3D implementation:
https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/iou_calculators/iou3d_calculator.py