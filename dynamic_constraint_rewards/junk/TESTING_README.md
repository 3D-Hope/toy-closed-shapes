# Dynamic Reward Functions - Test Suite Documentation

## Overview

This directory contains dynamic reward functions for evaluating 3D scene layouts. Each reward function is tested with comprehensive test cases that cover various scenarios.

## Data Structure

All reward functions receive a `parsed_scene` dictionary with the following structure:

```python
parsed_scene = {
    "one_hot": torch.Tensor,        # Shape: (B, N, num_classes) - One-hot encoded object types
    "positions": torch.Tensor,       # Shape: (B, N, 3) - 3D positions (x, y, z)
    "sizes": torch.Tensor,           # Shape: (B, N, 3) - Object sizes (width, height, depth)
    "orientations": torch.Tensor,    # Shape: (B, N, 2) - 2D orientation [cos(θ), sin(θ)]
    "object_indices": torch.Tensor,  # Shape: (B, N) - Object class indices
    "is_empty": torch.Tensor,        # Shape: (B, N) - Boolean mask for empty slots
    "device": torch.device           # Device (cuda/cpu)
}
```

Where:
- **B**: Batch size
- **N**: Maximum number of objects per scene (typically 12)
- **num_classes**: Number of object classes (typically 22, including 'empty')

### Additional Parameters

All reward functions also receive:
- **idx_to_labels** (dict): Mapping from object indices to string labels
  ```python
  idx_to_labels = {
      0: "armchair",
      1: "bookshelf",
      # ... more objects ...
      19: "tv_stand",
      20: "wardrobe",
      21: "empty"
  }
  ```

## Reward Functions

### 1. reward_tv_present.py

**Purpose**: Checks if a TV (tv_stand) is present in the scene.

**Returns**: 
- `1.0` if TV is present
- `0.0` if TV is absent

**Test Cases**:
- ✓ Scene with TV → 1.0
- ✓ Scene without TV → 0.0
- ✓ Empty room → 0.0

**Usage**:
```python
from dynamic_reward_functions import reward_tv_present

rewards = reward_tv_present.get_reward(parsed_scene, idx_to_labels=idx_to_labels)
```

---

### 2. reward_tv_distance.py

**Purpose**: Evaluates the distance between a bed/sofa and the TV using a Gaussian reward function.

**Parameters**:
- `ideal` (float, default=3.0): Ideal viewing distance in meters
- `sigma` (float, default=1.0): Standard deviation for Gaussian falloff

**Returns**: 
- Value in [0, 1] where 1.0 means optimal distance
- Reward = exp(-((distance - ideal)² / (2 * sigma²)))

**Test Cases**:
- ✓ TV at ideal distance (3m) → ~1.0
- ✓ TV too close (1m) → ~0.14
- ✓ TV too far (6m+) → ~0.0
- ✓ No TV or bed → 0.0

**Usage**:
```python
from dynamic_reward_functions import reward_tv_distance

# Using default parameters (ideal=3.0m, sigma=1.0)
rewards = reward_tv_distance.get_reward(parsed_scene, idx_to_labels=idx_to_labels)

# Custom parameters
rewards = reward_tv_distance.get_reward(
    parsed_scene, 
    ideal=4.0,  # 4 meters ideal distance
    sigma=1.5,  # More forgiving falloff
    idx_to_labels=idx_to_labels
)
```

---

### 3. reward_tv_viewing_angle.py

**Purpose**: Evaluates how well a bed/sofa is oriented toward the TV.

**Returns**: 
- Value in [0, 1] based on cosine similarity
- `1.0` = furniture facing directly at TV
- `0.0` = furniture facing away or perpendicular

**Implementation Details**:
- Projects 3D positions to 2D (XZ plane) to match orientation format
- Uses cosine similarity between furniture orientation and direction to TV
- Clamps negative values to 0 (only rewards positive alignment)

**Test Cases**:
- ✓ Bed facing TV directly → 1.0
- ✓ Bed facing away from TV → 0.0
- ✓ Bed perpendicular to TV → ~0.0
- ✓ No TV or bed → 0.0

**Usage**:
```python
from dynamic_reward_functions import reward_tv_viewing_angle

rewards = reward_tv_viewing_angle.get_reward(parsed_scene, idx_to_labels=idx_to_labels)
```

**Bug Fix Applied**: 
Fixed dimension mismatch by projecting 3D direction vector to 2D (XZ plane) before comparing with 2D orientation.

---

## Running Tests

### Run All Tests

```bash
cd /path/to/steerable-scene-generation
python dynamic_constraint_rewards/test_all_dynamic_rewards.py
```

### Run Individual Function Tests

Each reward function has a `test_reward()` function that can be called:

```python
from dynamic_reward_functions import reward_tv_present

reward_tv_present.test_reward()
```

## Test Suite Structure

The comprehensive test suite (`test_all_dynamic_rewards.py`) includes:

1. **Data Type Verification**: Ensures parsed_scene matches expected types
2. **7 Test Scenarios**:
   - Perfect TV setup (optimal placement)
   - No TV present
   - TV too close to bed
   - TV too far from bed
   - Poor viewing angle
   - Empty room
   - Batch processing (multiple scenes)

3. **Validation**: Compares outputs against expected results

## Test Results Summary

```
reward_tv_present:        ✓ PASS (7/7 tests)
reward_tv_distance:       ✓ PASS (7/7 tests)
reward_tv_viewing_angle:  ✓ PASS (7/7 tests)
```

## Example Test Output

```
Testing: reward_tv_present
  Scene: perfect_tv_setup
  Description: Bed facing TV at ideal distance with sofa
  ✓ Reward: 1.0000
  ✓ Matches expected: 1.0

  Scene: no_tv
  Description: Room with bed and furniture but no TV
  ✓ Reward: 0.0000
  ✓ Matches expected: 0.0
```

## Creating New Reward Functions

When creating a new reward function:

1. Create a new file: `reward_<name>.py`
2. Implement `get_reward(parsed_scene, **kwargs)` function
3. Add `test_reward()` function
4. Add test cases to `test_all_dynamic_rewards.py`
5. Update this README

### Template:

```python
import torch

def get_reward(parsed_scene, **kwargs):
    """
    Description of what this reward evaluates.
    
    Args:
        parsed_scene: Dict with scene data
        **kwargs: Additional parameters
        
    Returns:
        torch.Tensor: Reward values of shape (B,)
    """
    device = parsed_scene["device"]
    object_indices = parsed_scene["object_indices"]
    # ... implementation ...
    
    rewards = torch.zeros(len(object_indices), device=device)
    # ... compute rewards ...
    
    return rewards


def test_reward():
    """Test function for the reward."""
    print("Testing reward_<name>...")
    # Add test logic here
    pass
```

## Troubleshooting

### Common Issues:

1. **Dimension Mismatch**: 
   - Orientations are 2D (XZ plane): `[cos(θ), sin(θ)]`
   - Positions are 3D: `[x, y, z]`
   - Project positions to 2D when comparing with orientations

2. **Empty Tensor Handling**:
   - Always check `is_empty` mask
   - Handle cases where no valid objects exist
   - Return zeros for empty batches

3. **Batch Processing**:
   - Ensure rewards are computed per scene in batch
   - Return tensor of shape `(B,)` not scalar

## Object Index Reference

Common bedroom objects:
- 8: double_bed
- 15: single_bed
- 16: sofa
- 19: tv_stand
- 12: nightstand
- 20: wardrobe
- 21: empty (placeholder)

See `create_idx_to_labels()` in test file for complete mapping.

## Contact

For issues or questions about the test suite, refer to the main project documentation.
