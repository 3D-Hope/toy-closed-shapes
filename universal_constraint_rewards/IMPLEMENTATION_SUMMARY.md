# Physical Constraint Rewards - Implementation Summary
- [x] gravity
- [x] non penetration
- [x] must have furnitures(eg bed in bedroom)
- [x] number of furnitures in scene
#### Ignored in baseline rl (harder because no floor conditioning used)
- [ ] night table (apparantly flux model we are using is doing great job at this so ignoring for now)
- [ ] chair orientation
- [ ] path clearance
- [ ] could also enforce distribution of objects in each batch similar to datasets
!Warning using the batch level rewards like kl div of count distribution: we need to distribute the same value to all samples so no difference between good vs bad sample. and same sample gets different reward in different batch.

Understanding Tanh Scale Parameter
The scale parameter in tanh(-reward / scale) is NOT a maximum. It's a sensitivity parameter that controls when the penalty becomes "severe":

Scale interpretation:

scale = 1.0 for gravity means: "When penalty reaches -1.0, it's considered severe (-0.76 normalized)"
scale = 5.0 for penetration means: "When penalty reaches -5.0, it's considered severe (-0.76 normalized)"
These are tunable thresholds, not maximums!


## Overview
This module implements a modular reward system for evaluating 3D scene generation quality based on physical constraints and functional requirements.

## Architecture

### Core Components

1. **`commons.py`** - Central utilities
   - `parse_and_descale_scenes()`: Converts normalized scene tensors to world coordinates
   - `descale_pos()`, `descale_size()`: Denormalization functions
   - `get_composite_reward()`: **Main entry point** - combines multiple rewards with weights
   - Constants: `idx_to_labels`, `ceiling_objects`

2. **Scene Input Format**
   - Shape: `(Batch, 12, 30)`
   - Per object: `[one_hot(22) | position(3) | size(3) | cos_theta(1) | sin_theta(1)]`
   - Empty slots use class index 21

## Implemented Reward Functions

### 1. Gravity Following Reward (`gravity_following_reward.py`)
**Purpose**: Ensures ground objects rest on the floor (y_min ≈ 0)

**Logic**:
- Ignores ceiling objects (ceiling_lamp, pendant_lamp)
- Ignores empty slots
- Calculates distance from floor for each object
- Reward = -average_distance

**Test Results**:
- Objects on ground: -0.3667 ✓
- Floating objects: -2.1034 ✓
- Ceiling lamp correctly ignored ✓

### 2. Object Count Reward (`object_count_reward.py`)
**Purpose**: Encourages realistic object counts (prevents empty/overcrowded scenes)

**Logic**:
- Target: 5 objects (mean from training data)
- Std dev: 1.43
- Reward = -(deviation/std_dev)²

**Test Results**:
- 5 objects (optimal): -0.0000 ✓
- 4-6 objects: -0.4890 ✓
- 0 objects (degenerate): -12.2255 ✓
- 12 objects (overcrowded): -23.9621 ✓

### 3. Must-Have Furniture Reward (`must_have_furniture_reward.py`)
**Purpose**: Enforces room-specific furniture requirements

**Logic** (bedroom):
- Requires at least one bed (single_bed, double_bed, or kids_bed)
- Has bed: reward = 0
- No bed: reward = -10

**Test Results**:
- Bedroom with single_bed: 0.0 ✓
- Bedroom with double_bed: 0.0 ✓
- Bedroom without bed: -10.0 ✓

### 4. Non-Penetration Reward (`non_penetration_reward.py`)
**Purpose**: Prevents object overlap (collision detection)

**Logic**:
- Uses Axis-Aligned Bounding Box (AABB) overlap in XZ plane
- Implements Separating Axis Theorem (SAT) for axis-aligned boxes
- Ignores ceiling objects
- Ignores empty slots
- Reward = -total_overlap_area

**Test Results**:
- No overlap: -0.0000 ✓
- Full overlap: -2.7596 ✓
- Ceiling lamp correctly ignored ✓
- Partial overlap: -1.7480 ✓

## Usage

### Basic Usage
```python
from physical_constraint_rewards.commons import get_composite_reward

# Simple usage (gravity only)
total_reward, components = get_composite_reward(scenes)

# Custom weights for bedroom
reward_weights = {
    'gravity': 1.0,
    'object_count': 0.5,
    'must_have_furniture': 2.0,
    'non_penetration': 1.5,
}

total_reward, components = get_composite_reward(
    scenes,
    num_classes=22,
    reward_weights=reward_weights,
    room_type='bedroom'
)
```

### Output
- `total_reward`: Tensor of shape `(B,)` - weighted sum of all rewards
- `components`: Dict with individual reward values for analysis

## Composite Reward Test Results

**Good Bedroom** (5 objects, on ground, no overlap, has bed):
- Total: -0.3667
- gravity: -0.3667, object_count: 0.0, must_have: 0.0, non_penetration: 0.0

**Missing Bed**:
- Total: -21.3448 (heavy penalty from must_have_furniture)

**Floating Furniture**:
- Total: -4.3040 (penalty from gravity and object_count)

**Overlapping Furniture**:
- Total: -10.7133 (heavy penalty from non_penetration)

**Empty Bedroom**:
- Total: -26.1128 (worst - penalties from must_have and object_count)

**Rankings**: Good > Floating > Overlapping > Missing bed > Empty ✓

## Design Principles

1. **Modularity**: Each reward is independent and testable
2. **Efficiency**: Parsing/descaling done once in composite function
3. **Extensibility**: Easy to add new rewards
4. **Degeneracy Prevention**: Object count reward prevents empty scenes
5. **Semantic Awareness**: Ceiling objects handled specially
6. **Batched Processing**: All operations support batch processing

## Testing

Each module includes comprehensive unit tests:
```bash
python physical_constraint_rewards/gravity_following_reward.py
python physical_constraint_rewards/object_count_reward.py
python physical_constraint_rewards/must_have_furniture_reward.py
python physical_constraint_rewards/non_penetration_reward.py
python physical_constraint_rewards/test_composite_reward.py
```

All tests passing ✓

## Future Extensions

To add a new reward:
1. Create `new_reward.py` with `compute_new_reward(parsed_scene, ...)`
2. Add test cases
3. Update `get_composite_reward()` in `commons.py`:
   ```python
   if 'new_reward' in reward_weights:
       from physical_constraint_rewards.new_reward import compute_new_reward
       reward_components['new_reward'] = compute_new_reward(parsed_scene)
   ```

## Performance Notes

- All operations are batched and parallelizable
- Device-agnostic (CPU/CUDA)
- No circular imports (local imports in composite function)
- Efficient AABB overlap computation using broadcasting
