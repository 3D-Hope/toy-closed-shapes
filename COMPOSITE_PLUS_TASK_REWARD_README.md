# Composite + Task Reward System

## Overview

The composite + task reward system allows you to combine **general scene quality constraints** with **task-specific requirements** in a single reward function for RL training.

## Architecture

### Two-Level Reward Structure

```
Total Reward = Composite Reward + (task_weight × Task Reward)
```

#### 1. Composite Reward (General Scene Quality)
Combines 4 physics-based constraints, each normalized to [-1, 0]:

- **Gravity Following**: Objects should rest on the ground (not float)
- **Non-Penetration**: No overlapping objects
- **Must-Have Furniture**: Room should have appropriate base furniture
- **Object Count**: Realistic furniture density (not too sparse/cluttered)

Each component is:
1. Computed independently
2. Normalized to [-1, 0] range (0 = perfect, -1 = worst)
3. Weighted by importance multiplier

**Formula**:
```python
composite_reward = sum(normalized_reward_i * importance_weight_i)
```

#### 2. Task Reward (Specific Goal)
Binary or continuous reward for task completion:

- **has_sofa**: 1.0 if sofa present, 0.0 otherwise
- **has_table**: 1.0 if table present, 0.0 otherwise
- **two_beds**: Score based on bed count probability

## Configuration

### Option 1: Composite Reward Only
```yaml
algorithm.ddpo.use_composite_reward=True
algorithm.ddpo.composite_reward.room_type=bedroom
algorithm.ddpo.composite_reward.importance_weights.must_have_furniture=1.5
algorithm.ddpo.composite_reward.importance_weights.gravity=1.0
algorithm.ddpo.composite_reward.importance_weights.non_penetration=1.0
algorithm.ddpo.composite_reward.importance_weights.object_count=0.7
```

### Option 2: Composite + Task Reward
```yaml
algorithm.ddpo.use_composite_plus_task_reward=True
algorithm.ddpo.composite_plus_task.task_reward_type=has_sofa
algorithm.ddpo.composite_plus_task.task_weight=2.0
algorithm.ddpo.composite_plus_task.room_type=living_room
algorithm.ddpo.composite_plus_task.importance_weights.must_have_furniture=1.0
algorithm.ddpo.composite_plus_task.importance_weights.gravity=1.0
algorithm.ddpo.composite_plus_task.importance_weights.non_penetration=1.0
algorithm.ddpo.composite_plus_task.importance_weights.object_count=0.5
```

## Parameters

### `task_reward_type`
Specifies which task-specific reward to use:
- `'has_sofa'`: Encourage presence of sofa (living room)
- `'has_table'`: Encourage presence of table
- `'two_beds'`: Encourage two beds (guest bedroom)

### `task_weight`
How much to emphasize task completion vs general quality:
- **High (2.0-3.0)**: Strong focus on task, physics constraints secondary
- **Medium (1.0)**: Balanced between task and physics
- **Low (0.3-0.7)**: Mild preference for task, physics dominant

### `room_type`
Affects must-have furniture component:
- `'bedroom'`: Expects bed, nightstand, etc.
- `'living_room'`: Expects sofa, coffee table, etc.

### `importance_weights`
Fine-tune relative importance of each physics constraint:
- **1.0**: Baseline importance
- **>1.0**: More important (e.g., 1.5 = 50% more important)
- **<1.0**: Less important (e.g., 0.5 = 50% less important)

## Example Scenarios

### Scenario 1: Living Room Must Have Sofa
**Goal**: Generate living rooms with realistic physics AND a sofa.

**Config**:
```yaml
task_reward_type: has_sofa
task_weight: 2.0  # Strong task emphasis
room_type: living_room
importance_weights:
  must_have_furniture: 1.0  # Lower (task handles furniture requirement)
  gravity: 1.0
  non_penetration: 1.0
  object_count: 0.5
```

**Expected Behavior**:
- Model learns to place sofas in scenes (task_weight=2.0 provides strong signal)
- Still respects physics (no floating/overlapping furniture)
- Moderate furniture density (object_count weight is lower)

### Scenario 2: Bedroom with Two Beds (Guest Room)
**Goal**: Generate bedrooms with exactly two beds.

**Config**:
```yaml
task_reward_type: two_beds
task_weight: 3.0  # Very strong task emphasis
room_type: bedroom
importance_weights:
  must_have_furniture: 0.5  # Lower (task handles beds)
  gravity: 1.0
  non_penetration: 1.0
  object_count: 0.5
```

**Expected Behavior**:
- Model strongly encouraged to generate two beds
- Physics constraints still enforced
- Less strict about other bedroom furniture

### Scenario 3: General Bedroom Quality (No Task)
**Goal**: Just good physics, no specific requirements.

**Config**:
```yaml
use_composite_reward: True  # Not composite_plus_task
room_type: bedroom
importance_weights:
  must_have_furniture: 1.5  # Higher (main furniture guidance)
  gravity: 1.0
  non_penetration: 1.0
  object_count: 0.7
```

**Expected Behavior**:
- Focus on general bedroom quality
- Appropriate furniture (bed, nightstand)
- Good physics
- Realistic density

## Normalization Strategy

### Why Normalize?
Raw rewards have different scales:
- Gravity: -0.01 to -10.0 (unbounded)
- Non-penetration: -0.01 to -20.0 (unbounded)
- Must-have: -10.0 to 0.0 (bounded)
- Object count: -9.2 to -1.2 (bounded, log-probability)

Without normalization, importance weights would be hard to interpret.

### How It Works

#### Unbounded Rewards (Tanh)
```python
normalized = -tanh(-reward / scale)
```
- Smoothly maps to [-1, 0]
- `scale` parameter sets "severe violation" threshold
- Example: gravity scale=1.0 → 1m floating ≈ -0.76 normalized

#### Bounded Rewards (Linear)
```python
normalized = reward / max_penalty
```
- Direct mapping to [-1, 0]
- Uses true maximum penalty

#### Result
All rewards in [-1, 0] where:
- **0.0** = Perfect (no violations)
- **-1.0** = Worst case

## Logging

The system logs all reward components for analysis:

```python
reward_components/{name}_mean
```

Available components:
- `gravity_mean`: Average gravity violation
- `non_penetration_mean`: Average penetration penalty
- `must_have_furniture_mean`: Furniture requirement score
- `object_count_mean`: Density score
- `task_reward_mean`: Task completion score (0 or 1)
- `composite_reward_mean`: Sum of physics constraints
- `total_reward_mean`: Final combined reward

## Best Practices

### 1. Start with Composite Only
First train with just `use_composite_reward=True` to establish good physics baseline.

### 2. Add Task Gradually
Introduce task reward with low weight, then increase:
```yaml
# Phase 1: Physics only
use_composite_reward: True

# Phase 2: Add task
use_composite_plus_task_reward: True
task_weight: 0.5  # Start low

# Phase 3: Increase task emphasis
task_weight: 2.0  # Once physics is stable
```

### 3. Monitor Components
Watch individual component logs to diagnose issues:
- If `gravity_mean` stays negative → increase `importance_weights.gravity`
- If `task_reward_mean` stays at 0 → increase `task_weight`

### 4. Balance Task vs Physics
**Rule of thumb**:
```
task_weight ≈ sum(importance_weights)
```
This makes task and physics roughly equal in magnitude.

## Command Line Usage

```bash
# Full example with all options
PYTHONPATH=. python main.py \
    +name=living_room_sofa_experiment \
    load=checkpoint.ckpt \
    algorithm=scene_diffuser_flux_transformer \
    algorithm.trainer=rl_score \
    algorithm.ddpo.use_composite_plus_task_reward=True \
    algorithm.ddpo.composite_plus_task.task_reward_type=has_sofa \
    algorithm.ddpo.composite_plus_task.task_weight=2.0 \
    algorithm.ddpo.composite_plus_task.room_type=living_room \
    algorithm.ddpo.composite_plus_task.importance_weights.must_have_furniture=1.0 \
    algorithm.ddpo.composite_plus_task.importance_weights.gravity=1.0 \
    algorithm.ddpo.composite_plus_task.importance_weights.non_penetration=1.0 \
    algorithm.ddpo.composite_plus_task.importance_weights.object_count=0.5 \
    algorithm.ddpo.batch_size=256 \
    algorithm.ddpo.ddpm_reg_weight=200.0 \
    algorithm.custom.loss=true
```

## Testing

Run the integration tests:
```bash
PYTHONPATH=. python test_composite_reward_integration.py
```

This tests both:
1. Composite reward alone
2. Composite + task reward combination

## Technical Details

### Files Modified
- `configurations/algorithm/scene_diffuser_base_continous.yaml`: Config
- `steerable_scene_generation/algorithms/scene_diffusion/ddpo_helpers.py`: Reward functions
- `steerable_scene_generation/algorithms/scene_diffusion/trainer_rl.py`: Integration
- `physical_constraint_rewards/commons.py`: Core reward computation

### Key Functions
- `get_composite_reward()`: Computes normalized physics constraints
- `composite_reward()`: Wrapper for DDPO integration
- `composite_plus_task_reward()`: Combines physics + task rewards
- Individual task rewards: `has_sofa_reward()`, `two_beds_reward()`, etc.
