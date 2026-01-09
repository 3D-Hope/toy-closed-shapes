# Complete LLM Pipeline System
## Step-by-Step Implementation for Agentic Scene Generation

---

# PHASE 0: System Initialization

## Step 0.1: Load System Context (One-Time Setup)

### Files to Prepare:

```python
system_context = {
    "dataset_info": "context/3dfront_dataset_info.json",
    "object_classes": "context/bedroom_classes.json",
    "scene_format": "context/scene_representation.md",
    "utility_functions": "context/utility_functions.py",
    "universal_rewards": "context/universal_rewards.py",
    "reward_template": "context/reward_function_template.py"
}
```

### Context Files Content:

**File 1: `3dfront_dataset_info.json`**
```json
{
  "dataset_name": "3D-FRONT",
  "description": "Large-scale dataset of 3D indoor scenes with furniture layouts",
  "statistics": {
    "total_scenes": 18797,
    "bedroom_scenes": 3127,
    "living_room_scenes": 2304,
    "dining_room_scenes": 1089
  },
  "common_patterns": {
    "bedroom": {
      "typical_furniture": ["bed", "wardrobe", "nightstand", "desk"],
      "bed_placement": "usually against wall",
      "nightstand_placement": "70% have 1 nightstand, 40% have 2",
      "ceiling_lamps": "0.3% have 2+ ceiling lamps, 0% have 4+",
      "chairs_in_bedroom": "2% of scenes have chairs",
      "multiple_beds": "1% have 2+ beds"
    }
  },
  "coordinate_system": {
    "y_axis": "vertical (up)",
    "xz_plane": "floor plane",
    "units": "meters",
    "origin": "room center typically"
  }
}
```

**File 2: `bedroom_classes.json`**
```json
{
  "bedroom": {
    "unique_values": {
      "0": "armchair",
      "1": "bookshelf",
      "2": "cabinet",
      "3": "ceiling_lamp",
      "4": "chair",
      "5": "children_cabinet",
      "6": "coffee_table",
      "7": "desk",
      "8": "double_bed",
      "9": "dressing_chair",
      "10": "dressing_table",
      "11": "kids_bed",
      "12": "nightstand",
      "13": "pendant_lamp",
      "14": "shelf",
      "15": "single_bed",
      "16": "sofa",
      "17": "stool",
      "18": "table",
      "19": "tv_stand",
      "20": "wardrobe"
    },
    "num_classes": 21,
    "room_type": "bedroom",
    "max_objects": 12,
    "typical_sizes": {
      "double_bed": {"x": 2.0, "y": 0.5, "z": 2.2},
      "nightstand": {"x": 0.5, "y": 0.5, "z": 0.5},
      "wardrobe": {"x": 1.0, "y": 2.0, "z": 0.6},
      "ceiling_lamp": {"x": 0.3, "y": 0.15, "z": 0.3}
    }
  }
}
```

**File 3: `scene_representation.md`**
```markdown
# Scene Representation Format

## Parsed Scene Structure
Scenes are provided as dictionaries with PyTorch tensors:

- `positions`: (B, N, 3) - Object centroids in meters (x, y, z)
- `sizes`: (B, N, 3) - Half-extents (sx/2, sy/2, sz/2)
- `object_indices`: (B, N) - Class indices [0, num_classes-1]
- `one_hot`: (B, N, num_classes) - One-hot encoded classes
- `is_empty`: (B, N) - Boolean mask (True = empty slot)
- `orientations`: (B, N, 2) - [cos(θ), sin(θ)] for z-rotation
- `device`: torch.device

Where:
- B = Batch size
- N = Max objects per scene (typically 12 for bedroom)

## Coordinate System
- Y-axis: Vertical (up direction)
- XZ-plane: Floor plane
- Units: Meters (world coordinates, unnormalized)
- Empty slots: Have index (num_classes-1), near-zero size/position

## Important Facts
- Ceiling objects are at y ≈ ceiling_height (typically 2.8m)
- Floor objects have y ≈ object_height/2
- Ignore empty slots (is_empty == True) in calculations
```

**File 4: `utility_functions.py`**
```python
"""
Available utility functions for reward computation.
Import and use these in your reward functions.
"""

import torch
import numpy as np

def get_object_count(parsed_scenes, class_label, idx_to_labels):
    """
    Count objects of a specific class.
    
    Args:
        parsed_scenes: Scene dictionary
        class_label: String label (e.g., "ceiling_lamp")
        idx_to_labels: Index to label mapping
    
    Returns:
        counts: (B,) tensor of object counts per scene
    """
    pass  # Implementation provided

def distance_2d(pos1, pos2):
    """
    Compute 2D distance (ignoring y) between positions.
    
    Args:
        pos1: (..., 3) positions
        pos2: (..., 3) positions
    
    Returns:
        distances: (...,) L2 distances in XZ plane
    """
    return torch.sqrt((pos1[..., 0] - pos2[..., 0])**2 + 
                      (pos1[..., 2] - pos2[..., 2])**2)

def distance_3d(pos1, pos2):
    """3D Euclidean distance."""
    return torch.norm(pos1 - pos2, dim=-1)

def angle_between_objects(orient1, orient2):
    """
    Compute angle between two object orientations.
    
    Args:
        orient1: (..., 2) [cos, sin]
        orient2: (..., 2) [cos, sin]
    
    Returns:
        angles: (...,) angles in degrees [0, 180]
    """
    pass

def get_front_center(position, size, orientation):
    """
    Get the center point of object's front face.
    
    Args:
        position: (3,) centroid
        size: (3,) half-extents
        orientation: (2,) [cos, sin]
    
    Returns:
        front_center: (3,) position
    """
    pass

def closest_wall_distance(position, floor_vertices):
    """
    Distance from position to closest wall.
    
    Args:
        position: (2,) or (3,) position (uses XZ)
        floor_vertices: List of (x, z) vertices
    
    Returns:
        distance: float
        wall_segment: ((x1,z1), (x2,z2))
    """
    pass

def point_in_polygon(point, vertices):
    """Check if 2D point is inside polygon."""
    pass

def is_point_in_floor(position, floor_vertices):
    """Check if XZ position is within floor bounds."""
    pass

def get_room_center(floor_vertices):
    """Get centroid of floor polygon."""
    return np.mean(floor_vertices, axis=0)

def get_room_bounds(floor_vertices):
    """Get bounding box of floor."""
    vertices = np.array(floor_vertices)
    return {
        'min_x': vertices[:, 0].min(),
        'max_x': vertices[:, 0].max(),
        'min_z': vertices[:, 1].min(),
        'max_z': vertices[:, 1].max()
    }
```

**File 5: `universal_rewards.py`**
```python
"""
Universal constraint rewards that apply to all scenes.
These are already implemented and weighted in the system.
"""

# Already implemented universal rewards:
UNIVERSAL_REWARDS = {
    "must_have_furniture": {
        "weight": 1.0,
        "description": "Essential furniture present (e.g., bed in bedroom)"
    },
    "non_penetration": {
        "weight": 1.0,
        "description": "No object-object collisions"
    },
    "object_count": {
        "weight": 0.5,
        "description": "Appropriate number of objects"
    },
    "not_out_of_bound": {
        "weight": 1.0,
        "description": "Objects within room boundaries"
    },
    "accessibility": {
        "weight": 0.7,
        "description": "Objects reachable, not blocked"
    },
    "gravity_following": {
        "weight": 1.0,
        "description": "Objects on floor/ceiling, not floating"
    },
    "night_tables_on_head_side": {
        "weight": 0.6,
        "description": "Nightstands at bed head"
    },
    "axis_alignment": {
        "weight": 0.5,
        "description": "Objects aligned with room axes"
    },
    "furniture_against_wall": {
        "weight": 0.6,
        "description": "Large furniture near walls"
    }
}

# Note: Dynamic rewards will conflict with these.
# Example: Generating 4 ceiling lamps may increase collision risk.
# Adjust weights in your constraint specification accordingly.
```

**File 6: `reward_function_template.py`**
```python
"""
Template for writing reward functions.
COPY THIS TEMPLATE when generating reward code.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from utility_functions import (
    get_object_count,
    distance_2d,
    distance_3d,
    angle_between_objects,
    closest_wall_distance,
    is_point_in_floor,
    get_room_center
)

def reward_CONSTRAINT_NAME(
    parsed_scenes: Dict[str, torch.Tensor],
    idx_to_labels: Dict[int, str],
    room_type: str,
    max_objects: int,
    num_classes: int,
    floor_polygons: List[torch.Tensor],
    indices: List[int],
    is_val: bool,
    sdf_cache: Any,
    floor_plan_args: Dict[str, List],
    **kwargs
) -> List[float]:
    """
    Reward function for CONSTRAINT_NAME.
    
    Description: [Describe what this constraint checks]
    
    Args:
        parsed_scenes: Scene data with positions, sizes, orientations, etc.
        idx_to_labels: Mapping from object index to class label
        room_type: Type of room (e.g., "bedroom")
        max_objects: Maximum objects per scene
        num_classes: Number of object classes
        floor_polygons: Floor boundary vertices per scene
        indices: Scene indices in batch
        is_val: Validation flag
        sdf_cache: Signed distance field cache
        floor_plan_args: Floor plan geometry data
        **kwargs: Additional arguments
    
    Returns:
        rewards: List of float rewards (one per scene in batch)
                 Higher = better constraint satisfaction
                 Can be any scale (will be normalized to [-1, 1])
    """
    
    batch_size = parsed_scenes['positions'].shape[0]
    device = parsed_scenes['device']
    rewards = []
    
    for b in range(batch_size):
        # Extract scene data
        positions = parsed_scenes['positions'][b]  # (N, 3)
        sizes = parsed_scenes['sizes'][b]  # (N, 3)
        orientations = parsed_scenes['orientations'][b]  # (N, 2)
        object_indices = parsed_scenes['object_indices'][b]  # (N,)
        is_empty = parsed_scenes['is_empty'][b]  # (N,)
        one_hot = parsed_scenes['one_hot'][b]  # (N, num_classes)
        
        floor_vertices = floor_polygons[b].cpu().numpy()  # (M, 2)
        
        # Filter out empty slots
        valid_mask = ~is_empty
        valid_positions = positions[valid_mask]
        valid_sizes = sizes[valid_mask]
        valid_orientations = orientations[valid_mask]
        valid_indices = object_indices[valid_mask]
        
        # TODO: Implement constraint logic here
        
        # Example structure:
        # 1. Count specific objects
        # 2. Check spatial relationships
        # 3. Compute geometric properties
        # 4. Calculate reward score
        
        reward = 0.0  # Replace with actual computation
        
        rewards.append(float(reward))
    
    return rewards
```

---

# PHASE 1: Initial Constraint Decomposition

## Step 1.1: Prepare LLM Prompt for Constraint Analysis

### Prompt Template:

```markdown
# TASK: Constraint Decomposition for 3D Scene Generation

You are an expert in 3D scene generation, interior design, and reinforcement learning. Your task is to analyze a user prompt and decompose it into verifiable constraints with Python reward functions.

## USER PROMPT
"{user_prompt}"

## CONTEXT

### Dataset: 3D-FRONT
{paste content from 3dfront_dataset_info.json}

### Available Object Classes (Bedroom)
{paste content from bedroom_classes.json}

### Scene Representation
{paste content from scene_representation.md}

### Available Utility Functions
{paste content from utility_functions.py}

### Universal Rewards (Already Implemented)
{paste content from universal_rewards.py}

**IMPORTANT:** Your dynamic reward functions may conflict with universal rewards. For example:
- Generating 4 ceiling lamps (unusual) may increase collision risk
- Placing chairs in corners may conflict with accessibility
- Consider these conflicts when designing reward weights

## YOUR TASK

Analyze the user prompt and provide a comprehensive JSON response with the following structure:

### 1. CONSTRAINT DECOMPOSITION

List ALL constraints needed to satisfy the prompt. For each constraint:

```json
{
  "constraints": [
    {
      "id": "C1",
      "name": "descriptive_snake_case_name",
      "description": "Clear description of what this checks",
      "type": "generation|spatial|orientation|functional|aesthetic",
      "priority": 1,  // 1=highest (critical blocker)
      "in_distribution": true|false,  // Based on dataset statistics
      "depends_on": ["C0"],  // List of prerequisite constraint IDs
      "reasoning": "Why this constraint is necessary and its difficulty"
    }
  ]
}
```

### 2. REWARD FUNCTION CODE

For EACH constraint, write a complete Python reward function following the template provided.

**Requirements:**
- Use the exact function signature from the template
- Import necessary utilities from utility_functions
- Return List[float] with one reward per scene
- Rewards can be any scale (will be normalized)
- Higher values = better satisfaction
- Add docstring explaining logic
- Handle edge cases (no objects, empty slots, etc.)

**Code Format:**
```python
def reward_C1_constraint_name(...):
    """Docstring here"""
    # Implementation
    return rewards
```

### 3. SUCCESS THRESHOLDS

After normalization to [-1, 1], what threshold indicates success?

```json
{
  "thresholds": {
    "C1_constraint_name": 0.8,  // Strict
    "C2_another_constraint": 0.6,  // Moderate
    "C3_soft_constraint": 0.4  // Lenient
  },
  "threshold_reasoning": {
    "C1_constraint_name": "Must be nearly perfect - geometric precision required",
    "C2_another_constraint": "Allow some tolerance for aesthetic constraints"
  }
}
```

### 4. REWARD WEIGHTS

Provide weights for ALL rewards (universal + dynamic):

```json
{
  "reward_weights": {
    // Universal (copy from context, adjust if needed)
    "must_have_furniture": 1.0,
    "non_penetration": 1.0,
    "not_out_of_bound": 1.0,
    "gravity_following": 1.0,
    "accessibility": 0.7,
    "object_count": 0.5,
    "night_tables_on_head_side": 0.6,
    "axis_alignment": 0.5,
    "furniture_against_wall": 0.6,
    
    // Your dynamic constraints
    "C1_constraint_name": 0.8,
    "C2_another_constraint": 0.6,
    // ...
  },
  "weight_reasoning": "Explanation of weight choices and potential conflicts"
}
```

### 5. HARD-CODING RECOMMENDATION

Should we hard-code any objects? (Use sparingly!)

```json
{
  "hardcode_objects": true|false,
  "reasoning": "Why hard-coding is necessary (if at all)",
  "specification": {
    "num_objects": 4,
    "object_class": "ceiling_lamp",
    "positions": "random_ceiling"|"specific",  // If specific, provide coords
    "sizes": "default"|[x, y, z],
    "orientations": "default"|[cos, sin],
    "note": "Physics and realism will be handled by generator"
  }
}
```

### 6. PRELIMINARY CURRICULUM SKETCH

Based on constraint dependencies and expected dataset statistics, sketch a rough curriculum:

```json
{
  "preliminary_curriculum": [
    {
      "subgoal_id": 1,
      "name": "Short description",
      "target_constraints": ["C1"],
      "expected_difficulty": "easy|medium|hard|extreme",
      "reasoning": "Why this is first/this difficulty",
      "estimated_steps": 1000
    }
  ],
  "note": "This will be refined after baseline evaluation"
}
```

## OUTPUT FORMAT

Provide your response as valid JSON with the following top-level keys:
- `constraints`: List of constraint objects
- `reward_functions_code`: String containing all Python code
- `thresholds`: Dictionary of thresholds
- `reward_weights`: Dictionary of weights
- `hardcode_objects`: Hard-coding specification
- `preliminary_curriculum`: Rough curriculum sketch

**CRITICAL:** Ensure all JSON is valid and all Python code is syntactically correct with proper imports.
```

---

## Step 1.2: Call LLM with Prompt

```python
import json
import openai  # or anthropic for Claude

def call_llm_constraint_decomposition(user_prompt: str) -> dict:
    """
    Call LLM for initial constraint decomposition.
    """
    
    # Load context files
    context = load_system_context()
    
    # Build prompt
    prompt = build_constraint_decomposition_prompt(
        user_prompt=user_prompt,
        context=context
    )
    
    # Call LLM
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",  # or "claude-3-opus"
        messages=[
            {"role": "system", "content": "You are an expert in 3D scene generation and constraint reasoning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Low for consistency
        max_tokens=8000
    )
    
    # Parse response
    llm_output = response.choices[0].message.content
    
    # Extract JSON (handle markdown code blocks if present)
    llm_json = extract_json_from_response(llm_output)
    
    # Validate structure
    validate_llm_response(llm_json)
    
    return llm_json
```

---

## Step 1.3: Validate and Parse LLM Output

```python
def validate_llm_response(llm_json: dict):
    """
    Validate LLM output structure and content.
    """
    
    required_keys = [
        "constraints",
        "reward_functions_code",
        "thresholds",
        "reward_weights",
        "hardcode_objects",
        "preliminary_curriculum"
    ]
    
    # Check keys exist
    for key in required_keys:
        assert key in llm_json, f"Missing key: {key}"
    
    # Validate constraints
    for c in llm_json["constraints"]:
        assert "id" in c and "name" in c, "Invalid constraint format"
        assert c["priority"] > 0, "Priority must be positive"
    
    # Validate Python code syntax
    try:
        compile(llm_json["reward_functions_code"], "<string>", "exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid Python code: {e}")
    
    # Check threshold range
    for thresh in llm_json["thresholds"].values():
        assert -1 <= thresh <= 1, "Threshold must be in [-1, 1]"
    
    # Check weights are positive
    for weight in llm_json["reward_weights"].values():
        assert weight >= 0, "Weights must be non-negative"
    
    print("✓ LLM response validated successfully")
```

---

## Step 1.4: Execute Reward Functions (Test)

```python
def test_reward_functions(llm_json: dict, test_scenes: dict):
    """
    Test generated reward functions on sample scenes.
    """
    
    # Execute code to define functions
    exec(llm_json["reward_functions_code"], globals())
    
    # Extract function names
    constraint_ids = [c["id"] for c in llm_json["constraints"]]
    constraint_names = [c["name"] for c in llm_json["constraints"]]
    
    # Test each function
    for cid, cname in zip(constraint_ids, constraint_names):
        func_name = f"reward_{cid}_{cname}"
        
        try:
            func = globals()[func_name]
            rewards = func(
                test_scenes["parsed_scenes"],
                test_scenes["idx_to_labels"],
                test_scenes["room_type"],
                test_scenes["max_objects"],
                test_scenes["num_classes"],
                test_scenes["floor_polygons"],
                test_scenes["indices"],
                is_val=True,
                sdf_cache=test_scenes["sdf_cache"],
                floor_plan_args=test_scenes["floor_plan_args"]
            )
            
            assert len(rewards) == test_scenes["batch_size"], "Wrong number of rewards"
            assert all(isinstance(r, (int, float)) for r in rewards), "Invalid reward types"
            
            print(f"✓ {func_name} passed test")
            
        except Exception as e:
            print(f"✗ {func_name} failed: {e}")
            raise
```

---

# PHASE 2: Baseline Evaluation

## Step 2.1: Run Baseline Statistics

```python
def run_baseline_evaluation(llm_json: dict, config: dict) -> dict:
    """
    Run reward functions on baseline model and dataset.
    """
    
    print("=" * 60)
    print("PHASE 2: BASELINE EVALUATION")
    print("=" * 60)
    
    # Import the reward functions
    exec(llm_json["reward_functions_code"], globals())
    
    # Prepare reward functions dict
    reward_functions = {}
    for constraint in llm_json["constraints"]:
        cid = constraint["id"]
        cname = constraint["name"]
        func_name = f"reward_{cid}_{cname}"
        threshold = llm_json["thresholds"][f"{cid}_{cname}"]
        
        reward_functions[f"{cid}_{cname}"] = (globals()[func_name], threshold)
    
    # Call the baseline stats function (from your existing code)
    stats = get_reward_stats_from_baseline(
        reward_functions=reward_functions,
        load=config["model_checkpoint"],
        dataset="custom_scene",
        config=config,
        num_scenes=1000,
        # ... other config params
    )
    
    # Also run on dataset
    stats_dataset = get_reward_stats_from_dataset(
        reward_functions=reward_functions,
        config=config,
        num_scenes=1000
    )
    
    # Combine stats
    combined_stats = {
        "baseline_model": stats,
        "dataset": stats_dataset
    }
    
    # Save stats
    save_json(combined_stats, "baseline_statistics.json")
    
    print("\n✓ Baseline evaluation complete")
    print(f"Results saved to: baseline_statistics.json")
    
    return combined_stats
```

---

## Step 2.2: Format Statistics for LLM

```python
def format_stats_for_llm(combined_stats: dict) -> str:
    """
    Format statistics into LLM-readable text.
    """
    
    formatted = "# BASELINE EVALUATION RESULTS\n\n"
    
    for constraint_name, stats in combined_stats["baseline_model"].items():
        formatted += f"## Constraint: {constraint_name}\n\n"
        
        formatted += "### Baseline Model Performance\n"
        formatted += f"- Success Rate: {stats['success_metrics']['success_rate']:.1%}\n"
        formatted += f"- Mean Reward: {stats['basic_stats']['mean']:.4f}\n"
        formatted += f"- Max Reward: {stats['basic_stats']['max']:.4f}\n"
        formatted += f"- Distribution Shape: {stats['distribution']['shape']}\n"
        formatted += f"- Interpretation: {stats['distribution']['interpretation']}\n\n"
        
        formatted += "### Dataset Performance\n"
        dataset_stats = combined_stats["dataset"][constraint_name]
        formatted += f"- Success Rate: {dataset_stats['success_metrics']['success_rate']:.1%}\n"
        formatted += f"- Mean Reward: {dataset_stats['basic_stats']['mean']:.4f}\n"
        formatted += f"- Max Reward: {dataset_stats['basic_stats']['max']:.4f}\n\n"
        
        formatted += "### Learnability Assessment\n"
        formatted += f"- Difficulty: {stats['learnability']['difficulty'].upper()}\n"
        formatted += f"- Recommended Approach: {stats['learnability']['recommended_approach']}\n"
        formatted += f"- Estimated Steps: {stats['learnability']['estimated_training_steps']}\n"
        formatted += f"- Has Positive Examples: {stats['learnability']['has_positive_examples']}\n"
        formatted += f"- Positive Example Count: {stats['learnability']['num_positive_examples']}\n\n"
        
        formatted += "### Model vs Dataset Gap\n"
        model_mean = stats['basic_stats']['mean']
        dataset_mean = dataset_stats['basic_stats']['mean']
        gap = model_mean - dataset_mean
        formatted += f"- Gap: {gap:+.4f} (model - dataset)\n"
        formatted += f"- Analysis: "
        if abs(gap) < 0.05:
            formatted += "Similar performance\n"
        elif gap < 0:
            formatted += "Model underperforming dataset - learnable with gradient\n"
        else:
            formatted += "Model outperforming dataset - may have learned something\n"
        
        formatted += "\n" + "-" * 60 + "\n\n"
    
    return formatted
```

---

# PHASE 3: Curriculum Design

## Step 3.1: Prompt LLM for Detailed Curriculum

### Prompt Template:

```markdown
# TASK: Design Detailed Training Curriculum

You have previously decomposed the user prompt into constraints and reward functions. Now, based on ACTUAL baseline evaluation results, design a detailed training curriculum.

## USER PROMPT
"{user_prompt}"

## YOUR PREVIOUS OUTPUT

### Constraints
{paste constraints from Phase 1}

### Reward Functions
{list of function names and descriptions}

### Preliminary Curriculum
{paste preliminary_curriculum from Phase 1}

## BASELINE EVALUATION RESULTS

{paste formatted stats from Step 2.2}

## YOUR TASK

Design a detailed, step-by-step training curriculum. Each subgoal should:
1. Build on previous subgoals (progressive difficulty)
2. Have achievable success criteria based on baseline stats
3. Include a complete Python reward function
4. Specify training hyperparameters

### Output Format:

```json
{
  "curriculum_analysis": {
    "primary_bottleneck": "C1_constraint_name",
    "reasoning": "Based on stats, this constraint has 0% baseline success and blocks all others",
    "training_strategy": "Start with easiest achievable subgoal, progress to hardest"
  },
  
  "curriculum": [
    {
      "subgoal_id": 1,
      "name": "Descriptive name",
      "description": "Detailed description of what this subgoal achieves",
      
      "target_constraints": ["C1_relaxed"],
      "depends_on": [],  // Subgoal IDs this depends on
      
      "difficulty": "EASY|MEDIUM|HARD|EXTREME",
      "reasoning": "Why this difficulty based on baseline stats showing X% success",
      
      "success_criteria": {
        "success_rate_target": 0.7,
        "mean_reward_target": 0.65,
        "min_training_steps": 1000,
        "max_training_steps": 3000,
        "early_stop_condition": "success_rate > 0.75 for 2 consecutive evals"
      },
      
      "reward_function_code": "def reward_subgoal_1(...):\n    # Complete implementation\n    return rewards",
      
      "reward_composition": {
        "subgoal_1_reward": 0.7,
        "previous_subgoal_reward": 0.0,  // No previous for first subgoal
        "universal_collision": 0.3,
        "universal_gravity": 0.0,  // Included in default
        "note": "Reduced collision weight to allow exploration"
      },
      
      "training_config": {
        "learning_rate": 3e-4,
        "kl_penalty_weight": 0.01,
        "entropy_bonus": 0.05,
        "clip_range": 0.2,
        "note": "Higher entropy for exploration in early subgoals"
      },
      
      "expected_issues": [
        "Mode collapse: Model may refuse to generate multiple objects",
        "Collision increase: More objects = more collisions"
      ],
      
      "mitigation_strategies": [
        "Use higher temperature sampling (1.2)",
        "Add exploration bonus for diversity",
        "Monitor collision rate, may need to adjust weights"
      ],
      
      "evaluation_frequency": 500,  // Steps between evaluations
      "num_eval_scenes": 100
    },
    
    // ... more subgoals
  ],
  
  "total_estimated_steps": "8000-12000",
  "estimated_wall_clock_time": "4-6 hours on single A100",
  
  "adaptive_strategy": {
    "if_subgoal_fails": "Increase training steps by 50%, reduce success threshold by 0.1",
    "if_regression_occurs": "Add stabilization rewards from previous subgoals with higher weight",
    "if_plateau": "Adjust learning rate or add auxiliary rewards"
  }
}
```

## IMPORTANT NOTES

1. **Progressive Difficulty:** First subgoal should have baseline success > 0%