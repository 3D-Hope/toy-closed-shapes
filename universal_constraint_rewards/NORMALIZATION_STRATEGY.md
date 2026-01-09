# Normalization Strategy Analysis

## Problem: Unbounded Rewards

You correctly identified that my "empirical maximums" are NOT true bounds:

1. **Gravity**: Can be arbitrarily large if objects float very high
   - My observation: -2.7 (floating 2m)
   - Reality: Could be -100 (floating 10m), -10000 (floating 100m)
   - Formula: `-distance²` → unbounded

2. **Non-penetration**: Can be arbitrarily large with many/severe overlaps
   - My observation: -25 (some overlaps in test)
   - Reality: Could be -1000 (all objects overlapping severely)
   - Formula: sum of all penetration volumes → unbounded

3. **Must-have furniture**: Bounded
   - True max: -10 (missing bed)
   - This one is actually bounded ✓

4. **Object count (NLL)**: Bounded
   - True max: -9.2 (1 object, very rare)
   - This one is actually bounded ✓

---

## Solution Options

### Option 1: Soft Normalization (Tanh/Sigmoid)
Map unbounded rewards to [-1, 0] using smooth functions:

```python
def soft_normalize(reward, scale):
    """
    Soft normalization using tanh.
    - reward=0 → 0
    - reward=-scale → -0.76
    - reward=-2*scale → -0.96
    - reward=-∞ → -1.0
    """
    return -torch.tanh(-reward / scale)
```

**Pros:**
- Handles unbounded rewards
- Smooth gradients everywhere
- No clipping artifacts

**Cons:**
- Needs to choose `scale` parameter
- Diminishing returns for large violations (might be good!)

---

### Option 2: Clipped Normalization
Set a "reasonable maximum" and clip:

```python
def clip_normalize(reward, max_penalty):
    """
    Clip to reasonable range then normalize.
    """
    clipped = torch.clamp(reward, min=-max_penalty, max=0)
    return clipped / max_penalty
```

**Pros:**
- Simple, interpretable
- Can set "practical maximum" based on domain knowledge

**Cons:**
- Loses information beyond clip point
- Gradient becomes zero at extremes

---

### Option 3: Percentile-Based Normalization
Use training data statistics (median, 95th percentile):

```python
def percentile_normalize(reward, p50, p95):
    """
    Normalize based on empirical distribution.
    - p50 (median violation) → -0.5
    - p95 (severe violation) → -0.95
    """
    # Use robust scaling
    scale = p95 - p50
    return (reward - p50) / scale
```

**Pros:**
- Data-driven, adaptive
- Captures real distribution

**Cons:**
- Requires statistics from training data
- Can change over time as model improves

---

### Option 4: Logarithmic Normalization
For quadratic/exponential penalties:

```python
def log_normalize(reward, scale):
    """
    Log normalization for penalties that grow quadratically.
    Useful for gravity (distance²) and penetration (volume).
    """
    return -torch.log1p(-reward / scale) / torch.log1p(10)
```

**Pros:**
- Good for quadratic/volume penalties
- Compresses large violations while preserving small differences

**Cons:**
- Less intuitive
- Needs scale parameter

---

## Recommended: Hybrid Approach

Different normalization for different reward types:

### 1. Bounded Rewards (Must-have, Object count)
Use simple linear normalization:
```python
normalized = reward / max_penalty
```

### 2. Unbounded Quadratic (Gravity)
Use soft normalization with tanh:
```python
# Gravity penalty = -distance²
# Choose scale where -1.0² = "significant violation"
GRAVITY_SCALE = 1.0  # 1m floating → -0.76 penalty
normalized = -torch.tanh(-reward / GRAVITY_SCALE)
```

### 3. Unbounded Accumulative (Non-penetration)
Use soft normalization with larger scale:
```python
# Penetration is sum of volumes
# Choose scale where "typical severe overlap" → -0.8
NON_PENETRATION_SCALE = 10.0
normalized = -torch.tanh(-reward / NON_PENETRATION_SCALE)
```

---

## Implementation

```python
def normalize_reward(reward, reward_type):
    """
    Normalize reward to approximately [-1, 0] range.
    
    Different strategies for different reward characteristics:
    - Bounded: Linear normalization
    - Unbounded quadratic: Tanh normalization
    - Unbounded accumulative: Tanh with larger scale
    """
    
    if reward_type == 'gravity':
        # Quadratic penalty (distance²)
        # Scale: 1.0 means 1m floating → -0.76 normalized
        scale = 1.0
        return -torch.tanh(-reward / scale)
    
    elif reward_type == 'non_penetration':
        # Accumulative penalty (sum of volumes)
        # Scale: 10.0 means moderate overlaps → -0.76 normalized
        scale = 10.0
        return -torch.tanh(-reward / scale)
    
    elif reward_type == 'must_have_furniture':
        # Bounded: -10 max
        return reward / 10.0
    
    elif reward_type == 'object_count':
        # Bounded: -9.2 to -1.2
        # Shift to [0, -8] then normalize
        shifted = reward + 1.2  # Now in [-8, 0]
        return shifted / 8.0
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
```

---

## Tanh Normalization Scale Guide

For `normalized = -tanh(-reward / scale)`:

| Raw Penalty | Normalized (scale=1.0) | Normalized (scale=10.0) |
|-------------|------------------------|-------------------------|
| 0 | 0.00 | 0.00 |
| -0.5 | -0.46 | -0.05 |
| -1.0 | -0.76 | -0.10 |
| -2.0 | -0.96 | -0.20 |
| -5.0 | -0.9999 | -0.46 |
| -10.0 | -1.00 | -0.76 |
| -20.0 | -1.00 | -0.96 |
| -50.0 | -1.00 | -0.9999 |
| -100.0 | -1.00 | -1.00 |

**Interpretation:**
- **scale=1.0**: Penalty of -1.0 → normalized to -0.76 (severe)
- **scale=10.0**: Penalty of -10.0 → normalized to -0.76 (severe)

---

## Choosing Scale Parameters

### Gravity (distance²):
```python
GRAVITY_SCALE = 1.0  # 1m floating is severe
```
- Floating 0.5m: -0.25 raw → -0.24 normalized (moderate)
- Floating 1.0m: -1.0 raw → -0.76 normalized (severe)
- Floating 2.0m: -4.0 raw → -0.999 normalized (critical)

### Non-penetration (volumes):
```python
NON_PENETRATION_SCALE = 5.0  # Typical overlap severity
```
- Minor overlap (-2): -0.38 normalized
- Moderate overlap (-5): -0.76 normalized
- Severe overlap (-10): -0.96 normalized
- Extreme overlap (-50): -0.9999 normalized

---

## Final Recommendation

**Use soft normalization (tanh) for unbounded rewards:**

```python
NORMALIZATION_CONFIG = {
    'gravity': {
        'type': 'tanh',
        'scale': 1.0,  # 1m² → -0.76
    },
    'non_penetration': {
        'type': 'tanh',
        'scale': 5.0,  # Moderate overlap → -0.76
    },
    'must_have_furniture': {
        'type': 'linear',
        'max': 10.0,
    },
    'object_count': {
        'type': 'linear',
        'max': 8.0,
        'offset': 1.2,  # Shift to start at 0
    },
}
```

**Benefits:**
- ✅ Handles unbounded rewards gracefully
- ✅ Smooth gradients (no clipping artifacts)
- ✅ Diminishing returns for extreme violations (appropriate!)
- ✅ All normalized to approximately [-1, 0]
- ✅ Scale parameters have physical interpretation

**Next step:** Implement this in commons.py
