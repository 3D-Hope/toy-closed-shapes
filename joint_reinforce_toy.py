import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, DDIMScheduler
import math
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm
import pandas as pd
import time
from typing import Optional, Tuple
from diffusers.utils.torch_utils import randn_tensor

name = "joint_reinforce_toy"

config = {

    "load": None,
    # "load": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/toy_parallelogram/outputs/ppo_inc_2_3_4_5/ckpt_toy_rl/model_checkpoint_rl.pt",
    # "load": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/toy_parallelogram/outputs/only_3_steps/ckpt_toy_rl/model_checkpoint_rl_epoch_4000.pt",
    "load": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/toy_parallelogram/outputs/rect_pretraining_40_rl_40/ckpt_toy/model_checkpoint.pt", #baseline
    "noise_scheduler": "ddim", #or ddpm
    "num_train_steps": 40,
    "num_ddim_inference_steps": 40,
    "checkpoint_dir": f"outputs/{name}/ckpt_toy",
    "skip_training_if_ckpt_exists": True,
    "run_eval": True,  # when set true ckpt even if trained on the fly is not saved (also load should be not none # Set to True to run comprehensive evaluation
    
    # RL Fine-tuning config
    "run_rl": True,  # Set to True to run RL fine-tuning
    "rl_checkpoint_dir": f"outputs/{name}/ckpt_toy_rl",
    "rl_load_from_checkpoint": None,  # Path to checkpoint to fine-tune from (or None to use base checkpoint)
    "rl_num_epochs": 4000,
    "save_every": 1000,
    "rl_batch_size": 512,
    "rl_num_inference_steps": 5, #can not have 5 for eval for some reason, # FIXME:
    "rl_lr": 1e-6,
    "rl_ddpm_reg_weight": 0.0,  # Weight for DDPM regularization loss
    "rl_advantage_max": 10.0,  # Clipping for advantages
    "reward_type": "rectangle",  # "rectangle", "circles", or "ring" - choose reward function
    "ring_inner_distance": 0.5,  # For ring reward: inner distance threshold)
    "ring_outer_distance": 0.3,  # For ring reward: reward points within this distance from boundary
    
    # Joint multi-step training
    "rl_joint_training": True,  # Train on multiple step counts simultaneously
    "rl_step_counts": [2, 3, 4, 5],  # Different inference step counts to train on
    "rl_step_loss_weights": [1.0, 1.0, 1.0, 1.0],  # Optional per-step loss weighting
}

# config = {
#     "noise_scheduler": "ddim", #or ddpm
#     "num_train_steps": 1000,
#     "num_ddim_inference_steps": 150,
#     "checkpoint_dir": "outputs/pretraining_1000_rl_150/ckpt_toy",
#     "skip_training_if_ckpt_exists": True,
#     "run_eval": True,  # Set to True to run comprehensive evaluation
    
#     # RL Fine-tuning config
#     "run_rl": True,  # Set to True to run RL fine-tuning
#     "rl_checkpoint_dir": "outputs/pretraining_1000_rl_150/ckpt_toy_rl",
#     "rl_load_from_checkpoint": None,  # Path to checkpoint to fine-tune from (or None to use base checkpoint)
#     "rl_num_epochs": 100000,
#     "rl_batch_size": 512,
#     "rl_num_inference_steps": 150,
#     "rl_lr": 1e-6,
#     "rl_ddpm_reg_weight": 0.0,  # Weight for DDPM regularization loss
#     "rl_advantage_max": 10.0,  # Clipping for advantages
# }

# ==========================================
# 1. Advanced Model Architecture
#    (Encoder -> Residual Blocks -> Decoder)
# ==========================================

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Mimics a Transformer Feed-Forward block with residual connection"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x

class DiffuserTransformerMLP(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, num_layers=4):
        super().__init__()
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Encoder (Data Projection)
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        # 3. Transformer/Residual Backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])

        # 4. Decoder (Noise Prediction)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        # Embed time
        t_emb = self.time_mlp(t)
        
        # Embed input (Encoder)
        x = self.input_proj(x)
        
        # Add time info to the input embedding
        x = x + t_emb
        
        # Pass through backbone
        for block in self.blocks:
            x = block(x)
            
        # Project back to data dimension (Decoder)
        return self.output_proj(x)

# ==========================================
# 2. Data Preparation
# ==========================================

def make_parallelogram(n_samples=50000):
    rng = np.random.default_rng(42)
    a, b = rng.uniform(0, 1, (n_samples, 1)), rng.uniform(0, 1, (n_samples, 1))
    v1, v2 = np.array([3.0, 0.8]), np.array([0.6, 2.5])
    x = a * v1 + b * v2
    
    # Normalize to [-1, 1] (Crucial for Diffusion)
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x = 2 * (x - x_min) / (x_max - x_min) - 1
    
    return torch.tensor(x, dtype=torch.float32), (v1, v2, x_min, x_max)

device = "cuda" if torch.cuda.is_available() else "cpu"
X_train, parallelogram_params = make_parallelogram()
X_train = X_train.to(device)
dataloader = DataLoader(TensorDataset(X_train), batch_size=512, shuffle=True)

# Store parallelogram geometry for manifold checking
V1, V2, X_MIN, X_MAX = parallelogram_params

# Compute parallelogram corners in normalized space for plotting
def get_parallelogram_corners_normalized(v1, v2, x_min, x_max):
    """Get the 4 corners of the parallelogram in normalized [-1, 1] space."""
    # Define corners in original space
    corners_original = np.array([
        [0, 0],           # Origin
        v1,               # First vector
        v1 + v2,          # Opposite corner
        v2,               # Second vector
        [0, 0]            # Close the loop
    ])
    
    # Normalize to [-1, 1] (same as data normalization)
    corners_normalized = 2 * (corners_original - x_min) / (x_max - x_min) - 1
    
    return corners_normalized

PARALLELOGRAM_CORNERS = get_parallelogram_corners_normalized(V1, V2, X_MIN, X_MAX)

def get_offset_parallelogram_normalized(v1, v2, x_min, x_max, offset_distance):
    """Get corners of a parallelogram offset outward by a given distance.
    
    Args:
        v1, v2: Basis vectors of the parallelogram
        x_min, x_max: Normalization parameters
        offset_distance: Distance to offset outward from each edge (positive = outward, negative = inward)
        
    Returns:
        Array of corners in normalized space
    """
    # Original corners: [0,0], v1, v1+v2, v2
    # We need to offset each of the 4 edges outward by offset_distance
    
    # Compute outward normals for each edge
    # Edge 0->v1 (bottom edge)
    edge_01 = v1
    normal_01 = np.array([-edge_01[1], edge_01[0]])
    normal_01 = normal_01 / np.linalg.norm(normal_01)
    
    # Edge v1->v1+v2 (right edge)
    edge_12 = v2
    normal_12 = np.array([-edge_12[1], edge_12[0]])
    normal_12 = normal_12 / np.linalg.norm(normal_12)
    
    # Edge v1+v2->v2 (top edge)
    edge_23 = -v1
    normal_23 = np.array([-edge_23[1], edge_23[0]])
    normal_23 = normal_23 / np.linalg.norm(normal_23)
    
    # Edge v2->0 (left edge)
    edge_30 = -v2
    normal_30 = np.array([-edge_30[1], edge_30[0]])
    normal_30 = normal_30 / np.linalg.norm(normal_30)
    
    # Check if normals point outward (cross product with edge should be positive for CCW)
    # For outward normal, it should point away from center
    center = (v1 + v2) / 2
    
    # Check each normal and flip if needed
    if np.dot(normal_01, center - v1/2) > 0:
        normal_01 = -normal_01
    if np.dot(normal_12, center - (v1 + v2/2)) > 0:
        normal_12 = -normal_12
    if np.dot(normal_23, center - (v2 + v1/2)) > 0:
        normal_23 = -normal_23
    if np.dot(normal_30, center - v2/2) > 0:
        normal_30 = -normal_30
    
    # Offset each edge and find new corner intersections
    # New corner 0: intersection of offset edge_30 and edge_01
    # Offset edge_30: starts at v2 + normal_30 * offset_distance, direction -v2
    # Offset edge_01: starts at 0 + normal_01 * offset_distance, direction v1
    
    # Solve for intersection using parametric equations
    # Point on offset_30: (v2 + normal_30 * d) + t * (-v2)
    # Point on offset_01: (normal_01 * d) + s * v1
    # These are equal at intersection
    
    d = offset_distance
    
    # Corner 0: intersection of offset bottom and left edges
    # Parametric: normal_01 * d + s * v1 = v2 + normal_30 * d + t * (-v2)
    # s * v1 + t * v2 = v2 + normal_30 * d - normal_01 * d
    # Solve: [v1, v2] * [s, t]' = v2 + d * (normal_30 - normal_01)
    M = np.column_stack([v1, v2])
    rhs = v2 + d * (normal_30 - normal_01)
    params = np.linalg.solve(M, rhs)
    corner_0 = normal_01 * d + params[0] * v1
    
    # Corner 1: intersection of offset bottom and right edges  
    # normal_01 * d + s * v1 = v1 + normal_12 * d + t * v2
    rhs = v1 + d * (normal_12 - normal_01)
    params = np.linalg.solve(M, rhs)
    corner_1 = normal_01 * d + params[0] * v1
    
    # Corner 2: intersection of offset right and top edges
    # v1 + normal_12 * d + s * v2 = v1 + v2 + normal_23 * d + t * (-v1)
    # v1 + normal_12 * d + s * v2 = v1 + v2 + normal_23 * d - t * v1
    # t * v1 + s * v2 = v2 + d * (normal_23 - normal_12)
    rhs = v2 + d * (normal_23 - normal_12)
    params = np.linalg.solve(M, rhs)
    corner_2 = v1 + normal_12 * d + params[1] * v2
    
    # Corner 3: intersection of offset top and left edges
    # v1 + v2 + normal_23 * d + s * (-v1) = v2 + normal_30 * d + t * (-v2)
    # v1 + v2 + normal_23 * d - s * v1 = v2 + normal_30 * d - t * v2
    # s * v1 - t * v2 = v1 + d * (normal_23 - normal_30)
    # [v1, -v2] * [s, t]' = v1 + d * (normal_23 - normal_30)
    M2 = np.column_stack([v1, -v2])
    rhs = v1 + d * (normal_23 - normal_30)
    params = np.linalg.solve(M2, rhs)
    corner_3 = v1 + v2 + normal_23 * d - params[0] * v1
    
    corners_original = np.array([
        corner_0,
        corner_1,
        corner_2,
        corner_3,
        corner_0  # Close the loop
    ])
    
    # Normalize to [-1, 1] (same as data normalization)
    corners_normalized = 2 * (corners_original - x_min) / (x_max - x_min) - 1
    
    return corners_normalized

RECT_BOUNDS = (-0.5, 0.5, -0.5, 0.5)

x_min_rect, x_max_rect, y_min_rect, y_max_rect = RECT_BOUNDS

rectangle_corners = np.array([
    [x_min_rect, y_min_rect],
    [x_max_rect, y_min_rect],
    [x_max_rect, y_max_rect],
    [x_min_rect, y_max_rect],
    [x_min_rect, y_min_rect]
])

# Define two circles within the parallelogram (in normalized space)
# Circle 1: center at (-0.3, 0.0), radius 0.25
# Circle 2: center at (-0.3, 0.0), radius 0.25
CIRCLE_CENTERS = np.array([[-0.3, 0.0], [-0.3, 0.0]])
CIRCLE_RADII = np.array([0.25, 0.35])

# ==========================================
# Evaluation Metrics
# ==========================================

def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet Distance between two Gaussian distributions."""
    diff = mu1 - mu2
    # Product might be almost singular
    covmean = sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fd

def compute_coverage_and_mmd(real_samples, fake_samples, threshold=0.05):
    """
    Coverage: Percentage of real samples that have at least one fake sample nearby.
    MMD: Minimum Matching Distance - average distance from fake to nearest real.
    """
    # Compute pairwise distances
    distances = cdist(real_samples, fake_samples, metric='euclidean')
    
    # Coverage: for each real sample, find minimum distance to any fake sample
    min_dist_real_to_fake = distances.min(axis=1)
    coverage = (min_dist_real_to_fake < threshold).mean()
    
    # MMD: for each fake sample, find minimum distance to any real sample
    min_dist_fake_to_real = distances.min(axis=0)
    mmd = min_dist_fake_to_real.mean()
    
    return coverage, mmd

def check_points_in_parallelogram(points, v1, v2, x_min, x_max):
    """
    Check if normalized points are within the original parallelogram defined by v1, v2.
    
    The parallelogram is: p = a*v1 + b*v2, where 0 <= a, b <= 1
    
    Args:
        points: Normalized points in [-1, 1] range (N, 2) - can be numpy or torch
        v1, v2: Basis vectors of the parallelogram
        x_min, x_max: Min/max values used for normalization
        
    Returns:
        Boolean array indicating if each point is inside the parallelogram
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    # Inverse normalization: from [-1, 1] back to original space
    points_original = (points + 1) / 2 * (x_max - x_min) + x_min
    
    # Solve: point = a*v1 + b*v2
    # This is a linear system: [v1 v2] * [a; b] = point
    # Stack v1 and v2 as columns
    M = np.column_stack([v1, v2])
    
    # Solve for coefficients [a, b] for each point
    try:
        coeffs = np.linalg.solve(M, points_original.T).T  # Shape: (N, 2)
        a, b = coeffs[:, 0], coeffs[:, 1]
        
        # Check if 0 <= a, b <= 1 (with small tolerance for numerical errors)
        eps = 1e-6
        inside = (a >= -eps) & (a <= 1 + eps) & (b >= -eps) & (b <= 1 + eps)
        
        return inside
    except np.linalg.LinAlgError:
        # If singular, fall back to all False
        return np.zeros(len(points), dtype=bool)

def check_points_in_rectangle(points, rect_bounds):
    """
    Check if normalized points are within a rectangle defined in original space.
    
    Args:
        points: Normalized points in [-1, 1] range (N, 2) - can be numpy or torch
        rect_bounds: Tuple of (x_min_rect, x_max_rect, y_min_rect, y_max_rect) in original space
        
    Returns:
        Boolean array indicating if each point is inside the rectangle
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    
    # Unpack rectangle bounds
    x_min_rect, x_max_rect, y_min_rect, y_max_rect = rect_bounds
    
    # Check if points are inside rectangle
    eps = 1e-6
    inside = (
        (points[:, 0] >= x_min_rect - eps) & 
        (points[:, 0] <= x_max_rect + eps) &
        (points[:, 1] >= y_min_rect - eps) & 
        (points[:, 1] <= y_max_rect + eps)
    )
    
    return inside

def compute_rectangle_grid_coverage(points, rect_bounds, grid_size=10):
    """
    Compute grid cell coverage for rectangle to detect reward hacking.
    Measures what percentage of grid cells contain at least one point.
    
    Args:
        points: Generated samples (N, 2) - can be numpy or torch
        rect_bounds: Tuple of (x_min_rect, x_max_rect, y_min_rect, y_max_rect)
        grid_size: Number of cells per dimension (grid_size x grid_size grid)
        
    Returns:
        coverage: Percentage of cells with at least one point (0 to 1)
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    # Filter to only points inside rectangle
    inside_mask = check_points_in_rectangle(points, rect_bounds)
    points_inside = points[inside_mask]
    
    if len(points_inside) == 0:
        return 0.0
    
    # Unpack bounds
    x_min, x_max, y_min, y_max = rect_bounds
    
    # Create grid
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    
    # Compute 2D histogram
    hist, _, _ = np.histogram2d(
        points_inside[:, 0], 
        points_inside[:, 1], 
        bins=[x_bins, y_bins]
    )
    
    # Count cells with at least one point
    occupied_cells = np.sum(hist > 0)
    total_cells = grid_size * grid_size
    
    coverage = occupied_cells / total_cells
    
    return coverage

def compute_distance_from_parallelogram(points, v1, v2, x_min, x_max):
    """
    Compute minimum distance from each point to the parallelogram boundary.
    Returns positive distance for points outside, zero for points inside/on boundary.
    
    Args:
        points: Normalized points in [-1, 1] range (N, 2) - can be numpy or torch
        v1, v2: Basis vectors of the parallelogram
        x_min, x_max: Min/max values used for normalization
        
    Returns:
        Array of distances from parallelogram boundary (N,)
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    # Inverse normalization: from [-1, 1] back to original space
    points_original = (points + 1) / 2 * (x_max - x_min) + x_min
    
    # Get parallelogram corners in original space
    corners = np.array([
        [0, 0],
        v1,
        v1 + v2,
        v2
    ])
    
    # For each point, compute distance to parallelogram
    distances = np.zeros(len(points))
    
    for i, point in enumerate(points_original):
        # Solve: point = a*v1 + b*v2
        M = np.column_stack([v1, v2])
        try:
            coeffs = np.linalg.solve(M, point)
            a, b = coeffs[0], coeffs[1]
            
            # Clamp to [0, 1] to find nearest point on parallelogram
            a_clamped = np.clip(a, 0, 1)
            b_clamped = np.clip(b, 0, 1)
            
            # Nearest point on parallelogram
            nearest_point = a_clamped * v1 + b_clamped * v2
            
            # Distance
            distances[i] = np.linalg.norm(point - nearest_point)
        except np.linalg.LinAlgError:
            # If singular, use distance to nearest corner
            distances[i] = np.min([np.linalg.norm(point - corner) for corner in corners])
    
    return distances


def compute_signed_distance_from_parallelogram(points, v1, v2, x_min, x_max):
    """
    Compute signed distance from each point to the parallelogram boundary.
    Negative for points inside, positive for points outside.
    
    Args:
        points: Normalized points in [-1, 1] range (N, 2) - can be numpy or torch
        v1, v2: Basis vectors of the parallelogram
        x_min, x_max: Min/max values used for normalization
        
    Returns:
        Array of signed distances (N,)
        - Negative values: point is inside, abs(value) = distance from boundary inward
        - Positive values: point is outside, value = distance from boundary outward
        - Zero: point is on the boundary
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    # Inverse normalization: from [-1, 1] back to original space
    points_original = (points + 1) / 2 * (x_max - x_min) + x_min
    
    # For each point, compute signed distance to parallelogram
    signed_distances = np.zeros(len(points))
    
    for i, point in enumerate(points_original):
        # Solve: point = a*v1 + b*v2
        M = np.column_stack([v1, v2])
        try:
            coeffs = np.linalg.solve(M, point)
            a, b = coeffs[0], coeffs[1]
            
            # Check if point is inside (0 <= a, b <= 1)
            eps = 1e-6
            inside = (a >= -eps) and (a <= 1 + eps) and (b >= -eps) and (b <= 1 + eps)
            
            if inside:
                # Point is inside: compute distance to nearest edge (negative)
                # Distance to each edge defined by a=0, a=1, b=0, b=1
                dist_to_edges = [
                    a * np.linalg.norm(v1),           # distance to a=0 edge
                    (1-a) * np.linalg.norm(v1),       # distance to a=1 edge
                    b * np.linalg.norm(v2),           # distance to b=0 edge
                    (1-b) * np.linalg.norm(v2)        # distance to b=1 edge
                ]
                signed_distances[i] = -min(dist_to_edges)
            else:
                # Point is outside: compute distance to nearest point on boundary (positive)
                a_clamped = np.clip(a, 0, 1)
                b_clamped = np.clip(b, 0, 1)
                nearest_point = a_clamped * v1 + b_clamped * v2
                signed_distances[i] = np.linalg.norm(point - nearest_point)
                
        except np.linalg.LinAlgError:
            # If singular, compute distance to nearest corner (assume outside)
            corners = np.array([[0, 0], v1, v1 + v2, v2])
            signed_distances[i] = np.min([np.linalg.norm(point - corner) for corner in corners])
    
    return signed_distances


def check_points_in_circles(points, circle_centers, circle_radii):
    """
    Check if points are within any of the given circles.
    
    Args:
        points: Points to check (N, 2) - can be numpy or torch
        circle_centers: Centers of circles (num_circles, 2)
        circle_radii: Radii of circles (num_circles,)
        
    Returns:
        Boolean array indicating if each point is inside any circle
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    inside = np.zeros(len(points), dtype=bool)
    
    # Check each circle
    for center, radius in zip(circle_centers, circle_radii):
        distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
        inside |= (distances <= radius)
    
    return inside


def compute_rectangle_reward(samples):
    """Compute reward based on rectangle (existing reward)."""
    rect_bounds = RECT_BOUNDS
    inside = check_points_in_rectangle(samples, rect_bounds)
    
    # Convert to torch tensor
    rewards = torch.tensor(inside, dtype=torch.float32, device=samples.device)
    rewards = 2.0 * rewards - 1.0  # Map True->1.0, False->-1.0
    
    return rewards


def compute_circles_reward(samples):
    """Compute reward based on annulus: inside bigger circle but outside smaller circle."""
    # Convert to numpy if torch tensor
    if torch.is_tensor(samples):
        points = samples.detach().cpu().numpy()
    else:
        points = samples
    
    # Both circles have the same center, so we can compute distance once
    center = CIRCLE_CENTERS[0]  # Both circles share the same center
    smaller_radius = CIRCLE_RADII[0]  # 0.25
    bigger_radius = CIRCLE_RADII[1]   # 0.35
    
    # Compute distances from center
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    
    # Reward +1 only if inside bigger circle AND outside smaller circle (annulus/ring)
    in_annulus = (distances > smaller_radius) & (distances <= bigger_radius)
    
    # Convert to torch tensor
    rewards = torch.tensor(in_annulus, dtype=torch.float32, device=samples.device)
    rewards = 2.0 * rewards - 1.0  # Map True->1.0, False->-1.0
    
    return rewards


def compute_ring_reward(samples, v1, v2, x_min, x_max, inner_dist=0.0, outer_dist=0.3):
    """Compute reward based on ring/band around parallelogram boundary.
    
    Rewards points in the band region between inner and outer boundaries:
    - Inner boundary: inner_dist units INSIDE the parallelogram from the edge
    - Outer boundary: outer_dist units OUTSIDE the parallelogram from the edge
    - Points in the ring between these boundaries get +1 reward
    
    Uses signed distance:
    - Negative distance = point is inside parallelogram (abs value = distance from edge inward)
    - Positive distance = point is outside parallelogram (value = distance from edge outward)
    
    Reward = +1 if: -inner_dist <= signed_distance <= outer_dist
    Reward = -1 otherwise
    
    Args:
        samples: Points to evaluate (N, 2)
        v1, v2: Parallelogram basis vectors
        x_min, x_max: Normalization parameters
        inner_dist: Distance inside from boundary (0.0 = boundary itself)
        outer_dist: Distance outside from boundary
    """
    signed_distances = compute_signed_distance_from_parallelogram(samples, v1, v2, x_min, x_max)
    
    # Reward points in the ring:
    # - signed_distance >= -inner_dist (not too deep inside)
    # - signed_distance <= outer_dist (not too far outside)
    in_ring = (signed_distances >= -inner_dist) & (signed_distances <= outer_dist)
    
    # Convert to torch tensor
    rewards = torch.tensor(in_ring, dtype=torch.float32, device=samples.device)
    rewards = 2.0 * rewards - 1.0  # Map True->1.0, False->-1.0
    
    return rewards


def compute_geometric_reward(samples, v1, v2, x_min, x_max, reward_type="rectangle"):
    """Compute geometric reward based on the selected reward type."""
    # print(f"Computing {reward_type} reward...")
    if reward_type == "circles":
        return compute_circles_reward(samples)
    elif reward_type == "ring":
        inner_dist = config.get("ring_inner_distance", 0.3)
        outer_dist = config.get("ring_outer_distance", 0.3)
        return compute_ring_reward(samples, v1, v2, x_min, x_max, inner_dist, outer_dist)
    else:  # default to rectangle
        return compute_rectangle_reward(samples)


def evaluate_samples(real_samples, generated_samples):
    """Compute evaluation metrics."""
    # Convert to numpy if needed
    if torch.is_tensor(real_samples):
        real_samples = real_samples.cpu().numpy()
    if torch.is_tensor(generated_samples):
        generated_samples = generated_samples.cpu().numpy()
    
    # Compute statistics
    mu_real = np.mean(real_samples, axis=0)
    sigma_real = np.cov(real_samples, rowvar=False)
    
    mu_gen = np.mean(generated_samples, axis=0)
    sigma_gen = np.cov(generated_samples, rowvar=False)
    
    # Frechet Distance
    fd = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    # Coverage and MMD (use subset for efficiency)
    n_eval = min(5000, len(real_samples), len(generated_samples))
    coverage, mmd = compute_coverage_and_mmd(
        real_samples[:n_eval], 
        generated_samples[:n_eval]
    )
    rewards = compute_geometric_reward(
        torch.tensor(generated_samples[:n_eval], device=device), V1, V2, X_MIN, X_MAX,
        reward_type=config.get("reward_type", "rectangle")
    )
    
    # Rectangle grid coverage (detects reward hacking/center clustering)
    rectangle_coverage = compute_rectangle_grid_coverage(
        generated_samples[:n_eval], 
        RECT_BOUNDS, 
        grid_size=10
    )

    
    return {
        'frechet_distance': fd,
        'coverage': coverage,
        'mmd': mmd,
        'rewards_mean': rewards.mean().item(),
        'rectangle_coverage': rectangle_coverage,
    }

# ==========================================
# 3. Setup Diffusers Scheduler & Training
# ==========================================

# NOTE: beta_schedule="squaredcos_cap_v2" is vastly superior for 
# 2D geometric shapes compared to "linear"

noise_scheduler = DDPMScheduler(
    num_train_timesteps=config["num_train_steps"],
    beta_schedule="linear",
    prediction_type="epsilon" # We predict noise
)

model = DiffuserTransformerMLP(hidden_dim=256, num_layers=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# ==========================================
# Checkpoint Management
# ==========================================

checkpoint_path = Path(config["checkpoint_dir"])
checkpoint_path.mkdir(parents=True, exist_ok=True)
checkpoint_file = checkpoint_path / "model_checkpoint.pt"

if config["load"] is not None:
    checkpoint_file = Path(config["load"])
start_epoch = 0
num_epochs = 100

if checkpoint_file.exists() and config["skip_training_if_ckpt_exists"]:
    print(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")
    
    if start_epoch >= num_epochs:
        print("Training already completed. Skipping training.")
        num_epochs = start_epoch  # Skip training loop
elif config["rl_load_from_checkpoint"] is not None:
    start_epoch = num_epochs + 1  # Skip training loop
else:
    print(f"Training on {device} using Diffusers library...")
for epoch in range(start_epoch, num_epochs):
    epoch_loss = 0.0
    model.eval()
    
    for (batch_x,) in dataloader:
        batch_x = batch_x.to(device)
        noise = torch.randn_like(batch_x)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, 
            (batch_x.shape[0],), device=device
        ).long()

        # Diffusers: Add noise
        noisy_x = noise_scheduler.add_noise(batch_x, noise, timesteps)

        # Model Prediction
        noise_pred = model(noisy_x, timesteps)

        # Calculate Loss
        loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

if not config["run_eval"] and config["load"] is not None:
    # Save final checkpoint
    if start_epoch < num_epochs:
        print(f"Saving checkpoint to {checkpoint_file}")
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_file)
        print("Checkpoint saved.")

# ==========================================
# 4. Sampling with Diffusers Library
# ==========================================
def ddim_step_with_logprob(
    scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 1.0,
    use_clipped_model_output: bool = False,
    generator=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied and adapted from diffusers DDIM scheduler to return log probability.
    
    Predict the sample at the previous timestep by reversing the SDE.
    Returns both the predicted previous sample and its log probability.
    
    Args:
        scheduler: DDIMScheduler instance
        model_output: Direct output from learned diffusion model (predicted noise)
        timestep: Current discrete timestep in the diffusion chain
        sample: Current instance of sample being created by diffusion process
        eta: Weight of noise for added noise in diffusion step (0 = deterministic DDIM)
        use_clipped_model_output: If True, recompute from clipped x_0
        generator: Random number generator
        
    Returns:
        Tuple of (prev_sample, log_prob)
    """
    assert isinstance(
        scheduler, DDIMScheduler
    ), "scheduler must be an instance of DDIMScheduler"
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' "
            "after creating the scheduler"
        )
    
    # 1. Get previous step value (=t-1)
    prev_timestep = (
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )
    
    # 2. Compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    
    beta_prod_t = 1 - alpha_prod_t
    
    # 3. Compute predicted original sample from predicted noise (x_0)
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
    
    # 5. Compute variance: "sigma_t(η)"
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    
    if use_clipped_model_output:
        # Re-derive pred_epsilon from clipped x_0
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)
    
    # 6. Compute "direction pointing to x_t"
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon
    
    # 7. Compute x_{t-1} mean (without random noise)
    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )
    
    # 8. Sample from Gaussian
    variance_noise = randn_tensor(
        model_output.shape,
        generator=generator,
        device=model_output.device,
        dtype=model_output.dtype,
    )
    prev_sample = prev_sample_mean + std_dev_t * variance_noise
    
    # 9. Compute log probability of the sample
    # Key trick: detach prev_sample so gradients only flow through the mean
    std_dev_t = torch.clamp(std_dev_t, min=1e-6)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=std_dev_t.device)))
    )
    
    # Mean over all dimensions except batch
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob



@torch.no_grad()
def ddpm_sample_with_diffusers(model, scheduler, n_samples=10000, num_steps=None):
    model.eval()
    if num_steps is None:
        num_steps = scheduler.config.num_train_timesteps
    
    # 1. Start from random noise
    x = torch.randn(n_samples, 2, device=device)
    
    # 2. Initialize scheduler timesteps
    scheduler.set_timesteps(num_steps)
    
    # 3. Denoising Loop
    for t in scheduler.timesteps:
        # Create batch of timesteps (must be on same device as model)
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        model_output = model(x, t_batch)
        
        # Step back: x_{t-1} = scheduler.step(...)
        step_output = scheduler.step(model_output, t, x)
        x = step_output.prev_sample
        
    return x.cpu().numpy()

@torch.no_grad()
def ddim_sample_with_diffusers(model, scheduler, n_samples=10000, num_steps=None):
    assert scheduler.__class__ == DDIMScheduler, "Scheduler must be DDIMScheduler"
    model.eval()
    if num_steps is None:
        num_steps = config["num_ddim_inference_steps"]
    
    # 1. Start from random noise
    x = torch.randn(n_samples, 2, device=device)
    
    # 2. Initialize scheduler timesteps
    scheduler.set_timesteps(num_steps)
    
    # 3. Denoising Loop
    for t in scheduler.timesteps:
        # Create batch of timesteps (must be on same device as model)
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        model_output = model(x, t_batch)
        
        # Step back: x_{t-1} = scheduler.step(...)
        step_output = scheduler.step(model_output, t, x, eta=1.0)
        x = step_output.prev_sample
        # print(f"using ddim step with logprob at timestep {t}")
        # x, _ = ddim_step_with_logprob(
        #     scheduler=scheduler,
        #     model_output=model_output,
        #     timestep=t,
        #     sample=x,
        #     eta=1.0,
        # )
        
    return x.detach().cpu().numpy()


def ddim_sample_with_log_probs(model, scheduler, n_samples=256, num_steps=20, timesteps=None, eta=1.0):
    """
    Sample using DDIM while tracking log probabilities for RL training.
    
    Args:
        model: Diffusion model
        scheduler: DDIMScheduler instance
        n_samples: Number of samples to generate
        num_steps: Number of denoising steps
        timesteps: Optional custom timesteps to use (for aligned paths)
        eta: Stochasticity parameter (0 = deterministic DDIM)
        
    Returns:
        samples: Final samples (n_samples, 2)
        log_probs: Sum of log probabilities across all timesteps (n_samples,)
    """
    model.eval()
    
    # 1. Start from random noise
    x = torch.randn(n_samples, 2, device=device)
    
    # 2. Initialize scheduler timesteps
    if timesteps is None:
        scheduler.set_timesteps(num_steps)
        timesteps = scheduler.timesteps
    else:
        scheduler.set_timesteps(num_steps)
    
    # Track log probabilities
    trajectory = []
    trajectory_log_probs = []
    
    trajectory.append(x)
    # 3. Denoising Loop
    for t in timesteps:
        # Create batch of timesteps
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise (model forward pass)
        model_output = model(x, t_batch)
        
        # DDIM step with log probability
        xt, log_prob = ddim_step_with_logprob(
            scheduler=scheduler,
            model_output=model_output,
            timestep=t,
            sample=x,
            eta=eta,
        )
        
        # Accumulate log probabilities
        trajectory_log_probs.append(log_prob)
        trajectory.append(xt)
        x = xt
        
    
    trajectories = torch.stack(trajectory, dim=1)
    trajectories_log_probs = torch.stack(
            trajectory_log_probs, dim=1
        )
    return trajectories, trajectories_log_probs

# ==========================================
# Comprehensive Evaluation
# ==========================================

if config["run_eval"]:
    print("\n" + "="*60)
    print("Running Comprehensive Evaluation")
    print("="*60)
    
    # Define evaluation configurations
    # eval_configs = [
    #     {"scheduler": "ddpm", "steps": config["num_train_steps"], "name": f"DDPM-{config['num_train_steps']}"},
    #     {"scheduler": "ddim", "steps": config["num_ddim_inference_steps"]//10, "name": f"DDIM-{config['num_ddim_inference_steps']//10}"},
    #     {"scheduler": "ddim", "steps": config["num_ddim_inference_steps"]//8, "name": f"DDIM-{config['num_ddim_inference_steps']//8}"},
    #     {"scheduler": "ddim", "steps": config["num_ddim_inference_steps"]//2, "name": f"DDIM-{config['num_ddim_inference_steps']//2}"},
    #     {"scheduler": "ddim", "steps": config["num_ddim_inference_steps"], "name": f"DDIM-{config['num_ddim_inference_steps']}"},
    # ]
    eval_configs = [
        {"scheduler": "ddpm", "steps": 40, "name": f"DDPM-40"},
        {"scheduler": "ddim", "steps": 2, "name": f"DDIM-2"},
        {"scheduler": "ddim", "steps": 3, "name": f"DDIM-3"},
        {"scheduler": "ddim", "steps": 4, "name": f"DDIM-4"},
        {"scheduler": "ddim", "steps": 5, "name": f"DDIM-5"},
        {"scheduler": "ddim", "steps": 10, "name": f"DDIM-10"},
        {"scheduler": "ddim", "steps": 15, "name": f"DDIM-15"},
        {"scheduler": "ddim", "steps": 20, "name": f"DDIM-20"},
        {"scheduler": "ddim", "steps": 40, "name": f"DDIM-40"},   
    ] 
    results = []
    all_samples = {}
    
    for eval_cfg in eval_configs:
        print(f"\nEvaluating {eval_cfg['name']}...")
        
        start_time = time.time()
        
        if eval_cfg['scheduler'] == 'ddpm':
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=config["num_train_steps"],
                beta_schedule="linear",
                prediction_type="epsilon" # We predict noise
            )

            samples = ddpm_sample_with_diffusers(
                model, noise_scheduler, 
                n_samples=10000, 
                num_steps=eval_cfg['steps']
            )
        else:  # ddim
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=config["num_train_steps"],
                beta_schedule="linear",
                prediction_type="epsilon" # We predict noise
            )
            samples = ddim_sample_with_diffusers(
                model, noise_scheduler, 
                n_samples=10000, 
                num_steps=eval_cfg['steps']
            )
        
        sampling_time = time.time() - start_time
        
        # Compute metrics
        metrics = evaluate_samples(X_train[:512], samples)
        
        # Store results
        results.append({
            'scheduler': eval_cfg['scheduler'].upper(),
            'num_steps': eval_cfg['steps'],
            'name': eval_cfg['name'],
            'frechet_distance': metrics['frechet_distance'],
            'coverage': metrics['coverage'],
            'mmd': metrics['mmd'],
            "mean_reward": metrics['rewards_mean'],
            'rectangle_coverage': metrics['rectangle_coverage'],
            'sampling_time_sec': sampling_time,
        })
        
        # Store samples for visualization
        all_samples[eval_cfg['name']] = samples
        
        print(f"  FD: {metrics['frechet_distance']:.6f} | "
              f"Coverage: {metrics['coverage']:.4f} | "
              f"MMD: {metrics['mmd']:.6f} | "
              f"Mean Reward: {metrics['rewards_mean']:.4f} | "
              f"Rect Coverage: {metrics['rectangle_coverage']:.4f} | "
              f"Time: {sampling_time:.2f}s")
    
    # Save results to CSV
    df_results = pd.DataFrame(results)
    csv_path = checkpoint_path / "evaluation_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Evaluation results saved to {csv_path}")
    
    # Create comprehensive visualization
    print("\nCreating comparative visualization...")
    n_configs = len(eval_configs)
    fig = plt.figure(figsize=(20, 4 * ((n_configs + 1) // 2)))
    
    # Plot ground truth first
    plt.subplot(((n_configs + 2) // 2), 4, 1)
    orig_data = X_train[:5000].cpu().numpy()
    plt.scatter(orig_data[:, 0], orig_data[:, 1], s=1, alpha=0.5, c='blue')
    # Plot parallelogram boundary
    plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1], 
             'k-', linewidth=2, label='True Manifold')
    plt.title("Ground Truth", fontsize=12, fontweight='bold')
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # Plot each configuration
    for idx, eval_cfg in enumerate(eval_configs, start=2):
        plt.subplot(((n_configs + 2) // 2), 4, idx)
        samples_plot = all_samples[eval_cfg['name']][:5000]
        plt.scatter(samples_plot[:, 0], samples_plot[:, 1], s=1, alpha=0.5, c='red')
        # Plot parallelogram boundary
        plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1], 
                 'k-', linewidth=1.5, alpha=0.7)
        
        # # Plot reward shape based on reward_type
        # if config.get("reward_type", "rectangle") == "circles":
        #     for i, (center, radius) in enumerate(zip(CIRCLE_CENTERS, CIRCLE_RADII)):
        #         circle = plt.Circle(center, radius, fill=False, color='orange', linewidth=2, 
        #                           label='RL reward Manifold' if i == 0 else '')
        #         plt.gca().add_patch(circle)
        # elif config.get("reward_type", "rectangle") == "ring":
        #     # Draw inner and outer parallelograms to show the ring boundaries
        #     outer_dist = config.get("ring_outer_distance", 0.3)
        #     inner_dist = config.get("ring_inner_distance", 0.0)
            
        #     # Draw outer boundary (offset OUTWARD by outer_dist)
        #     outer_corners = get_offset_parallelogram_normalized(V1, V2, X_MIN, X_MAX, outer_dist)
        #     plt.plot(outer_corners[:, 0], outer_corners[:, 1],
        #              '-', color='orange', linewidth=2, label='RL reward Manifold (outer)')
            
        #     # Draw inner boundary (offset INWARD by inner_dist, negative offset goes inside)
        #     if inner_dist > 0:
        #         # Offset INWARD (negative direction)
        #         inner_corners = get_offset_parallelogram_normalized(V1, V2, X_MIN, X_MAX, -inner_dist)
        #         plt.plot(inner_corners[:, 0], inner_corners[:, 1],
        #                  '-', color='orange', linewidth=2, linestyle='--', label='RL reward Manifold (inner)')
        #     else:
        #         # inner_dist = 0 means inner boundary is the original parallelogram
        #         plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1],
        #                  '-', color='orange', linewidth=2, linestyle='--', label='RL reward Manifold (inner)')
        # else:
        #     plt.plot(rectangle_corners[:, 0], rectangle_corners[:, 1], '-', color='orange', linewidth=2, label='RL reward Manifold')
        
        # Get metrics for title
        row = df_results[df_results['name'] == eval_cfg['name']].iloc[0]
        title = f"{eval_cfg['name']}\n"
        title += f"FD: {row['frechet_distance']:.3f} | Cov: {row['coverage']:.3f}\n"
        title += f"Reward: {row['mean_reward']:.3f} | RectCov: {row['rectangle_coverage']:.3f}\n"
        title += f"{row['sampling_time_sec']:.1f}s"
        plt.title(title, fontsize=9)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = checkpoint_path / "evaluation_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to {comparison_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    
else:
    # Single sampling for non-eval mode
    print("\nGenerating samples...")
    
    if config["noise_scheduler"] == "ddim":
        noise_scheduler = DDIMScheduler(
                num_train_timesteps=config["num_train_steps"],
                beta_schedule="linear",
                prediction_type="epsilon" # We predict noise
            )
        print(f"Using DDIM sampling with {config['num_ddim_inference_steps']} steps.")
        samples = ddim_sample_with_diffusers(model, noise_scheduler)
    else:
        noise_scheduler = DDPMScheduler(
                num_train_timesteps=config["num_train_steps"],
                beta_schedule="linear",
                prediction_type="epsilon" # We predict noise
            )
        samples = ddpm_sample_with_diffusers(model, noise_scheduler)
    
    # Quick evaluation
    print("\nEvaluating generated samples...")
    metrics = evaluate_samples(X_train[:10000], samples[:10000])
    print(f"Frechet Distance: {metrics['frechet_distance']:.6f}")
    print(f"Coverage (threshold=0.05): {metrics['coverage']:.4f}")
    print(f"MMD (Mean Min Distance): {metrics['mmd']:.6f}")
    print(f"Mean Reward: {metrics['rewards_mean']:.4f}")
    print(f"Rectangle Coverage: {metrics['rectangle_coverage']:.4f}")

# ==========================================
# 5. Simple Visualization (non-eval mode)
# ==========================================

if not config["run_eval"]:
    plt.figure(figsize=(12, 6))

    # Plot Original Data (Subset)
    plt.subplot(1, 2, 1)
    orig_data = X_train[:5000].cpu().numpy()
    plt.scatter(orig_data[:, 0], orig_data[:, 1], s=1, alpha=0.5, c='blue')
    plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1], 
             'k-', linewidth=2, label='True Manifold')
    plt.title("Ground Truth (Normalized)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot Generated Data
    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='red')
    plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1], 
             'k-', linewidth=2, label='True Manifold')
    
    # Plot reward shape based on reward_type
    if config.get("reward_type", "rectangle") == "circles":
        for i, (center, radius) in enumerate(zip(CIRCLE_CENTERS, CIRCLE_RADII)):
            circle = plt.Circle(center, radius, fill=False, color='orange', linewidth=2, 
                              label='RL Reward Region' if i == 0 else '')
            plt.gca().add_patch(circle)
    elif config.get("reward_type", "rectangle") == "ring":
        # Draw inner and outer parallelograms to show the ring boundaries
        outer_dist = config.get("ring_outer_distance", 0.3)
        inner_dist = config.get("ring_inner_distance", 0.0)
        
        # Draw outer boundary (offset OUTWARD by outer_dist)
        outer_corners = get_offset_parallelogram_normalized(V1, V2, X_MIN, X_MAX, outer_dist)
        plt.plot(outer_corners[:, 0], outer_corners[:, 1],
                 '-', color='orange', linewidth=2, label='RL Reward Region (outer)')
        
        # Draw inner boundary (offset INWARD by inner_dist, negative offset goes inside)
        if inner_dist > 0:
            # Offset INWARD (negative direction)
            inner_corners = get_offset_parallelogram_normalized(V1, V2, X_MIN, X_MAX, -inner_dist)
            plt.plot(inner_corners[:, 0], inner_corners[:, 1],
                     '-', color='orange', linewidth=2, linestyle='--', label='RL Reward Region (inner)')
        else:
            # inner_dist = 0 means inner boundary is the original parallelogram
            plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1],
                     '-', color='orange', linewidth=2, linestyle='--', label='RL Reward Region (inner)')
    else:
        plt.plot(rectangle_corners[:, 0], rectangle_corners[:, 1], '-', color='orange', linewidth=2, label='RL Reward Region')
    
    plt.title(f"Diffusers Generated (Epoch {num_epochs}), with sampler {config['noise_scheduler']}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    filename = "diffusers_parallelogram_result.png"
    plt.savefig(filename)
    print(f"\nVisualization saved to {filename}")


# ==========================================
# 6. RL Fine-tuning (REINFORCE + DDPM Regularization)
# ==========================================

if config["run_rl"]:
    start_time = time.time()

    print("\n" + "="*60)
    print("RL Fine-tuning")
    print("="*60)
    
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config["num_train_steps"],
        beta_schedule="linear",
        prediction_type="epsilon" # We predict noise
    )

    # Setup RL checkpoint directory
    rl_checkpoint_path = Path(config["rl_checkpoint_dir"])
    rl_checkpoint_path.mkdir(parents=True, exist_ok=True)
    rl_checkpoint_file = rl_checkpoint_path / "model_checkpoint_rl.pt"
    
    # Load base model for RL fine-tuning
    if config["rl_load_from_checkpoint"] is not None:
        rl_load_path = Path(config["rl_load_from_checkpoint"])
        print(f"Loading checkpoint for RL from {rl_load_path}")
        checkpoint = torch.load(rl_load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Base model loaded for RL fine-tuning")
    else:
        print("Using current model state for RL fine-tuning")
    
    # Create separate optimizer for RL
    rl_optimizer = torch.optim.AdamW(model.parameters(), lr=config["rl_lr"])
    
    # Learning rate scheduler for RL with warmup
    warmup_epochs = int(0.1 * config["rl_num_epochs"])  # 10% warmup
    
    # Warmup scheduler: linearly increase LR from 0 to target LR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        rl_optimizer,
        start_factor=0.01,  # Start from 1% of lr
        end_factor=1.0,     # End at 100% of lr
        total_iters=warmup_epochs
    )
    
    # After warmup, keep LR nearly constant with very gentle decay
    # Cosine annealing with eta_min=0.95 means LR only drops to 95% of target
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        rl_optimizer,
        T_max=config["rl_num_epochs"] - warmup_epochs,
        eta_min=config["rl_lr"] * 0.95  # Only decrease to 95% of target LR
    )
    
    # Chain warmup and gentle cosine annealing
    rl_scheduler = torch.optim.lr_scheduler.SequentialLR(
        rl_optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Tracking metrics for plotting
    rl_metrics_history = {
        'total_loss': [],
        'reinforce_loss': [],
        'ddpm_reg_loss': [],
        'reward_mean': [],
        'learning_rate': [],
    }
    
    # Add per-step-count tracking if joint training
    if config.get("rl_joint_training", False):
        step_counts = config.get("rl_step_counts", [2, 3, 4, 5])
        for num_steps in step_counts:
            rl_metrics_history[f'reward_mean_{num_steps}step'] = []
            rl_metrics_history[f'reinforce_loss_{num_steps}step'] = []
    
    # Helper function to generate aligned timesteps for joint training
    def get_timesteps_for_joint_training(n_timesteps, max_timesteps=5, num_train_steps=40):
        """
        Generate timesteps such that shorter trajectories use subsets of longer trajectory paths.
        This ensures 2-step, 3-step, 4-step all follow the same denoising path as 5-step.
        
        Args:
            n_timesteps: Number of timesteps to generate
            max_timesteps: Maximum number of inference steps (default 5)
            num_train_steps: Total training timesteps (default 40)
        
        Returns:
            torch.Tensor of timesteps
        """
        step_ratio = num_train_steps // max_timesteps
        # Generate full timestep schedule for max_timesteps
        timesteps_full = (np.arange(0, max_timesteps) * step_ratio).round()[::-1].copy().astype(np.int64)
        
        # Sample equally spaced indices to get subset for n_timesteps
        indices = np.linspace(0, len(timesteps_full) - 1, n_timesteps).round().astype(np.int64)
        timesteps = timesteps_full[indices]
        
        return torch.tensor(timesteps, device=device)
    
    print(f"Starting RL training for {config['rl_num_epochs']} epochs...")
    if config.get("rl_joint_training", False):
        step_counts = config.get("rl_step_counts", [2, 3, 4, 5])
        print(f"Joint Training Mode: Step counts {step_counts}")
        print(f"Batch size: {config['rl_batch_size']} (split into {len(step_counts)} groups of {config['rl_batch_size']//len(step_counts)} each)")
        print(f"Aligned timestep paths: shorter trajectories use subsets of {max(step_counts)}-step path")
    else:
        print(f"Batch size: {config['rl_batch_size']}, Inference steps: {config['rl_num_inference_steps']}")
    print(f"DDPM regularization weight: {config['rl_ddpm_reg_weight']}")
    print(f"LR Scheduler: Warmup ({warmup_epochs} epochs) → Gentle decay (1e-6 → {config['rl_lr'] * 0.95:.2e})")
    
    for rl_epoch in range(config["rl_num_epochs"]):
        model.train() # Note: it was there when the rl worked but i think this is useless #Note: seems to work without it as well, but for some reason we must have train() in ppo to work.

        
        # ============ REINFORCE Loss ============
        if config.get("rl_joint_training", False):
            # ========== PHASE 1: Multi-Step Sampling ==========
            step_counts = config.get("rl_step_counts", [2, 3, 4, 5])
            step_loss_weights = config.get("rl_step_loss_weights", [1.0] * len(step_counts))
            samples_per_group = config["rl_batch_size"] // len(step_counts)
            
            all_trajectories = []
            all_log_probs = []
            all_final_samples = []
            
            for num_steps in step_counts:
                # Generate aligned timesteps for this step count
                timesteps = get_timesteps_for_joint_training(
                    n_timesteps=num_steps,
                    max_timesteps=max(step_counts),
                    num_train_steps=config["num_train_steps"]
                )
                
                trajectories, log_probs = ddim_sample_with_log_probs(
                    model, 
                    noise_scheduler, 
                    n_samples=samples_per_group,
                    num_steps=num_steps,
                    timesteps=timesteps
                )
                all_trajectories.append(trajectories)
                all_log_probs.append(log_probs)
                all_final_samples.append(trajectories[:, -1])
            
            # ========== PHASE 2: Unified Reward Computation ==========
            # Concatenate all final samples
            all_final_samples_concat = torch.cat(all_final_samples, dim=0)
            
            # Compute rewards for all samples at once
            rewards_all = compute_geometric_reward(
                all_final_samples_concat, V1, V2, X_MIN, X_MAX, 
                reward_type=config.get("reward_type", "rectangle")
            )
            
            # Normalize advantages globally (shared statistics across all step counts)
            reward_mean = rewards_all.mean()
            reward_std = rewards_all.std() + 1e-8
            advantages_all = (rewards_all - reward_mean) / reward_std
            advantages_all = torch.clamp(
                advantages_all,
                min=-config["rl_advantage_max"],
                max=config["rl_advantage_max"]
            )
            
            # Split advantages and rewards back to match each trajectory group
            advantages_split = torch.split(advantages_all, samples_per_group)
            rewards_split = torch.split(rewards_all, samples_per_group)
            
            # ========== PHASE 3: Per-Group Loss Computation ==========
            reinforce_loss = 0.0
            per_step_losses = []
            per_step_rewards = []
            
            for i, (log_probs, advantages, step_rewards, weight, num_steps) in enumerate(
                zip(all_log_probs, advantages_split, rewards_split, step_loss_weights, step_counts)
            ):
                # REINFORCE loss for this group: -mean(sum(log_probs) * advantages)
                group_loss = -torch.mean(torch.sum(log_probs, dim=1) * advantages)
                reinforce_loss += weight * group_loss
                
                # Track per-step metrics
                per_step_losses.append(group_loss.item())
                per_step_rewards.append(step_rewards.mean().item())
            
            # Store for logging
            rewards = rewards_all  # For overall reward tracking
            
        else:
            # Original single-step training
            trajectories, trajectories_log_probs = ddim_sample_with_log_probs(
                model, 
                noise_scheduler, 
                n_samples=config["rl_batch_size"],
                num_steps=config["rl_num_inference_steps"]
            )
            
            # Compute rewards (per-sample)
            x0 = trajectories[:, -1]
            rewards = compute_geometric_reward(x0, V1, V2, X_MIN, X_MAX, reward_type=config.get("reward_type", "rectangle"))
            
            # Calculate advantages
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8
            advantages = (rewards - reward_mean) / reward_std
            advantages = torch.clamp(
                advantages,
                min=-config["rl_advantage_max"],
                max=config["rl_advantage_max"]
            )
            
            # REINFORCE loss: -log_prob * advantage
            reinforce_loss = -torch.mean(torch.sum(trajectories_log_probs, dim=1) * advantages)
        # ========== PHASE 4: DDPM Regularization + Single Backward Pass ==========
        if config["rl_ddpm_reg_weight"] > 0.0:
            # ============ DDPM Regularization Loss ============
            # Sample random data points and timesteps for standard DDPM training
            reg_batch_size = config["rl_batch_size"]
            indices = torch.randint(0, len(X_train), (reg_batch_size,))
            reg_batch_x = X_train[indices]
            reg_noise = torch.randn_like(reg_batch_x)
            reg_timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (reg_batch_size,), device=device
            ).long()
            
            reg_noisy_x = noise_scheduler.add_noise(reg_batch_x, reg_noise, reg_timesteps)
            reg_noise_pred = model(reg_noisy_x, reg_timesteps)
            ddpm_reg_loss = F.mse_loss(reg_noise_pred, reg_noise)
            
            # ============ Combined Loss ============
        else:
            ddpm_reg_loss = torch.tensor(0.0, device=device)
        total_loss = reinforce_loss + config["rl_ddpm_reg_weight"] * ddpm_reg_loss
        
        # Single backward pass (gradients flow through all trajectory groups)
        rl_optimizer.zero_grad()
        total_loss.backward()
        rl_optimizer.step()
        rl_scheduler.step()
        
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
        #     else:
        #         print(f"{name}: NO GRADIENT!")
                
        # print(f"Reward stats: min={rewards.min():.3f}, max={rewards.max():.3f}, "
        # f"positive_ratio={(rewards > 0).float().mean():.3f}")
        
        # Track metrics
        current_lr = rl_scheduler.get_last_lr()[0]
        rl_metrics_history['total_loss'].append(total_loss.item())
        rl_metrics_history['reinforce_loss'].append(reinforce_loss.item())
        rl_metrics_history['ddpm_reg_loss'].append(ddpm_reg_loss.item())
        rl_metrics_history['reward_mean'].append(rewards.mean().item())
        rl_metrics_history['learning_rate'].append(current_lr)
        
        # Track per-step metrics if joint training
        if config.get("rl_joint_training", False):
            for i, (num_steps, step_loss, step_reward) in enumerate(
                zip(step_counts, per_step_losses, per_step_rewards)
            ):
                rl_metrics_history[f'reinforce_loss_{num_steps}step'].append(step_loss)
                rl_metrics_history[f'reward_mean_{num_steps}step'].append(step_reward)
        
        # Logging
        if rl_epoch % 5 == 0 or rl_epoch == config["rl_num_epochs"] - 1:
            with torch.no_grad():
                log_msg = (f"RL Epoch {rl_epoch:03d} | "
                          f"Total Loss: {total_loss.item():.6f} | "
                          f"REINFORCE: {reinforce_loss.item():.6f} | "
                          f"DDPM Reg: {ddpm_reg_loss.item():.6f} | "
                          f"Reward: {rewards.mean().item():.4f} | "
                          f"LR: {current_lr:.2e}")
                
                # Add per-step info if joint training
                if config.get("rl_joint_training", False):
                    step_info = " | Steps: " + ", ".join(
                        f"{num_steps}={step_reward:.3f}" 
                        for num_steps, step_reward in zip(step_counts, per_step_rewards)
                    )
                    log_msg += step_info
                
                print(log_msg)
        
        # Save checkpoint periodically
        if (rl_epoch + 1) % config["save_every"] == 0 or rl_epoch == config["rl_num_epochs"] - 1:
            checkpoint_save_path = rl_checkpoint_path / f"model_checkpoint_rl_epoch_{rl_epoch+1}.pt"
            torch.save({
                'epoch': rl_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': rl_optimizer.state_dict(),
                'scheduler_state_dict': rl_scheduler.state_dict(),
                'metrics_history': rl_metrics_history,
                'config': config,
            }, checkpoint_save_path)
            print(f"  → Checkpoint saved: {checkpoint_save_path.name}")
    
    # Plot training curves
    print("\nPlotting RL training curves...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('RL Training Metrics', fontsize=16, fontweight='bold')
    
    epochs_range = range(len(rl_metrics_history['total_loss']))
    
    # Total Loss
    axes[0, 0].plot(epochs_range, rl_metrics_history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # REINFORCE Loss
    axes[0, 1].plot(epochs_range, rl_metrics_history['reinforce_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('REINFORCE Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # DDPM Regularization Loss
    axes[0, 2].plot(epochs_range, rl_metrics_history['ddpm_reg_loss'], 'g-', linewidth=2)
    axes[0, 2].set_title('DDPM Regularization Loss', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Reward Mean
    axes[1, 0].plot(epochs_range, rl_metrics_history['reward_mean'], 'purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Mean Reward', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs_range, rl_metrics_history['learning_rate'], 'orange', linewidth=2)
    axes[1, 1].set_title('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Combined view: Losses
    axes[1, 2].plot(epochs_range, rl_metrics_history['total_loss'], 'b-', linewidth=2, label='Total', alpha=0.7)
    axes[1, 2].plot(epochs_range, rl_metrics_history['reinforce_loss'], 'r-', linewidth=2, label='REINFORCE', alpha=0.7)
    axes[1, 2].plot(epochs_range, rl_metrics_history['ddpm_reg_loss'], 'g-', linewidth=2, label='DDPM Reg', alpha=0.7)
    axes[1, 2].set_title('All Losses Combined', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend(loc='best', fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    training_curves_path = rl_checkpoint_path / "rl_training_curves.png"
    plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
    print(f"✓ RL training curves saved to {training_curves_path}")
    plt.close()
    
    # Save RL checkpoint
    print(f"\nSaving RL checkpoint to {rl_checkpoint_file}")
    torch.save({
        'epoch': config['rl_num_epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': rl_optimizer.state_dict(),
        'scheduler_state_dict': rl_scheduler.state_dict(),
        'metrics_history': rl_metrics_history,
        'config': config,
    }, rl_checkpoint_file)
    print("RL Checkpoint saved.")
    
    # Evaluate RL model
    print("\n" + "="*60)
    print("Evaluating RL Fine-tuned Model")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        rl_samples = ddim_sample_with_diffusers(
            model, noise_scheduler,
            n_samples=10000,
            num_steps=config["rl_num_inference_steps"]
        )
    
    rl_metrics = evaluate_samples(X_train[:10000], rl_samples)
    print(f"RL Model - Frechet Distance: {rl_metrics['frechet_distance']:.6f}")
    print(f"RL Model - Coverage: {rl_metrics['coverage']:.4f}")
    print(f"RL Model - MMD: {rl_metrics['mmd']:.6f}")
    print(f"RL Model - Mean Reward: {rl_metrics['rewards_mean']:.4f}")
    print(f"RL Model - Rectangle Coverage: {rl_metrics['rectangle_coverage']:.4f}")
    
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    orig_data = X_train[:5000].cpu().numpy()
    plt.scatter(orig_data[:, 0], orig_data[:, 1], s=1, alpha=0.5, c='blue')
    plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1],
             'k-', linewidth=2, label='Pretrained Manifold')
    plt.title("Ground Truth")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(rl_samples[:5000, 0], rl_samples[:5000, 1], s=1, alpha=0.5, c='green')
    plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1],
             'k-', linewidth=2, label='Pretrained Manifold')
    
    # Plot reward shape based on reward_type
    if config.get("reward_type", "rectangle") == "circles":
        for i, (center, radius) in enumerate(zip(CIRCLE_CENTERS, CIRCLE_RADII)):
            circle = plt.Circle(center, radius, fill=False, color='orange', linewidth=2, 
                              label='RL reward Manifold' if i == 0 else '')
            plt.gca().add_patch(circle)
    elif config.get("reward_type", "rectangle") == "ring":
        # Draw parallelogram in orange to show the ring is around it
        plt.plot(PARALLELOGRAM_CORNERS[:, 0], PARALLELOGRAM_CORNERS[:, 1],
                 '-', color='orange', linewidth=3, label='RL reward Manifold (ring around this)')
    else:
        plt.plot(rectangle_corners[:, 0], rectangle_corners[:, 1], '-', color='orange', linewidth=2, label='RL reward Manifold')
    
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    rl_viz_path = rl_checkpoint_path / "rl_result.png"
    plt.savefig(rl_viz_path)
    print(f"\n✓ RL visualization saved to {rl_viz_path}")
    
    print("="*60)
    
    

    print(f"Total time taken for RL: {time.time() - start_time:.2f} seconds")