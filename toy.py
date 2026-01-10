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

name = "only_2_steps"

config = {

    # "load": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/toy_parallelogram/outputs/only_3_steps/ckpt_toy_rl/model_checkpoint_rl_epoch_4000.pt",
    "load": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/toy_parallelogram/outputs/only_2_steps/ckpt_toy_rl/model_checkpoint_rl_epoch_20000.pt",
    # "load": None,
    "noise_scheduler": "ddim", #or ddpm
    "num_train_steps": 40,
    "num_ddim_inference_steps": 10,
    "checkpoint_dir": f"outputs/{name}/ckpt_toy",
    "skip_training_if_ckpt_exists": True,
    "run_eval": True,  # when set true ckpt even if trained on the fly is not saved (also load should be not none # Set to True to run comprehensive evaluation
    
    # RL Fine-tuning config
    "run_rl": False,  # Set to True to run RL fine-tuning
    "rl_checkpoint_dir": f"outputs/{name}/ckpt_toy_rl",
    "rl_load_from_checkpoint": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/toy_parallelogram/outputs/rect_pretraining_40_rl_40/ckpt_toy/model_checkpoint.pt",  # Path to checkpoint to fine-tune from (or None to use base checkpoint)
    "rl_num_epochs": 4000,
    "save_every": 500,
    "rl_batch_size": 512,
    "rl_num_inference_steps": 2, #can not have 5 for eval for some reason, # FIXME:
    "rl_lr": 1e-6,
    "rl_ddpm_reg_weight": 0.0,  # Weight for DDPM regularization loss
    "rl_advantage_max": 10.0,  # Clipping for advantages
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
RECT_BOUNDS = (-0.5, 0.5, -0.5, 0.5)

x_min_rect, x_max_rect, y_min_rect, y_max_rect = RECT_BOUNDS

rectangle_corners = np.array([
    [x_min_rect, y_min_rect],
    [x_max_rect, y_min_rect],
    [x_max_rect, y_max_rect],
    [x_min_rect, y_max_rect],
    [x_min_rect, y_min_rect]
])

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

def compute_geometric_reward(samples, v1, v2, x_min, x_max):
    rect_bounds = RECT_BOUNDS
    inside = check_points_in_rectangle(samples, rect_bounds)
    
    # Convert to torch tensor
    rewards = torch.tensor(inside, dtype=torch.float32, device=samples.device)
    rewards = 2.0 * rewards - 1.0  # Map True->1.0, False->-1.0
    
    return rewards


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
        torch.tensor(generated_samples[:n_eval], device=device), V1, V2, X_MIN, X_MAX
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
    model.train()
    
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


def ddim_sample_with_log_probs(model, scheduler, n_samples=256, num_steps=20, eta=1.0):
    """
    Sample using DDIM while tracking log probabilities for RL training.
    
    Args:
        model: Diffusion model
        scheduler: DDIMScheduler instance
        n_samples: Number of samples to generate
        num_steps: Number of denoising steps
        eta: Stochasticity parameter (0 = deterministic DDIM)
        
    Returns:
        samples: Final samples (n_samples, 2)
        log_probs: Sum of log probabilities across all timesteps (n_samples,)
    """
    model.eval()
    
    # 1. Start from random noise
    x = torch.randn(n_samples, 2, device=device)
    
    # 2. Initialize scheduler timesteps
    scheduler.set_timesteps(num_steps)
    
    # Track log probabilities
    trajectory = []
    trajectory_log_probs = []
    
    trajectory.append(x)
    # 3. Denoising Loop
    for t in scheduler.timesteps:
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
    #     {"scheduler": "ddim", "steps": config["rl_num_inference_steps"]//10, "name": f"DDIM-{config['rl_num_inference_steps']//10}"},
    #     {"scheduler": "ddim", "steps": config["rl_num_inference_steps"]//8, "name": f"DDIM-{config['rl_num_inference_steps']//8}"},
    #     {"scheduler": "ddim", "steps": config["rl_num_inference_steps"]//2, "name": f"DDIM-{config['rl_num_inference_steps']//2}"},
    #     {"scheduler": "ddim", "steps": config["rl_num_inference_steps"], "name": f"DDIM-{config['rl_num_inference_steps']}"},
    # ]
    eval_configs = [
        {"scheduler": "ddpm", "steps": 40, "name": f"DDPM-40"},
        {"scheduler": "ddim", "steps": 2, "name": f"DDIM-2"},
        {"scheduler": "ddim", "steps": 3, "name": f"DDIM-3"},
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
        
        plt.plot(rectangle_corners[:, 0], rectangle_corners[:, 1], '-', color='orange', linewidth=2, label='RL reward Manifold')
        
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
    
    print(f"Starting RL training for {config['rl_num_epochs']} epochs...")
    print(f"Batch size: {config['rl_batch_size']}, Inference steps: {config['rl_num_inference_steps']}")
    print(f"DDPM regularization weight: {config['rl_ddpm_reg_weight']}")
    print(f"LR Scheduler: Warmup ({warmup_epochs} epochs) → Gentle decay (1e-6 → {config['rl_lr'] * 0.95:.2e})")
    
    for rl_epoch in range(config["rl_num_epochs"]):
        # model.train() # Note: it was there when the rl worked but i think this is useless

        
        # ============ REINFORCE Loss ============
        # Generate trajectories with log probs
        trajectories, trajectories_log_probs = ddim_sample_with_log_probs(
            model, 
            noise_scheduler, 
            n_samples=config["rl_batch_size"],
            num_steps=config["rl_num_inference_steps"]
        )
        
        # Compute rewards (per-sample)
        x0 = trajectories[:, -1]
        rewards = compute_geometric_reward(x0, V1, V2, X_MIN, X_MAX)
        
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
        # REINFORCE loss
        reinforce_loss = -torch.mean(torch.sum(trajectories_log_probs, dim=1) * advantages)
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
        
        # Backprop and optimize
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
        rl_metrics_history['reward_mean'].append(reward_mean.item())
        rl_metrics_history['learning_rate'].append(current_lr)
        
        # Logging
        if rl_epoch % 5 == 0 or rl_epoch == config["rl_num_epochs"] - 1:
            with torch.no_grad():
                print(f"RL Epoch {rl_epoch:03d} | "
                      f"Total Loss: {total_loss.item():.6f} | "
                      f"REINFORCE: {reinforce_loss.item():.6f} | "
                      f"DDPM Reg: {ddpm_reg_loss.item():.6f} | "
                      f"Reward: {reward_mean.item():.4f} | "
                      f"LR: {current_lr:.2e}"
                )
        
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