import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
import math
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm
import pandas as pd
import time

config = {
    "noise_scheduler": "ddim", #or ddpm
    "num_train_steps": 40,
    "num_ddim_inference_steps": 10,
    "checkpoint_dir": "tmp/ckpt_toy",
    "skip_training_if_ckpt_exists": True,
    "run_eval": True,  # Set to True to run comprehensive evaluation
}
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
        points: Normalized points in [-1, 1] range (N, 2)
        v1, v2: Basis vectors of the parallelogram
        x_min, x_max: Min/max values used for normalization
        
    Returns:
        Boolean array indicating if each point is inside the parallelogram
    """
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

def compute_out_of_manifold_ratio(real_samples, generated_samples, threshold=0.1, parallelogram_params=None):
    """
    Compute the ratio of generated samples that are outside the data manifold.
    
    Uses two criteria:
    1. Distance-based: Points that are too far from any real sample
    2. Geometric (if parallelogram_params provided): Points outside the actual parallelogram
    
    Returns the ratio of out-of-manifold points.
    """
    # Distance-based criterion: too far from any real point
    distances = cdist(generated_samples, real_samples, metric='euclidean')
    min_dist_to_real = distances.min(axis=1)
    out_of_manifold_distance = (min_dist_to_real > threshold).sum()
    
    # Geometric parallelogram criterion
    if parallelogram_params is not None:
        v1, v2, x_min, x_max = parallelogram_params
        in_parallelogram = check_points_in_parallelogram(
            generated_samples, v1, v2, x_min, x_max
        )
        out_of_manifold_geom = (~in_parallelogram).sum()
        ratio_geom = out_of_manifold_geom / len(generated_samples)
    else:
        out_of_manifold_geom = 0
        ratio_geom = 0.0
    
    total_generated = len(generated_samples)
    ratio_distance = out_of_manifold_distance / total_generated
    
    return {
        'out_of_manifold_ratio_distance': ratio_distance,
        'out_of_manifold_ratio_geometric': ratio_geom,
        'out_of_manifold_count_distance': out_of_manifold_distance,
        'out_of_manifold_count_geometric': out_of_manifold_geom,
    }

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
    
    # Out-of-manifold ratio
    oom_metrics = compute_out_of_manifold_ratio(
        real_samples[:n_eval],
        generated_samples[:n_eval],
        threshold=0.1,
        parallelogram_params=(V1, V2, X_MIN, X_MAX)
    )
    
    return {
        'frechet_distance': fd,
        'coverage': coverage,
        'mmd': mmd,
        'out_of_manifold_ratio': oom_metrics['out_of_manifold_ratio_distance'],
        'out_of_manifold_geometric_ratio': oom_metrics['out_of_manifold_ratio_geometric'],
    }

# ==========================================
# 3. Setup Diffusers Scheduler & Training
# ==========================================

# NOTE: beta_schedule="squaredcos_cap_v2" is vastly superior for 
# 2D geometric shapes compared to "linear"

noise_scheduler = DDPMScheduler(
    num_train_timesteps=config["num_train_steps"],
    beta_schedule="squaredcos_cap_v2",
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
        step_output = scheduler.step(model_output, t, x)
        x = step_output.prev_sample
        
    return x.detach().cpu().numpy()

# ==========================================
# Comprehensive Evaluation
# ==========================================

if config["run_eval"]:
    print("\n" + "="*60)
    print("Running Comprehensive Evaluation")
    print("="*60)
    
    # Define evaluation configurations
    eval_configs = [
        {"scheduler": "ddpm", "steps": 10, "name": "DDPM-10"},
        {"scheduler": "ddpm", "steps": 20, "name": "DDPM-20"},
        {"scheduler": "ddpm", "steps": 40, "name": "DDPM-40"},
        {"scheduler": "ddim", "steps": 5, "name": "DDIM-5"},
        {"scheduler": "ddim", "steps": 10, "name": "DDIM-10"},
        {"scheduler": "ddim", "steps": 20, "name": "DDIM-20"},
        {"scheduler": "ddim", "steps": 40, "name": "DDIM-40"},
    ]
    
    results = []
    all_samples = {}
    
    for eval_cfg in eval_configs:
        print(f"\nEvaluating {eval_cfg['name']}...")
        
        start_time = time.time()
        
        if eval_cfg['scheduler'] == 'ddpm':
            samples = ddpm_sample_with_diffusers(
                model, noise_scheduler, 
                n_samples=10000, 
                num_steps=eval_cfg['steps']
            )
        else:  # ddim
            samples = ddim_sample_with_diffusers(
                model, noise_scheduler, 
                n_samples=10000, 
                num_steps=eval_cfg['steps']
            )
        
        sampling_time = time.time() - start_time
        
        # Compute metrics
        metrics = evaluate_samples(X_train[:10000], samples)
        
        # Store results
        results.append({
            'scheduler': eval_cfg['scheduler'].upper(),
            'num_steps': eval_cfg['steps'],
            'name': eval_cfg['name'],
            'frechet_distance': metrics['frechet_distance'],
            'coverage': metrics['coverage'],
            'mmd': metrics['mmd'],
            'outside_parallelogram_%': metrics['out_of_manifold_geometric_ratio'] * 100,
            'sampling_time_sec': sampling_time,
        })
        
        # Store samples for visualization
        all_samples[eval_cfg['name']] = samples
        
        print(f"  FD: {metrics['frechet_distance']:.6f} | "
              f"Coverage: {metrics['coverage']:.4f} | "
              f"MMD: {metrics['mmd']:.6f} | "
              f"Outside Parallelogram: {metrics['out_of_manifold_geometric_ratio']*100:.2f}% | "
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
        
        # Get metrics for title
        row = df_results[df_results['name'] == eval_cfg['name']].iloc[0]
        title = f"{eval_cfg['name']}\n"
        title += f"FD: {row['frechet_distance']:.3f} | Cov: {row['coverage']:.3f}\n"
        title += f"Outside Parallelogram: {row['outside_parallelogram_%']:.2f}%\n"
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
        print(f"Using DDIM sampling with {config['num_ddim_inference_steps']} steps.")
        samples = ddim_sample_with_diffusers(model, noise_scheduler)
    else:
        samples = ddpm_sample_with_diffusers(model, noise_scheduler)
    
    # Quick evaluation
    print("\nEvaluating generated samples...")
    metrics = evaluate_samples(X_train[:10000], samples[:10000])
    print(f"Frechet Distance: {metrics['frechet_distance']:.6f}")
    print(f"Coverage (threshold=0.05): {metrics['coverage']:.4f}")
    print(f"MMD (Mean Min Distance): {metrics['mmd']:.6f}")
    print(f"Outside Parallelogram: {metrics['out_of_manifold_geometric_ratio']*100:.2f}%")

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

print(f"\nCheckpoint saved to {checkpoint_file}")
