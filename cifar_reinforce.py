import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, DDIMScheduler
import math
from pathlib import Path
import pandas as pd
import time
from typing import Optional, Tuple
from diffusers.utils.torch_utils import randn_tensor
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# ==========================================
# Configuration
# ==========================================

name = "cifar_dit_jpeg_reward"

config = {
    # Base training config
    "load": None,  # Change to path if loading a pretrained model
    "noise_scheduler": "ddim", 
    "num_train_steps": 1000,
    "num_ddim_inference_steps": 50,
    "checkpoint_dir": f"outputs/{name}/ckpt",
    "skip_training_if_ckpt_exists": True,
    "run_eval": True,
    
    # RL Fine-tuning config
    "run_rl": True,
    "rl_checkpoint_dir": f"outputs/{name}/ckpt_rl",
    "rl_load_from_checkpoint": None,
    "rl_num_epochs": 100,
    "save_every": 10,
    "rl_batch_size": 32, # Batch size for RL
    "rl_num_inference_steps": 10, # Reduced steps for RL efficiency
    "rl_lr": 1e-5,
    "rl_ddpm_reg_weight": 0.01,
    "rl_advantage_max": 5.0,
    
    # Model Config
    "image_size": 32,
    "patch_size": 2, # Reduced from 4 to 2 to improve quality
    "hidden_dim": 256,
    "num_layers": 6,
    "num_heads": 4,
    "dropout": 0.1,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. Diffusion Transformer Architecture (DiT)
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

class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm (conditioning on time) could be added,
    but here we use simple addition of time embeddings + standard Transformer Encoder Layer
    for simplicity and stability."""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, SeqLen, Dim)
        # Self-Attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class DiffusionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # 1. Patch Embedding
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        
        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        
        # 3. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 4. Transformer Backbone
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        
        # 5. Output Projection
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.patch_dim)

    def forward(self, x, t):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Patchify: (B, C, H, W) -> (B, NumPatches, PatchDim)
        # Reshape to (B, C, H/P, P, W/P, P)
        p = self.patch_size
        x = x.view(B, C, H//p, p, W//p, p)
        # Permute to (B, H/P, W/P, C, P, P) then flatten to (B, NumPatches, PatchDim)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1, self.patch_dim)
        
        # Embed patches
        x = self.patch_embed(x) # (B, NumPatches, HiddenDim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Add time embedding
        t_emb = self.time_mlp(t) # (B, HiddenDim)
        x = x + t_emb.unsqueeze(1) # Broadcast time to all tokens
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        # Final Norm
        x = self.norm_final(x)
        
        # Output project
        x = self.output_proj(x) # (B, NumPatches, PatchDim)
        
        # Unpatchify: (B, NumPatches, PatchDim) -> (B, C, H, W)
        x = x.view(B, H//p, W//p, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        
        return x

# ==========================================
# 2. Data Preparation (CIFAR-10)
# ==========================================

print("Loading CIFAR-10 dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map [0, 1] to [-1, 1]
])

# Download and load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

print(f"Dataset loaded. {len(trainset)} training images.")

# ==========================================
# 3. Reward Function (JPEG Incompressibility)
# ==========================================

from scipy.linalg import sqrtm

class FIDMetric:
    def __init__(self, device):
        self.device = device
        # Load InceptionV3
        # Transform input: 299x299, normalize
        self.inception = torchvision.models.inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device)
        self.inception.eval()
        
        # Hook to get features from the pool3 layer (2048 dim)
        self.features = []
        def hook(module, input, output):
            self.features.append(output.view(output.size(0), -1))
        
        # In torchvision inception_v3, Mixed_7c is the last block before pool
        # But we usually want the average pooling layer output. 
        # The model.avgpool is applied after Mixed_7c.
        self.inception.avgpool.register_forward_hook(hook)
        
        # Pre-computed statistics for real data
        self.mu_real = None
        self.sigma_real = None
        
    @torch.no_grad()
    def get_features(self, loader_or_tensor, max_samples=2000, batch_size=32):
        self.features = []
        all_feats = []
        count = 0
        
        if isinstance(loader_or_tensor, DataLoader):
            for batch, _ in loader_or_tensor:
                batch = batch.to(self.device)
                
                # Process batch in smaller chunks to avoid OOM
                for i in range(0, batch.shape[0], batch_size):
                    sub_batch = batch[i:i+batch_size]
                    
                    # Resize to 299x299 for Inception
                    sub_batch = F.interpolate(sub_batch, size=(299, 299), mode='bilinear', align_corners=False)
                    # Map [-1, 1] to ImageNet normalization
                    sub_batch = (sub_batch + 1) / 2
                    sub_batch = transforms.functional.normalize(
                        sub_batch, 
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    )
                    
                    self.inception(sub_batch)
                    all_feats.append(self.features[-1])
                    self.features = [] # Clear hook format
                
                count += batch.shape[0]
                if count >= max_samples:
                    break
        else:
            # Tensor case
            tensor_data = loader_or_tensor
            for i in range(0, tensor_data.shape[0], batch_size):
                batch = tensor_data[i:i+batch_size].to(self.device)
                
                # Resize
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                batch = (batch + 1) / 2
                batch = transforms.functional.normalize(
                    batch, 
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
                self.inception(batch)
                all_feats.append(self.features[-1])
                self.features = []
            
        all_feats = torch.cat(all_feats, dim=0)[:max_samples]
        return all_feats.cpu().numpy()

    def compute_stats(self, features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def precompute_real_stats(self, loader, max_samples=1000):
        print(f"Computing FID stats for real data ({max_samples} samples)...")
        feats = self.get_features(loader, max_samples)
        self.mu_real, self.sigma_real = self.compute_stats(feats)
        print("Done.")

    def compute_fid(self, gen_samples):
        """
        gen_samples: (B, C, H, W) tensor in [-1, 1]
        """
        feats_gen = self.get_features(gen_samples, max_samples=len(gen_samples))
        mu_gen, sigma_gen = self.compute_stats(feats_gen)
        
        return self._calculate_frechet_distance(self.mu_real, self.sigma_real, mu_gen, sigma_gen)

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produced singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

fid_scorer = FIDMetric(device)

def jpeg_incompressibility():
    def _fn(images):
        # images: (B, C, H, W) tensor in [-1, 1]
        
        # Denormalize to [0, 255] for PIL
        # [-1, 1] -> [0, 1] -> [0, 255]
        images = (images + 1) / 2
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        
        # NCHW -> NHWC
        if images.ndim == 4:
            images = images.transpose(0, 2, 3, 1) 
        
        # images is now (B, H, W, C) numpy array of uint8
        
        pil_images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in pil_images]
        for image, buffer in zip(pil_images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        
        # Calculate size in KB
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return torch.tensor(sizes, dtype=torch.float32), {}

    return _fn

reward_fn = jpeg_incompressibility()

def compute_reward(samples):
    """
    Compute reward for a batch of samples.
    Args:
        samples: (B, C, H, W) tensor, values approx in [-1, 1]
    Returns:
        rewards: (B,) tensor
    """
    rewards, _ = reward_fn(samples)
    return rewards.to(samples.device)

# ==========================================
# 4. Sampling Helper (with LogProbs)
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
    Adapted from diffusers DDIM scheduler to return log probability.
    """
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
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(f"prediction_type {scheduler.config.prediction_type} not supported")
    
    # 4. Clip
    if scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )
    
    # 5. Compute variance: "sigma_t(η)"
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    
    # 6. Compute "direction pointing to x_t"
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
    
    # 7. Compute x_{t-1} mean
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
    
    # 9. Compute log probability
    # PDF of Normal(mean, std): 
    # log_p = -0.5 * ((x - mean)/std)^2 - log(std) - 0.5 * log(2pi)
    std_dev_t = torch.clamp(std_dev_t, min=1e-6)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - math.log(math.sqrt(2 * math.pi))
    )
    
    # Sum over all dimensions except batch (C, H, W)
    log_prob = log_prob.sum(dim=(1, 2, 3))
    
    return prev_sample, log_prob

def ddim_sample_with_log_probs(model, scheduler, n_samples=32, num_steps=20, eta=1.0):
    model.eval()
    
    # Start from random noise
    x = torch.randn(
        n_samples, 
        config['in_channels'] if 'in_channels' in config else 3, 
        config['image_size'], 
        config['image_size'], 
        device=device
    )
    
    scheduler.set_timesteps(num_steps)
    sum_log_probs = torch.zeros(n_samples, device=device)
    
    for t in scheduler.timesteps:
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        model_output = model(x, t_batch)
        
        # Step
        xt, log_prob = ddim_step_with_logprob(
            scheduler=scheduler,
            model_output=model_output,
            timestep=t,
            sample=x,
            eta=eta,
        )
        
        sum_log_probs += log_prob
        x = xt.detach() # Detach to prevent graph exploration
        
    return x, sum_log_probs

@torch.no_grad()
def simple_sample(model, scheduler, n_samples=16, num_steps=20):
    model.eval()
    x = torch.randn(n_samples, 3, config['image_size'], config['image_size'], device=device)
    scheduler.set_timesteps(num_steps)
    
    for t in scheduler.timesteps:
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        model_output = model(x, t_batch)
        step_output = scheduler.step(model_output, t, x, eta=1.0)
        x = step_output.prev_sample
        
    return x

# ==========================================
# 5. Helper: Save Image Grid
# ==========================================

def save_image_grid(images, path, title="Generated Images"):
    """
    images: (B, C, H, W) tensor in [-1, 1]
    """
    # Denormalize to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Make grid
    grid = torchvision.utils.make_grid(images, nrow=4, padding=2)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis('off')
    plt.savefig(path)
    plt.close()

# ==========================================
# 6. Main execution
# ==========================================

# Init Model
model = DiffusionTransformer(
    image_size=config['image_size'],
    patch_size=config['patch_size'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    dropout=config['dropout']
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Load Checkpoint if exists
checkpoint_path = Path(config["checkpoint_dir"])
checkpoint_path.mkdir(parents=True, exist_ok=True)
checkpoint_file = checkpoint_path / "model_checkpoint.pt"

start_epoch = 0
if config["load"] is not None:
    if Path(config["load"]).exists():
        print(f"Loading from {config['load']}")
        ckpt = torch.load(config["load"], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
elif checkpoint_file.exists() and config["skip_training_if_ckpt_exists"]:
    print(f"Loading checkpoint {checkpoint_file}")
    ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    print(f"Resuming from epoch {start_epoch}")

# Precompute FID stats
print("Precomputing FID statistics for real data...")
fid_scorer.precompute_real_stats(trainloader, max_samples=1000)

# Pre-training Loop (if needed)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=config["num_train_steps"],
    beta_schedule="linear",
    prediction_type="epsilon"
)

pretrain_metrics = {'loss': [], 'fid': []}

num_pretrain_epochs = 500 # Increased from 10 to 50
if start_epoch < num_pretrain_epochs:
    print("Starting Pre-training...")
    for epoch in range(start_epoch, num_pretrain_epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (images, _) in enumerate(trainloader):
            images = images.to(device)
            B = images.shape[0]
            
            # Sample noise
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
            
            # Add noise
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Predict
            noise_pred = model(noisy_images, timesteps)
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")
        
        # Save checkpoint
        avg_loss = epoch_loss/len(trainloader)
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
        
        # Compute FID
        print("Computing FID...")
        with torch.no_grad():
            samples = simple_sample(model, DDIMScheduler.from_config(noise_scheduler.config), n_samples=500, num_steps=20)
            fid = fid_scorer.compute_fid(samples)
            print(f"Epoch {epoch} | FID: {fid:.4f}")
            
        pretrain_metrics['loss'].append(avg_loss)
        pretrain_metrics['fid'].append(fid)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': pretrain_metrics,
        }, checkpoint_file)
        
        # Validate purely visually
        save_image_grid(samples[:16], checkpoint_path / f"epoch_{epoch}_samples.png", f"Epoch {epoch} (FID: {fid:.2f})")

    # Plot Pre-training Curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(pretrain_metrics['loss'])
    plt.title('Pre-training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(pretrain_metrics['fid'])
    plt.title('Pre-training FID')
    plt.savefig(checkpoint_path / 'pretrain_curves.png')
    plt.close()

# RL Loop
if config["run_rl"]:
    print("\n" + "="*60)
    print("Starting RL Fine-tuning (JPEG Reward)")
    print("="*60)
    
    rl_checkpoint_path = Path(config["rl_checkpoint_dir"])
    rl_checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Use scheduler for RL sampling (DDIM)
    rl_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    
    rl_optimizer = torch.optim.AdamW(model.parameters(), lr=config["rl_lr"])
    
    rl_metrics = {'reward': [], 'loss': [], 'fid': []}
    
    data_iter = iter(trainloader)

    for epoch in range(config["rl_num_epochs"]):
        model.eval() # Gradients required for params, but we can set eval mode for layers
        
        # 1. Sample with log probs
        samples, sum_log_probs = ddim_sample_with_log_probs(
            model, rl_scheduler, 
            n_samples=config["rl_batch_size"], 
            num_steps=config["rl_num_inference_steps"]
        )
        
        # 2. Compute Rewards
        rewards = compute_reward(samples)
        
        # 3. Advantages
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        advantages = (rewards - reward_mean) / reward_std
        advantages = torch.clamp(advantages, -config["rl_advantage_max"], config["rl_advantage_max"])
        
        # 4. REINFORCE Loss
        reinforce_loss = -torch.mean(sum_log_probs * advantages)
        
        # 5. DDPM Regularization (Optional, prevents catastrophic forgetting)
        ddpm_loss = 0.0
        if config["rl_ddpm_reg_weight"] > 0:
            # Grab a batch of real data
            try:
                real_batch, _ = next(data_iter)
            except:
                data_iter = iter(trainloader)
                real_batch, _ = next(data_iter)
            
            real_batch = real_batch.to(device)
            B_reg = real_batch.shape[0]
            
            noise = torch.randn_like(real_batch)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B_reg,), device=device).long()
            noisy_x = noise_scheduler.add_noise(real_batch, noise, timesteps)
            pred_noise = model(noisy_x, timesteps)
            ddpm_loss = F.mse_loss(pred_noise, noise)
        
        total_loss = reinforce_loss + config["rl_ddpm_reg_weight"] * ddpm_loss
        
        # Update
        rl_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        rl_optimizer.step()
        
        # Log
        rl_metrics['reward'].append(reward_mean.item())
        rl_metrics['loss'].append(total_loss.item())
        
        # Compute FID periodically
        fid = 0.0
        if epoch % 5 == 0:
             with torch.no_grad():
                # We can use the samples we just generated if batch size is large enough, 
                # but RL batch size is small (32). Let's generate a dedicated batch for FID.
                fid_samples = simple_sample(model, rl_scheduler, n_samples=500, num_steps=20)
                fid = fid_scorer.compute_fid(fid_samples)
        
        rl_metrics['fid'].append(fid) # Note: this might be 0.0 for non-eval epochs, handling in plot
        
        print(f"RL Epoch {epoch:03d} | Reward: {reward_mean.item():.4f} | "
              f"Loss: {total_loss.item():.4f} | FID: {fid:.4f}")
        
        if (epoch + 1) % config["save_every"] == 0:
            # Save checkpoint
            torch.save(model.state_dict(), rl_checkpoint_path / f"rl_model_epoch_{epoch}.pt")
            
            # Save visualization
            save_image_grid(samples[:16], rl_checkpoint_path / f"rl_epoch_{epoch}_samples.png", 
                           title=f"RL Epoch {epoch} (R={reward_mean.item():.2f}, FID={fid:.2f})")
            
            # Plot RL Curves
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(rl_metrics['reward'])
            plt.title('RL Reward')
            plt.subplot(1, 3, 2)
            plt.plot(rl_metrics['loss'])
            plt.title('RL Loss')
            plt.subplot(1, 3, 3)
            # Filter non-zero FIDs
            fids = [f for f in rl_metrics['fid'] if f > 0]
            if fids:
                plt.plot(range(0, len(fids)*5, 5), fids)
            plt.title('RL FID (every 5 epochs)')
            plt.savefig(rl_checkpoint_path / 'rl_training_curves.png')
            plt.close()
            
    print("RL Fine-tuning Complete.")