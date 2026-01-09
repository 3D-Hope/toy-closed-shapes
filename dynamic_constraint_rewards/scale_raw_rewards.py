import json

import numpy as np
import torch


class RewardNormalizer:
    """
    Normalize each to [-1, 1] using robust min–max scaling using EMA-updated percentiles.
    """

    def __init__(
        self,
        baseline_stats_path: str,
        beta: float = 0.99,
        eps: float = 1e-8,
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
    ):
        """
        Robust per-reward normalizer with percentile-based scaling and EMA updates.

        Args:
            baseline_stats_path (str): Path to baseline stats JSON file.
            beta (float): EMA smoothing factor for updating min/max.
            eps (float): Small constant to avoid division by zero.
            low_percentile (float): Lower percentile for clipping outliers.
            high_percentile (float): Upper percentile for clipping outliers.
        """
        self.beta = beta
        self.eps = eps
        self.low_p = low_percentile
        self.high_p = high_percentile

        # Load baseline stats
        baseline_stats = json.load(open(baseline_stats_path))
        self.stats = {}
        for name, s in baseline_stats.items():
            # Initialize from robust range if available, else use min/max
            self.stats[name] = {
                "min": torch.tensor(
                    s.get("percentile_1", s["min"]), dtype=torch.float32
                ),
                "max": torch.tensor(
                    s.get("percentile_99", s["max"]), dtype=torch.float32
                ),
            }

    def normalize(self, name: str, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards for a specific reward component using robust min–max scaling.

        Args:
            name (str): Reward name key from stats.
            rewards (torch.Tensor): Tensor of shape [B, 1] or [B].

        Returns:
            torch.Tensor: Normalized rewards in range [-1, 1].
        """
        if name not in self.stats:
            raise ValueError(f"Unknown reward name '{name}' — check baseline stats.")

        device = rewards.device
        r_min = self.stats[name]["min"].to(device)
        r_max = self.stats[name]["max"].to(device)

        # --- Compute robust batch percentiles to avoid outliers ---
        batch_np = rewards.detach().cpu().numpy().flatten()
        batch_low = np.percentile(batch_np, self.low_p)
        batch_high = np.percentile(batch_np, self.high_p)
        if batch_high <= batch_low:
            batch_high = batch_low + self.eps

        # --- EMA update of running min and max ---
        new_min = torch.tensor(batch_low, dtype=torch.float32, device=device)
        new_max = torch.tensor(batch_high, dtype=torch.float32, device=device)
        r_min = self.beta * r_min + (1 - self.beta) * new_min
        r_max = self.beta * r_max + (1 - self.beta) * new_max
        self.stats[name]["min"], self.stats[name]["max"] = r_min, r_max

        # --- Robust min–max normalization to [-1, 1] ---
        denom = torch.clamp(r_max - r_min, min=self.eps)
        scaled = 2.0 * (rewards - r_min) / denom - 1.0
        scaled = torch.clamp(scaled, -1.0, 1.0)

        return scaled

    def get_stats(self) -> dict:
        """Return current EMA stats for logging or checkpointing."""
        return {
            name: {
                "min": float(v["min"].cpu().item()),
                "max": float(v["max"].cpu().item()),
            }
            for name, v in self.stats.items()
        }
