import json
import logging
import os
import random

from functools import cache
from typing import Any, Union

import torch

from datasets import Dataset
from omegaconf import DictConfig, ListConfig
from torch.utils.data import WeightedRandomSampler
from transformers import BatchEncoding

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
from steerable_scene_generation.datasets.common import BaseDataset

random.seed(42)
np.random.seed(42)

# def draw_rectangle(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
#     """Draw a rectangle outline on the image."""
#     # Top and bottom edges
#     img[y, x:x+w] = 1
#     img[y+h-1, x:x+w] = 1
#     # Left and right edges
#     img[y:y+h, x] = 1
#     img[y:y+h, x+w-1] = 1
#     return img

# def draw_circle(img: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
#     """Draw a circle outline on the image."""
#     for angle in np.linspace(0, 2*np.pi, int(2*np.pi*radius)):
#         x = int(cx + radius * np.cos(angle))
#         y = int(cy + radius * np.sin(angle))
#         if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
#             img[y, x] = 1
#     return img

# def draw_ellipse(img: np.ndarray, cx: int, cy: int, a: int, b: int) -> np.ndarray:
#     """Draw an ellipse outline on the image."""
#     for angle in np.linspace(0, 2*np.pi, int(2*np.pi*max(a, b))):
#         x = int(cx + a * np.cos(angle))
#         y = int(cy + b * np.sin(angle))
#         if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
#             img[y, x] = 1
#     return img

# def draw_triangle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int) -> np.ndarray:
#     """Draw a triangle outline on the image."""
#     # Draw lines between vertices
#     def draw_line(xa, ya, xb, yb):
#         points = max(abs(xb - xa), abs(yb - ya))
#         for i in range(points + 1):
#             t = i / max(points, 1)
#             x = int(xa + t * (xb - xa))
#             y = int(ya + t * (yb - ya))
#             if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
#                 img[y, x] = 1
    
#     draw_line(x1, y1, x2, y2)
#     draw_line(x2, y2, x3, y3)
#     draw_line(x3, y3, x1, y1)
#     return img

# def draw_polygon(img: np.ndarray, vertices: List[Tuple[int, int]]) -> np.ndarray:
#     """Draw a polygon outline on the image."""
#     def draw_line(xa, ya, xb, yb):
#         points = max(abs(xb - xa), abs(yb - ya))
#         for i in range(points + 1):
#             t = i / max(points, 1)
#             x = int(xa + t * (xb - xa))
#             y = int(ya + t * (yb - ya))
#             if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
#                 img[y, x] = 1
    
#     for i in range(len(vertices)):
#         x1, y1 = vertices[i]
#         x2, y2 = vertices[(i + 1) % len(vertices)]
#         draw_line(x1, y1, x2, y2)
    
#     return img

# def generate_sample(size: int = 64) -> np.ndarray:
#     """Generate a single sample with a random closed surface."""
#     img = np.zeros((size, size), dtype=np.uint8)
    
#     shape_type = random.choice(['rectangle', 'circle', 'ellipse', 'triangle', 'pentagon', 'hexagon'])
    
#     if shape_type == 'rectangle':
#         w = random.randint(10, 30)
#         h = random.randint(10, 30)
#         x = random.randint(5, size - w - 5)
#         y = random.randint(5, size - h - 5)
#         img = draw_rectangle(img, x, y, w, h)
    
#     elif shape_type == 'circle':
#         radius = random.randint(8, 20)
#         cx = random.randint(radius + 5, size - radius - 5)
#         cy = random.randint(radius + 5, size - radius - 5)
#         img = draw_circle(img, cx, cy, radius)
    
#     elif shape_type == 'ellipse':
#         a = random.randint(8, 20)
#         b = random.randint(8, 20)
#         cx = random.randint(a + 5, size - a - 5)
#         cy = random.randint(b + 5, size - b - 5)
#         img = draw_ellipse(img, cx, cy, a, b)
    
#     elif shape_type == 'triangle':
#         margin = 10
#         x1 = random.randint(margin, size - margin)
#         y1 = random.randint(margin, size - margin)
#         x2 = random.randint(margin, size - margin)
#         y2 = random.randint(margin, size - margin)
#         x3 = random.randint(margin, size - margin)
#         y3 = random.randint(margin, size - margin)
#         img = draw_triangle(img, x1, y1, x2, y2, x3, y3)
    
#     elif shape_type in ['pentagon', 'hexagon']:
#         n_sides = 5 if shape_type == 'pentagon' else 6
#         cx = random.randint(20, size - 20)
#         cy = random.randint(20, size - 20)
#         radius = random.randint(10, 18)
#         vertices = []
#         for i in range(n_sides):
#             angle = 2 * np.pi * i / n_sides
#             x = int(cx + radius * np.cos(angle))
#             y = int(cy + radius * np.sin(angle))
#             vertices.append((x, y))
#         img = draw_polygon(img, vertices)
    
#     return img

# def create_dataset(n_samples: int = 10000, size: int = 64) -> np.ndarray:
#     """Create a dataset of n_samples closed surface images."""
#     dataset = np.zeros((n_samples, size, size), dtype=np.uint8)
    
#     print(f"Generating {n_samples} samples...")
#     for i in range(n_samples):
#         dataset[i] = generate_sample(size)
#         if (i + 1) % 1000 == 0:
#             print(f"Generated {i + 1}/{n_samples} samples")
    
#     print("Dataset generation complete!")
#     return torch.tensor(dataset)

# def visualize_samples(dataset: np.ndarray, n_samples: int = 16, seed: int = None) -> None:
#     """Visualize random samples from the dataset."""
#     if seed is not None:
#         np.random.seed(seed)
    
#     indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
#     rows = int(np.sqrt(n_samples))
#     cols = int(np.ceil(n_samples / rows))
    
#     fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
#     axes = axes.flatten() if n_samples > 1 else [axes]
    
#     for idx, ax in enumerate(axes):
#         if idx < len(indices):
#             ax.imshow(dataset[indices[idx]], cmap='binary', interpolation='nearest')
#             ax.set_title(f'Sample {indices[idx]}')
#         ax.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# Example usage
# Create dataset
# n_samples = 100
# dataset = create_dataset(n_samples=n_samples, size=64)


# Save dataset
# np.save('closed_surfaces_dataset.npy', dataset)
# print(f"Dataset shape: {dataset.shape}")
# print(f"Dataset saved to 'closed_surfaces_dataset.npy'")

# Visualize some samples
# visualize_samples(dataset, n_samples=16, seed=42)

# # To load the dataset later:
# # dataset = np.load('closed_surfaces_dataset.npy')
# print(f"Average number of 1s per sample: {(dataset.sum(axis=(1,2)).mean()):.2f}")
import numpy as np

def make_parallelogram(n_samples=100000,
                       origin=(0.0, 0.0),
                       v1=(2.0, 0.5),
                       v2=(0.5, 1.5),
                       noise=0.05,
                       random_state=42):
    """
    Generate 2D points uniformly sampled from a fixed parallelogram.

    Parameters
    ----------
    n_samples : int
        Number of points to sample
    origin : tuple(float, float)
        Origin of the parallelogram
    v1, v2 : tuple(float, float)
        Edge vectors defining the parallelogram
    noise : float
        Standard deviation of Gaussian noise added to points
    random_state : int or None
        Seed for reproducibility

    Returns
    -------
    x : ndarray of shape (n_samples, 2)
        Sampled points
    """

    rng = np.random.default_rng(random_state)

    a = rng.uniform(0.0, 1.0, size=(n_samples, 1))
    b = rng.uniform(0.0, 1.0, size=(n_samples, 1))

    origin = np.array(origin)
    v1 = np.array(v1)
    v2 = np.array(v2)

    x = origin + a * v1 + b * v2

    if noise > 0.0:
        x += rng.normal(scale=noise, size=x.shape)

    return x

def update_data_file_paths(config_data, config):
    config_data["dataset_directory"] = os.path.join(
        config["data"]["path_to_processed_data"], config_data["dataset_directory"]
    )
    config_data["annotation_file"] = os.path.join(
        config["data"]["path_to_dataset_files"], config_data["annotation_file"]
    )
    return config_data


console_logger = logging.getLogger(__name__)


class CustomDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        split: str | list | ListConfig,
        ckpt_path: str | None = None,
    ):
        """
        Args:
            cfg: a DictConfig object defined by `configurations/dataset/scene.yaml`.
            split: One of "training", "validation", "test".
            ckpt_path: The optional checkpoint path.
        """
        self.cfg = cfg
        n_samples = 100000
        self.raw_dataset = make_parallelogram(n_samples=n_samples)
        self.raw_dataset = (self.raw_dataset - self.raw_dataset.mean(axis=0)) / self.raw_dataset.std(axis=0)

        # HF dataset features are not used in this ThreedFront path
        self.hf_dataset = None
        self.use_subdataset_sampling = False
        self.subdataset_ranges = None
        self.subdataset_names = None


    def normalize_scenes(self, scenes: torch.Tensor) -> torch.Tensor:
        return scenes

    def inverse_normalize_scenes(self, scenes: torch.Tensor) -> torch.Tensor:
        return scenes

    def __len__(self) -> int:
        """
        Returns the length of the ThreedFront encoded dataset.
        """
        
        return len(self.raw_dataset)

        

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw_item = self.raw_dataset[idx]
        data_tensor = torch.tensor(raw_item.flatten())# flattened to 1 d tensor

        item: dict[str, Any] = {
            "scenes": data_tensor,
            "idx": idx,
        }


        return item


    def replace_cond_data(
        self, data: dict[str, Any], txt_labels: str | list[str]
    ) -> dict[str, Any]:
        return data

    @staticmethod
    def sample_data_dict(data: dict[str, Any], num_items: int) -> dict[str, Any]:
        """
        Sample `num_items` from `data`. Sample with replacement if `data` contains less
        than `num_items` items.

        Args:
            data (dict[str, Any]): The data to sample.

        Returns:
            dict[str, Any]: The sampled data of length `num_items`.
        """
        total_items = len(data["scenes"])
        # print(f"[Ashok] items in data {[ (key, type(value)) for key, value in data.items() ]}")
        if num_items <= total_items:
            # Sample without replacement.
            # sample_indices = torch.randperm(total_items)[:num_items] # TODO: use this for training rl but for inference it shuffles the idx so the floor cond renders are wrong
            # print(
            #     f"[Ashok] Sampling without replacement for {num_items} items, with total items {total_items}"
            # )
            sample_indices = torch.arange(num_items)
        else:
            # Sample with replacement.
            sample_indices = torch.randint(0, total_items, (num_items,))
            # raise NotImplementedError("please provide data <= num_items")

        # Create the sampled data dictionary.
        sampled_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                sampled_data[key] = value[sample_indices]
            elif isinstance(value, list):
                sampled_data[key] = [value[i] for i in sample_indices]
            elif isinstance(value, BatchEncoding):
                sampled_data[key] = BatchEncoding(
                    {k: v[sample_indices] for k, v in value.items()}
                )
            else:
                raise ValueError(
                    f"Unsupported data type '{type(value)}' for key '{key}'"
                )
        return sampled_data

    def get_sampler(self) -> Union[WeightedRandomSampler, None]:
        """
        Returns a sampler for weighted random sampling of the dataset based on the
        dataset labels.
        This is an alternative to the two-step sampling process used in
        `sample_subdataset_index` and `sample_from_subdataset`.
        """
        # raise NotImplementedError("get_sampler is not supported for toy dataset")
        print("[Ashok] WARNING get_sampler is not supported for toy dataset")
        if not self.cfg.custom_data_batch_mix.use:
            return None

        if "labels" not in self.hf_dataset.column_names:
            raise ValueError("Dataset does not contain labels!")

        labels = self.hf_dataset["labels"]

        # Calculate the number of samples for each class.
        class_counts = torch.bincount(labels)

        # Calculate weights for each class.
        class_weights = [
            p / count if count > 0 else 0.0
            for p, count in zip(
                self.cfg.custom_data_batch_mix.label_probs, class_counts
            )
        ]

        # Create a list of weights for each sample in the dataset.
        weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    @cache
    def get_all_data(
        self,
        normalized: bool = True,
        label: int = None,
        scene_indices: torch.Tensor | None = None,
        only_scenes: bool = False,
    ) -> dict[str, Any]:
        """
        Returns all data in the dataset, including scenes and additional attributes.

        Args:
            normalized (bool, optional): Whether to return normalized scenes.
            label (int, optional): If not None, only data that correspond to that
                label are returned. This option is ignored if the dataset does not
                contain labels.
            scene_indices (torch.Tensor, optional): If not None, only data at the
                specified indices are returned.
            only_scenes (bool, optional): If True, only the scenes are returned.

        Returns:
            dict[str, Any]: All data in the dataset, including "scenes" and
                additional attributes.
        """
        if self.encoded_dataset is None:
            raise ValueError("Dataset is not loaded!")

        if only_scenes and label is not None:
            raise ValueError("Cannot specify both 'only_scenes' and 'label'.")

        # Use indexing to fetch only the required data.
        # Aggregate from encoded dataset
        indices = (
            scene_indices.tolist()
            if scene_indices is not None
            else list(range(len(self)))
        )
        scenes_list = []
        for i in indices:
            sample = self.encoded_dataset[i]
            scenes_list.append(self._to_scene_tensor(sample))
        raw_scenes = torch.stack(scenes_list, dim=0)
        if only_scenes:
            return {"scenes": raw_scenes}
        return {"scenes": raw_scenes}

    def set_data(self, data: dict[str, torch.Tensor], normalized: bool = False) -> None:
        """
        Replaces the dataset with new data.

        Note that this will disable subdataset sampling if it was enabled.

        Args:
            data (dict[str, torch.Tensor]): The data dictionary containing "scenes" and
                optionally additional attributes such as "labels".
            normalized (bool): Whether the scenes are normalized.
        """
        # Not supported for ThreedFront-backed dataset
        raise NotImplementedError("set_data is not supported for ThreedFront dataset")

    def setup_subdataset_sampling(
        self, sampling_cfg: DictConfig, hf_dataset: Dataset
    ) -> None:
        """
        Sets up weighted subdataset sampling based on configuration.

        Args:
            sampling_cfg: Configuration for subdataset sampling.
            hf_dataset: HuggingFace dataset to sample from.
        """
        raise NotImplementedError("setup_subdataset_sampling is not supported for toy dataset")
        self.subdataset_probs = sampling_cfg.probabilities

        # Validate that all subdataset names have probabilities.
        if self.subdataset_probs is None:
            raise ValueError(
                "Subdataset sampling is enabled but no probabilities are specified."
            )
        for name in self.subdataset_names:
            if name not in self.subdataset_probs:
                raise ValueError(f"No probability specified for subdataset '{name}'")

        # Validate that probabilities approximately sum to 1.
        total_prob = sum(self.subdataset_probs.values())
        if not 0.98 <= total_prob <= 1.02:  # Allow for small floating point errors
            raise ValueError(
                f"Subdataset probabilities must sum to 1, got {total_prob}"
            )

        # Normalize probabilities to sum to 1.
        self.subdataset_probs = {
            name: prob / total_prob for name, prob in self.subdataset_probs.items()
        }

        # Create cumulative probabilities for efficient sampling
        self.subdataset_cum_probs = []
        cum_prob = 0.0
        for name in self.subdataset_names:
            cum_prob += self.subdataset_probs[name]
            self.subdataset_cum_probs.append(cum_prob)

        # Check if we should use infinite iterators.
        self.use_infinite_iterators = sampling_cfg.use_infinite_iterators
        if self.use_infinite_iterators:
            # Create infinite iterators for each subdataset.
            self.subdataset_iterators = []
            for start_idx, end_idx in self.subdataset_ranges:
                subdataset = hf_dataset.select(range(start_idx, end_idx))
                iterator = InfiniteShuffledStreamingDataset(
                    dataset=subdataset, buffer_size=sampling_cfg.buffer_size
                )
                self.subdataset_iterators.append(iterator)

            # Initialize iterators.
            self.subdataset_iterator_objects = [
                iter(iterator) for iterator in self.subdataset_iterators
            ]
            console_logger.info("Using infinite iterators for subdataset sampling.")

        console_logger.info(
            "Enabled weighted subdataset sampling with probabilities: "
            f"{self.subdataset_probs}"
        )

    def sample_subdataset_index(self) -> int:
        """
        Samples a subdataset index based on the configured probabilities.

        Returns:
            int: The index of the sampled subdataset.
        """
        raise NotImplementedError("sample_subdataset_index is not supported for toy dataset")
        if not self.use_subdataset_sampling:
            raise RuntimeError("Subdataset sampling is not enabled.")

        # Sample a random value between 0 and 1.
        r = random.random()

        # Find the subdataset whose cumulative probability range contains r.
        for i, cum_prob in enumerate(self.subdataset_cum_probs):
            if r <= cum_prob:
                return i

        # Something went wrong.
        raise RuntimeError("Failed to sample a subdataset index.")

    def sample_from_subdataset(self, subdataset_idx: int) -> int:
        """
        Samples an index from the specified subdataset.

        Args:
            subdataset_idx: Index of the subdataset to sample from.

        Returns:
            int: The global index of the sampled item.
        """
        raise NotImplementedError("sample_from_subdataset is not supported for toy dataset")
        # Use pre-computed indices for fast sampling.
        start_idx, end_idx = self.subdataset_ranges[subdataset_idx]
        return random.randint(start_idx, end_idx - 1)

    def get_subdataset_name_from_index(self, index: int) -> str:
        """
        Returns the subdataset name for the given dataset index.
        """
        raise NotImplementedError("get_subdataset_name_from_index is not supported for toy dataset")
        if self.subdataset_ranges is None:
            raise ValueError("Subdataset ranges are not set!")

        for i, (start_idx, end_idx) in enumerate(self.subdataset_ranges):
            if index >= start_idx and index < end_idx:
                return self.subdataset_names[i]
        raise ValueError(f"Index {index} is out of range for any subdataset.")

    def _validate_subdataset_config(self) -> None:
        """
        Validates the subdataset configuration, ensuring that necessary metadata
        is available when using subdataset features.
        """
        raise NotImplementedError("_validate_subdataset_config is not supported for toy dataset")
        # Check if static subdataset prompts are enabled but metadata is missing.
        if self.cfg.static_subdataset_prompts.use and (
            self.subdataset_ranges is None or self.subdataset_names is None
        ):
            raise ValueError(
                "Require subdataset ranges and names to be set when using static "
                "subdataset prompts!"
            )

        # Check if subdataset sampling is enabled but metadata is missing.
        if self.cfg.subdataset_sampling.use and (
            self.subdataset_ranges is None or self.subdataset_names is None
        ):
            raise ValueError(
                "Require subdataset ranges and names to be set when using "
                "subdataset sampling!"
            )

        # Check if static prompts are provided for all subdatasets.
        if self.cfg.static_subdataset_prompts.use and set(
            self.cfg.static_subdataset_prompts.name_to_prompt.keys()
        ) != set(self.subdataset_names):
            raise ValueError(
                "Require static subdataset prompts to be set for all subdatasets!\n"
                f"Subdataset names: {self.subdataset_names}\n"
                f"Prompts: {self.cfg.static_subdataset_prompts.name_to_prompt.keys()}"
            )

    def _validate_dataset_structure(
        self, hf_dataset: Dataset, metadata: dict[str, Any]
    ) -> None:
        raise NotImplementedError("_validate_dataset_structure is not supported for toy dataset")
        # Not used in ThreedFront-backed implementation
        return



