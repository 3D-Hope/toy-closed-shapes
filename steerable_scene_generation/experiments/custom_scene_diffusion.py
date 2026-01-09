from typing import Dict, Type

import torch

from steerable_scene_generation.algorithms.scene_diffusion import (
    SceneDiffuserTrainerDDPM,
    SceneDiffuserTrainerPPO,
    SceneDiffuserTrainerScore,
    create_scene_diffuser_diffuscene,
    create_scene_diffuser_flux_transformer,
)
from steerable_scene_generation.datasets.custom_scene import CustomDataset
from steerable_scene_generation.experiments.scene_diffusion import (
    SceneDiffusionExperiment,
)


class CustomSceneDiffusionExperiment(SceneDiffusionExperiment):
    """A scene diffusion experiment using a custom dataset."""

    # Inherit all the algorithm factories and trainers from SceneDiffusionExperiment

    # Override the compatible_datasets to include our custom dataset
    # compatible_datasets = dict(
    #     scene=CustomSceneDataset,  # The original key "scene" still points to our custom dataset
    #     custom_scene=CustomSceneDataset,  # Add a direct "custom_scene" key
    # )
    compatible_datasets = dict(
        scene=CustomDataset,  # The original key "scene" still points to our custom dataset
        custom_scene=CustomDataset,  # Add a direct "custom_scene" key
    )

    def inpaint(
        self,
        dataloader: torch.utils.data.DataLoader | None = None,
        use_ema: bool = False,
        scenes=None,
        inpaint_masks=None,
        to_hardcode=None,
        callbacks: list | None = None,
    ) -> list[torch.Tensor]:
        raise NotImplementedError("inpaint not implemented yet")
        if not self.algo:
            self.algo = self._build_algo(ckpt_path=self.ckpt_path)
            if self.ckpt_path is not None:
                print(f"[DEBUG] Loading checkpoint: {self.ckpt_path} SUI7")
                ckpt = torch.load(self.ckpt_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
                missing, unexpected = self.algo.load_state_dict(
                    state_dict, strict=False
                )
                if getattr(self.algo, "ema", None) and "ema_state_dict" in ckpt:
                    self.algo.ema.load_state_dict(ckpt["ema_state_dict"])
                print(f"[DEBUG] Missing keys: {missing}")
                print(f"[DEBUG] Unexpected keys: {unexpected}")
                
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo = self.algo.to(device)
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)
        device = next(self.algo.parameters()).device
        
        if dataloader is None:
            dataloader = self._build_test_loader(self.ckpt_path)
        dataloader = (
            torch.utils.data.DataLoader(dataloader, batch_size=self.cfg.test.batch_size)
            if isinstance(dataloader, torch.utils.data.Dataset)
            else dataloader
        )

        results: list[torch.Tensor] = []
        self.algo.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                inpainted = self.algo.inpaint_scenes(
                    batch, scenes=scenes, inpaint_masks=inpaint_masks, use_ema=use_ema,
                    to_hardcode=to_hardcode,
                )
                results.append(inpainted.cpu())
        return results
