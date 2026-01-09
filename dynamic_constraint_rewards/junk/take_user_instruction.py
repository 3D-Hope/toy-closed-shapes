# takes the user input
import argparse
import json
import os

import hydra
import torch

from call_agent import ConstraintGenerator, RewardGenerator
from commons import import_dynamic_reward_functions
from get_reward_stats_from_baseline import get_reward_stats_from_baseline
from get_reward_stats_from_dataset import get_reward_stats_from_dataset
from omegaconf import DictConfig, OmegaConf
from scale_raw_rewards import RewardNormalizer

from steerable_scene_generation.utils.omegaconf import register_resolvers
from universal_constraint_rewards.commons import get_all_universal_reward_functions


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig):
    register_resolvers()
    OmegaConf.resolve(cfg)
    algorithm_config = cfg.algorithm

    # Set default values if the config structure doesn't exist
    if hasattr(algorithm_config, "ddpo") and hasattr(
        algorithm_config.ddpo, "dynamic_constraint_rewards"
    ):
        user_input = algorithm_config.ddpo.dynamic_constraint_rewards.user_query
        reward_code_dir = (
            algorithm_config.ddpo.dynamic_constraint_rewards.reward_code_dir
        )
        room_type = algorithm_config.ddpo.dynamic_constraint_rewards.room_type
    else:
        raise
    os.makedirs(reward_code_dir, exist_ok=True)

    # TODO: Uncomment this when we have the llm working.
    # Call llm to parse the user input and create a list of reward functions to follow the user instruction.
    # constraint_generator = ConstraintGenerator()
    # reward_generator = RewardGenerator()

    # constraints = constraint_generator.generate_constraints({"room_type": room_type, "query": user_input})
    # print(constraints)

    # for i in range(len(constraints["constraints"])):
    #     constraint = constraints["constraints"][i]
    #     print(f"[SAUGAT] Constraint {i+1}: {constraint}")
    #     result = reward_generator.generate_reward({"room_type": room_type, "query": user_input, "constraint": constraint})
    #     print(f"[SAUGAT] Reward code {i+1}: {result["raw_response"]}")
    #     with open(f"{reward_code_dir}/{i}_code.py", "w") as f:
    #         f.write(result["raw_response"])

    # Test each reward function individually.

    get_reward_functions, test_reward_functions = import_dynamic_reward_functions(
        reward_code_dir
    )

    # Add all universal reward functions to the existing dynamic reward functions
    universal_reward_functions = get_all_universal_reward_functions()
    get_reward_functions.update(universal_reward_functions)

    from universal_constraint_rewards.physcene.room_layout_constraint import (
        room_layout_constraint,
    )

    get_reward_functions["room_layout_constraint"] = room_layout_constraint
    # Test each reward function individually.
    for file in test_reward_functions:
        test_reward_functions[file]()

    # Get stats from baseline model
    print("\n" + "=" * 80)
    print("Computing reward statistics from BASELINE MODEL")
    print("=" * 80)
    stats = get_reward_stats_from_baseline(
        get_reward_functions,
        num_scenes=1000,
        config=cfg,
        # algorithm="scene_diffuser_flux_transformer",
        load="9xplsx0a",
        # algorithm_classifier_free_guidance_use_floor=False,
    )
    print("Baseline Stats: ", stats)
    stats_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/dynamic_constraint_rewards/stats.json"

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved baseline stats to {stats_path}")

    # Get statistics from ground truth dataset
    print("\n" + "=" * 80)
    print("Computing statistics from GROUND TRUTH DATASET")
    print("=" * 80)
    dataset_stats = get_reward_stats_from_dataset(
        reward_functions=get_reward_functions,
        config=cfg,
        num_scenes=1000,
    )

    # Save dataset stats
    dataset_stats_path = os.path.join(reward_code_dir, "dataset_stats.json")
    with open(dataset_stats_path, "w") as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"\nDataset statistics saved to: {dataset_stats_path}")

    # Testing normalizer (commented out for now)
    # reward_normalizer = RewardNormalizer(stats)

    # for key, value in get_reward_functions.items():
    #     normalized_reward = reward_normalizer.normalize(key, torch.tensor([1.0, 2.0, 3.0]))
    #     print(f"Normalized reward for {key}: {normalized_reward}")


if __name__ == "__main__":
    main()

# Run Command:
# python take_user_instruction.py dataset=custom_scene algorithm=scene_diffuser_flux_transformer
