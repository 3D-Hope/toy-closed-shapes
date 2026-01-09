from dynamic_constraint_rewards.commons import import_dynamic_reward_functions, get_reward_stats_from_baseline_helper, get_reward_stats_from_dataset_helper
from universal_constraint_rewards.commons import idx_to_labels
import hydra
from commons import import_dynamic_reward_functions
from omegaconf import DictConfig, OmegaConf
from steerable_scene_generation.utils.omegaconf import register_resolvers

    
def verify_tests_for_reward_function(output_dir: str = None) -> bool:
    reward_code_dir = "dynamic_reward_functions"
    _, test_reward_functions = import_dynamic_reward_functions(reward_code_dir, output_dir=output_dir)  
    floor_polygons = [[-3,-3],[-3,3],[3,3],[3,-3]]
    ROOM_TYPE = 'bedroom'
    for test_reward_function_name, test_reward_function in test_reward_functions.items():
        try:
            print("Testing reward function: ", test_reward_function_name)
            test_reward_function(idx_to_labels[ROOM_TYPE], ROOM_TYPE, floor_polygons)
            print("Passed test", test_reward_function_name)
        except Exception as e:
            print("Failed test", test_reward_function_name, "with error: ", e)
            return False
    return True


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig):
    register_resolvers()
    OmegaConf.resolve(cfg)
    
    test_passed = verify_tests_for_reward_function(cfg.dataset.data.room_type)
    if not test_passed:
        print("Test verification failed!")
        raise RuntimeError("Test verification failed!")
    print("Test verification passed!")
    dataset_stats = get_reward_stats_from_dataset_helper(cfg)
    baseline_stats = get_reward_stats_from_baseline_helper(cfg)  
    print(f"Dataset stats: {dataset_stats}")
    print(f"Baseline stats: {baseline_stats}")

if __name__ == "__main__":
    main()

# python test_llm_interactions.py algorithm=scene_diffuser_midiffusion dataset=custom_scene algorithm.custom.old=True