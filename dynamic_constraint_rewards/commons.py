import json
import os
import torch
from universal_constraint_rewards.commons import idx_to_labels
from dynamic_constraint_rewards.get_reward_stats import get_reward_stats_from_baseline, get_reward_stats_from_dataset
from omegaconf import DictConfig


def import_dynamic_reward_functions(reward_code_dir: str):
    # Test each reward function individually.
    base_dir = os.path.dirname(__file__)
    reward_functions = {}
    # Iterate through each file inside reward_code_dir and import the reward function.
    for file in os.listdir(os.path.join(base_dir, reward_code_dir)):
        if file.endswith(".py") and file != "__init__.py":
            # Dynamically import the module and extract get_reward and test_reward.
            import importlib

            module_name = f"dynamic_constraint_rewards.{reward_code_dir}.{file.split('.')[0]}"
            module = importlib.import_module(module_name)

            get_reward = getattr(module, "get_reward")
            test_reward = getattr(module, "test_reward")

            # print(f"[SAUGAT] Imported reward function from {file.split('.')[0]}")

            reward_functions[file.split(".")[0]] = {
                "get_reward": get_reward,
                "test_reward": test_reward,
            }
    get_reward_functions = {}
    for key, value in reward_functions.items():
        get_reward_functions[key] = value["get_reward"]

    test_reward_functions = {}
    for key, value in reward_functions.items():
        test_reward_functions[key] = value["test_reward"]

    return get_reward_functions, test_reward_functions


def get_dynamic_reward(
    parsed_scenes,
    reward_normalizer,
    get_reward_functions,
    num_classes=22,
    dynamic_importance_weights=None,
    config=None,
    floor_polygons=None,
    indices=None,
    is_val=None,
    sdf_cache=None,
    floor_plan_args=None,
    **kwargs,
):
    """
    Entry point for computing dynamic reward from multiple reward functions.

    this function assumes, llm has already created variable number of rewards to follow user's instruction. these reward functions are runnable.

    llm should also give use a technique to normalize each reward to be bounded with in [0, 1] range.
    0 is worst 1 is the best scene.


    this function returns the sum of all those rewards weighted by importance weights. and the total should also be in range [0, 1].
    """
    rewards = {}
    room_type = config.ddpo.dynamic_constraint_rewards.room_type
    all_rooms_info = json.load(
        open(
            os.path.join(
                config.dataset.data.path_to_dataset_files, "all_rooms_info.json"
            )
        )
    )
    from universal_constraint_rewards.commons import idx_to_labels
    idx_to_label = idx_to_labels[room_type]
    max_objects = all_rooms_info[room_type]["max_objects"]
    num_classes = all_rooms_info[room_type]["num_classes_with_empty"]
    for key, value in get_reward_functions.items():
        reward = value(
            parsed_scenes,
            idx_to_labels=idx_to_label,
            room_type=room_type,
            floor_polygons=floor_polygons,
            indices=indices,
            is_val=True,
            sdf_cache=sdf_cache,
            floor_plan_args=floor_plan_args,
            **kwargs,
        )
        print(f"[Ashok] Raw reward for {key}: {reward}")
        rewards[key] = reward

    reward_components = {}
    if reward_normalizer is not None:
        for key, value in rewards.items():
            reward_components[key] = value
            rewards[key] = reward_normalizer.normalize(key, torch.tensor(value))
            print(f"[Ashok] Normalized reward for {key}: {rewards[key]}")
    else:
        for key, value in rewards.items():
            reward_components[key] = value
    rewards_sum = 0
    try:
        dynamic_importance_weights = dynamic_importance_weights.get("importance_weights", None)
    except AttributeError:
        dynamic_importance_weights = None
    if dynamic_importance_weights is None:
        dynamic_importance_weights = {key: 1.0 for key in rewards.keys()}

    for key, value in rewards.items():
        # print(f"[Ashok] key: {key}")
        matching_key = "_".join(key.split("_")[1:])
        # print(f"[Ashok] matching_key: {matching_key}")
        # print(f"[Ashok] value: {value}")
        importance = dynamic_importance_weights.get(matching_key, 1.0)
        # print(f"[Ashok] importance: {importance}")
        rewards_sum += importance * value
    
    # print(f"[Ashok] rewards_sum: {rewards_sum}")
    # import sys; sys.exit()

    return rewards_sum , reward_components

    
def verify_tests_for_reward_function(room_type: str, user_query: str, reward_code_dir: str = None) -> bool:
    if reward_code_dir is None:
        raise ValueError("reward_code_dir is required")
    _, test_reward_functions = import_dynamic_reward_functions(reward_code_dir=reward_code_dir)  
    floor_polygons = [[-3,-3],[-3,3],[3,3],[3,-3]]
    for test_reward_function_name, test_reward_function in test_reward_functions.items():
        # try:
        print("Testing reward function: ", test_reward_function_name)
        test_reward_function(idx_to_labels[room_type], room_type, floor_polygons)
        print("Passed test", test_reward_function_name)
        # except Exception as e:
        #     print("Failed test", test_reward_function_name, "with error: ", e)
        #     return False
    return True

def get_reward_stats_from_baseline_helper(cfg: DictConfig, load = None, inpaint_masks=None, threshold_dict=None, reward_code_dir: str = None):
    get_reward_functions, _ = import_dynamic_reward_functions(reward_code_dir=reward_code_dir)  
    stats = get_reward_stats_from_baseline(
        get_reward_functions,
        load=load,
        num_scenes=1000,
        config=cfg,
        algorithm_custom_old=cfg.algorithm.custom.old,
        inpaint_masks=inpaint_masks,
        threshold_dict=threshold_dict,
    )
    # print(f"[Ashok] Baseline stats: {stats}")
    return stats

def get_reward_stats_from_dataset_helper(cfg: DictConfig, reward_code_dir: str = None, threshold_dict=None):
    get_reward_functions, _ = import_dynamic_reward_functions(reward_code_dir=reward_code_dir)  
    stats = get_reward_stats_from_dataset(
        reward_functions=get_reward_functions,
        config=cfg,
        threshold_dict=threshold_dict,
    )
    return stats

def save_reward_functions(reward_functions, reward_code_dir: str = None):
    if reward_code_dir is None:
        raise ValueError("reward_code_dir is required")
    base_dir = os.path.dirname(__file__)
    reward_code_dir = os.path.join(base_dir, reward_code_dir)
    rewards = reward_functions["rewards"]
    if os.path.exists(reward_code_dir):
        import shutil
        shutil.rmtree(reward_code_dir)
    os.makedirs(reward_code_dir, exist_ok=True)

    for reward in rewards:
        print(reward["id"], reward["constraint_id"], reward["name"])
        file_path = os.path.join(reward_code_dir, f"{reward['id']}_{reward['name']}.py")
        with open(file_path, "w") as f:
            f.write(reward["code"])
        print(f"Saved code to {file_path}")
    return True

def get_stats_from_initial_rewards(reward_functions, cfg=None, load=None, reward_code_dir: str = None):  
    if cfg is None:
        raise ValueError("cfg is required")
    # inpaint_masks = reward_functions["inpaint"]
    # inpaint_masks = str(inpaint_masks).replace("'", '')
    # print(inpaint_masks)
    # print(type(inpaint_masks))
    inpaint_masks = None
    threshold_dict = {}
    for reward in reward_functions["rewards"]:
        threshold_dict[reward["name"]] = reward["success_threshold"]
    print("Ashok Threshold dict: ", threshold_dict)
    dataset_stats = get_reward_stats_from_dataset_helper(cfg, reward_code_dir=reward_code_dir, threshold_dict=threshold_dict)
    baseline_stats = get_reward_stats_from_baseline_helper(cfg, load=load, inpaint_masks=inpaint_masks, threshold_dict=threshold_dict, reward_code_dir=reward_code_dir)  
    print(f"Dataset stats: {dataset_stats}")
    print(f"Baseline stats: {baseline_stats}")

    return dataset_stats, baseline_stats
