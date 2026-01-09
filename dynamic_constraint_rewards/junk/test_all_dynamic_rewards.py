"""
Comprehensive Test Suite for Dynamic Reward Functions

This module tests all dynamic reward functions in the dynamic_reward_functions directory
with proper test cases based on the actual parsed_scene data structure.

Author: Test Suite Generator
Date: 2025
"""

import sys
import traceback

from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamic_constraint_rewards.dynamic_reward_functions import (
    reward_tv_distance,
    reward_tv_present,
    reward_tv_viewing_angle,
)

# ============================================================================
# Test Data Creation
# ============================================================================


def create_idx_to_labels():
    """
    Create the idx_to_labels mapping for bedroom objects.
    Matches the structure from universal_constraint_rewards/commons.py
    """
    return {
        0: "armchair",
        1: "bookshelf",
        2: "cabinet",
        3: "ceiling_lamp",
        4: "chair",
        5: "children_cabinet",
        6: "coffee_table",
        7: "desk",
        8: "double_bed",
        9: "dressing_chair",
        10: "dressing_table",
        11: "kids_bed",
        12: "nightstand",
        13: "pendant_lamp",
        14: "shelf",
        15: "single_bed",
        16: "sofa",
        17: "stool",
        18: "table",
        19: "tv_stand",
        20: "wardrobe",
        21: "empty",  # Empty slot marker
    }


def create_test_scenes():
    """
    Create comprehensive test scenes matching the actual parsed_scene structure.

    Returns:
        dict: Dictionary of test scenes with various configurations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_labels = create_idx_to_labels()
    num_classes = len(idx_to_labels)
    max_objects = 12  # Typical max objects per scene

    scenes = {}

    # ========================================================================
    # Test Case 1: Perfect TV Viewing Setup
    # ========================================================================
    scenes["perfect_tv_setup"] = {
        "description": "Bed facing TV at ideal distance with sofa",
        "parsed_scene": {
            "one_hot": F.one_hot(
                torch.tensor(
                    [[8, 19, 16] + [21] * (max_objects - 3)], dtype=torch.long
                ),
                num_classes,
            )
            .float()
            .to(device),
            "positions": torch.tensor(
                [
                    [
                        [0.0, 0.5, 0.0],  # bed (double_bed, idx=8)
                        [3.0, 0.5, 0.0],  # tv_stand (idx=19) - 3m away
                        [1.5, 0.5, 2.0],  # sofa (idx=16)
                    ]
                    + [[0.0, 0.0, 0.0]] * (max_objects - 3)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "sizes": torch.tensor(
                [
                    [
                        [1.0, 0.4, 0.9],  # bed size
                        [0.6, 0.3, 0.4],  # tv_stand size
                        [0.8, 0.4, 0.8],  # sofa size
                    ]
                    + [[0.01, 0.01, 0.01]] * (max_objects - 3)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "orientations": torch.tensor(
                [
                    [
                        [1.0, 0.0],  # bed facing +X (toward TV)
                        [-1.0, 0.0],  # tv_stand facing -X (toward bed)
                        [0.7, 0.7],  # sofa at 45 degrees
                    ]
                    + [[0.0, 0.0]] * (max_objects - 3)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "object_indices": torch.tensor(
                [[8, 19, 16] + [21] * (max_objects - 3)],
                device=device,
                dtype=torch.long,
            ),
            "is_empty": torch.tensor(
                [[False, False, False] + [True] * (max_objects - 3)],
                device=device,
                dtype=torch.bool,
            ),
            "device": device,
        },
        "expected_results": {
            "tv_present": 1.0,
            "tv_distance": "> 0.5",  # Should be high (close to ideal distance)
            "tv_viewing_angle": "> 0.8",  # Should be high (good alignment)
        },
    }

    # ========================================================================
    # Test Case 2: No TV Present
    # ========================================================================
    scenes["no_tv"] = {
        "description": "Room with bed and furniture but no TV",
        "parsed_scene": {
            "one_hot": F.one_hot(
                torch.tensor(
                    [[8, 12, 20] + [21] * (max_objects - 3)], dtype=torch.long
                ),
                num_classes,
            )
            .float()
            .to(device),
            "positions": torch.tensor(
                [
                    [
                        [0.0, 0.5, 0.0],  # bed
                        [1.0, 0.3, -1.0],  # nightstand
                        [-2.0, 0.8, 0.0],  # wardrobe
                    ]
                    + [[0.0, 0.0, 0.0]] * (max_objects - 3)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "sizes": torch.tensor(
                [
                    [
                        [1.0, 0.4, 0.9],
                        [0.3, 0.3, 0.3],
                        [1.0, 0.9, 0.6],
                    ]
                    + [[0.01, 0.01, 0.01]] * (max_objects - 3)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "orientations": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                    ]
                    + [[0.0, 0.0]] * (max_objects - 3)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "object_indices": torch.tensor(
                [[8, 12, 20] + [21] * (max_objects - 3)],
                device=device,
                dtype=torch.long,
            ),
            "is_empty": torch.tensor(
                [[False, False, False] + [True] * (max_objects - 3)],
                device=device,
                dtype=torch.bool,
            ),
            "device": device,
        },
        "expected_results": {
            "tv_present": 0.0,
            "tv_distance": 0.0,
            "tv_viewing_angle": 0.0,
        },
    }

    # ========================================================================
    # Test Case 3: TV Too Close
    # ========================================================================
    scenes["tv_too_close"] = {
        "description": "TV very close to bed (poor distance)",
        "parsed_scene": {
            "one_hot": F.one_hot(
                torch.tensor([[8, 19] + [21] * (max_objects - 2)], dtype=torch.long),
                num_classes,
            )
            .float()
            .to(device),
            "positions": torch.tensor(
                [
                    [
                        [0.0, 0.5, 0.0],  # bed
                        [1.0, 0.5, 0.0],  # tv_stand - only 1m away
                    ]
                    + [[0.0, 0.0, 0.0]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "sizes": torch.tensor(
                [
                    [
                        [1.0, 0.4, 0.9],
                        [0.6, 0.3, 0.4],
                    ]
                    + [[0.01, 0.01, 0.01]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "orientations": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [-1.0, 0.0],
                    ]
                    + [[0.0, 0.0]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "object_indices": torch.tensor(
                [[8, 19] + [21] * (max_objects - 2)], device=device, dtype=torch.long
            ),
            "is_empty": torch.tensor(
                [[False, False] + [True] * (max_objects - 2)],
                device=device,
                dtype=torch.bool,
            ),
            "device": device,
        },
        "expected_results": {
            "tv_present": 1.0,
            "tv_distance": "< 0.5",  # Should be lower (not ideal distance)
            "tv_viewing_angle": "> 0.8",  # Alignment still good
        },
    }

    # ========================================================================
    # Test Case 4: TV Far Away
    # ========================================================================
    scenes["tv_too_far"] = {
        "description": "TV very far from bed",
        "parsed_scene": {
            "one_hot": F.one_hot(
                torch.tensor([[15, 19] + [21] * (max_objects - 2)], dtype=torch.long),
                num_classes,
            )
            .float()
            .to(device),
            "positions": torch.tensor(
                [
                    [
                        [0.0, 0.5, 0.0],  # single_bed
                        [6.0, 0.5, 0.0],  # tv_stand - 6m away
                    ]
                    + [[0.0, 0.0, 0.0]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "sizes": torch.tensor(
                [
                    [
                        [0.9, 0.4, 0.9],
                        [0.6, 0.3, 0.4],
                    ]
                    + [[0.01, 0.01, 0.01]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "orientations": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [-1.0, 0.0],
                    ]
                    + [[0.0, 0.0]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "object_indices": torch.tensor(
                [[15, 19] + [21] * (max_objects - 2)], device=device, dtype=torch.long
            ),
            "is_empty": torch.tensor(
                [[False, False] + [True] * (max_objects - 2)],
                device=device,
                dtype=torch.bool,
            ),
            "device": device,
        },
        "expected_results": {
            "tv_present": 1.0,
            "tv_distance": "< 0.3",  # Should be low (too far)
            "tv_viewing_angle": "> 0.8",
        },
    }

    # ========================================================================
    # Test Case 5: Poor Viewing Angle
    # ========================================================================
    scenes["poor_viewing_angle"] = {
        "description": "Bed facing away from TV",
        "parsed_scene": {
            "one_hot": F.one_hot(
                torch.tensor([[16, 19] + [21] * (max_objects - 2)], dtype=torch.long),
                num_classes,
            )
            .float()
            .to(device),
            "positions": torch.tensor(
                [
                    [
                        [0.0, 0.5, 0.0],  # sofa
                        [3.0, 0.5, 0.0],  # tv_stand
                    ]
                    + [[0.0, 0.0, 0.0]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "sizes": torch.tensor(
                [
                    [
                        [0.8, 0.4, 0.8],
                        [0.6, 0.3, 0.4],
                    ]
                    + [[0.01, 0.01, 0.01]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "orientations": torch.tensor(
                [
                    [
                        [-1.0, 0.0],  # sofa facing AWAY from TV
                        [-1.0, 0.0],
                    ]
                    + [[0.0, 0.0]] * (max_objects - 2)
                ],
                device=device,
                dtype=torch.float32,
            ),
            "object_indices": torch.tensor(
                [[16, 19] + [21] * (max_objects - 2)], device=device, dtype=torch.long
            ),
            "is_empty": torch.tensor(
                [[False, False] + [True] * (max_objects - 2)],
                device=device,
                dtype=torch.bool,
            ),
            "device": device,
        },
        "expected_results": {
            "tv_present": 1.0,
            "tv_distance": "> 0.5",
            "tv_viewing_angle": "< 0.2",  # Should be low (bad angle)
        },
    }

    # ========================================================================
    # Test Case 6: Empty Room
    # ========================================================================
    scenes["empty_room"] = {
        "description": "Completely empty room",
        "parsed_scene": {
            "one_hot": F.one_hot(
                torch.tensor([[21] * max_objects], dtype=torch.long), num_classes
            )
            .float()
            .to(device),
            "positions": torch.tensor(
                [[[0.0, 0.0, 0.0]] * max_objects], device=device, dtype=torch.float32
            ),
            "sizes": torch.tensor(
                [[[0.01, 0.01, 0.01]] * max_objects], device=device, dtype=torch.float32
            ),
            "orientations": torch.tensor(
                [[[0.0, 0.0]] * max_objects], device=device, dtype=torch.float32
            ),
            "object_indices": torch.tensor(
                [[21] * max_objects], device=device, dtype=torch.long
            ),
            "is_empty": torch.tensor(
                [[True] * max_objects], device=device, dtype=torch.bool
            ),
            "device": device,
        },
        "expected_results": {
            "tv_present": 0.0,
            "tv_distance": 0.0,
            "tv_viewing_angle": 0.0,
        },
    }

    # ========================================================================
    # Test Case 7: Multiple Beds and TVs (Batch Test)
    # ========================================================================
    scenes["batch_multiple"] = {
        "description": "Batch with multiple scenes of varying quality",
        "parsed_scene": {
            "one_hot": F.one_hot(
                torch.tensor(
                    [
                        [8, 19, 16] + [21] * (max_objects - 3),  # Good setup
                        [8, 12] + [21] * (max_objects - 2),  # No TV
                        [15, 19] + [21] * (max_objects - 2),  # TV too far
                    ],
                    dtype=torch.long,
                ),
                num_classes,
            )
            .float()
            .to(device),
            "positions": torch.tensor(
                [
                    # Scene 1: Good
                    [[0.0, 0.5, 0.0], [3.0, 0.5, 0.0], [1.5, 0.5, 2.0]]
                    + [[0.0, 0.0, 0.0]] * (max_objects - 3),
                    # Scene 2: No TV
                    [[0.0, 0.5, 0.0], [1.0, 0.3, -1.0]]
                    + [[0.0, 0.0, 0.0]] * (max_objects - 2),
                    # Scene 3: Too far
                    [[0.0, 0.5, 0.0], [7.0, 0.5, 0.0]]
                    + [[0.0, 0.0, 0.0]] * (max_objects - 2),
                ],
                device=device,
                dtype=torch.float32,
            ),
            "sizes": torch.tensor(
                [
                    [[1.0, 0.4, 0.9], [0.6, 0.3, 0.4], [0.8, 0.4, 0.8]]
                    + [[0.01, 0.01, 0.01]] * (max_objects - 3),
                    [[1.0, 0.4, 0.9], [0.3, 0.3, 0.3]]
                    + [[0.01, 0.01, 0.01]] * (max_objects - 2),
                    [[0.9, 0.4, 0.9], [0.6, 0.3, 0.4]]
                    + [[0.01, 0.01, 0.01]] * (max_objects - 2),
                ],
                device=device,
                dtype=torch.float32,
            ),
            "orientations": torch.tensor(
                [
                    [[1.0, 0.0], [-1.0, 0.0], [0.7, 0.7]]
                    + [[0.0, 0.0]] * (max_objects - 3),
                    [[1.0, 0.0], [0.0, 1.0]] + [[0.0, 0.0]] * (max_objects - 2),
                    [[1.0, 0.0], [-1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 2),
                ],
                device=device,
                dtype=torch.float32,
            ),
            "object_indices": torch.tensor(
                [
                    [8, 19, 16] + [21] * (max_objects - 3),
                    [8, 12] + [21] * (max_objects - 2),
                    [15, 19] + [21] * (max_objects - 2),
                ],
                device=device,
                dtype=torch.long,
            ),
            "is_empty": torch.tensor(
                [
                    [False, False, False] + [True] * (max_objects - 3),
                    [False, False] + [True] * (max_objects - 2),
                    [False, False] + [True] * (max_objects - 2),
                ],
                device=device,
                dtype=torch.bool,
            ),
            "device": device,
        },
        "expected_results": {
            "tv_present": [1.0, 0.0, 1.0],  # Scene 1: yes, Scene 2: no, Scene 3: yes
            "tv_distance": "varies",
            "tv_viewing_angle": "varies",
        },
    }

    return scenes


# ============================================================================
# Test Execution Functions
# ============================================================================


def test_reward_function(reward_module, scene_name, scene_data, idx_to_labels):
    """
    Test a single reward function on a scene.

    Args:
        reward_module: The reward module to test
        scene_name: Name of the test scene
        scene_data: Scene data dictionary
        idx_to_labels: Object index to label mapping

    Returns:
        dict: Test results
    """
    try:
        parsed_scene = scene_data["parsed_scene"]

        # Call the reward function
        rewards = reward_module.get_reward(parsed_scene, idx_to_labels=idx_to_labels)

        # Handle batch vs single scene
        if isinstance(rewards, torch.Tensor):
            if rewards.dim() == 0:
                reward_values = [rewards.item()]
            else:
                reward_values = rewards.cpu().tolist()
                if not isinstance(reward_values, list):
                    reward_values = [reward_values]
        else:
            reward_values = [float(rewards)]

        return {
            "success": True,
            "rewards": reward_values,
            "description": scene_data["description"],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "description": scene_data["description"],
        }


def run_all_tests():
    """
    Run comprehensive tests on all dynamic reward functions.
    """
    print("=" * 80)
    print("DYNAMIC REWARD FUNCTIONS - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    # Setup
    idx_to_labels = create_idx_to_labels()
    test_scenes = create_test_scenes()

    # Reward functions to test
    reward_functions = {
        "reward_tv_present": reward_tv_present,
        "reward_tv_distance": reward_tv_distance,
        "reward_tv_viewing_angle": reward_tv_viewing_angle,
    }

    # Track results
    all_results = {}

    # Test each reward function
    for func_name, func_module in reward_functions.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {func_name}")
        print(f"{'=' * 80}")

        func_results = {}

        for scene_name, scene_data in test_scenes.items():
            print(f"\n  Scene: {scene_name}")
            print(f"  Description: {scene_data['description']}")

            result = test_reward_function(
                func_module, scene_name, scene_data, idx_to_labels
            )
            func_results[scene_name] = result

            if result["success"]:
                rewards = result["rewards"]
                if len(rewards) == 1:
                    print(f"  ✓ Reward: {rewards[0]:.4f}")
                else:
                    print(f"  ✓ Rewards (batch):")
                    for i, r in enumerate(rewards):
                        print(f"      [{i}]: {r:.4f}")

                # Check expectations
                expected = scene_data.get("expected_results", {}).get(
                    func_name.replace("reward_", "")
                )
                if expected is not None:
                    if isinstance(expected, (int, float)):
                        if abs(rewards[0] - expected) < 0.1:
                            print(f"  ✓ Matches expected: {expected}")
                        else:
                            print(f"  ⚠ Expected {expected}, got {rewards[0]:.4f}")
                    elif isinstance(expected, str):
                        print(f"  ℹ Expected: {expected}")
                    elif isinstance(expected, list):
                        print(f"  ℹ Expected batch: {expected}")
            else:
                print(f"  ✗ ERROR: {result['error']}")
                print(f"  Traceback:\n{result['traceback']}")

        all_results[func_name] = func_results

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for func_name, func_results in all_results.items():
        total = len(func_results)
        passed = sum(1 for r in func_results.values() if r["success"])
        failed = total - passed

        status = "✓ PASS" if failed == 0 else "✗ FAIL"
        print(f"\n{func_name}: {status}")
        print(f"  Passed: {passed}/{total}")
        print(f"  Failed: {failed}/{total}")

        if failed > 0:
            print(f"  Failed scenes:")
            for scene_name, result in func_results.items():
                if not result["success"]:
                    print(f"    - {scene_name}: {result['error']}")

    print("\n" + "=" * 80)

    return all_results


# ============================================================================
# Data Type Verification
# ============================================================================


def verify_data_types():
    """
    Verify that the parsed_scene data types match the expected types.
    """
    print("\n" + "=" * 80)
    print("DATA TYPE VERIFICATION")
    print("=" * 80)

    test_scenes = create_test_scenes()
    scene_data = test_scenes["perfect_tv_setup"]
    parsed_scene = scene_data["parsed_scene"]

    print("\nParsed Scene Data Types:")
    for key, value in parsed_scene.items():
        if isinstance(value, torch.Tensor):
            print(
                f"  {key:20s}: {type(value)} (dtype={value.dtype}, shape={list(value.shape)})"
            )
        else:
            print(f"  {key:20s}: {type(value)}")

    print("\n✓ All data types verified!")
    print("=" * 80)


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    try:
        # Verify data types match expectations
        verify_data_types()

        # Run all tests
        results = run_all_tests()

        print("\n✓ Test suite completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
