# import json
# import google.generativeai as genai
# from dotenv import load_dotenv
# import os
# import sys
# from time import sleep
# from create_prompt import create_prompt
# import torch
# import torch.nn.functional as F
# import traceback
# from pathlib import Path

# load_dotenv()
# # Configure Gemini API
# API_KEY = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=API_KEY)


# def generate_reward_function_with_gemini(room_type, query, model_name='gemini-2.0-flash-exp'):
#     """
#     Generate a reward function using Gemini API

#     Args:
#         room_type (str): Type of room (bedroom, livingroom, etc.)
#         query (str): Constraint description
#         model_name (str): Gemini model to use

#     Returns:
#         dict: Generated code and metadata
#     """
#     try:
#         # Initialize the model
#         model = genai.GenerativeModel(model_name)

#         # Create prompt
#         prompt = create_prompt(room_type, query)

#         # Generate response
#         print(f"🤖 Generating reward function for: '{query}'...")
#         response = model.generate_content(prompt)

#         # Extract and clean response
#         response_text = response.text.strip()
#         if response_text.startswith("```python"):
#             response_text = response_text[9:]
#         if response_text.startswith("```"):
#             response_text = response_text[3:]
#         if response_text.endswith("```"):
#             response_text = response_text[:-3]
#         response_text = response_text.strip()

#         return {
#             "success": True,
#             "code": response_text,
#             "room_type": room_type,
#             "query": query
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "traceback": traceback.format_exc(),
#             "room_type": room_type,
#             "query": query
#         }


# def test_reward_function_syntax(code, reward_name):
#     """
#     Test if the generated code compiles and has required functions

#     Args:
#         code (str): Python code to test
#         reward_name (str): Name identifier for this reward

#     Returns:
#         dict: Test results
#     """
#     print(f"\n📝 Testing syntax for {reward_name}...")

#     try:
#         # Try to compile the code
#         compile(code, f"<{reward_name}>", "exec")

#         # Execute in isolated namespace
#         namespace = {}
#         exec(code, namespace)

#         # Check for required functions
#         has_reward_func = any(
#             callable(v) and "reward" in k.lower() and "test" not in k.lower()
#             for k, v in namespace.items()
#         )
#         has_test_func = any(
#             callable(v) and "test" in k.lower()
#             for k, v in namespace.items()
#         )

#         if not has_reward_func:
#             return {
#                 "success": False,
#                 "error": "No reward function found in generated code"
#             }

#         if not has_test_func:
#             return {
#                 "success": False,
#                 "error": "No test function found in generated code"
#             }

#         # Extract function names
#         reward_func_name = [k for k, v in namespace.items()
#                            if callable(v) and "reward" in k.lower() and "test" not in k.lower()][0]
#         test_func_name = [k for k, v in namespace.items()
#                          if callable(v) and "test" in k.lower()][0]

#         print(f"✅ Syntax valid. Found functions: {reward_func_name}, {test_func_name}")

#         return {
#             "success": True,
#             "reward_func_name": reward_func_name,
#             "test_func_name": test_func_name,
#             "namespace": namespace
#         }

#     except SyntaxError as e:
#         return {
#             "success": False,
#             "error": f"Syntax error: {str(e)}",
#             "traceback": traceback.format_exc()
#         }
#     except Exception as e:
#         return {
#             "success": False,
#             "error": f"Execution error: {str(e)}",
#             "traceback": traceback.format_exc()
#         }


# def run_generated_test(namespace, test_func_name, reward_name):
#     """
#     Run the test function from generated code

#     Args:
#         namespace (dict): Namespace containing the functions
#         test_func_name (str): Name of test function
#         reward_name (str): Name identifier for this reward

#     Returns:
#         dict: Test execution results
#     """
#     print(f"\n🧪 Running generated tests for {reward_name}...")

#     try:
#         test_func = namespace[test_func_name]

#         # Capture stdout
#         from io import StringIO
#         old_stdout = sys.stdout
#         sys.stdout = captured_output = StringIO()

#         # Run the test
#         test_func()

#         # Restore stdout
#         sys.stdout = old_stdout
#         output = captured_output.getvalue()

#         print(f"✅ Generated tests passed!")
#         print(f"Test output:\n{output}")

#         return {
#             "success": True,
#             "output": output
#         }

#     except AssertionError as e:
#         sys.stdout = old_stdout
#         return {
#             "success": False,
#             "error": f"Test assertion failed: {str(e)}",
#             "traceback": traceback.format_exc()
#         }
#     except Exception as e:
#         sys.stdout = old_stdout
#         return {
#             "success": False,
#             "error": f"Test execution error: {str(e)}",
#             "traceback": traceback.format_exc()
#         }


# def create_predefined_test_scenes(room_type="bedroom"):
#     """
#     Create hand-designed test scenes for evaluation

#     Returns:
#         dict: Test scenes with descriptions
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Example for bedroom
#     idx_to_labels = {
#         0: "bed", 1: "tv_stand", 2: "armchair", 3: "nightstand",
#         4: "dresser", 5: "wardrobe", 6: "desk", 7: "chair", 8: "empty"
#     }

#     num_classes = len(idx_to_labels)
#     max_objects = 12

#     scenes = {}

#     # Scene 1: Well-organized bedroom
#     scenes["organized_bedroom"] = {
#         "description": "Well-organized bedroom with bed, tv_stand, armchair, nightstand",
#         "parsed_scene": {
#             "positions": torch.tensor([
#                 [[1.5, 0.5, 0.0], [3.5, 0.5, 0.0], [-1.0, 0.5, 1.5], [0.5, 0.3, -0.8]] +
#                 [[0.0, 0.0, 0.0]] * (max_objects - 4)
#             ], device=device),
#             "sizes": torch.tensor([
#                 [[1.0, 0.5, 0.9], [0.6, 0.5, 0.3], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]] +
#                 [[0.01, 0.01, 0.01]] * (max_objects - 4)
#             ], device=device),
#             "object_indices": torch.tensor([
#                 [0, 1, 2, 3] + [8] * (max_objects - 4)
#             ], device=device, dtype=torch.long),
#             "is_empty": torch.tensor([
#                 [False, False, False, False] + [True] * (max_objects - 4)
#             ], device=device),
#             "orientations": torch.tensor([
#                 [[1.0, 0.0], [1.0, 0.0], [0.7, 0.7], [1.0, 0.0]] +
#                 [[0.0, 0.0]] * (max_objects - 4)
#             ], device=device),
#             "one_hot": F.one_hot(
#                 torch.tensor([[0, 1, 2, 3] + [8] * (max_objects - 4)], dtype=torch.long),
#                 num_classes
#             ).float().to(device),
#             "device": device
#         },
#         "idx_to_labels": idx_to_labels
#     }

#     # Scene 2: Minimal bedroom (only bed)
#     scenes["minimal_bedroom"] = {
#         "description": "Minimal bedroom with only a bed",
#         "parsed_scene": {
#             "positions": torch.tensor([
#                 [[1.5, 0.5, 0.0]] + [[0.0, 0.0, 0.0]] * (max_objects - 1)
#             ], device=device),
#             "sizes": torch.tensor([
#                 [[1.0, 0.5, 0.9]] + [[0.01, 0.01, 0.01]] * (max_objects - 1)
#             ], device=device),
#             "object_indices": torch.tensor([
#                 [0] + [8] * (max_objects - 1)
#             ], device=device, dtype=torch.long),
#             "is_empty": torch.tensor([
#                 [False] + [True] * (max_objects - 1)
#             ], device=device),
#             "orientations": torch.tensor([
#                 [[1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 1)
#             ], device=device),
#             "one_hot": F.one_hot(
#                 torch.tensor([[0] + [8] * (max_objects - 1)], dtype=torch.long),
#                 num_classes
#             ).float().to(device),
#             "device": device
#         },
#         "idx_to_labels": idx_to_labels
#     }

#     # Scene 3: Cluttered bedroom
#     scenes["cluttered_bedroom"] = {
#         "description": "Cluttered bedroom with many furniture pieces",
#         "parsed_scene": {
#             "positions": torch.tensor([
#                 [[1.5, 0.5, 0.0], [3.5, 0.5, 0.0], [-1.0, 0.5, 1.5],
#                  [0.5, 0.3, -0.8], [-1.5, 0.8, -1.0], [2.0, 0.8, 2.0],
#                  [0.0, 0.4, 2.5], [-2.0, 0.4, 0.0]] +
#                 [[0.0, 0.0, 0.0]] * (max_objects - 8)
#             ], device=device),
#             "sizes": torch.tensor([
#                 [[1.0, 0.5, 0.9], [0.6, 0.5, 0.3], [0.4, 0.4, 0.4],
#                  [0.3, 0.3, 0.3], [0.8, 0.8, 0.5], [1.0, 0.9, 0.6],
#                  [0.6, 0.4, 0.5], [0.5, 0.4, 0.5]] +
#                 [[0.01, 0.01, 0.01]] * (max_objects - 8)
#             ], device=device),
#             "object_indices": torch.tensor([
#                 [0, 1, 2, 3, 4, 5, 6, 7] + [8] * (max_objects - 8)
#             ], device=device, dtype=torch.long),
#             "is_empty": torch.tensor([
#                 [False] * 8 + [True] * (max_objects - 8)
#             ], device=device),
#             "orientations": torch.tensor([
#                 [[1.0, 0.0], [1.0, 0.0], [0.7, 0.7], [1.0, 0.0],
#                  [0.0, 1.0], [-0.7, 0.7], [0.5, 0.87], [-1.0, 0.0]] +
#                 [[0.0, 0.0]] * (max_objects - 8)
#             ], device=device),
#             "one_hot": F.one_hot(
#                 torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7] + [8] * (max_objects - 8)], dtype=torch.long),
#                 num_classes
#             ).float().to(device),
#             "device": device
#         },
#         "idx_to_labels": idx_to_labels
#     }

#     # Scene 4: Empty bedroom
#     scenes["empty_bedroom"] = {
#         "description": "Empty bedroom with no furniture",
#         "parsed_scene": {
#             "positions": torch.tensor([
#                 [[0.0, 0.0, 0.0]] * max_objects
#             ], device=device),
#             "sizes": torch.tensor([
#                 [[0.01, 0.01, 0.01]] * max_objects
#             ], device=device),
#             "object_indices": torch.tensor([
#                 [8] * max_objects
#             ], device=device, dtype=torch.long),
#             "is_empty": torch.tensor([
#                 [True] * max_objects
#             ], device=device),
#             "orientations": torch.tensor([
#                 [[0.0, 0.0]] * max_objects
#             ], device=device),
#             "one_hot": F.one_hot(
#                 torch.tensor([[8] * max_objects], dtype=torch.long),
#                 num_classes
#             ).float().to(device),
#             "device": device
#         },
#         "idx_to_labels": idx_to_labels
#     }

#     return scenes


# def evaluate_on_predefined_scenes(namespace, reward_func_name, scenes, reward_name):
#     """
#     Evaluate reward function on predefined test scenes

#     Args:
#         namespace (dict): Namespace containing the reward function
#         reward_func_name (str): Name of reward function
#         scenes (dict): Dictionary of test scenes
#         reward_name (str): Name identifier for this reward

#     Returns:
#         dict: Evaluation results
#     """
#     print(f"\n🎯 Evaluating {reward_name} on predefined scenes...")

#     try:
#         reward_func = namespace[reward_func_name]
#         results = {}

#         for scene_name, scene_data in scenes.items():
#             print(f"\n  Testing on: {scene_data['description']}")

#             # Call reward function
#             reward = reward_func(
#                 scene_data["parsed_scene"],
#                 idx_to_labels=scene_data.get("idx_to_labels", {})
#             )

#             reward_value = reward.item() if torch.is_tensor(reward) else float(reward)
#             results[scene_name] = {
#                 "description": scene_data["description"],
#                 "reward": reward_value
#             }

#             print(f"    Reward: {reward_value:.4f}")

#         print(f"\n✅ Evaluation complete!")

#         return {
#             "success": True,
#             "results": results
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "error": f"Evaluation error: {str(e)}",
#             "traceback": traceback.format_exc()
#         }


# if __name__ == "__main__":
#     print("=" * 80)
#     print("MULTI-REWARD FUNCTION GENERATION AND TESTING")
#     print("=" * 80)

#     # Define multiple constraints to test
#     test_queries = [
#         {
#             "room_type": "bedroom",
#             "query": "a bed facing a tv stand and there should be an armchair as well"
#         },
#         {
#             "room_type": "bedroom",
#             "query": "bed should be near the wall and desk should face a window"
#         },
#         {
#             "room_type": "bedroom",
#             "query": "maximize open floor space while having essential furniture"
#         }
#     ]

#     # Create output directory
#     output_dir = Path("generated_rewards")
#     output_dir.mkdir(exist_ok=True)

#     # Store all results
#     all_results = []
#     predefined_scenes = create_predefined_test_scenes()

#     # Process each query
#     for idx, query_config in enumerate(test_queries, 1):
#         print(f"\n{'=' * 80}")
#         print(f"PROCESSING QUERY {idx}/{len(test_queries)}")
#         print(f"{'=' * 80}")
#         print(f"Room: {query_config['room_type']}")
#         print(f"Query: {query_config['query']}")

#         reward_name = f"reward_{idx}"
#         result = {"query_config": query_config, "reward_name": reward_name}

#         # Step 1: Generate code
#         gen_result = generate_reward_function_with_gemini(
#             query_config["room_type"],
#             query_config["query"]
#         )
#         result["generation"] = gen_result

#         if not gen_result["success"]:
#             print(f"❌ Generation failed: {gen_result['error']}")
#             all_results.append(result)
#             continue

#         # Save generated code
#         code_file = output_dir / f"{reward_name}.py"
#         with open(code_file, "w") as f:
#             f.write(gen_result["code"])
#         print(f"💾 Saved code to: {code_file}")

#         # Step 2: Test syntax
#         syntax_result = test_reward_function_syntax(gen_result["code"], reward_name)
#         result["syntax_test"] = syntax_result

#         if not syntax_result["success"]:
#             print(f"❌ Syntax test failed: {syntax_result['error']}")
#             all_results.append(result)
#             continue

#         # Step 3: Run generated tests
#         test_result = run_generated_test(
#             syntax_result["namespace"],
#             syntax_result["test_func_name"],
#             reward_name
#         )
#         result["generated_test"] = test_result

#         if not test_result["success"]:
#             print(f"⚠️  Generated tests failed: {test_result['error']}")
#             # Continue anyway to test on predefined scenes

#         # Step 4: Evaluate on predefined scenes
#         eval_result = evaluate_on_predefined_scenes(
#             syntax_result["namespace"],
#             syntax_result["reward_func_name"],
#             predefined_scenes,
#             reward_name
#         )
#         result["predefined_evaluation"] = eval_result

#         if not eval_result["success"]:
#             print(f"❌ Evaluation failed: {eval_result['error']}")

#         all_results.append(result)

#         # Small delay between API calls
#         if idx < len(test_queries):
#             sleep(2)

#     # Save comprehensive results
#     results_file = output_dir / "all_results.json"
#     with open(results_file, "w") as f:
#         # Convert results to JSON-serializable format
#         json_results = []
#         for r in all_results:
#             json_r = r.copy()
#             # Remove non-serializable namespace
#             if "syntax_test" in json_r and "namespace" in json_r["syntax_test"]:
#                 del json_r["syntax_test"]["namespace"]
#             json_results.append(json_r)

#         json.dump(json_results, f, indent=2)

#     print(f"\n{'=' * 80}")
#     print("SUMMARY")
#     print(f"{'=' * 80}")
#     print(f"Total queries processed: {len(all_results)}")
#     print(f"Successful generations: {sum(1 for r in all_results if r['generation']['success'])}")
#     print(f"Passed syntax tests: {sum(1 for r in all_results if r.get('syntax_test', {}).get('success'))}")
#     print(f"Passed generated tests: {sum(1 for r in all_results if r.get('generated_test', {}).get('success'))}")
#     print(f"Completed evaluations: {sum(1 for r in all_results if r.get('predefined_evaluation', {}).get('success'))}")
#     print(f"\n📁 All results saved to: {results_file}")
#     print(f"📁 Generated code saved to: {output_dir}/")
#     print(f"\n{'=' * 80}")


"""
Multi-Reward Function Generator for 3D Scene Generation

This module generates multiple reward function variations using Google's Gemini API,
validates them through compilation and runtime testing, and evaluates them against
predefined test scenes.

Author: Research Team
Date: 2025
"""

import importlib
import json
import os
import sys

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from create_prompt import create_prompt
from dotenv import load_dotenv

# ============================================================================
# Configuration and Constants
# ============================================================================


class Config:
    """Configuration constants for the reward function generator."""

    # API Configuration
    GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
    MODEL_NAME = "gemini-2.5-flash"
    API_RATE_LIMIT_DELAY = 1.0  # seconds between API calls

    # File paths
    OUTPUT_REWARDS_FILE = "ashok_test_mutable_rewards.py"
    OUTPUT_RESULTS_FILE = "multi_reward_results.json"
    PREDEFINED_SCENES_FILE = "predefined_test_scenes.json"

    # Generation settings
    DEFAULT_NUM_VARIATIONS = 1
    REWARD_FUNCTION_NAME_PREFIX = "compute_reward_v"
    MAX_RETRIES_PER_VARIATION = 3  # Maximum retries if generation/compilation fails
    RETRY_DELAY = 2.0  # seconds between retries

    # Code formatting
    PYTHON_CODE_MARKERS = ["```python", "```"]
    INDENT_SPACES = 4


class CompilationStatus(Enum):
    """Status codes for compilation results."""

    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    MISSING_FUNCTION = "missing_function"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RewardVariation:
    """Represents a single reward function variation."""

    variation_id: int
    raw_code: str
    success: bool
    compile_status: Optional[CompilationStatus] = None
    compile_message: Optional[str] = None
    compile_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.compile_status:
            data["compile_status"] = self.compile_status.value
        return data


@dataclass
class SceneTestResult:
    """Result of testing a reward function on a scene."""

    scene_name: str
    success: bool
    reward: Optional[float] = None
    error: Optional[str] = None


@dataclass
class GenerationResults:
    """Complete results from reward function generation and testing."""

    instructions: Dict[str, str]
    variations: List[RewardVariation]
    scene_test_results: Dict[str, List[SceneTestResult]]

    @property
    def total_generated(self) -> int:
        return len(self.variations)

    @property
    def successfully_compiled(self) -> int:
        return sum(
            1 for v in self.variations if v.compile_status == CompilationStatus.SUCCESS
        )

    @property
    def failed_compilation(self) -> int:
        return self.total_generated - self.successfully_compiled


# ============================================================================
# Utility Functions
# ============================================================================


def setup_api() -> None:
    """Initialize and configure the Gemini API."""
    load_dotenv()
    api_key = os.getenv(Config.GOOGLE_API_KEY_ENV)

    if not api_key:
        raise ValueError(
            f"API key not found. Please set {Config.GOOGLE_API_KEY_ENV} "
            "in your environment or .env file"
        )

    genai.configure(api_key=api_key)


def clean_code_response(response_text: str) -> str:
    """
    Clean up code response by removing markdown code blocks.

    Args:
        response_text: Raw response text from API

    Returns:
        Cleaned code string
    """
    text = response_text.strip()

    # Remove code block markers
    for marker in Config.PYTHON_CODE_MARKERS:
        if text.startswith(marker):
            text = text[len(marker) :]
        if text.endswith(marker):
            text = text[: -len(marker)]

    return text.strip()


# ============================================================================
# Core Generation Functions
# ============================================================================


class RewardFunctionGenerator:
    """Handles generation of multiple reward function variations."""

    def __init__(self, model_name: str = Config.MODEL_NAME):
        """
        Initialize the generator.

        Args:
            model_name: Name of the Gemini model to use
        """
        self.model = genai.GenerativeModel(model_name)

    def generate_variations(
        self,
        room_type: str,
        query: str,
        num_variations: int = Config.DEFAULT_NUM_VARIATIONS,
    ) -> List[RewardVariation]:
        """
        Generate multiple reward function variations with retry logic.

        Args:
            room_type: Type of room (e.g., "bedroom", "livingroom")
            query: User query describing desired layout
            num_variations: Number of variations to generate

        Returns:
            List of RewardVariation objects
        """
        variations = []

        print(f"\nGenerating {num_variations} reward function variation(s)...")
        print(f"Room Type: {room_type}")
        print(f"Query: {query}\n")

        for i in range(num_variations):
            variation_num = i + 1
            print(f"\n{'='*60}")
            print(f"VARIATION {variation_num}/{num_variations}")
            print(f"{'='*60}")

            # Try generating and validating this variation with retries
            variation = self._generate_with_retry(
                room_type, query, variation_num, num_variations
            )
            variations.append(variation)

            # Rate limiting between variations
            if i < num_variations - 1:
                sleep(Config.API_RATE_LIMIT_DELAY)

        return variations

    def _generate_with_retry(
        self, room_type: str, query: str, variation_num: int, total_variations: int
    ) -> RewardVariation:
        """
        Generate a single variation with retry logic if it fails.

        Args:
            room_type: Type of room
            query: User query
            variation_num: Current variation number
            total_variations: Total number of variations

        Returns:
            RewardVariation object (may be failed after all retries)
        """
        for attempt in range(1, Config.MAX_RETRIES_PER_VARIATION + 1):
            try:
                print(f"\nAttempt {attempt}/{Config.MAX_RETRIES_PER_VARIATION}:")

                # Generate the code
                variation = self._generate_single_variation(
                    room_type, query, variation_num, total_variations, attempt
                )

                if not variation.success:
                    print(f"  ✗ Generation failed")
                    if attempt < Config.MAX_RETRIES_PER_VARIATION:
                        print(f"  ⟳ Retrying in {Config.RETRY_DELAY}s...")
                        sleep(Config.RETRY_DELAY)
                        continue
                    return variation

                # Validate the generated code
                print(f"  ✓ Code generated successfully")
                print(f"  ⚙ Validating code...", end=" ")

                status, message, error = CodeValidator.validate_code(
                    variation.raw_code, variation_num
                )

                variation.compile_status = status
                variation.compile_message = message
                variation.compile_error = error

                if status == CompilationStatus.SUCCESS:
                    print(f"✓")
                    print(f"  ✅ Variation {variation_num} successful!")
                    return variation
                else:
                    print(f"✗")
                    print(f"  ✗ Validation failed: {status.value}")
                    if error:
                        print(f"     Error: {error}")

                    if attempt < Config.MAX_RETRIES_PER_VARIATION:
                        print(
                            f"  ⟳ Retrying with different approach in {Config.RETRY_DELAY}s..."
                        )
                        sleep(Config.RETRY_DELAY)
                    else:
                        print(
                            f"  ❌ All {Config.MAX_RETRIES_PER_VARIATION} attempts failed"
                        )
                        return variation

            except Exception as e:
                print(f"  ✗ Exception during attempt {attempt}: {str(e)}")
                if attempt < Config.MAX_RETRIES_PER_VARIATION:
                    print(f"  ⟳ Retrying in {Config.RETRY_DELAY}s...")
                    sleep(Config.RETRY_DELAY)
                else:
                    print(f"  ❌ All {Config.MAX_RETRIES_PER_VARIATION} attempts failed")
                    return RewardVariation(
                        variation_id=variation_num,
                        raw_code="",
                        success=False,
                        compile_error=f"All attempts failed. Last error: {str(e)}",
                    )

        # Should not reach here, but return failed variation as fallback
        return RewardVariation(
            variation_id=variation_num,
            raw_code="",
            success=False,
            compile_error="Maximum retries exceeded",
        )

    def _generate_single_variation(
        self,
        room_type: str,
        query: str,
        variation_num: int,
        total_variations: int,
        attempt: int = 1,
    ) -> RewardVariation:
        """Generate a single reward function variation."""
        print(f"    Calling Gemini API...", end=" ")

        # Create enhanced prompt with attempt-specific guidance
        base_prompt = create_prompt(room_type, query)
        enhanced_prompt = self._create_variation_prompt(
            base_prompt, variation_num, attempt
        )

        # Generate response
        response = self.model.generate_content(enhanced_prompt)
        code = clean_code_response(response.text)

        print("✓")

        return RewardVariation(variation_id=variation_num, raw_code=code, success=True)

    @staticmethod
    def _create_variation_prompt(
        base_prompt: str, variation_num: int, attempt: int = 1
    ) -> str:
        """Create an enhanced prompt for generating unique variations."""
        retry_guidance = ""
        if attempt > 1:
            retry_guidance = f"""
IMPORTANT - RETRY ATTEMPT {attempt}:
The previous attempt failed validation. Please ensure:
1. ALL required imports are included (torch, torch.nn.functional as F, math, numpy)
2. The function signature exactly matches: compute_reward(parsed_scene, **kwargs)
3. All code is syntactically correct Python
4. No undefined variables or functions are used
5. The function returns a torch.Tensor of shape (B,)
6. Include proper test function with if __name__ == "__main__" block

"""

        return f"""{base_prompt}

{retry_guidance}
VARIATION #{variation_num}:
Make this variation unique by:
- Using different weighting strategies for rewards
- Emphasizing different spatial relationships or constraints
- Applying different penalty/reward magnitude scales
- Considering alternative constraint satisfaction approaches

Return ONLY the complete Python code with imports, reward function, and test cases.
No explanations, no markdown formatting beyond code blocks."""


# ============================================================================
# Compilation and Validation
# ============================================================================


class CodeValidator:
    """Validates and compiles reward function code."""

    @staticmethod
    def validate_code(
        code: str, variation_id: int
    ) -> Tuple[CompilationStatus, str, Optional[str]]:
        """
        Validate and compile reward function code.

        Args:
            code: Python code string
            variation_id: ID of the variation

        Returns:
            Tuple of (status, message, error_details)
        """
        try:
            # Compile the code
            compile(code, f"<reward_function_v{variation_id}>", "exec")

            # Execute in isolated namespace
            namespace = {}
            exec(code, namespace)

            # Check for required function
            if "compute_reward" not in namespace and not any(
                name.startswith("compute_") and "reward" in name
                for name in namespace.keys()
            ):
                return (
                    CompilationStatus.MISSING_FUNCTION,
                    "Required compute_reward function not found",
                    None,
                )

            return (
                CompilationStatus.SUCCESS,
                "Code compiled and validated successfully",
                None,
            )

        except SyntaxError as e:
            error_msg = f"Line {e.lineno}, Column {e.offset}: {e.msg}"
            return (
                CompilationStatus.SYNTAX_ERROR,
                "Syntax error in generated code",
                error_msg,
            )

        except Exception as e:
            return (
                CompilationStatus.RUNTIME_ERROR,
                "Runtime error during validation",
                str(e),
            )

    @classmethod
    def validate_variations(cls, variations: List[RewardVariation]) -> None:
        """
        Validate all variations and update their status.

        Args:
            variations: List of RewardVariation objects to validate
        """
        print("\nValidating and compiling variations...")

        for var in variations:
            if not var.success:
                print(f"Variation {var.variation_id}: ⊘ SKIPPED (generation failed)")
                continue

            status, message, error = cls.validate_code(var.raw_code, var.variation_id)

            var.compile_status = status
            var.compile_message = message
            var.compile_error = error

            status_symbol = "✓" if status == CompilationStatus.SUCCESS else "✗"
            print(
                f"Variation {var.variation_id}: {status_symbol} {status.value.upper()}"
            )

            if error:
                print(f"  └─ Error: {error}")


# ============================================================================
# Code Generation and File Writing
# ============================================================================


class CodeWriter:
    """Writes validated reward functions to Python module."""

    @staticmethod
    def write_rewards_module(
        variations: List[RewardVariation], output_file: str = Config.OUTPUT_REWARDS_FILE
    ) -> int:
        """
        Write all valid reward functions to a Python module.

        Args:
            variations: List of RewardVariation objects
            output_file: Path to output file

        Returns:
            Number of functions written
        """
        valid_variations = [
            v for v in variations if v.compile_status == CompilationStatus.SUCCESS
        ]

        if not valid_variations:
            print("\n⚠ No valid variations to write!")
            return 0

        print(f"\nWriting {len(valid_variations)} valid variations to {output_file}...")

        with open(output_file, "w", encoding="utf-8") as f:
            CodeWriter._write_module_header(f)

            for var in valid_variations:
                CodeWriter._write_variation(f, var)

        print(f"✓ Successfully wrote module with {len(valid_variations)} functions")
        return len(valid_variations)

    @staticmethod
    def _write_module_header(file_handle) -> None:
        """Write module header and imports."""
        file_handle.write('"""\n')
        file_handle.write("Auto-generated Reward Functions\n")
        file_handle.write("\n")
        file_handle.write("Generated by Multi-Reward Function Generator\n")
        file_handle.write("DO NOT EDIT MANUALLY\n")
        file_handle.write('"""\n\n')
        file_handle.write("import torch\n")
        file_handle.write("import torch.nn.functional as F\n")
        file_handle.write("import numpy as np\n")
        file_handle.write("import math\n")
        file_handle.write("from typing import Dict, Any\n\n\n")

    @staticmethod
    def _write_variation(file_handle, variation: RewardVariation) -> None:
        """Write a single reward function variation."""
        file_handle.write(f'# {"=" * 70}\n')
        file_handle.write(f"# VARIATION {variation.variation_id}\n")
        file_handle.write(f'# {"=" * 70}\n\n')

        # Parse and rewrite the code with proper function naming
        lines = variation.raw_code.split("\n")
        in_function = False
        function_found = False

        for line in lines:
            stripped = line.strip()

            # Detect function definition
            if stripped.startswith("def compute_") and "reward" in stripped.lower():
                if not function_found:
                    # Rename the function
                    func_name = f"compute_reward_v{variation.variation_id}"
                    # Extract parameters
                    params_start = stripped.find("(")
                    params = (
                        stripped[params_start:]
                        if params_start != -1
                        else "(parsed_scene, **kwargs):"
                    )
                    file_handle.write(f"def {func_name}{params}\n")
                    in_function = True
                    function_found = True
                    continue

            # Write the line
            file_handle.write(line + "\n")

        file_handle.write("\n\n")


# ============================================================================
# Scene Testing
# ============================================================================


class SceneTester:
    """Tests reward functions against predefined scenes."""

    def __init__(self, scenes_file: str = Config.PREDEFINED_SCENES_FILE):
        """
        Initialize scene tester.

        Args:
            scenes_file: Path to predefined scenes JSON file
        """
        self.scenes_file = scenes_file
        self.scenes = self._load_or_create_scenes()

    def _load_or_create_scenes(self) -> Dict[str, Any]:
        """Load predefined scenes or create defaults."""
        if os.path.exists(self.scenes_file):
            print(f"\nLoading test scenes from {self.scenes_file}")
            with open(self.scenes_file, "r") as f:
                return json.load(f)
        else:
            print(f"\n⚠ {self.scenes_file} not found. Creating default scenes...")
            scenes = self._create_default_scenes()
            with open(self.scenes_file, "w") as f:
                json.dump(scenes, f, indent=2)
            print(f"✓ Created {self.scenes_file}")
            return scenes

    @staticmethod
    def _create_default_scenes() -> Dict[str, Any]:
        """Create default test scenes for bedroom layout."""
        return {
            "optimal_layout": {
                "description": "Optimal bedroom layout with all objects properly arranged",
                "objects": [
                    {
                        "type": "bed",
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0],
                        "size": [2, 1, 1.5],
                    },
                    {
                        "type": "tv_stand",
                        "position": [3, 0, 0],
                        "rotation": [0, 180, 0],
                        "size": [1, 0.5, 0.5],
                    },
                    {
                        "type": "armchair",
                        "position": [1, 0, 2],
                        "rotation": [0, -45, 0],
                        "size": [0.8, 0.8, 0.8],
                    },
                ],
                "room_dimensions": [5, 3, 3],
            },
            "poor_layout": {
                "description": "Poor layout with objects not properly arranged",
                "objects": [
                    {
                        "type": "bed",
                        "position": [0, 0, 0],
                        "rotation": [0, 90, 0],
                        "size": [2, 1, 1.5],
                    },
                    {
                        "type": "tv_stand",
                        "position": [0.5, 0, 0.5],
                        "rotation": [0, 0, 0],
                        "size": [1, 0.5, 0.5],
                    },
                    {
                        "type": "armchair",
                        "position": [4, 0, 0],
                        "rotation": [0, 180, 0],
                        "size": [0.8, 0.8, 0.8],
                    },
                ],
                "room_dimensions": [5, 3, 3],
            },
            "missing_object": {
                "description": "Layout missing required armchair",
                "objects": [
                    {
                        "type": "bed",
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0],
                        "size": [2, 1, 1.5],
                    },
                    {
                        "type": "tv_stand",
                        "position": [3, 0, 0],
                        "rotation": [0, 180, 0],
                        "size": [1, 0.5, 0.5],
                    },
                ],
                "room_dimensions": [5, 3, 3],
            },
            "cluttered_layout": {
                "description": "Overcrowded layout with overlapping objects",
                "objects": [
                    {
                        "type": "bed",
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0],
                        "size": [2, 1, 1.5],
                    },
                    {
                        "type": "tv_stand",
                        "position": [0.5, 0, 0],
                        "rotation": [0, 180, 0],
                        "size": [1, 0.5, 0.5],
                    },
                    {
                        "type": "armchair",
                        "position": [0.2, 0, 0.2],
                        "rotation": [0, -45, 0],
                        "size": [0.8, 0.8, 0.8],
                    },
                ],
                "room_dimensions": [5, 3, 3],
            },
        }

    def test_variations(
        self, variations: List[RewardVariation]
    ) -> Dict[str, List[SceneTestResult]]:
        """
        Test all valid variations against predefined scenes.

        Args:
            variations: List of RewardVariation objects

        Returns:
            Dictionary mapping variation names to test results
        """
        print("\nTesting variations against predefined scenes...")

        # Import the generated module
        try:
            import ashok_test_mutable_rewards as reward_module

            importlib.reload(reward_module)
        except Exception as e:
            print(f"✗ Error importing reward module: {str(e)}")
            return {}

        results = {}
        valid_variations = [
            v for v in variations if v.compile_status == CompilationStatus.SUCCESS
        ]

        for var in valid_variations:
            var_name = f"variation_{var.variation_id}"
            func_name = f"{Config.REWARD_FUNCTION_NAME_PREFIX}{var.variation_id}"

            if not hasattr(reward_module, func_name):
                print(f"⚠ Function {func_name} not found in module")
                continue

            print(f"\nTesting {var_name}...")
            compute_reward = getattr(reward_module, func_name)

            scene_results = []
            for scene_name, scene_data in self.scenes.items():
                result = self._test_single_scene(compute_reward, scene_name, scene_data)
                scene_results.append(result)

                status = "✓" if result.success else "✗"
                reward_str = (
                    f"{result.reward:.4f}" if result.reward is not None else "N/A"
                )
                print(f"  {status} {scene_name}: {reward_str}")

            results[var_name] = scene_results

        return results

    @staticmethod
    def _test_single_scene(
        compute_reward, scene_name: str, scene_data: Dict[str, Any]
    ) -> SceneTestResult:
        """Test a single scene with a reward function."""
        try:
            reward = compute_reward(scene_data)
            return SceneTestResult(
                scene_name=scene_name, success=True, reward=float(reward)
            )
        except Exception as e:
            return SceneTestResult(scene_name=scene_name, success=False, error=str(e))


# ============================================================================
# Results Management
# ============================================================================


class ResultsManager:
    """Manages and saves generation results."""

    @staticmethod
    def save_results(
        results: GenerationResults, output_file: str = Config.OUTPUT_RESULTS_FILE
    ) -> None:
        """
        Save generation results to JSON file.

        Args:
            results: GenerationResults object
            output_file: Path to output file
        """
        output_data = {
            "instructions": results.instructions,
            "variations": [var.to_dict() for var in results.variations],
            "scene_test_results": {
                var_name: [
                    {
                        "scene_name": r.scene_name,
                        "success": r.success,
                        "reward": r.reward,
                        "error": r.error,
                    }
                    for r in results_list
                ]
                for var_name, results_list in results.scene_test_results.items()
            },
            "summary": {
                "total_generated": results.total_generated,
                "successfully_compiled": results.successfully_compiled,
                "failed_compilation": results.failed_compilation,
            },
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

    @staticmethod
    def print_summary(results: GenerationResults) -> None:
        """Print a formatted summary of results."""
        print("\n" + "=" * 70)
        print("GENERATION SUMMARY")
        print("=" * 70)
        print(f"Total variations generated: {results.total_generated}")
        print(f"Successfully compiled: {results.successfully_compiled}")
        print(f"Failed compilation: {results.failed_compilation}")

        if results.scene_test_results:
            print("\n" + "-" * 70)
            print("SCENE TEST RESULTS")
            print("-" * 70)

            for var_name, scene_results in results.scene_test_results.items():
                print(f"\n{var_name}:")
                for result in scene_results:
                    if result.success:
                        print(f"  ✓ {result.scene_name:.<40} {result.reward:>8.4f}")
                    else:
                        print(f"  ✗ {result.scene_name:.<40} ERROR")
                        if result.error:
                            print(f"    └─ {result.error}")

        print("\n" + "=" * 70)


# ============================================================================
# Main Pipeline
# ============================================================================


def run_generation_pipeline(
    room_type: str, query: str, num_variations: int = Config.DEFAULT_NUM_VARIATIONS
) -> GenerationResults:
    """
    Run the complete reward function generation pipeline.

    Args:
        room_type: Type of room
        query: User query describing desired layout
        num_variations: Number of variations to generate

    Returns:
        GenerationResults object with all results
    """
    print("=" * 70)
    print("MULTI-REWARD FUNCTION GENERATOR")
    print("=" * 70)

    # Setup API
    setup_api()

    # Step 1: Generate variations (with built-in validation and retry)
    generator = RewardFunctionGenerator()
    variations = generator.generate_variations(room_type, query, num_variations)

    # Step 2: Write valid variations to file
    writer = CodeWriter()
    num_written = writer.write_rewards_module(variations)

    # Step 3: Test with scenes
    scene_results = {}
    if num_written > 0:
        tester = SceneTester()
        scene_results = tester.test_variations(variations)

    # Create results object
    results = GenerationResults(
        instructions={"room_type": room_type, "query": query},
        variations=variations,
        scene_test_results=scene_results,
    )

    return results


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Main entry point for the script."""
    # Configuration
    room_type = "bedroom"
    query = "a bed facing a tv stand and there should be an armchair as well"
    num_variations = 1  # Generate 1 variation with automatic retry on failure

    try:
        # Run pipeline
        results = run_generation_pipeline(room_type, query, num_variations)

        # Save and display results
        ResultsManager.save_results(results)
        ResultsManager.print_summary(results)

        # Print retry statistics
        print("\n" + "=" * 70)
        print("RETRY STATISTICS")
        print("=" * 70)
        for var in results.variations:
            status = (
                "SUCCESS"
                if var.compile_status == CompilationStatus.SUCCESS
                else "FAILED"
            )
            print(f"Variation {var.variation_id}: {status}")
            if var.compile_error:
                print(f"  Final error: {var.compile_error}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
