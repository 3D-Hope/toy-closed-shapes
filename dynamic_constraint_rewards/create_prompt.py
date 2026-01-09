import json
from universal_constraint_rewards.commons import get_all_universal_reward_functions
from dynamic_constraint_rewards.utilities import get_all_utility_functions
from dynamic_constraint_rewards.commons import save_reward_functions, verify_tests_for_reward_function, get_stats_from_initial_rewards
import hydra
from omegaconf import DictConfig, OmegaConf
from steerable_scene_generation.utils.omegaconf import register_resolvers


dataset_facts = """
    The dataset being used is 3D-FRONT which uses 3D-FUTURE dataset for furniture models. 3D-FRONT is a collection of synthetic, high-quality 3D indoor scenes, highlighted by professionally and distinctively designed layouts.

    In this dataset, the following facts are important to know:

    ## Coordinate System
    - Y-axis: Vertical (up direction)
    - XZ-plane: Floor plane
    - Units: Meters (world coordinates, unnormalized)
    - Empty slots: Have index (num_classes-1), near-zero size/position

    # Important Facts about 3D-FRONT dataset
    - Ceiling objects are at y ≈ ceiling_height (typically 2.8m)
    - Floor objects have y ≈ object_height/2
    - Ignore empty slots (is_empty == True) in calculations
    - East direction is along +X axis and West direction is along -X axis
    - North direction is along +Z axis and South direction is along -Z axis
    """

scene_representation = """
    A 3D scene is represented in batch format (parsed_scenes) a dictionary with the following keys and PyTorch tensors as values:
        - `positions`: (B, N, 3) - Object centroids in meters (x, y, z)
        - `sizes`: (B, N, 3) - Half-extents (sx/2, sy/2, sz/2)
        - `object_indices`: (B, N) - Class indices [0, num_classes-1]
        - `one_hot`: (B, N, num_classes) - One-hot encoded classes
        - `is_empty`: (B, N) - Boolean mask (True = empty slot)
        - `orientations`: (B, N, 2) - [cos(θ), sin(θ)] for z-rotation
        - `device`: torch.device
        Where:
            - B = Batch size
            - N = Max objects per scene
    """

room_stats = {
    "bedroom": {
        "max_objects": 12,
        "num_classes": 22,
        "num_classes_with_empty": 22,
        "num_classes_without_empty": 21,
    },
    "livingroom": {
        "max_objects": 21,
        "num_classes": 25,
        "num_classes_with_empty": 25,
        "num_classes_without_empty": 24,
    }
}

reward_function_template = f"""
    ```python
    def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
        '''
        Input:
            - parsed_scenes: list of parsed scenes
                Format:
                Scenes are provided as dictionaries with PyTorch tensors:
                    - `positions`: (B, N, 3) - Object centroids in meters (x, y, z)
                    - `sizes`: (B, N, 3) - Half-extents (sx/2, sy/2, sz/2)
                    - `object_indices`: (B, N) - Class indices [0, num_classes-1]
                    - `one_hot`: (B, N, num_classes) - One-hot encoded classes
                    - `is_empty`: (B, N) - Boolean mask (True = empty slot)
                    - `orientations`: (B, N, 2) - [cos(θ), sin(θ)] for z-rotation
                    - `device`: torch.device
                    Where:
                        - B = Batch size
                        - N = Max objects per scene
            
            - idx_to_labels: dictionary mapping class indices to class labels
            - room_type: string, Example: "bedroom" or "livingroom"
            - Floor Polygons (floor_polygons): A list of ordered floor_polygons in the format [(x1, z1), (x2, z2), ...(xn, zn)]  where n >= 4, and always forms a closed polygon
            - **kwargs: additional keyword arguments

        Output:
            reward: torch.Tensor of shape (len(parsed_scenes),)
        '''
        
        # Logic of reward function here
        return reward

    def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
        '''
        Input:
            - idx_to_labels: dictionary mapping class indices to class labels
            - room_type: string, Example: "bedroom" or "livingroom"
            - floor_polygons: A list of ordered floor_polygons in the format [(x1, z1), (x2, z2), ...(xn, zn)]  where n >= 4, and always forms a closed polygon
            - **kwargs: additional keyword arguments
        '''
        # Create some test scenes using create_scene_for_testing
        # Scene 1
        num_objects_1 = 5
        class_label_indices_1 = [0, 1, 2, 3, 4]
        translations_1 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
        sizes_1 = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        orientations_1 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
        scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
        
        # Scene 2
        num_objects_2 = 6
        class_label_indices_2 = [0, 1, 2, 3, 4, 5]
        translations_2 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0),]
        sizes_2 = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        orientations_2 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
        scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
        
        # Scene 3
        num_objects_3 = 4
        class_label_indices_3 = [0, 1, 3, 4]
        translations_3 = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        sizes_3 = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        orientations_3 = [(1, 0), (1, 0), (1, 0), (1, 0)]
        scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
        
        
        # Stack each key of the parsed_scene dicts into a batched dict
        tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
        parsed_scenes = {{
            k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
            for k in tensor_keys
        }}
        parsed_scenes['room_type'] = room_type
        parsed_scenes['device'] = scene_1['device']
        
        rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
        print("Rewards:", rewards)
        assert rewards.shape[0] == len(parsed_scenes)
        
        # You have to add different test cases here to verify that the reward function is working as expected
        assert <TEST_CASE_1>
        assert <TEST_CASE_2>
        ...
        assert <TEST_CASE_N>
    ```
    """


def get_dataset_context(room_type):
    if room_type == "bedroom":
        dataset_context = {  
            "room_type": "bedroom",
            "total_scenes": 4042,
            "class_frequencies": {
                "nightstand": 0.27245508982035926,
                "double_bed": 0.17138137518067315,
                "wardrobe": 0.16079909147222796,
                "pendant_lamp": 0.12693578360520338,
                "ceiling_lamp": 0.06308073508156102,
                "tv_stand": 0.029888498864340286,
                "chair": 0.022816436093330582,
                "single_bed": 0.021216188313029113,
                "dressing_table": 0.020854842040057817,
                "cabinet": 0.020183770390253975,
                "table": 0.019667561428866404,
                "desk": 0.016260582283708445,
                "stool": 0.011459838942804047,
                "shelf": 0.0081561015899236,
                "kids_bed": 0.0081561015899236,
                "bookshelf": 0.0071753045632872185,
                "children_cabinet": 0.0071753045632872185,
                "dressing_chair": 0.006142886640512079,
                "armchair": 0.003716704521990502,
                "sofa": 0.0014970059880239522,
                "coffee_table": 0.0009807970266363824
            },
            "furniture_counts": {
                "nightstand": 5278,
                "double_bed": 3320,
                "wardrobe": 3115,
                "pendant_lamp": 2459,
                "ceiling_lamp": 1222,
                "tv_stand": 579,
                "chair": 442,
                "single_bed": 411,
                "dressing_table": 404,
                "cabinet": 391,
                "table": 381,
                "desk": 315,
                "stool": 222,
                "shelf": 158,
                "kids_bed": 158,
                "bookshelf": 139,
                "children_cabinet": 139,
                "dressing_chair": 119,
                "armchair": 72,
                "sofa": 29,
                "coffee_table": 19
            },
            "idx_to_labels": {
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
                21: "empty"
            },
            "num_classes_with_empty": 22,
            "num_classes_without_empty": 21,
            "room_type": "bedroom",
            "max_objects": 12
        }
    elif room_type == "livingroom":
        dataset_context = {
            "room_type": "livingroom",
        "total_scenes": 2926,
        "class_frequencies": {
                "dining_chair": 0.25492085340674464,
                "pendant_lamp": 0.13282863041982107,
                "coffee_table": 0.08616655196145905,
                "corner_side_table": 0.07240192704748796,
                "dining_table": 0.06951135581555402,
                "tv_stand": 0.06221610461114935,
                "multi_seat_sofa": 0.05299380591878871,
                "armchair": 0.048313833448038544,
                "console_table": 0.037026841018582245,
                "lounge_chair": 0.03234686854783207,
                "stool": 0.0264280798348245,
                "cabinet": 0.023124569855471438,
                "bookshelf": 0.02202339986235375,
                "loveseat_sofa": 0.020922229869236062,
                "ceiling_lamp": 0.018169304886441844,
                "wine_cabinet": 0.012112869924294563,
                "l_shaped_sofa": 0.01032346868547832,
                "round_end_table": 0.0057811424638678595,
                "shelf": 0.0035788024776324846,
                "chinese_chair": 0.0031658637302133517,
                "wardrobe": 0.0027529249827942187,
                "chaise_longue_sofa": 0.0011011699931176876,
                "desk": 0.0009635237439779766,
                "lazy_sofa": 0.0008258774948382657
            },
            "furniture_counts": {
                "dining_chair": 1852,
                "pendant_lamp": 965,
                "coffee_table": 626,
                "corner_side_table": 526,
                "dining_table": 505,
                "tv_stand": 452,
                "multi_seat_sofa": 385,
                "armchair": 351,
                "console_table": 269,
                "lounge_chair": 235,
                "stool": 192,
                "cabinet": 168,
                "bookshelf": 160,
                "loveseat_sofa": 152,
                "ceiling_lamp": 132,
                "wine_cabinet": 88,
                "l_shaped_sofa": 75,
                "round_end_table": 42,
                "shelf": 26,
                "chinese_chair": 23,
                "wardrobe": 20,
                "chaise_longue_sofa": 8,
                "desk": 7,
                "lazy_sofa": 6
            },
            "idx_to_labels": {
                0: "armchair",
                1: "bookshelf",
                2: "cabinet",
                3: "ceiling_lamp",
                4: "chaise_longue_sofa",
                5: "chinese_chair",
                6: "coffee_table",
                7: "console_table",
                8: "corner_side_table",
                9: "desk",
                10: "dining_chair",
                11: "dining_table",
                12: "l_shaped_sofa",
                13: "lazy_sofa",
                14: "lounge_chair",
                15: "loveseat_sofa",
                16: "multi_seat_sofa",
                17: "pendant_lamp",
                18: "round_end_table",
                19: "shelf",
                20: "stool",
                21: "tv_stand",
                22: "wardrobe",
                23: "wine_cabinet",
                24: "empty"
                },
            "num_classes_with_empty": 25,
            "num_classes_without_empty": 24,
            "room_type": "livingroom",
            "max_objects": 21
        }
    else:
        raise ValueError(f"Room type {room_type} not supported")

    return dataset_context

def get_universal_rewards_info_with_docstrings():
    universal_rewards_info = get_all_universal_reward_functions()
    universal_rewards_info_with_docstrings = {}

    for reward_name, reward_info in universal_rewards_info.items():
        universal_rewards_info_with_docstrings[reward_name] = {
            "function": reward_info.__name__,
            "description": reward_info.__doc__,
        }
    return universal_rewards_info_with_docstrings

def create_prompt_1(user_prompt, room_type):
    dataset_context = get_dataset_context(room_type)
    universal_rewards_info_with_docstrings = get_universal_rewards_info_with_docstrings()
    
    # Avoid using f-string here to prevent invalid format specifier error
    llm_instruction_1 = f"""
    # TASK: Constraint Decomposition for 3D Scene Generation

    You are an expert in 3D scene generation, interior design, and reinforcement learning. Your task is to analyze a user prompt and decompose it into verifiable constraints with Python reward functions.

    ## CONTEXT

    ### Dataset: 3D-FRONT
    {dataset_facts}

    Note: While generating constraints, no need to verify these facts with your constraints, focus on the constraints other than these facts.

    In this task, you are provided with a user prompt and a dataset context. Your task is to decompose the user prompt into verifiable constraints.

    Here is the dataset information in JSON format about the specific room type: {room_type} you will be working on:
    ```json
    {dataset_context}
    ```

    Also, the baseline model is already trained on some universal constraints, so you do not need to consider these constraints while generating new ones. The universal constraints are:
    ```json
    {universal_rewards_info_with_docstrings}
    ```

    ## YOUR TASK

    Analyze the user prompt and provide a comprehensive JSON response with the following structure:

    ### 1. CONSTRAINT DECOMPOSITION

    Generate ALL constraints needed to satisfy the prompt strictly in following format.

    ```json
    {{
    "constraints": [
        {{
        "id": "C1",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        }},
        {{
        "id": "C2",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        }},
        ...
        {{
        "id": "Cn",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        }}
    ]
    }}
    ```
    """
    
    llm_user_prompt_1 = f"""
    User Prompt: {user_prompt}
    """
    return llm_instruction_1, llm_user_prompt_1

def create_prompt_2(user_prompt, constraints, room_type):
    dataset_context = get_dataset_context(room_type)
    utility_functions = get_all_utility_functions(is_prompt=True)
    universal_rewards_info_with_docstrings = get_universal_rewards_info_with_docstrings()
    llm_instruction_2 = f"""
    # TASK: Constraints to reward code mapping

    You are an expert in 3D scene generation, interior design, and reinforcement learning.
    Your task is to analyze given constraints and convert them into verifiable reward functions in Python.

    ## CONTEXT

    ### Dataset: 3D-FRONT
    {dataset_facts}

    In this task, you are provided with a user prompt and a dataset context. Your task is to decompose the user prompt into verifiable constraints with Python reward functions.

    Here is the dataset information in JSON format about the specific room type: {room_type} you will be working on:
    ```json
    {dataset_context}
    ```

    ### Scene Representation
    {scene_representation}

    ### You also have the following utility functions at your disposal which you can use according to the given docstrings.
    ```json
    {utility_functions}
    ```

    ### The baseline model is already trained on some universal constraints, so you do not need to consider these constraints while generating new ones. The universal constraints are:
    ```json
    {universal_rewards_info_with_docstrings}
    ```

    ## YOUR TASK

    Analyze the user prompt, constraints to be satisfied for that prompt and all other context i have provided, then provide a comprehensive JSON response with the following structure:

    The template for reward function to quantify each constraint satisfaction with python code is as follows:
    {reward_function_template}

    Note: While using the utility functions, you can use the following code snippet:

    ```python
    from dynamic_constraint_rewards.utilities import get_all_utility_functions

    utility_functions = get_all_utility_functions()
    return_values = utility_functions["function_name"]["function"](required_arguments(from docstring), **kwargs)
    ```

    Also, passing all required parameters to the utlity functions is a must
    (Example: don't miss room_type for create_scene_for_testing)



    Also, success_threshold is a float type, that indicates the constraint is satisfied (if unnormalized_raw_reward_value >= success_threshold)

    Very Very Important, all rewards should be bounded and within a reasonable range even if some samples are anamolous.
    
    Never ever give huge numbers like ('inf') in the rewards. even anomalous samples should be given rewards that are worse but capped at some reasonable value.
    
    Very Very Important, you should be mindful of the scale of the rewards for different test cases according to the particular reward function. ocassionally the assertions may fail because you expected different magnitude of the reward not because the scene does not satisfy the constraint. Therefore, in case of such assertions fails you need to print the informative message mentioning the expected and actual reward values for each test case so that you can debug and adjust the reward function accordingly in the next iteration.
    Only return the following JSON response (nothing else), follow this structure strictly:

    ```json
    {{
    "rewards": [
        {{
        "id": "R1",
        "constraint_id": "C1",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }},
        {{
        "id": "R2",
        "constraint_id": "C2",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }},
        ...
        {{
        "id": "Rn",
        "constraint_id": "Cn",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }}
    ]
    }}
    ```

    NOTE: If you are passing any other arguments other than specified in the function descriptions, make sure to get it from the kwargs dictionary. (kwargs.get("argument_name"))
    NOTE: You should use the utility functions exactly as the docstrings provided, all arguments should be passed in the same order as in the docstrings. (followed by kwargs if required)
    """
    llm_user_prompt_2 = f"""
    User Prompt: {user_prompt}
    Constraints to be satisfied: {constraints}
    """
    return llm_instruction_2, llm_user_prompt_2

def create_prompt_3(user_prompt, constraints, reward_functions, room_type):
    import os
    stats = ""
    base_dir = os.path.dirname(__file__)
    user_query = user_prompt.replace(' ', '_').replace('.', '')
    for txt_file in os.listdir(os.path.join(base_dir, f"{user_query}_reward_analysis_txt")):
        if txt_file.endswith(".txt"):
            with open(os.path.join(os.path.join(base_dir, f"{user_query}_reward_analysis_txt"), txt_file), "r") as f:
                content = f.read()
                stats += f"\n\n--- Stats from {txt_file} ---\n"
                stats += content
    dataset_context = get_dataset_context(room_type)
    utility_functions = get_all_utility_functions(is_prompt=True)
    universal_rewards_info_with_docstrings = get_universal_rewards_info_with_docstrings()

    llm_instruction_3 = f"""
    # TASK: Reward Functions Finetuning based on reward statistics

    You are an expert in 3D scene generation, interior design, and reinforcement learning.
    Your task is to analyze the user prompt, initial constraints, initial reward functions and the statistics of those reward functions on entire dataset as well as on synthetic scenes generated from baseline model (1000 scenes). Then, return the new constraints, new reward functions based on the analysis.

    Now, as an rl expert in reward shaping, think about the curriculum to teach the baseline diffusion model using rl post training. We do rl post training iteratively according to the curriculum. For now you have to prepare reward functions for the first iteration of rl post training. after finishing first iteration of rl post training, you will analyze the results and prepare for the next iteration of rl post training. Our goal is to gradually improve the model to satisfy the user prompt strictly this iterative agentic approach should achieve better results than trying to achieve everything in one go.
    
    ## CONTEXT

    ### Dataset: 3D-FRONT
    {dataset_facts}

    Here is the dataset information in JSON format about the specific room type: {room_type} you will be working on:
    ```json
    {dataset_context}
    ```

    ### Scene Representation
    {scene_representation}

    ### You also have the following utility functions at your disposal which you can use according to the given docstrings.
    ```json
    {utility_functions}
    ```

    ### The baseline model is already trained on some universal constraints, so you do not need to consider these constraints while generating new ones. The universal constraints are:
    ```json
    {universal_rewards_info_with_docstrings}
    ```

    ## YOUR TASK
    Analyze the user prompt, initial constraints, initial reward functions, the statistics of those reward functions on entire dataset as well as on synthetic scenes generated from baseline model (1000 scenes) and all other context i have provided, then provide a comprehensive JSON response with the following structure:

    The template for reward function to quantify each constraint satisfaction with python code is as follows:
    {reward_function_template}

    Note: While using the utility functions, you can use the following code snippet:

    ```python
    from dynamic_constraint_rewards.utilities import get_all_utility_functions

    utility_functions = get_all_utility_functions()
    return_values = utility_functions["function_name"]["function"](required_arguments(from docstring), **kwargs)
    ```

    Also, passing all required parameters to the utlity functions is a must
    (Example: don't miss room_type for create_scene_for_testing)


    Also, success_threshold is a float type, that indicates the constraint is satisfied (if unnormalized_raw_reward_value >= success_threshold)
    ---

    Note: Here is the example of the reward statistics format after evaluating the initial reward functions on the baseline model generated scenes (1000 scenes) as well as on the entire dataset for the given room type:

    --- Stats from R1_ceiling_lamps_count_llm_summary_baseline.txt ---                                                 

    === REWARD ANALYSIS: R1_ceiling_lamps_count ===                                                                    

    PERFORMANCE METRICS: • Success Rate: 100.0% (1000/1000 scenes) • Mean Reward: 0.0000 • Median Reward: 0.0000 •     
    Range: [0.0000, 0.0000] • Std Dev: 0.0000 • Percentiles: - P1: 0.0000 - P5: 0.0000 - P25: 0.0000 - P75: 0.0000 -   
    P95: 0.0000 - P99: 0.0000                                                                                          

    DISTRIBUTION CHARACTERISTICS:                                                                                      

    • Skewness: nan • Kurtosis: nan • Min Rate: 100.0% • Near Min Rate: 100.0% • Max Rate: 100.0% • Near Max Rate:     
    100.0%                                                                                                             

    ============================================================                                                       

    --- Stats from R1_ceiling_lamps_count_llm_summary_dataset.txt ---                                                  

    === REWARD ANALYSIS: R1_ceiling_lamps_count ===                                                                    

    PERFORMANCE METRICS: • Success Rate: 0.0% (0/4041 scenes) • Mean Reward: -3.6847 • Median Reward: -4.0000 • Range: 
    [-4.0000, -1.0000] • Std Dev: 0.4746 • Percentiles: - P1: -4.0000 - P5: -4.0000 - P25: -4.0000 - P75: -3.0000 -    
    P95: -3.0000 - P99: -3.0000                                                                                        

    DISTRIBUTION CHARACTERISTICS:                                                                                      

    • Skewness: 0.94 • Kurtosis: -0.66 • Min Rate: 68.9% • Near Min Rate: 68.9% • Max Rate: 0.0% • Near Max Rate: 0.0% 

    Where, stats from txt file with _baseline.txt eg R1_ceiling_lamps_count_llm_summary_baseline.txt indicates the reward statistics on the baseline model generated scenes (1000 scenes) and
    stats from txt file with _dataset.txt eg R1_ceiling_lamps_count_llm_summary_dataset.txt indicates the reward statistics on the entire dataset for the given room type.

    Success Rate = number of scenes satisfying the constraint / total number of scenes * 100%
    Mean Reward = average reward value across all scenes
    Median Reward = median reward value across all scenes
    Range = [minimum reward value, maximum reward value]
    Std Dev = standard deviation of reward values
    Percentiles = reward values at different percentiles (P1, P5, P25, P75, P95, P99)
    Skewness = measure of asymmetry of the reward
    Kurtosis = measure of "tailedness" of the reward distribution
    Near Max Rate = percentage of scenes with near-maximum reward
    Near Min Rate = percentage of scenes with near-minimum reward
    Max Rate = percentage of scenes with maximum reward
    Min Rate = percentage of scenes with minimum reward

    Ignore the nan values in the statistics(if any).

    Very Very Important, all rewards should be bounded and within a reasonable range even if some samples are anamolous.
    Never ever give huge numbers like ('inf') in the rewards. even anomalous samples should be given rewards that are worse but capped at some reasonable value.    
    
    Very Very Important, you should be mindful of the scale of the rewards for different test cases according to the particular reward function. ocassionally the assertions may fail because you expected different magnitude of the reward not because the scene does not satisfy the constraint. Therefore, in case of such assertions fails you need to print the informative message mentioning the expected and actual reward values for each test case so that you can debug and adjust the reward function accordingly in the next iteration.
    
    NOTE: Very important, while generating new constraints and reward functions, only provide those for the stage1 of curriculum you generated.
    ---
    Only return the following JSON response (nothing else), follow this structure strictly:

    ```json
    {{
    "curriculum": {{
       "stage1": "description of what to focus on in this stage",
        "stage2": "description of what to focus on in this stage",
        ...
        "stagen": "description of what to focus on in this stage" 
    }},
    "constraints": [
        {{
        "id": "SC1",
        "name": "descriptive_snake_case_name",
        "description": "Detailed description of the constraint in natural language explaining what needs to be satisfied in the scene."
        }},
        {{
        "id": "SC2",
        "name": "descriptive_snake_case_name",
        "description": "Detailed description of the constraint in natural language explaining what needs to be satisfied in the scene."
        }},
        ...
        {{
        "id": "SCn",
        "name": "descriptive_snake_case_name",
        "description": "Detailed description of the constraint in natural language explaining what needs to be satisfied in the scene."
        }}
    ]
    "rewards": [
        {{
        "id": "SR1",
        "constraint_id": "SC1",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }},
        {{
        "id": "SR2",
        "constraint_id": "SC2",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }},
        ...
        {{
        "id": "SRn",
        "constraint_id": "SCn",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }}
    ],
    }}
    ```
    
    NOTE: If you are passing any other arguments other than specified in the function descriptions, make sure to get it from the kwargs dictionary. (kwargs.get("argument_name"))
    NOTE: You should use the utility functions exactly as the docstrings provided, all arguments should be passed in the same order as in the docstrings. (followed by kwargs if required)
    """
    
    llm_user_prompt_3 = f"""
        User Prompt: {user_prompt}
        Initial Constraints: {constraints}
        Initial Reward Functions: {reward_functions}
        Reward Statistics = {stats}
    """
    return llm_instruction_3, llm_user_prompt_3

def create_prompt_4(user_prompt, final_constraints_and_dynamic_rewards, room_type):
    dataset_context = get_dataset_context(room_type)
    universal_rewards_info_with_docstrings = get_universal_rewards_info_with_docstrings()
    llm_instruction_4 = f"""
    # TASK: Assigning inportance weights to each of dynamic and universal reward components.

    You are an expert in 3D scene generation, interior design, and reinforcement learning.
    Your task is to analyze the user prompt, final constraints, final dynamic and universal reward functions. Then, return the weights to be applied to each of the rewards while training the reinforcement learning model.

    Now, as an reinforcement learning expert in reward shaping if any of the reward functions conflict then according to the desired behaviour as specified in the user prompt, return the weights to be applied to each of the rewards such that the final reward value will be the most suitable while training the reinforcement learning model.

    ## CONTEXT

    ### Dataset: 3D-FRONT
    {dataset_facts}

    Here is the dataset information in JSON format about the specific room type: {room_type} you will be working on:
    ```json
    {dataset_context}
    ```

    ### Scene Representation
    {scene_representation}


    ### The baseline model is already trained on some universal constraints and they are used as part of reward functions as well for regularization,


    ## YOUR TASK
    Analyze the original user prompt, final constraints, final dynamic reward functions and universal reward functions, then provide a comprehensive JSON response with the following structure.

    It should be noted that each reward components is converted to the range [-1, 1] before applying the weighted sum so the weights should purely be based on the importance of the rewards. This task is aimed to reduce conflicting rewards because some dynamic reward may try to conflict with these universal ones.

    ---
    Only return the following JSON response (nothing else), follow this structure strictly:

    ```json
    {{
    "importance_weights": {{
        "reward_name1": weight1(float),
        "reward_name2": weight2(float),
        ...
        "reward_namen": weightn(float)
    }}
    }}
    ```

    """
    llm_user_prompt_4 = f"""
        Original User Prompt: {user_prompt}
        Final Constraints: {final_constraints_and_dynamic_rewards["constraints"]}
        Final Dynamic Reward Functions: {final_constraints_and_dynamic_rewards["rewards"]}
        Final Universal Reward Functions: {universal_rewards_info_with_docstrings}
    """
    return llm_instruction_4, llm_user_prompt_4


def create_prompt_5(user_prompt, reward_reflection):
    llm_instruction_5 = f"""
    # TASK: Generating a list of reward functions to follow the user instruction.
    
    We have successfully trained a RL policy of stage 1 of the provided curriculum using the provided reward function code.
    
    Please carefully analyze the policy feedback and return the new constraints and reward functions for next stage of the curriculum that
    can better solve the task.
    
    ## YOUR TASK
    Only return the following JSON response (nothing else), follow this structure strictly:
    
    # TODO: Finalize this format
    ```json
    {{
    "curriculum": {{
       "stage1": "description of what to focus on in this stage",
        "stage2": "description of what to focus on in this stage",
        ...
        "stagen": "description of what to focus on in this stage" 
    }},
    "constraints": [
        {{
        "id": "SC1_stage2",
        "name": "descriptive_snake_case_name",
        "description": "Detailed description of the constraint in natural language explaining what needs to be satisfied in the scene."
        }},
        {{
        "id": "SC2_stage2",
        "name": "descriptive_snake_case_name",
        "description": "Detailed description of the constraint in natural language explaining what needs to be satisfied in the scene."
        }},
        ...
        {{
        "id": "SCn_stage2",
        "name": "descriptive_snake_case_name",
        "description": "Detailed description of the constraint in natural language explaining what needs to be satisfied in the scene."
        }}
    ]
    "rewards": [
        {{
        "id": "SR1_stage2",
        "constraint_id": "SC1_stage2",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }},
        {{
        "id": "SR2_stage2",
        "constraint_id": "SC2_stage2",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }},
        ...
        {{
        "id": "SRn_stage2",
        "constraint_id": "SCn_stage2",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }}
    ],
    }}
    ```
    
    
    """
    llm_user_prompt_5 = f"""
        User Prompt: {user_prompt}
        Reward Reflection: {reward_reflection}
        # TODO: What to add here?
    """
    return llm_instruction_5, llm_user_prompt_5



@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig):
    register_resolvers()
    OmegaConf.resolve(cfg)
    
    USER_PROMPT = "A classroom for 10 students."
    ROOM_TYPE = "livingroom"
    
    llm_instruction_1, llm_user_prompt_1 = create_prompt_1(USER_PROMPT, ROOM_TYPE)
    
    with open("prompts_tmp/llm_instruction_1.json", "w") as f:
        json.dump(llm_instruction_1, f)
    with open("prompts_tmp/llm_user_prompt_1.json", "w") as f:
        json.dump(llm_user_prompt_1, f)
    # CALL THE LLM HERE
    llm_response_1 = None
    
    constraints = None
    
    if constraints is not None:
        with open("responses_tmp/llm_response_1.json", "w") as f:
            json.dump(constraints, f)
    else:
        with open("responses_tmp/llm_response_1.json", "r") as f:
            constraints = json.load(f)
        
    llm_instruction_2, llm_user_prompt_2 = create_prompt_2(USER_PROMPT, constraints, ROOM_TYPE)
    with open("prompts_tmp/llm_instruction_2.json", "w") as f:
        json.dump(llm_instruction_2, f)
    with open("prompts_tmp/llm_user_prompt_2.json", "w") as f:
        json.dump(llm_user_prompt_2, f)
    
    # CALL THE LLM HERE
    llm_response_2 = None
    
    reward_functions = None
    
    if reward_functions is not None:
        with open("responses_tmp/llm_response_2.json", "w") as f:
            json.dump(reward_functions, f)
    else:
        with open("responses_tmp/llm_response_2.json", "r") as f:
            reward_functions = json.load(f)
    with open("prompts_tmp/llm_instruction_3.json", "w") as f:
        json.dump(llm_instruction_3, f)
    with open("prompts_tmp/llm_user_prompt_3.json", "w") as f:
        json.dump(llm_user_prompt_3, f)
        
    saved_reward_functions = save_reward_functions(reward_functions)
    
    if not saved_reward_functions:
        raise ValueError("Failed to save reward functions")
    
    verified_tests = verify_tests_for_reward_function(ROOM_TYPE, output_dir="dynamic_reward_function_initial")
    if not verified_tests:
        raise ValueError("Failed to verify tests for reward functions")
    
    dataset_stats, baseline_stats = get_stats_from_initial_rewards(reward_functions, cfg, output_dir="dynamic_reward_function_initial")
    if dataset_stats is None or baseline_stats is None:
        raise ValueError("Failed to get stats from initial rewards")
    
    print(f"Dataset stats: {dataset_stats}")
    print(f"Baseline stats: {baseline_stats}")
    
    # Stats shoulde be saved to txt files by now

    llm_instruction_3, llm_user_prompt_3 = create_prompt_3(USER_PROMPT, constraints, reward_functions, ROOM_TYPE)
    
    with open("prompts_tmp/llm_instruction_3.json", "w") as f:
        json.dump(llm_instruction_3, f)
    with open("prompts_tmp/llm_user_prompt_3.json", "w") as f:
        json.dump(llm_user_prompt_3, f)
    
    # CALL THE LLM HERE
    llm_response_3 = None
    
    final_constraints_and_dynamic_rewards = None
    
    if final_constraints_and_dynamic_rewards is not None:
        with open("responses_tmp/llm_response_3.json", "w") as f:
            json.dump(final_constraints_and_dynamic_rewards, f)
    else:
        with open("responses_tmp/llm_response_3.json", "r") as f:
            final_constraints_and_dynamic_rewards = json.load(f)
    
    llm_instruction_4, llm_user_prompt_4 = create_prompt_4(USER_PROMPT, final_constraints_and_dynamic_rewards, ROOM_TYPE)
    with open("prompts_tmp/llm_instruction_4.json", "w") as f:
        json.dump(llm_instruction_4, f)
    with open("prompts_tmp/llm_user_prompt_4.json", "w") as f:
        json.dump(llm_user_prompt_4, f)
        
    # CALL THE LLM HERE
    llm_response_4 = None
    
    reward_weights = None
    
    if reward_weights is not None:
        with open("responses_tmp/llm_response_4.json", "w") as f:
            json.dump(reward_weights, f)
    else:
        with open("responses_tmp/llm_response_4.json", "r") as f:
            reward_weights = json.load(f)
            
    print(f"Reward weights: {reward_weights}")
    
    print("Successfully completed the task")