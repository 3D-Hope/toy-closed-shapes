
    # TASK: Constraints to reward code mapping

    You are an expert in 3D scene generation, interior design, and reinforcement learning.
    Your task is to analyze given constraints and convert them into verifiable reward functions in Python.

    ## CONTEXT

    ### Dataset: 3D-FRONT
    
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
    

    In this task, you are provided with a user prompt and a dataset context. Your task is to decompose the user prompt into verifiable constraints with Python reward functions.

    Here is the dataset information in JSON format about the specific room type: bedroom you will be working on:
    ```json
    {'room_type': 'bedroom', 'total_scenes': 4042, 'class_frequencies': {'nightstand': 0.27245508982035926, 'double_bed': 0.17138137518067315, 'wardrobe': 0.16079909147222796, 'pendant_lamp': 0.12693578360520338, 'ceiling_lamp': 0.06308073508156102, 'tv_stand': 0.029888498864340286, 'chair': 0.022816436093330582, 'single_bed': 0.021216188313029113, 'dressing_table': 0.020854842040057817, 'cabinet': 0.020183770390253975, 'table': 0.019667561428866404, 'desk': 0.016260582283708445, 'stool': 0.011459838942804047, 'shelf': 0.0081561015899236, 'kids_bed': 0.0081561015899236, 'bookshelf': 0.0071753045632872185, 'children_cabinet': 0.0071753045632872185, 'dressing_chair': 0.006142886640512079, 'armchair': 0.003716704521990502, 'sofa': 0.0014970059880239522, 'coffee_table': 0.0009807970266363824}, 'furniture_counts': {'nightstand': 5278, 'double_bed': 3320, 'wardrobe': 3115, 'pendant_lamp': 2459, 'ceiling_lamp': 1222, 'tv_stand': 579, 'chair': 442, 'single_bed': 411, 'dressing_table': 404, 'cabinet': 391, 'table': 381, 'desk': 315, 'stool': 222, 'shelf': 158, 'kids_bed': 158, 'bookshelf': 139, 'children_cabinet': 139, 'dressing_chair': 119, 'armchair': 72, 'sofa': 29, 'coffee_table': 19}, 'idx_to_labels': {0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk', 8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed', 12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe', 21: 'empty'}, 'num_classes_with_empty': 22, 'num_classes_without_empty': 21, 'max_objects': 12}
    ```

    ### Scene Representation
    
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
    

    ### You also have the following utility functions at your disposal which you can use according to the given docstrings.
    ```json
    {'find_object_front_and_back': {'function': 'find_object_front_and_back', 'description': '\n    Find the coordinates of the front and back centers of a object.\n\n    Args:\n        position: (1,3) tensor - object centroid (x, y, z)\n        orientation: (1,2) tensor - [cos(θ), sin(θ)], z-rotation\n        size: (1,3) tensor - half-extents (sx/2, sy/2, sz/2)\n    \n    Returns:\n        front_center: (1,3) tensor - position of object front\n        back_center: (1,3) tensor - position of object back\n    '}, 'find_closest_wall_to_object': {'function': 'find_closest_wall_to_object', 'description': "\n    Find which wall is closest to the object's front or back and compute its distance.\n\n    Args:\n        position: (1,3) tensor - object centroid (x, y, z)\n        orientation: (1,2) tensor - z-rotation\n        size: (1,3) tensor - half-extents (sx/2, sy/2, sz/2)\n        floor_polygons: list of ordered floor polygon vertices in the format [(x1, z1), (x2, z2), ...(xn, zn)]  where n >= 4, and always forms a closed polygon\n    \n    Returns:\n        wall_index: (1) tensor - index of the wall in floor_polygons (i.e., wall_index = 0 means the wall formed by floor_polygons[0] and floor_polygons[1])\n        distance: (1) tensor - perpendicular distance from object centroid to wall\n    "}, 'get_object_count_in_a_scene': {'function': 'get_object_count_in_a_scene', 'description': '\n    Count number of objects of a specific class in each scene.\n\n    Args:\n        one_hot: (B, N, num_classes) - One-hot encoded classes\n        class_label: string, e.g. "ceiling_lamp"\n        idx_to_labels: dict, {idx: label}\n\n    Returns:\n        count: int, number of objects of class_label in the scene\n    '}, 'create_scene_for_testing': {'function': 'create_scene_for_testing', 'description': '\n    Create a scene for testing reward functions.\n    Input:\n        room_type: string, Example: "bedroom" or "livingroom"\n        num_objects: int, number of objects in the scene\n        class_label_indices: list of int, class indices\n        translations: list of tuple, (x, y, z) translations\n        sizes: list of tuple, (sx/2, sy/2, sz/2) sizes\n        orientations: list of tuple, (cos(θ), sin(θ)) orientations\n        \n    Output:\n        parsed_scene: dict, scene representation\n    '}}
    ```

    ### The baseline model is already trained on some universal constraints, so you do not need to consider these constraints while generating new ones. The universal constraints are:
    ```json
    {'non_penetration': {'function': 'compute_non_penetration_reward', 'description': '\n    Calculate reward based on non-penetration constraint using penetration depth.\n\n    Following the approach from original authors: reward = sum of negative signed distances.\n    When objects overlap, we get positive penetration depth, so reward is negative.\n\n    Args:\n        parsed_scene: Dict returned by parse_and_descale_scenes()\n\n    Returns:\n        rewards: Tensor of shape (B,) with non-penetration rewards for each scene\n    '}, 'not_out_of_bound': {'function': 'compute_boundary_violation_reward', 'description': "\n    Compute boundary violation reward using cached SDF grids.\n\n    **IMPORTANT**: Call `precompute_sdf_cache()` once before training to generate cache!\n\n    Args:\n        parsed_scene: Dictionary with positions, sizes, is_empty, device\n        floor_polygons: (B, num_vertices, 2) - only needed if cache doesn't exist\n        indices: (B,) - scene indices for SDF lookup\n        grid_resolution: SDF grid resolution\n        sdf_cache_dir: Directory containing cached SDF grids\n\n    Returns:\n        rewards: (B, 1) - sum of negative violation distances per scene\n    "}, 'accessibility': {'function': 'compute_accessibility_reward', 'description': "\n    Compute accessibility reward using cached floor grids or computing on-the-fly.\n\n    Returns dict with 3 components:\n    - coverage_ratio: [0, 1] - fraction of floor reachable from largest region\n    - num_regions: [1, ∞) - number of disconnected regions\n    - avg_clearance: meters - average distance to nearest obstacle in reachable area\n\n    Args:\n        parsed_scenes: Dictionary with positions, sizes, is_empty, device, object_types\n        floor_polygons: (B, num_vertices, 2) - floor polygon vertices\n        is_val: Whether this is validation split\n        indices: (B,) - scene indices for cache lookup\n        accessibility_cache: Pre-loaded AccessibilityCache instance (optional)\n        grid_resolution: Grid resolution in meters (default 0.2m = 20cm)\n        agent_radius: Agent radius in meters (default 0.15m = 15cm)\n        save_viz: Whether to save visualization images\n        viz_dir: Directory to save visualizations\n\n    Returns:\n        Dictionary with:\n        - 'coverage_ratio': (B,) - reachable area ratio [0, 1]\n        - 'num_regions': (B,) - number of disconnected regions [1, ∞)\n        - 'avg_clearance': (B,) - average clearance in meters\n    "}}
    ```

    ## YOUR TASK

    Analyze the user prompt, constraints to be satisfied for that prompt and all other context i have provided, then provide a comprehensive JSON response with the following structure:

    The template for reward function to quantify each constraint satisfaction with python code is as follows:
    
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
        parsed_scenes = {
            k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
            for k in tensor_keys
        }
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
    

    Note: While using the utility functions, you can use the following code snippet:

    ```python
    from dynamic_constraint_rewards.utilities import get_all_utility_functions

    utility_functions = get_all_utility_functions()
    return_values = utility_functions["function_name"]["function"](required_arguments(from docstring), **kwargs)
    ```

    Also, passing all required parameters to the utlity functions is a must
    (Example: don't miss room_type for create_scene_for_testing)

    Also, Given the reward constraints, analyze and if there are constraints like: a scene must have n number of objects of a particular class, then inpaint those objects. To inpaint, pass the class labels and counts in the json format as specified below.

    Example: If R1 = "a scene must have exactly 4 ceiling lamps", R2 = "a scene must have exactly 2 nightstands", then inpaint the objects with:

    ```
    "inpaint": {
    "ceiling_lamp": 4,
    "nightstand": 2
    }
    ```

    Also, success_threshold is a float type, that indicates the constraint is satisfied (if unnormalized_raw_reward_value >= success_threshold)

    Very Very Important, all rewards should be bounded and within a reasonable range even if some samples are anamolous.
    
    Never ever give huge numbers like ('inf') in the rewards. even anomalous samples should be given rewards that are worse but capped at some reasonable value.
    
    Very Very Important, you should be mindful of the scale of the rewards for different test cases according to the particular reward function. ocassionally the assertions may fail because you expected different magnitude of the reward not because the scene does not satisfy the constraint. Therefore, in case of such assertions fails you need to print the informative message mentioning the expected and actual reward values for each test case so that you can debug and adjust the reward function accordingly in the next iteration.
    Only return the following JSON response (nothing else), follow this structure strictly:

    ```json
    {
    "rewards": [
        {
        "id": "R1",
        "constraint_id": "C1",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        },
        {
        "id": "R2",
        "constraint_id": "C2",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        },
        ...
        {
        "id": "Rn",
        "constraint_id": "Cn",
        "name": "descriptive_snake_case_name",
        "code": "Python Code implementing get_reward and test_reward functions as per the template",
        "success_threshold": "Value in terms of raw reward units as implemented in Python code indicating satisfactory fulfillment of the constraint. This will be used to calculate success rate."
        }
    ],
    "inpaint": {
    "class_label1": count1,
    "class_label2": count2,
    ...,
    "class_labeln": countn
    }
    }
    ```

    NOTE: Even if you have inpainted objects due to some constraints, keep those constraints in the rewards list.
    NOTE: If you are passing any other arguments other than specified in the function descriptions, make sure to get it from the kwargs dictionary. (kwargs.get("argument_name"))
    NOTE: You should use the utility functions exactly as the docstrings provided, all arguments should be passed in the same order as in the docstrings. (followed by kwargs if required)
    