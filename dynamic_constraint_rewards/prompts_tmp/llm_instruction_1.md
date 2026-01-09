
    # TASK: Constraint Decomposition for 3D Scene Generation

    You are an expert in 3D scene generation, interior design, and reinforcement learning. Your task is to analyze a user prompt and decompose it into verifiable constraints with Python reward functions.

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
    

    Note: While generating constraints, no need to verify these facts with you constraints, focus on the constraints other than these facts.

    In this task, you are provided with a user prompt and a dataset context. Your task is to decompose the user prompt into verifiable constraints with Python reward functions.

    Here is the dataset information in JSON format about the specific room type: bedroom you will be working on:
    ```json
    {'room_type': 'bedroom', 'total_scenes': 4042, 'class_frequencies': {'nightstand': 0.27245508982035926, 'double_bed': 0.17138137518067315, 'wardrobe': 0.16079909147222796, 'pendant_lamp': 0.12693578360520338, 'ceiling_lamp': 0.06308073508156102, 'tv_stand': 0.029888498864340286, 'chair': 0.022816436093330582, 'single_bed': 0.021216188313029113, 'dressing_table': 0.020854842040057817, 'cabinet': 0.020183770390253975, 'table': 0.019667561428866404, 'desk': 0.016260582283708445, 'stool': 0.011459838942804047, 'shelf': 0.0081561015899236, 'kids_bed': 0.0081561015899236, 'bookshelf': 0.0071753045632872185, 'children_cabinet': 0.0071753045632872185, 'dressing_chair': 0.006142886640512079, 'armchair': 0.003716704521990502, 'sofa': 0.0014970059880239522, 'coffee_table': 0.0009807970266363824}, 'furniture_counts': {'nightstand': 5278, 'double_bed': 3320, 'wardrobe': 3115, 'pendant_lamp': 2459, 'ceiling_lamp': 1222, 'tv_stand': 579, 'chair': 442, 'single_bed': 411, 'dressing_table': 404, 'cabinet': 391, 'table': 381, 'desk': 315, 'stool': 222, 'shelf': 158, 'kids_bed': 158, 'bookshelf': 139, 'children_cabinet': 139, 'dressing_chair': 119, 'armchair': 72, 'sofa': 29, 'coffee_table': 19}, 'idx_to_labels': {0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk', 8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed', 12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe', 21: 'empty'}, 'num_classes_with_empty': 22, 'num_classes_without_empty': 21, 'max_objects': 12}
    ```

    Also, the baseline model is already trained on some universal constraints, so you do not need to consider these constraints while generating new ones. The universal constraints are:
    ```json
    {'non_penetration': {'function': 'compute_non_penetration_reward', 'description': '\n    Calculate reward based on non-penetration constraint using penetration depth.\n\n    Following the approach from original authors: reward = sum of negative signed distances.\n    When objects overlap, we get positive penetration depth, so reward is negative.\n\n    Args:\n        parsed_scene: Dict returned by parse_and_descale_scenes()\n\n    Returns:\n        rewards: Tensor of shape (B,) with non-penetration rewards for each scene\n    '}, 'not_out_of_bound': {'function': 'compute_boundary_violation_reward', 'description': "\n    Compute boundary violation reward using cached SDF grids.\n\n    **IMPORTANT**: Call `precompute_sdf_cache()` once before training to generate cache!\n\n    Args:\n        parsed_scene: Dictionary with positions, sizes, is_empty, device\n        floor_polygons: (B, num_vertices, 2) - only needed if cache doesn't exist\n        indices: (B,) - scene indices for SDF lookup\n        grid_resolution: SDF grid resolution\n        sdf_cache_dir: Directory containing cached SDF grids\n\n    Returns:\n        rewards: (B, 1) - sum of negative violation distances per scene\n    "}, 'accessibility': {'function': 'compute_accessibility_reward', 'description': "\n    Compute accessibility reward using cached floor grids or computing on-the-fly.\n\n    Returns dict with 3 components:\n    - coverage_ratio: [0, 1] - fraction of floor reachable from largest region\n    - num_regions: [1, ∞) - number of disconnected regions\n    - avg_clearance: meters - average distance to nearest obstacle in reachable area\n\n    Args:\n        parsed_scenes: Dictionary with positions, sizes, is_empty, device, object_types\n        floor_polygons: (B, num_vertices, 2) - floor polygon vertices\n        is_val: Whether this is validation split\n        indices: (B,) - scene indices for cache lookup\n        accessibility_cache: Pre-loaded AccessibilityCache instance (optional)\n        grid_resolution: Grid resolution in meters (default 0.2m = 20cm)\n        agent_radius: Agent radius in meters (default 0.15m = 15cm)\n        save_viz: Whether to save visualization images\n        viz_dir: Directory to save visualizations\n\n    Returns:\n        Dictionary with:\n        - 'coverage_ratio': (B,) - reachable area ratio [0, 1]\n        - 'num_regions': (B,) - number of disconnected regions [1, ∞)\n        - 'avg_clearance': (B,) - average clearance in meters\n    "}, 'gravity_following': {'function': 'compute_gravity_following_reward', 'description': '\n    Calculate gravity-following reward based on how close objects are to the ground.\n\n    Objects should rest on the floor (y_min ≈ 0), except for ceiling objects.\n    Only penalizes objects that are MORE than tolerance away from the floor(both sinking and floating cases).\n\n    Args:\n        parsed_scene: Dict returned by parse_and_descale_scenes()\n        tolerance: Distance threshold in meters (default 0.01m = 1cm)\n\n    Returns:\n        rewards: Tensor of shape (B,) with gravity-following rewards\n    '}}
    ```

    ## YOUR TASK

    Analyze the user prompt and provide a comprehensive JSON response with the following structure:

    ### 1. CONSTRAINT DECOMPOSITION

    Generate ALL constraints needed to satisfy the prompt strictly in following format.

    ```json
    {
    "constraints": [
        {
        "id": "C1",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        },
        {
        "id": "C2",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        },
        ...
        {
        "id": "Cn",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        }
    ]
    }
    ```
    