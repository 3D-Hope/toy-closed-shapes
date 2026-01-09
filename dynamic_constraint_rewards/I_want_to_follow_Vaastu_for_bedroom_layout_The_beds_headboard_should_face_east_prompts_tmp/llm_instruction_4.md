
    # TASK: Assigning inportance weights to each of dynamic and universal reward components.

    You are an expert in 3D scene generation, interior design, and reinforcement learning.
    Your task is to analyze the user prompt, final constraints, final dynamic and universal reward functions. Then, return the weights to be applied to each of the rewards while training the reinforcement learning model.

    Now, as an reinforcement learning expert in reward shaping if any of the reward functions conflict then according to the desired behaviour as specified in the user prompt, return the weights to be applied to each of the rewards such that the final reward value will be the most suitable while training the reinforcement learning model.

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
    


    ### The baseline model is already trained on some universal constraints and they are used as part of reward functions as well for regularization,


    ## YOUR TASK
    Analyze the original user prompt, final constraints, final dynamic reward functions and universal reward functions, then provide a comprehensive JSON response with the following structure.

    It should be noted that each reward components is converted to the range [-1, 1] before applying the weighted sum so the weights should purely be based on the importance of the rewards. This task is aimed to reduce conflicting rewards because some dynamic reward may try to conflict with these universal ones.

    ---
    Only return the following JSON response (nothing else), follow this structure strictly:

    ```json
    {
    "importance_weights": {
        "reward_name1": weight1(float),
        "reward_name2": weight2(float),
        ...
        "reward_namen": weightn(float)
    }
    }
    ```

    