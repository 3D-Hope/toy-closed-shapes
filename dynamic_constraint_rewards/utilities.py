# utility functions for reward calculations
import torch

# Function templates for utility functions used in reward computations
def find_object_front_and_back(position, orientation, size):
    """
    Find the coordinates of the front and back centers of a object.

    Args:
        position: (1,3) tensor - object centroid (x, y, z)
        orientation: (1,2) tensor - [cos(θ), sin(θ)], z-rotation
        size: (1,3) tensor - half-extents (sx/2, sy/2, sz/2)
    
    Returns:
        front_center: (1,3) tensor - position of object front
        back_center: (1,3) tensor - position of object back
    """
    
    position_x, position_y, position_z = position[0]
    orientation_cos, orientation_sin = orientation[0]
    size_x, _, size_z = size[0]

    front_center = torch.tensor([position_x + size_x * orientation_cos, position_y, position_z + size_z * orientation_sin], device=position.device)
    back_center = torch.tensor([position_x - size_x * orientation_cos, position_y, position_z - size_z * orientation_sin], device=position.device)
    return front_center, back_center

def find_closest_wall_to_object(position, orientation, size, floor_polygons):
    """
    Find which wall is closest to the object's front or back and compute its distance.

    Args:
        position: (1,3) tensor - object centroid (x, y, z)
        orientation: (1,2) tensor - z-rotation
        size: (1,3) tensor - half-extents (sx/2, sy/2, sz/2)
        floor_polygons: list of ordered floor polygon vertices in the format [(x1, z1), (x2, z2), ...(xn, zn)]  where n >= 4, and always forms a closed polygon
    
    Returns:
        wall_index: (1) tensor - index of the wall in floor_polygons (i.e., wall_index = 0 means the wall formed by floor_polygons[0] and floor_polygons[1])
        distance: (1) tensor - perpendicular distance from object centroid to wall
    """
    front_center, back_center = find_object_front_and_back(position, orientation, size)
    distances = []
    for i in range(len(floor_polygons)):
        point1 = floor_polygons[i]
        point2 = floor_polygons[(i+1)%len(floor_polygons)]
        midpoint = (point1 + point2) / 2
        distance_front = distance_2d(front_center, midpoint)
        distance_back = distance_2d(back_center, midpoint)
        distances.append(min(distance_front, distance_back))
    return distances.index(min(distances)), min(distances)

def compute_angle_between_objects(orientation1, orientation2):
    """
    Calculate angle in degrees between two objects in xz-plane.

    Args:
        orientation1: orientation of object 1,  (2,) tensor - [cos(θ), sin(θ)], z-rotation
        orientation2: orientation of object 2, (2,) tensor - [cos(θ), sin(θ)], z-rotation

    Returns:
        angle_radians: (,) tensor - angle between objects in radians
    """
    # Calculate angle in degrees between the facing direction vectors of two objects in xz-plane
    # (i.e., project orientation onto xz-plane, compute relative angle)
    orientation1_angle = torch.atan2(orientation1[1], orientation1[0])
    orientation2_angle = torch.atan2(orientation2[1], orientation2[0])
    delta = orientation2_angle - orientation1_angle
    # unwrap to [-pi, pi]
    angle_radians = (delta + torch.pi) % (2 * torch.pi) - torch.pi
    
    return angle_radians
    
def distance_2d(point1, point2):
    """
    Compute Euclidean distance in the XZ plane between two sets of points.

    Args:
        point1: (3,) tensor - xyz
        point2: (3,) tensor - xyz

    Returns:
        distances: (1,) tensor - distance between points
    """
    return torch.sqrt((point1[0] - point2[0])**2 + (point1[2] - point2[2])**2)

def get_object_count_in_a_scene(one_hot, class_label, idx_to_labels):
    """
    Count number of objects of a specific class in each scene.

    Args:
        one_hot: (B, N, num_classes) - One-hot encoded classes
        class_label: string, e.g. "ceiling_lamp"
        idx_to_labels: dict, {idx: label}

    Returns:
        count: int, number of objects of class_label in the scene
    """
    
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    count = 0
    for i in range(one_hot.shape[1]):
        # print(f"one hot at index {i}: tv stand idx {labels_to_idx[class_label]}, one hot argmax {one_hot[0, i].argmax().item()}")
        if labels_to_idx[class_label] == one_hot[0, i].argmax().item():
            count += 1
    return count

import torch

def get_object_present_reward_potential(one_hot, class_label, idx_to_labels, 
                                       object_indices, count=1, threshold=0.5, 
                                       overplacement_penalty=2.0,
                                       underplacement_penalty=1.0):
    """
    Reward function with smooth gradients:
    - Perfect match (exactly `count` objects): +1.0
    - Too few objects: small negative, scaled by deficit
    - Too many objects: larger negative, scaled by excess
    """
    label_to_idx = {v: k for k, v in idx_to_labels.items()}
    
    if class_label not in label_to_idx:
        raise ValueError(f"Class label '{class_label}' not found in idx_to_labels")
    
    target_idx = label_to_idx[class_label]
    
    # Count detections: (B,)
    matches_target = (object_indices == target_idx)
    num_detections = torch.sum(matches_target, dim=1).float()
    #simplified reward: +1 if at least one object is present, else 0
    # rewards = torch.where(
    #     num_detections >= 1,
    #     torch.ones_like(num_detections),  # At least one object: +1.0
    #     torch.zeros_like(num_detections)  # No object: 0.0
    # )
    
    # # Calculate deviation from target count
    deviation = num_detections - count
    # Shaped reward with asymmetric penalties
    rewards = torch.where(
        deviation == 0,
        torch.ones_like(num_detections),  # Perfect: +1.0
        torch.where(
            deviation < 0,
            # Too few: -0.5 per missing object (less harsh)
            deviation * underplacement_penalty,  # e.g., -1 TV: -0.5, -2 TVs: -1.0
            # Too many: -2.0 per extra object (more harsh)
            -deviation * overplacement_penalty  # e.g., +1 TV: -2.0, +2 TVs: -4.0
        )
    )
    
    return rewards # range [-21, 1]  #-18

def has_x_meter_clearance(parsed_scenes, x, direction):
    """
    Check whether there is at least x meters of path clearance around objects.

    Args:
        parsed_scenes: list/dict with scene info
        x: float, required clearance in meters
        direction: float, z_angle in radians
    Returns:
        clearance_mask: tensor/list (B,) - True if scene satisfies clearance
    """
    raise NotImplementedError("This function is not implemented")
    
# Utility function to create a scene for testing reward functions
def create_scene_for_testing(room_type, num_objects, class_label_indices, translations, sizes, orientations):
    """
    Create a scene for testing reward functions.
    Input:
        room_type: string, Example: "bedroom" or "livingroom"
        num_objects: int, number of objects in the scene
        class_label_indices: list of int, class indices
        translations: list of tuple, (x, y, z) translations
        sizes: list of tuple, (sx/2, sy/2, sz/2) sizes
        orientations: list of tuple, (cos(θ), sin(θ)) orientations
        
    Output:
        parsed_scene: dict, scene representation
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
    max_num_objects, num_classes = room_stats[room_type]["max_objects"], room_stats[room_type]["num_classes"]
    B = 1  # Single scene for testing
    N = max_num_objects
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create empty arrays
    one_hot = torch.zeros(B, N, num_classes, dtype=torch.float32, device=device)
    positions = torch.zeros(B, N, 3, dtype=torch.float32, device=device)
    obj_sizes = torch.zeros(B, N, 3, dtype=torch.float32, device=device)
    obj_orientations = torch.zeros(B, N, 2, dtype=torch.float32, device=device)
    object_indices = torch.full((B, N), num_classes-1, dtype=torch.long, device=device)  # default is empty class
    is_empty = torch.ones(B, N, dtype=torch.bool, device=device)  # mark all empty by default

    for i in range(num_objects):
        idx = class_label_indices[i]
        one_hot[0, i, idx] = 1.0
        positions[0, i] = torch.tensor(translations[i], dtype=torch.float32, device=device)
        obj_sizes[0, i] = torch.tensor(sizes[i], dtype=torch.float32, device=device)
        obj_orientations[0, i] = torch.tensor(orientations[i], dtype=torch.float32, device=device)
        object_indices[0, i] = idx
        is_empty[0, i] = False

    # Fill the one_hot for empty slots (if any)
    for i in range(num_objects, N):
        one_hot[0, i, num_classes - 1] = 1.0  # Mark as "empty"

    parsed_scene = {
        "one_hot": one_hot,  # (B, N, num_classes)
        "positions": positions,  # (B, N, 3)
        "sizes": obj_sizes,      # (B, N, 3) (half-extents)
        "orientations": obj_orientations,  # (B, N, 2)
        "object_indices": object_indices,  # (B, N)
        "is_empty": is_empty,              # (B, N)
        "room_type": room_type,
        "device": torch.device(device),
    }
    return parsed_scene


def get_all_utility_functions(is_prompt=False):
    """
    Get all utility functions.
    """
    return {
        "find_object_front_and_back": {
            "function": find_object_front_and_back.__name__ if is_prompt else find_object_front_and_back,
            "description": find_object_front_and_back.__doc__,
        },
        "find_closest_wall_to_object": {
            "function": find_closest_wall_to_object.__name__ if is_prompt else find_closest_wall_to_object,
            "description": find_closest_wall_to_object.__doc__,
        },
        # "compute_angle_between_objects": {
        #     "function": compute_angle_between_objects.__name__ if is_prompt else compute_angle_between_objects,
        #     "description": compute_angle_between_objects.__doc__,
        # },
        # "distance_2d": {
        #     "function": distance_2d.__name__ if is_prompt else distance_2d,
        #     "description": distance_2d.__doc__,
        # },
        "get_object_count_in_a_scene": {
            "function": get_object_count_in_a_scene.__name__ if is_prompt else get_object_count_in_a_scene,
            "description": get_object_count_in_a_scene.__doc__,
        },
        "get_object_present_reward_potential": {
            "function": get_object_present_reward_potential.__name__ if is_prompt else get_object_present_reward_potential,
            "description": get_object_present_reward_potential.__doc__,
        },
        # TODO Not implemented yet
        # "has_x_meter_clearance": {
        #     "function": has_x_meter_clearance.__name__,
        #     "description": has_x_meter_clearance.__doc__,
        # },
        "create_scene_for_testing": {
            "function": create_scene_for_testing.__name__ if is_prompt else create_scene_for_testing,
            "description": create_scene_for_testing.__doc__,
        },
    }