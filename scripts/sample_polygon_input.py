import numpy as np

# Bounds for fpoc are [-2.995450019836426, -3.0, 2.995450019836426, 3.0]
fpoc_min_x = -2.995450019836426
fpoc_min_z = -3.0
fpoc_max_x = 2.995450019836426
fpoc_max_z = 3.0

# A simple square floor plan
# polygon_world = np.array([[-2.0, -2.0 ],
#        [-2.0,  2.0 ],
#        [ 2.0,  2.0 ],
#        [ 2.0, -2.0 ]])

# L shaped floor plan
# Points must be ordered going around the OUTER perimeter (counter-clockwise when viewed from above, Y+)
# For CCW winding: start at bottom-left, trace counter-clockwise around the perimeter
#   (-2, -2) → (-2, 2) → (0, 2) → (0, 0) → (2, 0) → (2, -2) → back to start
polygon_world = np.array([[-2.0, -2.0 ],  # Bottom-left corner (start)
       [-2.0,  2.0 ],  # Top-left corner
       [ -1.0,  2.0 ],  # Left edge at x=0, top
       [ -1.0,  1.0 ],  # Inner corner (where L meets)
       [ 2.0,  1.0 ],  # Right edge at z=0
       [ 2.0, -2.0 ]])  # Bottom-right corner

# Save normalized polygon for use with preprocess_floorplan_custom.py
out_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp/polygon_world.npy"
np.save(out_path, polygon_world)
print("Saved normalized polygon to:", out_path)