# vo_project/parameters.py

# Output directories (global constants)
OUTPUT_DIR = 'output_map'
TRAJECTORY_DIR_2D = 'output_trajectory_2d'
TRAJECTORY_DIR_3D = 'output_trajectory_3d'

DEBUG = True

DEBUG_DIRS = {
    'sparsity': 'debug_sparsity',
    'matches': 'debug_matches',
    'keyframes': 'debug_keyframes',
    'trajectory_2d': TRAJECTORY_DIR_2D,
    'trajectory_3d': TRAJECTORY_DIR_3D,
    'lba_steps': 'output_map/lba_steps'
}
MIN_TRACKED_FEATURES = 20 # between two frames
BA_WINDOW_SIZE = 5
CAMERA_POSE_INLIER_RATIO = 0.7
CAMERA_POSE_INLIER_NUMBERS = 20