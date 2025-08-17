# vo_project/main.py

import os
import shutil
import cv2
import numpy as np
import open3d as o3d

from pipeline import VisualOdometryPipeline
from features import ORBExtractor, BruteForceMatcher
from parameters import OUTPUT_DIR, DEBUG_DIRS

def clean_directories(folders):
    """Deletes specified folders if they exist."""
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")

def main():
    # --- Configuration ---
    VIDEO_PATH = '../data/video_0001.mp4'
    #VIDEO_PATH = '/home/students/girgine/ros2_ws/src/visual_odometry/data/my_dataset_7/video_0001.mp4'
    #VIDEO_PATH = '/home/students/girgine/ros2_ws/src/visual_odometry/data/desk.avi'

    # Keyframe selection criteria
    KEYFRAME_CRITERIA = {
        "min_pixel_displacement": 20.0,
        "min_rotation": 0.15,
        "min_feature_ratio": 0.25,
        "min_parallax_deg": 1.0,
        "min_tracked_for_parallax": 20
    }

    # Camera intrinsics
    CAMERA_MATRIX = np.array([
        [912.7820434570312, 0.0, 650.2929077148438],
        [0.0, 913.0294189453125, 362.7241516113281],
        [0.0, 0.0, 1.0]
    ])
    DIST_COEFFS = np.zeros(5)

    # --- Setup ---
    clean_directories([OUTPUT_DIR] + list(DEBUG_DIRS.values()))
    print(list(DEBUG_DIRS.values()))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for debug_dir in DEBUG_DIRS.values():
        os.makedirs(debug_dir, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{VIDEO_PATH}'")
        return

    # --- Pipeline Initialization ---
    print("Initializing visual odometry pipeline...")
    vo_pipeline = VisualOdometryPipeline(
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
        feature_extractor=ORBExtractor(n_features=4000),
        feature_matcher=BruteForceMatcher(),
        keyframe_criteria=KEYFRAME_CRITERIA
    )

    # --- Main Loop ---
    print("Opening video...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing frame {frame_idx}...")
        vo_pipeline.process_frame(frame)
        frame_idx += 1
    
    cap.release()
    print("\nVideo processing finished.")

    # --- Final Global Bundle Adjustment ---
    print("\n--- Running Final Global Bundle Adjustment ---")
    num_keyframes = vo_pipeline.map.next_keyframe_id
    if num_keyframes > 2:
        original_window_size = vo_pipeline.bundle_adjuster.window_size
        vo_pipeline.bundle_adjuster.window_size = num_keyframes 
        vo_pipeline.bundle_adjuster.run(vo_pipeline.map)
        vo_pipeline.bundle_adjuster.window_size = original_window_size
        print("--- Global Bundle Adjustment Complete ---")
    else:
        print("--- Global Bundle Adjustment Skipped: Not enough keyframes ---")

    # --- Save and Display Final Map ---
    final_pcd = vo_pipeline.map.get_pcd()
    if final_pcd.has_points():
        final_map_path = os.path.join(OUTPUT_DIR, "final_map_global_ba.pcd")
        print(f"\nSaving final map to '{final_map_path}'...")
        o3d.io.write_point_cloud(final_map_path, final_pcd)
        print("Map saved successfully.")
        
        print("\nDisplaying the final map. Press 'q' in the window to close.")
        o3d.visualization.draw_geometries([final_pcd])
    else:
        print("\nNo points were generated for the final map.")

if __name__ == '__main__':
    main()
