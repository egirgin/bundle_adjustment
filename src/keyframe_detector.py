# vo_project/keyframe_detector.py

import cv2
import numpy as np
from typing import List

from map_structures import Keyframe, Map

class KeyframeDetector:
    """
    Determines if a frame should be a keyframe based on motion and feature tracking.
    """
    def __init__(self, keyframe_criteria: dict):
        self.keyframe_criteria = keyframe_criteria
        print("KeyframeDetector initialized with criteria:", keyframe_criteria)

    def _calculate_median_displacement(self, pts1: np.ndarray, pts2: np.ndarray) -> float:
        """Calculates the median pixel displacement between two sets of points."""
        return 0 if len(pts1) == 0 else np.median(np.linalg.norm(pts2 - pts1, axis=1))

    def is_keyframe(
        self,
        relative_R: np.ndarray,
        relative_t: np.ndarray,
        all_matches: list,
        inlier_indices: np.ndarray,
        inlier_pts1: np.ndarray,
        inlier_pts2: np.ndarray,
        last_kf: Keyframe,
        gmap: Map
    ) -> bool:
        """
        Determines if the current frame should be a keyframe based on motion,
        feature ratio, and geometric baseline (parallax).
        """
        # --- Condition 1: Check for sufficient parallax (good baseline) ---
        # This is crucial for stable triangulation.
        last_kf_obs_lookup = {kp_idx: mp_id for mp_id, kp_idx in last_kf.observations}
        parallaxes = []

        # Get the pose of the current potential keyframe in the world
        current_R_world = last_kf.R @ relative_R
        current_t_world = last_kf.t + last_kf.R @ relative_t

        for i in range(len(inlier_indices)):
            match_idx = inlier_indices[i]
            last_kf_kp_idx = all_matches[match_idx].queryIdx

            # Check if the matched keypoint in the last frame corresponds to a known 3D map point
            mp_id = last_kf_obs_lookup.get(last_kf_kp_idx)
            if mp_id and mp_id in gmap.map_points:
                map_point = gmap.map_points[mp_id]

                # Vector from last keyframe's center to the 3D point
                vec1 = map_point.position - last_kf.t
                # Vector from current frame's center to the 3D point
                vec2 = map_point.position - current_t_world

                # Calculate the angle between the two viewing rays
                cos_angle = vec1.T @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                parallaxes.append(angle)

        if len(parallaxes) > self.keyframe_criteria['min_tracked_for_parallax']:
            median_parallax_rad = np.median(parallaxes)
            median_parallax_deg = np.rad2deg(median_parallax_rad)
            if median_parallax_deg > self.keyframe_criteria['min_parallax_deg']:
                print(f"    -> Keyframe Trigger: Parallax ({median_parallax_deg:.2f}° > {self.keyframe_criteria['min_parallax_deg']}°)")
                return True

        # --- Condition 2: Fallback checks if parallax is insufficient (e.g., during initialization) ---
        median_displacement = self._calculate_median_displacement(inlier_pts1, inlier_pts2)
        if median_displacement > self.keyframe_criteria['min_pixel_displacement']:
            print(f"    -> Keyframe Trigger: Pixel Displacement ({median_displacement:.2f} > {self.keyframe_criteria['min_pixel_displacement']})")
            return True

        angle, _ = cv2.Rodrigues(relative_R)
        rotation_magnitude = np.linalg.norm(angle)
        if rotation_magnitude > self.keyframe_criteria['min_rotation']:
            print(f"    -> Keyframe Trigger: Rotation ({rotation_magnitude:.4f} > {self.keyframe_criteria['min_rotation']})")
            return True

        feature_ratio = len(inlier_indices) / len(last_kf.keypoints) if len(last_kf.keypoints) > 0 else 0
        if feature_ratio < self.keyframe_criteria['min_feature_ratio']:
            print(f"    -> Keyframe Trigger: Feature Ratio ({feature_ratio:.2f} < {self.keyframe_criteria['min_feature_ratio']})")
            return True

        return False