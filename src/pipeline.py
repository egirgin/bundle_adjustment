# vo_project/pipeline.py

from doctest import debug
import cv2
import numpy as np
import os

from map_structures import Map, Keyframe, MapPoint
from features import FeatureExtractor, FeatureMatcher
from bundle_adjuster import BundleAdjuster
from visualization import plot_and_save_trajectory_2d, plot_and_save_trajectory_3d
from parameters import TRAJECTORY_DIR_2D, TRAJECTORY_DIR_3D, MIN_TRACKED_FEATURES, BA_WINDOW_SIZE, DEBUG, DEBUG_DIRS, CAMERA_POSE_INLIER_RATIO, CAMERA_POSE_INLIER_NUMBERS


class VisualOdometryPipeline:
    """
    Implements the main visual odometry pipeline, managing the entire process.
    """
    def __init__(self, camera_matrix, dist_coeffs, feature_extractor: FeatureExtractor, 
                 feature_matcher: FeatureMatcher, keyframe_criteria: dict):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.keyframe_criteria = keyframe_criteria
        self.min_matches_to_track = MIN_TRACKED_FEATURES

        print("Visual Odometry Pipeline initialized with criteria:", keyframe_criteria)

        self.trajectory_output_dir_2d = TRAJECTORY_DIR_2D
        self.trajectory_output_dir_3d = TRAJECTORY_DIR_3D

        self.map = Map()
        self.bundle_adjuster = BundleAdjuster(camera_matrix, window_size=BA_WINDOW_SIZE)

    def process_frame(self, frame):
        """Processes a single video frame."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)
        
        if self.map.next_keyframe_id == 0:
            self._initialize_map(frame, keypoints, descriptors)
            return

        last_kf = self.map.keyframes[self.map.next_keyframe_id - 1]
        matches = self.feature_matcher.match(last_kf.descriptors, descriptors)

        if len(matches) < self.min_matches_to_track:
            print("    -> Frame Discarded: Not enough matches to track.")
            return

        if DEBUG:
            img_matches = cv2.drawMatches(
                last_kf.img,  # Last keyframe's RGB image
                last_kf.keypoints,
                frame,        # Current frame's RGB image
                keypoints,
                matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            debug_dir = DEBUG_DIRS['matches']
            img_filename = os.path.join(debug_dir, f"matches_{self.map.next_keyframe_id}.png")
            cv2.imwrite(img_filename, img_matches)

        relative_R, relative_t, inlier_pts1, inlier_pts2, inlier_indices = self._estimate_pose(matches, last_kf.keypoints, keypoints)
        
        if relative_R is None or relative_t is None:
            print("    -> Frame Discarded: Could not estimate pose.")
            return
        
        # Calculate inlier ratio
        num_inliers = len(inlier_indices)
        num_matches = len(matches)
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
        is_reliable = inlier_ratio > CAMERA_POSE_INLIER_RATIO and num_inliers > CAMERA_POSE_INLIER_NUMBERS

        if not is_reliable:
            print("    -> Frame Discarded: Low inlier ratio or insufficient inliers.")
            return
        
        if self._is_keyframe(relative_R, relative_t, matches, inlier_indices, inlier_pts1, inlier_pts2, last_kf):

            if DEBUG:
                img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags=0)
                debug_dir = DEBUG_DIRS['keyframes']
                img_filename = os.path.join(debug_dir, f"keyframe_{self.map.next_keyframe_id}.png")
                cv2.imwrite(img_filename, img_with_keypoints)
                
            self._add_new_keyframe(frame, keypoints, descriptors, last_kf, matches, relative_R, relative_t, inlier_indices, inlier_pts1, inlier_pts2)
            self.bundle_adjuster.run(self.map)
    
    def _initialize_map(self, frame, keypoints, descriptors):
        """
        Initializes the map with the first keyframe with unit rotation and translation.
        """
        print("Initializing with first keyframe...")
        kf = Keyframe(
            id=self.map.next_keyframe_id, R=np.eye(3), t=np.zeros((3, 1)),
            keypoints=keypoints, descriptors=descriptors, observations=[], img=frame.copy()
        )
        self.map.add_keyframe(kf)

    def _add_new_keyframe(self, frame, keypoints, descriptors, last_kf, matches, R_rel, t_rel, inlier_indices, inlier_pts1, inlier_pts2):
        """
            Adds a new keyframe to the map if if the frame is a keyframe
        Steps:
            - Create a new Keyframe object.
            - Add the new keyframe to the map.
        args:
            - frame: The current frame being processed.
            - keypoints: 2D Keypoints detected in the current frame.
            - descriptors: Descriptors for the keypoints.
            - last_kf: The last keyframe in the map.
            - matches: Feature matches between the last keyframe and the current frame.
            - R_rel: Relative rotation between the last keyframe and the current frame.
            - t_rel: Relative translation between the last keyframe and the current frame.
            - inlier_indices: Indices of inlier matches.
            - inlier_pts1: Inlier points from the last keyframe.
            - inlier_pts2: Inlier points from the current frame.
        """
        world_R = last_kf.R @ R_rel
        world_t = last_kf.t + last_kf.R @ t_rel
        new_kf = Keyframe(id=self.map.next_keyframe_id, R=world_R, t=world_t, keypoints=keypoints, descriptors=descriptors, observations=[], img=frame.copy())
        
        # Build a reverse lookup for the last keyframe's observations
        # This lets us quickly find if a keypoint corresponds to a map point
        # 2D point -> 3D point hashtable for the last KF
        last_kf_obs_lookup = {kp_idx: mp_id for mp_id, kp_idx in last_kf.observations}

        # Points to be registered to the map first time
        newly_triangulated_pts = []
        newly_triangulated_indices = []

        # for each point contributed to the essential matrix calculation
        for i in range(len(inlier_indices)):
            match_idx = inlier_indices[i]

            # Keypoint indices (queryIdx/trainIdx) are only meaningful within their respective keyframes.
            # The indices below are the same across different matches of the same keyframe.

            # Get the keypoint index from the last keyframe.
            last_kf_kp_idx = matches[match_idx].queryIdx

            # Get the keypoint index from the new keyframe.
            new_kf_kp_idx = matches[match_idx].trainIdx

            # Check if this 2D keypoint from the last keyframe already corresponds to a 3D map point.
            # If yes, this is a re-observation: associate the new keyframe and keypoint index with the existing map point.
            # If not, this is a new point: record the indices for triangulation and later addition to the map as evidence.

            if last_kf_kp_idx in last_kf_obs_lookup:
                # Re-observation of an existing map point
                mp_id = last_kf_obs_lookup[last_kf_kp_idx]
                self.map.map_points[mp_id].observations.append((new_kf.id, new_kf_kp_idx))
                new_kf.observations.append((mp_id, new_kf_kp_idx))
            else:
                # New 3D point to be triangulated and added to the map
                newly_triangulated_pts.append((inlier_pts1[i], inlier_pts2[i]))
                newly_triangulated_indices.append(match_idx)

        # Triangulate only the points that are actually new
        if newly_triangulated_pts:
            new_pts1 = np.array([p[0] for p in newly_triangulated_pts])
            new_pts2 = np.array([p[1] for p in newly_triangulated_pts])
            points_3d_relative, _ = self._triangulate_points(R_rel, t_rel, new_pts1, new_pts2)

            if points_3d_relative is not None:
                # move the point to the world coordinate system
                points_3d_world = (last_kf.R @ points_3d_relative) + last_kf.t

                for i in range(points_3d_world.shape[1]):
                    match_idx = newly_triangulated_indices[i]
                    last_kf_kp_idx = matches[match_idx].queryIdx
                    new_kf_kp_idx = matches[match_idx].trainIdx

                    # Get the color of the keypoint in the new keyframe
                    bgr_color = frame[int(new_pts2[i][1]), int(new_pts2[i][0])]
                    rgb_color = (bgr_color[::-1] / 255.0).astype(np.float64)

                    # Create a new map point
                    mp = MapPoint(id=self.map.next_map_point_id, position=points_3d_world[:, i].reshape(3,1), observations=[], color=rgb_color)
                    mp.observations.extend([(last_kf.id, last_kf_kp_idx), (new_kf.id, new_kf_kp_idx)])
                    last_kf.observations.append((mp.id, last_kf_kp_idx))
                    new_kf.observations.append((mp.id, new_kf_kp_idx))
                    self.map.add_map_point(mp)

        self.map.add_keyframe(new_kf)
        
        plot_and_save_trajectory_2d(self.map, self.trajectory_output_dir_2d)
        plot_and_save_trajectory_3d(self.map, self.trajectory_output_dir_3d)

    def _calculate_median_displacement(self, pts1, pts2):
        return 0 if len(pts1) == 0 else np.median(np.linalg.norm(pts2 - pts1, axis=1))

    def _is_keyframe(self, relative_R, relative_t, all_matches, inlier_indices, inlier_pts1, inlier_pts2, last_kf: Keyframe):
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
            if mp_id and mp_id in self.map.map_points:
                map_point = self.map.map_points[mp_id]

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

    def _estimate_pose(self, matches, kp1, kp2):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        if E is None:
            return None, None, None, None, None
            
        _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)

        num_matches = len(matches)
        num_inliers = np.sum(mask)
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
        print(f"    -> Pose Estimation: {num_inliers} inliers out of {num_matches} matches (Ratio: {inlier_ratio:.2f})")

        # A very low ratio indicates a poor geometric match
        if inlier_ratio < 0.4: # You can tune this threshold
            print("    -> WARNING: Low inlier ratio. Pose estimate may be unreliable.")

        inlier_mask = mask.ravel() == 1
        return R_rel, t_rel, pts1[inlier_mask], pts2[inlier_mask], np.where(inlier_mask)[0]

    def _triangulate_points(self, R_rel, t_rel, pts1, pts2):
        if pts1.shape[0] == 0:
            return None, None
        
        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.camera_matrix @ np.hstack((R_rel, t_rel))
        
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d_hom / (points_4d_hom[3] + 1e-6)
        
        # --- Cheirality Check ---
        # 1. Check if points are in front of the first camera (z > 0)
        valid_mask_cam1 = points_3d[2, :] > 0
        # 2. Check if points are in front of the second camera
        # Transform points into the second camera's frame and check z > 0
        points_3d_in_cam2 = R_rel @ points_3d[:3, :] + t_rel
        valid_mask_cam2 = points_3d_in_cam2[2, :] > 0
        # The final mask requires the point to be valid in both views
        valid_mask = np.logical_and(valid_mask_cam1, valid_mask_cam2)
        
        print(f"    -> Triangulation: Kept {np.sum(valid_mask)} of {points_4d_hom.shape[1]} points.")
        return points_3d[:3, valid_mask], np.where(valid_mask)[0]