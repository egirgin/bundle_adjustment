import sys
import cv2
import matplotlib
import numpy as np
import os
import open3d as o3d
from scipy.optimize import least_squares
from dataclasses import dataclass
from scipy.sparse import lil_matrix
import shutil
matplotlib.use('Agg') # Use the Agg backend
import matplotlib.pyplot as plt

@dataclass
class MapPoint:
    id: int
    position: np.ndarray  # 3x1 vector
    observations: list  # List of (keyframe_id, keypoint_index)
    color: np.ndarray  # 3x1 vector

@dataclass
class Keyframe:
    id: int
    R: np.ndarray
    t: np.ndarray
    keypoints: list
    descriptors: np.ndarray
    observations: list
    img: np.ndarray  # Store the RGB image for visualization

class Map:
    """
    Represents a full in-memory map for visual odometry, managing keyframes and 
    3D map points. This class provides functionality to:
    - Store and manage keyframes and map points, each with unique IDs.
    - Add new keyframes and map points to the map.
    - Generate an Open3D point cloud from the stored map points for visualization or further processing.
    Attributes:
        keyframes (dict): Dictionary mapping keyframe IDs to Keyframe objects.
        map_points (dict): Dictionary mapping map point IDs to MapPoint objects.
        next_keyframe_id (int): Counter for assigning unique keyframe IDs.
        next_map_point_id (int): Counter for assigning unique map point IDs.
    Methods:
        add_keyframe(keyframe: Keyframe): Adds a keyframe to the map.
        add_map_point(map_point: MapPoint): Adds a map point to the map.
        get_pcd(): Generates an Open3D point cloud from the map points.
    """

    def __init__(self):
        self.keyframes = {}
        self.map_points = {}
        self.next_keyframe_id = 0
        self.next_map_point_id = 0

    def add_keyframe(self, keyframe: Keyframe):
        self.keyframes[keyframe.id] = keyframe
        self.next_keyframe_id += 1

    def add_map_point(self, map_point: MapPoint):
        self.map_points[map_point.id] = map_point
        self.next_map_point_id += 1
    
    def get_pcd(self):
        """Generates an Open3D point cloud from the map points."""
        positions = [p.position for p in self.map_points.values()]
        if not positions:
            return o3d.geometry.PointCloud()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.squeeze(positions))
        pcd.colors = o3d.utility.Vector3dVector([p.color for p in self.map_points.values()])
        return pcd

class BundleAdjuster:
    """
    BundleAdjuster performs Local Bundle Adjustment (LBA) for visual odometry.
    This class is responsible for optimizing the poses of keyframes and 
    the positions of 3D map points within a local sliding window, 
    using non-linear least squares minimization of re-projection error.
    It is independent of the main odometry pipeline and 
    leverages `scipy.optimize.least_squares` for optimization.
    Attributes:
        camera_matrix (np.ndarray): The intrinsic camera matrix used for projection.
        window_size (int): Number of recent keyframes to include in the local window for adjustment.
    Methods:
        __init__(camera_matrix, window_size=5):
            Initializes the BundleAdjuster with camera intrinsics and window size.
        _cost_function(params, keyframe_ids, map_point_ids, observations, keypoints_2d):
            Computes the re-projection error for all observations in the local window.
            Used as the objective function for optimization.
        run(gmap: Map):
            Executes Local Bundle Adjustment on the most recent window of keyframes and updates
            the map with optimized poses and point positions.
    """
    def __init__(self, camera_matrix, window_size=5):
        self.camera_matrix = camera_matrix
        self.window_size = window_size

    def _cost_function(self, params, fixed_kf_pose, fixed_kf_id, adjustable_kf_ids, map_point_ids, observations, keypoints_2d):
        
        """
        Computes the re-projection error for Bundle Adjustment optimization.
        Args:
            params (np.ndarray): Optimization parameter vector containing rotation vectors, translation vectors for adjustable keyframes, and 3D map point coordinates. [R1, R2, ..., t1, t2, ..., P1, P2, ..., PN]
            fixed_kf_pose (tuple): Pose of the fixed keyframe as (R, t), where R is a rotation matrix and t is a translation vector.
            fixed_kf_id (int): ID of the fixed keyframe.
            adjustable_kf_ids (list of int): List of IDs for adjustable keyframes.
            map_point_ids (list of int): List of IDs for map points.
            observations (list of tuple): List of (keyframe_id, map_point_id) tuples representing all observations. (Fixed + Adjusted)
            keypoints_2d (dict): Dictionary mapping (keyframe_id, map_point_id) to 2D keypoint coordinates.
        Returns:
            np.ndarray: Flattened array of re-projection errors for all observations.
        """
        num_adjustable_keyframes = len(adjustable_kf_ids)
        num_map_points = len(map_point_ids)
        
        # Unpack parameters for an adjustable keyframes
        poses_rvec = params[0 : num_adjustable_keyframes * 3].reshape((num_adjustable_keyframes, 3))
        poses_tvec = params[num_adjustable_keyframes * 3 : num_adjustable_keyframes * 6].reshape((num_adjustable_keyframes, 3))
        points_3d = params[num_adjustable_keyframes * 6 :].reshape((num_map_points, 3))

        # Create dictionaries for quick lookups (global_enumeration : local_enumeration)
        adj_kf_id_to_idx = {kf_id: i for i, kf_id in enumerate(adjustable_kf_ids)}
        mp_id_to_idx = {mp_id: i for i, mp_id in enumerate(map_point_ids)}

        errors = []
        fixed_R, fixed_t = fixed_kf_pose # Unpack fixed pose

        for obs_kf_id, obs_mp_id in observations: # (keyframe_id, 3D map_point_id) 
            mp_idx = mp_id_to_idx.get(obs_mp_id) # get local 3D point ID
            if mp_idx is None: # is it ever be zero ? 
                continue

            point_3d = points_3d[mp_idx] # get estimated 3D point

            # Check if the observation is from the fixed keyframe or an adjustable one
            if obs_kf_id == fixed_kf_id:
                rvec, _ = cv2.Rodrigues(fixed_R) # Rotation to Rodrigues
                tvec = fixed_t
            else:
                kf_idx = adj_kf_id_to_idx.get(obs_kf_id)
                if kf_idx is None:
                    continue
                rvec = poses_rvec[kf_idx] # already in rodrugues form
                tvec = poses_tvec[kf_idx]
            
            # Project the 3D point into the keyframe's image plane
            projected_points, _ = cv2.projectPoints(point_3d, rvec, tvec, self.camera_matrix, None)
            
            # Get the original 2D keypoint detection
            observed_point = keypoints_2d[(obs_kf_id, obs_mp_id)]

            # Calculate the re-projection error
            error = (observed_point - projected_points.ravel()).ravel()
            errors.extend(error)
            
        return np.array(errors)
    
    def _prepare_sparsity_matrix(self, num_adj_kfs, num_mps, adj_kf_ids, mp_ids, observations):
        """
        Creates the sparse Jacobian matrix structure for the optimizer.
        """
        num_params = num_adj_kfs * 6 + num_mps * 3
        num_residuals = len(observations) * 2

        A = lil_matrix((num_residuals, num_params), dtype=int)

        adj_kf_id_to_idx = {kf_id: i for i, kf_id in enumerate(adj_kf_ids)}
        mp_id_to_idx = {mp_id: i for i, mp_id in enumerate(mp_ids)}

        for i, (obs_kf_id, obs_mp_id) in enumerate(observations):
            mp_idx = mp_id_to_idx.get(obs_mp_id)
            if mp_idx is None:
                continue

            # Each observation contributes 2 residuals (x and y error)
            row = i * 2

            # The residuals depend on the 3D point's parameters
            point_param_idx_start = num_adj_kfs * 6 + mp_idx * 3
            A[row:row+2, point_param_idx_start:point_param_idx_start+3] = 1

            # If the observation is from an adjustable keyframe,
            # the residuals also depend on that keyframe's pose parameters
            if obs_kf_id in adj_kf_id_to_idx:
                kf_idx = adj_kf_id_to_idx[obs_kf_id]
                # Rotation parameters
                r_param_idx_start = kf_idx * 3
                A[row:row+2, r_param_idx_start:r_param_idx_start+3] = 1
                # Translation parameters
                t_param_idx_start = num_adj_kfs * 3 + kf_idx * 3
                A[row:row+2, t_param_idx_start:t_param_idx_start+3] = 1

        return A


    def run(self, gmap: Map):
        """
        Run Local Bundle Adjustment (LBA) on the most recent window of keyframes.
        This function:
        - Selects a sliding window of keyframes, fixing the oldest pose and optimizing the rest.
        - Gathers all map points observed by these keyframes and their 2D observations.
        - Prepares initial parameter vectors for adjustable keyframe poses and map point positions.
        - Runs non-linear least squares optimization to minimize reprojection error.
        - Updates the map with optimized keyframe poses and map point positions.
        """
        print("--- Running Local Bundle Adjustment ---")
        
        # 1. Select the local window of keyframes
        all_kf_ids = sorted(gmap.keyframes.keys())
        if len(all_kf_ids) < 3: # Need at least 1 fixed, 1 adjustable, and the latest frame (not included)
            print("    -> LBA Skipped: Not enough keyframes to form a window.")
            return
            
        local_kf_ids = all_kf_ids[-(self.window_size + 1) : -1]

        # Separate the oldest keyframe to be fixed
        fixed_kf_id = local_kf_ids[0]
        adjustable_kf_ids = local_kf_ids[1:]

        if not adjustable_kf_ids:
            print("    -> LBA Skipped: No adjustable keyframes in the window.")
            return

        fixed_keyframe = gmap.keyframes[fixed_kf_id]
        adjustable_keyframes = {i: gmap.keyframes[i] for i in adjustable_kf_ids}
        
        # 2. Find all map points observed by these keyframes (both fixed and adjustable)
        local_map_point_ids = set()
        observations = []
        keypoints_2d = {}

        # Gather observations from the fixed keyframe (mapPoint id, keypoint id)
        for mp_id, kp_idx in fixed_keyframe.observations:
            if mp_id in gmap.map_points:
                local_map_point_ids.add(mp_id)
                observations.append((fixed_kf_id, mp_id))
                keypoints_2d[(fixed_kf_id, mp_id)] = fixed_keyframe.keypoints[kp_idx].pt

        # Gather observations from adjustable keyframes
        for kf_id, kf in adjustable_keyframes.items(): # (21, <KeyFrame>) etc.
            for mp_id, kp_idx in kf.observations: # 3D point id, 2D point id
                if mp_id in gmap.map_points:
                    local_map_point_ids.add(mp_id)
                    observations.append((kf_id, mp_id))
                    keypoints_2d[(kf_id, mp_id)] = kf.keypoints[kp_idx].pt

        local_map_point_ids = sorted(list(local_map_point_ids))

        if not local_map_point_ids:
            print("    -> LBA Skipped: No points in the local window.")
            return

        # 3. Prepare initial parameters for the optimizer (ONLY for adjustable frames)
        # (R1, R1, R1, R2, R2, R2, ...)
        initial_rvecs = np.array([cv2.Rodrigues(gmap.keyframes[i].R)[0].ravel() for i in adjustable_kf_ids])
        # (t1, t1, t1, t2, t2, t2, ...)
        initial_tvecs = np.array([gmap.keyframes[i].t.ravel() for i in adjustable_kf_ids])
        # (p1x, p1y, p1z, p2x, p2y, p2z, ...)
        initial_points_3d = np.array([gmap.map_points[i].position.ravel() for i in local_map_point_ids])
        
        params = np.concatenate([initial_rvecs.flatten(), initial_tvecs.flatten(), initial_points_3d.flatten()])
        print("    -> LBA Parameter size:", params.size)

        # MODIFICATION: Get the pose of the fixed keyframe to pass to the cost function
        fixed_kf_pose = (fixed_keyframe.R, fixed_keyframe.t)

        initial_residuals = self._cost_function(
            params, fixed_kf_pose, fixed_kf_id, adjustable_kf_ids, 
            local_map_point_ids, observations, keypoints_2d
        )
        initial_cost = np.sum(initial_residuals**2)

        sparsity_matrix = self._prepare_sparsity_matrix(
            len(adjustable_kf_ids), len(local_map_point_ids),
            adjustable_kf_ids, local_map_point_ids, observations
        )

        # --- DEBUG: Visualize Sparsity ---
        # This saves a plot of the Jacobian's structure.
        plt.figure(figsize=(8, 8))
        plt.spy(sparsity_matrix, markersize=1)
        plt.title(f"Jacobian Sparsity (LBA for KF {fixed_kf_id+1} to {local_kf_ids[-1]})")
        plt.xlabel("Parameters (Camera Poses + 3D Points)")
        plt.ylabel("Residuals (Observations)")
        sparsity_debug_dir = "debug_sparsity"
        os.makedirs(sparsity_debug_dir, exist_ok=True)
        plt.savefig(os.path.join(sparsity_debug_dir, f"sparsity_{fixed_kf_id}.png"))
        plt.close() # Close the plot to prevent it from displaying blocking the run
        # --- END DEBUG BLOCK ---

        # 4. Run the optimization
        res = least_squares(
            self._cost_function,
            params,
            jac_sparsity=sparsity_matrix,  # <-- Tell the solver to use it!
            loss='huber', # Using robust loss is also highly recommended
            args=(fixed_kf_pose, fixed_kf_id, adjustable_kf_ids, local_map_point_ids, observations, keypoints_2d),
            verbose=0, # Set to 1 for more detailed output
            xtol=1e-5,
            ftol=1e-5,
            max_nfev=50
        )

        final_cost = np.sum(res.fun**2)
        # --- CRITICAL FIX: Add this check ---
        if final_cost >= initial_cost:
            print(f"    -> LBA Diverged! Cost increased from {initial_cost:.2f} to {final_cost:.2f}. Discarding optimization results.")
            return # Exit the function without updating the map
        # --- END FIX ---
        
        # 5. Update the map with the optimized parameters (ONLY for adjustable frames)
        optimized_params = res.x
        num_adjustable_keyframes = len(adjustable_kf_ids)
        
        opt_rvecs = optimized_params[0 : num_adjustable_keyframes * 3].reshape((num_adjustable_keyframes, 3))
        opt_tvecs = optimized_params[num_adjustable_keyframes * 3 : num_adjustable_keyframes * 6].reshape((num_adjustable_keyframes, 3))
        opt_points_3d = optimized_params[num_adjustable_keyframes * 6 :].reshape((len(local_map_point_ids), 3))

        # Update adjustable keyframes
        for i, kf_id in enumerate(adjustable_kf_ids):
            R, _ = cv2.Rodrigues(opt_rvecs[i])
            gmap.keyframes[kf_id].R = R
            gmap.keyframes[kf_id].t = opt_tvecs[i].reshape(3, 1)

        # Update map points
        for i, mp_id in enumerate(local_map_point_ids):
            gmap.map_points[mp_id].position = opt_points_3d[i].reshape(3, 1)
        
        # --- MODIFICATION START ---
        # Save the updated point cloud after a successful BA run
        lba_steps_dir = "output_map/lba_steps"
        os.makedirs(lba_steps_dir, exist_ok=True)
        updated_pcd = gmap.get_pcd()
        if updated_pcd.has_points():
            pcd_filename = os.path.join(lba_steps_dir, f"map_after_lba_kf_{fixed_kf_id}.pcd")
            o3d.io.write_point_cloud(pcd_filename, updated_pcd)
            print(f"    -> Saved intermediate map to {pcd_filename}")
        # --- MODIFICATION END ---
            
        print(f"    -> LBA Complete. Fixed KF {fixed_kf_id}. Optimized {num_adjustable_keyframes} KFs and {len(local_map_point_ids)} MPs.")
        improvement = 100.0 * (initial_cost - final_cost) / (initial_cost + 1e-8)
        print(f"    -> LBA Complete. Initial Cost: {initial_cost:.2f}, Final Cost: {final_cost:.2f}, Improvement: {improvement:.2f}%")

class FeatureExtractor:
    def extract(self, image):
        raise NotImplementedError

class ORBExtractor(FeatureExtractor):
    def __init__(self, n_features=3000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def extract(self, image):
        return self.orb.detectAndCompute(image, None)

class FeatureMatcher:
    def match(self, des1, des2):
        raise NotImplementedError

class BruteForceMatcher(FeatureMatcher):
    def __init__(self, norm_type=cv2.NORM_HAMMING):
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass
        return good_matches

class VisualOdometryPipeline:
    """
    VisualOdometryPipeline implements a visual odometry pipeline for estimating camera motion and building a sparse 3D map from a sequence of images.
    This class manages feature extraction, matching, keyframe selection, pose estimation, triangulation, and local bundle adjustment. It maintains a map of keyframes and 3D points, and triggers local bundle adjustment when new keyframes are added.
    Args:
        camera_matrix (np.ndarray): Intrinsic camera matrix.
        dist_coeffs (np.ndarray): Distortion coefficients for the camera.
        feature_extractor (FeatureExtractor): Object for extracting features from images.
        feature_matcher (FeatureMatcher): Object for matching features between frames.
        keyframe_criteria (dict): Criteria for keyframe selection, including thresholds for pixel displacement, rotation, and feature ratio.
    Attributes:
        camera_matrix (np.ndarray): Intrinsic camera matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.
        feature_extractor (FeatureExtractor): Feature extraction object.
        feature_matcher (FeatureMatcher): Feature matching object.
        keyframe_criteria (dict): Keyframe selection thresholds.
        map (Map): Map object storing keyframes and map points.
        bundle_adjuster (BundleAdjuster): Local bundle adjustment optimizer.
    Methods:
        process_frame(frame):
            Processes a new frame, extracts features, matches with previous keyframe, estimates pose, checks keyframe criteria, triangulates new points, and triggers local bundle adjustment if a new keyframe is added.
        _calculate_median_displacement(pts1, pts2):
            Computes the median pixel displacement between matched points.
        _is_keyframe(relative_R, num_matches, inlier_pts1, inlier_pts2, last_kf_feature_count):
            Determines if the current frame should be a keyframe based on displacement, rotation, and feature ratio.
        _estimate_pose(matches, kp1, kp2):
            Estimates the relative pose between frames using matched keypoints and the essential matrix.
        _triangulate_points(R_rel, t_rel, pts1, pts2, inlier_indices):
            Triangulates 3D points from matched keypoints between two frames.
    Note:
        This class requires external definitions for Map, Keyframe, MapPoint, BundleAdjuster, FeatureExtractor, and FeatureMatcher.
    """
    # --- MODIFICATION: Add trajectory_output_dir to __init__ ---
    def __init__(self, camera_matrix, dist_coeffs, feature_extractor, feature_matcher, keyframe_criteria, trajectory_output_dir_2d, trajectory_output_dir_3d):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.keyframe_criteria = keyframe_criteria
        self.min_matches_to_track = 20


        self.trajectory_output_dir_2d = trajectory_output_dir_2d
        self.trajectory_output_dir_3d = trajectory_output_dir_3d

        self.map = Map()
        self.bundle_adjuster = BundleAdjuster(camera_matrix, window_size=5)

    def _calculate_median_displacement(self, pts1, pts2):
        if len(pts1) == 0:
            return 0
        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        return np.median(displacements)

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
        
    # --- MODIFICATION: Add a new method to plot and save the trajectory ---
    def _plot_and_save_trajectory_2d(self):
        """
        Generates a top-down (X-Z) plot of the camera's trajectory so far
        and saves it to the specified output directory.
        """
        # Ensure the output directory exists
        os.makedirs(self.trajectory_output_dir_2d, exist_ok=True)

        # Get all keyframe positions in order
        positions = []
        sorted_kf_ids = sorted(self.map.keyframes.keys())
        for kf_id in sorted_kf_ids:
            # The translation vector 't' represents the camera's world position
            positions.append(self.map.keyframes[kf_id].t.flatten())

        # Need at least two points to draw a line
        if len(positions) < 2:
            return

        positions_np = np.array(positions)
        x_coords = positions_np[:, 0]
        y_coords = positions_np[:, 1] # Use X and Y for a top-down view

        plt.figure(figsize=(10, 8))
        # Plot the trajectory path
        plt.plot(x_coords, y_coords, marker='o', markersize=4, linestyle='-', color='royalblue', label='Camera Path')
        # Highlight the starting and current positions
        plt.scatter(x_coords[0], y_coords[0], c='lime', s=100, label='Start', zorder=5, edgecolors='black')
        plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, label='Current', zorder=5, edgecolors='black')

        plt.title(f"Camera Trajectory (Top-Down View) - Keyframe {sorted_kf_ids[-1]}")
        plt.xlabel("X Position (meters)")
        plt.ylabel("Y Position (meters)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        # Use 'equal' aspect ratio for a true-to-scale representation
        plt.axis('equal') 

        # Save the plot to a file
        plot_filename = os.path.join(self.trajectory_output_dir_2d, f"trajectory_kf_{sorted_kf_ids[-1]:04d}.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free memory
        print(f"    -> Saved 2D trajectory plot to {plot_filename}")

    # --- MODIFICATION: Add a new method to plot and save the 3D trajectory ---
    def _plot_and_save_trajectory_3d(self):
        """
        Generates a 3D plot of the camera's trajectory, showing each pose as an arrow.
        """
        os.makedirs(self.trajectory_output_dir_3d, exist_ok=True)
        sorted_kf_ids = sorted(self.map.keyframes.keys())

        if not sorted_kf_ids:
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        positions = []
        orientations = []

        for kf_id in sorted_kf_ids:
            kf = self.map.keyframes[kf_id]
            positions.append(kf.t.flatten())
            # The camera's viewing direction is the Z-axis of its local coordinate frame.
            # In the world frame, this direction is given by the third column of the rotation matrix R.
            orientations.append(kf.R[:, 2].flatten())

        positions = np.array(positions)
        orientations = np.array(orientations)

        # Plot the trajectory path line
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='grey', linestyle='--', label='Path')

        # Plot each camera pose as an arrow (quiver)
        # The arrow starts at the camera's position (t) and points in its viewing direction.
        ax.quiver(
            positions[:, 0], positions[:, 1], positions[:, 2],  # Arrow starting points
            orientations[:, 0], orientations[:, 1], orientations[:, 2],  # Arrow direction vectors
            length=0.5,  # Length of the arrows in the plot
            normalize=True,
            color='blue',
            label='Camera Pose'
        )

        # Highlight the start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='lime', s=100, label='Start', edgecolors='black')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, label='Current', edgecolors='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Camera Trajectory - Keyframe {sorted_kf_ids[-1]}')
        ax.legend()
        
        # --- Automatic axis scaling to keep aspect ratio somewhat consistent ---
        x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
        y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
        z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])

        max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0

        mid_x = (x_max+x_min) * 0.5
        mid_y = (y_max+y_min) * 0.5
        mid_z = (z_max+z_min) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        # --- End of axis scaling ---

        plot_filename = os.path.join(self.trajectory_output_dir_3d, f"trajectory_3d_kf_{sorted_kf_ids[-1]:04d}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"    -> Saved 3D trajectory plot to {plot_filename}")

    def process_frame(self, frame):
        """
        Processes a single video frame for visual odometry and local bundle adjustment.
        This function performs the following steps:
        1. Converts the input frame to grayscale and extracts feature keypoints and descriptors.
        2. Initializes the map with the first keyframe if none exist.
        3. Matches features between the current frame and the last keyframe.
        4. Estimates the relative pose between the current frame and the last keyframe using matched features.
        5. Determines if the current frame should be added as a new keyframe based on pose change and match quality.
        6. Updates map points and keyframe observations:
            - Re-observes existing map points if matched.
            - Triangulates new points and adds them to the map.
        7. Adds the new keyframe and runs local bundle adjustment to optimize poses and map points.
        Args:
            frame (np.ndarray): The current video frame in BGR format.
        Returns:
            None
        """
        debug = True  # Set to False to disable visualization

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)
        
        if self.map.next_keyframe_id == 0:
            print("Initializing with first keyframe...")
            initial_pose_R = np.eye(3)
            initial_pose_t = np.zeros((3, 1))
            
            kf = Keyframe(
                id=self.map.next_keyframe_id,
                R=initial_pose_R,
                t=initial_pose_t,
                keypoints=keypoints,
                descriptors=descriptors,
                observations=[],
                img=frame.copy()  # Save the RGB frame for visualization
            )
            self.map.add_keyframe(kf)
            return

        last_kf = self.map.keyframes[self.map.next_keyframe_id - 1]
        matches = self.feature_matcher.match(last_kf.descriptors, descriptors)

        # Visualize matched keypoints between last keyframe and current frame
        

        if debug:
            img_matches = cv2.drawMatches(
                last_kf.img,  # Last keyframe's RGB image
                last_kf.keypoints,
                frame,        # Current frame's RGB image
                keypoints,
                matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            debug_dir = "debug_matches"
            os.makedirs(debug_dir, exist_ok=True)
            img_filename = os.path.join(debug_dir, f"matches_{self.map.next_keyframe_id}.png")
            cv2.imwrite(img_filename, img_matches)
            print(f"Saved matched keypoints visualization to {img_filename}")
            # Optionally display the image:
            # cv2.imshow(f"Matched Keypoints (Frame {self.map.next_keyframe_id})", img_matches)
            # cv2.waitKey(1)

        if len(matches) < self.min_matches_to_track:
            print("    -> Frame Discarded: Not enough matches to track.")
            return

        relative_R, relative_t, inlier_pts1, inlier_pts2, inlier_indices = self._estimate_pose(matches, last_kf.keypoints, keypoints)
        
        if relative_R is None or relative_t is None:
            print("    -> Frame Discarded: Could not estimate pose.")
            return
        
        # Calculate inlier ratio
        num_inliers = len(inlier_indices)
        num_matches = len(matches)
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
        is_reliable = inlier_ratio > 0.7 and num_inliers > 20

        if not is_reliable:
            print("    -> Frame Discarded: Low inlier ratio or insufficient inliers.")
            return

        if is_reliable and self._is_keyframe(relative_R, relative_t, matches, inlier_indices, inlier_pts1, inlier_pts2, last_kf):

            if debug:
                img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags=0)
                debug_dir = "debug_keyframes"
                os.makedirs(debug_dir, exist_ok=True)
                img_filename = os.path.join(debug_dir, f"keyframe_{self.map.next_keyframe_id}.png")
                cv2.imwrite(img_filename, img_with_keypoints)
                #cv2.imshow(f"Detected Keypoints (Frame {self.map.next_keyframe_id})", img_with_keypoints)
                print(f"Saved keyframe visualization to {img_filename}")
                #print("Press any key to proceed to the next frame...")
                #cv2.waitKey(0)

            world_R = last_kf.R @ relative_R
            world_t = last_kf.t + last_kf.R @ relative_t
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

                # Keypoint indices are only meaningful within their respective keyframes.
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

                points_3d_relative, _ = self._triangulate_points(relative_R, relative_t, new_pts1, new_pts2)

                if points_3d_relative is not None:
                    # move the point to the world coordinate system
                    points_3d_world = (last_kf.R @ points_3d_relative) + last_kf.t
                    
                    for i in range(points_3d_world.shape[1]):
                        match_idx = newly_triangulated_indices[i]
                        last_kf_kp_idx = matches[match_idx].queryIdx
                        new_kf_kp_idx = matches[match_idx].trainIdx

                        bgr_color = frame[int(new_pts2[i][1]), int(new_pts2[i][0])]
                        rgb_color_normalized = (bgr_color[::-1] / 255.0).astype(np.float64)
                        mp = MapPoint(id=self.map.next_map_point_id, position=points_3d_world[:, i].reshape(3,1), observations=[], color=rgb_color_normalized)
                        mp.observations.append((last_kf.id, last_kf_kp_idx))
                        mp.observations.append((new_kf.id, new_kf_kp_idx))
                        
                        last_kf.observations.append((mp.id, last_kf_kp_idx))
                        new_kf.observations.append((mp.id, new_kf_kp_idx))
                        self.map.add_map_point(mp)

            self.map.add_keyframe(new_kf)
            
            # --- MODIFICATION: Call the new plotting method after adding a keyframe ---
            self._plot_and_save_trajectory_2d()
            self._plot_and_save_trajectory_3d()


            self.bundle_adjuster.run(self.map)

    def _estimate_pose(self, matches, kp1, kp2):
        """
        Estimates the relative pose between two sets of keypoints using the Essential matrix.
        Returned rotation is in rotation matrix form
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        if E is None:
            return None, None, None, None, None
            
        _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        
        # --- DEBUG: Log Inlier Ratio ---
        num_matches = len(matches)
        num_inliers = np.sum(mask)
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
        print(f"    -> Pose Estimation: {num_inliers} inliers out of {num_matches} matches (Ratio: {inlier_ratio:.2f})")
        # --- END DEBUG BLOCK ---

        # A very low ratio indicates a poor geometric match
        if inlier_ratio < 0.4: # You can tune this threshold
            print("    -> WARNING: Low inlier ratio. Pose estimate may be unreliable.")
    
        # Get inlier points and their original indices from the matches list
        inlier_mask = mask.ravel() == 1
        inlier_indices = np.where(inlier_mask)[0]
        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]

        return R_rel, t_rel, inlier_pts1, inlier_pts2, inlier_indices

    def _triangulate_points(self, R_rel, t_rel, pts1, pts2):
        """
        Triangulates 3D points from two sets of 2D points and their relative pose.
        """
        if pts1.shape[0] == 0:
            return None, None
        
        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.camera_matrix @ np.hstack((R_rel, t_rel))
        
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d_hom / (points_4d_hom[3] + 1e-6)
        
        initial_count = points_4d_hom.shape[1]
        # --- Cheirality Check ---
        # 1. Check if points are in front of the first camera (z > 0)
        valid_mask_cam1 = points_3d[2, :] > 0

        # 2. Check if points are in front of the second camera
        # Transform points into the second camera's frame and check z > 0
        points_3d_in_cam2 = R_rel @ points_3d[:3, :] + t_rel
        valid_mask_cam2 = points_3d_in_cam2[2, :] > 0
        
        # The final mask requires the point to be valid in both views
        valid_mask = np.logical_and(valid_mask_cam1, valid_mask_cam2)
        # --- Cheirality Check Over ---

        points_3d = points_3d[:3, valid_mask]

        # --- DEBUG: Log Triangulation Results ---
        final_count = points_3d.shape[1]
        print(f"    -> Triangulation: Kept {final_count} of {initial_count} points (z > 0 filter).")
        # --- END DEBUG BLOCK ---
        
        if points_3d.shape[1] == 0:
            return None, None

        return points_3d, np.where(valid_mask)[0]

def main():

    # Clean up output directories before starting

    folders_to_delete = [
        'output_trajectory_2d',
        'output_trajectory_3d',
        'debug_sparsity',
        'debug_matches',
        'debug_keyframes',
        'output_map/lba_steps'
    ]
    for folder in folders_to_delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")

    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/my_dataset_7/video_0001.mp4'
    output_dir = 'output_map'
    
    # --- MODIFICATION: Define directories for both 2D and 3D trajectory plots ---
    trajectory_dir_2d = 'output_trajectory_2d'
    trajectory_dir_3d = 'output_trajectory_3d'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(trajectory_dir_2d, exist_ok=True)
    os.makedirs(trajectory_dir_3d, exist_ok=True)
    final_map_path = os.path.join(output_dir, "final_map_lba.pcd")

    KEYFRAME_CRITERIA = {
        "min_pixel_displacement": 20.0, # higher more selective
        "min_rotation": 0.15, # higher more selective
        "min_feature_ratio": 0.25, # lower more selective
        "min_parallax_deg": 1.0, # degrees
        "min_tracked_for_parallax": 20 # frames
    }

    camera_matrix = np.array([
        [912.7820434570312, 0.0, 650.2929077148438],
        [0.0, 913.0294189453125, 362.7241516113281],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    print("Initializing visual odometry pipeline...")
    # --- MODIFICATION: Pass both trajectory directories to the pipeline ---
    vo_pipeline = VisualOdometryPipeline(
        camera_matrix,
        dist_coeffs,
        ORBExtractor(n_features=4000),
        BruteForceMatcher(),
        KEYFRAME_CRITERIA,
        trajectory_output_dir_2d=trajectory_dir_2d,
        trajectory_output_dir_3d=trajectory_dir_3d
    )
    print("Opening video...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if 0 <= frame_idx: # 90 -> 1400
            print(f"Processing frame {frame_idx}...")
            #cv2.imshow("Current Frame", frame)
            #cv2.waitKey(1)
            vo_pipeline.process_frame(frame)
        frame_idx += 1

    print("\nVideo processing finished.")
    
    cap.release()
    cv2.destroyAllWindows()

    # --- INSERT THIS CODE FOR GLOBAL BA ---
    print("\n--- Running Final Global Bundle Adjustment ---")
    num_keyframes = vo_pipeline.map.next_keyframe_id
    if num_keyframes > 2:
        # Temporarily set the window size to include all keyframes
        # The run method fixes the first KF and adjusts the rest in the window
        original_window_size = vo_pipeline.bundle_adjuster.window_size
        vo_pipeline.bundle_adjuster.window_size = num_keyframes 
        
        vo_pipeline.bundle_adjuster.run(vo_pipeline.map)
        
        # Restore the original window size (optional, but good practice)
        vo_pipeline.bundle_adjuster.window_size = original_window_size
        print("--- Global Bundle Adjustment Complete ---")
    else:
        print("--- Global Bundle Adjustment Skipped: Not enough keyframes ---")
    # --- END OF INSERTED CODE ---

    final_pcd = vo_pipeline.map.get_pcd()
    if final_pcd.has_points():
        print(f"\nSaving final map from {vo_pipeline.map.next_keyframe_id} keyframes to '{final_map_path}'...")
        downsampled_pcd = final_pcd.voxel_down_sample(voxel_size=0.02)
        o3d.io.write_point_cloud(final_map_path, downsampled_pcd)
        print("Map saved successfully.")

        print("\nDisplaying the final map. Press 'q' in the window to close.")
        o3d.visualization.draw_geometries([downsampled_pcd])
    else:
        print("\nNo points were generated for the final map.")


if __name__ == '__main__':
    main()