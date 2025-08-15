# vo_project/bundle_adjuster.py

import cv2
import numpy as np
import os
import open3d as o3d
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from map_structures import Map
from visualization import plot_and_save_sparsity

from parameters import DEBUG_DIRS


class BundleAdjuster:
    """
    Performs Local Bundle Adjustment (LBA) to optimize keyframe poses and map points.
    """
    def __init__(self, camera_matrix, window_size=5):
        self.camera_matrix = camera_matrix
        self.window_size = window_size

    def _cost_function(self, params, fixed_kf_pose, fixed_kf_id, adjustable_kf_ids, map_point_ids, observations, keypoints_2d):
        """Computes the re-projection error for Bundle Adjustment optimization.
        Steps:
            - Compute the 3D points from the map point IDs.
            - For each observation, project the 3D point into the image plane.
            - Compute the reprojection error between the observed 2D point and the projected point.
        args:
            - params: The optimization parameters (keyframe poses and map points).
            - fixed_kf_pose: The pose (rotation and translation) of the fixed keyframe.
            - fixed_kf_id: The ID of the fixed keyframe.
            - adjustable_kf_ids: List of IDs for the adjustable keyframes.
            - map_point_ids: List of IDs for the map points.
            - observations: List of observations (keyframe ID, map point ID).
            - keypoints_2d: Dictionary mapping (keyframe ID, map point ID) to 2D keypoints.
        """
        num_adjustable_keyframes = len(adjustable_kf_ids)
        num_map_points = len(map_point_ids)
        
        poses_rvec = params[0 : num_adjustable_keyframes * 3].reshape((num_adjustable_keyframes, 3))
        poses_tvec = params[num_adjustable_keyframes * 3 : num_adjustable_keyframes * 6].reshape((num_adjustable_keyframes, 3))
        points_3d = params[num_adjustable_keyframes * 6 :].reshape((num_map_points, 3))

        adj_kf_id_to_idx = {kf_id: i for i, kf_id in enumerate(adjustable_kf_ids)}
        mp_id_to_idx = {mp_id: i for i, mp_id in enumerate(map_point_ids)}

        errors = []
        fixed_R, fixed_t = fixed_kf_pose

        for obs_kf_id, obs_mp_id in observations:
            mp_idx = mp_id_to_idx.get(obs_mp_id)
            if mp_idx is None: continue

            point_3d = points_3d[mp_idx]
            
            if obs_kf_id == fixed_kf_id:
                rvec, _ = cv2.Rodrigues(fixed_R)
                tvec = fixed_t
            else:
                kf_idx = adj_kf_id_to_idx.get(obs_kf_id)
                if kf_idx is None: continue
                rvec = poses_rvec[kf_idx]
                tvec = poses_tvec[kf_idx]
            
            projected_points, _ = cv2.projectPoints(point_3d, rvec, tvec, self.camera_matrix, None)
            observed_point = keypoints_2d[(obs_kf_id, obs_mp_id)]
            error = (observed_point - projected_points.ravel()).ravel()
            errors.extend(error)
            
        return np.array(errors)
    
    def _prepare_sparsity_matrix(self, num_adj_kfs, num_mps, adj_kf_ids, mp_ids, observations):
        """
        Creates the sparse Jacobian matrix structure for the optimizer.
        Steps:
            - Initialize the sparse matrix A with the appropriate dimensions.
            - Iterate over each observation to populate the Jacobian structure.
        args:
            - num_adj_kfs: Number of adjustable keyframes.
            - num_mps: Number of map points.
            - adj_kf_ids: List of adjustable keyframe IDs.
            - mp_ids: List of map point IDs.
            - observations: List of observations (keyframe ID, map point ID).
        return:
            A sparse matrix representing the Jacobian structure.
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
        """Run LBA on the most recent window of keyframes.
        Steps:
            1. Gather local keyframes and map points.
            2. Initialize parameters for optimization.
            3. Set up the cost function and sparsity matrix.
            4. Run the optimization.
            5. Update the map with the optimized parameters.
        args:
            gmap: The global map to optimize.
        """
        print("    --- Running Local Bundle Adjustment ---")
        
        all_kf_ids = sorted(gmap.keyframes.keys())
        if len(all_kf_ids) < self.window_size:
            print("    -> LBA Skipped: Not enough keyframes.")
            return
            
        local_kf_ids = all_kf_ids[-(self.window_size + 1) : -1]
        fixed_kf_id = local_kf_ids[0]
        adjustable_kf_ids = local_kf_ids[1:]

        if not adjustable_kf_ids:
            print("    -> LBA Skipped: No adjustable keyframes.")
            return

        fixed_keyframe = gmap.keyframes[fixed_kf_id]
        
        local_map_point_ids, observations, keypoints_2d = self._gather_local_data(gmap, local_kf_ids)
        if not local_map_point_ids:
            print("    -> LBA Skipped: No points in the local window.")
            return

        # Prepare initial parameters for the optimizer (ONLY for adjustable frames)
        # (R1, R1, R1, R2, R2, R2, ...)
        initial_rvecs = np.array([cv2.Rodrigues(gmap.keyframes[i].R)[0].ravel() for i in adjustable_kf_ids])
        # (t1, t1, t1, t2, t2, t2, ...)
        initial_tvecs = np.array([gmap.keyframes[i].t.ravel() for i in adjustable_kf_ids])
        # (p1x, p1y, p1z, p2x, p2y, p2z, ...)
        initial_points_3d = np.array([gmap.map_points[i].position.ravel() for i in local_map_point_ids])
        params = np.concatenate([initial_rvecs.flatten(), initial_tvecs.flatten(), initial_points_3d.flatten()])
        
        fixed_kf_pose = (fixed_keyframe.R, fixed_keyframe.t)
        initial_cost = np.sum(self._cost_function(params, fixed_kf_pose, fixed_kf_id, adjustable_kf_ids, local_map_point_ids, observations, keypoints_2d)**2)

        sparsity_matrix = self._prepare_sparsity_matrix(len(adjustable_kf_ids), len(local_map_point_ids), adjustable_kf_ids, local_map_point_ids, observations)
        plot_and_save_sparsity(sparsity_matrix, fixed_kf_id, local_kf_ids[-1], "debug_sparsity")

        res = least_squares(
            self._cost_function, params, jac_sparsity=sparsity_matrix, loss='huber',
            args=(fixed_kf_pose, fixed_kf_id, adjustable_kf_ids, local_map_point_ids, observations, keypoints_2d),
            verbose=0, xtol=1e-5, ftol=1e-5, max_nfev=50
        )

        final_cost = np.sum(res.fun**2)
        if final_cost >= initial_cost:
            print(f"    -> LBA Diverged! Cost increased from {initial_cost:.2f} to {final_cost:.2f}. Discarding results.")
            return
        
        self._update_map(gmap, res.x, adjustable_kf_ids, local_map_point_ids)
        
        improvement = 100.0 * (initial_cost - final_cost) / (initial_cost + 1e-8)
        print(f"    -> LBA Complete. Initial Cost: {initial_cost:.2f}, Final Cost: {final_cost:.2f}, Improvement: {improvement:.2f}%")
        
        # Save intermediate point cloud
        lba_steps_dir = DEBUG_DIRS['lba_steps']

        updated_pcd = gmap.get_pcd()
        if updated_pcd.has_points():
            pcd_filename = os.path.join(lba_steps_dir, f"map_after_lba_kf_{fixed_kf_id}.pcd")
            o3d.io.write_point_cloud(pcd_filename, updated_pcd)
            print(f"    -> Saved intermediate map to {pcd_filename}")

    def _gather_local_data(self, gmap: Map, local_kf_ids: list):
        """
        Collects map points and observations from a window of keyframes.
        args:
            - gmap: The global map containing keyframes and map points.
            - local_kf_ids: List of keyframe IDs in the local window.
        return:
            - local_map_point_ids: Sorted list of map point IDs in the local window.
            - observations: List of (keyframe_id, map_point_id) tuples representing all observations.
            - keypoints_2d: Dictionary mapping (keyframe_id, map_point_id) to 2D keypoint coordinates.
        """
        local_map_point_ids = set()
        observations = []
        keypoints_2d = {}

        for kf_id in local_kf_ids: # eg 21
            kf = gmap.keyframes[kf_id]  # Keyframe object
            for mp_id, kp_idx in kf.observations: # 3D point id, 2D point id
                if mp_id in gmap.map_points:
                    local_map_point_ids.add(mp_id)
                    observations.append((kf_id, mp_id))
                    keypoints_2d[(kf_id, mp_id)] = kf.keypoints[kp_idx].pt
        
        return sorted(list(local_map_point_ids)), observations, keypoints_2d

    def _update_map(self, gmap: Map, optimized_params: np.ndarray, adjustable_kf_ids: list, local_map_point_ids: list):
        """Updates the map with optimized parameters.
        args:
            - gmap: The global map to update.
            - optimized_params: The optimized parameters from the bundle adjustment.
            - adjustable_kf_ids: List of keyframe IDs that were adjusted.
            - local_map_point_ids: List of map point IDs that were adjusted.            
        """
        num_adj_kfs = len(adjustable_kf_ids)
        
        opt_rvecs = optimized_params[0 : num_adj_kfs * 3].reshape((num_adj_kfs, 3))
        opt_tvecs = optimized_params[num_adj_kfs * 3 : num_adj_kfs * 6].reshape((num_adj_kfs, 3))
        opt_points_3d = optimized_params[num_adj_kfs * 6 :].reshape((len(local_map_point_ids), 3))

        for i, kf_id in enumerate(adjustable_kf_ids):
            R, _ = cv2.Rodrigues(opt_rvecs[i])
            gmap.keyframes[kf_id].R = R
            gmap.keyframes[kf_id].t = opt_tvecs[i].reshape(3, 1)

        for i, mp_id in enumerate(local_map_point_ids):
            gmap.map_points[mp_id].position = opt_points_3d[i].reshape(3, 1)
