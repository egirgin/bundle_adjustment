import sys
import cv2
import numpy as np
import os
import open3d as o3d
from scipy.optimize import least_squares
from dataclasses import dataclass
from scipy.sparse import lil_matrix


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

        # 4. Run the optimization
        res = least_squares(
            self._cost_function,
            params,
            jac_sparsity=sparsity_matrix,  # <-- Tell the solver to use it!
            loss='huber', # Using robust loss is also highly recommended
            args=(fixed_kf_pose, fixed_kf_id, adjustable_kf_ids, local_map_point_ids, observations, keypoints_2d),
            verbose=1, # Set to 1 for more detailed output
            xtol=1e-5,
            ftol=1e-5,
            max_nfev=50
        )
        
        # 5. Update the map with the optimized parameters (ONLY for adjustable frames)
        optimized_params = res.x
        final_cost = np.sum(res.fun**2)
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
            
        print(f"    -> LBA Complete. Fixed KF {fixed_kf_id}. Optimized {num_adjustable_keyframes} KFs and {len(local_map_point_ids)} MPs.")
        print(f"    -> LBA Complete. Initial Cost: {initial_cost:.2f}, Final Cost: {final_cost:.2f}")

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

    def __init__(self, camera_matrix, dist_coeffs, feature_extractor, feature_matcher, keyframe_criteria):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.keyframe_criteria = keyframe_criteria
        self.min_matches_to_track = 20

        self.map = Map()
        self.bundle_adjuster = BundleAdjuster(camera_matrix, window_size=5)

    def _calculate_median_displacement(self, pts1, pts2):
        if len(pts1) == 0:
            return 0
        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        return np.median(displacements)

    def _is_keyframe(self, relative_R, num_matches, inlier_pts1, inlier_pts2, last_kf_feature_count):
    
        median_displacement = self._calculate_median_displacement(inlier_pts1, inlier_pts2)
        # higher values mean more dramatic change in the translation of matrix.
        if median_displacement > self.keyframe_criteria['min_pixel_displacement']:
            print(f"    -> Keyframe Trigger: Pixel Displacement ({median_displacement:.2f} > {self.keyframe_criteria['min_pixel_displacement']})")
            return True

        angle, _ = cv2.Rodrigues(relative_R)
        rotation_magnitude = np.linalg.norm(angle)
        # higher values mean more dramatic change in the rotation of matrix
        if rotation_magnitude > self.keyframe_criteria['min_rotation']:
            print(f"    -> Keyframe Trigger: Rotation ({rotation_magnitude:.4f} > {self.keyframe_criteria['min_rotation']})")
            return True

        feature_ratio = num_matches / last_kf_feature_count if last_kf_feature_count > 0 else 0
        # 1 if all features from the last keyframe are matched in the current frame so the visible keypoints are mostly the same
        # 0 if none of the last keyframe's features are matched so the keypoints between two frame are not the same.
        if feature_ratio < self.keyframe_criteria['min_feature_ratio']:
            print(f"    -> Keyframe Trigger: Feature Ratio ({feature_ratio:.2f} < {self.keyframe_criteria['min_feature_ratio']})")
            return True

        return False

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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)
        
        if self.map.next_keyframe_id == 0:
            print("Initializing with first keyframe...")
            initial_pose_R = np.eye(3)
            initial_pose_t = np.zeros((3, 1))
            
            kf = Keyframe(id=self.map.next_keyframe_id, R=initial_pose_R, t=initial_pose_t, keypoints=keypoints, descriptors=descriptors, observations=[])
            self.map.add_keyframe(kf)
            return

        last_kf = self.map.keyframes[self.map.next_keyframe_id - 1]
        matches = self.feature_matcher.match(last_kf.descriptors, descriptors)

        
        if len(matches) < self.min_matches_to_track:
            print("    -> Frame Discarded: Not enough matches to track.")
            return

        relative_R, relative_t, inlier_pts1, inlier_pts2, inlier_indices = self._estimate_pose(matches, last_kf.keypoints, keypoints)
        
        if relative_R is None or relative_t is None:
            print("    -> Frame Discarded: Could not estimate pose.")
            return
        
        if self._is_keyframe(relative_R, len(matches), inlier_pts1, inlier_pts2, len(last_kf.keypoints)):
            
            # Visualization and saving of detected keyframes
            debug = True  # Set this to False to disable visualization and saving

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
            new_kf = Keyframe(id=self.map.next_keyframe_id, R=world_R, t=world_t, keypoints=keypoints, descriptors=descriptors, observations=[])

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

                        mp = MapPoint(id=self.map.next_map_point_id, position=points_3d_world[:, i].reshape(3,1), observations=[], color=frame[int(new_pts2[i][1]), int(new_pts2[i][0])])
                        mp.observations.append((last_kf.id, last_kf_kp_idx))
                        mp.observations.append((new_kf.id, new_kf_kp_idx))
                        
                        last_kf.observations.append((mp.id, last_kf_kp_idx))
                        new_kf.observations.append((mp.id, new_kf_kp_idx))
                        self.map.add_map_point(mp)

            self.map.add_keyframe(new_kf)
            self.bundle_adjuster.run(self.map)

    def _estimate_pose(self, matches, kp1, kp2):
        """
        Estimates the relative pose between two sets of keypoints using the Essential matrix.
        Returned rotation is in rotation matrix form
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None, None, None
            
        _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        
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
        
        valid_mask = points_3d[2, :] > 0
        points_3d = points_3d[:3, valid_mask]
        
        if points_3d.shape[1] == 0:
            return None, None

        return points_3d, np.where(valid_mask)[0]

def main():
    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/srge_lab.avi'
    output_dir = 'output_map'
    os.makedirs(output_dir, exist_ok=True)
    final_map_path = os.path.join(output_dir, "final_map_lba.pcd")

    KEYFRAME_CRITERIA = {
        "min_pixel_displacement": 30.0,
        "min_rotation": 1.0,
        "min_feature_ratio": 0.2
    }

    camera_matrix = np.array([
        [431.39865, 0.0, 429.08605],
        [0.0, 431.39865, 235.27142],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    print("Initializing visual odometry pipeline...")
    vo_pipeline = VisualOdometryPipeline(camera_matrix, dist_coeffs, ORBExtractor(), BruteForceMatcher(), KEYFRAME_CRITERIA)
    print("Opening video...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if 90 <= frame_idx < 500: # 90 -> 1400
            print(f"Processing frame {frame_idx}...")
            #cv2.imshow("Current Frame", frame)
            #cv2.waitKey(1)
            vo_pipeline.process_frame(frame)
        frame_idx += 1

    print("\nVideo processing finished.")
    
    cap.release()
    cv2.destroyAllWindows()

    final_pcd = vo_pipeline.map.get_pcd()
    if final_pcd.has_points():
        print(f"\nSaving final map from {vo_pipeline.map.next_keyframe_id} keyframes to '{final_map_path}'...")
        downsampled_pcd = final_pcd.voxel_down_sample(voxel_size=0.1)
        o3d.io.write_point_cloud(final_map_path, downsampled_pcd)
        print("Map saved successfully.")

        print("\nDisplaying the final map. Press 'q' in the window to close.")
        o3d.visualization.draw_geometries([downsampled_pcd])
    else:
        print("\nNo points were generated for the final map.")


if __name__ == '__main__':
    main()