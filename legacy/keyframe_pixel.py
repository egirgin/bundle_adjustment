import cv2
import numpy as np
import os
import open3d as o3d

class FeatureExtractor:
    """Abstract base class for feature extractors."""
    def extract(self, image):
        raise NotImplementedError

class ORBExtractor(FeatureExtractor):
    """Concrete ORB feature extractor."""
    def __init__(self, n_features=3000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def extract(self, image):
        return self.orb.detectAndCompute(image, None)

class FeatureMatcher:
    """Abstract base class for feature matchers."""
    def match(self, des1, des2):
        raise NotImplementedError

class BruteForceMatcher(FeatureMatcher):
    """Concrete Brute-Force feature matcher with Lowe's ratio test."""
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
    The main class to orchestrate Visual Odometry with KeyFrame Management.
    """
    def __init__(self, camera_matrix, dist_coeffs, feature_extractor, feature_matcher, keyframe_criteria):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        
        self.keyframe_criteria = keyframe_criteria
        
        self.last_keyframe_data = None
        self.last_keyframe_features_count = 0

        self.world_R = np.eye(3)
        self.world_t = np.zeros((3, 1))

    def _calculate_median_displacement(self, pts1, pts2):
        """Calculates the median pixel displacement between two sets of points."""
        if len(pts1) == 0:
            return 0
        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        return np.median(displacements)

    def _is_keyframe(self, relative_R, num_matches, inlier_pts1, inlier_pts2):
        """
        Decides if the current frame should be a keyframe based on selection criteria.
        """
        # **MODIFIED**: Criterion 1: Minimum pixel displacement of features
        median_displacement = self._calculate_median_displacement(inlier_pts1, inlier_pts2)
        if median_displacement > self.keyframe_criteria['min_pixel_displacement']:
            print(f"    -> Keyframe Trigger: Pixel Displacement ({median_displacement:.2f} > {self.keyframe_criteria['min_pixel_displacement']})")
            return True

        # Criterion 2: Minimum rotation since last keyframe
        angle, _ = cv2.Rodrigues(relative_R)
        rotation_magnitude = np.linalg.norm(angle)
        if rotation_magnitude > self.keyframe_criteria['min_rotation']:
            print(f"    -> Keyframe Trigger: Rotation ({rotation_magnitude:.4f} > {self.keyframe_criteria['min_rotation']})")
            return True

        # Criterion 3: Minimum number of tracked features (as a ratio)
        feature_ratio = num_matches / self.last_keyframe_features_count if self.last_keyframe_features_count > 0 else 0
        if feature_ratio < self.keyframe_criteria['min_feature_ratio']:
            print(f"    -> Keyframe Trigger: Feature Ratio ({feature_ratio:.2f} < {self.keyframe_criteria['min_feature_ratio']})")
            return True

        return False

    def process_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)
        
        if self.last_keyframe_data is None:
            print("Initializing with first keyframe...")
            self.last_keyframe_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
            self.last_keyframe_features_count = len(keypoints)
            return None

        matches = self.feature_matcher.match(self.last_keyframe_data['descriptors'], descriptors)

        if len(matches) < 20:
            print("    -> Frame Discarded: Not enough matches to track.")
            return None

        # **MODIFIED**: Estimate pose and get inlier points for displacement calculation
        relative_R, relative_t, inlier_pts1, inlier_pts2 = self._estimate_pose(matches, self.last_keyframe_data['keypoints'], keypoints)
        
        if relative_R is None or relative_t is None:
            print("    -> Frame Discarded: Could not estimate pose.")
            return None
        
        # **** KEYFRAME DECISION LOGIC ****
        if self._is_keyframe(relative_R, len(matches), inlier_pts1, inlier_pts2):
            points_3d_relative = self._triangulate_points(relative_R, relative_t, inlier_pts1, inlier_pts2)
            
            if points_3d_relative is not None:
                points_3d_world = (self.world_R @ points_3d_relative) + self.world_t

                self.world_t = self.world_t + self.world_R @ relative_t
                self.world_R = self.world_R @ relative_R

                self.last_keyframe_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
                self.last_keyframe_features_count = len(keypoints)
                return points_3d_world
        else:
            print("    -> Frame Discarded: Did not meet keyframe criteria.")

        return None

    def _estimate_pose(self, matches, kp1, kp2):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None, None
            
        _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]

        # **MODIFIED**: Return inlier points as well
        return R_rel, t_rel, inlier_pts1, inlier_pts2

    def _triangulate_points(self, R_rel, t_rel, pts1, pts2):
        if pts1.shape[0] == 0 or pts2.shape[0] == 0:
            return None
        
        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.camera_matrix @ np.hstack((R_rel, t_rel))
        
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        points_3d = points_4d_hom / (points_4d_hom[3] + 1e-6)
        
        valid_mask = points_3d[2, :] > 0
        points_3d = points_3d[:, valid_mask]

        if points_3d.shape[1] == 0:
            return None

        return points_3d[0:3, :]

def main():
    """Main function to run the Visual Odometry and Point Cloud Registration pipeline."""
    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/srge_lab.avi'
    output_dir = 'output_map'
    os.makedirs(output_dir, exist_ok=True)
    final_map_path = os.path.join(output_dir, "final_map_keyframes.pcd")

    # --- **MODIFIED**: Keyframe Selection Criteria ---
    # Tune these values to control how frequently keyframes are created.
    # More frequent keyframes = denser map, more computation.
    # Less frequent keyframes = sparser map, faster, but might miss details.
    KEYFRAME_CRITERIA = {
        "min_pixel_displacement": 30.0, # Minimum median feature displacement (in pixels) to be a keyframe
        "min_rotation": 0.6,            # Minimum rotation (in radians) to be a keyframe
        "min_feature_ratio": 0.3        # Add keyframe if tracked features fall below this ratio
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

    vo_pipeline = VisualOdometryPipeline(camera_matrix, dist_coeffs, ORBExtractor(), BruteForceMatcher(), KEYFRAME_CRITERIA)
    
    global_pcd = o3d.geometry.PointCloud()
    keyframe_count = 0

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_idx}...")
        new_points_world = vo_pipeline.process_frame(frame)
        
        if new_points_world is not None:
            keyframe_count += 1
            print(f"  -> KEYFRAME {keyframe_count} CREATED. Adding {new_points_world.shape[1]} new points to the map.")
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(new_points_world.T)
            global_pcd += temp_pcd

        frame_idx += 1

    print("\nVideo processing finished.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if global_pcd.has_points():
        print(f"\nSaving final map from {keyframe_count} keyframes to '{final_map_path}'...")
        downsampled_pcd = global_pcd.voxel_down_sample(voxel_size=0.1)
        o3d.io.write_point_cloud(final_map_path, downsampled_pcd)
        print("Map saved successfully.")

        print("\nDisplaying the final map. Press 'q' in the window to close.")
        o3d.visualization.draw_geometries([downsampled_pcd])
    else:
        print("\nNo points were generated for the final map.")


if __name__ == '__main__':
    main()