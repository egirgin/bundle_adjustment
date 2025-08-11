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
        # Extracts keypoints and descriptors from an image.
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
        # Finds the best matches between two sets of descriptors.
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
    The main class to orchestrate Visual Odometry and Point Cloud registration.
    """
    def __init__(self, camera_matrix, dist_coeffs, feature_extractor, feature_matcher):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.previous_frame_data = None

        self.world_R = np.eye(3)
        self.world_t = np.zeros((3, 1))

    def process_frame(self, frame):
        """
        Processes a single frame to estimate pose and returns triangulated points
        in the global coordinate system along with their colors.
        """
        # Feature extraction is done on the grayscale image.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)
        
        # On the first frame, just store its data and exit.
        if self.previous_frame_data is None:
            print("Initializing first frame...")
            # MODIFIED: Store the original color frame for color sampling later.
            self.previous_frame_data = {'image': frame, 'keypoints': keypoints, 'descriptors': descriptors}
            return None, None

        matches = self.feature_matcher.match(self.previous_frame_data['descriptors'], descriptors)

        if len(matches) > 15:
            relative_R, relative_t, pts1, pts2 = self._estimate_pose(matches, self.previous_frame_data['keypoints'], keypoints)
            
            if relative_R is not None and relative_t is not None:
                # Triangulate points and get the mask of valid points.
                points_3d_relative, valid_mask = self._triangulate_points(relative_R, relative_t, pts1, pts2)
                
                if points_3d_relative is not None:
                    # NEW: Get the colors for the valid points from the *previous* frame.
                    # We use the coordinates from pts1, which correspond to the previous frame.
                    valid_pts1 = pts1[valid_mask]
                    point_indices = np.int32(valid_pts1)
                    # Use the integer indices to look up the BGR color in the original color image.
                    colors_bgr = self.previous_frame_data['image'][point_indices[:, 1], point_indices[:, 0]]
                    
                    # Convert colors from BGR to RGB and normalize to [0, 1] for Open3D.
                    colors_rgb = colors_bgr[:, ::-1] / 255.0

                    # Transform the relative 3D points into the global coordinate system.
                    points_3d_world = (self.world_R @ points_3d_relative) + self.world_t

                    # Update the global pose.
                    self.world_t = self.world_t + self.world_R @ relative_t
                    self.world_R = self.world_R @ relative_R

                    # MODIFIED: Store the new color frame for the next iteration.
                    self.previous_frame_data = {'image': frame, 'keypoints': keypoints, 'descriptors': descriptors}
                    # Return both the world points and their colors.
                    return points_3d_world, colors_rgb

        # If processing fails, update frame data and return None.
        self.previous_frame_data = {'image': frame, 'keypoints': keypoints, 'descriptors': descriptors}
        return None, None

    def _estimate_pose(self, matches, kp1, kp2):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None, None
            
        _, R_rel, t_rel, pose_mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        
        # Use the mask from recoverPose to filter out outlier matches.
        inlier_mask = pose_mask.ravel() == 1
        return R_rel, t_rel, pts1[inlier_mask], pts2[inlier_mask]

    def _triangulate_points(self, R_rel, t_rel, pts1, pts2):
        if pts1.shape[0] == 0 or pts2.shape[0] == 0:
            return None, None
        
        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.camera_matrix @ np.hstack((R_rel, t_rel))
        
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d_hom / (points_4d_hom[3] + 1e-6)
        
        # MODIFIED: Keep track of the mask of points that are in front of the camera.
        valid_mask = points_3d[2, :] > 0
        
        # Also filter the points based on the mask.
        filtered_points_3d = points_3d[:, valid_mask]

        if filtered_points_3d.shape[1] == 0:
            return None, None

        # Return both the points and the mask, so we can filter colors accordingly.
        return filtered_points_3d[0:3, :], valid_mask

def main():
    """Main function to run the Visual Odometry and Point Cloud Registration pipeline."""
    # --- Configuration ---
    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/srge_lab.avi' # <--- CHANGE THIS
    output_dir = 'output_map'
    os.makedirs(output_dir, exist_ok=True)
    final_map_path = os.path.join(output_dir, "final_colored_map.pcd")

    # --- Camera Intrinsics ---
    camera_matrix = np.array([
        [431.39865, 0.0, 429.08605],
        [0.0, 431.39865, 235.27142],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # --- Initialization ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    vo_pipeline = VisualOdometryPipeline(camera_matrix, dist_coeffs, ORBExtractor(), BruteForceMatcher())
    
    global_pcd = o3d.geometry.PointCloud()

    # --- Main Processing Loop ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_count}...")
        # MODIFIED: Process frame to get both world points and their colors.
        new_points_world, new_colors_rgb = vo_pipeline.process_frame(frame)
        
        # NEW: If new points and colors were generated, add them to the global map.
        if new_points_world is not None and new_colors_rgb is not None:
            print(f"  -> Adding {new_points_world.shape[1]} new colored points to the map.")
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(new_points_world.T)
            temp_pcd.colors = o3d.utility.Vector3dVector(new_colors_rgb)
            global_pcd += temp_pcd

        frame_count += 1

    print("\nVideo processing finished.")
    
    # --- Finalization and Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    
    if global_pcd.has_points():
        print(f"\nSaving final registered and colored point cloud to '{final_map_path}'...")
        # Voxel downsampling cleans up the point cloud and merges colors in each voxel.
        downsampled_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)
        o3d.io.write_point_cloud(final_map_path, downsampled_pcd)
        print("Map saved successfully.")

        # --- Visualization ---
        print("\nDisplaying the final map. Press 'q' in the window to close.")
        o3d.visualization.draw_geometries([downsampled_pcd])
    else:
        print("\nNo points were generated for the final map.")


if __name__ == '__main__':
    main()